#!/usr/bin/env python3
"""V20 results_step4 CSV -> <=30aa search accuracy (no re-prediction needed).

A spectrum's candidate pool is length-homogeneous: candidates match the
precursor within +-10 ppm, and peptides differing by one residue differ in
neutral mass by >= one Gly (~57.02 Da; variable mods only add mass), far outside
any +-10 ppm window.  Hence a <=30aa-GT spectrum's <=32 pool holds only <=30aa
candidates, so its stored top-1 ``<branch>_hit`` / ``<branch>_gt_rank`` already
are the <=30aa-pool values; >30aa-GT spectra (pools all >30aa) drop out of the
<=30 denominator entirely -- the same behaviour as the Prosit run's collect
filter.  Filtering the stored per-scan results CSV to GT backbone <=30aa (with
GT in candidates) therefore reproduces the <=30aa search accuracy exactly.

top1/top3/top10_acc = fraction of n_with_gt_le30 with hit=1 / gt_rank<=3/<=10,
identical definitions to the Prosit summary (step4_prosit_score.py).

Usage:
    python results_len30_accuracy.py \
        --csv <out>/results_step4_<sp>.csv \
        --out <out>/len30_accuracy_<sp>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

NUMERIC_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
FLANK_RE = re.compile(r"^[A-Z-]*\.(.*?)\.[A-Z-]*$")


def backbone_len(seq: str) -> int:
    m = FLANK_RE.match(seq)
    core = m.group(1) if m else seq
    core = re.sub(r"\[[^\]]*\]", "", core)
    core = NUMERIC_RE.sub("", core)
    return sum(1 for ch in core if ch.isalpha())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", required=True, help="results_step4_<sp>.csv")
    p.add_argument("--out", default=None, help="summary JSON to write")
    p.add_argument("--max-len", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.csv, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        fieldnames = rd.fieldnames or []
        branches = sorted({h[:-4] for h in fieldnames if h.endswith("_hit")})
        have_gt_in = "gt_in_candidates" in fieldnames
        rows = [r for r in rd]
    if not rows:
        print("empty csv")
        return 1
    print(f"rows={len(rows)} branches={len(branches)}")

    def gt_in(r) -> bool:
        if have_gt_in:
            return r.get("gt_in_candidates", "0").strip() in ("1", "True", "true")
        # fallback: GT present in candidates iff its rank column is filled
        for b in branches:
            v = r.get(f"{b}_gt_rank", "")
            if v not in ("", "0", "0.0"):
                return True
        return False

    le30 = [r for r in rows if gt_in(r) and backbone_len(r.get("gt_seq", "")) <= args.max_len]
    denom_all = sum(1 for r in rows if gt_in(r))
    denom_le30 = len(le30)
    print(f"gt in candidates: all={denom_all}, backbone<={args.max_len}: {denom_le30} "
          f"(excluded GT>{args.max_len}: {denom_all - denom_le30})")

    out = {"n_with_gt_full": denom_all,
           "n_with_gt_le30": denom_le30,
           "n_excluded_gt_gt30": denom_all - denom_le30,
           "branches_le30": {}}
    warn_gt30_best = {}
    for b in branches:
        hits = ranks = 0
        n_best_gt30 = 0
        for r in le30:
            try:
                hit = int(float(r.get(f"{b}_hit", "0")))
            except (TypeError, ValueError):
                hit = 0
            try:
                rank = int(float(r.get(f"{b}_gt_rank", "0")))
            except (TypeError, ValueError):
                rank = 0
            hits += hit
            ranks += 1 if rank <= 3 else 0
            if backbone_len(r.get(f"{b}_best_seq", "")) > args.max_len:
                n_best_gt30 += 1
        warn_gt30_best[b] = n_best_gt30
        out["branches_le30"][b] = {
            "hits": hits,
            "top1_acc": hits / denom_le30 if denom_le30 else 0.0,
            "top3_acc": ranks / denom_le30 if denom_le30 else 0.0,
        }
    # top10 needs full rank lists; recompute once per branch
    for b in branches:
        n10 = 0
        for r in le30:
            try:
                rank = int(float(r.get(f"{b}_gt_rank", "0")))
            except (TypeError, ValueError):
                rank = 0
            n10 += 1 if 0 < rank <= 10 else 0
        out["branches_le30"][b]["top10_acc"] = n10 / denom_le30 if denom_le30 else 0.0

    gt30_best_rows = {b: n for b, n in warn_gt30_best.items() if n}
    if gt30_best_rows:
        print("WARN rows (GT<=30, in candidates) whose best_seq is >30aa "
              "(should be 0 if pools are length-homogeneous):")
        for b, n in gt30_best_rows.items():
            print(f"  {b}: {n}")
    else:
        print("sanity OK: no GT<=30 row has a >30aa best_seq (pools are length-homogeneous)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary -> {args.out}")
    print(f"\n<=30 search accuracy (no FDR):")
    for b in branches:
        g = out["branches_le30"][b]
        print(f"  {b:<24} top1={g['top1_acc']:.4f} top3={g['top3_acc']:.4f} "
              f"top10={g['top10_acc']:.4f} hits={g['hits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
