#!/usr/bin/env python3
"""DS step4 candidates_detail -> search accuracy on the <=30aa pool (fair vs Prosit).

Koina Prosit accepts backbones of at most 30 residues, so the V20 branches
(``hs_d020_f020`` = hsdf, ``..._mby`` = hsdf_mby, and the single d/f/m_by_prec
branches) must be re-ranked on the per-scan pool restricted to <=30aa and the
denominator limited to the spectra Prosit itself can score.  This script does
that purely as post-processing of ``--save-detail`` output
(``candidates_detail_<sp>.csv.gz``) -- no GPU, no re-prediction.

Semantics replicated from the runs themselves:
  * per (ordinal, branch) candidates are sorted by score descending
    (argsort(-score)); best = top; hit = best matches GT; gt_rank = position
    of the best GT-matching candidate (I/L-equivalent mass matching).
  * fusion hsdf / hsdf_mby are NOT taken from the stored rows: their min-max
    normalization is per-pool, so stored full-32 fusion scores would not equal
    a <=30-pool fusion.  They are rebuilt from the single-branch scores stored
    in the same file (``d_spec_rel<lvl>`` + ``f_frag_rel<lvl>`` [+ ``m_by_prec_rel<lvl>``])
    over the kept pool -- exactly the pool-restricted definition.
  * denominator n_with_gt = spectra with a GT that were processed with a
    non-empty pool (after the length restriction), matching ds step4 and the
    Prosit script; spectra whose GT backbone is >30aa stay in the denominator
    only as misses, exactly as the Prosit run treats them.
  * top3/top10_acc = fraction of n_with_gt with gt_rank <= 3 / <= 10.

Both the full (<=32) and the <=30 numbers are computed in one pass; the full
line validates against the original step4 summary.

Usage:
    python detail_len30_accuracy.py \
        --detail <out>/candidates_detail_<sp>.csv.gz \
        --mgf <root>/<sp>/high.<sp>...mgf \
        --out <out>/len30_accuracy_<sp>.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

_SELF = Path(__file__).resolve().parent
for _p in (_SELF, _SELF.parent / "transfomer" / "ds"):
    if (_p / "ds_common.py").exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ds_common as dc

NUMERIC_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def backbone_len(seq: str) -> int:
    core = re.sub(r"\[[^\]]*\]", "", seq)
    core = NUMERIC_RE.sub("", core)
    return sum(1 for ch in core if ch.isalpha())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--detail", required=True, help="step4 candidates_detail_<sp>.csv.gz")
    p.add_argument("--mgf", required=True, help="source MGF (SEQ= ground truth)")
    p.add_argument("--out", required=True, help="summary JSON to write")
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--species", default="")
    args = p.parse_args()

    # ordinal -> gt_seq
    mgf_gt: dict[int, str] = {}
    for ordinal, (meta, _pk) in enumerate(dc.iter_mgf_spectra(args.mgf)):
        gt = meta.get("SEQ", "")
        if gt:
            mgf_gt[ordinal] = gt

    # per ordinal -> cand_idx -> seq  (candidate identity is shared across branches)
    cands: dict[int, list[str]] = {}
    # per ordinal -> base-branch score per candidate: {branch_key: [score,...]}
    base: dict[int, dict[str, list[float]]] = {}
    other: dict[int, dict[str, list[float]]] = {}  # non-base branches (stored scores)
    with gzip.open(args.detail, "rt", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd, None)
        if header is None or header[0] != "ordinal":
            print(f"unexpected header: {header}")
            return 1
        for row in rd:
            if len(row) < 7:
                continue
            ordinal, branch = int(row[0]), row[1]
            idx = int(row[2])
            seq = row[3]
            try:
                score = float(row[5])
            except ValueError:
                continue
            if ordinal not in cands:
                cands[ordinal] = []
                base[ordinal] = {}
                other[ordinal] = {}
            while len(cands[ordinal]) <= idx:
                cands[ordinal].append("")
            if cands[ordinal][idx] == "":
                cands[ordinal][idx] = seq
            if re.match(r"^(d_spec_rel|f_frag_rel|m_by_prec_rel)\d{3}$", branch):
                base[ordinal].setdefault(branch, [None] * len(cands[ordinal]))
                arr = base[ordinal][branch]
                while len(arr) <= idx:
                    arr.append(None)
                arr[idx] = score
            else:
                other[ordinal].setdefault(branch, [None] * len(cands[ordinal]))
                arr = other[ordinal][branch]
                while len(arr) <= idx:
                    arr.append(None)
                arr[idx] = score
    print(f"detail: {len(cands)} spectra, "
          f"{sorted({k for d in base.values() for k in d})} base branches")

    def pool_scores(ordinal: int, branch: str, max_len: int | None):
        """[score,...] over the kept pool; returns None if the pool is empty."""
        kept = [i for i, s in enumerate(cands[ordinal])
                if max_len is None or backbone_len(s) <= max_len]
        if not kept:
            return None
        if branch in base[ordinal]:
            vals = [base[ordinal][branch][i] for i in kept]
            if any(v is None for v in vals):
                return None
            return vals
        if branch in other[ordinal]:
            vals = [other[ordinal][branch][i] for i in kept]
            if any(v is None for v in vals):
                return None
            return vals
        return None

    # fusion base keys present per level (same level on both sides in minimal set)
    rel_levels = sorted({re.match(r"^d_spec_rel(\d{3})$", k).group(1)
                         for d in base.values() for k in d
                         if re.match(r"^d_spec_rel\d{3}$", k)})
    fused: dict[str, list[str]] = {}
    for lvl in rel_levels:
        fb = [f"d_spec_rel{lvl}", f"f_frag_rel{lvl}"]
        if all(any(f in d for d in base.values()) for f in fb):
            fused[f"hs_d{lvl}_f{lvl}"] = fb
        mfb = fb + [f"m_by_prec_rel{lvl}"]
        if all(any(f in d for d in base.values()) for f in mfb):
            fused[f"hs_d{lvl}_f{lvl}_mby"] = mfb
    branch_keys = sorted({k for d in base.values() for k in d}
                         | {k for d in other.values() for k in d}
                         | set(fused))

    def score_avg(vals_per_cand: list[list[float]]) -> list[float]:
        n = len(vals_per_cand[0])
        acc = [0.0] * n
        for comp in vals_per_cand:
            lo, hi = min(comp), max(comp)
            if hi - lo > 1e-12:
                for i, v in enumerate(comp):
                    acc[i] += (v - lo) / (hi - lo)
        return [a / len(vals_per_cand) for a in acc]

    def run(max_len: int | None) -> dict:
        denom = 0
        agg: dict[str, dict] = {}
        for b in branch_keys:
            agg[b] = {"hits": 0, "ranks": []}
        for ordinal in sorted(cands):
            gt_seq = mgf_gt.get(ordinal, "")
            pool = [i for i, s in enumerate(cands[ordinal])
                    if max_len is None or backbone_len(s) <= max_len]
            if not pool:
                continue
            if gt_seq:
                denom += 1
            idx_map = {old: new for new, old in enumerate(pool)}
            kept = [cands[ordinal][i] for i in pool]
            for b in branch_keys:
                if b in fused:
                    comps = []
                    ok = True
                    for fk in fused[b]:
                        vals = pool_scores(ordinal, fk, max_len)
                        if vals is None:
                            ok = False
                            break
                        comps.append(vals)
                    if not ok:
                        continue
                    scores = score_avg(comps)
                else:
                    vals = pool_scores(ordinal, b, max_len)
                    if vals is None:
                        continue
                    scores = vals
                order = sorted(range(len(pool)), key=lambda i: -scores[i])
                hit = gt_rank = 0
                if gt_seq:
                    gt_idx = None
                    gt_score = -float("inf")
                    for i in order:
                        if dc.peptide_matches(kept[i], gt_seq):
                            sc = scores[i]
                            if sc > gt_score:
                                gt_score = sc
                                gt_idx = i
                    if gt_idx is not None:
                        gt_rank = next(r for r, i in enumerate(order, 1) if i == gt_idx)
                        hit = int(dc.peptide_matches(kept[order[0]], gt_seq))
                agg[b]["hits"] += hit
                if gt_rank > 0:
                    agg[b]["ranks"].append(gt_rank)
        out = {"n_with_gt": denom, "branches": {}}
        for b in branch_keys:
            ranks = agg[b]["ranks"]
            g = {"hits": agg[b]["hits"],
                 "top1_acc": agg[b]["hits"] / denom if denom else 0.0,
                 "top3_acc": sum(1 for r in ranks if r <= 3) / denom if denom else 0.0,
                 "top10_acc": sum(1 for r in ranks if r <= 10) / denom if denom else 0.0}
            out["branches"][b] = g
        return out

    def verify_fusions():
        """Rebuilt fusion scores must equal the stored full-pool fusion rows."""
        n_bad = 0
        max_diff = 0.0
        for ordinal in cands:
            for b in fused:
                if b not in other[ordinal]:
                    continue
                comps = []
                for fk in fused[b]:
                    vals = pool_scores(ordinal, fk, None)
                    if vals is None:
                        break
                    comps.append(vals)
                if len(comps) != len(fused[b]):
                    continue
                rebuilt = score_avg(comps)
                stored = other[ordinal][b]
                for r, s in zip(rebuilt, stored):
                    if s is None:
                        continue
                    d = abs(r - s)
                    max_diff = max(max_diff, d)
                    if d > 1e-6:
                        n_bad += 1
        print(f"fusion rebuild vs stored rows: mismatches={n_bad} max_diff={max_diff:.3e}")
        return n_bad, max_diff

    full = run(None)
    verify_fusions()
    le30 = run(args.max_len)
    summary = {
        "species": args.species,
        "source": str(args.detail),
        "max_len": args.max_len,
        "n_spectra": len(cands),
        "n_with_gt_full": full["n_with_gt"],
        "n_with_gt_le30": le30["n_with_gt"],
        "n_excluded_gt_gt30": full["n_with_gt"] - le30["n_with_gt"],
        "branches_le30": le30["branches"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"summary -> {args.out}")

    print(f"\nfull (<=32): n_with_gt = {full['n_with_gt']}")
    for b, g in full["branches"].items():
        print(f"  {b:<26} top1={g['top1_acc']:.4f} top3={g['top3_acc']:.4f} "
              f"top10={g['top10_acc']:.4f} hits={g['hits']}")
    print(f"\nlen<={args.max_len}: n_with_gt = {le30['n_with_gt']} "
          f"(excluded GT>{args.max_len}: {full['n_with_gt'] - le30['n_with_gt']})")
    for b, g in le30["branches"].items():
        print(f"  {b:<26} top1={g['top1_acc']:.4f} top3={g['top3_acc']:.4f} "
              f"top10={g['top10_acc']:.4f} hits={g['hits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
