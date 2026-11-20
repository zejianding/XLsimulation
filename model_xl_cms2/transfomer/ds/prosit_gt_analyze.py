#!/usr/bin/env python3
"""Aggregate Prosit results on the GT-family pool into the two metrics.

Reads the per-spectrum results CSV from step4_prosit_score.py run with a
GT-family candidate pool (build_gt_family_candidates.py):

  * GT fidelity (A):  the <branch>_gt_score column = Prosit's prediction of the
      true peptide (b/y filtered, rel-weighted) vs the experimental spectrum.
      Reported as mean/median and pass-rate >= thresholds.
  * family top1 (B):  the <branch>_hit / _gt_rank columns = whether the correct
      variant of the true-peptide family ranks first within that spectrum's pool.

Note: hit/rank use the V20 I/L-equivalent mass-matching rule, so isobaric
variants (same oxidation count on a different M-site, I/L swaps) all count as
matching the GT — metric B therefore reflects neutral-mass-family discrimination
rather than exact mod-site placement.

Usage:
    python prosit_gt_analyze.py --species mouse \
        --csv output/prosit_gt/mouse/results_prosit_mouse.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--csv", required=True, help="results_prosit_<sp>.csv on the GT-family pool")
    p.add_argument("--branches", default="d020,f020,hdf020_020,hdf020_020_mby",
                   help="comma-separated branch prefixes to report")
    p.add_argument("--out", default=None, help="summary JSON to write")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    with open(args.csv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        print("empty csv")
        return 1
    header = set(rows[0].keys())
    missing = [b for b in branches if f"{b}_gt_score" not in header]
    if missing:
        print(f"branches missing in csv: {missing}; columns example: "
              f"{sorted(h for h in header if h.endswith('_gt_score'))[:10]}")
    # denominator = spectra whose GT family was scored (all rows have GT in pool)
    denom = len(rows)
    print(f"{args.species}: spectra_scored={denom}")

    def flt(r, col):
        try:
            return float(r.get(col, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    summary = {"species": args.species, "n_spectra": denom,
               "metric": {"A_gt_fidelity": "mean/median/pass-rate of <branch>_gt_score",
                          "B_family_top1": "fraction with <branch>_hit==1 (GT family rank 1)"},
               "branches": {}}
    for b in branches:
        gcol, hcol, rcol = f"{b}_gt_score", f"{b}_hit", f"{b}_gt_rank"
        if gcol not in header:
            continue
        gts = [flt(r, gcol) for r in rows]
        gts = [g for g in gts if g >= 0.0]  # -1 = GT not found / unscored
        hits = sum(1 for r in rows
                   if r.get(hcol, "").strip() in ("1", "True", "true"))
        rank1 = rank3 = 0
        for r in rows:
            try:
                rk = int(float(r.get(rcol, "0")))
            except (TypeError, ValueError):
                rk = 0
            rank1 += rk == 1
            rank3 += 0 < rk <= 3
        g = {
            "gt_fidelity": {
                "mean": round(sum(gts) / len(gts), 4) if gts else None,
                "median": round(sorted(gts)[len(gts) // 2], 4) if gts else None,
                "pass_rate": {f">={t:.2f}": round(sum(1 for x in gts if x >= t) / len(gts), 4)
                              for t in THRESHOLDS} if gts else {},
            },
            "family_top1": {
                "hit_rate": round(hits / denom, 4) if denom else None,
                "gt_rank_le1": round(rank1 / denom, 4) if denom else None,
                "gt_rank_le3": round(rank3 / denom, 4) if denom else None,
            },
        }
        summary["branches"][b] = g
        print(f"\n  branch {b}:")
        print(f"    [A fidelity] mean={g['gt_fidelity']['mean']} "
              f"median={g['gt_fidelity']['median']}")
        print("    pass_rate: " + "  ".join(
            f"{k}={v:.3f}" for k, v in g["gt_fidelity"]["pass_rate"].items()))
        print(f"    [B family top1] hit_rate={g['family_top1']['hit_rate']} "
              f"rank<=3={g['family_top1']['gt_rank_le3']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nsummary -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
