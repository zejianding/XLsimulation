#!/usr/bin/env python3
"""Select high-similarity ordinals for viz_candidates.py.

Reads a results_step4_<sp>.csv, keeps rows where a chosen branch is a HIT
(true peptide ranked #1), and prints the top-N by a chosen branch score.
Lets you eyeball good real-vs-predicted matches instead of hunting ordinals.

Branch CSV prefixes (minimal mode, rel R -> pp = 3 digits):
  d_spec_rel<R>     -> d{pp}          (d{pp}_score / d{pp}_gt_score / d{pp}_hit)
  f_frag_rel<R>     -> f{pp}
  m_by_prec_rel<R>  -> mp{pp}
  hs_d<R>_f<R>      -> hdf{pp}_{pp}
  hs_d<R>_f<R>_mby  -> hdf{pp}_{pp}_mby
  soft_ppm          -> sppm ; comp_by -> cby ; sat_consistency -> sat

Usage:
  python pick_top_scans.py \
      --results output/mouse/full_sqrtclip/results_step4_mouse.csv \
      --hit-prefix hdf020_020_mby --sort-prefix d020 --gt-score --top 10
  (--gt-score sorts by the TRUE peptide's score d{pp}_gt_score; without it,
   sorts by the branch's best score d{pp}_score.)
"""

from __future__ import annotations

import argparse
import csv
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True, help="results_step4_<sp>.csv path")
    p.add_argument("--hit-prefix", default="hdf020_020_mby",
                   help="branch prefix whose _hit==1 rows are kept")
    p.add_argument("--sort-prefix", default="d020",
                   help="branch prefix to sort by (score or gt_score)")
    p.add_argument("--gt-score", action="store_true",
                   help="sort by the TRUE candidate's score ({prefix}_gt_score) "
                        "instead of the branch best ({prefix}_score)")
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(open(args.results, encoding="utf-8")))
    hit_col = f"{args.hit_prefix}_hit"
    if hit_col not in rows[0]:
        print(f"column {hit_col!r} not in CSV")
        return 1
    score_col = f"{args.sort_prefix}_{'gt_score' if args.gt_score else 'score'}"
    if score_col not in rows[0]:
        print(f"column {score_col!r} not in CSV")
        return 1

    hits = [r for r in rows if r.get(hit_col) == "1"]
    scored = []
    for r in hits:
        try:
            scored.append((float(r[score_col]), r))
        except (TypeError, ValueError):
            continue
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"kept {len(hits)} hits (of {len(rows)} spectra) by {hit_col}==1, "
          f"sorted by {score_col}\n")
    print(f"{'ordinal':>8} {'score':>8}  gt_seq")
    for s, r in scored[: args.top]:
        print(f"{int(r['ordinal']):>8} {s:>8.4f}  {r.get('gt_seq', '')[:40]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
