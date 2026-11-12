#!/usr/bin/env python3
"""Analyze a step4 results CSV: exact-match vs mass-match accuracy per branch.

Exact match = I/L-normalized string equality (conservative peptide-ID, the
convention database-search tools like DeepSearch count).
Mass match  = the V20 mass-matching rule used for the `hit` column (lenient:
0.5 Da cumulative / 0.1 Da per-residue AA-mass alignment).

The gap (mass% - exact%) shows how many 'hits' are mass-similar-but-DIFFERENT
sequences that the lenient match counts as correct.  Compare exact% against
DeepSearch-style numbers.

Branches are auto-discovered from the CSV header (prefixes like a, b005, b010,
c, d005, ..., e, f005, ...).

Usage:
    python analyze_results.py output/mouse/ppm5/results_step4_mouse.csv
"""

from __future__ import annotations

import csv
import sys


def il_norm(s: str) -> str:
    return s.replace("I", "L")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "output/mouse/results_step4_mouse.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = len(rows)
    if n == 0:
        print("no rows")
        return 1
    header = set(rows[0].keys())
    prefixes = sorted({k[:-len("_hit")] for k in header if k.endswith("_hit")})
    gt_in = sum(1 for r in rows if r.get("gt_in_candidates", "0") == "1")

    print(f"file: {path}")
    print(f"spectra: {n}   gt_in_candidates: {gt_in}/{n} = {gt_in / n:.4f}\n")
    print(f"{'branch':<16}{'exact%':>8}{'mass%':>8}{'lenient-only':>13}")
    for pre in prefixes:
        exact = mass = 0
        lenient_examples = []
        for r in rows:
            gt = r.get("gt_seq", "")
            best = r.get(f"{pre}_best_seq", "")
            hit = r.get(f"{pre}_hit", "0") == "1"
            if not gt or not best:
                continue
            if hit:
                mass += 1
            if il_norm(gt) == il_norm(best):
                exact += 1
            elif hit:
                lenient_examples.append((r.get("ordinal"), gt, best))
        print(f"{pre:<16}{exact / n * 100:>7.1f}%{mass / n * 100:>7.1f}%{mass - exact:>13}")
        if lenient_examples:
            print(f"    lenient-only examples: {lenient_examples[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
