#!/usr/bin/env python3
"""Re-evaluate a results CSV with STRICT string matching (exact chars except I/L).

For each branch, recompute top-1 accuracy where the best candidate must be
EXACTLY equal to the ground truth after replacing I->L on both sides (all other
characters — including modification annotations like C+57.021 / M+15.995 —
must match exactly).  Compare against the stored mass-matching `_hit` column:

    strict%      = il_norm(best_seq) == il_norm(gt_seq)   (exact except I/L)
    mass%        = the stored `_hit` column (V20 mass matching)
    lenient-only = mass hits that are NOT strict matches
                   (i.e. mass-lenient but string-different — the '水分')

Usage:
    python strict_match_check.py output/mouse/full/results_step4_mouse.csv
"""

from __future__ import annotations

import csv
import sys


def il_norm(s: str) -> str:
    return s.replace("I", "L")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "output/mouse/full/results_step4_mouse.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = len(rows)
    if n == 0:
        print("no rows")
        return 1
    header = set(rows[0].keys())
    prefixes = sorted({k[: -len("_hit")] for k in header if k.endswith("_hit")})

    print(f"file: {path}")
    print(f"spectra: {n}  (accuracy denominator = all rows with GT)\n")
    print(f"{'branch':<18}{'strict%':>9}{'mass%':>8}{'lenient-only':>13}")
    for pre in prefixes:
        strict = mass = 0
        examples = []
        for r in rows:
            gt = r.get("gt_seq", "")
            best = r.get(f"{pre}_best_seq", "")
            hit = r.get(f"{pre}_hit", "0") == "1"
            if not gt or not best:
                continue
            if hit:
                mass += 1
            if gt and best and il_norm(gt) == il_norm(best):
                strict += 1
            elif hit:
                examples.append((r.get("ordinal"), gt, best))
        print(f"{pre:<18}{strict / n * 100:>8.1f}%{mass / n * 100:>7.1f}%{mass - strict:>13}")
        if examples:
            print(f"    lenient-only examples: {examples[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
