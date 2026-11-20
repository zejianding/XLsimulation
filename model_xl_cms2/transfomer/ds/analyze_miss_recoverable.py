#!/usr/bin/env python3
"""Diagnose whether the rank-2/3 misses of a branch are recoverable.

For each spectrum where the GT ranks 2 or 3 (not a hit), measure:
  1. TRUE_BY_COVERAGE : fraction of the TRUE peptide's theoretical main b/y ions
     that actually appear in the experimental spectrum (within tol).  If LOW,
     the spectrum is b/y-sparse -> b/y-based signals (complementary pairs etc.)
     will likely CONFLICT rather than help.
  2. TOP1_TRUE_OVERLAP : fraction of the top-1 candidate's theoretical main b/y
     ions that coincide with the TRUE peptide's b/y ions.  If HIGH, the two
     candidates are almost indistinguishable by b/y -> unlikely recoverable.

Reads the MGF once, keys spectra by ordinal, then iterates the miss rows.

Usage:
    python analyze_miss_recoverable.py \
        --results output/mouse/full/results_step4_mouse.csv \
        --mgf /home/wangyuhang/seq2mass/v1-search/mouse/high.mouse.PXD004948_len7_32_mod_within_scope_10000.mgf \
        --branch hdf020_020 --max-misses 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter

import numpy as np

import ds_common as dc
from predict_mgf import generate_by_ions

TOL = 0.05


def main_by_mzs(seq: str) -> list[float]:
    """Theoretical main b/y ion m/z positions (token 23, no offset)."""
    return sorted(float(ion["mz"]) for ion in generate_by_ions(seq, max_charge=3))


def coverage(exp_mz: np.ndarray, theo_mz: np.ndarray, tol: float = TOL) -> float:
    """Fraction of theo m/z positions that have an exp peak within tol."""
    if theo_mz.size == 0:
        return 0.0
    exp_sorted = np.sort(exp_mz)
    lo = np.searchsorted(exp_sorted, theo_mz - tol, side="left")
    hi = np.searchsorted(exp_sorted, theo_mz + tol, side="right")
    return float((hi > lo).mean())


def overlap(mz_a: list[float], mz_b: list[float], tol: float = TOL) -> float:
    """Fraction of A's m/z within tol of some B m/z."""
    if not mz_a:
        return 0.0
    arr_b = np.array(sorted(mz_b), dtype=np.float64)
    arr_a = np.array(mz_a, dtype=np.float64)
    lo = np.searchsorted(arr_b, arr_a - tol, side="left")
    hi = np.searchsorted(arr_b, arr_a + tol, side="right")
    return float((hi > lo).mean())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True, help="results_step4_<sp>.csv path")
    p.add_argument("--mgf", required=True, help="source MGF path")
    p.add_argument("--branch", default="hdf020_020",
                   help="branch csv prefix to analyze (e.g. hdf020_020)")
    p.add_argument("--max-misses", type=int, default=100,
                   help="cap on how many rank2-3 misses to analyze (0 = all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pre = args.branch
    rows = list(csv.DictReader(open(args.results, encoding="utf-8")))
    if f"{pre}_hit" not in rows[0]:
        print(f"branch prefix {pre!r} not in CSV columns")
        return 1

    # load MGF once -> ordinal -> peaks
    mgf_peaks: dict[int, np.ndarray] = {}
    for ordinal, (_meta, peaks) in enumerate(dc.iter_mgf_spectra(args.mgf)):
        mgf_peaks[ordinal] = peaks
    print(f"loaded {len(mgf_peaks)} spectra from MGF")

    misses = [r for r in rows
              if r.get(f"{pre}_hit") == "0" and r.get(f"{pre}_gt_rank") in ("2", "3")]
    if args.max_misses:
        misses = misses[: args.max_misses]
    print(f"analyzing {len(misses)} rank2-3 misses\n")

    true_cov, top_overlap, gt_in_mgf = [], [], 0
    examples = []
    for r in misses:
        ordinal = int(r["ordinal"])
        gt = r["gt_seq"]
        top1 = r.get(f"{pre}_best_seq", "")
        peaks = mgf_peaks.get(ordinal)
        if peaks is None or peaks.size == 0:
            continue
        exp_mz = peaks[:, 0]
        true_mzs = main_by_mzs(gt)
        top1_mzs = main_by_mzs(top1)
        c = coverage(exp_mz, np.array(true_mzs))
        o = overlap(top1_mzs, true_mzs)
        true_cov.append(c)
        top_overlap.append(o)
        examples.append((ordinal, c, o, gt[:20], top1[:20]))
        gt_in_mgf += 1

    if not true_cov:
        print("no analyzable misses")
        return 1

    arr_c = np.array(true_cov)
    arr_o = np.array(top_overlap)
    print(f"{'metric':<24}{'mean':>8}{'median':>8}{'p25':>8}{'p75':>8}")
    print(f"{'TRUE b/y coverage':<24}{arr_c.mean():>8.3f}{np.median(arr_c):>8.3f}"
          f"{np.percentile(arr_c,25):>8.3f}{np.percentile(arr_c,75):>8.3f}")
    print(f"{'TOP1-TRUE b/y overlap':<24}{arr_o.mean():>8.3f}{np.median(arr_o):>8.3f}"
          f"{np.percentile(arr_o,25):>8.3f}{np.percentile(arr_o,75):>8.3f}")

    # bucket
    cov_bucket = Counter(("low<0.5" if c < 0.5 else "high>=0.5") for c in true_cov)
    ov_bucket = Counter(("sep<0.8" if o < 0.8 else "amb>=0.8") for o in top_overlap)
    print(f"\ntrue b/y coverage buckets: {dict(cov_bucket)}")
    print(f"top1-true overlap buckets : {dict(ov_bucket)}")

    print("\nexamples (ordinal, true_cov, overlap, GT, top1):")
    for e in examples[:10]:
        print(f"  ord={e[0]:>5}  true_cov={e[1]:.2f}  overlap={e[2]:.2f}  "
              f"GT={e[3]:<22} top1={e[4]:<22}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
