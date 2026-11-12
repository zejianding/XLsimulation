#!/usr/bin/env python3
"""Quantify adjacent-swap (near-isobaric) isomer interference.

Two peptides with the same residue multiset but adjacent positions swapped
("...XY..." vs "...YX...") have the same neutral mass, so step3 puts BOTH
into the same spectrum's candidate list.  Their predicted spectra differ
only in the b/y ions spanning the swapped boundary.  This script measures
how often such isomers actually co-occur, and whether a scoring branch
still ranks the ground truth above them.

An "adjacent-swap isomer" here = same (residue, delta) multiset, differing
by exactly one adjacent transposition, with the same neutral mass.  Swaps
between mass-identical residues (I/L, which GT matching already treats as
equivalent) are counted separately and excluded from the "challenging"
subset, because they can never be an error.

For each spectrum (joined by ordinal between the results CSV and the
step3 candidates TSV.gz), it reports:
  - how many spectra have >=1 swap-isomer candidate (co-occurrence)
  - branch top1_acc on the challenging-swap subset vs the rest
  - how many branch misses had a challenging swap present
  - how often the top-1 candidate IS a swap isomer (GT lost to a swap)

Usage:
    python analyze_swap_isomers.py \
        --results output/mouse/full_sqrtclip/results_step4_mouse.csv \
        --candidates data/mouse/candidates_mouse_full.tsv.gz \
        --prefix hdf020_020_mby
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys

import numpy as np

import ds_common as dc


def tokens(annoseq: str) -> list[tuple[str, float]]:
    residues, deltas, _ = dc.parse_modified_sequence(annoseq)
    return list(zip(residues, deltas))


def token_mass(t: tuple[str, float]) -> float:
    aa, delta = t
    return dc.AA_MASS.get(aa, 0.0) + delta


def is_adjacent_swap(a: list[tuple[str, float]],
                     b: list[tuple[str, float]]) -> bool:
    """True if ``b`` equals ``a`` with exactly one adjacent token swap."""
    if len(a) != len(b):
        return False
    n = len(a)
    for i in range(n):
        if a[i] != b[i]:
            if (i + 1 < n and a[i] == b[i + 1] and a[i + 1] == b[i]
                    and a[i + 2:] == b[i + 2:]):
                return True
            return False
    return False  # identical sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True, help="results_step4_<sp>.csv")
    p.add_argument("--candidates", required=True, help="step3 TSV.gz")
    p.add_argument("--prefix", default="hdf020_020_mby",
                   help="branch CSV prefix (hdf020_020_mby = hs_d020_f020_mby)")
    p.add_argument("--max-scans", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    hit_col = f"{args.prefix}_hit"
    best_col = f"{args.prefix}_best_seq"
    rank_col = f"{args.prefix}_gt_rank"

    rows = list(csv.DictReader(open(args.results, encoding="utf-8")))
    if hit_col not in rows[0]:
        print(f"column {hit_col!r} not in results CSV")
        return 1

    # ordinal -> candidate sequences (join key is ordinal, see step4)
    cand_by_ord: dict[int, list[str]] = {}
    with gzip.open(args.candidates, "rt", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd, None)
        for row in rd:
            if not row:
                continue
            cand_by_ord.setdefault(int(row[0]), []).append(row[6])
    print(f"loaded candidates for {len(cand_by_ord)} ordinals")

    n_total = n_with_swap = n_chall = 0
    n_miss_with_swap = n_miss_with_chall = 0
    n_lost_to_swap = n_lost_to_chall = 0
    hit_total = hit_swap = hit_chall = 0
    rank_chall: list[int] = []

    for r in rows:
        ordinal = int(r["ordinal"])
        gt = r.get("gt_seq", "")
        if not gt:
            continue
        n_total += 1
        cands = cand_by_ord.get(ordinal, [])
        gt_tok = tokens(gt)
        gt_mass = dc.compute_neutral_mass_annotated(gt)
        swap_seqs: list[str] = []
        chall_seqs: list[str] = []
        for c in cands:
            if c == gt:
                continue
            ct = tokens(c)
            if not is_adjacent_swap(gt_tok, ct):
                continue
            if abs(dc.compute_neutral_mass_annotated(c) - gt_mass) > 1e-3:
                continue  # mass changed -> not a pure adjacent swap
            swap_seqs.append(c)
            # is the swapped pair mass-identical (I/L etc.)? then GT matching
            # already accepts it -> not "challenging"
            for i in range(len(gt_tok) - 1):
                if (gt_tok[i] == ct[i + 1] and gt_tok[i + 1] == ct[i]
                        and abs(token_mass(gt_tok[i]) - token_mass(ct[i])) < 1e-4
                        and abs(token_mass(gt_tok[i + 1]) - token_mass(ct[i + 1])) < 1e-4):
                    break
            else:
                chall_seqs.append(c)

        hit = r.get(hit_col) == "1"
        best = r.get(best_col, "")
        if hit:
            hit_total += 1
        if swap_seqs:
            n_with_swap += 1
            if hit:
                hit_swap += 1
            else:
                n_miss_with_swap += 1
            if best in swap_seqs:
                n_lost_to_swap += 1
        if chall_seqs:
            n_chall += 1
            if hit:
                hit_chall += 1
            else:
                n_miss_with_chall += 1
            if best in chall_seqs:
                n_lost_to_chall += 1
            try:
                rank_chall.append(int(r[rank_col]))
            except (TypeError, ValueError):
                pass

    def acc(hits, denom) -> float:
        return round(hits / denom, 4) if denom else 0.0

    print(f"\nspecies-wide top1_acc [{args.prefix}]: {acc(hit_total, n_total)} "
          f"({hit_total}/{n_total})")
    print(f"\nadjacent-swap co-occurrence:")
    print(f"  spectra with >=1 swap-isomer candidate : {n_with_swap} "
          f"({n_with_swap/n_total:.3%} of {n_total})")
    print(f"  ... of which 'challenging' (non I/L-equal): {n_chall} "
          f"({n_chall/n_total:.3%})")
    print(f"\ntop1_acc on subsets:")
    print(f"  with a swap candidate present : {acc(hit_swap, n_with_swap)} "
          f"({hit_swap}/{n_with_swap})")
    print(f"  with a CHALLENGING swap present: {acc(hit_chall, n_chall)} "
          f"({hit_chall}/{n_chall})")
    print(f"  without any swap candidate     : "
          f"{acc(hit_total - hit_swap, n_total - n_with_swap)} "
          f"({hit_total - hit_swap}/{n_total - n_with_swap})")
    print(f"\nmisses explained by swaps:")
    n_miss = n_total - hit_total
    print(f"  total branch misses             : {n_miss}")
    print(f"  misses with a swap candidate    : {n_miss_with_swap}")
    print(f"  misses with a CHALLENGING swap  : {n_miss_with_chall}")
    print(f"  top-1 IS a swap isomer          : {n_lost_to_swap}")
    print(f"  top-1 IS a CHALLENGING swap     : {n_lost_to_chall}")
    if rank_chall:
        arr = np.array(rank_chall)
        print(f"\nGT rank on challenging-swap spectra: "
              f"mean={arr.mean():.2f} median={np.median(arr):.0f} "
              f"p95={np.percentile(arr, 95):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
