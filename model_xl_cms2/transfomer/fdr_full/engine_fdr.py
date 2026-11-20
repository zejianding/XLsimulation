#!/usr/bin/env python3
"""Generic 1% FDR counting for a search-engine PSM table (ground-truth free).

Input is a tab-separated PSM table from MS-GF+ (MzidToTsvConverter) or MSFragger.
Every PSM row has a peptide, a confidence score (lower = better: MS-GF+
``SpecEValue``, MSFragger ``expectscore``) and a decoy marker in the protein
accession (``rev_`` prefix when searching our rev_ target-decoy FASTA; ``XXX``
when MS-GF+ used -tda 1).

Method (paper formula, +1 correction): sort all PSMs by score ascending,
accumulate decoy/target counts, FDR_i = min_{j<=i} (n_decoy_j + 1) / n_target_j.
The 1% line = the largest high-scoring prefix with FDR <= target.  Report
``n_target`` (identifications at the FDR line — the headline number) and
``n_decoy`` (should be ~target% of n_target).  No comparison to any GT is done.

Usage:
    # MS-GF+ tsv (searching our rev_ TD FASTA)
    python engine_fdr.py --engine msgfplus --tsv out/yeast.tsv \
        --decoy-token rev_ --fdr 0.01
    # MSFragger tsv
    python engine_fdr.py --engine msfragger --tsv mouse10000_fixed.tsv \
        --decoy-token rev_ --fdr 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# per-engine defaults: (score column, scan column, lower-better)
ENGINE = {
    "msgfplus": {"score": "SpecEValue", "scan": "ScanNum", "lower_better": True,
                 "peptide": "Peptide", "protein": "Protein"},
    "msfragger": {"score": "expectscore", "scan": "scannum", "lower_better": True,
                  "peptide": "peptide", "protein": "proteins"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tsv", required=True, help="search-engine PSM tsv")
    p.add_argument("--engine", default="msgfplus",
                   choices=list(ENGINE), help="engine defaults for score/protein columns")
    p.add_argument("--score", default=None,
                   help="score column (default = engine default)")
    p.add_argument("--scan", default=None,
                   help="scan column for per-spectrum dedup (default = engine default; "
                        "set '' to count every PSM row instead)")
    p.add_argument("--protein", default=None,
                   help="protein/accession column (default = engine default)")
    p.add_argument("--decoy-token", default="rev_",
                   help="marker in protein accession identifying decoys (rev_ or XXX)")
    p.add_argument("--fdr", type=float, default=0.01, help="FDR line (default 0.01)")
    p.add_argument("--max-len", type=int, default=None,
                   help="only count PSMs whose peptide backbone <= max-len "
                        "(e.g. 30 to align with Prosit); others are dropped")
    p.add_argument("--out-json", default=None, help="optional JSON summary output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eng = ENGINE[args.engine]
    score_col = args.score or eng["score"]
    scan_col = args.scan if args.scan is not None else eng["scan"]
    prot_col = args.protein or eng["protein"]

    # best PSM per scan (dedupe I/L unroll rows): lower score wins
    best_scan: dict[str, tuple[float, bool]] = {}
    n_rows = 0
    for r in csv.DictReader(open(args.tsv, encoding="utf-8", errors="ignore"),
                            delimiter="\t"):
        if score_col not in r:
            print(f"column {score_col!r} not in tsv (engine {args.engine})")
            return 1
        try:
            sc = float(r[score_col])
        except (TypeError, ValueError):
            continue
        n_rows += 1
        if args.max_len is not None:
            pep = r.get(eng.get("peptide", "Peptide"), "") or ""
            if sum(1 for ch in pep if ch.isalpha()) > args.max_len:
                continue
        decoy = args.decoy_token in (r.get(prot_col, "") or "")
        if scan_col:
            key = r.get(scan_col, "")
            cur = best_scan.get(key)
            if cur is None or sc < cur[0]:
                best_scan[key] = (sc, decoy)
        else:
            best_scan[str(n_rows)] = (sc, decoy)
    psms = list(best_scan.values())
    if not psms:
        print("no parseable PSMs")
        return 1
    # lower-better for both engine E-values
    psms.sort(key=lambda x: x[0])

    n_t = n_d = 0
    best = None  # (n_target, n_decoy, fdr) at loosest prefix with fdr <= target
    for i, (sc, decoy) in enumerate(psms):
        if decoy:
            n_d += 1
        else:
            n_t += 1
        fdr = (n_d + 1) / max(n_t, 1)
        if fdr <= args.fdr:
            best = (i, n_t, n_d, fdr)
    if best is None:
        print(f"no PSM prefix passes {args.fdr:.0%} FDR")
        return 1
    cut, n_target, n_decoy, real_fdr = best
    threshold = psms[cut][0]

    print(f"PSM rows: {n_rows}  ->  best-per-scan spectra: {len(psms)}")
    print(f"{args.fdr:.0%} FDR ({args.engine}, decoy='{args.decoy_token}', "
          f"score='{score_col}'):")
    print(f"  threshold ({score_col} <=) = {threshold:.3e}")
    print(f"  n_target = {n_target}   n_decoy = {n_decoy}   "
          f"achieved FDR = {real_fdr:.4%}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "engine": args.engine, "fdr": args.fdr, "score_col": score_col,
            "decoy_token": args.decoy_token, "n_psms": len(psms),
            "threshold": threshold, "n_target": n_target,
            "n_decoy": n_decoy, "achieved_fdr": real_fdr,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary -> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
