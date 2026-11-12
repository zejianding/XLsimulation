#!/usr/bin/env python3
"""Step 5 (MS-GF+) — top-1 search accuracy against the small GT library.

Reads the MzidToTsv PSM table from an MS-GF+ run over the GT plain TD FASTA
(base_plain + rev_ decoys), picks the best PSM per spectrum (min SpecEValue),
and reports: top-1 accuracy vs the MGF SEQ= GT and the decoy-win rate.

ordinal mapping: MzidToTsv ScanNum is 1-based; pass --scan-offset so that
ordinal = ScanNum + offset (0 for msconvert mzML with 0-based SpecId index=scan,
-1 for fixed-MGF TITLE where scan = ordinal+1).

Usage:
    python step5_msgfplus_acc.py --species mouse --work <fullsearch> \
        --tsv <out>/mouse.tsv [--scan-offset 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from fullsearch_common import dc

FLANK = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--tsv", required=True, help="MS-GF+ MzidToTsv .tsv")
    p.add_argument("--scan-offset", type=int, default=0,
                   help="ordinal = ScanNum + offset")
    p.add_argument("--out", default=None)
    return p.parse_args()


def strip(peptide: str) -> str:
    if "." in peptide:
        return peptide.split(".", 1)[1].rsplit(".", 1)[0]
    return peptide


def main() -> int:
    import re
    args = parse_args()
    work = Path(args.work)
    _fasta, mgf = dc.resolve_species_files(Path(args.root), args.species)
    mgf_gt = {}
    for ordinal, (meta, _pk) in enumerate(dc.iter_mgf_spectra(mgf)):
        g = meta.get("SEQ", "")
        if g:
            mgf_gt[ordinal] = g

    best = {}
    with open(args.tsv, encoding="utf-8", errors="ignore") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            try:
                scan = int(r.get("ScanNum", "").strip())
            except (TypeError, ValueError):
                continue
            try:
                ev = float(r.get("SpecEValue", "inf"))
            except (TypeError, ValueError):
                ev = float("inf")
            try:
                ms = float(r.get("MSGFScore", "-1e9"))
            except (TypeError, ValueError):
                ms = -1e9
            ord_ = scan + args.scan_offset
            if ord_ not in mgf_gt:
                continue
            cur = best.get(ord_)
            if cur is None or ev < cur["ev"] or (ev == cur["ev"] and ms > cur["ms"]):
                best[ord_] = {"seq": strip(r.get("Peptide", "")),
                              "prot": r.get("Protein", ""), "ev": ev, "ms": ms}

    n_with_gt = n_identified = n_hit = n_decoy = 0
    rows = []
    for ord_, gt in mgf_gt.items():
        n_with_gt += 1
        top = best.get(ord_)
        hit = decoy = 0
        if top:
            n_identified += 1
            decoy = 1 if "rev_" in top["prot"] else 0
            n_decoy += decoy
            hit = int((not decoy) and dc.peptide_matches(top["seq"], gt))
            n_hit += hit
        rows.append({"ordinal": ord_, "identified": int(top is not None),
                     "best_seq": top["seq"] if top else "", "decoy": decoy,
                     "hit": hit, "gt_seq": gt})

    denom = n_with_gt or 1
    summary = {
        "species": args.species, "predictor": "msgfplus",
        "n_with_gt": n_with_gt, "n_identified": n_identified,
        "top1_acc": round(n_hit / denom, 6),
        "decoy_win": round(n_decoy / denom, 6),
        "scan_offset": args.scan_offset,
    }
    out = args.out or (work / "output" / args.species /
                       f"summary_msgfplus_{args.species}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[msgfplus-acc] {args.species}: n_with_gt={n_with_gt} "
          f"identified={n_identified} top1_acc={summary['top1_acc']:.4f} "
          f"decoy_win={summary['decoy_win']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
