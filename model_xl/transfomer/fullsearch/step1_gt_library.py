#!/usr/bin/env python3
"""Step 1 — collect the GT peptide library from a species' MGF.

Reads every spectrum's SEQ= ground truth, deduplicates the backbones and writes:
  data/<sp>/base_plain.fasta        unique plain backbones (for the MS-GF+ leg
                                    and as the seed for variant expansion)
  data/<sp>/scans_meta.json         charge multiset + GT/backbone stats
  output/<sp>/records/record_step1_gt_<sp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from fullsearch_common import (dc, backbone_of, MAX_BACKBONE_PRO)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--work", required=True, help="fullsearch data/output root")
    p.add_argument("--max-scans", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.work)
    ddir = work / "data" / args.species
    odir = work / "output" / args.species / "records"
    ddir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)

    _fasta, mgf = dc.resolve_species_files(Path(args.root), args.species)
    total = args.max_scans or dc.count_mgf_spectra(mgf)

    backbones: dict[str, int] = {}
    charge_counter: Counter = Counter()
    n_spectra = n_with_gt = n_gt_too_long = 0
    for _ordinal, (meta, _pk) in enumerate(
        dc.progress(dc.iter_mgf_spectra(mgf, max_scans=args.max_scans),
                    total=total, desc="step1 collect GT", unit="spec", mininterval=1.0)
    ):
        n_spectra += 1
        ch = dc.parse_charge_from_meta(meta)
        if ch > 0:
            charge_counter[ch] += 1
        gt = meta.get("SEQ", "")
        if not gt:
            continue
        n_with_gt += 1
        bb = backbone_of(gt)
        if len(bb) > MAX_BACKBONE_PRO:
            n_gt_too_long += 1
        backbones[bb] = backbones.get(bb, 0) + 1

    base_path = ddir / "base_plain.fasta"
    with open(base_path, "w", encoding="utf-8") as fh:
        for i, bb in enumerate(sorted(backbones), start=1):
            fh.write(f">gt_{i} len={len(bb)}\n{bb}\n")

    meta = {
        "species": args.species,
        "n_spectra": n_spectra,
        "n_with_gt": n_with_gt,
        "n_unique_backbone": len(backbones),
        "n_gt_gt30": n_gt_too_long,
        "charges": sorted(charge_counter),
        "charge_counts": dict(charge_counter),
        "backbone_len_hist": {str(k): v for k, v in sorted(
            Counter(len(b) for b in backbones).items())},
        "base_fasta": str(base_path),
    }
    (ddir / "scans_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (odir / f"record_step1_gt_{args.species}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gtlib] {args.species}: spectra={n_spectra} with_gt={n_with_gt} "
          f"unique_backbones={len(backbones)} charges={meta['charges']} "
          f"gt>30={n_gt_too_long} -> {base_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
