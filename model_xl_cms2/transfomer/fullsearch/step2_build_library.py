#!/usr/bin/env python3
"""Step 2 — expand GT backbones into the annotated target+decoy variant library.

Targets = fixed C+57.021 + variable M+15.995 (<= max-var-mods) annotated variants
of every unique GT backbone.  Decoys = reverse_annotated(target variant); a decoy
that collides with any target or an already-accepted decoy is dropped (mirror of
fdr/step1).  Writes the annotated library TSV.gz consumed by the spectral
predictors.  base_plain.fasta (step1) remains the MS-GF+ seed FASTA.

Writes: data/<sp>/library_annotated.tsv.gz + record json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from fullsearch_common import (dc, fc, enumerate_variants, neutral_of, backbone_of,
                               FIXED_MODS, VARIABLE_MODS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--work", required=True, help="fullsearch data/output root")
    p.add_argument("--max-var-mods", type=int, default=3)
    p.add_argument("--plain-fasta", default=None,
                   help="seed base FASTA (default <work>/data/<sp>/base_plain.fasta)")
    return p.parse_args()


def read_plain(fasta: Path):
    out = []
    with open(fasta, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith(">"):
                continue
            out.append(raw)
    return out


def main() -> int:
    args = parse_args()
    work = Path(args.work)
    ddir = work / "data" / args.species
    odir = work / "output" / args.species / "records"
    fasta = Path(args.plain_fasta) if args.plain_fasta else ddir / "base_plain.fasta"
    base = read_plain(fasta)

    # targets = unique annotated variants of each backbone
    targets: list[str] = []
    seen: set[str] = set()
    n_base_with_m = 0
    for bb in base:
        if "M" in bb:
            n_base_with_m += 1
        for a in enumerate_variants(bb, FIXED_MODS, VARIABLE_MODS, args.max_var_mods):
            if a not in seen:
                seen.add(a)
                targets.append(a)

    target_set = seen
    decoys: list[str] = []
    decoy_set: set[str] = set()
    n_collision = 0
    for a in targets:
        r = fc.reverse_annotated(a)
        if r is None or r in target_set or r in decoy_set:
            n_collision += 1
            continue
        decoys.append(r)
        decoy_set.add(r)

    entries = [(a, fc.LABEL_TARGET) for a in targets] + \
              [(a, fc.LABEL_DECOY) for a in decoys]
    with gzip.open(ddir / "library_annotated.tsv.gz", "wt",
                   encoding="utf-8", compresslevel=6) as fh:
        fh.write("idx\tannoseq\tbackbone\tlabel\tmass\n")
        for i, (a, lab) in enumerate(entries):
            bb = backbone_of(a)
            fh.write(f"{i}\t{a}\t{bb}\t{lab}\t{neutral_of(a):.6f}\n")

    rec = {
        "species": args.species,
        "n_base_peptide": len(base),
        "n_base_with_M": n_base_with_m,
        "n_target_variants": len(targets),
        "n_decoy": len(decoys),
        "n_collision_dropped": n_collision,
        "n_entries": len(entries),
        "library": str(ddir / "library_annotated.tsv.gz"),
    }
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"record_step2_lib_{args.species}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gtlib] {args.species}: base={len(base)} (M-containing={n_base_with_m}) "
          f"targets={len(targets)} decoys={len(decoys)} "
          f"collision_dropped={n_collision} entries={len(entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
