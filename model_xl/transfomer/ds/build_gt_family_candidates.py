#!/usr/bin/env python3
"""Per-spectrum candidate pool = the MGF ground truth + its mod variants.

Purpose: an isolated Prosit "GT fidelity / mod-family discrimination" analysis.
For every MGF spectrum we build a SMALL candidate pool containing only that
spectrum's true peptide (SEQ=) and its modification variants (fixed C+57.021 on
every C + variable M+15.995 on up to N M-sites, identical rules to the DS
step2/step3 candidate enumeration), mass-filtered to +-PPM of the precursor.
Feeding this pool to step4_prosit_score.py then yields, with a fraction of the
Koina calls (predictions are per backbone x charge and GTs are a tiny subset of
the full candidate space):

  * GT fidelity      : <branch>_gt_score columns = how well Prosit's prediction
                       of the true peptide reproduces the experimental spectrum
                       (b/y filtered, rel-level intensity weighted).
  * mod-family top1  : <branch>_hit / _gt_rank = whether the correct variant of
                       the true peptide family ranks first.

Output columns follow the step3 TSV convention (seq at index 6, neutral mass 7,
ppm 8) so step4_prosit_score.py can read it unchanged.

Usage:
    python build_gt_family_candidates.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --out-candidates data/mouse/candidates_mouse_gtfam.tsv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import ds_common as dc
from predict_mgf import AA_MASS
from step2_build_candidate_fasta import enumerate_variants, neutral_mass_with_table

PROSIT_MAX_LEN = 30
FIXED_MODS = {"C": ["57.021"]}
VARIABLE_MODS = {"M": ["15.995"]}
MASS_TABLE = dict(AA_MASS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--out-candidates", required=True, help="output TSV.gz (step3 format)")
    p.add_argument("--ppm", type=float, default=10.0, help="mass window ppm (default 10)")
    p.add_argument("--max-var-mods", type=int, default=3,
                   help="max variable M-ox sites per peptide (default 3)")
    p.add_argument("--max-scans", type=int, default=None)
    return p.parse_args()


def backbone_of(anno: str) -> str:
    residues, _deltas, _nterm = dc.parse_modified_sequence(anno)
    return "".join(residues)


def main() -> int:
    args = parse_args()
    out = Path(args.out_candidates)
    out.parent.mkdir(parents=True, exist_ok=True)
    _fasta, mgf_path = dc.resolve_species_files(Path(args.root), args.species)
    total = args.max_scans or dc.count_mgf_spectra(mgf_path)

    n_spec = n_with_gt = n_gt_too_long = n_family = 0
    fam_sizes: list[int] = []
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fh:
        fh.write("ordinal\tscan_id\tcharge\tpepmass\trt\tn_fam\tseq\tneutral_mass\tppm\n")
        for ordinal, (meta, _peaks) in enumerate(
            dc.progress(dc.iter_mgf_spectra(mgf_path, max_scans=args.max_scans),
                        total=total, desc="gt family", unit="spec", mininterval=1.0)
        ):
            n_spec += 1
            charge = dc.parse_charge_from_meta(meta)
            pepmass = dc.parse_pepmass_from_meta(meta)
            rt = float(meta.get("RTINSECONDS", dc.DEFAULT_RT))
            gt = meta.get("SEQ", "")
            if not gt or charge <= 0 or pepmass <= 0:
                continue
            n_with_gt += 1
            bb = backbone_of(gt)
            if len(bb) > PROSIT_MAX_LEN:
                n_gt_too_long += 1
                continue  # unresolvable by Prosit (same rule as step4 collect)
            variants = list(dict.fromkeys(
                enumerate_variants(bb, FIXED_MODS, VARIABLE_MODS, args.max_var_mods)))
            # guarantee the exact annotated GT is a candidate even if its
            # formatting differs from our enumeration.
            if gt not in variants:
                variants.insert(0, gt)
            kept = []
            for a in variants:
                m = neutral_mass_with_table(a, MASS_TABLE)
                ppm = (m - pepmass) / pepmass * 1e6 if pepmass else float("inf")
                if abs(ppm) <= args.ppm:
                    kept.append((a, m, ppm))
            if not kept:
                continue
            scan_id = meta.get("SCANS", str(ordinal))
            for a, m, ppm in kept:
                fh.write(f"{ordinal}\t{scan_id}\t{charge}\t{pepmass:.6f}\t{rt:.3f}\t"
                         f"{len(kept)}\t{a}\t{m:.6f}\t{ppm:.4f}\n")
            n_family += 1
            fam_sizes.append(len(kept))

    mean_fam = float(sum(fam_sizes)) / len(fam_sizes) if fam_sizes else 0.0
    print(f"[gtfam] {args.species}: spectra={n_spec} with_gt={n_with_gt} "
          f"gt>30_skipped={n_gt_too_long} family_scored={n_family} "
          f"mean_family={mean_fam:.2f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
