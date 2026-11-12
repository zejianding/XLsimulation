#!/usr/bin/env python3
"""Step 3 — match MGF spectra to candidate variants by precursor mass (± ppm).

Builds an in-memory mass index over the variants FASTA produced by step2, then
for each MGF spectrum computes the neutral precursor mass from PEPMASS/CHARGE
and retrieves every candidate variant within ``--ppm`` (default 10) of it
(optionally re-centered by ``--ppm-offset``).  Per-spectrum GT diagnostics are
computed (is the GT annotated sequence among the candidates? its ppm) to catch
systematic precursor bias.

Writes:
    <out-candidates>  candidates_<sp>_scans.tsv.gz
          columns: ordinal, scan_id, charge, pepmass, rt, cand_row, seq,
                   neutral_mass, ppm
    <out-dir>/records/record_step3_<sp>.json   process record

Usage:
    python step3_match_by_mass.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --fasta <ds>/data/mouse/variants_mouse.fasta \
        --out-candidates <ds>/data/mouse/candidates_mouse_scans.tsv.gz \
        --out-dir <ds>/output/mouse [--ppm 10] [--ppm-offset 0] [--max-scans N]
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

import ds_common as dc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--fasta", required=True, help="variants FASTA from step2")
    p.add_argument("--out-candidates", required=True, help="output TSV.gz path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ppm", type=float, default=10.0, help="mass window ppm")
    p.add_argument("--ppm-offset", type=float, default=0.0,
                   help="systematic ppm re-centering (additive, before window)")
    p.add_argument("--max-scans", type=int, default=None,
                   help="limit spectra matched (fast iteration)")
    return p.parse_args()


def load_variant_index(fasta_path: Path):
    """Return (mass_arr float64, seq_list) for all variant records."""
    masses: list[float] = []
    seqs: list[str] = []
    with open(fasta_path, encoding="utf-8") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                header = line[1:]
                continue
            if header is None:
                continue
            # header: HEADER MASS=<x> MOD=<annoseq>
            parts = header.split()
            mass = float(parts[1][5:])
            seq = line.strip()
            masses.append(mass)
            seqs.append(seq)
            header = None
    mass_arr = np.asarray(masses, dtype=np.float64)
    return mass_arr, seqs


def main() -> int:
    args = parse_args()
    out_candidates = Path(args.out_candidates)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    _fasta_in, mgf_path = dc.resolve_species_files(args.root, args.species)

    print(f"[step3] loading variant index from {args.fasta} ...")
    mass_arr, seq_list = load_variant_index(Path(args.fasta))
    order = np.argsort(mass_arr)
    sorted_mass = mass_arr[order]
    sorted_row = np.arange(mass_arr.shape[0], dtype=np.int64)[order]
    print(f"[step3] index ready: {mass_arr.shape[0]} variants")

    ppm = float(args.ppm)
    offset = float(args.ppm_offset)
    n_spectra = 0
    n_with_gt = 0
    n_gt_in_candidates = 0
    gt_ppm_list: list[float] = []
    n_candidates_list: list[int] = []

    total_spectra = args.max_scans or dc.count_mgf_spectra(mgf_path)
    with gzip.open(out_candidates, "wt", encoding="utf-8", compresslevel=6) as fout:
        fout.write("ordinal\tscan_id\tcharge\tpepmass\trt\tcand_row\tseq\tneutral_mass\tppm\n")
        for ordinal, (meta, _peaks) in enumerate(
            dc.progress(dc.iter_mgf_spectra(mgf_path, max_scans=args.max_scans),
                        total=total_spectra, desc="step3 match", unit="spec",
                        mininterval=1.0)
        ):
            n_spectra += 1
            charge = dc.parse_charge_from_meta(meta)
            pepmass = dc.parse_pepmass_from_meta(meta)
            rt = float(meta.get("RTINSECONDS", dc.DEFAULT_RT))
            if charge <= 0 or pepmass <= 0:
                continue
            neutral = dc.neutral_from_mz(pepmass, charge)
            center = neutral * (1.0 + offset * 1e-6)
            tol_da = center * ppm * 1e-6
            lo = int(np.searchsorted(sorted_mass, center - tol_da, side="left"))
            hi = int(np.searchsorted(sorted_mass, center + tol_da, side="right"))

            rows = sorted_row[lo:hi]
            scan_id = meta.get("SCANS", str(ordinal))
            for row in rows:
                seq = seq_list[row]
                cand_mass = mass_arr[row]
                ppm_i = (cand_mass - neutral) / neutral * 1e6
                fout.write(
                    f"{ordinal}\t{scan_id}\t{charge}\t{pepmass:.6f}\t{rt:.3f}"
                    f"\t{row}\t{seq}\t{cand_mass:.6f}\t{ppm_i:.4f}\n"
                )
            n_candidates_list.append(len(rows))

            # GT diagnostic uses the V20-rescoring mass-matching rule.
            gt_seq = meta.get("SEQ", "")
            if gt_seq:
                n_with_gt += 1
                for row in rows:
                    if dc.peptide_matches(seq_list[row], gt_seq):
                        n_gt_in_candidates += 1
                        gt_ppm_list.append(
                            (mass_arr[row] - neutral) / neutral * 1e6
                        )
                        break

    gt_in_rate = n_gt_in_candidates / n_with_gt if n_with_gt else 0.0
    mean_candidates = float(np.mean(n_candidates_list)) if n_candidates_list else 0.0
    gt_ppm_summary = {}
    if gt_ppm_list:
        arr = np.asarray(gt_ppm_list)
        gt_ppm_summary = {
            "n": int(arr.size),
            "median": round(float(np.median(arr)), 4),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
        }

    record = {
        "step": 3,
        "species": args.species,
        "n_variants_in_index": int(mass_arr.shape[0]),
        "n_spectra": n_spectra,
        "n_spectra_with_gt": n_with_gt,
        "ppm": ppm,
        "ppm_offset": offset,
        "mean_candidates_per_spectrum": round(mean_candidates, 2),
        "gt_in_candidates_rate": round(gt_in_rate, 6),
        "gt_ppm": gt_ppm_summary,
        "out_candidates": str(out_candidates),
        "out_size_bytes": out_candidates.stat().st_size if out_candidates.exists() else 0,
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_step3_{args.species}.json", record)
    print(f"[step3] {args.species}: {n_spectra} spectra, "
          f"mean_candidates={mean_candidates:.1f}, "
          f"gt_in_candidates={gt_in_rate:.4f}, gt_ppm={gt_ppm_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
