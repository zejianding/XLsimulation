#!/usr/bin/env python3
"""Step 2 — match MGF spectra to target/decoy candidates by precursor mass (± ppm).

Builds an in-memory mass index over the step1 target+decoy FASTA, then for each
MGF spectrum computes the neutral precursor mass and retrieves every candidate
within ``--ppm`` (default 10).  Every candidate carries its label (T/D) so the
FDR experiment knows which top-1s are guaranteed-wrong decoys.

Writes:
    <out-candidates>  candidates_<species>_td.tsv.gz
          columns: ordinal, scan_id, charge, pepmass, rt, seq, label,
                   neutral_mass, ppm
    <out-dir>/records/record_step2_<species>.json   process record

Usage:
    python step2_match_td.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --fasta <fdr>/data/mouse/mouse_td.fasta \
        --out-candidates <fdr>/data/mouse/candidates_mouse_td.tsv.gz \
        --out-dir <fdr>/output/mouse [--ppm 10] [--ppm-offset 0] [--max-scans N]
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np

import fdr_common as fc
import ds_common as dc  # noqa: F401  (path bootstrap + helpers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=fc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--fasta", required=True, help="target+decoy FASTA from step1")
    p.add_argument("--out-candidates", required=True, help="output TSV.gz path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ppm", type=float, default=10.0, help="mass window ppm")
    p.add_argument("--ppm-offset", type=float, default=0.0,
                   help="systematic ppm re-centering (additive, before window)")
    p.add_argument("--max-scans", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_candidates = Path(args.out_candidates)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    _fasta_in, mgf_path = dc.resolve_species_files(args.root, args.species)

    print(f"[step2] loading target+decoy index from {args.fasta} ...")
    mass_arr, seq_list, label_arr = fc.load_td_fasta(Path(args.fasta))
    order = np.argsort(mass_arr)
    sorted_mass = mass_arr[order]
    sorted_row = np.arange(mass_arr.shape[0], dtype=np.int64)[order]
    n_target = int((label_arr == fc.LABEL_TARGET).sum())
    n_decoy = int((label_arr == fc.LABEL_DECOY).sum())
    print(f"[step2] index ready: {mass_arr.shape[0]} entries "
          f"({n_target} target / {n_decoy} decoy)")

    ppm = float(args.ppm)
    offset = float(args.ppm_offset)
    n_spectra = 0
    n_with_candidates = 0
    n_with_both = 0
    n_with_gt = 0
    n_gt_in_targets = 0
    n_cand_list: list[int] = []
    n_target_list: list[int] = []
    n_decoy_list: list[int] = []
    gt_ppm_list: list[float] = []

    total_spectra = args.max_scans or dc.count_mgf_spectra(mgf_path)
    with gzip.open(out_candidates, "wt", encoding="utf-8", compresslevel=6) as fout:
        fout.write("ordinal\tscan_id\tcharge\tpepmass\trt\tseq\tlabel\tneutral_mass\tppm\n")
        for ordinal, (meta, _peaks) in enumerate(
            dc.progress(dc.iter_mgf_spectra(mgf_path, max_scans=args.max_scans),
                        total=total_spectra, desc="step2 match", unit="spec",
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
            n_t = n_d = 0
            for row in rows:
                seq = seq_list[row]
                lab = label_arr[row]
                if lab == fc.LABEL_TARGET:
                    n_t += 1
                else:
                    n_d += 1
                fout.write(
                    f"{ordinal}\t{scan_id}\t{charge}\t{pepmass:.6f}\t{rt:.3f}"
                    f"\t{seq}\t{lab}\t{mass_arr[row]:.6f}\t"
                    f"{(mass_arr[row] - neutral) / neutral * 1e6:.4f}\n"
                )
            n_cand_list.append(len(rows))
            n_target_list.append(n_t)
            n_decoy_list.append(n_d)
            if len(rows) > 0:
                n_with_candidates += 1
                if n_t > 0 and n_d > 0:
                    n_with_both += 1

            gt_seq = meta.get("SEQ", "")
            if gt_seq:
                n_with_gt += 1
                for row in rows:
                    if label_arr[row] == fc.LABEL_TARGET and dc.peptide_matches(seq_list[row], gt_seq):
                        n_gt_in_targets += 1
                        gt_ppm_list.append((mass_arr[row] - neutral) / neutral * 1e6)
                        break

    arr = np.asarray(n_cand_list, dtype=np.float64) if n_cand_list else np.zeros(1)
    gt_in_rate = n_gt_in_targets / n_with_gt if n_with_gt else 0.0
    gt_ppm_summary = {}
    if gt_ppm_list:
        g = np.asarray(gt_ppm_list)
        gt_ppm_summary = {"n": int(g.size), "median": round(float(np.median(g)), 4),
                          "mean": round(float(np.mean(g)), 4), "std": round(float(np.std(g)), 4)}

    record = {
        "step": 2,
        "species": args.species,
        "n_db_entries": int(mass_arr.shape[0]),
        "n_target": n_target,
        "n_decoy": n_decoy,
        "n_spectra": n_spectra,
        "n_spectra_with_candidates": n_with_candidates,
        "n_spectra_with_target_and_decoy": n_with_both,
        "ppm": ppm,
        "ppm_offset": offset,
        "mean_candidates": round(float(arr.mean()), 2),
        "median_candidates": round(float(np.median(arr)), 1),
        "mean_target_candidates": round(float(np.mean(n_target_list)), 2) if n_target_list else 0.0,
        "mean_decoy_candidates": round(float(np.mean(n_decoy_list)), 2) if n_decoy_list else 0.0,
        "n_spectra_with_gt": n_with_gt,
        "gt_in_target_candidates_rate": round(gt_in_rate, 6),
        "gt_ppm": gt_ppm_summary,
        "out_candidates": str(out_candidates),
        "out_size_bytes": out_candidates.stat().st_size if out_candidates.exists() else 0,
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_step2_{args.species}.json", record)
    print(f"[step2] {args.species}: {n_spectra} spectra, mean_candidates={float(arr.mean()):.1f}, "
          f"mean_t={float(np.mean(n_target_list)):.1f}/mean_d={float(np.mean(n_decoy_list)):.1f}, "
          f"both={n_with_both}/{n_spectra}, gt_in_targets={gt_in_rate:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
