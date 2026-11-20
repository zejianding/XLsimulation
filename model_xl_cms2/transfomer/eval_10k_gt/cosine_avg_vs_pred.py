#!/usr/bin/env python3
"""Prediction-vs-average-spectrum cosine (B2 右半边), CSV only.

Companion to cosine_detail_csv.py (which averages the prediction against every
original spectrum, case B1). Here each peptide has ONE averaged experimental
spectrum, so the comparison is a single cosine: PiPrime embedding of the
predicted spectrum vs PiPrime embedding of the average spectrum.

    <run>/Embeddings/<stem>.npz       prediction + originals (from the group MGF)
    <run>/Embeddings_Avg/<stem>.npz   the averaged spectrum (one row, row 0)

Outputs:
    cosine_avg_summary.csv  group_file, sequence, charge, num_original,
                            num_prediction, pred_vs_avg_cosine, orig_vs_avg_mean
    cosine_avg_detail.csv   group_file, sequence, charge, idx_in_group, source_tag,
                            species, spectrum_id, cosine_to_average

``orig_vs_avg_mean`` is a sanity column: how well the individual originals agree
with their own average. If the prediction's cosine sits near it, the predicted
spectrum is about as close to the average as the real spectra are to each other.

Usage:
    python cosine_avg_vs_pred.py --mgf-dir <run>/Grouped_MGFs \
        --npz-dir <run>/Embeddings --avg-npz-dir <run>/Embeddings_Avg \
        --out-dir <run> [--limit 0]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from cosine_detail_csv import read_group_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mgf-dir", required=True)
    p.add_argument("--npz-dir", required=True, help="embeddings of the group MGFs")
    p.add_argument("--avg-npz-dir", required=True, help="embeddings of Average_MGFs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all groups (debug cap)")
    return p.parse_args()


def cos_to(vec: np.ndarray, ref: np.ndarray) -> float:
    nv, nr = np.linalg.norm(vec), np.linalg.norm(ref)
    if nv == 0 or nr == 0:
        return float("nan")
    return float(vec @ ref / (nv * nr))


def main() -> int:
    args = parse_args()
    mgf_dir = Path(args.mgf_dir)
    npz_dir, avg_dir = Path(args.npz_dir), Path(args.avg_npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "cosine_avg_detail.csv"
    summ_path = out_dir / "cosine_avg_summary.csv"

    n_groups = 0
    with open(detail_path, "w", newline="", encoding="utf-8") as fd, \
         open(summ_path, "w", newline="", encoding="utf-8") as fs:
        wd, ws = csv.writer(fd), csv.writer(fs)
        wd.writerow(["group_file", "sequence", "charge", "idx_in_group", "source_tag",
                     "species", "spectrum_id", "cosine_to_average"])
        ws.writerow(["group_file", "sequence", "charge", "num_original", "num_prediction",
                     "pred_vs_avg_cosine", "orig_vs_avg_mean"])

        for mgf_path in sorted(mgf_dir.glob("*.mgf")):
            npz_path = npz_dir / (mgf_path.stem + ".npz")
            avg_path = avg_dir / (mgf_path.stem + ".npz")
            if not npz_path.exists() or not avg_path.exists():
                continue
            metas = read_group_meta(mgf_path)
            try:
                emb = np.load(npz_path)["embeddings"].astype(np.float64)
                avg = np.load(avg_path)["embeddings"].astype(np.float64)
            except Exception as exc:
                print(f"[skip] {mgf_path.name}: {exc}")
                continue
            if emb.shape[0] != len(metas) or emb.shape[0] < 2 or avg.shape[0] < 1:
                print(f"[skip] {mgf_path.name}: emb {emb.shape[0]} vs spectra {len(metas)}, "
                      f"avg {avg.shape[0]}")
                continue

            ref = avg[0]
            src = [str(m.get("SOURCE", "UNKNOWN")).upper() for m in metas]
            pred_idx = [i for i, s in enumerate(src) if "PREDICTION" in s]
            orig_idx = [i for i, s in enumerate(src) if "PREDICTION" not in s]
            if not pred_idx or not orig_idx:
                continue

            cos = [cos_to(emb[i], ref) for i in range(emb.shape[0])]
            seq = metas[0].get("SEQ", "")
            charge = re.sub(r"\D", "", metas[0].get("CHARGE", "")) or ""
            for i, m in enumerate(metas):
                wd.writerow([mgf_path.name, seq, charge, i, src[i],
                             m.get("COMMENT", "").replace("species:", ""),
                             m.get("SCANS", ""),
                             f"{cos[i]:.6f}" if cos[i] == cos[i] else "nan"])

            pv = cos[pred_idx[-1]]
            ov = [cos[i] for i in orig_idx]
            ov_mean = sum(ov) / len(ov)
            ws.writerow([mgf_path.name, seq, charge, len(orig_idx), len(pred_idx),
                         f"{pv:.6f}" if pv == pv else "nan", f"{ov_mean:.6f}"])
            n_groups += 1
            if n_groups % 500 == 0:
                print(f"  {n_groups} groups ...", flush=True)
            if args.limit and n_groups >= args.limit:
                break

    print(f"done: {n_groups} groups -> {summ_path} (+ {detail_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
