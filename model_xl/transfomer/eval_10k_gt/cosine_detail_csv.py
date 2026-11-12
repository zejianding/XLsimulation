#!/usr/bin/env python3
"""Streaming per-spectrum cosine details + per-peptide summary, CSV only.

Companion to batch_cosine_summary.py: that script writes a per-group CSV summary;
this one additionally keeps the FULL detail — one row per original spectrum, with
its cosine to the predicted spectrum of the same peptide — and writes everything
as CSV (no JSON), one group at a time so memory stays bounded:

    cosine_detail.csv    group_file, sequence, charge, idx_in_group, source_tag,
                         species, spectrum_id, cosine_to_prediction
    cosine_summary.csv   group_file, sequence, charge, num_original, num_prediction,
                         pred_vs_orig_mean, pred_vs_orig_std, pred_vs_orig_min,
                         pred_vs_orig_max, orig_orig_mean

Inputs are the standard eval artifacts:
  MGF dir : <run>/Grouped_MGFs/<seq>_charge<z>_<hash>.mgf  (SOURCE=ORIGINAL /
            SOURCE=MODEL_PREDICTION tags, COMMENT=species:<sp>)
  NPZ dir : <run>/Embeddings/<same stem>.npz  from mgf_to_embedding_batch.py
            (embeddings in the SAME spectrum order as the MGF)

Usage:
    python cosine_detail_csv.py --mgf-dir <run>/Grouped_MGFs --npz-dir <run>/Embeddings \
        --out-dir <run> [--limit 0]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mgf-dir", required=True)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all groups (debug cap)")
    return p.parse_args()


def read_group_meta(mgf_path: Path):
    """Yield per-spectrum dicts in file order: source, species, spectrum_id, seq, charge."""
    meta = {}
    out = []
    with open(mgf_path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if s == "BEGIN IONS":
                meta = {}
                continue
            if s == "END IONS":
                out.append(meta)
                continue
            if "=" in s and not s[0].isdigit():
                k, _, v = s.partition("=")
                meta[k.strip().upper()] = v.strip()
    return out


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1e-12
    e = emb / n
    return e @ e.T


def main() -> int:
    args = parse_args()
    mgf_dir, npz_dir, out_dir = Path(args.mgf_dir), Path(args.npz_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "cosine_detail.csv"
    summ_path = out_dir / "cosine_summary.csv"

    n_groups = n_spectra = 0
    with open(detail_path, "w", newline="", encoding="utf-8") as fd, \
         open(summ_path, "w", newline="", encoding="utf-8") as fs:
        wd = csv.writer(fd)
        ws = csv.writer(fs)
        wd.writerow(["group_file", "sequence", "charge", "idx_in_group", "source_tag",
                     "species", "spectrum_id", "cosine_to_prediction"])
        ws.writerow(["group_file", "sequence", "charge", "num_original", "num_prediction",
                     "pred_vs_orig_mean", "pred_vs_orig_std", "pred_vs_orig_min",
                     "pred_vs_orig_max", "orig_orig_mean"])

        for mgf_path in sorted(mgf_dir.glob("*.mgf")):
            npz_path = npz_dir / (mgf_path.stem + ".npz")
            if not npz_path.exists():
                continue
            metas = read_group_meta(mgf_path)
            try:
                emb = np.load(npz_path)["embeddings"].astype(np.float64)
            except Exception as exc:
                print(f"[skip] {mgf_path.name}: {exc}")
                continue
            if emb.shape[0] != len(metas) or len(metas) < 2:
                print(f"[skip] {mgf_path.name}: emb {emb.shape[0]} vs spectra {len(metas)}")
                continue
            sim = cosine_matrix(emb)
            src = [str(m.get("SOURCE", "UNKNOWN")).upper() for m in metas]
            pred_idx = [i for i, s in enumerate(src) if "PREDICTION" in s]
            orig_idx = [i for i, s in enumerate(src) if "PREDICTION" not in s]
            if not pred_idx or not orig_idx:
                continue
            pi = pred_idx[-1]
            pv = sim[pi, orig_idx]
            for j, oi in enumerate(orig_idx):
                m = metas[oi]
                wd.writerow([mgf_path.name, metas[0].get("SEQ", ""),
                             re.sub(r"\D", "", metas[0].get("CHARGE", "")) or "",
                             oi, src[oi], m.get("COMMENT", "").replace("species:", ""),
                             m.get("SCANS", ""), f"{pv[j]:.6f}"])
            oo = sim[np.ix_(orig_idx, orig_idx)]
            oo_mean = float((oo.sum() - np.trace(oo)) / (len(orig_idx) * (len(orig_idx) - 1))) \
                if len(orig_idx) > 1 else float("nan")
            ws.writerow([mgf_path.name, metas[0].get("SEQ", ""),
                         re.sub(r"\D", "", metas[0].get("CHARGE", "")) or "",
                         len(orig_idx), len(pred_idx),
                         f"{float(pv.mean()):.6f}",
                         f"{float(pv.std()):.6f}" if len(pv) > 1 else "nan",
                         f"{float(pv.min()):.6f}", f"{float(pv.max()):.6f}",
                         f"{oo_mean:.6f}" if oo_mean == oo_mean else "nan"])
            n_groups += 1
            n_spectra += len(orig_idx)
            if n_groups % 500 == 0:
                print(f"  {n_groups} groups, {n_spectra} original spectra ...", flush=True)
            if args.limit and n_groups >= args.limit:
                break
    print(f"done: {n_groups} groups / {n_spectra} original-spectrum rows -> {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
