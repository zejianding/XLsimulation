#!/usr/bin/env python3
"""Visualize candidate spectra + d/f/m_by_prec scores for one scan.

Loads the V20 checkpoint, predicts the spectrum (peaks) for the top-1 and the
true candidate of a chosen scan, and plots:
  - panel 1: experimental spectrum (black) + TRUE candidate predicted peaks (green)
  - panel 2: experimental spectrum (black) + TOP-1 candidate predicted peaks (red)
  - panel 3: grouped bars of d / f / m_by_prec for TRUE vs TOP-1

Usage:
    python viz_candidates.py --ordinal 0 \
        --results output/mouse/full/results_step4_mouse.csv \
        --candidates data/mouse/candidates_mouse_full.tsv.gz \
        --mgf /home/wangyuhang/seq2mass/v1-search/mouse/high.mouse.PXD004948_len7_32_mod_within_scope_10000.mgf \
        --checkpoint /home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt \
        --branch hdf020_020 --rel 0.20 --out viz_scan0.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import ds_common as dc
import step4_predict_score as s4
from predict_mgf import generate_by_ions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ordinal", type=int, required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--mgf", required=True)
    p.add_argument("--checkpoint", default=dc.SERVER_CKPT)
    p.add_argument("--branch", default="hdf020_020", help="csv prefix (e.g. hdf020_020)")
    p.add_argument("--rel", type=float, default=0.20, help="rel filter level")
    p.add_argument("--out", default="viz_scan.png")
    p.add_argument("--min-mz", type=float, default=100.0)
    p.add_argument("--topk", type=int, default=2, help="show top-N ranked candidates")
    p.add_argument("--exp-transform", default="sqrt_clip",
                   choices=["none", "sqrt_clip", "sqrt"],
                   help="map experimental intensities into the model's training domain "
                        "before plotting/scoring (same as step4): sqrt_clip=sqrt(min(x,50000))/"
                        "sqrt(50000), none=raw linear, sqrt=sqrt(x)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model + candidates + results row
    model, model_config, stats, _ = dc.load_checkpoint(Path(args.checkpoint), device)
    rel_intensities = [args.rel]
    cands = s4.load_candidates_by_ordinal(Path(args.candidates)).get(args.ordinal, [])
    seqs = [c[0] for c in cands]
    rows = list(csv.DictReader(open(args.results, encoding="utf-8")))
    row = next((r for r in rows if int(r["ordinal"]) == args.ordinal), None)
    if row is None:
        print(f"ordinal {args.ordinal} not in results")
        return 1
    gt_seq = row.get("gt_seq", "")
    pre = args.branch

    # get meta + experimental peaks (transform into model's training domain so the
    # plotted relative shapes AND the d/f/m_by_prec scores match a sqrt_clip step4 run;
    # keep the raw copy for a faint overlay showing what the transform changed)
    meta = peaks = raw_peaks = None
    for ordinal, (m, pk) in enumerate(dc.iter_mgf_spectra(args.mgf)):
        if ordinal == args.ordinal:
            meta, raw_peaks = m, pk
            peaks = dc.transform_exp_intensity(pk, method=args.exp_transform)
            break
    if meta is None:
        print(f"ordinal {args.ordinal} not in MGF")
        return 1
    charge = dc.parse_charge_from_meta(meta)
    pepmass = dc.parse_pepmass_from_meta(meta)
    rt = float(meta.get("RTINSECONDS", dc.DEFAULT_RT))

    # predict grids for all candidates, get scores + peaks
    pred, ig, vm = s4.predict_candidate_grids(
        model, model_config, stats, device, seqs, charge, rt, args.min_mz, 64)
    scores_all = {}
    for i, seq in enumerate(seqs):
        sc = s4.score_candidate(
            pred[i], ig[i], vm[i], peaks,
            0.01, args.min_mz, 0.05, 0.05, rel_intensities, minimal=True)
        scores_all[i] = sc
    # score arrays for ranking
    keys = [f"d_spec_rel{int(args.rel*100):03d}", f"f_frag_rel{int(args.rel*100):03d}",
            f"m_by_prec_rel{int(args.rel*100):03d}"]
    sc_arr = {k: np.array([scores_all[i][k] for i in range(len(seqs))]) for k in keys}
    fused = (
        np.clip((sc_arr[keys[0]]-sc_arr[keys[0]].min())/(sc_arr[keys[0]].max()-sc_arr[keys[0]].min()+1e-12),0,1)
        + np.clip((sc_arr[keys[1]]-sc_arr[keys[1]].min())/(sc_arr[keys[1]].max()-sc_arr[keys[1]].min()+1e-12),0,1)
        + np.clip((sc_arr[keys[2]]-sc_arr[keys[2]].min())/(sc_arr[keys[2]].max()-sc_arr[keys[2]].min()+1e-12),0,1)
    ) / 3.0
    order = np.argsort(-fused)

    # choose candidates to show: topk + true (if not already)
    show_idx = list(order[:args.topk])
    true_idx = next((i for i in range(len(seqs)) if dc.peptide_matches(seqs[i], gt_seq)), None)
    if true_idx is not None and true_idx not in show_idx:
        show_idx.append(true_idx)

    labels = {}
    for i in show_idx:
        is_true = (true_idx == i)
        labels[i] = ("TRUE" if is_true else "rank%d" % (int(np.where(order == i)[0][0]) + 1))
        labels[i] += " " + seqs[i][:20]

    def peaks_for(i):
        eff = max(0.01, args.rel * float(pred[i].max()))
        return dc.prediction_to_mgf_peaks(ig[i], pred[i], vm[i], min_intensity=eff,
                                          min_mz=args.min_mz, mz_merge_tol=0.05)

    # ---- mirror plot, one figure per shown candidate ----
    # Same m/z axis: predicted peaks UP (top half), experimental peaks DOWN
    # (bottom half, mirrored).  Main b/y ions annotated by (B/Y, charge) color.
    ION_COLORS = {("B", 1): "#d62728", ("B", 2): "#ff7f0e", ("B", 3): "#bcbd22",
                  ("Y", 1): "#1f77b4", ("Y", 2): "#17becf", ("Y", 3): "#9467bd"}

    def by_ions_for(seq, i):
        """Main b/y ions as (m/z, model-predicted intensity) colored by (type, charge)."""
        out: dict[tuple[str, int], list[tuple[float, float]]] = {}
        for ion in generate_by_ions(seq, max_charge=3):
            pos = int(ion["cleavage_index"]) - 1
            z = int(ion["charge"])
            ch = (z - 1) if ion["ion_type"] == "B" else (z + 2)
            mz = float(ig[i][pos, s4.MAIN_TOKEN_ID, ch])
            inten = float(pred[i][pos, s4.MAIN_TOKEN_ID, ch])
            key = (ion["ion_type"], z)
            out.setdefault(key, []).append((mz, inten))
        return out

    exp_mz = peaks[:, 0]
    exp_int = peaks[:, 1]
    exp_max = exp_int.max() if exp_int.size else 1.0
    show_raw = (args.exp_transform != "none") and raw_peaks is not None and raw_peaks.size > 0
    if show_raw:
        raw_mz = raw_peaks[:, 0]
        raw_int = raw_peaks[:, 1]
        raw_max = raw_int.max() if raw_int.size else 1.0

    for k, i in enumerate(show_idx):
        pk = peaks_for(i)
        pz = pk[:, 0]
        pv = pk[:, 1]
        pred_max = pv.max() if pv.size else 1.0
        seq = seqs[i]
        byions = by_ions_for(seq, i)
        is_true = (i == true_idx)
        tag = "TRUE" if is_true else ("rank%d" % (int(np.where(order == i)[0][0]) + 1))

        fig, ax = plt.subplots(figsize=(15, 8))
        # predicted UP (normalized to its own max)
        ax.vlines(pz, 0, pv / pred_max, color="#888888", lw=1.0, alpha=0.75, label="predicted")
        # experimental DOWN (normalized, mirrored): transformed (dark blue) over a
        # faint raw (gray) so the sqrt_clip compression is directly visible
        if show_raw:
            ax.vlines(raw_mz, 0, -raw_int / raw_max, color="#cccccc", lw=0.5, alpha=0.6,
                      label="experimental (raw)")
        ax.vlines(exp_mz, 0, -exp_int / exp_max, color="#1a1a5e", lw=0.9, alpha=0.95,
                  label=f"experimental ({args.exp_transform})")
        # main b/y ladder: ONLY ions whose predicted intensity passes the
        # threshold (same eff as the predicted peaks).  Bar height = predicted
        # intensity; absent = model did not predict that b/y ion strongly.
        eff = max(0.01, args.rel * pred_max)
        for key, items in byions.items():
            color = ION_COLORS.get(key, "#555555")
            for mz, inten in items:
                if inten >= eff:
                    ax.vlines(mz, 0, inten / pred_max, color=color, lw=2.0, alpha=0.95)
        ax.axhline(0, color="black", lw=0.9)
        ax.set_ylim(-1.15, 1.35)
        ax.set_xlim(100, exp_mz.max() + 50)
        ax.set_xlabel("m/z")
        ax.set_ylabel("normalized intensity  (predicted / experimental)")
        ax.set_title(
            f"scan {args.ordinal}  GT={gt_seq}  (charge {charge}+)   [{tag}] {seq}\n"
            f"exp={args.exp_transform}  d={scores_all[i][keys[0]]:.3f}  "
            f"f={scores_all[i][keys[1]]:.3f}  m_by_prec={scores_all[i][keys[2]]:.3f}",
            fontsize=10)

        # legend: ion colors + predicted/experimental
        handles = [plt.Line2D([0], [0], lw=3, color="#888888", label="predicted"),
                   plt.Line2D([0], [0], lw=2, color="#1a1a5e", label="experimental")]
        handles += [plt.Line2D([0], [0], lw=3, color=ION_COLORS[key],
                               label=f"{key[0]}{key[1]}+") for key in ION_COLORS]
        ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

        out_name = args.out.replace(".png", f"_{tag}.png") if args.out.endswith(".png") \
            else f"{args.out}_{tag}.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close(fig)
        print(f"saved -> {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
