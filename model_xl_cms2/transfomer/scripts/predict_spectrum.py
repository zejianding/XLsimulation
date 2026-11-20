#!/usr/bin/env python3
"""从肽段序列预测理论质谱，输出 MGF 格式。

复用 predict_npz_to_mgf 的 cluster_postprocess 后处理流程，
从 build_input_mz_matrix 取真实 m/z 值，从 model 预测取强度，
经 run_cluster_postprocess + merge_overlapping_peaks → MGF。

用法:
  python scripts/predict_spectrum.py \
      --checkpoint /path/to/best_model.pt \
      --sequence "LLPLAYK" --charge 2 \
      --output predicted.mgf

  python scripts/predict_spectrum.py \
      --checkpoint /path/to/best_model.pt \
      --fasta peptides.fasta --charge 2 \
      --output predicted.mgf
"""
from __future__ import annotations

import argparse, re, sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRANSFORMER_DIR = _SCRIPT_DIR.parent
if str(_TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_DIR))

from spectrum_transformer.config import MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS
from spectrum_transformer.models.factory import build_model, prepare_checkpoint_for_inference

# Same postprocess pipeline as predict_npz_to_mgf.py
from cluster_postprocess import (
    SCENE_TEMPLATES, resolve_scene_params, run_cluster_postprocess,
    load_n_true_mean_from_stats,
)
import candidate_rescoring as _cr


def merge_overlapping_peaks(candidates, merge_tol=0.05, mz_mode="max"):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x["mz"])
    merged, i, n = [], 0, len(candidates)
    while i < n:
        anchor_mz = candidates[i]["mz"]
        cluster = [candidates[i]]
        j = i + 1
        while j < n and candidates[j]["mz"] <= anchor_mz + merge_tol:
            cluster.append(candidates[j]); j += 1
        intensities = np.array([x["intensity"] for x in cluster], dtype=np.float64)
        mzs = np.array([x["mz"] for x in cluster], dtype=np.float64)
        best_idx = int(np.argmax(intensities))
        merged.append({"mz": float(mzs[best_idx]), "intensity": float(intensities[best_idx])})
        i = j
    return merged


def predict_grid(model, model_config, sequence, charge, pepmass, rt, stats, device, min_mz=100.0):
    """Predict spectral grid for a single peptide. Returns (pred_grid, input_mz, valid_mask)."""
    K = 1
    use_seq_context = bool(model_config.get("use_sequence_context", True))

    input_grids = np.zeros((K, MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    valid_masks = np.zeros((K, MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    lengths = np.zeros(K, dtype=np.int64)
    aa_ids_batch = np.zeros((K, MAX_CLEAVAGE_LEN, 2), dtype=np.int64)
    scalar_batch = np.zeros((K, MAX_CLEAVAGE_LEN, 7), dtype=np.float32)

    input_grids[0] = _cr.build_input_mz_matrix(sequence, min_mz=min_mz)
    pos_mask, cc = _cr.build_position_mask(sequence)
    ch_mask = _cr.build_channel_mask(charge)
    mz_mask = _cr.build_input_mz_validity(sequence, min_mz=min_mz)
    vm = pos_mask[:, None, None] * ch_mask[None, None, :]
    vm = np.broadcast_to(vm, (MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS)).copy()
    valid_masks[0] = (vm * mz_mask).astype(np.float32)
    lengths[0] = cc + 1

    if use_seq_context:
        aa_ids, scalar = _cr.encode_cleavage_context(sequence, cc + 1, MAX_CLEAVAGE_LEN)
        aa_ids_batch[0] = aa_ids.astype(np.int64)
        scalar_batch[0] = scalar.astype(np.float32)

    expected_in_channels = int(model_config.get("in_channels", 6))
    norm_inputs = np.zeros_like(input_grids)
    norm_inputs[0] = _cr.prepare_model_input(
        input_grids[0], valid_masks[0], stats, expected_in_channels
    )[0]

    raw_aux = np.array([[float(charge), pepmass, rt]], dtype=np.float32)
    aux_norm = ((raw_aux - stats["aux_feature_means"]) / np.maximum(stats["aux_feature_stds"], 1e-6)).astype(np.float32)

    forward_kwargs = {
        "aux_features": torch.from_numpy(aux_norm).to(device),
        "valid_mask": torch.from_numpy(valid_masks).to(device),
        "charge": torch.from_numpy(np.array([charge], dtype=np.int64)).to(device),
        "length": torch.from_numpy(lengths.astype(np.int64)).to(device),
    }
    if use_seq_context:
        forward_kwargs["cleavage_aa_ids"] = torch.from_numpy(aa_ids_batch).to(device)
        forward_kwargs["cleavage_scalar_features"] = torch.from_numpy(scalar_batch).to(device)

    with torch.inference_mode():
        out = model(torch.from_numpy(norm_inputs).to(device), **forward_kwargs)
    pred = out["prediction"].cpu().numpy().astype(np.float32)[0]
    return pred, input_grids[0], valid_masks[0]


def grid_to_peaks_cluster(pred_grid, input_mz_grid, valid_mask, charge, peptide_length,
                          scene="v20_pp4", merge_tol=0.05, mz_mode="max"):
    """Use cluster postprocess to extract peaks from predicted grid."""
    params = resolve_scene_params(scene)
    cells, _ = run_cluster_postprocess(
        pred_matrix=pred_grid.copy(),
        input_mz=input_mz_grid,
        peptide_length=peptide_length,
        precursor_charge=charge,
        scene=scene,
        params=params,
        n_true_mean=None,
    )
    return merge_overlapping_peaks(cells, merge_tol=merge_tol, mz_mode=mz_mode)


def write_mgf_block(fout, seq, charge, pepmass, rt, peaks, scan_id, source_idx):
    fout.write("BEGIN IONS\n")
    fout.write(f"TITLE=PREDICTED_{source_idx} scan={scan_id}\n")
    fout.write(f"PEPMASS={pepmass:.6f}\n")
    fout.write(f"CHARGE={charge}+\n")
    fout.write(f"RTINSECONDS={rt:.6f}\n")
    fout.write(f"SEQ={seq}\n")
    for p in sorted(peaks, key=lambda x: x["mz"]):
        fout.write(f"{p['mz']:.5f} {p['intensity']:.6f}\n")
    fout.write("END IONS\n\n")


def parse_fasta(fasta_path):
    from Bio import SeqIO
    return [(r.id, str(r.seq)) for r in SeqIO.parse(fasta_path, "fasta")]


def count_peptide_residues(seq):
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    return sum(1 for ch in str(seq or "") if ch in aa)


def main():
    p = argparse.ArgumentParser(description="Predict theoretical spectra from peptide sequences using V20 model")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt")
    p.add_argument("--sequence", type=str, default=None, help="Single peptide sequence")
    p.add_argument("--fasta", type=Path, default=None, help="FASTA file with multiple sequences")
    p.add_argument("--charge", type=int, default=2, help="Precursor charge")
    p.add_argument("--pepmass", type=float, default=None, help="Override precursor mass")
    p.add_argument("--rt", type=float, default=1800.0, help="Retention time (sec)")
    p.add_argument("--output", type=Path, default=Path("predicted.mgf"), help="Output MGF file")
    p.add_argument("--min-mz", type=float, default=100.0)
    p.add_argument("--merge-tol", type=float, default=0.05, help="m/z merge tolerance (Da)")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    if not args.sequence and not args.fasta:
        p.error("Must provide --sequence or --fasta")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_config = prepare_checkpoint_for_inference(checkpoint)
    model_config = ckpt_config.get("model_config", ckpt_config)
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval()
    print(f"Model: in_channels={model_config.get('in_channels',6)}, "
          f"d_model={model_config.get('d_model','?')}, "
          f"seq_context={bool(model_config.get('use_sequence_context', True))}")

    stats = checkpoint.get("stats", checkpoint.get("training_stats", {}))
    if not stats:
        print("Warning: no stats in checkpoint, using defaults")
        stats = {
            "aux_feature_means": np.array([2.0, 800.0, 1800.0], dtype=np.float32),
            "aux_feature_stds": np.array([1.0, 400.0, 600.0], dtype=np.float32),
        }

    sequences = []
    if args.sequence:
        sequences = [(f"seq_0", args.sequence)]
    if args.fasta:
        sequences.extend(parse_fasta(args.fasta))

    with open(args.output, 'w') as fout:
        for idx, (seq_id, seq) in enumerate(sequences):
            if args.pepmass:
                pepmass = args.pepmass
            else:
                pepmass = sum(_cr._AA_MASSES.get(t, _cr._token_mass_fallback(t))
                             for t in _cr._tokenize_peptide(seq)) + 18.010565

            length = max(count_peptide_residues(seq), 1)
            print(f"[{idx+1}/{len(sequences)}] {seq} mass={pepmass:.2f} charge={args.charge}+ length={length} ...")

            pred_grid, input_mz, valid_mask = predict_grid(
                model, model_config, seq, args.charge, pepmass, args.rt,
                stats, device, args.min_mz
            )
            peaks = grid_to_peaks_cluster(
                pred_grid, input_mz, valid_mask, args.charge, length,
                merge_tol=args.merge_tol
            )
            print(f"  → {len(peaks)} peaks")
            write_mgf_block(fout, seq, args.charge, pepmass, args.rt, peaks, seq_id, idx)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
