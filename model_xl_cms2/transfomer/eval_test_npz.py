#!/usr/bin/env python3
"""Evaluate a V20 checkpoint on a raw NPZ split (test/val) using the exact
training-time data path, so the reported metrics match train.py's test eval.

Reuses ``_build_dataset_from_path`` (training-time normalization, handles both
valid_masks and position_masks/channel_masks layouts) and ``evaluate_v20``
(masked cosine, MAE, subgroup/family metrics).

Usage (server):
    cd /home/wangyuhang/seq2mass/v20_0701/transfomer
    python eval_test_npz.py \
        --checkpoint /home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt \
        --npz /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_test.npz \
        --batch-size 64 --gpu 0 --out-json test_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Checkpoints saved with NumPy 2 may pickle under ``numpy._core``; register
# aliases so torch.load can read server checkpoints on older local environments.
try:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
except Exception:
    pass

if os.name == "nt":
    try:
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass

from spectrum_transformer.warnings import configure_v20_warning_filters  # noqa: E402
configure_v20_warning_filters()

from spectrum_transformer.data.dataset import load_npz_data  # noqa: E402
from spectrum_transformer.models.factory import (  # noqa: E402
    build_model,
    prepare_checkpoint_for_inference,
)
from spectrum_transformer.cli.train import _build_dataset_from_path, set_seed  # noqa: E402
from spectrum_transformer.training.engine import evaluate_v20  # noqa: E402


KEY_METRICS = [
    "cosine", "masked_mae", "positive_mae", "density_gap",
    "pred_nonzero_ratio", "target_nonzero_ratio",
    "z2_cosine", "z3_cosine", "z2_short_cosine", "z3_short_cosine",
    "main_cosine", "satellite_cosine", "residue_event_cosine",
    "peak_recall_0p5_2", "peak_recall_2_10", "peak_recall_10_plus",
]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a V20 checkpoint on an NPZ split.")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--npz", required=True, help="Path to test/val .npz")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--min-mz", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default=None, help="Optional path to save full metrics JSON")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    # --gpu <0 或缺 CUDA 时退回 CPU（否则在有 GPU 的机器上会拼出非法的 "cuda:-1"）
    if args.gpu is not None and int(args.gpu) >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config, state_dict = prepare_checkpoint_for_inference(ckpt)
    model = build_model(model_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    ch_means = np.asarray(ckpt["channel_means"], dtype=np.float32)
    ch_stds = np.asarray(ckpt["channel_stds"], dtype=np.float32)
    aux_means = np.asarray(ckpt["aux_feature_means"], dtype=np.float32)
    aux_stds = np.asarray(ckpt["aux_feature_stds"], dtype=np.float32)

    raw = load_npz_data(args.npz)
    n = raw["inputs"].shape[0]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"NPZ:        {args.npz}  ({n} samples, inputs {raw['inputs'].shape})")
    print(f"Device:     {device}")

    ds = _build_dataset_from_path(
        Path(args.npz), ch_means, ch_stds, aux_means, aux_stds,
        raw=raw, min_valid_mz=args.min_mz,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics = evaluate_v20(model, loader, device, desc="test")

    print("\n===== TEST METRICS =====")
    for k in KEY_METRICS:
        if k in metrics:
            print(f"  {k:>24} = {metrics[k]:.6f}")

    if args.out_json:
        out = Path(args.out_json)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull metrics saved to: {out}")


if __name__ == "__main__":
    main()
