"""V20 native training CLI.

This is a reduced training surface intended only for the V20 clean baseline.
It deliberately removes the historical multi-generation CLI complexity.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectrum_transformer.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_D_MODEL,
    DEFAULT_EPOCHS,
    DEFAULT_FF_DIM,
    DEFAULT_LR,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_MIN_VALID_MZ,
    DEFAULT_WEIGHT_DECAY,
)
from spectrum_transformer.data.dataset import (
    LazySpectrumDataset,
    ShardedSpectrumDataset,
    SpectrumDataset,
    compute_channel_stats,
    compute_channel_stats_from_position_channel_masks,
    compute_feature_stats,
    load_npz_data,
    normalize_features,
    normalize_inputs,
)
from spectrum_transformer.models.factory import build_model
from spectrum_transformer.training.losses import masked_cosine_loss
from spectrum_transformer.training.engine import (
    collect_v20_monitor_stats,
    compute_family_metrics,
    compute_prediction_metrics,
    evaluate_v20,
)
from spectrum_transformer.training.logger import TrainingLogger
from spectrum_transformer.training.losses import v20_compound_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_v20_selection_score(metrics: dict) -> tuple[float, dict]:
    cosine = float(metrics.get("cosine", 0.0))
    positive_mae = float(metrics.get("positive_mae", 0.0))
    density_gap = float(metrics.get("density_gap", 0.0))
    z2_gen_gap = float(metrics.get("z2_gen_gap", 0.0))
    z3_gen_gap = float(metrics.get("z3_gen_gap", 0.0))
    z2s_gen_gap = float(metrics.get("z2s_gen_gap", 0.0))
    z3s_gen_gap = float(metrics.get("z3s_gen_gap", 0.0))
    z2_cosine = float(metrics.get("z2_cosine", 0.0))
    z3_cosine = float(metrics.get("z3_cosine", 0.0))
    z2_short_cosine = float(metrics.get("z2_short_cosine", 0.0))
    z2_masked_mae = float(metrics.get("z2_masked_mae", 0.0))
    z2_short_masked_mae = float(metrics.get("z2_short_masked_mae", 0.0))
    z3_short_cosine = float(metrics.get("z3_short_cosine", 0.0))
    z3_masked_mae = float(metrics.get("z3_masked_mae", 0.0))
    z3_short_masked_mae = float(metrics.get("z3_short_masked_mae", 0.0))
    z2_long_cosine = float(metrics.get("z2_long_cosine", 0.0))
    z3_long_cosine = float(metrics.get("z3_long_cosine", 0.0))
    z2_long_masked_mae = float(metrics.get("z2_long_masked_mae", 0.0))
    z3_long_masked_mae = float(metrics.get("z3_long_masked_mae", 0.0))
    density_penalty = max(abs(density_gap) - 0.02, 0.0)

    z2_block = (
        0.42 * z2_cosine
        + 0.24 * z2_short_cosine
        + 0.18 * z2_long_cosine
        - 0.10 * z2_masked_mae
        - 0.10 * z2_short_masked_mae
        - 0.06 * z2_long_masked_mae
    )
    z3_block = (
        0.45 * z3_cosine
        + 0.28 * z3_short_cosine
        + 0.20 * z3_long_cosine
        - 0.11 * z3_masked_mae
        - 0.12 * z3_short_masked_mae
        - 0.07 * z3_long_masked_mae
    )
    dual_floor = min(z2_block, z3_block)
    z3_gap_penalty = max(z3_gen_gap - 0.08, 0.0) + max(z3s_gen_gap - 0.08, 0.0)
    z2_gap_penalty = max(z2_gen_gap - 0.08, 0.0) + max(z2s_gen_gap - 0.08, 0.0)

    score = (
        0.80 * dual_floor
        + 0.50 * z2_block
        + 0.70 * z3_block
        + 0.60 * cosine
        - 0.20 * positive_mae
        - 0.20 * density_penalty
        - 0.20 * z3_gap_penalty
        - 0.10 * z2_gap_penalty
    )

    explain = {
        "dual_floor_term": 0.80 * dual_floor,
        "z2_block_term": 0.50 * z2_block,
        "z3_block_term": 0.70 * z3_block,
        "overall_cosine_term": 0.60 * cosine,
        "positive_mae_penalty": -0.20 * positive_mae,
        "density_penalty": -0.20 * density_penalty,
        "z3_gap_penalty": -0.20 * z3_gap_penalty,
        "z2_gap_penalty": -0.10 * z2_gap_penalty,
        "z2_block": z2_block,
        "z3_block": z3_block,
        "dual_floor": dual_floor,
        "total": score,
    }
    return score, explain


def _mean_mapping(total_mapping: dict[str, float], count: int) -> dict[str, float]:
    denom = max(count, 1)
    return {key: float(value) / denom for key, value in total_mapping.items()}


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metric(metrics: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def _format_metrics_line(prefix: str, metrics: dict) -> str:
    ordered_keys = [
        "cosine",
        "masked_mae",
        "positive_mae",
        "target_nonzero_ratio",
        "density_gap",
        "pred_nonzero_ratio",
        "z2_fraction",
        "z3_fraction",
        "short_fraction",
        "z2_short_fraction",
        "z3_short_fraction",
        "z2_long_fraction",
        "z3_long_fraction",
        "z2_cosine",
        "z3_cosine",
        "z2_short_cosine",
        "z3_short_cosine",
        "z2_long_cosine",
        "z3_long_cosine",
        "z2_masked_mae",
        "z3_masked_mae",
        "z2_short_masked_mae",
        "z3_short_masked_mae",
        "z2_long_masked_mae",
        "z3_long_masked_mae",
    ]
    parts = [prefix]
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={_format_metric_value(metrics[key])}")
    return " ".join(parts)


def _format_monitor_block(label: str, metrics: dict) -> list[str]:
    z2_block = (
        0.42 * _metric(metrics, "z2_cosine")
        + 0.24 * _metric(metrics, "z2_short_cosine")
        + 0.18 * _metric(metrics, "z2_long_cosine")
        - 0.10 * _metric(metrics, "z2_masked_mae")
        - 0.10 * _metric(metrics, "z2_short_masked_mae")
        - 0.06 * _metric(metrics, "z2_long_masked_mae")
    )
    z3_block = (
        0.45 * _metric(metrics, "z3_cosine")
        + 0.28 * _metric(metrics, "z3_short_cosine")
        + 0.20 * _metric(metrics, "z3_long_cosine")
        - 0.11 * _metric(metrics, "z3_masked_mae")
        - 0.12 * _metric(metrics, "z3_short_masked_mae")
        - 0.07 * _metric(metrics, "z3_long_masked_mae")
    )
    dual_floor = min(z2_block, z3_block)
    return [
        (
            f"{label}: "
            f"cos={_metric(metrics, 'cosine'):.4f} "
            f"mae={_metric(metrics, 'masked_mae'):.4f} "
            f"pos_mae={_metric(metrics, 'positive_mae'):.4f} "
            f"p_f1={_metric(metrics, 'presence_f1'):.4f}"
        ),
        (
            f"  dual: "
            f"dual_floor={dual_floor:.4f} "
            f"z2_block={z2_block:.4f} "
            f"z3_block={z3_block:.4f}"
        ),
        (
            f"  z23: "
            f"z2_cos={_metric(metrics, 'z2_cosine'):.4f} "
            f"z2_mae={_metric(metrics, 'z2_masked_mae'):.4f} "
            f"z3_cos={_metric(metrics, 'z3_cosine'):.4f} "
            f"z3_mae={_metric(metrics, 'z3_masked_mae'):.4f}"
        ),
        (
            f"  z23+ : "
            f"z2s_cos={_metric(metrics, 'z2_short_cosine'):.4f} "
            f"z2s_mae={_metric(metrics, 'z2_short_masked_mae'):.4f} "
            f"z3s_cos={_metric(metrics, 'z3_short_cosine'):.4f} "
            f"z3s_mae={_metric(metrics, 'z3_short_masked_mae'):.4f}"
        ),
        (
            f"  z23++: "
            f"z2l_cos={_metric(metrics, 'z2_long_cosine'):.4f} "
            f"z2l_mae={_metric(metrics, 'z2_long_masked_mae'):.4f} "
            f"z3l_cos={_metric(metrics, 'z3_long_cosine'):.4f} "
            f"z3l_mae={_metric(metrics, 'z3_long_masked_mae'):.4f}"
        ),
        (
            f"  family: "
            f"main_cos={_metric(metrics, 'main_cosine'):.4f} "
            f"sat_cos={_metric(metrics, 'satellite_cosine'):.4f} "
            f"res_cos={_metric(metrics, 'residue_event_cosine'):.4f}"
        ),
        (
            f"  family_nz: "
            f"main_nz={_metric(metrics, 'main_pred_nonzero_ratio'):.4f} "
            f"sat_nz={_metric(metrics, 'satellite_pred_nonzero_ratio'):.4f} "
            f"res_nz={_metric(metrics, 'residue_event_pred_nonzero_ratio'):.4f}"
        ),
        (
            f"  density: "
            f"target_nz={_metric(metrics, 'target_nonzero_ratio'):.4f} "
            f"pred_nz={_metric(metrics, 'pred_nonzero_ratio'):.4f} "
            f"gap={_metric(metrics, 'density_gap'):.4f}"
        ),
        (
            f"  peak/energy: "
            f"peak05_2={_metric(metrics, 'peak_recall_0p5_2'):.4f} "
            f"peak2_10={_metric(metrics, 'peak_recall_2_10'):.4f} "
            f"peak10p={_metric(metrics, 'peak_recall_10_plus'):.4f} "
            f"pred={_metric(metrics, 'prediction_energy'):.4f} "
            f"target={_metric(metrics, 'target_energy'):.4f} "
            f"energy_gap={_metric(metrics, 'energy_gap'):.4f}"
        ),
    ]


def _format_epoch_summary(
    epoch: int,
    train_loss: float,
    selection_score: float,
    val_metrics: dict,
    epoch_terms: dict,
    is_best: bool = False,
) -> list[str]:
    best_mark = " *BEST*" if is_best else ""
    return [
        (
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.6f} "
            f"sel={selection_score:.6f} "
            f"val_cos={_metric(val_metrics, 'cosine'):.6f} "
            f"val_z2s_cos={_metric(val_metrics, 'z2_short_cosine'):.6f}"
            f"{best_mark}"
        ),
        (
            f"  val density: "
            f"target_nz={_metric(val_metrics, 'target_nonzero_ratio'):.4f} "
            f"pred_nz={_metric(val_metrics, 'pred_nonzero_ratio'):.4f} "
            f"gap={_metric(val_metrics, 'density_gap'):.4f}"
        ),
        (
            f"  val dual: "
            f"z2_cos={_metric(val_metrics, 'z2_cosine'):.4f} "
            f"z3_cos={_metric(val_metrics, 'z3_cosine'):.4f} "
            f"z2s_cos={_metric(val_metrics, 'z2_short_cosine'):.4f} "
            f"z3s_cos={_metric(val_metrics, 'z3_short_cosine'):.4f}"
        ),
        (
            f"  losses: "
            f"bb_global={_metric(epoch_terms, 'v20_backbone_global_loss'):.4f} "
            f"z2_recon={_metric(epoch_terms, 'v20_z2_recon_loss'):.4f} "
            f"z3_recon={_metric(epoch_terms, 'v20_z3_recon_loss'):.4f} "
            f"z3s_recon={_metric(epoch_terms, 'v20_z3_short_recon_loss'):.4f} "
            f"amp={_metric(epoch_terms, 'v20_amp_strict_loss'):.4f}"
        ),
        (
            f"  regularizers: "
            f"density_pen={_metric(epoch_terms, 'v20_density_gap_penalty'):.4f} "
            f"fam_weighted={_metric(epoch_terms, 'v20_family_weighted_loss'):.4f} "
            f"detail_cos={_metric(epoch_terms, 'v20_backbone_cosine_loss'):.4f}"
        ),
    ]


def _select_metrics(source: dict | None, keys: list[str]) -> dict:
    source = source or {}
    return {key: source.get(key) for key in keys if key in source}


CORE_METRIC_KEYS = [
    "cosine",
    "masked_mae",
    "positive_mae",
    "density_gap",
    "target_nonzero_ratio",
    "pred_nonzero_ratio",
    "z2_cosine",
    "z3_cosine",
    "z2_masked_mae",
    "z3_masked_mae",
    "z2_short_cosine",
    "z3_short_cosine",
    "z2_short_masked_mae",
    "z3_short_masked_mae",
    "z2_long_cosine",
    "z3_long_cosine",
    "z2_long_masked_mae",
    "z3_long_masked_mae",
    "main_cosine",
    "satellite_cosine",
    "residue_event_cosine",
    "peak_recall_0p5_2",
    "peak_recall_2_10",
    "peak_recall_10_plus",
    "prediction_energy",
    "target_energy",
    "energy_gap",
    "z2_gen_gap",
    "z3_gen_gap",
    "z2s_gen_gap",
    "z3s_gen_gap",
]


SECONDARY_METRIC_KEYS = [
    "presence_precision",
    "presence_recall",
    "presence_f1",
    "consensus_predicted_nonzero_ratio",
    "consensus_target_nonzero_ratio",
    "z2_fraction",
    "z3_fraction",
    "short_fraction",
    "z2_short_fraction",
    "z3_short_fraction",
    "z2_long_fraction",
    "z3_long_fraction",
    "short_cosine",
    "short_masked_mae",
    "main_masked_mae",
    "satellite_masked_mae",
    "residue_event_masked_mae",
    "main_target_nonzero_ratio",
    "main_pred_nonzero_ratio",
    "satellite_target_nonzero_ratio",
    "satellite_pred_nonzero_ratio",
    "residue_event_target_nonzero_ratio",
    "residue_event_pred_nonzero_ratio",
]


def _attach_generalization_gaps(train_natural_metrics: dict, val_metrics: dict) -> dict:
    enriched = dict(val_metrics)
    enriched["train_natural_z2_cos"] = float(train_natural_metrics.get("z2_cosine", 0.0))
    enriched["train_natural_z3_cos"] = float(train_natural_metrics.get("z3_cosine", 0.0))
    enriched["train_natural_z2s_cos"] = float(train_natural_metrics.get("z2_short_cosine", 0.0))
    enriched["train_natural_z3s_cos"] = float(train_natural_metrics.get("z3_short_cosine", 0.0))
    enriched["z2_gen_gap"] = enriched["train_natural_z2_cos"] - float(val_metrics.get("z2_cosine", 0.0))
    enriched["z3_gen_gap"] = enriched["train_natural_z3_cos"] - float(val_metrics.get("z3_cosine", 0.0))
    enriched["z2s_gen_gap"] = enriched["train_natural_z2s_cos"] - float(val_metrics.get("z2_short_cosine", 0.0))
    enriched["z3s_gen_gap"] = enriched["train_natural_z3s_cos"] - float(val_metrics.get("z3_short_cosine", 0.0))
    enriched["train_natural_density_gap"] = float(train_natural_metrics.get("density_gap", 0.0))
    return enriched


def build_parser():
    parser = argparse.ArgumentParser(description="Train the V20 clean baseline.")
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--ff-dim", type=int, default=DEFAULT_FF_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-packed", action="store_true")
    parser.add_argument("--min-valid-mz", type=float, default=DEFAULT_MIN_VALID_MZ)
    parser.add_argument("--v20-backbone-weight", type=float, default=1.0)
    parser.add_argument("--v20-strict-presence-weight", type=float, default=0.30)
    parser.add_argument("--v20-strict-threshold", type=float, default=0.03)
    parser.add_argument("--v20-group-family-focus-weight", type=float, default=0.40)
    return parser


def _build_dataset(raw, ch_means, ch_stds, aux_means, aux_stds):
    if "position_masks" in raw and "channel_masks" in raw:
        aux = normalize_features(raw["aux_features"], aux_means, aux_stds)
        return LazySpectrumDataset(
            inputs=raw["inputs"],
            targets=raw["targets"],
            aux_features=aux,
            charges=raw["charges"],
            lengths=raw["lengths"],
            dataset_indices=np.arange(raw["inputs"].shape[0], dtype=np.int64),
            channel_means=ch_means.reshape(-1),
            channel_stds=ch_stds.reshape(-1),
            position_masks=raw["position_masks"],
            channel_masks=raw["channel_masks"],
            min_mz=DEFAULT_MIN_VALID_MZ,
            cleavage_aa_ids=raw.get("cleavage_aa_ids"),
            cleavage_scalar_features=raw.get("cleavage_scalar_features"),
        )
    inputs = normalize_inputs(raw["inputs"], raw["valid_masks"], ch_means, ch_stds)
    aux = normalize_features(raw["aux_features"], aux_means, aux_stds)
    return SpectrumDataset(
        inputs=inputs,
        targets=raw["targets"],
        valid_masks=raw["valid_masks"],
        aux_features=aux,
        charges=raw["charges"],
        lengths=raw["lengths"],
        cleavage_aa_ids=raw.get("cleavage_aa_ids"),
        cleavage_scalar_features=raw.get("cleavage_scalar_features"),
    )


def _describe_source(path: Path) -> str:
    if _is_shard_dir(path):
        return "shard_dir"
    if _is_memmap_dir(path):
        return "memmap_dir"
    return "single_npz"


def _is_shard_dir(path: Path) -> bool:
    return path.is_dir() and (path / "index.json").exists()


def _is_memmap_dir(path: Path) -> bool:
    return path.is_dir() and (path / "meta.json").exists()


def _build_dataset_from_path(path: Path, ch_means, ch_stds, aux_means, aux_stds, raw=None, min_valid_mz=DEFAULT_MIN_VALID_MZ):
    if _is_shard_dir(path):
        return ShardedSpectrumDataset(
            shard_dir=path,
            channel_means=ch_means.reshape(-1),
            channel_stds=ch_stds.reshape(-1),
            aux_means=aux_means.reshape(-1),
            aux_stds=aux_stds.reshape(-1),
            min_mz=min_valid_mz,
        )
    if raw is None:
        raw = load_npz_data(path)
    if _is_memmap_dir(path):
        n = raw["inputs"].shape[0]
        aux = normalize_features(raw["aux_features"], aux_means, aux_stds)
        return LazySpectrumDataset(
            inputs=raw["inputs"],
            targets=raw["targets"],
            aux_features=aux,
            charges=raw["charges"],
            lengths=raw["lengths"],
            dataset_indices=np.arange(n, dtype=np.int64),
            channel_means=ch_means.reshape(-1),
            channel_stds=ch_stds.reshape(-1),
            position_masks=raw.get("position_masks"),
            channel_masks=raw.get("channel_masks"),
            valid_masks=raw.get("valid_masks"),
            min_mz=min_valid_mz,
            cleavage_aa_ids=raw.get("cleavage_aa_ids"),
            cleavage_scalar_features=raw.get("cleavage_scalar_features"),
        )
    return _build_dataset(raw, ch_means, ch_stds, aux_means, aux_stds)


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(args.output_dir)
    logger.log("Training run started (V20 native).")
    logger.log_kv_block("Arguments:", vars(args))
    logger.console(
        "Dataset sources: "
        f"train={_describe_source(args.train_npz)} "
        f"val={_describe_source(args.val_npz)} "
        f"test={_describe_source(args.test_npz)}"
    )

    # --gpu <0 或缺 CUDA 时退回 CPU（否则在有 GPU 的机器上会拼出非法的 "cuda:-1"）
    if args.gpu is not None and int(args.gpu) >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.console(f"Device: {device}")

    metadata_iter = tqdm(
        [("train", args.train_npz)],
        desc="load metadata",
        leave=False,
        dynamic_ncols=True,
    )
    split_raw = {}
    for split_name, split_path in metadata_iter:
        split_raw[split_name] = load_npz_data(split_path)
        split_size = split_raw[split_name]["inputs"].shape[0]
        metadata_iter.set_postfix(split=split_name, samples=split_size)
    metadata_iter.close()
    train_raw = split_raw["train"]

    if "position_masks" in train_raw and "channel_masks" in train_raw:
        ch_means, ch_stds = compute_channel_stats_from_position_channel_masks(
            train_raw["inputs"],
            train_raw["position_masks"],
            train_raw["channel_masks"],
            min_mz=args.min_valid_mz,
            chunk_size=2048,
        )
    else:
        ch_means, ch_stds = compute_channel_stats(train_raw["inputs"], train_raw["valid_masks"])
    aux_means, aux_stds = compute_feature_stats(train_raw["aux_features"])

    dataset_iter = tqdm(
        [
            ("train", args.train_npz),
            ("val", args.val_npz),
            ("test", args.test_npz),
        ],
        desc="build datasets",
        leave=False,
        dynamic_ncols=True,
    )
    built_datasets = {}
    for split_name, split_path in dataset_iter:
        if split_name not in split_raw:
            split_raw[split_name] = load_npz_data(split_path)
        built_datasets[split_name] = _build_dataset_from_path(
            split_path, ch_means, ch_stds, aux_means, aux_stds, raw=split_raw[split_name], min_valid_mz=args.min_valid_mz
        )
        dataset_iter.set_postfix(split=split_name, samples=len(built_datasets[split_name]))
    dataset_iter.close()
    train_ds = built_datasets["train"]
    val_ds = built_datasets["val"]
    test_ds = built_datasets["test"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_config = {
        "in_channels": 6,
        "out_channels": 6,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "aux_dim": 3,
        "sequence_context_dim": 64,
        "use_packed": not args.no_packed,
    }
    model = build_model(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = float("-inf")
    best_path = args.output_dir / "best_model.pt"
    best_epoch = 0
    best_val_metrics = None
    best_val_selection_explain = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_terms_total = {}
        epoch_monitor_total = {}
        epoch_global_total = {}
        batches = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"train epoch {epoch:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in train_pbar:
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            aux = batch["aux_features"].to(device)
            cleavage_aa_ids = batch.get("cleavage_aa_ids")
            cleavage_scalar_features = batch.get("cleavage_scalar_features")
            if cleavage_aa_ids is not None:
                cleavage_aa_ids = cleavage_aa_ids.to(device)
            if cleavage_scalar_features is not None:
                cleavage_scalar_features = cleavage_scalar_features.to(device)
            charge = batch.get("charge")
            length = batch.get("length")
            if charge is not None:
                charge = charge.to(device)
            if length is not None:
                length = length.to(device)

            optimizer.zero_grad()
            output = model(
                inputs,
                aux,
                valid_mask=valid_mask,
                cleavage_aa_ids=cleavage_aa_ids,
                cleavage_scalar_features=cleavage_scalar_features,
                charge=charge,
                length=length,
            )
            prediction = output["prediction"] if isinstance(output, dict) else output
            loss, loss_terms = v20_compound_loss(
                prediction,
                target,
                valid_mask,
                backbone_pred=output.get("backbone_pred") if isinstance(output, dict) else None,
                detail_pred=output.get("detail_pred") if isinstance(output, dict) else None,
                detail_logits=output.get("detail_logits") if isinstance(output, dict) else None,
                main_anchor_prob=output.get("main_anchor_prob") if isinstance(output, dict) else None,
                main_anchor_logits=output.get("main_anchor_logits") if isinstance(output, dict) else None,
                charges=charge,
                lengths=length,
                v20_backbone_weight=args.v20_backbone_weight,
                v20_strict_presence_weight=args.v20_strict_presence_weight,
                v20_strict_threshold=args.v20_strict_threshold,
                v20_group_family_focus_weight=args.v20_group_family_focus_weight,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_cosine = float(1.0 - masked_cosine_loss(prediction.detach(), target.detach(), valid_mask.detach()).cpu())
            batch_global_metrics = compute_prediction_metrics(
                prediction.detach(),
                target.detach(),
                valid_mask.detach(),
                effective_peak_threshold=0.03,
                consensus_nonzero_threshold=0.02,
            )
            for key, value in loss_terms.items():
                epoch_terms_total[key] = epoch_terms_total.get(key, 0.0) + float(value)
            for key, value in batch_global_metrics.items():
                if isinstance(value, (int, float)):
                    epoch_global_total[key] = epoch_global_total.get(key, 0.0) + float(value)
            monitor_stats = collect_v20_monitor_stats(
                prediction,
                target,
                valid_mask,
                output=output,
                charges=charge,
                lengths=length,
            )
            batch_family_metrics = compute_family_metrics(prediction, target, valid_mask)
            for key, value in monitor_stats.items():
                epoch_monitor_total[key] = epoch_monitor_total.get(key, 0.0) + float(value)
            for key, value in batch_family_metrics.items():
                epoch_monitor_total[key] = epoch_monitor_total.get(key, 0.0) + float(value)
            batches += 1
            train_pbar.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                cos=f"{batch_cosine:.4f}",
                pred_nz=f"{monitor_stats.get('pred_nonzero_ratio', 0.0):.4f}",
                gap=f"{monitor_stats.get('pred_nonzero_ratio', 0.0) - monitor_stats.get('target_nonzero_ratio', 0.0):.4f}",
                amp=f"{loss_terms.get('v20_amp_strict_loss', 0.0):.4f}",
            )
        train_pbar.close()

        epoch_terms = _mean_mapping(epoch_terms_total, batches)
        train_monitor = _mean_mapping(epoch_monitor_total, batches)
        train_global = _mean_mapping(epoch_global_total, batches)
        train_monitor.update(train_global)
        train_monitor["density_gap"] = train_monitor.get("pred_nonzero_ratio", 0.0) - train_monitor.get("target_nonzero_ratio", 0.0)
        train_monitor["energy_gap"] = train_monitor.get("prediction_energy", 0.0) - train_monitor.get("target_energy", 0.0)
        val_metrics_raw = evaluate_v20(model, val_loader, device, desc=f"val   epoch {epoch:03d}")
        val_metrics = _attach_generalization_gaps({}, val_metrics_raw)
        selection_score, selection_explain = compute_v20_selection_score(val_metrics)
        is_best = selection_score > best_val_score
        history_row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "val_selection_score": selection_score,
        }
        history_row.update({f"train_{k}": v for k, v in epoch_terms.items()})
        history_row.update({f"train_weighted_{k}": v for k, v in train_monitor.items()})
        history_row.update({f"val_{k}": v for k, v in val_metrics.items()})
        logger.log_history_row(history_row)

        for line in _format_epoch_summary(
            epoch=epoch,
            train_loss=epoch_loss / max(batches, 1),
            selection_score=selection_score,
            val_metrics=val_metrics,
            epoch_terms=epoch_terms,
            is_best=is_best,
        ):
            logger.log(line)
        for line in _format_monitor_block(f"val_monitor   epoch={epoch:03d}", val_metrics):
            logger.console(line)

        if is_best:
            best_val_score = selection_score
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_val_selection_explain = dict(selection_explain)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "channel_means": ch_means.reshape(-1).tolist(),
                    "channel_stds": ch_stds.reshape(-1).tolist(),
                    "aux_feature_means": aux_means.reshape(-1).tolist(),
                    "aux_feature_stds": aux_stds.reshape(-1).tolist(),
                    "best_val_score": best_val_score,
                    "best_val_selection_explain": best_val_selection_explain,
                    "best_val_metrics_core": _select_metrics(best_val_metrics, CORE_METRIC_KEYS),
                    "best_val_metrics_secondary": _select_metrics(best_val_metrics, SECONDARY_METRIC_KEYS),
                    "epoch": epoch,
                },
                best_path,
            )

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_v20(model, test_loader, device, desc="test final")
    test_selection_score, test_selection_explain = compute_v20_selection_score(test_metrics)
    summary = {
        "v20_identity": {"version": "v20"},
        "v20_loss_profile": {
            "v20_backbone_weight": args.v20_backbone_weight,
            "v20_strict_presence_weight": args.v20_strict_presence_weight,
            "v20_strict_threshold": args.v20_strict_threshold,
            "v20_group_family_focus_weight": args.v20_group_family_focus_weight,
        },
        "best_val_score": best_val_score,
        "best_epoch": best_epoch,
        "best_val_metrics_core": _select_metrics(best_val_metrics, CORE_METRIC_KEYS),
        "best_val_metrics_secondary": _select_metrics(best_val_metrics, SECONDARY_METRIC_KEYS),
        "best_val_selection_explain": best_val_selection_explain,
        "best_checkpoint": str(best_path),
        "test_selection_score": test_selection_score,
        "test_selection_explain": test_selection_explain,
        "test_metrics_core": _select_metrics(test_metrics, CORE_METRIC_KEYS),
        "test_metrics_secondary": _select_metrics(test_metrics, SECONDARY_METRIC_KEYS),
        "training_log_path": str(logger.log_path),
        "training_history_path": str(logger.history_path),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.log(f"Best val selection score: {best_val_score:.6f} at epoch {best_epoch:03d}")
    if best_val_metrics is not None:
        logger.console(_format_metrics_line("Best val core:", summary["best_val_metrics_core"]))
    logger.log(
        f"Test cosine: {test_metrics['cosine']:.6f} "
        f"test_selection_score={test_selection_score:.6f} "
        f"test_masked_mae={test_metrics.get('masked_mae', 0.0):.6f} "
        f"test_density_gap={test_metrics.get('density_gap', 0.0):.6f} "
        f"test_energy_gap={test_metrics.get('energy_gap', 0.0):.6f}"
    )
    logger.console(_format_metrics_line("Test core:", summary["test_metrics_core"]))


if __name__ == "__main__":
    main()
