"""Inference script for trained Spectrum Transformer.

Usage:
    python predict.py --checkpoint runs/spectrum_transformer/best_model.pt \\
                      --npz-path datasets/spectrum_prediction_examples_len40.npz \\
                      --output-dir predictions/

    python predict.py --checkpoint runs/spectrum_transformer/best_model.pt \\
                      --npz-path datasets/spectrum_prediction_examples_len40.npz \\
                      --indices 0 1 2 3 4 \\
                      --save-npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectrum_transformer.config import AUX_FEATURE_NAMES, DEFAULT_MIN_VALID_MZ
from spectrum_transformer.data.dataset import (
    build_valid_mask,
    build_cleavage_context_arrays,
    load_npz_data,
    normalize_features,
    normalize_inputs,
    SpectrumDataset,
)
from spectrum_transformer.models.factory import build_model, prepare_checkpoint_for_inference
from spectrum_transformer.training.engine import (
    apply_prediction_thresholds,
    compute_prediction_metrics,
)
from spectrum_transformer.training.target_processing import (
    relative_percent_to_normalized_threshold,
)
from spectrum_transformer.warnings import configure_v20_warning_filters

configure_v20_warning_filters()


DEFAULT_MULTI_THRESHOLD_PCTS = (0.5, 1.0, 2.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Spectrum Transformer."
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best_model.pt checkpoint.")
    parser.add_argument("--npz-path", type=Path,
                        default=Path("dataset/spectrum_prediction_examples_len40.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Specific sample indices to predict. If not given, predicts all.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-mz", type=float, default=DEFAULT_MIN_VALID_MZ)
    parser.add_argument("--save-npz", action="store_true",
                        help="Also save predictions as .npz file.")
    parser.add_argument("--presence-threshold", type=float, default=None,
                        help="Optional presence probability threshold for filtering predictions.")
    parser.add_argument("--prediction-threshold", type=float, default=None,
                        help="Optional final prediction cutoff for filtering weak peaks.")
    parser.add_argument("--prediction-threshold-pct", type=float, default=None,
                        help="Prediction cutoff expressed in relative-intensity percent. Converted to normalized threshold internally.")
    parser.add_argument("--mz-consolidation-radius", type=int, default=None,
                        help="Optional post-processing radius over neighbouring m/z bins.")
    parser.add_argument("--multi-threshold-pcts", type=float, nargs="*", default=None,
                        help="Export multiple prediction-cutoff results at once using relative-intensity percent thresholds, e.g. 0.5 1.0 2.0.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (0, 1, ...). Only used when CUDA is available.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: 'cpu' or 'cuda:N'. If not set, uses --gpu to select.")
    return parser.parse_args()


def format_threshold_tag(threshold_pct):
    return f"{float(threshold_pct):g}".replace(".", "p")


def to_builtin_metrics(metrics):
    return {
        key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
        for key, value in metrics.items()
    }


def build_threshold_specs(args, recommended_thresholds):
    mz_radius = (
        args.mz_consolidation_radius
        if args.mz_consolidation_radius is not None
        else recommended_thresholds.get("mz_consolidation_radius", 0)
    )
    if args.multi_threshold_pcts:
        specs = []
        for threshold_pct in args.multi_threshold_pcts:
            specs.append(
                {
                    "name": f"thr_{format_threshold_tag(threshold_pct)}pct",
                    "presence_threshold": args.presence_threshold,
                    "prediction_threshold_pct": float(threshold_pct),
                    "prediction_threshold": relative_percent_to_normalized_threshold(
                        float(threshold_pct)
                    ),
                    "mz_consolidation_radius": mz_radius,
                }
            )
        return specs

    if (
        args.prediction_threshold is None
        and args.prediction_threshold_pct is None
        and args.presence_threshold is None
    ):
        specs = []
        for threshold_pct in DEFAULT_MULTI_THRESHOLD_PCTS:
            specs.append(
                {
                    "name": f"thr_{format_threshold_tag(threshold_pct)}pct",
                    "presence_threshold": recommended_thresholds.get("presence_threshold"),
                    "prediction_threshold_pct": float(threshold_pct),
                    "prediction_threshold": relative_percent_to_normalized_threshold(
                        float(threshold_pct)
                    ),
                    "mz_consolidation_radius": mz_radius,
                }
            )
        return specs

    prediction_threshold = args.prediction_threshold
    prediction_threshold_pct = args.prediction_threshold_pct
    if prediction_threshold_pct is not None:
        prediction_threshold = relative_percent_to_normalized_threshold(
            float(prediction_threshold_pct)
        )
    if prediction_threshold is None:
        prediction_threshold = recommended_thresholds.get("prediction_threshold")

    return [
        {
            "name": "default",
            "presence_threshold": (
                args.presence_threshold
                if args.presence_threshold is not None
                else recommended_thresholds.get("presence_threshold")
            ),
            "prediction_threshold_pct": prediction_threshold_pct,
            "prediction_threshold": prediction_threshold,
            "mz_consolidation_radius": mz_radius,
        }
    ]


def main():
    args = parse_args()
    if args.mz_consolidation_radius is not None and args.mz_consolidation_radius < 0:
        raise ValueError("--mz-consolidation-radius must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config, model_state_dict = prepare_checkpoint_for_inference(ckpt)
    model = build_model(model_config).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    best_epoch = ckpt.get("best_epoch", "N/A")
    best_score = ckpt.get("best_val_score")
    print(f"Loaded checkpoint from: {args.checkpoint}")
    if best_score is None:
        print(f"Best epoch: {best_epoch}, best score: N/A")
    else:
        print(f"Best epoch: {best_epoch}, best score: {best_score:.5f}")
    print(f"Min m/z constraint: >= {float(args.min_mz):.1f}")

    recommended_thresholds = ckpt.get("recommended_thresholds", {})
    effective_peak_threshold = ckpt.get("effective_peak_threshold")
    if effective_peak_threshold is None:
        effective_peak_threshold = relative_percent_to_normalized_threshold(
            ckpt.get("effective_peak_threshold_pct", 0.0)
        )
    threshold_specs = build_threshold_specs(args, recommended_thresholds)
    print("Threshold configurations:")
    for spec in threshold_specs:
        print(
            f"  {spec['name']}: "
            f"presence={spec['presence_threshold']} "
            f"prediction={spec['prediction_threshold']} "
            f"prediction_pct={spec['prediction_threshold_pct']} "
            f"mz_radius={spec['mz_consolidation_radius']}"
        )
    print(
        "Effective peak threshold: "
        f"{ckpt.get('effective_peak_threshold_pct', 0.0)}% "
        f"(normalized={effective_peak_threshold:.6f})"
    )

    # Load data
    raw_data = load_npz_data(args.npz_path)
    channel_means = ckpt["channel_means"]
    channel_stds = ckpt["channel_stds"]
    aux_means = ckpt["aux_feature_means"]
    aux_stds = ckpt["aux_feature_stds"]

    # Select indices
    if args.indices is not None:
        indices = np.array(args.indices)
    else:
        indices = np.arange(raw_data["inputs"].shape[0])

    # Normalize inputs
    if "valid_masks" in raw_data:
        selected_valid_masks = raw_data["valid_masks"][indices]
    else:
        vm = build_valid_mask(
            raw_data["position_masks"][indices],
            raw_data["channel_masks"][indices],
        ).astype(np.float32)
        selected_valid_masks = np.broadcast_to(vm, raw_data["inputs"][indices].shape).copy()
    selected_valid_masks = selected_valid_masks * (
        raw_data["inputs"][indices] >= float(args.min_mz)
    ).astype(np.float32)
    norm_inputs = normalize_inputs(
        raw_data["inputs"][indices], selected_valid_masks,
        channel_means, channel_stds,
    )
    expected_in_channels = int(model_config.get("in_channels", norm_inputs.shape[-1]))
    if expected_in_channels == norm_inputs.shape[-1] + selected_valid_masks.shape[-1]:
        norm_inputs = np.concatenate(
            [norm_inputs, selected_valid_masks.astype(np.float32)], axis=-1
        )
    elif expected_in_channels != norm_inputs.shape[-1]:
        raise ValueError(
            f"Checkpoint expects {expected_in_channels} input channels, "
            f"but prepared {norm_inputs.shape[-1]} channels."
        )
    norm_aux = normalize_features(raw_data["aux_features"][indices], aux_means, aux_stds)
    cleavage_aa_ids, cleavage_scalar_features = build_cleavage_context_arrays(
        raw_data.get("seqs", None)[indices] if raw_data.get("seqs", None) is not None else None,
        raw_data["lengths"][indices],
    )

    dataset = SpectrumDataset(
        inputs=norm_inputs,
        targets=raw_data["targets"][indices],
        valid_masks=raw_data["valid_masks"][indices],
        aux_features=norm_aux,
        charges=raw_data["charges"][indices],
        lengths=raw_data["lengths"][indices],
        dataset_indices=indices,
        cleavage_aa_ids=cleavage_aa_ids,
        cleavage_scalar_features=cleavage_scalar_features,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    all_predictions = {spec["name"]: [] for spec in threshold_specs}
    all_targets = []
    all_indices = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="predict batches", leave=False, dynamic_ncols=True)
        for batch in pbar:
            inputs_batch = batch["input"].to(device)
            aux_batch = batch["aux_features"].to(device)
            valid_mask_batch = batch["valid_mask"].to(device)
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
            output = model(
                inputs_batch,
                aux_batch,
                valid_mask=valid_mask_batch,
                cleavage_aa_ids=cleavage_aa_ids,
                cleavage_scalar_features=cleavage_scalar_features,
                charge=charge,
                length=length,
            )
            for spec in threshold_specs:
                prediction_source = output["prediction"]
                pred = apply_prediction_thresholds(
                    prediction_source,
                    valid_mask_batch,
                    presence_threshold=spec["presence_threshold"],
                    prediction_threshold=spec["prediction_threshold"],
                    mz_consolidation_radius=spec["mz_consolidation_radius"],
                ).cpu().numpy()
                all_predictions[spec["name"]].append(pred)
            tgt = batch["target"].cpu().numpy()
            idx = batch["dataset_index"].cpu().numpy()
            all_targets.append(tgt)
            all_indices.append(idx)
            total_seen = sum(arr.shape[0] for arr in all_targets)
            pbar.set_postfix(samples=total_seen)
        pbar.close()

    predictions_by_threshold = {
        name: np.concatenate(preds, axis=0)
        for name, preds in all_predictions.items()
    }
    targets = np.concatenate(all_targets, axis=0)
    pred_indices = np.concatenate(all_indices, axis=0)

    # Compute summary metrics
    valid_mask = raw_data["valid_masks"][indices]
    metrics_by_threshold = {}
    print(f"Predictions: {targets.shape[0]} samples")
    for spec in threshold_specs:
        name = spec["name"]
        metrics = compute_prediction_metrics(
            torch.from_numpy(predictions_by_threshold[name]),
            torch.from_numpy(targets),
            torch.from_numpy(valid_mask),
            effective_peak_threshold=effective_peak_threshold,
        )
        metrics = to_builtin_metrics(metrics)
        metrics_by_threshold[name] = metrics
        print(
            f"[{name}] "
            f"MAE={metrics['masked_mae']:.5f} "
            f"PosMAE={metrics['positive_mae']:.5f} "
            f"Cosine={metrics['cosine']:.5f} "
            f"PredNZ={metrics['predicted_nonzero_ratio']:.5f} "
            f"TargetNZ={metrics['target_nonzero_ratio']:.5f}"
        )

    # Save results
    results = {
        "checkpoint": str(args.checkpoint),
        "npz_path": str(args.npz_path),
        "indices": indices.tolist(),
        "effective_peak_threshold_pct": ckpt.get("effective_peak_threshold_pct", 0.0),
        "effective_peak_threshold": effective_peak_threshold,
        "threshold_configs": threshold_specs,
        "predictions_shape": list(targets.shape),
        "metrics_by_threshold": metrics_by_threshold,
    }
    results_path = args.output_dir / "prediction_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to: {results_path}")

    if args.save_npz:
        for spec in threshold_specs:
            name = spec["name"]
            npz_path = args.output_dir / f"predictions_{name}.npz"
            np.savez_compressed(
                npz_path,
                predictions=predictions_by_threshold[name],
                targets=targets,
                indices=pred_indices,
                prediction_threshold=(
                    np.asarray(spec["prediction_threshold"], dtype=np.float32)
                    if spec["prediction_threshold"] is not None else np.asarray(np.nan, dtype=np.float32)
                ),
                prediction_threshold_pct=(
                    np.asarray(spec["prediction_threshold_pct"], dtype=np.float32)
                    if spec["prediction_threshold_pct"] is not None else np.asarray(np.nan, dtype=np.float32)
                ),
            )
            print(f"Predictions saved to: {npz_path}")


if __name__ == "__main__":
    main()
