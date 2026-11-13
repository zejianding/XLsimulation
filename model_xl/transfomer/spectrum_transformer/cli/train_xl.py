"""Train the six-channel XL Spectrum Transformer or smoke-test one real batch."""

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import islice
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from spectrum_transformer.config import (
    CHANNEL_NAMES, DEFAULT_D_MODEL, DEFAULT_DROPOUT, DEFAULT_EPOCHS, DEFAULT_FF_DIM, DEFAULT_LR,
    DEFAULT_N_HEADS, DEFAULT_N_LAYERS, DEFAULT_SEED, DEFAULT_SEQUENCE_CONTEXT_DIM, DEFAULT_WEIGHT_DECAY,
)
from spectrum_transformer.data.xl_dataset import CLEAVAGE_SCALAR_FEATURE_NAMES, XLSpectrumDataset
from spectrum_transformer.models.xl_transformer import XLSpectrumTransformer
from spectrum_transformer.training.engine import _cosine_per_sample, _masked_mae_per_sample
from spectrum_transformer.training.logger import TrainingLogger
from spectrum_transformer.training.xl_engine import (
    ANCHOR_PROBABILITY_THRESHOLD, compute_xl_batch_loss, evaluate_xl, predict_xl_batch, prepare_xl_batch,
    xl_density_per_sample,
)
from spectrum_transformer.training.xl_losses import MAIN_PEAK_THRESHOLD, SHORT_CHAIN_MAX_LENGTH, XL_LOSS_WEIGHTS
from spectrum_transformer.warnings import configure_v20_warning_filters
from token_table import CHARGE_TOKENS, MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, SATELLITE_TOKEN_IDS


DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "dataset" / "xl_spectrum_training_data"
MODEL_CONFIG = {
    "d_model": DEFAULT_D_MODEL, "n_heads": DEFAULT_N_HEADS, "n_layers": DEFAULT_N_LAYERS,
    "ff_dim": DEFAULT_FF_DIM, "dropout": DEFAULT_DROPOUT, "sequence_context_dim": DEFAULT_SEQUENCE_CONTEXT_DIM,
    "in_channels": len(CHANNEL_NAMES), "out_channels": len(CHANNEL_NAMES), "aux_dim": 3,
}


def _loader(dataset, args, device, *, shuffle=False, indices=None):
    source = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(
        source, batch_size=args.batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )


def _smoke_indices(dataset, batch_size, seed):
    order = np.random.default_rng(seed).permutation(len(dataset))
    _, first = np.unique(dataset.precursor_charges[order], return_index=True)
    representatives = order[first].tolist()
    if batch_size <= len(representatives):
        return representatives[:batch_size]
    seen = set(representatives)
    remaining = [index for index in order.tolist() if index not in seen]
    return (representatives + remaining)[:batch_size]


def _checkpoint(model, optimizer, dataset, args, epoch, metrics):
    return {
        "model_type": "xl_spectrum_transformer",
        "model_config": MODEL_CONFIG,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation": metrics,
        "selection": {"primary": "cosine", "tie_breaker": "masked_mae"},
        "data_schema_version": dataset.schema_version,
        "data_metadata": dataset.metadata,
        "channel_names": CHANNEL_NAMES,
        "cleavage_scalar_feature_names": CLEAVAGE_SCALAR_FEATURE_NAMES,
        "chain_order": "bare_sequence,modifications,link_site",
        "token_offsets_by_fragment_charge": CHARGE_TOKENS,
        "token_families": {"main": MAIN_TOKEN_IDS, "satellite": SATELLITE_TOKEN_IDS,
                           "residue_event": RESIDUE_EVENT_TOKEN_IDS},
        "loss": {"weights": XL_LOSS_WEIGHTS, "shared_main_peak_weight": 1.0,
                 "main_peak_threshold": MAIN_PEAK_THRESHOLD, "short_chain_max_length": SHORT_CHAIN_MAX_LENGTH,
                 "short_chain_scope": "per_chain", "empty_peak_weight": args.empty_peak_weight},
        "anchor_probability_threshold": ANCHOR_PROBABILITY_THRESHOLD,
        "training": {"seed": args.seed, "batch_size": args.batch_size, "num_workers": args.num_workers,
                     "epochs": args.epochs, "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
    }


def _save_checkpoint(checkpoint, destination):
    with NamedTemporaryFile(dir=destination.parent, suffix=".pt", delete=False) as stream:
        staging = Path(stream.name)
    try:
        torch.save(checkpoint, staging)
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


def _backward(model, optimizer, batch, *, empty_peak_weight):
    optimizer.zero_grad(set_to_none=True)
    output = predict_xl_batch(model, batch)
    loss, terms = compute_xl_batch_loss(output, batch, empty_peak_weight=empty_peak_weight)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite XL training loss")
    loss.backward()
    return output, loss, terms


def _history_fields(prefix, metrics):
    for name, value in metrics.items():
        field = f"{prefix}_{name}"
        if isinstance(value, dict):
            yield from _history_fields(field, value)
        else:
            yield field, value


def _smoke(model, optimizer, dataset, args, device):
    indices = _smoke_indices(dataset, args.batch_size, args.seed)
    raw_batch = next(iter(_loader(dataset, args, device, indices=indices)))
    batch = prepare_xl_batch(raw_batch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output, loss, terms = _backward(model, optimizer, batch, empty_peak_weight=args.empty_peak_weight)
    expected_shape = batch["target"].shape
    for key in ("prediction", "backbone_pred", "detail_pred", "detail_logits", "main_anchor_logits", "main_anchor_prob"):
        if output[key].shape != expected_shape or not torch.isfinite(output[key]).all():
            raise RuntimeError(f"Invalid XL output {key}")
    for key in ("prediction", "main_anchor_prob"):
        if torch.count_nonzero(output[key][~batch["valid_mask"]]):
            raise RuntimeError(f"Nonzero XL {key} outside valid_mask")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not torch.stack([torch.isfinite(grad).all() for grad in gradients]).all():
        raise RuntimeError("Missing or non-finite XL gradients")
    parameter = next((p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad)), None)
    if parameter is None:
        raise RuntimeError("All XL gradients are zero")
    before_step = parameter.detach().clone()
    optimizer.step()
    if torch.equal(before_step, parameter):
        raise RuntimeError("XL optimizer did not change a parameter")
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    del output

    metrics = evaluate_xl(model, [raw_batch], device)
    checkpoint = io.BytesIO()
    torch.save(_checkpoint(model, optimizer, dataset, args, 0, metrics), checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded_model = XLSpectrumTransformer(**restored["model_config"]).to(device).eval()
    reloaded_model.load_state_dict(restored["model_state_dict"])
    with torch.inference_mode():
        torch.testing.assert_close(predict_xl_batch(model, batch)["prediction"],
                                   predict_xl_batch(reloaded_model, batch)["prediction"])
    report = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "shape": list(expected_shape), "charges": raw_batch["precursor_charge"].tolist(),
        "lengths": raw_batch["lengths"].tolist(), "loss": loss.item(), "loss_terms": terms,
        "gradient_tensors": len(gradients), "peak_cuda_allocated_gib": peak_gib,
        "checkpoint_reload": "passed", "post_step_batch_cosine": metrics["cosine"],
    }
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def _probe_indices(dataset, seed, per_charge):
    rng = np.random.default_rng(seed)
    charges = dataset.precursor_charges
    return [index for charge in np.unique(charges)
            for index in rng.permutation(np.flatnonzero(charges == charge))[:per_charge].tolist()]


def _token_variation(hidden, valid):
    """Mean per-pair RMS of the channel-wise standard deviations over valid tokens."""
    valid = valid.unsqueeze(-1)
    count = valid.sum(dim=1).clamp_min(1)
    mean = (hidden * valid).sum(dim=1) / count
    variance = ((hidden - mean[:, None, :]).square() * valid).sum(dim=1) / count
    return variance.mean(dim=-1).sqrt().mean().item()


def _activation_probe(model, batch):
    counts = batch["valid_mask"].any(dim=-1).flatten(1).sum(1)
    counts = counts[counts > 0]
    variation = {}

    def record(name, hidden):
        packed_valid = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < counts[:, None]
        variation[name] = _token_variation(hidden, packed_valid)

    def record_layer(layer_number):
        def hook(_module, inputs, output):
            if layer_number == 1:
                record("encoder_input", inputs[0])
            record(f"shared_layer_{layer_number}", output)
        return hook

    layers = model.shared_encoder.encoder.layers
    handles = [layers[number - 1].register_forward_hook(record_layer(number)) for number in (1, 3, 5)]
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        # Keep the packed tensor dense so eval-mode layer hooks see the padding layout.
        torch.backends.mha.set_fastpath_enabled(False)
        prediction = predict_xl_batch(model, batch)["prediction"]
        altered = dict(batch)
        for key in ("input", "aux_features", "cleavage_aa_ids", "cleavage_scalar_features"):
            altered[key] = torch.zeros_like(batch[key])
        for handle in handles:
            handle.remove()
        handles.clear()
        perturbed = predict_xl_batch(model, altered)["prediction"]
    finally:
        for handle in handles:
            handle.remove()
        torch.backends.mha.set_fastpath_enabled(fastpath)
    valid = batch["valid_mask"]
    delta = (prediction - perturbed).abs()[valid]
    return {"token_variation_rms": variation,
            "input_ablation_prediction_mae": delta.mean().item() if delta.numel() else None}


def _fit_prediction_template(model, loader, device):
    family = model.token_family_index
    sums, counts = {}, {}
    for raw in loader:
        batch = prepare_xl_batch(raw, device)
        prediction = predict_xl_batch(model, batch)["prediction"]
        for charge in raw["precursor_charge"].unique().tolist():
            selected = batch["precursor_charge"] == charge
            valid = batch["valid_mask"][selected]
            total = (prediction[selected] * valid).sum(dim=(0, 1, 2), dtype=torch.float64)
            count = valid.sum(dim=(0, 1, 2))
            sums[charge] = sums.get(charge, 0) + total
            counts[charge] = counts.get(charge, 0) + count
    templates = {}
    for charge, total in sums.items():
        template = torch.zeros_like(total, dtype=torch.float32)
        for family_id in range(3):
            mask = family == family_id
            template[mask] = (total[mask].sum(0) / counts[charge][mask].sum(0).clamp_min(1)).float()
        templates[charge] = template
    return templates


def _evaluate_step(model, loader, templates, device):
    rows, template_rows = defaultdict(list), defaultdict(list)
    for raw in loader:
        batch = prepare_xl_batch(raw, device)
        prediction = predict_xl_batch(model, batch)["prediction"]
        target, valid = batch["target"], batch["valid_mask"]
        cosine = _cosine_per_sample(prediction, target, valid)
        positive_mae = _masked_mae_per_sample(prediction, target, (target > 0) & valid)
        pred_density, target_density = xl_density_per_sample(prediction, target, valid)
        for charge in raw["precursor_charge"].unique().tolist():
            selected = batch["precursor_charge"] == charge
            rows[charge].append(torch.stack((cosine, positive_mae, pred_density, target_density), dim=1)[selected])
            if charge in templates:
                reference = templates[charge].expand_as(target[selected])
                reference_cosine = _cosine_per_sample(reference, target[selected], valid[selected])
                template_rows[charge].append(torch.stack((reference_cosine, cosine[selected] - reference_cosine), dim=1))

    def summarize(metrics, comparison):
        values = torch.cat(metrics)
        names = ("cosine", "positive_mae", "pred_nonzero_ratio", "target_nonzero_ratio")
        report = dict(zip(names, values.mean(dim=0).tolist()))
        report["samples"] = len(values)
        if comparison:
            paired = torch.cat(comparison)
            report["template_cosine"], report["cosine_gain_over_template"] = paired.mean(dim=0).tolist()
            report["template_comparison_samples"] = len(paired)
        return report

    report = summarize([row for group in rows.values() for row in group],
                       [row for group in template_rows.values() for row in group])
    report["by_charge"] = {charge: summarize(group, template_rows[charge]) for charge, group in rows.items()}
    return report


def _step_probe(model, probe_batch, fit_loader, val_loader, device):
    was_training = model.training
    try:
        model.eval()
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            activations = _activation_probe(model, probe_batch)
            templates = _fit_prediction_template(model, fit_loader, device)
            validation = _evaluate_step(model, val_loader, templates, device)
        return {**activations, "validation": validation}
    finally:
        model.train(was_training)


def _gradient_norms(model):
    layers = model.shared_encoder.encoder.layers
    modules = {"input_proj": model.input_proj, "shared_layer_1": layers[0], "shared_layer_3": layers[2],
               "shared_layer_5": layers[4], "backbone_head": model.backbone_head}
    norms = {}
    for name, module in modules.items():
        values = [parameter.grad.detach().float().square().sum()
                  for parameter in module.parameters() if parameter.grad is not None]
        norms[name] = torch.stack(values).sum().sqrt().item() if values else None
    return norms


def _train_steps(model, optimizer, train_data, args, device):
    val_data = XLSpectrumDataset(args.dataset_dir, "val")
    if not len(val_data):
        raise ValueError("XL step diagnostics require a nonempty validation split")
    fit_indices = _probe_indices(train_data, args.seed, per_charge=16)
    val_indices = _probe_indices(val_data, args.seed, per_charge=32)
    probe_indices = _smoke_indices(val_data, args.batch_size, args.seed)
    train_loader = _loader(train_data, args, device, shuffle=True)
    fit_loader = _loader(train_data, args, device, indices=fit_indices)
    val_loader = _loader(val_data, args, device, indices=val_indices)
    raw_probe = val_data.collate_fn([val_data[index] for index in probe_indices])
    probe_batch = prepare_xl_batch(raw_probe, device)
    probe_steps = {0, 50, 100, 200, 500, args.max_steps}
    gradient_steps = probe_steps | {1}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log_path = args.output_dir / "step_metrics.jsonl"
    with log_path.open("x", encoding="utf-8", buffering=1) as stream:
        def log(event, **fields):
            stream.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")

        log("config", lr=args.lr, max_steps=args.max_steps, seed=args.seed, batch_size=args.batch_size,
            num_workers=args.num_workers, weight_decay=args.weight_decay, optimizer="AdamW",
            device=str(device), dataset_dir=str(args.dataset_dir.resolve()), model_config=MODEL_CONFIG,
            loss_weights=XL_LOSS_WEIGHTS, empty_peak_weight=args.empty_peak_weight,
            template_train_indices=fit_indices, validation_indices=val_indices,
            activation_probe_group_ids=raw_probe["group_id"])
        log("probe", step=0, **_step_probe(model, probe_batch, fit_loader, val_loader, device))
        step = 0
        with tqdm(total=args.max_steps, desc="XL optimizer steps", unit="step", file=sys.stderr) as progress:
            while step < args.max_steps:
                model.train()
                for raw in islice(train_loader, args.max_steps - step):
                    batch = prepare_xl_batch(raw, device)
                    output, _, terms = _backward(model, optimizer, batch, empty_peak_weight=args.empty_peak_weight)
                    del output
                    step += 1
                    gradients = _gradient_norms(model) if step in gradient_steps else None
                    optimizer.step()
                    log("train", step=step, group_ids=raw["group_id"], loss_terms=terms,
                        cosine=1.0 - terms["v20_backbone_cosine_loss"], gradient_l2=gradients)
                    if step in probe_steps:
                        log("probe", step=step, **_step_probe(model, probe_batch, fit_loader, val_loader, device))
                    progress.update(1)
        log("complete", step=step)
    print(f"Completed {step} optimizer steps. Diagnostics: {log_path}", flush=True)


def _train(model, optimizer, train_data, args, device):
    val_data = XLSpectrumDataset(args.dataset_dir, "val")
    test_data = XLSpectrumDataset(args.dataset_dir, "test")
    if not len(val_data) or not len(test_data):
        raise ValueError("XL validation and test splits must contain samples")
    train_loader = _loader(train_data, args, device, shuffle=True)
    val_loader = _loader(val_data, args, device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logger = TrainingLogger(args.output_dir)
    best_path = args.output_dir / "best.pt"
    final_epoch_path = best_path
    best = (-float("inf"), -float("inf"))
    batch_count = len(train_loader)
    total_epoch_seconds = 0.0
    total_validation_and_save_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()
        model.train()
        epoch_terms = defaultdict(float)
        completed_epochs = epoch - 1
        post_train_estimate = total_validation_and_save_seconds / max(completed_epochs, 1)
        with tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True,
                  bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}|{postfix}",
                  file=sys.stderr, disable=None) as progress:
            for batch_index, batch in enumerate(progress, start=1):
                batch = prepare_xl_batch(batch, device)
                output, _, terms = _backward(model, optimizer, batch, empty_peak_weight=args.empty_peak_weight)
                optimizer.step()
                for name, value in terms.items():
                    epoch_terms[name] += value * batch["target"].shape[0]
                if not progress.disable:
                    pred_nz, target_nz = xl_density_per_sample(
                        output["prediction"].detach(), batch["target"], batch["valid_mask"],
                    )
                    pred_density, density_gap = torch.stack((pred_nz.mean(), (pred_nz - target_nz).mean())).tolist()
                    seconds_per_batch = (perf_counter() - epoch_start) / batch_index
                    epoch_estimate = (total_epoch_seconds / completed_epochs if completed_epochs
                                      else seconds_per_batch * batch_count)
                    remaining = (seconds_per_batch * (batch_count - batch_index) + post_train_estimate
                                 + epoch_estimate * (args.epochs - epoch))
                    progress.set_postfix_str(
                        f"total ETA {timedelta(seconds=round(remaining))} "
                        f"loss={terms['loss']:.3f} cos={1.0 - terms['v20_backbone_cosine_loss']:.3f} "
                        f"gap={density_gap:+.3f} pred_nz={pred_density:.3f} "
                        f"peak_amp={terms['v20_main_peak_amp_loss']:.3f}",
                        refresh=False,
                    )
                del output
        validation_start = perf_counter()
        validation = evaluate_xl(model, val_loader, device, desc=f"Val {epoch}/{args.epochs}")
        train_losses = {name: total / len(train_data) for name, total in epoch_terms.items()}
        logger.log(json.dumps({"epoch": epoch, "train_loss_terms": train_losses,
                               "validation": validation}, allow_nan=False))
        score = (validation["cosine"], -validation["masked_mae"])
        is_best = score > best
        if is_best:
            _save_checkpoint(_checkpoint(model, optimizer, train_data, args, epoch, validation), best_path)
            best = score
        elif args.empty_peak_weight and epoch == args.epochs:
            # Keep the last ablation epoch available for a comparison at equal training budgets.
            final_epoch_path = args.output_dir / "last.pt"
            _save_checkpoint(_checkpoint(model, optimizer, train_data, args, epoch, validation), final_epoch_path)
        epoch_end = perf_counter()
        epoch_seconds = epoch_end - epoch_start
        total_epoch_seconds += epoch_seconds
        total_validation_and_save_seconds += epoch_end - validation_start
        history = {"epoch": epoch, "is_best": is_best, "epoch_seconds": epoch_seconds,
                   "train_cosine": 1.0 - train_losses["v20_backbone_cosine_loss"]}
        history.update(_history_fields("train", train_losses))
        history.update(_history_fields("val", validation))
        logger.log_history_row(history)
        remaining = (total_epoch_seconds / epoch) * (args.epochs - epoch)
        best_marker = " | best" if is_best else ""
        tqdm.write(f"Epoch {epoch}/{args.epochs} | elapsed {timedelta(seconds=round(epoch_seconds))}"
                   f" | total ETA {timedelta(seconds=round(remaining))}"
                   f" | val cosine {validation['cosine']:.4f}{best_marker}", file=sys.stderr)

    del train_loader, val_loader
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_xl(model, _loader(test_data, args, device), device, desc="Test final")
    summary = {
        "model_type": checkpoint["model_type"],
        "best_epoch": checkpoint["epoch"],
        "best_checkpoint": str(best_path.resolve()),
        "selection": checkpoint["selection"],
        "best_val_metrics": checkpoint["validation"],
        "test_metrics": test_metrics,
        "training_log_path": str(logger.log_path.resolve()),
        "training_history_path": str(logger.history_path.resolve()),
    }
    if args.empty_peak_weight:
        summary["empty_peak_weight"] = args.empty_peak_weight
        summary["final_epoch_checkpoint"] = str(final_epoch_path.resolve())
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    final_message = (f"Best epoch {summary['best_epoch']} | val cosine {summary['best_val_metrics']['cosine']:.4f}"
                     f" | test cosine {test_metrics['cosine']:.4f} | test masked MAE {test_metrics['masked_mae']:.4f}")
    logger.log(final_message, also_print=False)
    tqdm.write(final_message, file=sys.stderr)


def main():
    configure_v20_warning_filters()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs") / datetime.now().strftime("xl_%Y%m%d_%H%M%S"),
        help="new directory for logs and best.pt (default: runs/xl_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--smoke", action="store_true", help="run one real train batch and reload an in-memory checkpoint")
    parser.add_argument("--max-steps", type=int, help="run this many optimizer updates and write step diagnostics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--empty-peak-weight", type=float, default=0.0,
                        help="MAE weight for valid zero-target main and satellite cells (default: 0)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.num_workers < 0:
        parser.error("--batch-size and --epochs must be positive and --num-workers must be nonnegative")
    if args.max_steps is not None and (args.max_steps < 1 or args.smoke):
        parser.error("--max-steps must be positive and cannot be combined with --smoke")
    if not isfinite(args.empty_peak_weight) or args.empty_peak_weight < 0:
        parser.error("--empty-peak-weight must be finite and nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = XLSpectrumDataset(args.dataset_dir, "train")
    if len(dataset) == 0:
        raise ValueError("The XL training split is empty")
    model = XLSpectrumTransformer(**MODEL_CONFIG).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.smoke:
        _smoke(model, optimizer, dataset, args, device)
    elif args.max_steps is not None:
        _train_steps(model, optimizer, dataset, args, device)
    else:
        _train(model, optimizer, dataset, args, device)


if __name__ == "__main__":
    main()
