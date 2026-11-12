"""Train the six-channel XL Spectrum Transformer or smoke-test one real batch."""

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
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
                 "short_chain_scope": "per_chain"},
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


def _backward(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    output = predict_xl_batch(model, batch)
    loss, terms = compute_xl_batch_loss(output, batch)
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
    output, loss, terms = _backward(model, optimizer, batch)
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
                output, _, terms = _backward(model, optimizer, batch)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.num_workers < 0:
        parser.error("--batch-size and --epochs must be positive and --num-workers must be nonnegative")

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
    else:
        _train(model, optimizer, dataset, args, device)


if __name__ == "__main__":
    main()
