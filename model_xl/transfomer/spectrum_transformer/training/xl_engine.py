"""Batch execution and validation for paired XL spectra."""

from contextlib import nullcontext

import torch
from tqdm import tqdm

from spectrum_transformer.config import MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, SATELLITE_TOKEN_IDS
from spectrum_transformer.training.engine import PEAK_RECALL_BANDS, _cosine_per_sample, _masked_mae_per_sample
from spectrum_transformer.training.xl_losses import SHORT_CHAIN_MAX_LENGTH, xl_compound_loss


_TRAINING_FIELDS = (
    "input", "target", "valid_mask", "aux_features", "precursor_charge", "lengths",
    "cleavage_aa_ids", "cleavage_scalar_features", "main_anchor_target", "main_anchor_mask",
)
_TOKEN_FAMILIES = {
    "main": MAIN_TOKEN_IDS,
    "satellite": SATELLITE_TOKEN_IDS,
    "residue_event": RESIDUE_EVENT_TOKEN_IDS,
}
ANCHOR_PROBABILITY_THRESHOLD = 0.5
_DENSITY_THRESHOLD = 0.03
_PRESENCE_PREDICTION_THRESHOLD = 0.05


def prepare_xl_batch(batch, device):
    return {name: batch[name].to(device, non_blocking=True) for name in _TRAINING_FIELDS}


def predict_xl_batch(model, batch):
    return model(
        batch["input"], batch["aux_features"], valid_mask=batch["valid_mask"],
        charge=batch["precursor_charge"], lengths=batch["lengths"],
        cleavage_aa_ids=batch["cleavage_aa_ids"],
        cleavage_scalar_features=batch["cleavage_scalar_features"],
    )


def compute_xl_batch_loss(output, batch):
    return xl_compound_loss(
        output["prediction"], batch["target"], batch["valid_mask"],
        charges=batch["precursor_charge"], lengths=batch["lengths"],
        main_anchor_prob=output["main_anchor_prob"],
        main_anchor_target=batch["main_anchor_target"], main_anchor_mask=batch["main_anchor_mask"],
    )


def xl_density_per_sample(prediction, target, valid_mask):
    """Return prediction and target density above 0.03 for each complete XL pair."""
    valid_cells = valid_mask.flatten(1).sum(1).clamp_min(1)
    pred_nz = ((prediction > _DENSITY_THRESHOLD) & valid_mask).flatten(1).sum(1) / valid_cells
    target_nz = ((target > _DENSITY_THRESHOLD) & valid_mask).flatten(1).sum(1) / valid_cells
    return pred_nz, target_nz


def _intensity_diagnostics(prediction, target, valid):
    pred_nz, target_nz = xl_density_per_sample(prediction, target, valid)
    valid_cells = valid.flatten(1).sum(1).clamp_min(1)
    prediction_energy = (prediction * valid).flatten(1).sum(1) / valid_cells
    target_energy = (target * valid).flatten(1).sum(1) / valid_cells

    predicted_peaks = (prediction > _PRESENCE_PREDICTION_THRESHOLD) & valid
    target_peaks = (target > _DENSITY_THRESHOLD) & valid
    predicted_count = predicted_peaks.flatten(1).sum(1)
    target_count = target_peaks.flatten(1).sum(1)
    hit_count = (predicted_peaks & target_peaks).flatten(1).sum(1)
    metrics = {
        "pred_nonzero_ratio": pred_nz,
        "target_nonzero_ratio": target_nz,
        "density_gap": pred_nz - target_nz,
        "prediction_energy": prediction_energy,
        "target_energy": target_energy,
        "energy_gap": prediction_energy - target_energy,
        "presence_precision": hit_count / predicted_count.clamp_min(1),
        "presence_recall": hit_count / target_count.clamp_min(1),
        "presence_f1": 2 * hit_count / (predicted_count + target_count).clamp_min(1),
    }
    # V20 bands are in percent of the normalization scale before the square root.
    intensity_percent = target.clamp_min(0).square() * 100
    band_counts = []
    for _, lower, upper in PEAK_RECALL_BANDS:
        band = (intensity_percent >= lower) & valid
        if upper is not None:
            band &= intensity_percent < upper
        band_hits = (band & predicted_peaks).flatten(1).sum(1)
        band_counts.append(torch.stack((band_hits, band.flatten(1).sum(1)), dim=1))
    return metrics, torch.stack(band_counts, dim=1)


def _sample_metrics(output, batch):
    prediction, target, valid = output["prediction"], batch["target"], batch["valid_mask"]
    positive = (target > 0) & valid
    anchor_mask = batch["main_anchor_mask"]
    anchor_target = (batch["main_anchor_target"] > 0) & anchor_mask
    anchor_prediction = (output["main_anchor_prob"] >= ANCHOR_PROBABILITY_THRESHOLD) & anchor_mask
    anchor_tp = (anchor_prediction & anchor_target).flatten(1).sum(1)
    metrics = {
        "cosine": _cosine_per_sample(prediction, target, valid),
        "masked_mae": _masked_mae_per_sample(prediction, target, valid),
        "positive_mae": _masked_mae_per_sample(prediction, target, positive),
        "anchor_precision": anchor_tp / anchor_prediction.flatten(1).sum(1).clamp_min(1),
        "anchor_recall": anchor_tp / anchor_target.flatten(1).sum(1).clamp_min(1),
        "zero_positive": ~positive.flatten(1).any(1),
    }
    intensity_metrics, band_counts = _intensity_diagnostics(prediction, target, valid)
    metrics.update(intensity_metrics)
    for name, tokens in _TOKEN_FAMILIES.items():
        family_pred, family_target, family_valid = (x[..., tokens, :] for x in (prediction, target, valid))
        metrics[f"{name}_cosine"] = _cosine_per_sample(family_pred, family_target, family_valid)
        metrics[f"{name}_masked_mae"] = _masked_mae_per_sample(family_pred, family_target, family_valid)
    return metrics, band_counts


def _chain_length_metrics(rows, short, charges, charge_values):
    def summarize(selected):
        if not len(selected):
            return {"chains": 0, "cosine": None, "masked_mae": None}
        cosine, masked_mae = selected.mean(dim=0).tolist()
        return {"chains": len(selected), "cosine": cosine, "masked_mae": masked_mae}

    report = {}
    for name, mask in (("short", short), ("long", ~short)):
        report[name] = summarize(rows[mask])
        report[name]["by_charge"] = {
            str(charge): summarize(rows[mask & (charges == charge)]) for charge in charge_values
        }
    return report


@torch.inference_mode()
def evaluate_xl(model, loader, device, desc=None):
    model.eval()
    charge_batches, metric_batches, chain_batches, short_batches = [], [], [], []
    band_batches = []
    running_cosine = running_mae = 0.0
    evaluated = 0
    iterator = (
        tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True, disable=None)
        if desc is not None else nullcontext(loader)
    )
    with iterator as batches:
        show_progress = desc is not None and not batches.disable
        for batch in batches:
            batch = prepare_xl_batch(batch, device)
            output = predict_xl_batch(model, batch)
            metrics, band_counts = _sample_metrics(output, batch)
            names = tuple(metrics)
            rows = torch.stack(tuple(metrics.values()), dim=1).cpu()
            charge_batches.append(batch["precursor_charge"].cpu())
            metric_batches.append(rows)
            band_batches.append(band_counts.cpu())

            chain_pred = output["prediction"].flatten(0, 1)
            chain_target = batch["target"].flatten(0, 1)
            chain_valid = batch["valid_mask"].flatten(0, 1)
            chain_cosine = _cosine_per_sample(chain_pred, chain_target, chain_valid)
            chain_mae = _masked_mae_per_sample(chain_pred, chain_target, chain_valid)
            chain_batches.append(torch.stack((chain_cosine, chain_mae), dim=1).cpu())
            short_batches.append((batch["lengths"].flatten() <= SHORT_CHAIN_MAX_LENGTH).cpu())

            if show_progress:
                evaluated += len(rows)
                running_cosine += rows[:, names.index("cosine")].sum().item()
                running_mae += rows[:, names.index("masked_mae")].sum().item()
                batches.set_postfix(cosine=f"{running_cosine / evaluated:.4f}",
                                    mae=f"{running_mae / evaluated:.4f}", refresh=False)
    if not metric_batches:
        raise ValueError("Cannot evaluate an empty XL split")

    charges = torch.cat(charge_batches)
    charge_values = charges.unique().tolist()
    rows = torch.cat(metric_batches)
    band_counts = torch.cat(band_batches)

    def summarize(group, bands):
        averages = dict(zip(names, group.mean(dim=0).tolist()))
        del averages["zero_positive"]
        averages["samples"] = len(group)
        averages["zero_positive_samples"] = int(group[:, names.index("zero_positive")].sum())
        for (name, _, _), (hits, peaks) in zip(PEAK_RECALL_BANDS, bands.sum(dim=0).tolist()):
            averages[name] = hits / peaks if peaks else None
        return averages

    report = summarize(rows, band_counts)
    report["by_charge"] = {}
    for charge in charge_values:
        selected = charges == charge
        report["by_charge"][str(charge)] = summarize(rows[selected], band_counts[selected])
    report["by_chain_length"] = _chain_length_metrics(
        torch.cat(chain_batches), torch.cat(short_batches), charges.repeat_interleave(2), charge_values
    )
    return report
