"""V20 native engine helpers."""

from __future__ import annotations

import torch.nn.functional as F
import torch
from tqdm import tqdm

from spectrum_transformer.config import MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, SATELLITE_TOKEN_IDS
from .losses import masked_cosine_loss
from .target_processing import build_effective_peak_mask, build_main_peak_mask, suppress_subthreshold_targets


PEAK_RECALL_BANDS = (
    ("peak_recall_0p5_2", 0.5, 2.0),
    ("peak_recall_2_10", 2.0, 10.0),
    ("peak_recall_10_plus", 10.0, None),
)


def _safe_div(numerator, denominator):
    return numerator / max(denominator, 1.0)


def compute_peak_recall_band_counts(predictions, targets, valid_masks, detection_threshold):
    valid_bool = valid_masks > 0
    predicted_positive = (predictions > detection_threshold) & valid_bool
    relative_pct = targets.clamp_min(0.0).pow(2) * 100.0

    counts = {}
    for name, lower, upper in PEAK_RECALL_BANDS:
        band_mask = valid_bool & (relative_pct >= lower)
        if upper is not None:
            band_mask = band_mask & (relative_pct < upper)
        total = band_mask.sum().item()
        hit = (predicted_positive & band_mask).sum().item()
        counts[name] = {"hit": hit, "total": total}
    return counts


def consolidate_mz_neighbors(predictions, valid_masks, radius=0):
    radius = int(radius or 0)
    if radius <= 0:
        return predictions * valid_masks
    pooled = F.max_pool2d(
        (predictions * valid_masks).permute(0, 3, 1, 2),
        kernel_size=(1, 2 * radius + 1),
        stride=1,
        padding=(0, radius),
    ).permute(0, 2, 3, 1)
    return pooled * valid_masks


def apply_prediction_thresholds(
    predictions,
    valid_masks,
    presence_prob=None,
    intensity_prob=None,
    presence_threshold=None,
    intensity_threshold=None,
    prediction_threshold=None,
    mz_consolidation_radius=0,
):
    filtered = predictions * valid_masks
    if presence_prob is not None and presence_threshold is not None:
        filtered = filtered * (presence_prob >= presence_threshold).to(filtered.dtype)
    if intensity_prob is not None and intensity_threshold is not None:
        filtered = filtered * (intensity_prob >= intensity_threshold).to(filtered.dtype)
    if prediction_threshold is not None:
        filtered = filtered * (filtered >= prediction_threshold).to(filtered.dtype)
    filtered = consolidate_mz_neighbors(filtered, valid_masks, radius=mz_consolidation_radius)
    return filtered * valid_masks


def compute_prediction_metrics(
    predictions,
    targets,
    valid_masks,
    nonzero_threshold=None,
    effective_peak_threshold=0.0,
    consensus_nonzero_threshold=0.02,
):
    batch_size = predictions.size(0)
    predictions = predictions * valid_masks
    targets = suppress_subthreshold_targets(targets, valid_masks, effective_peak_threshold)
    if nonzero_threshold is None:
        nonzero_threshold = max(float(effective_peak_threshold), 0.05)

    abs_error = (predictions - targets).abs() * valid_masks
    total_mae = abs_error.sum().item()
    total_valid_count = valid_masks.sum().item()

    positive_mask = build_effective_peak_mask(targets, valid_masks, 0.0)
    main_peak_mask = build_main_peak_mask(targets, valid_masks)
    total_positive_mae = (abs_error * positive_mask).sum().item()
    total_positive_count = positive_mask.sum().item()
    consensus_positive_mask = build_effective_peak_mask(
        targets, valid_masks, consensus_nonzero_threshold
    )
    total_consensus_count = consensus_positive_mask.sum().item()

    predicted_positive_count = (
        (predictions > nonzero_threshold).float() * valid_masks
    ).sum().item()
    predicted_consensus_count = (
        (predictions > consensus_nonzero_threshold).float() * valid_masks
    ).sum().item()
    predicted_positive = (predictions > nonzero_threshold).float() * valid_masks
    presence_tp = (predicted_positive * positive_mask).sum().item()
    presence_fp = (predicted_positive * (valid_masks - positive_mask)).sum().item()
    presence_fn = ((valid_masks - predicted_positive) * positive_mask).sum().item()
    main_tp = (predicted_positive * main_peak_mask).sum().item()
    main_fp = (predicted_positive * (valid_masks - main_peak_mask)).sum().item()

    pred_vectors = predictions.reshape(batch_size, -1)
    target_vectors = targets.reshape(batch_size, -1)
    cosine = F.cosine_similarity(pred_vectors, target_vectors, dim=1, eps=1e-8).mean().item()

    metrics = {
        "masked_mae": total_mae / max(total_valid_count, 1.0),
        "positive_mae": total_positive_mae / max(total_positive_count, 1.0),
        "cosine": cosine,
        "predicted_nonzero_ratio": predicted_positive_count / max(total_valid_count, 1.0),
        "target_nonzero_ratio": total_positive_count / max(total_valid_count, 1.0),
        "consensus_predicted_nonzero_ratio": predicted_consensus_count / max(total_valid_count, 1.0),
        "consensus_target_nonzero_ratio": total_consensus_count / max(total_valid_count, 1.0),
        "strict_nonzero_threshold": float(nonzero_threshold),
        "consensus_nonzero_threshold": float(consensus_nonzero_threshold),
        "presence_precision": presence_tp / max(presence_tp + presence_fp, 1.0),
        "presence_recall": presence_tp / max(presence_tp + presence_fn, 1.0),
        "presence_precision_main": main_tp / max(main_tp + main_fp, 1.0),
    }
    p = metrics["presence_precision"]
    r = metrics["presence_recall"]
    metrics["presence_f1"] = 2.0 * p * r / max(p + r, 1e-8)
    band_counts = compute_peak_recall_band_counts(
        predictions, targets, valid_masks, nonzero_threshold
    )
    for name, counts in band_counts.items():
        metrics[name] = counts["hit"] / max(counts["total"], 1.0)
    metrics["score"] = metrics["cosine"] - 0.5 * metrics["positive_mae"]
    return metrics


def _masked_mae_per_sample(prediction, target, valid_mask):
    abs_error = (prediction - target).abs() * valid_mask
    denom = valid_mask.reshape(valid_mask.shape[0], -1).sum(dim=1).clamp_min(1.0)
    return abs_error.reshape(abs_error.shape[0], -1).sum(dim=1) / denom


def _cosine_per_sample(prediction, target, valid_mask):
    pred = (prediction * valid_mask).reshape(prediction.shape[0], -1)
    tgt = (target * valid_mask).reshape(target.shape[0], -1)
    return torch.nn.functional.cosine_similarity(pred, tgt, dim=1, eps=1e-8)


def _subset_mean(values, subset_mask):
    subset_mask = subset_mask.bool()
    if subset_mask.sum().item() <= 0:
        return 0.0
    return float(values[subset_mask].mean().detach().cpu())


def _subset_sum_and_count(values, subset_mask):
    subset_mask = subset_mask.bool()
    count = int(subset_mask.sum().item())
    if count <= 0:
        return 0.0, 0
    return float(values[subset_mask].sum().detach().cpu()), count


def _masked_region_mean(values, mask):
    denom = mask.sum().clamp_min(1.0)
    return float(((values * mask).sum() / denom).detach().cpu())


def _family_mask(valid_mask, token_ids):
    mask = torch.zeros_like(valid_mask)
    mask[:, :, token_ids, :] = valid_mask[:, :, token_ids, :]
    return mask


def compute_family_metrics(prediction, target, valid_mask):
    out = {}
    families = {
        "main": MAIN_TOKEN_IDS,
        "satellite": SATELLITE_TOKEN_IDS,
        "residue_event": RESIDUE_EVENT_TOKEN_IDS,
    }
    for name, token_ids in families.items():
        fam_mask = _family_mask(valid_mask, token_ids)
        if fam_mask.sum().item() <= 0:
            out[f"{name}_cosine"] = 0.0
            out[f"{name}_masked_mae"] = 0.0
            out[f"{name}_target_nonzero_ratio"] = 0.0
            out[f"{name}_pred_nonzero_ratio"] = 0.0
            continue
        fam_pred = prediction * fam_mask
        fam_target = target * fam_mask
        fam_cos = 1.0 - masked_cosine_loss(fam_pred, fam_target, fam_mask)
        fam_mae = ((fam_pred - fam_target).abs().sum() / fam_mask.sum().clamp_min(1.0)).detach().cpu().item()
        target_nz = (((fam_target > 0.03).float() * fam_mask).sum() / fam_mask.sum().clamp_min(1.0)).detach().cpu().item()
        pred_nz = (((fam_pred > 0.03).float() * fam_mask).sum() / fam_mask.sum().clamp_min(1.0)).detach().cpu().item()
        out[f"{name}_cosine"] = float(fam_cos.detach().cpu())
        out[f"{name}_masked_mae"] = float(fam_mae)
        out[f"{name}_target_nonzero_ratio"] = float(target_nz)
        out[f"{name}_pred_nonzero_ratio"] = float(pred_nz)
    return out


@torch.no_grad()
def collect_v20_monitor_stats(prediction, target, valid_mask, output=None, charges=None, lengths=None):
    active_mask = (valid_mask > 0).float()
    valid_cells = float(active_mask.sum().detach().cpu())
    if valid_cells <= 0:
        return {
            "masked_mae": 0.0,
            "target_nonzero_ratio": 0.0,
            "pred_nonzero_ratio": 0.0,
            "prediction_energy": 0.0,
            "target_energy": 0.0,
            "z2_fraction": 0.0,
            "z3_fraction": 0.0,
            "short_fraction": 0.0,
            "z2_short_fraction": 0.0,
            "z3_short_fraction": 0.0,
            "z2_long_fraction": 0.0,
            "z3_long_fraction": 0.0,
            "z2_cosine": 0.0,
            "z3_cosine": 0.0,
            "short_cosine": 0.0,
            "z2_short_cosine": 0.0,
            "z3_short_cosine": 0.0,
            "z2_long_cosine": 0.0,
            "z3_long_cosine": 0.0,
            "short_masked_mae": 0.0,
            "z2_masked_mae": 0.0,
            "z3_masked_mae": 0.0,
            "z2_short_masked_mae": 0.0,
            "z3_short_masked_mae": 0.0,
            "z2_long_masked_mae": 0.0,
            "z3_long_masked_mae": 0.0,
            "z2_count": 0,
            "z3_count": 0,
            "short_count": 0,
            "z2_short_count": 0,
            "z3_short_count": 0,
            "z2_long_count": 0,
            "z3_long_count": 0,
            "z2_cosine_sum": 0.0,
            "z3_cosine_sum": 0.0,
            "short_cosine_sum": 0.0,
            "z2_short_cosine_sum": 0.0,
            "z3_short_cosine_sum": 0.0,
            "z2_long_cosine_sum": 0.0,
            "z3_long_cosine_sum": 0.0,
            "short_masked_mae_sum": 0.0,
            "z2_masked_mae_sum": 0.0,
            "z3_masked_mae_sum": 0.0,
            "z2_short_masked_mae_sum": 0.0,
            "z3_short_masked_mae_sum": 0.0,
            "z2_long_masked_mae_sum": 0.0,
            "z3_long_masked_mae_sum": 0.0,
        }

    abs_error = (prediction - target).abs() * active_mask
    target_positive = ((target > 0.03).float() * active_mask).sum()
    prediction_positive = ((prediction > 0.03).float() * active_mask).sum()

    stats = {
        "masked_mae": float(abs_error.sum().detach().cpu()) / valid_cells,
        "target_nonzero_ratio": float(target_positive.detach().cpu()) / valid_cells,
        "pred_nonzero_ratio": float(prediction_positive.detach().cpu()) / valid_cells,
        "prediction_energy": float((prediction * active_mask).sum().detach().cpu()) / valid_cells,
        "target_energy": float((target * active_mask).sum().detach().cpu()) / valid_cells,
        "z2_fraction": 0.0,
        "z3_fraction": 0.0,
        "short_fraction": 0.0,
        "z2_short_fraction": 0.0,
        "z3_short_fraction": 0.0,
        "z2_long_fraction": 0.0,
        "z3_long_fraction": 0.0,
        "z2_cosine": 0.0,
        "z3_cosine": 0.0,
        "short_cosine": 0.0,
        "z2_short_cosine": 0.0,
        "z3_short_cosine": 0.0,
        "z2_long_cosine": 0.0,
        "z3_long_cosine": 0.0,
        "short_masked_mae": 0.0,
        "z2_masked_mae": 0.0,
        "z3_masked_mae": 0.0,
        "z2_short_masked_mae": 0.0,
        "z3_short_masked_mae": 0.0,
        "z2_long_masked_mae": 0.0,
        "z3_long_masked_mae": 0.0,
        "z2_count": 0,
        "z3_count": 0,
        "short_count": 0,
        "z2_short_count": 0,
        "z3_short_count": 0,
        "z2_long_count": 0,
        "z3_long_count": 0,
        "z2_cosine_sum": 0.0,
        "z3_cosine_sum": 0.0,
        "short_cosine_sum": 0.0,
        "z2_short_cosine_sum": 0.0,
        "z3_short_cosine_sum": 0.0,
        "z2_long_cosine_sum": 0.0,
        "z3_long_cosine_sum": 0.0,
        "short_masked_mae_sum": 0.0,
        "z2_masked_mae_sum": 0.0,
        "z3_masked_mae_sum": 0.0,
        "z2_short_masked_mae_sum": 0.0,
        "z3_short_masked_mae_sum": 0.0,
        "z2_long_masked_mae_sum": 0.0,
        "z3_long_masked_mae_sum": 0.0,
    }

    if charges is not None or lengths is not None:
        charges = charges if charges is not None else torch.zeros(prediction.shape[0], device=prediction.device)
        lengths = lengths if lengths is not None else torch.full((prediction.shape[0],), 999, device=prediction.device)
        z2_mask = charges == 2
        z3_mask = charges == 3
        short_mask = lengths <= 16
        z2_short_mask = z2_mask & short_mask
        z3_short_mask = z3_mask & short_mask
        z2_long_mask = z2_mask & (~short_mask)
        z3_long_mask = z3_mask & (~short_mask)
        sample_cosine = _cosine_per_sample(prediction, target, valid_mask)
        sample_mae = _masked_mae_per_sample(prediction, target, valid_mask)
        batch_size = float(prediction.shape[0])
        stats["z2_fraction"] = float(z2_mask.float().mean().detach().cpu()) if batch_size > 0 else 0.0
        stats["z3_fraction"] = float(z3_mask.float().mean().detach().cpu()) if batch_size > 0 else 0.0
        stats["short_fraction"] = float(short_mask.float().mean().detach().cpu()) if batch_size > 0 else 0.0
        stats["z2_short_fraction"] = float(z2_short_mask.float().mean().detach().cpu()) if batch_size > 0 else 0.0
        stats["z3_short_fraction"] = float(z3_short_mask.float().mean().detach().cpu()) if batch_size > 0 else 0.0
        stats["z2_long_fraction"] = float(z2_long_mask.float().mean().detach().cpu()) if batch_size > 0 else 0.0
        stats["z3_long_fraction"] = float(z3_long_mask.float().mean().detach().cpu()) if batch_size > 0 else 0.0
        for name, subset_mask in (
            ("z2", z2_mask),
            ("z3", z3_mask),
            ("short", short_mask),
            ("z2_short", z2_short_mask),
            ("z3_short", z3_short_mask),
            ("z2_long", z2_long_mask),
            ("z3_long", z3_long_mask),
        ):
            cos_sum, count = _subset_sum_and_count(sample_cosine, subset_mask)
            mae_sum, _ = _subset_sum_and_count(sample_mae, subset_mask)
            stats[f"{name}_count"] = count
            stats[f"{name}_cosine_sum"] = cos_sum
            stats[f"{name}_masked_mae_sum"] = mae_sum
            stats[f"{name}_cosine"] = cos_sum / max(count, 1)
            stats[f"{name}_masked_mae"] = mae_sum / max(count, 1)

    if isinstance(output, dict):
        pass

    return stats


@torch.no_grad()
def evaluate_v20(model, loader, device, desc=None):
    model.eval()
    total_cos = 0.0
    total_masked_mae = 0.0
    total_target_nonzero_ratio = 0.0
    total_pred_nonzero_ratio = 0.0
    total_prediction_energy = 0.0
    total_target_energy = 0.0
    total_z2_fraction = 0.0
    total_z3_fraction = 0.0
    total_short_fraction = 0.0
    total_z2_short_fraction = 0.0
    total_z3_short_fraction = 0.0
    total_z2_long_fraction = 0.0
    total_z3_long_fraction = 0.0
    subgroup_totals = {
        "z2": {"cos_sum": 0.0, "mae_sum": 0.0, "count": 0},
        "z3": {"cos_sum": 0.0, "mae_sum": 0.0, "count": 0},
        "short": {"cos_sum": 0.0, "mae_sum": 0.0, "count": 0},
        "z2_short": {"cos_sum": 0.0, "mae_sum": 0.0, "count": 0},
        "z3_short": {"cos_sum": 0.0, "mae_sum": 0.0, "count": 0},
        "z2_long": {"cos_sum": 0.0, "mae_sum": 0.0, "count": 0},
        "z3_long": {"cos_sum": 0.0, "mae_sum": 0.0, "count": 0},
    }
    total_positive_mae = 0.0
    total_presence_precision = 0.0
    total_presence_recall = 0.0
    total_presence_f1 = 0.0
    total_consensus_predicted_nonzero_ratio = 0.0
    total_consensus_target_nonzero_ratio = 0.0
    total_peak_recall_0p5_2 = 0.0
    total_peak_recall_2_10 = 0.0
    total_peak_recall_10_plus = 0.0
    family_totals = {}
    total_batches = 0
    iterator = loader
    pbar = None
    if desc is not None:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        iterator = pbar
    for batch in iterator:
        inputs = batch["input"].to(device)
        target = batch["target"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        active_mask = (valid_mask > 0).float()
        valid_cells = active_mask.sum().clamp_min(1.0)
        aux = batch["aux_features"].to(device)
        cleavage_aa_ids = batch.get("cleavage_aa_ids")
        cleavage_scalar_features = batch.get("cleavage_scalar_features")
        charge = batch.get("charge")
        length = batch.get("length")
        if cleavage_aa_ids is not None:
            cleavage_aa_ids = cleavage_aa_ids.to(device)
        if cleavage_scalar_features is not None:
            cleavage_scalar_features = cleavage_scalar_features.to(device)
        if charge is not None:
            charge = charge.to(device)
        if length is not None:
            length = length.to(device)
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
        loss_cos = masked_cosine_loss(prediction, target, valid_mask)
        total_cos += float(1.0 - loss_cos.detach().cpu())
        pred_metrics = compute_prediction_metrics(
            prediction.detach(),
            target.detach(),
            valid_mask.detach(),
            effective_peak_threshold=0.03,
            consensus_nonzero_threshold=0.02,
        )
        total_positive_mae += float(pred_metrics.get("positive_mae", 0.0))
        total_presence_precision += float(pred_metrics.get("presence_precision", 0.0))
        total_presence_recall += float(pred_metrics.get("presence_recall", 0.0))
        total_presence_f1 += float(pred_metrics.get("presence_f1", 0.0))
        total_consensus_predicted_nonzero_ratio += float(pred_metrics.get("consensus_predicted_nonzero_ratio", 0.0))
        total_consensus_target_nonzero_ratio += float(pred_metrics.get("consensus_target_nonzero_ratio", 0.0))
        total_peak_recall_0p5_2 += float(pred_metrics.get("peak_recall_0p5_2", 0.0))
        total_peak_recall_2_10 += float(pred_metrics.get("peak_recall_2_10", 0.0))
        total_peak_recall_10_plus += float(pred_metrics.get("peak_recall_10_plus", 0.0))
        monitor_stats = collect_v20_monitor_stats(
            prediction,
            target,
            valid_mask,
            output=output,
            charges=charge,
            lengths=length,
        )
        family_metrics = compute_family_metrics(prediction, target, valid_mask)
        total_masked_mae += monitor_stats["masked_mae"]
        total_target_nonzero_ratio += monitor_stats["target_nonzero_ratio"]
        total_pred_nonzero_ratio += monitor_stats["pred_nonzero_ratio"]
        total_prediction_energy += monitor_stats["prediction_energy"]
        total_target_energy += monitor_stats["target_energy"]
        total_z2_fraction += monitor_stats["z2_fraction"]
        total_z3_fraction += monitor_stats["z3_fraction"]
        total_short_fraction += monitor_stats["short_fraction"]
        total_z2_short_fraction += monitor_stats["z2_short_fraction"]
        total_z3_short_fraction += monitor_stats["z3_short_fraction"]
        total_z2_long_fraction += monitor_stats["z2_long_fraction"]
        total_z3_long_fraction += monitor_stats["z3_long_fraction"]
        for name in subgroup_totals:
            subgroup_totals[name]["cos_sum"] += float(monitor_stats.get(f"{name}_cosine_sum", 0.0))
            subgroup_totals[name]["mae_sum"] += float(monitor_stats.get(f"{name}_masked_mae_sum", 0.0))
            subgroup_totals[name]["count"] += int(monitor_stats.get(f"{name}_count", 0))
        for key, value in family_metrics.items():
            family_totals[key] = family_totals.get(key, 0.0) + float(value)
        total_batches += 1
        if pbar is not None:
            z2s_count = subgroup_totals["z2_short"]["count"]
            z3s_count = subgroup_totals["z3_short"]["count"]
            z2s_text = (
                f"{subgroup_totals['z2_short']['cos_sum'] / z2s_count:.4f}"
                if z2s_count > 0
                else "na"
            )
            z3s_text = (
                f"{subgroup_totals['z3_short']['cos_sum'] / z3s_count:.4f}"
                if z3s_count > 0
                else "na"
            )
            pbar.set_postfix(
                cosine=f"{total_cos / max(total_batches, 1):.4f}",
                z2s_cos=z2s_text,
                z3s_cos=z3s_text,
            )
    if pbar is not None:
        pbar.close()
    result = {
        "cosine": total_cos / max(total_batches, 1),
        "masked_mae": total_masked_mae / max(total_batches, 1),
        "positive_mae": total_positive_mae / max(total_batches, 1),
        "target_nonzero_ratio": total_target_nonzero_ratio / max(total_batches, 1),
        "pred_nonzero_ratio": total_pred_nonzero_ratio / max(total_batches, 1),
        "density_gap": (total_pred_nonzero_ratio - total_target_nonzero_ratio) / max(total_batches, 1),
        "consensus_predicted_nonzero_ratio": total_consensus_predicted_nonzero_ratio / max(total_batches, 1),
        "consensus_target_nonzero_ratio": total_consensus_target_nonzero_ratio / max(total_batches, 1),
        "presence_precision": total_presence_precision / max(total_batches, 1),
        "presence_recall": total_presence_recall / max(total_batches, 1),
        "presence_f1": total_presence_f1 / max(total_batches, 1),
        "peak_recall_0p5_2": total_peak_recall_0p5_2 / max(total_batches, 1),
        "peak_recall_2_10": total_peak_recall_2_10 / max(total_batches, 1),
        "peak_recall_10_plus": total_peak_recall_10_plus / max(total_batches, 1),
        "prediction_energy": total_prediction_energy / max(total_batches, 1),
        "target_energy": total_target_energy / max(total_batches, 1),
        "energy_gap": (total_prediction_energy - total_target_energy) / max(total_batches, 1),
        "z2_fraction": total_z2_fraction / max(total_batches, 1),
        "z3_fraction": total_z3_fraction / max(total_batches, 1),
        "short_fraction": total_short_fraction / max(total_batches, 1),
        "z2_short_fraction": total_z2_short_fraction / max(total_batches, 1),
        "z3_short_fraction": total_z3_short_fraction / max(total_batches, 1),
        "z2_long_fraction": total_z2_long_fraction / max(total_batches, 1),
        "z3_long_fraction": total_z3_long_fraction / max(total_batches, 1),
        "z2_cosine": subgroup_totals["z2"]["cos_sum"] / max(subgroup_totals["z2"]["count"], 1),
        "z3_cosine": subgroup_totals["z3"]["cos_sum"] / max(subgroup_totals["z3"]["count"], 1),
        "short_cosine": subgroup_totals["short"]["cos_sum"] / max(subgroup_totals["short"]["count"], 1),
        "z2_short_cosine": subgroup_totals["z2_short"]["cos_sum"] / max(subgroup_totals["z2_short"]["count"], 1),
        "z3_short_cosine": subgroup_totals["z3_short"]["cos_sum"] / max(subgroup_totals["z3_short"]["count"], 1),
        "z2_long_cosine": subgroup_totals["z2_long"]["cos_sum"] / max(subgroup_totals["z2_long"]["count"], 1),
        "z3_long_cosine": subgroup_totals["z3_long"]["cos_sum"] / max(subgroup_totals["z3_long"]["count"], 1),
        "short_masked_mae": subgroup_totals["short"]["mae_sum"] / max(subgroup_totals["short"]["count"], 1),
        "z2_masked_mae": subgroup_totals["z2"]["mae_sum"] / max(subgroup_totals["z2"]["count"], 1),
        "z3_masked_mae": subgroup_totals["z3"]["mae_sum"] / max(subgroup_totals["z3"]["count"], 1),
        "z2_short_masked_mae": subgroup_totals["z2_short"]["mae_sum"] / max(subgroup_totals["z2_short"]["count"], 1),
        "z3_short_masked_mae": subgroup_totals["z3_short"]["mae_sum"] / max(subgroup_totals["z3_short"]["count"], 1),
        "z2_long_masked_mae": subgroup_totals["z2_long"]["mae_sum"] / max(subgroup_totals["z2_long"]["count"], 1),
        "z3_long_masked_mae": subgroup_totals["z3_long"]["mae_sum"] / max(subgroup_totals["z3_long"]["count"], 1),
    }
    for prefix in ("main", "satellite", "residue_event"):
        for suffix in ("cosine", "masked_mae", "target_nonzero_ratio", "pred_nonzero_ratio"):
            key = f"{prefix}_{suffix}"
            result[key] = float(family_totals.get(key, 0.0)) / max(total_batches, 1)
    return result
