"""V20 loss terms on an XL batch of paired peptide spectra."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spectrum_transformer.config import MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, SATELLITE_TOKEN_IDS

from .losses import masked_cosine_loss, masked_l1, weighted_masked_l1
from .target_processing import build_main_peak_mask


MAIN_PEAK_THRESHOLD = 0.03
SHORT_CHAIN_MAX_LENGTH = 16
EMPTY_PEAK_TOKEN_IDS = MAIN_TOKEN_IDS + SATELLITE_TOKEN_IDS
PEAK_RANKING_TOP_K = 8
PEAK_RANKING_MARGIN = 0.02
XL_LOSS_WEIGHTS = {
    "v20_backbone_global_loss": 0.20,
    "v20_main_peak_amp_loss": 0.80,
    "v20_family_weighted_loss": 0.35,
    "v20_main_anchor_z23_loss": 0.45,
    "v20_density_gap_penalty": 0.15,
}


def _local_peaks(target, valid_mask):
    # Treat the chains independently for local pooling, then restore the XL batch.
    return build_main_peak_mask(target.flatten(0, 1), valid_mask.flatten(0, 1)).reshape_as(target).bool()


def _weighted_anchor_bce(cell_bce, anchor_labels, subthreshold, cell_weight, *, negative_weight, positive_boost):
    peak_weight = 1.0 + positive_boost * anchor_labels + negative_weight * subthreshold
    return (cell_bce * peak_weight * cell_weight).sum() / cell_weight.sum().clamp_min(1.0)


def _peak_ranking_loss(prediction, target, valid_mask):
    group_losses, eligible_groups = [], []
    for token_ids in (MAIN_TOKEN_IDS, SATELLITE_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS):
        scores, labels, valid = (grid[..., token_ids, :].flatten(1) for grid in (prediction, target, valid_mask))
        positives = valid & (labels > MAIN_PEAK_THRESHOLD)
        empty = valid & (labels == 0)
        # Mine hard negatives independently within each peptide pair and family.
        k = min(PEAK_RANKING_TOP_K, scores.shape[1])
        hard_scores = scores.masked_fill(~empty, -torch.inf).topk(k, dim=1).values
        has_negative = hard_scores.isfinite()
        hard_scores = hard_scores.masked_fill(~has_negative, 0.0)
        comparisons = positives[:, :, None] & has_negative[:, None, :]
        violations = F.relu(PEAK_RANKING_MARGIN + hard_scores[:, None, :] - scores[:, :, None])
        count = comparisons.sum(dim=(1, 2))
        group_losses.append(violations.masked_fill(~comparisons, 0.0).sum(dim=(1, 2)) / count.clamp_min(1))
        eligible_groups.append(count > 0)
    return torch.stack(group_losses).sum() / torch.stack(eligible_groups).sum().clamp_min(1)


def xl_compound_loss(
    prediction,
    target,
    valid_mask,
    *,
    charges,
    lengths,
    main_anchor_prob,
    main_anchor_target=None,
    main_anchor_mask=None,
    empty_peak_weight=0.0,
    peak_ranking_weight=0.0,
):
    """Return the scalar loss and its components for grids shaped ``(B, 2, H, T, C)``."""
    if prediction.ndim != 5 or prediction.shape[1] != 2:
        raise ValueError("XL loss expects predictions shaped (batch, 2, cleavage, token, channel)")
    batch_size = prediction.shape[0]
    if charges.shape != (batch_size,) or lengths.shape != (batch_size, 2):
        raise ValueError("XL loss expects one precursor charge and two chain lengths per sample")
    for name, grid in (
        ("target", target),
        ("valid_mask", valid_mask),
        ("main_anchor_prob", main_anchor_prob),
        ("main_anchor_target", main_anchor_target),
        ("main_anchor_mask", main_anchor_mask),
    ):
        if grid is not None and grid.shape != prediction.shape:
            raise ValueError(f"{name} must match the XL prediction shape")

    valid_mask = valid_mask.bool()
    shared = (charges >= 2)[:, None, None, None, None]

    global_l1 = masked_l1(prediction, target, valid_mask)
    global_cosine = masked_cosine_loss(prediction, target, valid_mask)

    main_target = target[..., MAIN_TOKEN_IDS, :]
    main_valid = valid_mask[..., MAIN_TOKEN_IDS, :]

    family_weight = target.new_zeros((batch_size, target.shape[-2]))
    family_weight[:, RESIDUE_EVENT_TOKEN_IDS] = 1.0
    family_weight[:, SATELLITE_TOKEN_IDS] = 3.0
    family_weight[:, MAIN_TOKEN_IDS] = 8.0
    family_weighted_recon = weighted_masked_l1(prediction, target, valid_mask * family_weight[:, None, None, :, None])

    # Amplitude uses local maxima of the full grid. Anchor BCE uses only the main token.
    strong_peaks = _local_peaks(target, valid_mask) & (target >= MAIN_PEAK_THRESHOLD)
    main_peak_amp = weighted_masked_l1(prediction, target, strong_peaks & shared)
    anchor_valid = main_valid if main_anchor_mask is None else main_valid & main_anchor_mask[..., MAIN_TOKEN_IDS, :].bool()
    if main_anchor_target is None:
        anchor_labels = (_local_peaks(main_target, anchor_valid) & (main_target >= MAIN_PEAK_THRESHOLD)).float()
    else:
        anchor_labels = main_anchor_target[..., MAIN_TOKEN_IDS, :].float()
    anchor_prob = main_anchor_prob[..., MAIN_TOKEN_IDS, :].float().clamp(1e-6, 1.0 - 1e-6)
    cell_bce = F.binary_cross_entropy(anchor_prob, anchor_labels, reduction="none")
    subthreshold = main_target < MAIN_PEAK_THRESHOLD
    shared_anchor = _weighted_anchor_bce(
        cell_bce, anchor_labels, subthreshold, anchor_valid & shared, negative_weight=0.35, positive_boost=4.0
    )

    # Preserve the reported hard-threshold density term and its lack of gradient.
    pred_count = ((prediction >= MAIN_PEAK_THRESHOLD) & valid_mask).flatten(1).sum(dim=1)
    target_count = ((target >= MAIN_PEAK_THRESHOLD) & valid_mask).flatten(1).sum(dim=1)
    valid_count = valid_mask.flatten(1).sum(dim=1).clamp_min(1)
    density_gap = ((pred_count - target_count).abs() / valid_count).mean()

    metrics = {
        "v20_backbone_l1": global_l1,
        "v20_backbone_cosine_loss": global_cosine,
        "v20_backbone_global_loss": global_l1 + global_cosine,
        "v20_main_peak_amp_loss": main_peak_amp,
        "v20_family_weighted_loss": family_weighted_recon,
        "v20_main_anchor_z23_loss": shared_anchor,
        "v20_density_gap_penalty": density_gap,
    }
    loss = sum(weight * metrics[name] for name, weight in XL_LOSS_WEIGHTS.items())
    if empty_peak_weight:
        empty_peak_target = target[..., EMPTY_PEAK_TOKEN_IDS, :]
        empty_peak_mask = valid_mask[..., EMPTY_PEAK_TOKEN_IDS, :] & (empty_peak_target == 0)
        empty_peak_mae = masked_l1(prediction[..., EMPTY_PEAK_TOKEN_IDS, :], empty_peak_target, empty_peak_mask)
        metrics["empty_peak_mae"] = empty_peak_mae
        loss = loss + empty_peak_weight * empty_peak_mae
    if peak_ranking_weight:
        peak_ranking = _peak_ranking_loss(prediction, target, valid_mask)
        metrics["peak_ranking_loss"] = peak_ranking
        loss = loss + peak_ranking_weight * peak_ranking
    metrics["loss"] = loss
    return loss, dict(zip(metrics, torch.stack(tuple(metrics.values())).detach().tolist()))
