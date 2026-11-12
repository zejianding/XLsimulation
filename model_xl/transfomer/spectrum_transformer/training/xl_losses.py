"""V20 loss terms on an XL batch of paired peptide spectra."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spectrum_transformer.config import MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, SATELLITE_TOKEN_IDS

from .losses import masked_cosine_loss, masked_l1, weighted_masked_cosine_loss, weighted_masked_l1
from .target_processing import build_main_peak_mask


MAIN_PEAK_THRESHOLD = 0.03
SHORT_CHAIN_MAX_LENGTH = 16
XL_LOSS_WEIGHTS = {
    "v20_backbone_global_loss": 0.20,
    "v20_z2_recon_loss": 0.90,
    "v20_z3_recon_loss": 1.20,
    "v20_z3_main_cosine_loss": 1.30,
    "v20_z3_short_recon_loss": 1.10,
    "v20_main_peak_amp_loss": 0.80,
    "v20_family_weighted_loss": 0.35,
    "v20_main_anchor_z23_loss": 0.45,
    "v20_z3_main_anchor_loss": 0.65,
    "v20_density_gap_penalty": 0.15,
}


def _local_peaks(target, valid_mask):
    # Treat the chains independently for local pooling, then restore the XL batch.
    return build_main_peak_mask(target.flatten(0, 1), valid_mask.flatten(0, 1)).reshape_as(target).bool()


def _weighted_recon(prediction, target, valid_mask, sample_weight):
    cell_weight = valid_mask * sample_weight[:, None, None, None, None]
    return weighted_masked_l1(prediction, target, cell_weight) + weighted_masked_cosine_loss(
        prediction, target, valid_mask, sample_weight=sample_weight
    )


def _weighted_anchor_bce(cell_bce, anchor_labels, subthreshold, cell_weight, *, negative_weight, positive_boost):
    peak_weight = 1.0 + positive_boost * anchor_labels + negative_weight * subthreshold
    return (cell_bce * peak_weight * cell_weight).sum() / cell_weight.sum().clamp_min(1.0)


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
    z2_weight = (charges == 2).float()
    z3 = charges == 3
    z3_weight = 1.85 * z3.float()
    shared = (charges >= 2)[:, None, None, None, None]
    short_chain = (lengths <= SHORT_CHAIN_MAX_LENGTH)[:, :, None, None, None]
    short_valid = valid_mask & short_chain
    z3_short_weight = 2.4 * (z3 & short_valid.flatten(1).any(dim=1)).float()

    global_l1 = masked_l1(prediction, target, valid_mask)
    global_cosine = masked_cosine_loss(prediction, target, valid_mask)
    z2_recon = _weighted_recon(prediction, target, valid_mask, z2_weight)
    z3_recon = _weighted_recon(prediction, target, valid_mask, z3_weight)
    z3_short_recon = _weighted_recon(prediction, target, short_valid, z3_short_weight)

    main_target = target[..., MAIN_TOKEN_IDS, :]
    main_valid = valid_mask[..., MAIN_TOKEN_IDS, :]
    z3_main_cosine = weighted_masked_cosine_loss(
        prediction[..., MAIN_TOKEN_IDS, :], main_target, main_valid, sample_weight=z3_weight
    )

    family_weight = target.new_zeros((batch_size, target.shape[-2]))
    family_weight[:, RESIDUE_EVENT_TOKEN_IDS] = 1.0
    family_weight[:, SATELLITE_TOKEN_IDS] = 3.0
    family_weight[:, MAIN_TOKEN_IDS] = 8.0 + 3.0 * z3[:, None]
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
    z3_short_anchor_weight = (anchor_valid & short_chain) * z3_short_weight[:, None, None, None, None]
    z3_short_anchor = _weighted_anchor_bce(
        cell_bce, anchor_labels, subthreshold, z3_short_anchor_weight, negative_weight=0.25, positive_boost=5.0
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
        "v20_z2_recon_loss": z2_recon,
        "v20_z3_recon_loss": z3_recon,
        "v20_z3_main_cosine_loss": z3_main_cosine,
        "v20_z3_short_recon_loss": z3_short_recon,
        "v20_main_peak_amp_loss": main_peak_amp,
        "v20_family_weighted_loss": family_weighted_recon,
        "v20_main_anchor_z23_loss": shared_anchor,
        "v20_z3_main_anchor_loss": z3_short_anchor,
        "v20_density_gap_penalty": density_gap,
    }
    loss = sum(weight * metrics[name] for name, weight in XL_LOSS_WEIGHTS.items())
    metrics["loss"] = loss
    return loss, dict(zip(metrics, torch.stack(tuple(metrics.values())).detach().tolist()))
