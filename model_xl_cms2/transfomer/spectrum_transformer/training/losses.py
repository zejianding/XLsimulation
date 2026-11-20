"""V20 native clean-baseline losses.

This is the first real break from the inherited multi-generation loss stack.
The implementation is intentionally compact and mission-focused.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from spectrum_transformer.config import MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, SATELLITE_TOKEN_IDS
from .target_processing import build_main_peak_mask


def masked_l1(prediction, target, valid_mask):
    diff = (prediction - target).abs() * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_cosine_loss(prediction, target, valid_mask):
    pred = prediction * valid_mask
    tgt = target * valid_mask
    pred_flat = pred.reshape(pred.shape[0], -1)
    tgt_flat = tgt.reshape(tgt.shape[0], -1)
    cos = F.cosine_similarity(pred_flat, tgt_flat, dim=1, eps=1e-8)
    return 1.0 - cos.mean()


def weighted_masked_l1(prediction, target, weight_mask):
    diff = (prediction - target).abs() * weight_mask
    denom = weight_mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def weighted_masked_cosine_loss(prediction, target, valid_mask, sample_weight=None):
    pred = (prediction * valid_mask).reshape(prediction.shape[0], -1)
    tgt = (target * valid_mask).reshape(target.shape[0], -1)
    cos = F.cosine_similarity(pred, tgt, dim=1, eps=1e-8)
    per_sample_loss = 1.0 - cos
    if sample_weight is None:
        return per_sample_loss.mean()
    weights = sample_weight.reshape(-1).to(per_sample_loss.dtype)
    active = weights > 0
    if active.sum().item() <= 0:
        return prediction.new_tensor(0.0)
    return (per_sample_loss[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-6)


def build_v20_soft_target(target, valid_mask, threshold=0.03):
    return ((target > threshold).float() * valid_mask).float()


def v20_density_ratio_loss(prediction, target, valid_mask, eps=1e-6):
    pred_sum = (prediction * valid_mask).sum(dim=(1, 2, 3))
    tgt_sum = (target * valid_mask).sum(dim=(1, 2, 3))
    ratio = pred_sum / tgt_sum.clamp(min=eps)
    excess = F.relu(ratio - 1.0)
    return excess.mean()


def v20_mass_cap_loss(prediction, target, valid_mask, mass_cap_ratio=3.0):
    pred_mean = (prediction * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    target_mean = (target * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    return F.relu(pred_mean - target_mean * float(mass_cap_ratio))


def v20_peak_budget_loss(prediction, target, valid_mask, threshold=0.05, temperature=0.01):
    temperature = max(float(temperature), 1e-6)
    threshold = float(threshold)
    pred_soft = torch.sigmoid((prediction - threshold) / temperature) * valid_mask
    target_count = ((target >= threshold).float() * valid_mask).sum(dim=(1, 2, 3))
    pred_count = pred_soft.sum(dim=(1, 2, 3))
    return torch.abs(pred_count - target_count).mean() / valid_mask[0].numel()


def _family_valid_mask(valid_mask, token_ids):
    mask = torch.zeros_like(valid_mask)
    mask[:, :, token_ids, :] = valid_mask[:, :, token_ids, :]
    return mask


def _family_focus_mask(valid_mask):
    token_ids = tuple(SATELLITE_TOKEN_IDS) + tuple(RESIDUE_EVENT_TOKEN_IDS)
    return _family_valid_mask(valid_mask, token_ids)


def _main_family_mask(valid_mask):
    return _family_valid_mask(valid_mask, MAIN_TOKEN_IDS)


def _subgroup_sample_weight(charges, lengths):
    if charges is None or lengths is None:
        return None
    weights = torch.full_like(charges, 0.5, dtype=torch.float32)
    z2 = charges == 2
    z3 = charges == 3
    short = lengths <= 16
    long_mask = ~short
    weights = torch.where(z2 & short, torch.full_like(weights, 1.0), weights)
    weights = torch.where(z2 & long_mask, torch.full_like(weights, 1.3), weights)
    weights = torch.where(z3 & short, torch.full_like(weights, 1.85), weights)
    weights = torch.where(z3 & long_mask, torch.full_like(weights, 1.85), weights)
    return weights


def _subset_sample_weight(charges, lengths, charge_value, short_only=False, long_only=False, weight_value=1.0):
    if charges is None:
        return None
    mask = charges == int(charge_value)
    if lengths is not None and short_only:
        mask = mask & (lengths <= 16)
    if lengths is not None and long_only:
        mask = mask & (lengths > 16)
    weights = torch.zeros_like(charges, dtype=torch.float32)
    weights = torch.where(mask, torch.full_like(weights, float(weight_value)), weights)
    return weights


def _family_weight_map(valid_mask, charges):
    residue_mask = _family_valid_mask(valid_mask, RESIDUE_EVENT_TOKEN_IDS)
    satellite_mask = _family_valid_mask(valid_mask, SATELLITE_TOKEN_IDS)
    main_mask = _family_valid_mask(valid_mask, MAIN_TOKEN_IDS)
    weight = valid_mask.new_zeros(valid_mask.shape)
    weight = weight + 1.0 * residue_mask + 3.0 * satellite_mask
    if charges is None:
        return weight + 8.0 * main_mask
    z2 = (charges == 2).float().view(-1, 1, 1, 1)
    z3 = (charges == 3).float().view(-1, 1, 1, 1)
    weight = weight + (8.0 * z2 + 11.0 * z3 + 8.0 * (1.0 - z2 - z3).clamp_min(0.0)) * main_mask
    return weight


def _expand_local_mask(mask, radius=1):
    radius = int(radius)
    if radius <= 0:
        return mask
    pooled = F.max_pool2d(
        mask.permute(0, 3, 1, 2),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return pooled.permute(0, 2, 3, 1) * (mask > 0).to(mask.dtype).amax(dim=(1, 2, 3), keepdim=True)


def _subgroup_weight(valid_mask, charges=None, lengths=None):
    weight = valid_mask.new_ones((valid_mask.shape[0], 1, 1, 1))
    if charges is not None:
        z2 = (charges == 2).float().view(-1, 1, 1, 1)
        z3 = (charges == 3).float().view(-1, 1, 1, 1)
        z4plus = (charges >= 4).float().view(-1, 1, 1, 1)
        weight = weight + 0.32 * z2
        weight = weight + 0.52 * z3
        weight = weight + 0.02 * z4plus
    if lengths is not None:
        short = (lengths <= 16).float().view(-1, 1, 1, 1)
        weight = weight + 0.04 * short
        if charges is not None:
            z2_short = ((charges == 2) & (lengths <= 16)).float().view(-1, 1, 1, 1)
            z3_short = ((charges == 3) & (lengths <= 16)).float().view(-1, 1, 1, 1)
            weight = weight + 0.08 * z2_short
            weight = weight + 0.12 * z3_short
    return weight


def v20_strict_presence_push_loss(
    prediction,
    target,
    valid_mask,
    threshold=0.03,
    margin=0.012,
    negative_margin=0.010,
    subgroup_weight=None,
    main_peak_only=False,
):
    positive_mask = (((target >= threshold).float()) * valid_mask).float()
    if main_peak_only:
        positive_mask = positive_mask * build_main_peak_mask(target, valid_mask)
    negative_mask = (((target < threshold).float()) * valid_mask).float()

    pos_loss = prediction.new_tensor(0.0)
    if positive_mask.sum().item() > 0:
        pos_gap = F.relu((float(threshold) + float(margin)) - prediction)
        pos_term = pos_gap * positive_mask
        if subgroup_weight is not None:
            pos_term = pos_term * subgroup_weight
            pos_denom = (positive_mask * subgroup_weight).sum().clamp_min(1.0)
        else:
            pos_denom = positive_mask.sum().clamp_min(1.0)
        pos_loss = pos_term.sum() / pos_denom

    neg_loss = prediction.new_tensor(0.0)
    if negative_mask.sum().item() > 0:
        neg_excess = F.relu(prediction - (float(threshold) + float(negative_margin)))
        neg_term = neg_excess * negative_mask
        if subgroup_weight is not None:
            neg_term = neg_term * subgroup_weight
            neg_denom = (negative_mask * subgroup_weight).sum().clamp_min(1.0)
        else:
            neg_denom = negative_mask.sum().clamp_min(1.0)
        neg_loss = neg_term.sum() / neg_denom

    return pos_loss + 0.20 * neg_loss


def v20_main_peak_amplitude_loss(
    prediction,
    target,
    valid_mask,
    threshold=0.03,
    sample_weight=None,
):
    main_peak_mask = build_main_peak_mask(target, valid_mask)
    strict_main_mask = (((target >= float(threshold)).float()) * main_peak_mask * valid_mask).float()
    if sample_weight is not None:
        strict_main_mask = strict_main_mask * sample_weight.view(-1, 1, 1, 1)
    if strict_main_mask.sum().item() <= 0:
        return prediction.new_tensor(0.0)
    return weighted_masked_l1(prediction, target, strict_main_mask)


def v20_main_anchor_bce_loss(
    main_anchor_prob,
    target,
    valid_mask,
    threshold=0.03,
    sample_weight=None,
    negative_weight=0.55,
    positive_boost=3.0,
):
    positive_mask = (((target >= float(threshold)).float()) * build_main_peak_mask(target, valid_mask) * valid_mask).float()
    negative_mask = (((target < float(threshold)).float()) * valid_mask).float()
    labels = positive_mask
    probs = main_anchor_prob.clamp(1e-6, 1.0 - 1e-6)
    loss = F.binary_cross_entropy(probs, labels, reduction="none")
    if sample_weight is not None:
        loss = loss * sample_weight.view(-1, 1, 1, 1)
        denom = (valid_mask * sample_weight.view(-1, 1, 1, 1)).sum().clamp_min(1.0)
    else:
        denom = valid_mask.sum().clamp_min(1.0)
    weight_map = (
        1.0
        + float(positive_boost) * positive_mask
        + float(negative_weight) * negative_mask
    )
    return ((loss * weight_map) * valid_mask).sum() / denom


def _family_budget_targets(target, valid_mask):
    main = _family_valid_mask(valid_mask, MAIN_TOKEN_IDS) * target
    sat = _family_valid_mask(valid_mask, SATELLITE_TOKEN_IDS) * target
    residue = _family_valid_mask(valid_mask, RESIDUE_EVENT_TOKEN_IDS) * target
    main_sum = main.sum(dim=2)
    sat_sum = sat.sum(dim=2)
    residue_sum = residue.sum(dim=2)
    total = (main_sum + sat_sum + residue_sum).clamp_min(1e-8)
    return torch.stack([residue_sum / total, sat_sum / total, main_sum / total], dim=-1)


def v20_compound_loss(
    prediction,
    target,
    valid_mask,
    backbone_pred=None,
    detail_pred=None,
    detail_logits=None,
    main_anchor_prob=None,
    main_anchor_logits=None,
    charges=None,
    lengths=None,
    v20_backbone_weight=1.0,
    v20_strict_presence_weight=0.18,
    v20_strict_threshold=0.03,
    v20_group_family_focus_weight=0.28,
):
    backbone_source = backbone_pred if backbone_pred is not None else prediction
    z2_sample_weight = _subset_sample_weight(charges, lengths, charge_value=2, weight_value=1.0)
    z3_sample_weight = _subset_sample_weight(charges, lengths, charge_value=3, weight_value=1.85)
    z3_short_weight = _subset_sample_weight(charges, lengths, charge_value=3, short_only=True, weight_value=2.4)
    # Give Z2, Z3, and Z>=4 equal weight in the shared main-peak losses.
    amp_sample_weight = (charges >= 2).float() if charges is not None else None
    family_mask = _family_focus_mask(valid_mask)
    main_mask = _main_family_mask(valid_mask)
    detail_source = detail_pred if detail_pred is not None else backbone_source
    family_weight_map = _family_weight_map(valid_mask, charges)

    backbone_global_l1 = masked_l1(detail_source, target, valid_mask)
    backbone_global_cos = masked_cosine_loss(detail_source, target, valid_mask)
    backbone_global = backbone_global_l1 + backbone_global_cos

    z2_recon = prediction.new_tensor(0.0)
    if z2_sample_weight is not None and (z2_sample_weight > 0).sum().item() > 0:
        z2_weight_mask = valid_mask * z2_sample_weight.view(-1, 1, 1, 1)
        z2_recon = weighted_masked_l1(detail_source, target, z2_weight_mask) + weighted_masked_cosine_loss(
            detail_source,
            target,
            valid_mask,
            sample_weight=z2_sample_weight,
        )

    z3_recon = prediction.new_tensor(0.0)
    z3_main_cosine = prediction.new_tensor(0.0)
    if z3_sample_weight is not None and (z3_sample_weight > 0).sum().item() > 0:
        z3_weight_mask = valid_mask * z3_sample_weight.view(-1, 1, 1, 1)
        z3_recon = weighted_masked_l1(detail_source, target, z3_weight_mask) + weighted_masked_cosine_loss(
            detail_source,
            target,
            valid_mask,
            sample_weight=z3_sample_weight,
        )
        z3_main_cosine = weighted_masked_cosine_loss(
            detail_source * main_mask,
            target * main_mask,
            main_mask,
            sample_weight=z3_sample_weight,
        )
    z3_short_recon = prediction.new_tensor(0.0)
    if z3_short_weight is not None and (z3_short_weight > 0).sum().item() > 0:
        z3s_weight_mask = valid_mask * z3_short_weight.view(-1, 1, 1, 1)
        z3_short_recon = weighted_masked_l1(detail_source, target, z3s_weight_mask) + weighted_masked_cosine_loss(
            detail_source,
            target,
            valid_mask,
            sample_weight=z3_short_weight,
        )
    family_weighted_recon = weighted_masked_l1(detail_source, target, family_weight_map)

    anchor_z23 = v20_main_anchor_bce_loss(
        main_anchor_prob if main_anchor_prob is not None else prediction,
        target,
        main_mask,
        threshold=v20_strict_threshold,
        sample_weight=amp_sample_weight,
        negative_weight=0.35,
        positive_boost=4.0,
    )
    z3_main_anchor = v20_main_anchor_bce_loss(
        main_anchor_prob if main_anchor_prob is not None else prediction,
        target,
        main_mask,
        threshold=v20_strict_threshold,
        sample_weight=z3_short_weight if z3_short_weight is not None else z3_sample_weight,
        negative_weight=0.25,
        positive_boost=5.0,
    )
    strict_presence = v20_strict_presence_push_loss(
        prediction,
        target,
        valid_mask,
        threshold=v20_strict_threshold,
        subgroup_weight=amp_sample_weight.view(-1, 1, 1, 1) if amp_sample_weight is not None else None,
        main_peak_only=True,
    )
    main_peak_amp = v20_main_peak_amplitude_loss(
        prediction,
        target,
        valid_mask,
        threshold=v20_strict_threshold,
        sample_weight=amp_sample_weight,
    )
    density_gap_penalty = torch.abs(
        (((prediction >= v20_strict_threshold).float() * valid_mask).sum(dim=(1, 2, 3)) / valid_mask.sum(dim=(1, 2, 3)).clamp_min(1.0))
        - (((target >= v20_strict_threshold).float() * valid_mask).sum(dim=(1, 2, 3)) / valid_mask.sum(dim=(1, 2, 3)).clamp_min(1.0))
    ).mean()
    amp_strict = main_peak_amp + float(v20_strict_presence_weight) * strict_presence
    total = (
        0.20 * backbone_global
        + 0.90 * z2_recon
        + 1.20 * z3_recon
        + 1.30 * z3_main_cosine
        + 1.10 * z3_short_recon
        + 0.80 * main_peak_amp
        + 0.35 * family_weighted_recon
        + 0.45 * anchor_z23
        + 0.65 * z3_main_anchor
        + 0.15 * density_gap_penalty
    )

    return total, {
        "loss": total.detach().item(),
        "v20_backbone_l1": backbone_global_l1.detach().item(),
        "v20_backbone_cosine_loss": backbone_global_cos.detach().item(),
        "v20_backbone_global_loss": backbone_global.detach().item(),
        "v20_z2_recon_loss": z2_recon.detach().item(),
        "v20_z3_recon_loss": z3_recon.detach().item(),
        "v20_main_anchor_z23_loss": anchor_z23.detach().item(),
        "v20_z3_main_anchor_loss": z3_main_anchor.detach().item(),
        "v20_z3_main_cosine_loss": z3_main_cosine.detach().item(),
        "v20_z3_short_recon_loss": z3_short_recon.detach().item(),
        "v20_family_weighted_loss": family_weighted_recon.detach().item(),
        "v20_main_peak_amp_loss": main_peak_amp.detach().item(),
        "v20_amp_strict_loss": amp_strict.detach().item(),
        "v20_strict_presence_loss": strict_presence.detach().item(),
        "v20_density_gap_penalty": density_gap_penalty.detach().item(),
        "v20_group_family_focus_weight": float(v20_group_family_focus_weight),
    }
