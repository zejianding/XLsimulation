"""Helpers for interpreting averaged-spectrum targets."""

import math

import torch


def relative_percent_to_normalized_threshold(relative_percent):
    """Convert relative-intensity percent to sqrt-normalized target space."""
    relative_percent = float(relative_percent)
    if relative_percent <= 0.0:
        return 0.0
    return math.sqrt(relative_percent / 100.0)


def build_effective_peak_mask(targets, valid_mask, peak_threshold=0.0):
    """Mask of target cells considered effective peaks under the given threshold."""
    return (targets > float(peak_threshold)).to(targets.dtype) * valid_mask


def build_presence_target(
    targets,
    valid_mask,
    peak_threshold=0.0,
    mode="binary",
    min_relative_pct=0.5,
    mid_relative_pct=2.0,
    strong_relative_pct=10.0,
    min_label=0.25,
    mid_label=0.75,
):
    """Build hard or graded presence targets in sqrt-normalized spectrum space.

    ``channel_soft`` matches the default model path: a position/channel gate is
    derived by max-pooling presence probabilities over m/z bins. It teaches the
    gate which ion type should open at a cleavage, while leaving exact m/z
    placement to the bin head.
    """
    mode = str(mode or "binary").lower()
    if mode in ("binary", "hard"):
        return build_effective_peak_mask(targets, valid_mask, peak_threshold)
    if mode not in ("cell_soft", "channel_soft"):
        raise ValueError(f"Unsupported presence target mode: {mode!r}")

    min_relative_pct = float(min_relative_pct)
    mid_relative_pct = float(mid_relative_pct)
    strong_relative_pct = float(strong_relative_pct)
    if mid_relative_pct <= min_relative_pct:
        raise ValueError("mid_relative_pct must be greater than min_relative_pct")
    if strong_relative_pct <= mid_relative_pct:
        raise ValueError("strong_relative_pct must be greater than mid_relative_pct")

    min_label = float(min_label)
    mid_label = float(mid_label)
    if not (0.0 <= min_label <= mid_label <= 1.0):
        raise ValueError("presence soft labels must satisfy 0 <= min <= mid <= 1")

    relative_pct = targets.clamp_min(0.0).pow(2) * 100.0
    labels = torch.zeros_like(targets)

    weak_band = (relative_pct >= min_relative_pct) & (relative_pct < mid_relative_pct)
    weak_pos = (relative_pct - min_relative_pct) / (mid_relative_pct - min_relative_pct)
    weak_label = min_label + weak_pos.clamp(0.0, 1.0) * (mid_label - min_label)
    labels = torch.where(weak_band, weak_label, labels)

    mid_band = (relative_pct >= mid_relative_pct) & (relative_pct < strong_relative_pct)
    mid_pos = (relative_pct - mid_relative_pct) / (strong_relative_pct - mid_relative_pct)
    strong_label = mid_label + mid_pos.clamp(0.0, 1.0) * (1.0 - mid_label)
    labels = torch.where(mid_band, strong_label, labels)

    labels = torch.where(relative_pct >= strong_relative_pct, torch.ones_like(labels), labels)
    labels = labels * (targets > float(peak_threshold)).to(labels.dtype) * valid_mask

    if mode == "channel_soft":
        labels = labels.max(dim=2, keepdim=True).values.expand_as(labels) * valid_mask
    return labels


def suppress_subthreshold_targets(targets, valid_mask, peak_threshold=0.0):
    """Zero out low-intensity target cells so losses/metrics focus on effective peaks."""
    effective_mask = build_effective_peak_mask(targets, valid_mask, peak_threshold)
    return targets * effective_mask


def build_main_peak_mask(targets, valid_mask, radius=1):
    """Mask of target cells that are local maxima in their 3×3 neighbourhood.

    This excludes isotope satellites and neutral-loss side-peaks from
    strict-P metrics — they are real peaks but not the "main" peak.
    Only the dominant cell in each local peak cluster counts.
    """
    t = targets * valid_mask
    # (B, C, H, W) — max-pool over H×W with kernel 2*radius+1
    t_bc = t.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
    local_max = torch.nn.functional.max_pool2d(
        t_bc, kernel_size=2 * radius + 1, stride=1, padding=radius
    )
    is_main = (t_bc >= local_max) & (t_bc > 0)
    return is_main.permute(0, 2, 3, 1).float() * valid_mask


def build_intensity_soft_presence_target(
    targets,
    valid_mask,
    min_relative_pct=0.5,
    mid_relative_pct=2.0,
    strong_relative_pct=10.0,
    strong_label=1.00,
    mid_label=0.75,
    weak_label=0.40,
):
    """Soft gate labels based on target intensity bands.

    Instead of binary (target>0.03 → 1.0, else 0), this produces per-cell
    soft labels that converge gate values to band-specific targets:

        strong (≥10% rel):  label=1.00 → gate ≈ 0.85
        mid    (2-10%):     label=0.75 → gate ≈ 0.65
        weak   (0.5-2%):    label=0.40 → gate ≈ 0.35
        negative:           label=0.00 → gate ≈ 0.18

    The soft labels create natural gate separation between strong/mid/weak
    peaks without changing the BCE loss structure.  cN and hard_neg still
    operate on the same gate output.
    """
    dtype = targets.dtype
    relative_pct = targets.clamp_min(0.0).pow(2) * 100.0
    valid = valid_mask.to(dtype)

    strong_mask = (relative_pct >= float(strong_relative_pct)).to(dtype)
    mid_mask = (
        (relative_pct >= float(mid_relative_pct))
        * (relative_pct < float(strong_relative_pct))
    ).to(dtype)
    weak_mask = (
        (relative_pct >= float(min_relative_pct))
        * (relative_pct < float(mid_relative_pct))
    ).to(dtype)
    negative_mask = 1.0 - strong_mask - mid_mask - weak_mask

    labels = (
        float(strong_label) * strong_mask
        + float(mid_label) * mid_mask
        + float(weak_label) * weak_mask
        + 0.0 * negative_mask
    )
    return (labels * valid).to(dtype=targets.dtype, device=targets.device)
