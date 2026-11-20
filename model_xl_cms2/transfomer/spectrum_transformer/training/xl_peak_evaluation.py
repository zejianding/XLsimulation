"""Calibrate XL peak selection while preserving predicted peak amplitudes."""

from math import isfinite

import numpy as np
import torch
from tqdm import tqdm

from spectrum_transformer.config import SATELLITE_TOKEN_IDS
from spectrum_transformer.training.xl_engine import predict_xl_batch, prepare_xl_batch


TARGET_THRESHOLD = 0.03
MZ_TOLERANCE = 0.05


def _merge_peaks(mz, intensity):
    """Merge fixed-anchor 0.05 Da windows, retaining the strongest cell's m/z."""
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    merged_mz, merged_intensity = [], []
    start = 0
    while start < len(mz):
        end = int(np.searchsorted(mz, mz[start] + MZ_TOLERANCE, side="right"))
        strongest = start + int(np.argmax(intensity[start:end]))
        merged_mz.append(mz[strongest])
        merged_intensity.append(intensity[strongest])
        start = end
    return np.asarray(merged_mz), np.asarray(merged_intensity)


def _match_peaks(predicted_mz, target_mz):
    """Sorted, one-to-one matching within 0.05 Da, pairing the earliest match."""
    predicted_indices, target_indices = [], []
    predicted = target = 0
    while predicted < len(predicted_mz) and target < len(target_mz):
        difference = predicted_mz[predicted] - target_mz[target]
        if difference < -MZ_TOLERANCE:
            predicted += 1
        elif difference > MZ_TOLERANCE:
            target += 1
        else:
            predicted_indices.append(predicted)
            target_indices.append(target)
            predicted += 1
            target += 1
    return np.asarray(predicted_indices, dtype=np.int64), np.asarray(target_indices, dtype=np.int64)


def _merged_metrics(mz, prediction, target, selected):
    predicted_mz, predicted_intensity = _merge_peaks(mz[selected], prediction[selected])
    reference = target > TARGET_THRESHOLD
    reference_mz, _ = _merge_peaks(mz[reference], target[reference])
    hits, _ = _match_peaks(predicted_mz, reference_mz)
    positive = target > 0
    full_mz, full_intensity = _merge_peaks(mz[positive], target[positive])
    predicted_indices, target_indices = _match_peaks(predicted_mz, full_mz)
    denominator = np.linalg.norm(predicted_intensity) * np.linalg.norm(full_intensity)
    cosine = np.dot(predicted_intensity[predicted_indices], full_intensity[target_indices]) / max(denominator, 1e-8)
    return (len(hits) / max(len(predicted_mz), 1), len(hits) / max(len(reference_mz), 1),
            float(cosine), len(predicted_mz), len(reference_mz))


def _count_above(scores, thresholds):
    return len(scores) - np.searchsorted(np.sort(scores), thresholds, side="right")


def _summarize_at_threshold(rows, threshold):
    precision = recall = density = filtered_cosine = 0.0
    weak_hits = weak_count = satellite_hits = satellite_empty_count = 0
    satellite_true_hits = satellite_true_count = 0
    merged = np.zeros(5, dtype=np.float64)
    merged_samples = 0
    for scores, prediction, target, satellite, charge, mz in rows:
        selected = scores > np.float32(threshold)
        positive = target > TARGET_THRESHOLD
        hits = int((selected & positive).sum())
        precision += hits / max(int(selected.sum()), 1)
        recall += hits / max(int(positive.sum()), 1)
        density += int(selected.sum()) / max(len(scores), 1)
        filtered = np.where(selected, prediction, 0)
        denominator = np.linalg.norm(filtered) * np.linalg.norm(target)
        filtered_cosine += float(np.dot(filtered, target) / max(denominator, 1e-8))
        if charge == 3:
            intensity_percent = target ** 2 * 100
            weak = (intensity_percent >= 0.5) & (intensity_percent < 2.0)
            weak_hits += int((selected & weak).sum())
            weak_count += int(weak.sum())
        satellite_empty = satellite & (target == 0)
        satellite_true = satellite & positive
        satellite_hits += int((selected & satellite_empty).sum())
        satellite_empty_count += int(satellite_empty.sum())
        satellite_true_hits += int((selected & satellite_true).sum())
        satellite_true_count += int(satellite_true.sum())
        if mz is not None:
            merged += _merged_metrics(mz, prediction, target, selected)
            merged_samples += 1
    count = len(rows)
    merged_report = None
    if merged_samples:
        merged /= merged_samples
        merged_report = {
            "samples": merged_samples, "precision": float(merged[0]), "recall": float(merged[1]),
            "cosine": float(merged[2]), "predicted_peaks_mean": float(merged[3]),
            "target_peaks_mean": float(merged[4]), "mz_tolerance_da": MZ_TOLERANCE,
            "merge_rule": "joint A/B fixed-anchor windows <= 0.05 Da, strongest cell m/z and intensity",
            "matching_rule": "sorted one-to-one earliest match within 0.05 Da",
            "intensity_space": "sqrt_clip", "postprocessing": "max merge without PP4",
            "reference": "target > 0.03 for precision/recall; full target > 0 for cosine",
        }
    return {
        "samples": count, "precision": precision / count, "recall": recall / count,
        "density": density / count, "filtered_cosine": filtered_cosine / count,
        "z3_weak_recall": weak_hits / weak_count if weak_count else None,
        "z3_weak_hits": weak_hits, "z3_weak_count": weak_count,
        "satellite_empty_fpr": satellite_hits / satellite_empty_count if satellite_empty_count else None,
        "satellite_true_recall": satellite_true_hits / satellite_true_count if satellite_true_count else None,
        "satellite_empty_hits": satellite_hits, "satellite_empty_count": satellite_empty_count,
        "merged_spectrum": merged_report,
    }


@torch.inference_mode()
def evaluate_xl_peak_selection(
    model, loader, device, *, threshold=None, score_source="intensity",
    recall_floor=0.9195575, recall_ceiling=0.9200575, weak_recall_floor=None, desc=None,
):
    """Calibrate on validation or evaluate an explicitly fixed test threshold.

    Precision, recall and density are averaged per peptide pair. Z3 weak recall
    and satellite zero-label false-positive rates pool their respective cells.
    Presence probabilities select cells; amplitudes always come from prediction.
    Optional ``input_theo_mz`` enables the merged-spectrum diagnostic.
    """
    if score_source not in ("intensity", "presence"):
        raise ValueError("score_source must be 'intensity' or 'presence'")
    if not (0 <= recall_floor <= recall_ceiling <= 1):
        raise ValueError("Recall bounds must satisfy 0 <= floor <= ceiling <= 1")
    if weak_recall_floor is not None and not 0 <= weak_recall_floor <= 1:
        raise ValueError("weak_recall_floor must be in [0, 1]")
    if threshold is not None and (not isfinite(threshold) or threshold < 0 or
                                  (score_source == "presence" and threshold > 1)):
        raise ValueError("Invalid fixed threshold for the chosen score source")

    model.eval()
    rows = []
    batches = tqdm(loader, desc=desc, leave=False, disable=desc is None, dynamic_ncols=True)
    for raw_batch in batches:
        batch = prepare_xl_batch(raw_batch, device)
        output = predict_xl_batch(model, batch)
        prediction = output["prediction"].detach().float().cpu().numpy()
        scores = prediction
        if score_source == "presence":
            scores = output["main_anchor_logits"].detach().float().sigmoid().cpu().numpy()
        target = batch["target"].float().cpu().numpy()
        valid = batch["valid_mask"].bool().cpu().numpy()
        charges = batch["precursor_charge"].cpu().tolist()
        raw_mz = raw_batch.get("input_theo_mz")
        mz = raw_mz.detach().cpu().numpy() if raw_mz is not None else None
        for index, charge in enumerate(charges):
            mask = valid[index]
            satellite = np.zeros_like(mask)
            satellite[..., SATELLITE_TOKEN_IDS, :] = True
            row = (scores[index][mask], prediction[index][mask], target[index][mask],
                   satellite[mask], charge, mz[index][mask] if mz is not None else None)
            if not all(np.isfinite(values).all() for values in row[:3]):
                raise ValueError("Non-finite XL scores, predictions or targets")
            if row[5] is not None and not np.isfinite(row[5]).all():
                raise ValueError("Non-finite valid XL theoretical m/z")
            rows.append(row)
    if not rows:
        raise ValueError("Cannot evaluate an empty XL split")

    mode = "fixed" if threshold is not None else "calibrated"
    reason = None
    if threshold is None:
        last_index = 2000 if score_source == "presence" else 800
        thresholds = np.arange(last_index + 1, dtype=np.float32) / np.float32(2000)
        precision = np.zeros(len(thresholds), dtype=np.float64)
        recall = np.zeros_like(precision)
        weak_hits = np.zeros(len(thresholds), dtype=np.int64)
        weak_count = 0
        for scores, _, target, _, charge, _ in rows:
            positive = target > TARGET_THRESHOLD
            hits = _count_above(scores[positive], thresholds)
            precision += hits / np.maximum(_count_above(scores, thresholds), 1)
            recall += hits / max(int(positive.sum()), 1)
            if charge == 3:
                intensity_percent = target ** 2 * 100
                weak = (intensity_percent >= 0.5) & (intensity_percent < 2.0)
                weak_hits += _count_above(scores[weak], thresholds)
                weak_count += int(weak.sum())
        precision /= len(rows)
        recall /= len(rows)
        acceptable = recall >= recall_floor
        if weak_recall_floor is not None:
            acceptable &= (weak_hits / max(weak_count, 1) >= weak_recall_floor) & (weak_count > 0)
        candidates = np.flatnonzero(acceptable & (recall <= recall_ceiling))
        if not len(candidates):
            candidates = np.flatnonzero(acceptable)
            reason = "no_threshold_in_preferred_recall_band"
        if len(candidates):
            index = int(candidates[np.argmax(precision[candidates])])
        else:
            index = 0
            reason = "no_threshold_meets_recall_constraints"
        threshold = float(thresholds[index])

    report = _summarize_at_threshold(rows, threshold)
    weak_ok = (weak_recall_floor is None or
               (report["z3_weak_recall"] is not None and report["z3_weak_recall"] >= weak_recall_floor))
    report.update({
        "threshold": float(threshold), "score_source": score_source, "mode": mode,
        "recall_floor": recall_floor, "recall_ceiling": recall_ceiling,
        "weak_recall_floor": weak_recall_floor,
        "feasible": bool(report["recall"] >= recall_floor and weak_ok),
        "recall_band_satisfied": bool(recall_floor <= report["recall"] <= recall_ceiling),
        "calibration_note": reason,
    })
    return report
