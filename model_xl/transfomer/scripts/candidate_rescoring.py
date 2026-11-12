#!/usr/bin/env python3
"""
V20 Candidate Re-scoring via Spectrum Consistency Validation.

For each scan with K beam-search candidate peptides:
  1. Batch-predict K theoretical spectra using the V20 transformer model
  2. Project experimental peaks onto each candidate's theoretical grid
  3. Score via PiPrime embedding cosine (predicted vs filtered spectrum)
  4. Select the best candidate; optionally fuse with PrimeNovo scores

Modes:
  piprime_cosine  — V20 PiPrime embedding scoring only
  pn_fusion       — V20 PiPrime + PrimeNovo scores weighted fusion

Usage:
  cd v20/transfomer
  python scripts/candidate_rescoring.py \
      --parquet /data/v1/high.mouse.PXD004948.parquet \
      --mgf-dir /data/v1/high.mouse.PXD004948/ \
      --checkpoint runs/v20_evidence_v2_clean/best_model.pt \
      --score-method piprime_cosine \
      --piprime-model /path/to/piprime/model.ckpt \
      --max-scans 500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_TRANSFORMER_DIR = _SCRIPT_DIR.parent
if str(_TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_DIR))

try:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
except Exception:
    pass

if os.name == "nt":
    try:
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass

from predict_mgf import (
    MAX_CLEAVAGE_LEN,
    NUM_CHANNELS,
    NUM_TOKENS,
    build_channel_mask,
    build_input_mz_matrix,
    build_input_mz_validity,
    build_position_mask,
    calculate_pepmass,
    load_checkpoint,
    prediction_to_mgf_peaks,
    prepare_model_input,
)
from spectrum_transformer.data.dataset import encode_cleavage_context
from spectrum_transformer.warnings import configure_v20_warning_filters
from scripts.mgf_row_policy import (
    MgfPolicyStats,
    MgfRowPolicy,
    apply_mgf_row_policy,
    build_mgf_row_catalog,
    build_occurrence_aligned_pairs,
    parse_charge_from_meta as parse_charge_from_meta_policy,
    parse_pepmass_from_meta,
)

configure_v20_warning_filters()

SCORING_METHODS = [
    "piprime_cosine",
    "pn_fusion",
    "rank_fusion",
    "zscore_fusion",
]

BASE_SCORE_METHODS = [
    "piprime_cosine",
]

# ===================================================================
# Section 1: MGF I/O
# ===================================================================

def _index_one_mgf(mgf_path: Path) -> tuple[dict[str, int], int]:
    """Build scan_id → byte_offset index for a single MGF file.

    Stores both the raw SCANS value and the trailing numeric token as
    fallback keys, so both "F1:2478" and "2478" resolve to the same block.

    Returns (index, n_spectra).
    """
    index: dict[str, int] = {}
    n_spectra: int = 0
    duplicates: int = 0

    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as f:
        pos = f.tell()
        line = f.readline()
        while line:
            stripped = line.strip()
            if stripped == "BEGIN IONS":
                n_spectra += 1
                block_offset = pos
                scan_id = ""
                while True:
                    pos = f.tell()
                    inner = f.readline()
                    if not inner:
                        break
                    s = inner.strip()
                    if s == "END IONS" or s == "":
                        break
                    if "=" in s:
                        k, v = s.split("=", 1)
                        k_upper = k.strip().upper()
                        v_stripped = v.strip()
                        if k_upper == "SCANS" and v_stripped:
                            scan_id = v_stripped
                            break
                        elif k_upper == "TITLE" and v_stripped and not scan_id:
                            scan_id = v_stripped
                        continue
                    parts = s.split()
                    if len(parts) == 2:
                        try:
                            float(parts[0])
                            float(parts[1])
                            break
                        except ValueError:
                            pass

                if scan_id:
                    if scan_id in index:
                        duplicates += 1
                    index[scan_id] = block_offset
                    nums = re.findall(r"\d+", scan_id)
                    if nums:
                        fallback = nums[-1]
                        if fallback not in index:
                            index[fallback] = block_offset

                while inner and inner.strip() != "END IONS":
                    pos = f.tell()
                    inner = f.readline()
                pos = f.tell()
                line = f.readline()
            else:
                pos = f.tell()
                line = f.readline()

    return index, n_spectra


def build_mgf_dir_index(mgf_dir: Path) -> dict[str, tuple[Path, int]]:
    """Build scan_id → (mgf_path, byte_offset) index across all .mgf files."""
    mgf_files = sorted(mgf_dir.glob("*.mgf"))
    if not mgf_files:
        raise FileNotFoundError(f"No .mgf files found in {mgf_dir}")

    combined: dict[str, tuple[Path, int]] = {}
    total_spectra = 0
    total_duplicates = 0

    for mgf_path in mgf_files:
        file_index, n_spec = _index_one_mgf(mgf_path)
        dups_in_file = 0
        for sid, offset in file_index.items():
            if sid in combined:
                dups_in_file += 1
            combined[sid] = (mgf_path, offset)
        total_spectra += n_spec
        total_duplicates += dups_in_file

    print(f"  [mgf index] {len(mgf_files)} .mgf files, {total_spectra} spectra, "
          f"index keys={len(combined)} (duplicates={total_duplicates})")
    return combined


def load_spectrum_by_offset(mgf_path: Path, byte_offset: int) -> tuple[dict, np.ndarray]:
    """Load a single spectrum by byte offset. Returns (meta, peaks)."""
    meta: dict[str, str] = {}
    peak_list: list[tuple[float, float]] = []

    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(byte_offset)
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s == "END IONS":
                break
            if s == "BEGIN IONS":
                continue
            if "=" in s:
                k, v = s.split("=", 1)
                meta[k.upper()] = v.strip()
                continue
            parts = s.split()
            if len(parts) == 2:
                try:
                    peak_list.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    peaks = np.array(peak_list, dtype=np.float32) if peak_list else np.empty((0, 2), dtype=np.float32)
    if peaks.size > 0:
        peaks = peaks[peaks[:, 0].argsort()]
    return meta, peaks


def parse_charge_from_meta(meta: dict) -> int:
    return parse_charge_from_meta_policy(meta)


def iter_mgf_spectra_in_order(
    mgf_dir: Path,
    row_policy: MgfRowPolicy | None = None,
    stats: MgfPolicyStats | None = None,
    include_ordinal: bool = False,
    load_peaks: bool = True,
):
    """Yield (meta, peaks) in the same order as the original row-order loader."""
    mgf_files = sorted(mgf_dir.glob("*.mgf"))
    ordinal = 0
    for mgf_path in mgf_files:
        with open(mgf_path, "r", encoding="utf-8", errors="ignore") as f:
            block: list[str] = []
            in_block = False
            for line in f:
                s = line.strip()
                if s == "BEGIN IONS":
                    in_block = True
                    block = [line]
                    continue
                if not in_block:
                    continue
                block.append(line)
                if s != "END IONS":
                    continue

                meta: dict[str, str] = {}
                plist: list[tuple[float, float]] = []
                for bl in block:
                    st = bl.strip()
                    if not st:
                        continue
                    if "=" in st and not st.upper().startswith(("BEGIN IONS", "END IONS")):
                        k, v = st.split("=", 1)
                        meta[k.upper()] = v.strip()
                        continue
                    parts = st.split()
                    if load_peaks and len(parts) == 2:
                        try:
                            plist.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
                peaks = np.array(plist, dtype=np.float32) if plist else np.empty((0, 2), dtype=np.float32)
                if peaks.size > 0:
                    peaks = peaks[peaks[:, 0].argsort()]
                keep, fixed_meta, diag = apply_mgf_row_policy(meta, row_policy)
                if stats is not None:
                    stats.total_spectra += 1
                    if keep:
                        stats.kept_spectra += 1
                        if diag["precursor_mismatch_corrected"]:
                            stats.fixed_precursor_mismatch += 1
                    else:
                        if diag["row_drop_reason"] == "seq_contains_O":
                            stats.dropped_seq_with_o += 1
                        elif diag["row_drop_reason"] == "precursor_mismatch":
                            stats.dropped_precursor_mismatch += 1
                if keep:
                    if include_ordinal:
                        yield ordinal, fixed_meta, peaks, diag
                    else:
                        yield fixed_meta, peaks
                ordinal += 1
                block = []
                in_block = False


# ===================================================================
# Section 2: Grid projection
# ===================================================================

def project_experimental_to_grid(
    exp_peaks: np.ndarray,
    input_mz: np.ndarray,
    valid_mask: np.ndarray,
    mz_tol: float = 0.05,
) -> np.ndarray:
    """Project experimental peaks onto a candidate's theoretical (39,46,6) grid."""
    grid = np.zeros((MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    if exp_peaks.size == 0:
        return grid

    valid_rc = np.argwhere(valid_mask > 0)
    if valid_rc.size == 0:
        return grid
    slot_mz = np.array([input_mz[p, t, c] for p, t, c in valid_rc], dtype=np.float64)
    sort_idx = np.argsort(slot_mz)
    valid_rc = valid_rc[sort_idx]
    slot_mz = slot_mz[sort_idx]

    for mz, intensity in exp_peaks:
        idx = int(np.searchsorted(slot_mz, mz))
        best_dist = float("inf")
        best_i = -1
        for ni in (idx - 1, idx, idx + 1):
            if 0 <= ni < len(slot_mz):
                d = abs(float(slot_mz[ni]) - mz)
                if d <= mz_tol and d < best_dist:
                    best_dist = d
                    best_i = ni
        if best_i < 0:
            continue
        p, t, c = valid_rc[best_i]
        if float(intensity) > grid[p, t, c]:
            grid[p, t, c] = float(intensity)

    return grid


def compute_mass_ppm_deviation(seq: str, exp_pepmass: float, exp_charge: int) -> float:
    calc_pepmass = calculate_pepmass(seq, exp_charge)
    if exp_pepmass <= 0:
        return 0.0
    return abs(calc_pepmass - exp_pepmass) / exp_pepmass * 1e6


def compute_mass_da_deviation(seq: str, exp_pepmass: float, exp_charge: int) -> float:
    """Absolute neutral-mass deviation (Da), matching PrimeNovo PMC's tolerance.

    ``calculate_pepmass`` returns m/z; multiplying the m/z gap by the charge
    recovers the neutral-mass gap. PrimeNovo's PMC knapsack decode uses a
    0.1 Da tolerance on the neutral peptide mass, so this is the equivalent
    scale for a Da-based hard filter.
    """
    calc_pepmass = calculate_pepmass(seq, exp_charge)
    if exp_pepmass <= 0 or exp_charge <= 0:
        return 0.0
    return abs(calc_pepmass - exp_pepmass) * exp_charge


# ===================================================================
# Section 3: Scoring
# ===================================================================

def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _grid_to_peak_arrays(
    grid: np.ndarray,
    input_mz: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (39,46,6) grid to (mz_array, intensity_array) for PiPrime."""
    valid = (valid_mask > 0) & (grid > 0)
    if not np.any(valid):
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    mz = input_mz[valid].astype(np.float32)
    intensity = grid[valid].astype(np.float32)
    order = np.argsort(intensity)[::-1]
    return mz[order], intensity[order]


# ---- PiPrime embedding (lazy-loaded) ----

_piprime_model = None
_piprime_config = None
_piprime_device = None


def _load_piprime(piprime_checkpoint: str, target_device: object = None):
    global _piprime_model, _piprime_config, _piprime_device
    if _piprime_model is not None:
        return

    # 切换默认 CUDA 设备，确保 PiPrime 内部所有 tensor 都在目标 GPU 上创建
    if target_device is not None and str(target_device).startswith("cuda"):
        import torch as _torch
        _device_idx = int(str(target_device).split(":")[-1])
        _torch.cuda.set_device(_device_idx)

    _PIPRIME_DIR = os.environ.get(
        "V20_PIPRIME_DIR",
        str(Path(__file__).resolve().parents[3] / "piprime_embedding" / "piprime_embedding"),
    )
    if str(_PIPRIME_DIR) not in sys.path:
        sys.path.insert(0, str(_PIPRIME_DIR))

    import run_iterative_correction as ric

    _piprime_model, _piprime_config, _piprime_device = ric.load_model(piprime_checkpoint)
    if target_device is not None:
        target_str = str(target_device)
        if str(_piprime_device) != target_str:
            _piprime_model = _piprime_model.to(target_device)
            _piprime_device = target_device
    _piprime_model.eval()


def _extract_piprime_embedding(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
) -> np.ndarray:
    import run_iterative_correction as ric
    return ric.extract_embedding(
        mz_array, intensity_array,
        float(precursor_mz), int(precursor_charge),
        _piprime_model, _piprime_config, _piprime_device,
    )


def score_piprime_cosine(
    pred_grid: np.ndarray,
    filt_grid: np.ndarray,
    valid_mask: np.ndarray,
    input_mz: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
) -> float:
    """PiPrime embedding cosine between predicted and filtered spectrum."""
    pred_mz, pred_int = _grid_to_peak_arrays(pred_grid, input_mz, valid_mask)
    filt_mz, filt_int = _grid_to_peak_arrays(filt_grid, input_mz, valid_mask)

    if pred_mz.size == 0 or filt_mz.size == 0:
        return 0.0

    emb_pred = _extract_piprime_embedding(pred_mz, pred_int, precursor_mz, precursor_charge)
    emb_filt = _extract_piprime_embedding(filt_mz, filt_int, precursor_mz, precursor_charge)

    return _safe_cosine(emb_pred.astype(np.float32), emb_filt.astype(np.float32))


def _score_embedding_cosine_from_peaks(
    pred_peaks: np.ndarray,
    filt_peaks: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
) -> float:
    if pred_peaks.size == 0 or filt_peaks.size == 0:
        return 0.0
    emb_pred = _extract_piprime_embedding(
        pred_peaks[:, 0].astype(np.float32, copy=False),
        pred_peaks[:, 1].astype(np.float32, copy=False),
        precursor_mz,
        precursor_charge,
    )
    emb_filt = _extract_piprime_embedding(
        filt_peaks[:, 0].astype(np.float32, copy=False),
        filt_peaks[:, 1].astype(np.float32, copy=False),
        precursor_mz,
        precursor_charge,
    )
    return _safe_cosine(emb_pred.astype(np.float32), emb_filt.astype(np.float32))


def score_postprocessed_piprime_cosine(
    pred_grid: np.ndarray,
    filt_grid: np.ndarray,
    valid_mask: np.ndarray,
    input_mz: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
    min_mz: float,
    min_intensity: float = 0.005,
    mz_merge_tol: float = 0.05,
) -> float:
    """Export-style PiPrime cosine after lightweight peak filtering/merging."""
    pred_peaks = prediction_to_mgf_peaks(
        input_mz,
        pred_grid,
        valid_mask,
        min_intensity=float(min_intensity),
        min_mz=float(min_mz),
        mz_merge_tol=float(mz_merge_tol),
    )
    filt_peaks = prediction_to_mgf_peaks(
        input_mz,
        filt_grid,
        valid_mask,
        min_intensity=0.0,
        min_mz=float(min_mz),
        mz_merge_tol=float(mz_merge_tol),
    )
    return _score_embedding_cosine_from_peaks(
        pred_peaks,
        filt_peaks,
        precursor_mz=float(precursor_mz),
        precursor_charge=int(precursor_charge),
    )




















































def _score_to_scalar(v):
    """Coerce one candidate score to a scalar float, tolerating nested parquet
    encodings (e.g. list<list<float>> → its single value)."""
    if isinstance(v, (list, tuple)):
        a = np.asarray(v, dtype=np.float64).ravel()
        return float(a[0]) if a.size else float("nan")
    if isinstance(v, np.ndarray):
        a = v.reshape(-1)
        return float(a[0]) if a.size else float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _safe_score_array(values: list[float]) -> np.ndarray:
    values = [_score_to_scalar(v) for v in values]
    arr = np.array(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr)
    floor = np.min(arr[finite_mask]) - 1.0
    return np.where(finite_mask, arr, floor)


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)


def _zscore_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mean = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return np.ones_like(arr) * 0.5
    order = np.argsort(-arr, kind="mergesort")
    norm = np.zeros_like(arr, dtype=np.float64)
    denom = max(arr.size - 1, 1)
    for rank_idx, arr_idx in enumerate(order):
        norm[arr_idx] = 1.0 - (rank_idx / denom)
    return norm


def _compute_desc_ranks(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(-arr, kind="mergesort")
    ranks = np.zeros_like(order, dtype=np.int64)
    for rank_idx, arr_idx in enumerate(order, start=1):
        ranks[arr_idx] = rank_idx
    return ranks


def _annotate_candidate_context(results: list[dict]) -> None:
    if not results:
        return

    pn_scores = _safe_score_array([r["original_score"] for r in results])
    piprime_scores = _safe_score_array([r["piprime_score"] for r in results])
    model_scores = _safe_score_array([r["v20_score_model"] for r in results])

    pn_ranks = _compute_desc_ranks(pn_scores)
    piprime_ranks = _compute_desc_ranks(piprime_scores)
    model_ranks = _compute_desc_ranks(model_scores)

    pn_best = float(pn_scores.max()) if pn_scores.size else 0.0
    piprime_best = float(piprime_scores.max()) if piprime_scores.size else 0.0
    model_best = float(model_scores.max()) if model_scores.size else 0.0

    for idx, r in enumerate(results):
        r["pn_rank"] = int(pn_ranks[idx])
        r["piprime_rank"] = int(piprime_ranks[idx])
        r["v20_model_rank"] = int(model_ranks[idx])
        r["pn_gap_to_best"] = round(float(pn_best - pn_scores[idx]), 6)
        r["piprime_gap_to_best"] = round(float(piprime_best - piprime_scores[idx]), 6)
        r["v20_model_gap_to_best"] = round(float(model_best - model_scores[idx]), 6)


def _apply_fusion_raw(results: list[dict], alpha: float) -> None:
    """No-norm PN fusion: alpha*v20_model + (1-alpha)*clip(pn).

    No per-spectrum min-max normalization, preserving score magnitude for
    cross-spectrum FDR calibration. PN scores clipped to [0,1] to match
    v20_model range.
    """
    for r in results:
        if not r["passes_hard_filter"]:
            r["v20_score"] = r["v20_score_model"]
            continue
        v20 = r["v20_score_model"]
        pn = max(0.0, min(1.0, float(r["original_score"])))
        r["v20_score"] = round(float(alpha * v20 + (1 - alpha) * pn), 6)


def _apply_fusion(results: list[dict], mode: str, alpha: float) -> None:
    if not results:
        return

    v20_raw = _safe_score_array([r["v20_score_model"] for r in results])
    pn_raw = _safe_score_array([r["original_score"] for r in results])

    if mode == "pn_fusion":
        v20_norm = _minmax_normalize(v20_raw)
        pn_norm = _minmax_normalize(pn_raw)
    elif mode == "rank_fusion":
        v20_norm = _rank_normalize(v20_raw)
        pn_norm = _rank_normalize(pn_raw)
    elif mode == "zscore_fusion":
        v20_norm = _zscore_normalize(v20_raw)
        pn_norm = _zscore_normalize(pn_raw)
    else:
        raise ValueError(f"Unsupported fusion mode: {mode}")

    for idx, r in enumerate(results):
        if not r["passes_hard_filter"]:
            continue
        r["v20_score_raw"] = r["v20_score_model"]
        r["v20_score"] = round(float(alpha * v20_norm[idx] + (1 - alpha) * pn_norm[idx]), 6)





















def batch_predict_candidates(
    model: torch.nn.Module,
    model_config: dict,
    candidates: list[str],
    precursor_charge: int,
    pepmass: float,
    rt: float,
    stats: dict,
    device: torch.device,
    min_mz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch predict spectra for K candidate peptides.

    Returns (pred_grids, input_grids, valid_masks), each (K, 39, 46, 6).
    """
    K = len(candidates)
    expected_in_channels = int(model_config.get("in_channels", 6))
    use_seq_context = bool(model_config.get("use_sequence_context", True))

    input_grids = np.zeros((K, MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    valid_masks = np.zeros((K, MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    lengths = np.zeros(K, dtype=np.int64)
    aa_ids_batch = np.zeros((K, MAX_CLEAVAGE_LEN, 2), dtype=np.int64)
    scalar_batch = np.zeros((K, MAX_CLEAVAGE_LEN, 7), dtype=np.float32)

    for i, seq in enumerate(candidates):
        input_grids[i] = build_input_mz_matrix(seq, min_mz=min_mz)
        pos_mask, cc = build_position_mask(seq)
        ch_mask = build_channel_mask(precursor_charge)
        mz_mask = build_input_mz_validity(seq, min_mz=min_mz)
        vm = pos_mask[:, None, None] * ch_mask[None, None, :]
        vm = np.broadcast_to(vm, (MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS)).copy()
        valid_masks[i] = (vm * mz_mask).astype(np.float32)
        lengths[i] = cc + 1

        if use_seq_context:
            aa_ids, scalar = encode_cleavage_context(
                seq, cc + 1, max_cleavage_len=MAX_CLEAVAGE_LEN
            )
            aa_ids_batch[i] = aa_ids.astype(np.int64)
            scalar_batch[i] = scalar.astype(np.float32)

    norm_inputs = np.zeros_like(input_grids)
    for i in range(K):
        norm_inputs[i] = prepare_model_input(
            input_grids[i], valid_masks[i], stats, expected_in_channels
        )[0]

    raw_aux = np.array([[float(precursor_charge), pepmass, rt]], dtype=np.float32)
    aux_norm = (
        (raw_aux - stats["aux_feature_means"])
        / np.maximum(stats["aux_feature_stds"], 1e-6)
    ).astype(np.float32)
    aux_batch = np.repeat(aux_norm, K, axis=0)

    forward_kwargs: dict = {
        "aux_features": torch.from_numpy(aux_batch).to(device),
        "valid_mask": torch.from_numpy(valid_masks).to(device),
        "charge": torch.from_numpy(np.full(K, int(precursor_charge), dtype=np.int64)).to(device),
        "length": torch.from_numpy(lengths.astype(np.int64)).to(device),
    }
    if use_seq_context:
        forward_kwargs["cleavage_aa_ids"] = torch.from_numpy(aa_ids_batch).to(device)
        forward_kwargs["cleavage_scalar_features"] = torch.from_numpy(scalar_batch).to(device)

    with torch.inference_mode():
        out = model(torch.from_numpy(norm_inputs).to(device), **forward_kwargs)
    pred_grids = out["prediction"].cpu().numpy().astype(np.float32)

    return pred_grids, input_grids, valid_masks


def _valid_mask_for_seq(seq: str, precursor_charge: int, min_mz: float) -> np.ndarray:
    pos_mask, _ = build_position_mask(seq)
    ch_mask = build_channel_mask(precursor_charge)
    mz_mask = build_input_mz_validity(seq, min_mz=min_mz)
    vm = pos_mask[:, None, None] * ch_mask[None, None, :]
    vm = np.broadcast_to(vm, (MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS)).copy()
    return (vm * mz_mask).astype(np.float32)


# ===================================================================
# Section 5: Main rescoring logic
# ===================================================================

# ---- Amino-acid mass matching (exact table & logic from piprime_metrics.py) ----

# ---- Mass table & tokenizer from piprime_metrics.py ----

_AA_MASSES = {
    "G": 57.021463735, "A": 71.037113805, "S": 87.032028435,
    "P": 97.052763875, "V": 99.068413945, "T": 101.047678505,
    "C+57.021": 103.009184505 + 57.02146,
    "L": 113.084064015, "I": 113.084064015, "N": 114.042927470,
    "D": 115.026943065, "Q": 128.058577540, "K": 128.094963050,
    "E": 129.042593135, "M": 131.040484645, "H": 137.058911875,
    "F": 147.068413945, "R": 156.101111050, "Y": 163.063328575,
    "W": 186.079312980,
    "+42.011": 42.010565, "+43.006": 43.005814, "-17.027": -17.026549,
    "+43.006-17.027": 43.006814 - 17.026549,
    "M+15.995": 131.040484645 + 15.994915,
    "N+0.984": 114.042927470 + 0.984016, "Q+0.984": 128.058577540 + 0.984016,
}

# Precision variant → canonical token map.
# E.g. "C+57.02" → "C+57.021", "+43.01" → "+43.006", "-17.03" → "-17.027"
# Uses rounding (not truncation) to match MGF's 2dp representation.
_PRECISION_MAP: dict[str, str] = {}
for _tok, _mass in _AA_MASSES.items():
    # Extract AA prefix and numeric modifier, round modifier to 2dp
    _m = re.match(r"^([A-Z]?)([+-]?\d+\.\d{3,})$", _tok)
    if not _m: continue
    _prefix = _m.group(1)
    _mod_str = _m.group(2)
    _mod_val = float(_mod_str)
    _rounded = round(_mod_val, 2)
    # Build 2dp variant string
    _sign = "-" if _rounded < 0 else "+"
    _abs_val = abs(_rounded)
    _int_part = int(_abs_val)
    _frac_part = int(round((_abs_val - _int_part) * 100))
    _variant = f"{_prefix}{_sign}{_int_part}.{_frac_part:02d}"
    if _variant != _tok and _variant not in _AA_MASSES:
        _PRECISION_MAP[_variant] = _tok

# Build TOKEN_OPTIONS: mass-equivalent token groups (compute_candidate_quality_hit_rates.py)
_TOKEN_OPTIONS: dict[str, tuple[str, ...]] = {}
for _tok, _mass in _AA_MASSES.items():
    _group = tuple(sorted(
        (other for other, other_mass in _AA_MASSES.items() if abs(other_mass - _mass) < 0.1),
        key=lambda item: (-len(item), item),
    ))
    _TOKEN_OPTIONS[_tok] = _group

_TOKEN_SPLIT_RE = re.compile(r"(?<=.)(?=[A-Z])")
_PROFORMA_BRACKET_RE = re.compile(r"([A-Z])(\[[^\]]+\])")
_NTERM_BRACKET_RE = re.compile(r"^(\[[^\]]+\])")


def _tokenize_peptide(peptide: str) -> list[str]:
    if not peptide: return []
    p = str(peptide).strip().replace(" ", "").replace("$", "")
    # Convert round-bracket mods to inline: (+43.01)→+43.01, (-17.03)→-17.03, (+.98)→+.98
    p = re.sub(r"\(([+-]?\d*\.?\d+)\)", r"\1", p)
    # Convert remaining square-bracket mods: [+43.01] → +43.01
    p = _PROFORMA_BRACKET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)[1:-1]}", p)
    p = _NTERM_BRACKET_RE.sub(lambda m: m.group(1)[1:-1], p)
    tokens = [tok for tok in _TOKEN_SPLIT_RE.split(p) if tok]
    normalized = []
    for tok in tokens:
        # Normalize various token formats to standard "AA+mass" form:
        # "C57.02" → "C+57.02", "C.98" → "C+0.98", "57.02" → "+57.02",
        # ".98" → "+0.98", "+.98" → "+0.98"
        _m = re.match(r"^([A-Z])(\d+\.?\d*)$", tok)
        if _m:
            tok = f"{_m.group(1)}+{_m.group(2)}"
        elif re.match(r"^([A-Z])\.(\d+)$", tok):
            _dm = re.match(r"^([A-Z])\.(\d+)$", tok)
            tok = _dm.group(1) + "+0." + _dm.group(2)
        elif re.match(r"^([A-Z])\+\.(\d+)$", tok):
            _dm = re.match(r"^([A-Z])\+\.(\d+)$", tok)
            tok = _dm.group(1) + "+0." + _dm.group(2)
        elif re.match(r"^\d+\.?\d*$", tok):
            tok = f"+{tok}"
        elif re.match(r"^\.(\d+)$", tok):
            _dm = re.match(r"^\.(\d+)$", tok)
            tok = "+0." + _dm.group(1)
        elif re.match(r"^\+\.(\d+)$", tok):
            _dm = re.match(r"^\+\.(\d+)$", tok)
            tok = "+0." + _dm.group(1)
        elif re.match(r"^\-\.(\d+)$", tok):
            _dm = re.match(r"^\-\.(\d+)$", tok)
            tok = "-0." + _dm.group(1)
        # Precision map (single mod): "C+57.02" → "C+57.021"
        tok = _PRECISION_MAP.get(tok, tok)
        # Handle combo mods: "X+43.01-17.03" → map each mod separately
        _cm = re.match(r"^([A-Z]?)([+-]\d+\.?\d*)([+-]\d+\.?\d*)$", tok)
        if _cm:
            _p2 = _PRECISION_MAP.get(_cm.group(2), _cm.group(2))
            _p3 = _PRECISION_MAP.get(_cm.group(3), _cm.group(3))
            tok = _cm.group(1) + _p2 + _p3
        normalized.append(tok)
    return normalized


@lru_cache(maxsize=500_000)
def _tokenize_cached(sequence: str) -> tuple[str, ...]:
    return tuple(_tokenize_peptide(sequence))


# ---- Peptide matching: AA mass + variant expansion ----

def _token_mass_fallback(tok: str) -> float:
    """Compute mass for tokens not in _AA_MASSES (e.g. L+43.006, Q-17.027)."""
    if tok in _AA_MASSES:
        return _AA_MASSES[tok]
    # Try: AA + mod + mod  (e.g., L+43.006, C+57.021+42.011)
    m = re.match(r"^([A-Z])([+-]\d+\.?\d*)([+-]\d+\.?\d*)?$", tok)
    if m:
        base = _AA_MASSES.get(m.group(1), 0.0)
        mass = base + float(m.group(2))
        if m.group(3): mass += float(m.group(3))
        return mass
    # Try: mod only (e.g., +43.01, -17.027)
    m = re.match(r"^([+-]\d+\.?\d*)$", tok)
    if m: return float(m.group(1))
    return 0.0


def _aa_match_prefix_suffix(pep1: list[str], pep2: list[str]) -> tuple[list[bool], bool]:
    """Exact copy of piprime_metrics.py aa_match_prefix_suffix."""
    def _prefix(p1, p2):
        matches = [False] * max(len(p1), len(p2))
        i1, i2, cm1, cm2 = 0, 0, 0.0, 0.0
        while i1 < len(p1) and i2 < len(p2):
            m1, m2 = _token_mass_fallback(p1[i1]), _token_mass_fallback(p2[i2])
            if abs(cm1 + m1 - (cm2 + m2)) < 0.5:
                matches[max(i1, i2)] = abs(m1 - m2) < 0.1
                i1 += 1; i2 += 1; cm1 += m1; cm2 += m2
            elif cm2 + m2 > cm1 + m1:
                i1 += 1; cm1 += m1
            else:
                i2 += 1; cm2 += m2
        return matches, all(matches)

    aa_matches, pep_match = _prefix(pep1, pep2)
    if pep_match: return aa_matches, pep_match
    i1, i2 = len(pep1) - 1, len(pep2) - 1
    try: i_stop = aa_matches.index(False)
    except ValueError: i_stop = 0
    cm1 = cm2 = 0.0
    while i1 >= i_stop and i2 >= i_stop:
        m1, m2 = _token_mass_fallback(pep1[i1]), _token_mass_fallback(pep2[i2])
        if abs(cm1 + m1 - (cm2 + m2)) < 0.5:
            aa_matches[max(i1, i2)] = abs(m1 - m2) < 0.1
            i1 -= 1; i2 -= 1; cm1 += m1; cm2 += m2
        elif cm2 + m2 > cm1 + m1:
            i1 -= 1; cm1 += m1
        else:
            i2 -= 1; cm2 += m2
    return aa_matches, all(aa_matches)


def _build_variant_set(truth_tokens: tuple[str, ...]) -> frozenset | None:
    opts_per_pos = [_TOKEN_OPTIONS.get(t, (t,)) for t in truth_tokens]
    variant_count = 1
    for opts in opts_per_pos:
        variant_count *= len(opts)
        if variant_count > 1024: return None
    return frozenset("".join(tokens) for tokens in product(*opts_per_pos))


def _peptide_matches(candidate: str, truth: str) -> bool:
    cand_tokens = _tokenize_cached(candidate)
    truth_tokens = _tokenize_cached(truth)
    if not cand_tokens or not truth_tokens: return False

    # Primary: AA mass matching (piprime_metrics.py)
    _, m = _aa_match_prefix_suffix(list(cand_tokens), list(truth_tokens))
    if m: return True

    # Fallback: variant expansion (compute_candidate_quality_hit_rates.py)
    vs = _build_variant_set(truth_tokens)
    cand_str = "".join(cand_tokens)
    if vs is not None:
        return cand_str in vs
    # Regex fallback for large variant spaces
    parts = []
    for t in truth_tokens:
        opts = _TOKEN_OPTIONS.get(t, (t,))
        parts.append("(?:" + "|".join(re.escape(o) for o in opts) + ")")
    return bool(re.fullmatch("".join(parts), cand_str))


def _normalize_peptide(seq: str) -> str:
    """Legacy helper — kept for backward compatibility in diagnostics output."""
    s = str(seq)
    s = re.sub(r"\(([+-]?\d+\.?\d*)\)", lambda m: m.group(1) if m.group(1).startswith(('+','-')) else '+' + m.group(1), s)
    s = re.sub(r"\[([+-]?\d+\.?\d*)\]", lambda m: m.group(1) if m.group(1).startswith(('+','-')) else '+' + m.group(1), s)
    s = re.sub(r"([+-]\d+)\.(\d{2})\d*", r"\1.\2", s)
    s = s.replace("I", "L")
    return s.strip().upper()


def rescore_scan(
    scan_id: str,
    candidates: list[str],
    original_scores: list[float],
    model: torch.nn.Module,
    model_config: dict,
    stats: dict,
    device: torch.device,
    exp_peaks: np.ndarray,
    exp_charge: int,
    exp_pepmass: float,
    exp_rt: float,
    args: argparse.Namespace,
    ground_truth: str | None = None,
) -> dict:
    """Re-score candidates for a single scan (simplified: piprime_cosine + fusion)."""
    mz_tol = float(args.mz_tol)
    ppm_deviations = np.array([
        compute_mass_ppm_deviation(seq, exp_pepmass, exp_charge) for seq in candidates
    ])
    hard_pass = ppm_deviations <= float(args.precursor_hard_threshold)

    pred_grids, input_grids, valid_masks = batch_predict_candidates(
        model=model, model_config=model_config,
        candidates=candidates,
        precursor_charge=exp_charge, pepmass=exp_pepmass, rt=exp_rt,
        stats=stats, device=device, min_mz=float(args.min_mz),
    )

    results: list[dict] = []
    for i, (seq, pg, ig, vm, ppm_dev, passes_hard) in enumerate(
        zip(candidates, pred_grids, input_grids, valid_masks, ppm_deviations, hard_pass)
    ):
        if passes_hard:
            fg = project_experimental_to_grid(exp_peaks, ig, vm, mz_tol=mz_tol)
            n_filt_slots = int((fg > 0).sum())
            base_score = score_piprime_cosine(
                pg, fg, vm, ig, float(exp_pepmass), int(exp_charge)
            )
        else:
            n_filt_slots = 0
            base_score = 0.0
        results.append({
            "seq": seq,
            "original_score": original_scores[i] if i < len(original_scores) else np.nan,
            "piprime_score": round(float(base_score), 6),
            "v20_score_model": round(float(base_score), 6),
            "v20_score": round(float(base_score), 6),
            "grid_cosine": 0.0,
            "mass_ppm": round(float(ppm_dev), 2),
            "n_filt_slots": n_filt_slots,
            "passes_hard_filter": bool(passes_hard),
        })

    _annotate_candidate_context(results)

    if args.score_method in ("pn_fusion", "rank_fusion", "zscore_fusion"):
        alpha = float(getattr(args, "fusion_alpha", 0.7))
        if getattr(args, "fusion_mode", "norm") == "raw":
            _apply_fusion_raw(results, alpha)
        else:
            _apply_fusion(results, args.score_method, alpha)

    # Post-fusion mass hard filter: non-passing candidates sink to the bottom.
    for r in results:
        if not r["passes_hard_filter"]:
            r["v20_score"] = -1e10

    results.sort(key=lambda r: r["v20_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["v20_rank"] = rank

    best = results[0]
    gt_rank = 0
    aa_matched = 0
    aa_truth_len = 0
    aa_pred_len = 0
    if ground_truth is not None:
        for r in results:
            if _peptide_matches(r["seq"], ground_truth):
                gt_rank = r["v20_rank"]
                break
        best_tokens = _tokenize_cached(best["seq"])
        truth_tokens = _tokenize_cached(ground_truth)
        if best_tokens and truth_tokens:
            matches, _ = _aa_match_prefix_suffix(list(best_tokens), list(truth_tokens))
            aa_matched = int(sum(matches))
            aa_truth_len = len(truth_tokens)
            aa_pred_len = len(best_tokens)

    return {
        "scan_id": scan_id,
        "gt_seq": ground_truth,
        "num_candidates": len(candidates),
        "exp_charge": exp_charge,
        "exp_pepmass": round(exp_pepmass, 4),
        "n_exp_peaks": int(exp_peaks.shape[0]),
        "best_seq": best["seq"],
        "best_score": best["v20_score"],
        "best_original_score": best["original_score"],
        "best_mass_ppm": best["mass_ppm"],
        "gt_rank": gt_rank,
        "gt_hit": gt_rank > 0,
        "aa_matched": aa_matched,
        "aa_truth_len": aa_truth_len,
        "aa_pred_len": aa_pred_len,
        "candidates": results,
    }


# Columns for streaming diagnostics CSV (must match cand dict keys).
DIAGNOSTIC_COLUMNS = [
    "scan_id", "seq", "v20_rank", "v20_score", "v20_score_model", "v20_score_raw",
    "piprime_score", "grid_cosine", "original_score", "pn_rank", "piprime_rank",
    "v20_model_rank", "pn_gap_to_best", "piprime_gap_to_best", "v20_model_gap_to_best",
    "mass_ppm", "mass_da", "n_filt_slots", "passes_hard_filter", "is_best",
]

# Columns for the streaming per-candidate "full results" CSV (--save-full-results):
# one row per candidate, with the scan-level fields repeated on every row of that scan.
# 内容等价于 rescoring_results.json 里每个谱的 candidates[]，但**流式**写出（内存不随谱数增长），
# 所以 26 万谱 × 上百候选的量级也能落盘。
# 列名尽量对齐 rescore_hsdf.py 的 full_results.csv（ordinal / scan_id / charge / pepmass /
# cand_seq / pn_rank / v20_rank / pn_score / fused / mass_ppm / passes_hard / is_best /
# gt_match），因此 scripts/rescoring/save_predictions_cases.py 的 load_full_results_csv()
# 也能直接读这份文件。
FULL_RESULT_COLUMNS = [
    # --- 谱级（同一谱的每个候选行都重复）---
    "ordinal", "scan_id", "charge", "pepmass", "gt_seq",
    "n_exp_peaks", "num_candidates",
    "gt_rank", "gt_hit", "aa_matched", "aa_truth_len", "aa_pred_len",
    "best_seq", "best_score", "best_mass_ppm",
    "rescue_applied", "best_rescue_seq",
    # --- 候选级（HSDF 同名者放在前面）---
    "cand_seq", "pn_rank", "v20_rank", "pn_score", "fused",
    "mass_ppm", "mass_da", "passes_hard", "is_best", "gt_match",
    "piprime_score", "grid_cosine", "v20_score_model", "v20_score_raw",
    "piprime_rank", "v20_model_rank",
    "pn_gap_to_best", "piprime_gap_to_best", "v20_model_gap_to_best",
    "n_filt_slots",
]


def flatten_full_result(ordinal: int, scan_id: str, gt_seq, result: dict, cand: dict) -> dict:
    """One full-results CSV row: scan-level fields repeated + candidate-level scores.

    ``gt_match`` 用 ``v20_rank == gt_rank`` 判定 —— gt_rank 已由 rescore_scan 用质量匹配
    (_peptide_matches) 定位到那条命中候选，所以这里是 O(1)。**不能**对每个候选再跑一遍
    _peptide_matches：它未命中时会走变体展开正则回退，在 26 万谱 × 上百候选的量级上会慢到
    不可接受。
    """
    gt_rank = int(result.get("gt_rank") or 0)
    v20_rank = int(cand.get("v20_rank") or 0)
    return {
        "ordinal": ordinal, "scan_id": scan_id,
        "charge": result.get("exp_charge"), "pepmass": result.get("exp_pepmass"),
        "gt_seq": gt_seq or "",
        "n_exp_peaks": result.get("n_exp_peaks"),
        "num_candidates": result.get("num_candidates"),
        "gt_rank": gt_rank, "gt_hit": bool(result.get("gt_hit")),
        "aa_matched": result.get("aa_matched"),
        "aa_truth_len": result.get("aa_truth_len"),
        "aa_pred_len": result.get("aa_pred_len"),
        "best_seq": result.get("best_seq"), "best_score": result.get("best_score"),
        "best_mass_ppm": result.get("best_mass_ppm"),
        "rescue_applied": bool(result.get("rescue_applied")),
        "best_rescue_seq": result.get("best_rescue_seq") or "",
        "cand_seq": cand.get("seq"),
        "pn_rank": cand.get("pn_rank"), "v20_rank": v20_rank,
        "pn_score": cand.get("original_score"), "fused": cand.get("v20_score"),
        "mass_ppm": cand.get("mass_ppm"), "mass_da": cand.get("mass_da"),
        "passes_hard": cand.get("passes_hard_filter"),
        "is_best": v20_rank == 1,
        "gt_match": bool(gt_rank) and v20_rank == gt_rank,
        "piprime_score": cand.get("piprime_score"),
        "grid_cosine": cand.get("grid_cosine"),
        "v20_score_model": cand.get("v20_score_model"),
        "v20_score_raw": cand.get("v20_score_raw"),
        "piprime_rank": cand.get("piprime_rank"),
        "v20_model_rank": cand.get("v20_model_rank"),
        "pn_gap_to_best": cand.get("pn_gap_to_best"),
        "piprime_gap_to_best": cand.get("piprime_gap_to_best"),
        "v20_model_gap_to_best": cand.get("v20_model_gap_to_best"),
        "n_filt_slots": cand.get("n_filt_slots"),
    }

# ===================================================================
# Section 6: CLI + main()
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V20 Candidate Re-scoring via Spectrum Consistency Validation"
    )
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--mgf-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--piprime-model", type=str, default=None,
                   help="Path to PiPrime model checkpoint (required).")

    p.add_argument("--output-dir", type=Path, default=Path("./rescoring_output"))
    p.add_argument("--save-diagnostics", action="store_true",
                   help="额外保存逐候选的 rescoring_diagnostics.csv (与 --save-full-results 可同时开)。")
    p.add_argument("--save-full-results", action="store_true",
                   help="额外保存每候选一行的 full_results.csv：谱级字段(GT/命中/AA) + 每候选的 "
                        "v20/piprime/grid/pn 分、名次、质量偏差、gt_match。流式写出(内存安全)，"
                        "内容等价于 rescoring_results.json 的 candidates[]；列名对齐 "
                        "rescore_hsdf.py 的 full_results.csv。")
    p.add_argument("--skip-results-save", action="store_true",
                   help="Do not save rescoring_results.json (summary still saved). "
                        "也会挡掉 --save-diagnostics / --save-full-results。")
    p.add_argument("--save-summary-csv", type=str, default=None,
                   help="Save per-scan summary CSV (scan_id, gt_seq, pn_top1, v20_top1, scores, etc.)")

    p.add_argument("--score-method", type=str, default="piprime_cosine",
                   choices=SCORING_METHODS,
                   help="Scoring mode: piprime_cosine, pn_fusion (minmax), rank_fusion, zscore_fusion.")
    p.add_argument("--base-score-method", type=str, default="piprime_cosine",
                   choices=BASE_SCORE_METHODS,
                   help=("Base single-candidate score used before fusion/rerank. "
                         "Ignored when score-method itself is a base score mode."))
    p.add_argument("--fusion-mode", type=str, default="raw",
                   choices=("norm", "raw"),
                   help="Fusion normalization: raw=no-norm (alpha*v20 + (1-alpha)*clip(pn)), norm=min-max per-spectrum")
    p.add_argument("--fusion-alpha", type=float, default=0.7,
                   help="V20 weight in pn_fusion mode (default: 0.7).")
    p.add_argument("--grid-weight", type=float, default=0.5,
                   help="Weight of grid_cosine (gc) vs piprime (pp) in the simplified "
                        "spectral score: score=(1-w)*pp + w*gc. w=1.0 -> gc only, "
                        "w=0.0 -> pp only (default: 0.5).")
    p.add_argument("--gc-mode", type=str, default="cosine",
                   choices=("cosine", "jaccard", "support"),
                   help="Grid similarity metric: cosine (intensity), jaccard (peak "
                        "presence/absence), support (cosine over mutually-present "
                        "peaks only). Default: cosine.")
    p.add_argument("--exp-int-transform", type=str, default="off",
                   choices=("off", "sqrtclip"),
                   help="实验峰强度变换 (投影到网格前, 影响 pp/gc/f 全部分支): "
                        "'off' = 原始 MGF 强度 (默认); "
                        "'sqrtclip' = sqrt(clip(强度,0,50000))/sqrt(50000), "
                        "与模型训练空间 (pred_grid) 对齐, 压缩动态范围。")
    p.add_argument("--gc-topk", type=int, default=0,
                   help="Keep only the top-K predicted ions (by predicted intensity) "
                        "in the grid similarity; 0 = keep all (default).")
    p.add_argument("--gc-min-intens", type=float, default=0.0,
                   help="Relative intensity threshold for grid similarity: drop a cell "
                        "only when its predicted intensity is below min_intens*pred_max "
                        "AND it has no projected experimental peak (filt==0), i.e. pure "
                        "model noise. Experimental cells (filt>0) are always kept. "
                        "0.0 = no filter (default).")
    p.add_argument("--pp-min-intens", type=float, default=0.005,
                   help="Relative intensity threshold (fraction of pred max) for the "
                        "PiPrime (pp) component's predicted-peak list. Predicted peaks "
                        "below pp_min_intens*pred_max are dropped before the pp embedding "
                        "cosine. 0.0 = no filter. Experimental peaks are never filtered.")
    p.add_argument("--fuse-mode", type=str, default="weighted",
                   help="Score fusion across pp/gc components. 'weighted' = raw weighted "
                        "sum as in _simplified_score (default); 'score-avg' = per-spectrum "
                        "minmax-normalize each component then weighted average (DS-style "
                        "hs_d020_f020).")

    p.add_argument("--mz-tol", type=float, default=0.05)
    p.add_argument("--min-mz", type=float, default=100.0)
    p.add_argument("--row-order", action="store_true",
                   help="Match MGF and parquet by row order instead of scan_id.")
    p.add_argument("--row-alignment-mode", type=str, default="auto",
                   choices=("auto", "row_order", "occurrence_aware"),
                   help="Row-order MGF/parquet alignment mode. 'auto' enables mmazei occurrence-aware alignment.")
    p.add_argument("--drop-seq-with-o", action="store_true",
                   help="Drop row-order spectra whose SEQ contains the illegal amino acid O.")
    p.add_argument("--precursor-mismatch-mode", type=str, default="off",
                   choices=("off", "fix", "drop"),
                   help="In row-order mode, fix or drop spectra whose PEPMASS mismatches SEQ mass.")
    p.add_argument("--precursor-mismatch-ppm", type=float, default=50.0,
                   help="PPM threshold used by --precursor-mismatch-mode.")

    p.add_argument("--precursor-hard-threshold", type=float, default=50.0)
    p.add_argument("--mass-threshold-unit", type=str, default="ppm",
                   choices=("ppm", "da"),
                   help="Unit of --precursor-hard-threshold: 'ppm' (relative m/z) "
                        "or 'da' (absolute neutral-mass Da, matches PrimeNovo PMC 0.1 Da).")

    p.add_argument("--mass-rescue", type=str, default="off",
                   choices=("off", "v20_alone", "fused_all"),
                   help="Mass-killed rescue (lazy/triggered). off: current behavior "
                        "(mass-failed candidates get v20_score_model=0 and "
                        "v20_score=-1e10). v20_alone/fused_all: only for spectra whose "
                        "fused top-1 is mass-failed (rescue_applied=True, i.e. all "
                        "candidates exceed the hard filter) compute real V20 scores for "
                        "every candidate (field v20_rescue_score), rank them "
                        "(v20_alone by V20 model score, fused_all by unfiltered "
                        "PN+V20 fusion) and use the winner AS best_seq (deployment "
                        "oracle: summary GT metrics are the deployed ones); "
                        "best_rescue_seq + rescue_applied are also emitted. Normal "
                        "spectra pay zero extra cost (V20 rescoring includes a piprime "
                        "embedding forward pass).")
    p.add_argument("--filter-mode", type=str, default="before",
                   choices=("before", "after"),
                   help="前体质量过滤与重打分的顺序: "
                        "'before' = 先按质量过滤, 只对通过的候选做 pp/gc 重打分, "
                        "未通过的 v20_score_model=0 (默认, 当前行为); "
                        "'after' = 先对所有候选做 pp/gc 重打分, 再按质量过滤 "
                        "(参考 HSDF: 若存在质量通过的候选则只保留通过者, "
                        "若全部超标则全量兜底不过滤)。")

    p.add_argument("--col-scan-id", type=str, default="scan_id")
    p.add_argument("--col-candidates", type=str, default="candidates")
    p.add_argument("--col-scores-raw", type=str, default="scores_raw")
    p.add_argument("--col-scores", type=str, default="scores")

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--random-sample", type=int, default=None,
                   help="Randomly sample N scans from parquet for evaluation.")
    p.add_argument("--random-seed", type=int, default=42,
                   help="Random seed used with --random-sample.")

    return p.parse_args()


def _resolve_row_alignment_mode(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "row_alignment_mode", "auto")).lower()
    if requested != "auto":
        return requested

    parquet_text = str(getattr(args, "parquet", "")).lower()
    mgf_text = str(getattr(args, "mgf_dir", "")).lower()
    if "mmazei" in parquet_text or "methanosarcina-mazei" in mgf_text:
        return "occurrence_aware"
    return "row_order"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # V20 checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_config, stats, incompatible = load_checkpoint(args.checkpoint, device)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        print(f"  [checkpoint] missing={incompatible.missing_keys}, "
              f"unexpected={incompatible.unexpected_keys}")
    model.eval()
    print(f"  in_channels={model_config.get('in_channels', 6)}, "
          f"d_model={model_config.get('d_model', 'N/A')}, "
          f"seq_context={model_config.get('use_sequence_context', True)}")

    # Parquet
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  rows={len(df)}, columns={list(df.columns)}")

    col_scan = args.col_scan_id
    col_cand = args.col_candidates
    col_scores = args.col_scores

    if col_scan not in df.columns:
        raise KeyError(f"Column '{col_scan}' not found. Available: {list(df.columns)}")
    if col_cand not in df.columns:
        raise KeyError(f"Column '{col_cand}' not found. Available: {list(df.columns)}")

    row_policy = MgfRowPolicy(
        drop_seq_with_o=bool(args.drop_seq_with_o),
        precursor_mismatch_mode=str(args.precursor_mismatch_mode).lower(),
        precursor_mismatch_ppm=float(args.precursor_mismatch_ppm),
    )
    row_alignment_mode = _resolve_row_alignment_mode(args) if args.row_order else "scan_id"

    # MGF index (scan_id mode) or row-order / occurrence-aware catalog
    mgf_index: dict = {}
    mgf_total_count = 0
    row_policy_stats = MgfPolicyStats()
    occurrence_catalog = []
    occurrence_entry_by_row: dict[int, object] = {}
    occurrence_align_diag = {
        "compared_rows": 0,
        "missing_labels": 0,
        "overflow_reuse": 0,
        "skipped_dropped_rows": 0,
    }
    available_row_indices = np.arange(len(df), dtype=np.int64)
    if args.row_order:
        print(f"Loading MGF spectra from: {args.mgf_dir}")
        if row_alignment_mode == "occurrence_aware":
            occurrence_catalog, row_policy_stats = build_mgf_row_catalog(args.mgf_dir, row_policy)
            mgf_total_count = row_policy_stats.total_spectra
            occurrence_pairs, occurrence_align_diag = build_occurrence_aligned_pairs(
                df[col_scan].astype(str).tolist(),
                occurrence_catalog,
            )
            occurrence_entry_by_row = {
                int(row_idx): entry
                for row_idx, entry in occurrence_pairs
                if row_idx < len(df)
            }
            available_row_indices = np.asarray(sorted(occurrence_entry_by_row), dtype=np.int64)
            print(
                "  [mgf occurrence] "
                f"kept={row_policy_stats.kept_spectra}/{mgf_total_count} spectra after policy "
                f"(drop_O={row_policy_stats.dropped_seq_with_o}, "
                f"drop_precursor={row_policy_stats.dropped_precursor_mismatch}, "
                f"fix_precursor={row_policy_stats.fixed_precursor_mismatch}); "
                f"aligned_rows={occurrence_align_diag['compared_rows']}, "
                f"missing={occurrence_align_diag['missing_labels']}, "
                f"overflow={occurrence_align_diag['overflow_reuse']}, "
                f"skipped_dropped={occurrence_align_diag['skipped_dropped_rows']}"
            )
        else:
            kept_ordinals: list[int] = []
            for ordinal, _meta, _peaks, _diag in iter_mgf_spectra_in_order(
                args.mgf_dir,
                row_policy=row_policy,
                stats=row_policy_stats,
                include_ordinal=True,
                load_peaks=False,
            ):
                if ordinal < len(df):
                    kept_ordinals.append(ordinal)
            mgf_total_count = row_policy_stats.total_spectra
            available_row_indices = np.asarray(kept_ordinals, dtype=np.int64)
            print(
                "  [mgf seq] "
                f"kept={row_policy_stats.kept_spectra}/{mgf_total_count} spectra after policy "
                f"(drop_O={row_policy_stats.dropped_seq_with_o}, "
                f"drop_precursor={row_policy_stats.dropped_precursor_mismatch}, "
                f"fix_precursor={row_policy_stats.fixed_precursor_mismatch})"
            )
    else:
        print(f"Building MGF index from: {args.mgf_dir}")
        mgf_index = build_mgf_dir_index(args.mgf_dir)

    # PiPrime
    print(f"Loading PiPrime model: {args.piprime_model}")
    _load_piprime(args.piprime_model, target_device=device)

    # Main loop
    if args.random_sample is not None:
        rng = np.random.default_rng(int(args.random_seed))
        sample_n = min(int(args.random_sample), len(available_row_indices))
        sampled_positions = np.sort(rng.choice(len(available_row_indices), size=sample_n, replace=False))
        row_indices = available_row_indices[sampled_positions]
    else:
        n_scans = min(len(available_row_indices), args.max_scans) if args.max_scans else len(available_row_indices)
        row_indices = available_row_indices[:n_scans]

    n_scans = int(len(row_indices))
    if n_scans == 0:
        raise ValueError("No scans available after applying row-order MGF policy.")
    save_diag = bool(args.save_diagnostics) and not args.skip_results_save
    save_full = bool(args.save_full_results) and not args.skip_results_save
    streaming = save_diag or save_full
    keep_all_results = (not args.skip_results_save) and (not streaming)
    all_results: list[dict] = [] if keep_all_results else []
    missing_scans: list[str] = []
    n_processed = 0
    n_gt_hit = 0
    n_gt_top1 = 0
    sum_aa_matched = 0
    sum_aa_truth = 0
    sum_aa_pred = 0
    selected_row_set = {int(idx) for idx in row_indices.tolist()} if args.row_order and row_alignment_mode == "row_order" else set()
    max_selected_row = int(row_indices[-1]) if args.row_order and row_alignment_mode == "row_order" and n_scans > 0 else -1

    # --- CSV streaming file handles ---
    diag_csv_path = args.output_dir / "rescoring_diagnostics.csv"
    full_csv_path = args.output_dir / "full_results.csv"
    diag_csv_file: object = None
    diag_writer: object = None
    full_csv_file: object = None
    full_writer: object = None
    if streaming:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        if save_diag:
            diag_csv_file = open(diag_csv_path, "w", newline="", encoding="utf-8")
            diag_writer = _csv.DictWriter(diag_csv_file, fieldnames=DIAGNOSTIC_COLUMNS, extrasaction="ignore")
            diag_writer.writeheader()
        if save_full:
            full_csv_file = open(full_csv_path, "w", newline="", encoding="utf-8")
            full_writer = _csv.DictWriter(full_csv_file, fieldnames=FULL_RESULT_COLUMNS, extrasaction="ignore")
            full_writer.writeheader()

    def _flatten_candidate(scan_id: str, cand: dict) -> dict:
        """Build a flat CSV row for one candidate."""
        row_dict = {"scan_id": scan_id, "is_best": cand.get("v20_rank") == 1}
        for key in DIAGNOSTIC_COLUMNS:
            if key in ("scan_id", "is_best"):
                continue
            row_dict[key] = cand.get(key)
        return row_dict

    def process_one_row(row_idx: int, exp_meta: dict, exp_peaks: np.ndarray) -> None:
        nonlocal n_processed, n_gt_hit, n_gt_top1
        nonlocal sum_aa_matched, sum_aa_truth, sum_aa_pred
        row = df.iloc[row_idx]
        scan_id = str(row[col_scan])
        candidates = list(row[col_cand])
        raw_scores = row[col_scores] if col_scores in df.columns else []
        original_scores = [_score_to_scalar(s) for s in raw_scores] if raw_scores is not None else []

        if not candidates:
            return

        exp_charge = parse_charge_from_meta(exp_meta)
        exp_pepmass = parse_pepmass_from_meta(exp_meta)
        exp_rt = float(exp_meta.get("RTINSECONDS", 1800.0))
        gt_seq = exp_meta.get("SEQ") or None

        try:
            result = rescore_scan(
                scan_id=scan_id, candidates=candidates,
                original_scores=original_scores,
                model=model, model_config=model_config, stats=stats,
                device=device, exp_peaks=exp_peaks,
                exp_charge=exp_charge, exp_pepmass=exp_pepmass, exp_rt=exp_rt,
                args=args, ground_truth=gt_seq,
            )
        except Exception as e:
            tqdm.write(f"  [ERROR] Rescoring failed for {scan_id}: {e}")
            import traceback; traceback.print_exc()
            return

        n_processed += 1
        if result["gt_hit"]:
            n_gt_hit += 1
        # top-1 命中 = 官方答案 (best_seq, 含 --mass-rescue 替换) 匹配 GT;
        # 与主口径 gt_rank 解耦, 使 rescue 的部署答案直接反映在 summary。
        if gt_seq and _peptide_matches(str(result["best_seq"]), gt_seq):
            n_gt_top1 += 1
        if result.get("aa_matched") is not None:
            sum_aa_matched += int(result["aa_matched"])
            sum_aa_truth += int(result["aa_truth_len"])
            sum_aa_pred += int(result["aa_pred_len"])

        if keep_all_results:
            all_results.append(result)

        if diag_writer is not None:
            for cand in result["candidates"]:
                diag_writer.writerow(_flatten_candidate(scan_id, cand))
        if full_writer is not None:
            for cand in result["candidates"]:
                full_writer.writerow(
                    flatten_full_result(row_idx, scan_id, gt_seq, result, cand))

    if args.row_order and row_alignment_mode == "row_order":
        with tqdm(total=n_scans, desc="rescoring scans", unit="scan") as pbar:
            for ordinal, exp_meta, exp_peaks, _diag in iter_mgf_spectra_in_order(
                args.mgf_dir,
                row_policy=row_policy,
                include_ordinal=True,
            ):
                if ordinal > max_selected_row:
                    break
                if ordinal not in selected_row_set:
                    continue
                process_one_row(int(ordinal), exp_meta, exp_peaks)
                pbar.update(1)
    elif args.row_order and row_alignment_mode == "occurrence_aware":
        for row_idx in tqdm(row_indices, desc="rescoring scans", unit="scan"):
            row = df.iloc[row_idx]
            scan_id = str(row[col_scan])
            entry = occurrence_entry_by_row.get(int(row_idx))
            if entry is None:
                missing_scans.append(scan_id)
                continue
            try:
                exp_meta, exp_peaks = load_spectrum_by_offset(entry.mgf_path, entry.byte_offset)
                keep, exp_meta, _diag = apply_mgf_row_policy(exp_meta, row_policy)
                if not keep:
                    missing_scans.append(scan_id)
                    continue
            except Exception as e:
                tqdm.write(f"  [ERROR] MGF load failed for {scan_id}: {e}")
                missing_scans.append(scan_id)
                continue
            process_one_row(int(row_idx), exp_meta, exp_peaks)
    else:
        for row_idx in tqdm(row_indices, desc="rescoring scans", unit="scan"):
            row = df.iloc[row_idx]
            scan_id = str(row[col_scan])
            index_entry = mgf_index.get(scan_id)
            if index_entry is None:
                nums = re.findall(r"\d+", scan_id)
                if nums:
                    index_entry = mgf_index.get(nums[-1])
            if index_entry is None:
                missing_scans.append(scan_id)
                continue

            mgf_path, byte_offset = index_entry
            try:
                exp_meta, exp_peaks = load_spectrum_by_offset(mgf_path, byte_offset)
            except Exception as e:
                tqdm.write(f"  [ERROR] MGF load failed for {scan_id}: {e}")
                missing_scans.append(scan_id)
                continue

            process_one_row(int(row_idx), exp_meta, exp_peaks)

    # Summary
    gt_hit_rate = n_gt_hit / n_processed if n_processed > 0 else 0.0
    gt_top1_rate = n_gt_top1 / n_processed if n_processed > 0 else 0.0

    summary = {
        "checkpoint": str(args.checkpoint),
        "parquet": str(args.parquet),
        "mgf_dir": str(args.mgf_dir),
        "score_method": args.score_method,
        "base_score_method": args.base_score_method,
        "fusion_alpha": float(args.fusion_alpha),
        "n_scans_in_parquet": n_scans,
        "random_sample": int(args.random_sample) if args.random_sample is not None else None,
        "random_seed": int(args.random_seed) if args.random_sample is not None else None,
        "mgf_row_alignment_mode": row_alignment_mode,
        "mgf_row_policy": {
            "drop_seq_with_o": bool(args.drop_seq_with_o),
            "precursor_mismatch_mode": str(args.precursor_mismatch_mode),
            "precursor_mismatch_ppm": float(args.precursor_mismatch_ppm),
        },
        "mgf_total_spectra": int(mgf_total_count) if args.row_order else None,
        "mgf_kept_spectra": int(row_policy_stats.kept_spectra) if args.row_order else None,
        "mgf_dropped_seq_with_o": int(row_policy_stats.dropped_seq_with_o) if args.row_order else None,
        "mgf_dropped_precursor_mismatch": int(row_policy_stats.dropped_precursor_mismatch) if args.row_order else None,
        "mgf_fixed_precursor_mismatch": int(row_policy_stats.fixed_precursor_mismatch) if args.row_order else None,
        "mgf_occurrence_aligned_rows": int(occurrence_align_diag["compared_rows"]) if args.row_order and row_alignment_mode == "occurrence_aware" else None,
        "mgf_occurrence_missing_labels": int(occurrence_align_diag["missing_labels"]) if args.row_order and row_alignment_mode == "occurrence_aware" else None,
        "mgf_occurrence_overflow_reuse": int(occurrence_align_diag["overflow_reuse"]) if args.row_order and row_alignment_mode == "occurrence_aware" else None,
        "mgf_occurrence_skipped_dropped_rows": int(occurrence_align_diag["skipped_dropped_rows"]) if args.row_order and row_alignment_mode == "occurrence_aware" else None,
        "n_processed": n_processed,
        "n_missing_in_mgf": len(missing_scans),
        "n_gt_hit": n_gt_hit,
        "gt_hit_rate": round(gt_hit_rate, 4),
        "gt_top1": n_gt_top1,
        "gt_top1_rate": round(gt_top1_rate, 4),
        "aa_recall": round(sum_aa_matched / sum_aa_truth, 4) if sum_aa_truth > 0 else None,
        "aa_precision": round(sum_aa_matched / sum_aa_pred, 4) if sum_aa_pred > 0 else None,
        "aa_matched_total": sum_aa_matched,
        "aa_truth_total": sum_aa_truth,
        "aa_pred_total": sum_aa_pred,
    }

    print("\n===== Summary =====")
    print(f"Scans processed:   {n_processed}/{n_scans}")
    print(f"Missing in MGF:    {len(missing_scans)}")
    if args.row_order:
        print(f"Alignment mode:    {row_alignment_mode}")
        print(
            "MGF policy:        "
            f"keep={row_policy_stats.kept_spectra}/{mgf_total_count}, "
            f"drop_O={row_policy_stats.dropped_seq_with_o}, "
            f"drop_prec={row_policy_stats.dropped_precursor_mismatch}, "
            f"fix_prec={row_policy_stats.fixed_precursor_mismatch}"
        )
        if row_alignment_mode == "occurrence_aware":
            print(
                "Occurrence align:  "
                f"aligned={occurrence_align_diag['compared_rows']}, "
                f"missing={occurrence_align_diag['missing_labels']}, "
                f"overflow={occurrence_align_diag['overflow_reuse']}, "
                f"skipped_dropped={occurrence_align_diag['skipped_dropped_rows']}"
            )
    print(f"GT hit rate:       {gt_hit_rate:.4f} ({n_gt_hit}/{n_processed})")
    print(f"GT top-1 rate:     {gt_top1_rate:.4f} ({n_gt_top1}/{n_processed})")
    if sum_aa_truth > 0:
        print(f"AA recall (micro): {sum_aa_matched/sum_aa_truth:.4f} ({sum_aa_matched}/{sum_aa_truth})")
        print(f"AA precision (micro):{sum_aa_matched/sum_aa_pred:.4f} ({sum_aa_matched}/{sum_aa_pred})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "rescoring_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")

    if args.save_summary_csv:
        csv_path = args.output_dir / args.save_summary_csv
        csv_rows = []
        for r in all_results:
            cands = r.get("candidates", [])
            pn_top1 = cands[0]["seq"] if cands else ""
            pn_score = cands[0].get("original_score", 0) if cands else 0
            gt_seq = r.get("gt_seq") or ""
            csv_rows.append({
                "scan_id": r["scan_id"],
                "gt_seq": gt_seq,
                "pn_top1": pn_top1,
                "v20_top1": r.get("best_seq", ""),
                "v20_score": round(r.get("best_score", 0), 6),
                "pn_score": round(float(pn_score), 6),
                "gt_rank": r.get("gt_rank", 0),
                "gt_hit": r.get("gt_hit", False),
                "ppm": round(r.get("best_mass_ppm", 0), 2),
                "charge": r.get("exp_charge", 0),
                "pepmass": r.get("exp_pepmass", 0),
                "n_candidates": r.get("num_candidates", 0),
                "aa_matched": r.get("aa_matched", 0),
                "aa_truth_len": r.get("aa_truth_len", 0),
                "aa_pred_len": r.get("aa_pred_len", 0),
            })
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")

    if streaming:
        modes = [f for f, on in (("--save-diagnostics", save_diag),
                                 ("--save-full-results", save_full)) if on]
        print(f"Results JSON save skipped ({'+'.join(modes)}, CSV only)")
    elif not args.skip_results_save:
        results_path = args.output_dir / "rescoring_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to: {results_path}")
    else:
        print("Results JSON save skipped (--skip-results-save)")

    if diag_csv_file is not None:
        diag_csv_file.close()
        print(f"Diagnostics saved to: {diag_csv_path}")
    if full_csv_file is not None:
        full_csv_file.close()
        print(f"Full results saved to: {full_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
