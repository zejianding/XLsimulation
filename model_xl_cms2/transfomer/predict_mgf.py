#!/usr/bin/env python3
"""Export a Transformer checkpoint prediction as an MGF spectrum.

This is the Transformer-native replacement for the older
``resnet46New/predict_mgf.py`` helper.  It builds the theoretical m/z grid,
normalizes it with checkpoint statistics, runs the transformer, and writes one
or more predicted spectra to MGF.

Example:
    python transfomer/predict_mgf.py ^
      --checkpoint runs/transformer/best_model.pt ^
      --peptide QLEEAEEEAQR --charge 2 ^
      --output out.mgf --min-intensity 0.03 --min-mz 50 ^
      --source-label MODEL_PREDICTION
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

from spectrum_transformer.data.dataset import encode_cleavage_context, parse_modified_sequence
from spectrum_transformer.config import DEFAULT_MIN_VALID_MZ
from spectrum_transformer.models.factory import build_model, prepare_checkpoint_for_inference
from spectrum_transformer.warnings import configure_v20_warning_filters

configure_v20_warning_filters()


# Checkpoints saved with NumPy 2 may pickle references under ``numpy._core``.
# NumPy 1 exposes the same modules under ``numpy.core``; register aliases so
# torch.load can read server checkpoints on older local environments.
try:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray

    import sys

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
except Exception:
    pass


if os.name == "nt":
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass


# Shared token vocabulary — see <project_root>/token_table.py.
_project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from token_table import (  # noqa: E402
    AA_MASS,
    CHARGE_TOKENS,
    H2O,
    MAX_CLEAVAGE_LEN,
    NUM_CHANNELS,
    NUM_TOKENS,
    PROTON,
)


def parse_sequence(seq: str) -> list[tuple[str, float]]:
    """Parse a peptide string into (AA, residue_mass) pairs.

    Handles both old bracket (M(+15.99)) and MassiveKB digital (M+15.995) formats.
    """
    residues, residue_mod_deltas, _ = parse_modified_sequence(seq)
    result = []
    for aa, delta in zip(residues, residue_mod_deltas):
        mass = AA_MASS.get(aa, 0.0) + delta
        result.append((aa, mass))
    return result


def calculate_pepmass(seq: str, charge: int) -> float:
    residues, residue_mod_deltas, nterm_mod = parse_modified_sequence(seq)
    neutral_mass = sum(AA_MASS.get(aa, 0.0) + delta for aa, delta in zip(residues, residue_mod_deltas)) + nterm_mod
    return (neutral_mass + H2O + charge * PROTON) / charge


def generate_by_ions(seq: str, max_charge: int = 3) -> list[dict[str, float | int | str]]:
    residues, residue_mod_deltas, nterm_shift = parse_modified_sequence(seq)
    masses = [AA_MASS.get(aa, 0.0) + delta for aa, delta in zip(residues, residue_mod_deltas)]
    if len(masses) < 2:
        return []
    prefix = np.cumsum(masses)
    total = float(prefix[-1]) + nterm_shift
    ions = []
    for cleavage_index in range(1, len(masses)):
        b_neutral = float(prefix[cleavage_index - 1]) + nterm_shift
        y_neutral = (total - b_neutral) + H2O
        for z in range(1, max_charge + 1):
            ions.append(
                {
                    "ion_type": "B",
                    "cleavage_index": cleavage_index,
                    "charge": z,
                    "mz": (b_neutral + z * PROTON) / z,
                }
            )
            ions.append(
                {
                    "ion_type": "Y",
                    "cleavage_index": cleavage_index,
                    "charge": z,
                    "mz": (y_neutral + z * PROTON) / z,
                }
            )
    return ions


def build_input_mz_matrix(seq: str, min_mz: float) -> np.ndarray:
    grid = np.zeros((MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    for ion in generate_by_ions(seq, max_charge=3):
        row = int(ion["cleavage_index"]) - 1
        z = int(ion["charge"])
        if row >= MAX_CLEAVAGE_LEN:
            continue
        channel = (z - 1) if ion["ion_type"] == "B" else (z + 2)
        for token_index, token in enumerate(CHARGE_TOKENS[z]):
            mz = float(ion["mz"]) + token
            if mz >= min_mz:
                grid[row, token_index, channel] = mz
    return grid


def build_input_mz_validity(seq: str, min_mz: float) -> np.ndarray:
    mask = np.zeros((MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    for ion in generate_by_ions(seq, max_charge=3):
        row = int(ion["cleavage_index"]) - 1
        z = int(ion["charge"])
        if row >= MAX_CLEAVAGE_LEN:
            continue
        channel = (z - 1) if ion["ion_type"] == "B" else (z + 2)
        for token_index, token in enumerate(CHARGE_TOKENS[z]):
            mask[row, token_index, channel] = 1.0 if float(ion["mz"]) + token >= min_mz else 0.0
    return mask


def build_position_mask(seq: str) -> tuple[np.ndarray, int]:
    residues, _, _ = parse_modified_sequence(seq)
    cleavage_count = len(residues) - 1
    mask = np.zeros((MAX_CLEAVAGE_LEN,), dtype=np.float32)
    mask[:cleavage_count] = 1.0
    return mask, cleavage_count


def build_channel_mask(precursor_charge: int) -> np.ndarray:
    mask = np.zeros((NUM_CHANNELS,), dtype=np.float32)
    for z in range(1, min(3, precursor_charge) + 1):
        mask[z - 1] = 1.0
        mask[z + 2] = 1.0
    return mask


def build_valid_mask(seq: str, precursor_charge: int, min_mz: float) -> tuple[np.ndarray, int]:
    position_mask, cleavage_count = build_position_mask(seq)
    channel_mask = build_channel_mask(precursor_charge)
    mz_mask = build_input_mz_validity(seq, min_mz=min_mz)
    valid = position_mask[:, None, None] * channel_mask[None, None, :]
    valid = np.broadcast_to(valid, (MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS)).copy()
    return (valid * mz_mask).astype(np.float32, copy=False), cleavage_count


def reshape_stat(value, is_channel: bool) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if is_channel and arr.ndim == 1:
        return arr.reshape(1, 1, 1, -1)
    if not is_channel and arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _assert_token_axis_matches(model, state_dict):
    """Fail loudly when a checkpoint's token axis does not match this build.

    ``load_state_dict(strict=False)`` silently drops the per-token positional
    embedding (``packed_pos_w.weight``) when the checkpoint was trained with a
    different column count, leaving that embedding randomly initialised and the
    predictions plausible-looking but wrong.  Guard against it explicitly.
    """
    key = "packed_pos_w.weight"
    ckpt_weight = state_dict.get(key)
    model_weight = model.state_dict().get(key)
    if ckpt_weight is None or model_weight is None:
        return
    if tuple(ckpt_weight.shape) != tuple(model_weight.shape):
        raise ValueError(
            f"Checkpoint token axis mismatch: '{key}' is {tuple(ckpt_weight.shape)} "
            f"in the checkpoint but {tuple(model_weight.shape)} in this build "
            f"(NUM_TOKENS={NUM_TOKENS}). Retrain for the current token table."
        )


def load_checkpoint(checkpoint_path: Path, device: torch.device, no_packed: bool = False):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config, state_dict = prepare_checkpoint_for_inference(ckpt)
    if no_packed:
        # 带 packed 训练的模型，packed_pos_h / packed_pos_w 是学到的位置嵌入；
        # 关掉 packed 后这两张表整个不参与计算，且 load_state_dict(strict=False)
        # 会静默忽略它们 —— 分数会全线大跌却不报错。这里直接拦下。
        if model_config.get("use_packed", False):
            raise SystemExit(
                f"ERROR: 该 checkpoint 是按 packed 训练的 "
                f"(model_config.use_packed=True: {checkpoint_path})，"
                f"但调用方传了 --no-packed。关掉 packed 会丢弃 packed_pos_h/w "
                f"两张已学到的位置嵌入，让所有分数静默下降。\n"
                f"       去掉 --no-packed / 设 NOPACKED=0 即可（推荐）。"
            )
        model_config["use_packed"] = False

    model = build_model(model_config).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    _assert_token_axis_matches(model, state_dict)
    model.eval()

    stats = {
        "channel_means": reshape_stat(ckpt["channel_means"], is_channel=True),
        "channel_stds": reshape_stat(ckpt["channel_stds"], is_channel=True),
        "aux_feature_means": reshape_stat(ckpt["aux_feature_means"], is_channel=False),
        "aux_feature_stds": reshape_stat(ckpt["aux_feature_stds"], is_channel=False),
    }
    return model, model_config, stats, incompatible


def prepare_model_input(
    input_mz: np.ndarray,
    valid_mask: np.ndarray,
    stats: dict[str, np.ndarray],
    expected_in_channels: int,
) -> np.ndarray:
    inp = input_mz[np.newaxis, ...]
    mask = valid_mask[np.newaxis, ...]
    norm_inp = (inp - stats["channel_means"]) / np.maximum(stats["channel_stds"], 1e-6)
    norm_inp = np.where(mask, norm_inp, 0.0).astype(np.float32, copy=False)
    if expected_in_channels == norm_inp.shape[-1]:
        return norm_inp
    if expected_in_channels == norm_inp.shape[-1] + mask.shape[-1]:
        return np.concatenate([norm_inp, mask.astype(np.float32)], axis=-1)
    raise ValueError(
        f"Checkpoint expects {expected_in_channels} input channels, "
        f"but prepared {norm_inp.shape[-1]} or {norm_inp.shape[-1] + mask.shape[-1]}."
    )


def predict_spectrum(
    model: torch.nn.Module,
    model_input: np.ndarray,
    valid_mask: np.ndarray,
    aux_norm: np.ndarray,
    peptide: str,
    precursor_charge: int,
    cleavage_count: int,
    device: torch.device,
) -> np.ndarray:
    forward_kwargs = {
        "aux_features": torch.from_numpy(aux_norm).unsqueeze(0).to(device),
        "valid_mask": torch.from_numpy(valid_mask[np.newaxis, ...]).to(device),
        "charge": torch.tensor([int(precursor_charge)], dtype=torch.long, device=device),
        "length": torch.tensor([int(cleavage_count + 1)], dtype=torch.long, device=device),
    }
    if getattr(model, "use_sequence_context", False):
        aa_ids, scalar = encode_cleavage_context(
            peptide, cleavage_count + 1, max_cleavage_len=MAX_CLEAVAGE_LEN
        )
        forward_kwargs["cleavage_aa_ids"] = (
            torch.from_numpy(aa_ids.astype(np.int64, copy=False)).unsqueeze(0).to(device)
        )
        forward_kwargs["cleavage_scalar_features"] = (
            torch.from_numpy(scalar.astype(np.float32, copy=False)).unsqueeze(0).to(device)
        )

    with torch.inference_mode():
        out = model(torch.from_numpy(model_input).to(device), **forward_kwargs)
    score = out["prediction"]
    return score.detach().cpu().numpy()[0]


def prediction_to_mgf_peaks(
    input_mz: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    min_intensity: float,
    min_mz: float,
    mz_merge_tol: float,
) -> np.ndarray:
    pos = np.where((valid_mask > 0) & (prediction > min_intensity) & (input_mz >= min_mz))
    if pos[0].size == 0:
        return np.empty((0, 2), dtype=np.float32)

    mz_vals = input_mz[pos]
    int_vals = prediction[pos]
    order = np.argsort(mz_vals)
    mz_vals = mz_vals[order]
    int_vals = int_vals[order]

    merged_mz = []
    merged_int = []
    i = 0
    while i < len(mz_vals):
        j = i + 1
        while j < len(mz_vals) and (mz_vals[j] - mz_vals[i]) < mz_merge_tol:
            j += 1
        best = i + int(np.argmax(int_vals[i:j]))
        merged_mz.append(mz_vals[best])
        merged_int.append(int_vals[best])
        i = j

    return np.column_stack(
        [np.asarray(merged_mz, dtype=np.float32), np.asarray(merged_int, dtype=np.float32)]
    )


def write_mgf(
    output_path: Path,
    spectra_peaks: list[np.ndarray],
    seq: str,
    charge: int,
    pepmass: float,
    source_label: str,
    rt_values: list[float],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, peaks in enumerate(spectra_peaks):
            f.write("BEGIN IONS\n")
            f.write(f"TITLE=TRANSFORMER_PRED_{i + 1}\n")
            f.write(f"PEPMASS={pepmass:.6f}\n")
            f.write(f"CHARGE={charge}+\n")
            f.write(f"SEQ={seq}\n")
            f.write(f"SOURCE={source_label}\n")
            f.write(f"RTINSECONDS={rt_values[i]:.4f}\n")
            for mz, intensity in peaks:
                f.write(f"{float(mz):.5f} {float(intensity):.6f}\n")
            f.write("END IONS\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export transformer-predicted spectra to MGF.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--peptide", required=True)
    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-spectra", type=int, default=1)
    parser.add_argument("--rt", type=float, default=None, help="Use a fixed RTINSECONDS value.")
    parser.add_argument("--rt-mean", type=float, default=1800.0)
    parser.add_argument("--rt-std", type=float, default=300.0)
    parser.add_argument("--min-intensity", type=float, default=0.01)
    parser.add_argument("--min-mz", type=float, default=DEFAULT_MIN_VALID_MZ)
    parser.add_argument("--mz-merge-tol", type=float, default=0.05)
    parser.add_argument("--source-label", default="MODEL_PREDICTION")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-packed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_spectra < 1:
        raise ValueError("--num-spectra must be >= 1")
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    model, model_config, stats, incompatible = load_checkpoint(
        args.checkpoint, device, no_packed=args.no_packed
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Checkpoint loaded with non-strict state dict: "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    pepmass = calculate_pepmass(args.peptide, args.charge)
    input_mz = build_input_mz_matrix(args.peptide, min_mz=args.min_mz)
    valid_mask, cleavage_count = build_valid_mask(args.peptide, args.charge, min_mz=args.min_mz)
    model_input = prepare_model_input(
        input_mz,
        valid_mask,
        stats,
        expected_in_channels=int(model_config.get("in_channels", NUM_CHANNELS)),
    )

    rng = np.random.default_rng(args.seed)
    all_peaks = []
    rt_values = []
    for _ in range(args.num_spectra):
        rt = float(args.rt) if args.rt is not None else float(max(60.0, rng.normal(args.rt_mean, args.rt_std)))
        raw_aux = np.array([[float(args.charge), pepmass, rt]], dtype=np.float32)
        aux_norm = (
            (raw_aux - stats["aux_feature_means"])
            / np.maximum(stats["aux_feature_stds"], 1e-6)
        ).astype(np.float32)[0]
        prediction = predict_spectrum(
            model,
            model_input,
            valid_mask,
            aux_norm,
            args.peptide,
            args.charge,
            cleavage_count,
            device,
        )
        peaks = prediction_to_mgf_peaks(
            input_mz,
            prediction,
            valid_mask,
            min_intensity=args.min_intensity,
            min_mz=args.min_mz,
            mz_merge_tol=args.mz_merge_tol,
        )
        all_peaks.append(peaks)
        rt_values.append(rt)

    write_mgf(
        args.output,
        all_peaks,
        args.peptide,
        args.charge,
        pepmass,
        args.source_label,
        rt_values,
    )
    print(
        f"Wrote {len(all_peaks)} spectrum(s) to {args.output} | "
        f"peaks={','.join(str(len(peaks)) for peaks in all_peaks)} | "
        f"cleavage_context_residual={'on' if getattr(model, 'use_sequence_context', False) else 'off'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
