#!/usr/bin/env python3
"""Single-peptide prediction and reproducible grid-to-MGF export for V20_0918."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_DIR))
from predict_mgf import (  # noqa: E402
    build_input_mz_matrix, build_valid_mask, calculate_pepmass, load_checkpoint,
    predict_spectrum, prediction_to_mgf_peaks, prepare_model_input, write_mgf,
)
from spectrum_transformer.data.dataset import parse_modified_sequence  # noqa: E402
from token_table import AA_MASS, CHANNEL_NAMES, CHARGE_TOKENS  # noqa: E402

SHAPE = (39, 46, 6)
NUMBER = r"[+-](?:\d+(?:\.\d+)?|\.\d+)"
MOD = rf"(?:{NUMBER}|\({NUMBER}\)|\[{NUMBER}\])"
SEQUENCE = re.compile(rf"(?:{MOD})*(?:[ACDEFGHIKLMNPQRSTVWY](?:{MOD})*)+")


def validate_peptide(peptide, charge):
    if not SEQUENCE.fullmatch(peptide):
        raise ValueError("Use uppercase standard amino acids and signed numeric modifications, e.g. M+15.995 or C(+57.021).")
    residues, deltas, nterm = parse_modified_sequence(peptide)
    if not 2 <= len(residues) <= 40:
        raise ValueError("The 39-row model supports peptides of length 2..40; truncation is not allowed.")
    if charge < 1:
        raise ValueError("Precursor charge must be a positive integer.")
    if not math.isfinite(nterm) or any(not math.isfinite(d) or AA_MASS[a] + d <= 0 for a, d in zip(residues, deltas)):
        raise ValueError("Modification masses must be finite and residue masses positive.")
    return len(residues)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_grid(mz, prediction, mask):
    for name, array in (("input_mz", mz), ("prediction", prediction), ("valid_mask", mask)):
        if array.shape != SHAPE or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite array with shape {SHAPE}.")
    if not np.isin(mask, [0, 1]).all() or np.any(mz[mask > 0] <= 0):
        raise ValueError("Mask must be binary and valid m/z values positive.")


def export_peaks(mz, prediction, mask, min_intensity, mz_merge_tol, intensity_space):
    validate_grid(mz, prediction, mask)
    if not math.isfinite(min_intensity) or min_intensity < 0:
        raise ValueError("--min-intensity must be finite and >= 0.")
    if not math.isfinite(mz_merge_tol) or mz_merge_tol <= 0:
        raise ValueError("--mz-merge-tol must be finite and > 0.")
    # The threshold is always in model output space, before optional plotting conversion.
    peaks = prediction_to_mgf_peaks(mz, prediction, mask, min_intensity, 0.0, mz_merge_tol)
    if intensity_space == "relative100" and len(peaks):
        peaks[:, 1] = np.square(peaks[:, 1] / peaks[:, 1].max()) * 100.0
    return peaks


def write_outputs(out_dir, mz, prediction, mask, metadata, args, save_grid):
    peaks = export_peaks(mz, prediction, mask, args.min_intensity, args.mz_merge_tol, args.intensity_space)
    # Re-running replaces this command's result files, preserving other files.
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata["export"] = {
        "min_intensity_model_space": args.min_intensity,
        "mz_merge_tol_da": args.mz_merge_tol,
        "intensity_space": args.intensity_space,
        "merge_rule": "sorted anchor window: mz - first_mz < tolerance; keep strongest cell",
        "n_peaks": len(peaks),
    }
    if save_grid:
        np.savez_compressed(out_dir / "grid.npz", input_mz=mz, prediction=prediction,
                            valid_mask=mask, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
        np.save(out_dir / "prediction_grid.npy", prediction)
        with (out_dir / "grid.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cleavage_index", "ion_ordinal", "token_index", "offset_mz", "channel", "mz", "predicted_intensity", "valid"])
            for row, token, channel in np.ndindex(SHAPE):
                ion = row + 1 if channel < 3 else metadata["peptide_length"] - row - 1
                ordinal = ion if row < metadata["peptide_length"] - 1 else ""
                writer.writerow([row + 1, ordinal, token, CHARGE_TOKENS[channel % 3 + 1][token],
                                 CHANNEL_NAMES[channel], float(mz[row, token, channel]),
                                 float(prediction[row, token, channel]), int(mask[row, token, channel])])
    write_mgf(out_dir / "prediction.mgf", [peaks], metadata["peptide"], metadata["charge"],
              metadata["pepmass"], "V20_0918_CASESTUDY", [metadata["rt_seconds"]])
    with (out_dir / "peaks.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["mz", "intensity"])
        writer.writerows((f"{mz:.5f}", f"{intensity:.6f}") for mz, intensity in peaks)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(peaks)} peaks to {out_dir.resolve()} (grid shape {SHAPE}).")
    if not len(peaks):
        print("WARNING: No peaks pass the threshold. The MGF contains an empty spectrum; lower --min-intensity or inspect the grid.")


def run_predict(args):
    import torch
    length = validate_peptide(args.peptide, args.charge)
    if not math.isfinite(args.min_mz) or args.min_mz <= 0:
        raise ValueError("--min-mz must be finite and > 0.")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model, config, stats, incompatible = load_checkpoint(args.checkpoint, device)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint is incompatible; refusing partial weight loading: {incompatible}")
    if any(not np.isfinite(value).all() for value in stats.values()):
        raise ValueError("Checkpoint normalization statistics must be finite.")
    pepmass = args.pepmass if args.pepmass is not None else calculate_pepmass(args.peptide, args.charge)
    rt = args.rt if args.rt is not None else float(stats["aux_feature_means"].reshape(-1)[2])
    if not math.isfinite(pepmass) or pepmass <= 0 or not math.isfinite(rt) or rt < 0:
        raise ValueError("PEPMASS must be finite and positive; RT must be finite and >= 0.")
    mz = build_input_mz_matrix(args.peptide, args.min_mz)
    mask, cleavage_count = build_valid_mask(args.peptide, args.charge, args.min_mz)
    inp = prepare_model_input(mz, mask, stats, int(config.get("in_channels", 6)))
    raw_aux = np.array([[args.charge, pepmass, rt]], dtype=np.float32)
    aux = ((raw_aux - stats["aux_feature_means"]) / np.maximum(stats["aux_feature_stds"], 1e-6)).astype(np.float32)[0]
    prediction = predict_spectrum(model, inp, mask, aux, args.peptide, args.charge, cleavage_count, device)
    metadata = {
        "schema_version": 1, "peptide": args.peptide, "peptide_length": length,
        "charge": args.charge, "pepmass": pepmass, "pepmass_source": "provided" if args.pepmass is not None else "theoretical",
        "rt_seconds": rt, "rt_source": "provided" if args.rt is not None else "checkpoint_training_mean",
        "min_mz": args.min_mz, "grid_shape": list(SHAPE), "channels": list(CHANNEL_NAMES),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "model_config": config, "normalization": {k: v.tolist() for k, v in stats.items()},
        "device": str(device), "torch_version": torch.__version__, "numpy_version": np.__version__,
        "grid_intensity_space": "model_output_unchanged",
        "code_sha256": {str(p.relative_to(TRANSFORMER_DIR.parent)): sha256(p) for p in (
            Path(__file__).resolve(), TRANSFORMER_DIR / "predict_mgf.py", TRANSFORMER_DIR.parent / "token_table.py",
            TRANSFORMER_DIR / "spectrum_transformer/models/transformer.py",
            TRANSFORMER_DIR / "spectrum_transformer/models/factory.py",
            TRANSFORMER_DIR / "spectrum_transformer/data/dataset.py")},
    }
    write_outputs(args.out_dir, mz, prediction, mask, metadata, args, save_grid=True)


def run_export(args):
    with np.load(args.grid, allow_pickle=False) as saved:
        mz, prediction, mask = (saved[k] for k in ("input_mz", "prediction", "valid_mask"))
        metadata = json.loads(str(saved["metadata_json"].item()))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported grid schema; provide a grid.npz generated by this script.")
    length = validate_peptide(metadata["peptide"], metadata["charge"])
    if length != metadata["peptide_length"]:
        raise ValueError("Grid peptide length metadata is inconsistent.")
    metadata["export_from_grid"] = str(args.grid.resolve())
    metadata["export_from_grid_sha256"] = sha256(args.grid)
    write_outputs(args.out_dir, mz, prediction, mask, metadata, args, save_grid=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="Peptide + charge -> grid, CSV, MGF and provenance")
    predict.add_argument("--peptide", required=True)
    predict.add_argument("--charge", type=int, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--rt", type=float, help="Seconds; default: checkpoint training mean RT")
    predict.add_argument("--pepmass", type=float, help="Precursor m/z; default: theoretical")
    predict.add_argument("--min-mz", type=float, default=100.0)
    predict.add_argument("--device", help="cpu, cuda:0, etc.; default: auto")
    predict.add_argument("--threads", type=int, default=4, help="Torch CPU threads")
    export = sub.add_parser("export", help="Saved grid.npz -> MGF/TSV with new export settings; no checkpoint needed")
    export.add_argument("--grid", type=Path, required=True)
    for command in (predict, export):
        command.add_argument("--out-dir", type=Path, required=True, help="Output directory; existing result files are overwritten")
        command.add_argument("--min-intensity", type=float, default=0.01, help="Strict lower bound in model output space")
        command.add_argument("--mz-merge-tol", type=float, default=0.05, help="Anchor-window tolerance, Da")
        command.add_argument("--intensity-space", choices=["model", "relative100"], default="model",
                             help="model: unchanged; relative100: square positive model scores and scale base peak to 100")
    args = parser.parse_args(argv)
    if args.out_dir.exists() and not args.out_dir.is_dir():
        parser.error("--out-dir must be a directory, not a file.")
    try:
        (run_predict if args.command == "predict" else run_export)(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
