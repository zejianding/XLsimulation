#!/usr/bin/env python3
"""Project ORIGINAL spectra in grouped MGFs onto the V20 39x46x6 grid.

This keeps MODEL_PREDICTION blocks unchanged, but rewrites ORIGINAL blocks by
mapping experimental peaks to the nearest valid theoretical slot for the same
sequence/charge. The output is a new grouped-MGF directory that can be passed
through the existing PiPrime embedding + cosine summary pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from predict_mgf import build_input_mz_matrix, build_valid_mask

MANIFEST_NAME = "_projection_manifest.json"


def iter_mgf_blocks(path: Path):
    block = []
    in_block = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s == "BEGIN IONS":
                in_block = [line]
                in_block = True
                block = [line]
            elif in_block:
                block.append(line)
                if s == "END IONS":
                    yield block
                    block = []
                    in_block = False


def parse_block(block_lines: list[str]) -> tuple[dict, list[tuple[float, float]]]:
    meta = {}
    peaks = []
    for line in block_lines:
        s = line.strip()
        if not s:
            continue
        upper = s.upper()
        if "=" in s and not upper.startswith(("BEGIN IONS", "END IONS")):
            k, v = s.split("=", 1)
            meta[k.upper()] = v.strip()
            continue
        parts = s.split()
        if len(parts) == 2:
            try:
                peaks.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return meta, peaks


def _nearest_slot_indices(slot_mz: np.ndarray, peak_mz: float, tol: float) -> list[int]:
    idx = int(np.searchsorted(slot_mz, peak_mz))
    candidates = []
    if idx < slot_mz.size:
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if idx + 1 < slot_mz.size:
        candidates.append(idx + 1)
    valid = [i for i in candidates if abs(float(slot_mz[i]) - peak_mz) <= tol]
    if not valid:
        return []
    best = min(valid, key=lambda i: abs(float(slot_mz[i]) - peak_mz))
    return [best]


def build_slot_cache(seq: str, charge: int, min_mz: float):
    input_mz = build_input_mz_matrix(seq, min_mz=min_mz)
    valid_mask, _ = build_valid_mask(seq, precursor_charge=charge, min_mz=min_mz)
    valid_pos = np.argwhere(valid_mask > 0)
    slot_mz = np.asarray([input_mz[pos, tok, ch] for pos, tok, ch in valid_pos], dtype=np.float64)
    order = np.argsort(slot_mz)
    return valid_pos[order], slot_mz[order]


def project_peaks_to_grid(peaks: list[tuple[float, float]], seq: str, charge: int, min_mz: float, tol: float, aggregate: str, cache: dict):
    key = (seq, int(charge), float(min_mz))
    if key not in cache:
        cache[key] = build_slot_cache(seq, charge, min_mz=min_mz)
    valid_pos, slot_mz = cache[key]
    if slot_mz.size == 0:
        return []

    slot_intensity = np.zeros(slot_mz.shape[0], dtype=np.float64)
    for mz, inten in peaks:
        if mz < min_mz:
            continue
        matched = _nearest_slot_indices(slot_mz, mz, tol)
        if not matched:
            continue
        idx = matched[0]
        if aggregate == "sum":
            slot_intensity[idx] += float(inten)
        else:
            slot_intensity[idx] = max(slot_intensity[idx], float(inten))

    projected = []
    for idx, inten in enumerate(slot_intensity):
        if inten <= 0.0:
            continue
        projected.append({"mz": float(slot_mz[idx]), "intensity": float(inten)})
    projected.sort(key=lambda x: x["mz"])
    return projected


def write_block(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
        f.write("\n")


def make_projected_block(meta: dict, projected_peaks: list[dict], tol: float, aggregate: str) -> list[str]:
    lines = ["BEGIN IONS\n"]
    source = meta.get("SOURCE", "ORIGINAL")
    lines.append(f"SOURCE={source}\n")
    for key in ("TITLE", "PEPMASS", "RTINSECONDS", "CHARGE", "SCANS", "SEQ"):
        if key in meta:
            lines.append(f"{key}={meta[key]}\n")
    comment = meta.get("COMMENT", "")
    extra = f"projected_to_v20_grid tol={tol} aggregate={aggregate}"
    if comment:
        lines.append(f"COMMENT={comment} {extra}\n")
    else:
        lines.append(f"COMMENT={extra}\n")
    for peak in projected_peaks:
        lines.append(f"{peak['mz']:.6f} {peak['intensity']:.6f}\n")
    lines.append("END IONS\n")
    return lines


def projection_params(args, in_dir: Path, groups: list[Path]) -> dict:
    return {
        "input_dir": str(in_dir.resolve()),
        "min_mz": float(args.min_mz),
        "match_tol": float(args.match_tol),
        "aggregate": args.aggregate,
        "project_prediction": bool(args.project_prediction),
        "input_mgf_names": [p.name for p in groups],
    }


def projection_is_current(manifest_path: Path, out_dir: Path, params: dict) -> bool:
    """True only if a previous run finished with the same parameters and wrote every file.

    Checking ``out_dir.is_dir()`` is not enough: the directory is created before any
    work, so an interrupted run leaves a half-written tree that would then be reused
    silently and produce grid numbers over a truncated spectrum set.
    """
    if not manifest_path.is_file():
        return False
    try:
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if prev.get("params") != params:
        return False
    return all((out_dir / n).is_file() for n in params["input_mgf_names"])


def main():
    parser = argparse.ArgumentParser(description="Project ORIGINAL grouped MGFs onto the V20 39x46x6 grid.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-mz", type=float, default=100.0)
    parser.add_argument("--match-tol", type=float, default=0.05)
    parser.add_argument("--aggregate", choices=["max", "sum"], default="max")
    parser.add_argument("--project-prediction", action="store_true",
                        help="ALSO snap MODEL_PREDICTION peaks onto the same grid "
                             "(symmetric metric: both sides on the V20 grid)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    groups = sorted(in_dir.glob("*.mgf"))
    params = projection_params(args, in_dir, groups)
    manifest_path = out_dir / MANIFEST_NAME

    if projection_is_current(manifest_path, out_dir, params):
        print(f"Projection already complete for these parameters, reusing: {out_dir}")
        print(f"Grouped MGF files: {len(groups)}")
        print(f"Output dir: {out_dir}")
        return

    # Stale manifest, interrupted previous run, or changed parameters -> rebuild.
    # The manifest is removed first so a crash mid-rebuild cannot look complete.
    if out_dir.exists():
        for stale in list(out_dir.glob("*.mgf")) + [manifest_path]:
            stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    print(f"Grouped MGF files: {len(groups)}")
    total_orig = 0
    total_pred = 0
    total_pred_proj = 0
    total_proj_peaks = 0

    for mgf_path in groups:
        out_path = out_dir / mgf_path.name
        if out_path.exists():
            out_path.unlink()
        blocks = list(iter_mgf_blocks(mgf_path))
        if not blocks:
            # keep the file present so the completeness check above can succeed
            out_path.write_text("", encoding="utf-8")
            continue
        for block in blocks:
            meta, peaks = parse_block(block)
            source = meta.get("SOURCE", "").upper()
            if source == "ORIGINAL" or (args.project_prediction and source != "ORIGINAL"):
                seq = meta.get("SEQ", "")
                charge_str = meta.get("CHARGE", "0")
                charge_match = re.search(r"(\d+)", str(charge_str))
                charge = int(charge_match.group(1)) if charge_match else 0
                projected = project_peaks_to_grid(
                    peaks=peaks,
                    seq=seq,
                    charge=charge,
                    min_mz=args.min_mz,
                    tol=args.match_tol,
                    aggregate=args.aggregate,
                    cache=cache,
                )
                write_block(out_path, make_projected_block(meta, projected, tol=args.match_tol, aggregate=args.aggregate))
                if source == "ORIGINAL":
                    total_orig += 1
                    total_proj_peaks += len(projected)
                else:
                    total_pred_proj += 1
            else:
                write_block(out_path, block)
                total_pred += 1

    manifest_path.write_text(json.dumps({
        "params": params,
        "projected_original_spectra": total_orig,
        "copied_model_prediction_spectra": total_pred,
        "projected_model_prediction_spectra": total_pred_proj,
        "avg_projected_peaks_per_original": round(total_proj_peaks / max(total_orig, 1), 4),
    }, indent=2), encoding="utf-8")

    print(f"Projected ORIGINAL spectra: {total_orig}")
    print(f"Copied MODEL_PREDICTION spectra: {total_pred}")
    print(f"Projected MODEL_PREDICTION spectra: {total_pred_proj}")
    print(f"Average projected peaks per ORIGINAL: {total_proj_peaks / max(total_orig, 1):.2f}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
