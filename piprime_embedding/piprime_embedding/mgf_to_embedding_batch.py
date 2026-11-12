#!/usr/bin/env python3
"""Generate spectrum embeddings for every spectrum in an MGF or a folder of MGF files.

Output format: NPZ (np.savez_compressed)
  - embeddings: float32 array, shape [N, D]
  - titles: unicode str array, shape [N]
  - precursor_mz: float32 array, shape [N]
  - charges: int32 array, shape [N]

Examples:
  Single file:
    python mgf_to_embedding.py --mgf GITIATSHVEYESDTR.mgf --model model.ckpt --output out.npz

  Batch folder:
    python mgf_to_embedding.py --mgf_dir D:\\CasanovoV2\\SameSequence\\unmodified_with_prosit --model model.ckpt --output_dir D:\\CasanovoV2\\SameSequence\\npz_outputs
"""

import argparse
import os
import sys
import re
from pathlib import Path

import numpy as np


def _parse_precursor_mz(params: dict) -> float:
    """Parse PEPMASS from pyteomics MGF params."""
    pepmass = None
    if "pepmass" in params:
        pepmass = params.get("pepmass")
    elif "PEPMASS" in params:
        pepmass = params.get("PEPMASS")

    if pepmass is None:
        return 0.0

    if isinstance(pepmass, (list, tuple)) and len(pepmass) > 0:
        pepmass = pepmass[0]

    try:
        return float(pepmass)
    except (TypeError, ValueError):
        return 0.0


def _parse_charge(params: dict) -> int:
    """Parse CHARGE from pyteomics MGF params."""
    charge = None
    if "charge" in params:
        charge = params.get("charge")
    elif "CHARGE" in params:
        charge = params.get("CHARGE")

    if charge is None:
        return 1

    if isinstance(charge, (list, tuple)) and len(charge) > 0:
        charge = charge[0]

    if isinstance(charge, (int, np.integer)):
        c = int(charge)
        return c if c >= 1 else 1

    if isinstance(charge, str):
        m = re.search(r"(\d+)", charge)
        if m:
            c = int(m.group(1))
            return c if c >= 1 else 1

    return 1


def extract_embeddings_from_one_mgf(mgf_path: str, ric, model, config, device, limit=None):
    """Read one MGF and return arrays ready to save into NPZ."""
    from pyteomics import mgf

    embeddings = []
    titles = []
    precursor_mzs = []
    charges = []
    sources = []
    spectrum_ids = []
    invalid_reasons = []

    with mgf.MGF(mgf_path) as reader:
        for i, spec in enumerate(reader):
            if limit is not None and i >= limit:
                break

            params = spec.get("params", {})
            title = params.get("title") or params.get("TITLE") or f"index_{i}"
            precursor_mz = _parse_precursor_mz(params)
            precursor_charge = _parse_charge(params)
            source = params.get("source") or params.get("SOURCE") or "UNKNOWN"

            reason = ""
            try:
                emb = ric.extract_embedding(
                    spec.get("m/z array", np.array([], dtype=np.float32)),
                    spec.get("intensity array", np.array([], dtype=np.float32)),
                    precursor_mz, precursor_charge, model, config, device,
                )
                if not np.isfinite(emb).all() or np.linalg.norm(emb) == 0:
                    reason = "nonfinite_or_zero_embedding"
            except ValueError as exc:
                if str(exc) != "NO_USABLE_PEAKS":
                    raise
                reason = "no_usable_peaks"
                emb = np.full(int(config["dim_model"]), np.nan, dtype=np.float32)
            spectrum_ids.append(str(params.get("spectrum_id", title)))
            invalid_reasons.append(reason)

            embeddings.append(emb)
            titles.append(str(title))
            precursor_mzs.append(precursor_mz)
            charges.append(precursor_charge)
            sources.append(str(source))

    if len(embeddings) == 0:
        emb_arr = np.zeros((0, int(config.get("dim_model", 0) or 0)), dtype=np.float32)
    else:
        emb_arr = np.stack(embeddings, axis=0).astype(np.float32, copy=False)

    titles_arr = np.array(titles, dtype=np.str_)
    precursor_mz_arr = np.asarray(precursor_mzs, dtype=np.float32)
    charges_arr = np.asarray(charges, dtype=np.int32)
    sources_arr = np.array(sources, dtype=np.str_)

    return emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr, np.asarray(spectrum_ids), np.asarray(invalid_reasons)


def save_npz(output_path: str, emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr=None, spectrum_ids=None, invalid_reasons=None):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp = str(output_path) + ".partial.npz"
    np.savez_compressed(
        tmp,
        embeddings=emb_arr,
        titles=titles_arr,
        precursor_mz=precursor_mz_arr,
        charges=charges_arr,
        spectrum_ids=spectrum_ids if spectrum_ids is not None else titles_arr,
        invalid_reasons=invalid_reasons if invalid_reasons is not None else np.full(len(emb_arr), ""),
        sources=sources_arr if sources_arr is not None else np.array([], dtype=np.str_),
    )
    os.replace(tmp, output_path)


def process_single_file(mgf_path: str, output_path: str, ric, model, config, device, limit=None):
    emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr, spectrum_ids, invalid_reasons = extract_embeddings_from_one_mgf(
        mgf_path, ric, model, config, device, limit=limit
    )
    save_npz(output_path, emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr, spectrum_ids, invalid_reasons)

    print(f"Wrote: {output_path}")
    print(f"  embeddings: {emb_arr.shape} ({emb_arr.dtype})")
    print(f"  titles: {titles_arr.shape} ({titles_arr.dtype})")
    print(f"  precursor_mz: {precursor_mz_arr.shape} ({precursor_mz_arr.dtype})")
    print(f"  charges: {charges_arr.shape} ({charges_arr.dtype})")


def process_folder(mgf_dir: str, output_dir: str, ric, model, config, device, limit=None):
    mgf_dir_path = Path(mgf_dir)
    mgf_files = sorted(mgf_dir_path.glob("*.mgf"))

    if len(mgf_files) == 0:
        raise FileNotFoundError(f"No .mgf files found in folder: {mgf_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(mgf_files)} mgf files in: {mgf_dir}")

    failures = []
    for mgf_file in mgf_files:
        output_path = os.path.join(output_dir, f"{mgf_file.stem}.npz")
        try:
            process_single_file(
                str(mgf_file),
                output_path,
                ric,
                model,
                config,
                device,
                limit=limit,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {mgf_file.name}: {e}")
            failures.append(mgf_file.name)
    if failures:
        raise RuntimeError(f"Embedding failed for {len(failures)} files: {failures[:10]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MGF(s) to spectrum embeddings (NPZ)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mgf", help="Input single MGF path")
    group.add_argument("--mgf_dir", help="Input folder containing .mgf files")

    parser.add_argument("--model", required=True, help="Spec2Pep checkpoint path")
    parser.add_argument("--output", help="Output .npz path for single-file mode")
    parser.add_argument("--output_dir", help="Output folder for batch mode")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N spectra per file (debug)")
    parser.add_argument("--strict-preprocessing", action="store_true")
    args = parser.parse_args()

    # Import pyteomics lazily so `-h` works even if deps are missing.
    try:
        import pyteomics  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pyteomics is required to read MGF files. Install it first, e.g.: pip install pyteomics"
        ) from e

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    if args.mgf is not None:
        if not os.path.exists(args.mgf):
            raise FileNotFoundError(f"MGF file not found: {args.mgf}")
        if not args.output:
            raise ValueError("Single-file mode requires --output")
    else:
        if not os.path.isdir(args.mgf_dir):
            raise FileNotFoundError(f"MGF folder not found: {args.mgf_dir}")
        if not args.output_dir:
            raise ValueError("Batch mode requires --output_dir")

    # Ensure we can import local modules when invoked from other working dirs.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # Reuse existing, known-good logic.
    import encoder_only as ric

    print("Loading model...")
    model, config, device = ric.load_model(args.model)
    if args.strict_preprocessing:
        import piprime_utils
        if piprime_utils.sus is None:
            raise RuntimeError("spectrum_utils is required; fallback is forbidden")
        config["strict_preprocessing"] = True
    import json, importlib.metadata
    versions = {}
    for package in ("numpy", "torch", "pyteomics", "spectrum_utils", "pytorch-lightning"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    provenance_dir = Path(args.output_dir) if args.output_dir else Path(args.output).parent
    provenance_dir.mkdir(parents=True, exist_ok=True)
    config_keys = ("min_mz", "max_mz", "remove_precursor_tol", "min_intensity", "n_peaks", "dim_model", "strict_preprocessing")
    (provenance_dir / "embedding_provenance.json").write_text(json.dumps(dict(versions=versions, config={k:config.get(k) for k in config_keys}, model=str(Path(args.model).resolve()), device=str(device)), indent=2), encoding="utf-8")
    print("Model loaded.")

    if args.mgf is not None:
        process_single_file(args.mgf, args.output, ric, model, config, device, limit=args.limit)
    else:
        process_folder(args.mgf_dir, args.output_dir, ric, model, config, device, limit=args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())