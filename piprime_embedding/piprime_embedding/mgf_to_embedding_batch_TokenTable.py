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
def _parse_source(params: dict) -> str:
    """Parse SOURCE from pyteomics MGF params."""
    source = None
    if "source" in params:
        source = params.get("source")
    elif "SOURCE" in params:
        source = params.get("SOURCE")

    if source is None:
        return "UNKNOWN"

    return str(source)

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

    with mgf.MGF(mgf_path) as reader:
        for i, spec in enumerate(reader):
            if limit is not None and i >= limit:
                break

            params = spec.get("params", {})
            title = params.get("title") or params.get("TITLE") or f"index_{i}"
            precursor_mz = _parse_precursor_mz(params)
            precursor_charge = _parse_charge(params)
            source = _parse_source(params)

            emb = ric.extract_embedding(
                spec.get("m/z array", np.array([], dtype=np.float32)),
                spec.get("intensity array", np.array([], dtype=np.float32)),
                precursor_mz,
                precursor_charge,
                model,
                config,
                device,
            )

            embeddings.append(emb)
            titles.append(str(title))
            precursor_mzs.append(precursor_mz)
            charges.append(precursor_charge)
            sources.append(source)

    if len(embeddings) == 0:
        emb_arr = np.zeros((0, int(config.get("dim_model", 0) or 0)), dtype=np.float32)
    else:
        emb_arr = np.stack(embeddings, axis=0).astype(np.float32, copy=False)

    titles_arr = np.array(titles, dtype=np.str_)
    precursor_mz_arr = np.asarray(precursor_mzs, dtype=np.float32)
    charges_arr = np.asarray(charges, dtype=np.int32)
    sources_arr = np.array(sources, dtype=np.str_)

    return emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr

def save_npz(output_path: str, emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=emb_arr,
        titles=titles_arr,
        precursor_mz=precursor_mz_arr,
        charges=charges_arr,
        sources=sources_arr,
    )


def process_single_file(mgf_path: str, output_path: str, ric, model, config, device, limit=None):
    emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr = extract_embeddings_from_one_mgf(
        mgf_path, ric, model, config, device, limit=limit
    )
    save_npz(output_path, emb_arr, titles_arr, precursor_mz_arr, charges_arr, sources_arr)

    print(f"Wrote: {output_path}")
    print(f"  embeddings: {emb_arr.shape} ({emb_arr.dtype})")
    print(f"  titles: {titles_arr.shape} ({titles_arr.dtype})")
    print(f"  precursor_mz: {precursor_mz_arr.shape} ({precursor_mz_arr.dtype})")
    print(f"  charges: {charges_arr.shape} ({charges_arr.dtype})")
    print(f"  sources: {sources_arr.shape} ({sources_arr.dtype})")


def process_folder(mgf_dir: str, output_dir: str, ric, model, config, device, limit=None):
    mgf_dir_path = Path(mgf_dir)
    mgf_files = sorted(mgf_dir_path.glob("*.mgf"))

    if len(mgf_files) == 0:
        raise FileNotFoundError(f"No .mgf files found in folder: {mgf_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(mgf_files)} mgf files in: {mgf_dir}")

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MGF(s) to spectrum embeddings (NPZ)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mgf", help="Input single MGF path")
    group.add_argument("--mgf_dir", help="Input folder containing .mgf files")

    parser.add_argument("--model", required=True, help="Spec2Pep checkpoint path")
    parser.add_argument("--output", help="Output .npz path for single-file mode")
    parser.add_argument("--output_dir", help="Output folder for batch mode")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N spectra per file (debug)")
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
    import run_iterative_correction as ric

    print("Loading model...")
    model, config, device = ric.load_model(args.model)
    print("Model loaded.")

    if args.mgf is not None:
        process_single_file(args.mgf, args.output, ric, model, config, device, limit=args.limit)
    else:
        process_folder(args.mgf_dir, args.output_dir, ric, model, config, device, limit=args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())