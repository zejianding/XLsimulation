#!/usr/bin/env python3
"""Generate spectrum embeddings for every spectrum in an MGF.

This is a standalone utility that reuses the existing PiPrime/PrimeNovo
embedding extraction logic used by run_iterative_correction.py, but without
requiring any TSV predictions or running the correction workflow.

Output format: NPZ (np.savez_compressed)
  - embeddings: float32 array, shape [N, D]
  - titles: unicode str array, shape [N]
  - precursor_mz: float32 array, shape [N]
  - charges: int32 array, shape [N]

Example:
  python mgf_to_embedding.py --mgf GITIATSHVEYESDTR.mgf --model model.ckpt --output out.npz --limit 10
"""

import argparse
import os
import sys
import re

import numpy as np



def _parse_precursor_mz(params: dict) -> float:
    """Parse PEPMASS from pyteomics MGF params.

    pyteomics typically exposes it as params['pepmass'].
    Value can be a float, or (mz, intensity).
    """
    pepmass = None
    if 'pepmass' in params:
        pepmass = params.get('pepmass')
    elif 'PEPMASS' in params:
        pepmass = params.get('PEPMASS')

    if pepmass is None:
        return 0.0

    if isinstance(pepmass, (list, tuple)) and len(pepmass) > 0:
        pepmass = pepmass[0]

    try:
        return float(pepmass)
    except (TypeError, ValueError):
        return 0.0


def _parse_charge(params: dict) -> int:
    """Parse CHARGE from pyteomics MGF params.

    Common formats:
      - int
      - [2]
      - "2+"
      - "2+ and 3+"
    """
    charge = None
    if 'charge' in params:
        charge = params.get('charge')
    elif 'CHARGE' in params:
        charge = params.get('CHARGE')

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert MGF to spectrum embeddings (NPZ)")
    parser.add_argument("--mgf", required=True, help="Input MGF path")
    parser.add_argument("--model", required=True, help="Spec2Pep checkpoint path")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N spectra (debug)")
    args = parser.parse_args()

    # Import pyteomics lazily so `-h` works even if deps are missing.
    try:
        from pyteomics import mgf
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pyteomics is required to read MGF files. Install it first, e.g.: pip install pyteomics"
        ) from e

    if not os.path.exists(args.mgf):
        raise FileNotFoundError(f"MGF file not found: {args.mgf}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    # Ensure we can import local modules when invoked from other working dirs.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # Reuse existing, known-good logic.
    import run_iterative_correction as ric

    model, config, device = ric.load_model(args.model)

    embeddings = []
    titles = []
    precursor_mzs = []
    charges = []

    with mgf.MGF(args.mgf) as reader:
        for i, spec in enumerate(reader):
            if args.limit is not None and i >= args.limit:
                break

            params = spec.get('params', {})
            title = params.get('title') or params.get('TITLE') or f"index_{i}"
            precursor_mz = _parse_precursor_mz(params)
            precursor_charge = _parse_charge(params)

            emb = ric.extract_embedding(
                spec.get('m/z array', np.array([], dtype=np.float32)),
                spec.get('intensity array', np.array([], dtype=np.float32)),
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

    if len(embeddings) == 0:
        emb_arr = np.zeros((0, int(config.get('dim_model', 0) or 0)), dtype=np.float32)
    else:
        emb_arr = np.stack(embeddings, axis=0).astype(np.float32, copy=False)

    titles_arr = np.array(titles, dtype=np.str_)
    precursor_mz_arr = np.asarray(precursor_mzs, dtype=np.float32)
    charges_arr = np.asarray(charges, dtype=np.int32)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.output,
        embeddings=emb_arr,
        titles=titles_arr,
        precursor_mz=precursor_mz_arr,
        charges=charges_arr,
    )

    print(f"Wrote: {args.output}")
    print(f"  embeddings: {emb_arr.shape} ({emb_arr.dtype})")
    print(f"  titles: {titles_arr.shape} ({titles_arr.dtype})")
    print(f"  precursor_mz: {precursor_mz_arr.shape} ({precursor_mz_arr.dtype})")
    print(f"  charges: {charges_arr.shape} ({charges_arr.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
