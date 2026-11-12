#!/usr/bin/env python3
"""Read and inspect a best_grids.npz produced by step4 --save-predictions.

For a chosen spectrum it prints the stored metadata (seq/charge/pepmass/score),
grid statistics, the reconstructed m/z meaning of the top predicted cells
(input_mz grid rebuilt from seq+charge), and the predicted peak list (the same
peaks written to predicted_spectra.mgf).

Usage:
    python read_predictions.py output/mouse/ppm10/predictions/b_grid_rel005/best_grids.npz [index]
    python read_predictions.py <npz>            # spectrum index 0
    python read_predictions.py <npz> 12 --top 5 # spectrum 12, show 5 cells/peaks
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import ds_common as dc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", help="path to best_grids.npz")
    p.add_argument("index", type=int, nargs="?", default=0,
                   help="spectrum index (0-based, default 0)")
    p.add_argument("--top", type=int, default=10,
                   help="number of top predicted cells / peaks to show")
    p.add_argument("--min-int", type=float, default=0.01,
                   help="min intensity for the predicted-peak list (default 0.01)")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    n = int(d["grids"].shape[0])
    print(f"file: {args.npz}")
    print(f"spectra: {n}")
    print("keys / shapes / dtypes:")
    for k in d.keys():
        a = d[k]
        print(f"  {k:<8} shape={str(a.shape):<22} dtype={a.dtype}")

    idx = args.index
    if not (0 <= idx < n):
        print(f"\nindex {idx} out of range [0, {n})")
        return 1

    seq = str(d["seqs"][idx])
    charge = int(d["charges"][idx])
    pepmass = float(d["pepmasses"][idx])
    rt = float(d["rts"][idx])
    score = float(d["scores"][idx])
    grid = np.asarray(d["grids"][idx], dtype=np.float32)

    print(f"\n=== spectrum index {idx} (ordinal {d['ordinals'][idx]}) ===")
    print(f"  scan_id : {d['scan_ids'][idx]}")
    if "ranks" in d.files:
        print(f"  rank    : {d['ranks'][idx]}   (candidate rank in this branch)")
    print(f"  seq     : {seq}   (candidate)")
    if "gt_seqs" in d.files:
        print(f"  gt_seq  : {d['gt_seqs'][idx]}   (source MGF SEQ=)")
    print(f"  charge  : {charge}+   pepmass: {pepmass:.5f}   rt: {rt:.1f}")
    print(f"  score   : {score:.6f}")
    print(f"  grid    : shape={grid.shape} dtype={d['grids'].dtype}  "
          f"nonzero={int((grid > 0).sum())}/{grid.size}  max={grid.max():.6f}")

    # Rebuild the m/z interpretation (same as during scoring).
    input_mz = dc.build_input_mz_matrix(seq, min_mz=100.0)
    valid_mask, cleavage_count = dc.build_valid_mask(seq, charge, min_mz=100.0)

    print(f"\n  top {args.top} predicted cells (pos, token, channel -> m/z, intensity):")
    flat_idx = np.argsort(-grid.ravel())[: args.top]
    for fi in flat_idx:
        pos, tok, ch = np.unravel_index(fi, grid.shape)
        mz = float(input_mz[pos, tok, ch])
        print(f"    pos={pos:2d} tok={tok:2d} ch={ch}  m/z={mz:9.3f}  I={float(grid.flat[fi]):.5f}")

    peaks = dc.prediction_to_mgf_peaks(
        input_mz, grid, valid_mask,
        min_intensity=args.min_int, min_mz=100.0, mz_merge_tol=0.05,
    )
    print(f"\n  predicted peaks (min_int={args.min_int}): {len(peaks)}")
    for mz, inten in peaks[: args.top]:
        print(f"    m/z={float(mz):9.4f}  I={float(inten):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
