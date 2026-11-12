#!/usr/bin/env python3
"""Diagnose one spectrum's blind-library scoring for a spectral predictor.

Loads the MGF spectrum at --ordinal, the per-charge library npz for the chosen
predictor, scores the GT entry vs the whole same-charge library, and reports:
  * GT entry d/f/m scores and its RANK over the library (d / f separately),
  * how many TARGET entries and DECOY entries beat the GT,
  * fraction of GT predicted peaks (>= rel floor) matched by an experimental peak,
  * the argmax entry by d and by f (seq/label/backbone).

Use to decide whether a bad blind top-1 is a representation/axis bug (GT scores
low or beaten by random decoys) or genuine discrimination limit.

Usage:
    python prosit_blind_diag.py --species bacillus --work . \
        --root /home/wangyuhang/seq2mass/v1-search \
        --predictor prosit --ordinal 0 [--rel 0.05] [--both]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fullsearch_common import dc, read_library_tsv, TOL, backbone_of
from step4_blind_score import load_lib, per_rel_arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--predictor", default="prosit", choices=["prosit", "v20"])
    p.add_argument("--ordinal", type=int, default=0)
    p.add_argument("--rel", type=float, default=0.05)
    p.add_argument("--max-gt-len", type=int, default=None,
                   help="30 for prosit (exclude GT > 30aa)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.work)
    ddir = work / "data" / args.species
    rows = read_library_tsv(ddir / "library_annotated.tsv.gz")
    seqs = {r["idx"]: r for r in rows}
    _fasta, mgf = dc.resolve_species_files(Path(args.root), args.species)

    ordinal = None
    for i, (meta, peaks) in enumerate(dc.iter_mgf_spectra(mgf)):
        if i == args.ordinal:
            ordinal = (meta, peaks)
            break
    if ordinal is None:
        print(f"ordinal {args.ordinal} out of range")
        return 1
    meta, peaks = ordinal
    gt = meta.get("SEQ", "")
    charge = dc.parse_charge_from_meta(meta)
    if not gt or charge <= 0:
        print("no GT / charge")
        return 1
    gtb = backbone_of(gt)
    print(f"ordinal={args.ordinal} charge={charge} gt_len={len(gtb)} gt={gt}")

    lib_path = ddir / f"lib_{args.predictor}_c{charge}.npz"
    if not lib_path.exists():
        print(f"no library {lib_path}; run step3_predict_{args.predictor} first")
        return 1
    lib = load_lib(ddir, args.predictor, charge)
    E = lib["n_entries"]

    # GT entry: first target row matching the GT (mass / I-L equivalent)
    gt_idx = None
    for r in rows:
        if r["label"] == "T" and dc.peptide_matches(r["annoseq"], gt):
            gt_idx = r["idx"]
            break
    if gt_idx is None:
        print(f"GT not found among target entries (len>30? backbone={len(gtb)})")
        return 1

    exp = dc.transform_exp_intensity(peaks, method="sqrt_clip")
    emz = np.sort(exp[:, 0])
    eint = exp[np.argsort(exp[:, 0]), 1].astype(np.float64)
    d, f, m = per_rel_arrays(emz, eint, lib, args.rel)

    starts = lib["starts"]
    s0, s1 = starts[gt_idx], starts[gt_idx + 1]
    tmz_seg = lib["tmz"][s0:s1]
    tint_seg = lib["tint"][s0:s1]
    pmax = float(tint_seg.max()) if tint_seg.size else 0.0
    thr = max(0.01, args.rel * pmax)
    kept = tint_seg > thr
    if kept.any():
        lo = np.searchsorted(emz, tmz_seg[kept] - TOL, side="left")
        hi = np.searchsorted(emz, tmz_seg[kept] + TOL, side="right")
        frac = float((hi > lo).mean())
    else:
        frac = float("nan")

    def ranks(score, gt_val):
        return int(np.sum(score > gt_val))

    d_gt = float(d[gt_idx]); f_gt = float(f[gt_idx]); m_gt = float(m[gt_idx])
    nd_beats = int(np.sum(d > d_gt)); nf_beats = int(np.sum(f > f_gt))
    nd_dec = int(np.sum((lib["label"] == 0) & (d > d_gt)))
    nf_dec = int(np.sum((lib["label"] == 0) & (f > f_gt)))
    nd_tgt = int(np.sum((lib["label"] == 1) & (d > d_gt)))
    nf_tgt = int(np.sum((lib["label"] == 1) & (f > f_gt)))

    top_d = int(np.argmax(d)); top_f = int(np.argmax(f))
    print(f"\nlib entries(E)={E} peaks/entry(GT)={tint_seg.size} "
          f"kept@rel{args.rel}={int(kept.sum())} matched_frac={frac:.3f}")
    print(f"GT={seqs[gt_idx]['annoseq']}  label T")
    print(f"  d={d_gt:.4f}  f={f_gt:.4f}  m={m_gt:.4f}")
    print(f"  rank: #entries with d>gt_d = {nd_beats}, f>gt_f = {nf_beats}")
    print(f"  beating GT: targets d>{d_gt}:{nd_tgt} decoys d>{d_gt}:{nd_dec} | "
          f"targets f>{f_gt}:{nf_tgt} decoys f>{f_gt}:{nf_dec}")
    for name, arr in (("d", d), ("f", f)):
        bi = int(np.argmax(arr))
        r = seqs[bi]
        print(f"  argmax {name}: idx={bi} label={r['label']} "
              f"backbone={r['backbone'][:22]}... annoseq={r['annoseq'][:40]} "
              f"score={float(arr[bi]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
