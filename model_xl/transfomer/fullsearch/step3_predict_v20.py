#!/usr/bin/env python3
"""Step 3 (V20) — predict a library spectrum for every annotated library entry.

For each charge level present in the species, batch-predict each annotated
target/decoy variant with the local V20 model and store the merged theoretical
peak list per entry as per-charge .npz:

  lib_v20_c<z>.npz   tmz,tint (concatenated peaks), starts (E+1), entry_ids (E),
                     label (E, 1=target), pmax (E), blen (E)

Usage:
    python step3_predict_v20.py --species mouse --work <fullsearch> \
        [--checkpoint ...] [--charges 2,3] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fullsearch_common import (dc, fc, read_library_tsv, SPEC_MIN, MZ_MERGE,
                               DEFAULT_RT, TOL)
from predict_mgf import prediction_to_mgf_peaks, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--checkpoint", default=fc.SERVER_CKPT)
    p.add_argument("--charges", default=None, help="comma-separated (default from scans_meta)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-mz", type=float, default=dc.DEFAULT_MIN_MZ)
    p.add_argument("--min-rel", type=float, default=0.05,
                   help="store only predicted peaks above min-rel * entry pmax "
                        "(== the lowest rel filter the scorer will apply; no info lost)")
    p.add_argument("--top-peaks", type=int, default=400,
                   help="safety cap on peaks per entry after the rel floor")
    p.add_argument("--exp-transform", default="sqrt_clip",
                   choices=["none", "sqrt_clip", "sqrt"])  # accepted for symmetry
    return p.parse_args()


def main() -> int:
    import torch
    args = parse_args()
    work = Path(args.work)
    ddir = work / "data" / args.species
    odir = work / "output" / args.species / "records"
    meta = json.loads((ddir / "scans_meta.json").read_text(encoding="utf-8"))
    charges = [int(x) for x in args.charges.split(",")] if args.charges else meta["charges"]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    model, model_config, stats, _ = load_checkpoint(Path(args.checkpoint), device,
                                                    no_packed=False)
    print(f"[v20lib] species={args.species} device={device} charges={charges}")

    rows = read_library_tsv(ddir / "library_annotated.tsv.gz")
    annoseqs = [r["annoseq"] for r in rows]
    label = np.array([1 if r["label"] == "T" else 0 for r in rows], dtype=np.int8)
    blen = np.array([len(r["backbone"]) for r in rows], dtype=np.int16)
    eids = np.arange(len(rows), dtype=np.int32)

    for c in charges:
        pred, inmz, vm = fc.predict_candidate_grids(
            model, model_config, stats, device, annoseqs, c, DEFAULT_RT,
            args.min_mz, args.batch_size)
        tmz_l, tint_l, starts_l, pmax_l = [], [], [0], []
        for i in range(len(rows)):
            theo = prediction_to_mgf_peaks(inmz[i], pred[i], vm[i],
                                           min_intensity=SPEC_MIN,
                                           min_mz=args.min_mz,
                                           mz_merge_tol=MZ_MERGE)
            if theo.size:
                thr = max(SPEC_MIN, args.min_rel * float(theo[:, 1].max()))
                theo = theo[theo[:, 1] > thr]
                if theo.size and theo.shape[0] > args.top_peaks:
                    sel = np.argsort(theo[:, 1])[-args.top_peaks:]
                    theo = theo[sel]
                if theo.size:
                    theo = theo[np.argsort(theo[:, 0])]
            if theo.size:
                tmz_l.append(theo[:, 0]); tint_l.append(theo[:, 1])
                pmax_l.append(float(theo[:, 1].max()))
            else:
                pmax_l.append(0.0)
            starts_l.append(starts_l[-1] + (theo.shape[0] if theo.size else 0))
        tmz = np.concatenate(tmz_l).astype(np.float32) if tmz_l else np.empty(0, np.float32)
        tint = np.concatenate(tint_l).astype(np.float32) if tint_l else np.empty(0, np.float32)
        np.savez_compressed(
            ddir / f"lib_v20_c{c}.npz",
            tmz=tmz, tint=tint,
            starts=np.asarray(starts_l, dtype=np.int64),
            entry_ids=eids, label=label, pmax=np.asarray(pmax_l, dtype=np.float32),
            blen=blen)
        print(f"[v20lib] charge {c}: entries={len(rows)} peaks={len(tmz)}")
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"record_step3_v20_{args.species}.json").write_text(json.dumps(
        {"species": args.species, "charges": charges, "n_entries": len(rows)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
