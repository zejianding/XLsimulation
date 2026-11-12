#!/usr/bin/env python3
"""Step 3 (Prosit) — predict a library spectrum for every annotated entry.

Same .npz layout as step3_predict_v20.py (lib_prosit_c<z>.npz).  Koina
predictions are per (plain backbone, charge); annotated mod variants of the same
backbone share one prediction (reanchor), so only unique <=30aa backbones are
predicted.  Entries whose backbone is >30aa get empty spectra (excluded by the
scorer).  MUST be run as its own process: pandas is imported before ds_common /
predict_mgf (numpy-alias hazard).

Usage:
    python step3_predict_prosit.py --species mouse --work <fullsearch> \
        [--charges 2,3] [--batch-size 1500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  must precede ds_common/predict_mgf

from fullsearch_common import (dc, read_library_tsv, TOL, SPEC_MIN,
                               MAX_BACKBONE_PRO)
from step4_prosit_score import (PrositPredictor, build_candidate_peaks,
                                PROSIT_MAX_LEN)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--charges", default=None)
    p.add_argument("--cache", default=None,
                   help="Koina cache path (default <work>/data/<sp>/prosit_lib_cache.tsv.gz)")
    p.add_argument("--batch-size", type=int, default=1500)
    p.add_argument("--min-rel", type=float, default=0.0,
                   help="store predicted peaks above min-rel * entry pmax (Prosit "
                        "intensities are spiky, so default 0.0 = keep all b/y up to "
                        "the top-peaks cap; rel filtering happens at scoring time)")
    p.add_argument("--top-peaks", type=int, default=400,
                   help="safety cap on peaks per entry after the rel floor")
    p.add_argument("--exp-transform", default="sqrt_clip",
                   choices=["none", "sqrt_clip", "sqrt"])  # symmetry only
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.work)
    ddir = work / "data" / args.species
    odir = work / "output" / args.species / "records"
    meta = json.loads((ddir / "scans_meta.json").read_text(encoding="utf-8"))
    charges = [int(x) for x in args.charges.split(",")] if args.charges else meta["charges"]
    rows = read_library_tsv(ddir / "library_annotated.tsv.gz")
    n_rows = len(rows)

    cache = Path(args.cache) if args.cache else ddir / "prosit_lib_cache.tsv.gz"
    pred = PrositPredictor(cache_path=cache, batch_size=args.batch_size,
                           mods_mode="reanchor")
    backbones = sorted({r["backbone"] for r in rows if len(r["backbone"]) <= PROSIT_MAX_LEN})
    print(f"[prositlib] species={args.species} charges={charges} "
          f"unique_backbones_lt30={len(backbones)} / n_entries={n_rows}")
    needed = {(bb, c) for c in charges for bb in backbones}
    pred.predict_missing(needed)

    label = np.array([1 if r["label"] == "T" else 0 for r in rows], dtype=np.int8)
    blen = np.array([len(r["backbone"]) for r in rows], dtype=np.int16)
    eids = np.arange(n_rows, dtype=np.int32)

    for c in charges:
        tmz_l, tint_l, starts_l, pmax_l = [], [], [0], []
        for r in rows:
            bb = r["backbone"]
            if len(bb) > PROSIT_MAX_LEN:
                starts_l.append(starts_l[-1])
                pmax_l.append(0.0)
                continue
            peaks, annos = pred.get(bb, c)
            theo = build_candidate_peaks(peaks, annos, r["annoseq"], "reanchor",
                                         tol=TOL)
            if theo.size:
                # sqrt-transform Prosit predicted intensities to the same
                # intensity domain as the sqrt_clip experimental peaks (V20
                # predicted peaks are already in that domain).  Raw Prosit
                # intensities are heavily skewed (one dominant peak), which makes
                # the rel-<x>*pmax filter far too aggressive otherwise.
                theo[:, 1] = np.sqrt(theo[:, 1])
                # keep ALL positive b/y peaks (no absolute floor): scoring-time
                # rel filtering is handled in step4.
                thr = args.min_rel * float(theo[:, 1].max())
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
            ddir / f"lib_prosit_c{c}.npz",
            tmz=tmz, tint=tint,
            starts=np.asarray(starts_l, dtype=np.int64),
            entry_ids=eids, label=label, pmax=np.asarray(pmax_l, dtype=np.float32),
            blen=blen)
        print(f"[prositlib] charge {c}: entries={n_rows} peaks={len(tmz)}")
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"record_step3_prosit_{args.species}.json").write_text(json.dumps(
        {"species": args.species, "charges": charges, "n_entries": n_rows,
         "n_backbone_predicted": len(backbones)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
