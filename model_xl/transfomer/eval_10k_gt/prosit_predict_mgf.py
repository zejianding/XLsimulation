#!/usr/bin/env python3
"""Prosit side of the 10k-peptide comparison: peptide+charge -> predicted MGF.

Uses the SAME (sequence, charge) list as the V20 run (v20_10k_9species.npz),
predicts with Koina `Prosit_2025_intensity_MultiFrag`, and writes an MGF with the
SAME format as `predict_npz_to_mgf.py` (BEGIN IONS / TITLE / PEPMASS / CHARGE= /
SEQ=<annotated> / mz intensity), so the existing downstream steps
(collect_original_with_prediction.py -> PiPrime embeddings -> batch_cosine_summary)
run unchanged.

Modified peptides: Prosit only takes unmodified backbones, so we predict the
backbone and place each annotated b/y fragment at its MOD-corrected theoretical
m/z (the `reanchor` convention already used by ds/step4_prosit_score.py).

Usage:
    python prosit_predict_mgf.py --npz ../../datasets/v20_10k_9species.npz \
        --out <run_dir>/predicted.mgf [--batch-size 1500] [--cache <file>]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_TF = Path(__file__).resolve().parents[1]
_DS = _TF / "ds"
for p in (_TF, _DS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import pandas as pd  # noqa: E402  (before ds_common/predict_mgf on some numpy builds)
# Import the optional client before predict_mgf's legacy numpy._core aliases.
# Recent Triton clients import ml_dtypes, whose native extension must see NumPy's
# original module layout. Offline export remains available without koinapy.
if importlib.util.find_spec("koinapy") is not None:
    import koinapy  # noqa: F401,E402
import predict_mgf as pm  # noqa: E402
import step4_prosit_score as p4  # noqa: E402  (Koina predictor + reanchor helpers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--npz", required=True)
    p.add_argument("--out", required=True, help="output predicted MGF")
    p.add_argument("--batch-size", type=int, default=1500)
    p.add_argument("--cache", default=None, help="Koina cache tsv.gz (resumable)")
    p.add_argument("--min-mz", type=float, default=0.0)
    p.add_argument("--skipped-out", help="CSV with sequence, charge, reason, detail")
    p.add_argument("--by-only", action="store_true",
                   help="stop at the b/y filter (API-returned m/z, no reanchor, no merge). "
                        "Default = OFF: b/y intensities are reanchored to the MOD-corrected "
                        "theoretical m/z (same convention as the FDR/no-FDR pipelines)")
    p.add_argument("--raw-api", action="store_true",
                   help="write ALL Koina rows verbatim (original mz, no b/y filter, "
                        "no reanchor, no merge) instead of the b/y reanchored export")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw = dict(np.load(args.npz, allow_pickle=True))
    seqs = [str(s) for s in raw["seqs"]]
    charges = [int(c) for c in raw["charges"]]
    if len(seqs) != len(charges) or len(set(zip(seqs, charges))) != len(seqs):
        raise ValueError("Expected unique sequence+charge rows of equal length")
    raw_api = bool(args.raw_api)
    pred = p4.PrositPredictor(cache_path=Path(args.cache) if args.cache else None,
                              batch_size=args.batch_size, mods_mode="reanchor")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1) batch-predict the unique (backbone, charge) set first (keeps the Koina
    #    cache as a few segment files instead of one per peptide)
    needed = {(p4.strip_mods(s), c) for s, c in zip(seqs, charges)
              if len(p4.strip_mods(s)) <= p4.PROSIT_MAX_LEN}
    print(f"[prosit] unique (backbone,charge) = {len(needed)}")
    pred.predict_missing(needed)

    # 2) stream the MGF group by group (bounded memory: one peptide at a time)
    n_ok = n_empty = 0
    skipped = []
    tmp = Path(str(args.out) + ".partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        for seq, chg in zip(seqs, charges):
            backbone = p4.strip_mods(seq)
            if len(backbone) > p4.PROSIT_MAX_LEN:
                skipped.append((seq, chg, "backbone_too_long", str(len(backbone))))
                n_empty += 1
                continue
            peaks, annos = pred.get(backbone, chg)
            if peaks.size == 0:
                reason = "request_failed" if (backbone, chg) in pred.failures else "empty_prediction"
                skipped.append((seq, chg, reason, pred.failures.get((backbone, chg), "")))
                n_empty += 1
                continue
            if raw_api:
                theo = peaks            # verbatim Koina rows (all ion types, API m/z)
            elif args.by_only:
                keep = [i for i, a in enumerate(annos) if p4.parse_annotation(a) is not None]                     if annos is not None else list(range(len(peaks)))
                theo = peaks[keep] if keep else np.empty((0, 2), dtype=np.float32)
            else:
                # DEFAULT: b/y reanchored to the MOD-corrected theoretical m/z
                # (identical convention to ds/step4_prosit_score.py used in FDR/no-FDR)
                theo = p4.build_candidate_peaks(peaks, annos, seq, "reanchor")
            if theo.size:
                theo = theo[np.isfinite(theo).all(axis=1) & (theo[:, 0] >= args.min_mz) & (theo[:, 1] > 0)]
            if theo.size == 0:
                skipped.append((seq, chg, "empty_after_reanchor_or_peak_filter", ""))
                n_empty += 1
                continue
            mass = pm.calculate_pepmass(seq, chg)
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"Invalid precursor mass: {seq}/{chg}")
            fh.write("BEGIN IONS\n")
            fh.write(f"TITLE={seq}_{chg}\n")
            fh.write(f"PEPMASS={mass:.6f}\n")
            fh.write(f"CHARGE={chg}+\n")
            fh.write("RTINSECONDS=0\n")
            fh.write(f"SEQ={seq}\n")
            for mz, it in theo:
                fh.write(f"{float(mz):.6f} {float(it):.10f}\n")
            fh.write("END IONS\n\n")
            n_ok += 1
    pred.save_cache()
    skipped_path = Path(args.skipped_out or str(args.out) + ".skipped.csv")
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_tmp = skipped_path.with_name(skipped_path.name + ".partial")
    with skipped_tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sequence", "charge", "reason", "detail"]); w.writerows(skipped)
    skipped_tmp.replace(skipped_path)
    tmp.replace(args.out)
    print(f"prosit predicted {n_ok} peptides ({n_empty} empty/skipped) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
