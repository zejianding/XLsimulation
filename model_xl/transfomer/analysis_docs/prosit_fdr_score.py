#!/usr/bin/env python3
"""Prosit 1% FDR scoring on the fdr target+decoy candidate pool.

Consumes fdr step2 output (candidates_<sp>_td.tsv.gz, per-spectrum T/D annotated
variants, precursor-mass matched).  Per MGF spectrum every <=30aa candidate is
scored with Prosit (reanchor + sqrt intensity alignment, special handling) using
the same branch family as V20 fdr: d/f/m_by per rel level (rel 0 = no
predicted-side intensity filter) + hsdf / hsdf_mby (min-max) + dfexp / dfexp_mby
(raw mean).  For each branch the argmax candidate's T/D label and fused score are
written per spectrum — the input for the generic (D+1)/T counter (fdr step4).

Length cap 30 AA: candidates with backbone >30 are dropped; spectra with an empty
<=30 pool are skipped -> Prosit-FDR is intrinsically <=30.  Engine / V20 FDR
results need their own <=30 pass to compare on the same footing.

Usage:
    python prosit_fdr_score.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --candidates <fdr>/data/mouse/candidates_mouse_td.tsv.gz \
        --out-dir <fdr>/output/mouse [--cache <path>]
    # counting (reuse fdr step4):
    python step4_fdr_curve.py --species mouse --out-dir <out-dir> \
        --results <out-dir>/results_prositfdr_mouse.csv --fdr-targets 0.01,0.05,0.10
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  must precede ds_common/predict_mgf

_FDR = Path(__file__).resolve().parent
_TRANSFORMER = _FDR.parent
for _p in (_TRANSFORMER, _TRANSFORMER / "scripts", _TRANSFORMER / "ds",
           _TRANSFORMER / "fdr", _TRANSFORMER / "fdr_full"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ds_common as dc  # noqa: E402
import step4_prosit_score as p4  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--candidates", required=True, help="fdr step2 TD TSV.gz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache", default=None, help="Koina cache path")
    p.add_argument("--rel-intensities", default="0,0.05,0.10,0.20")
    p.add_argument("--exp-transform", default="sqrt_clip",
                   choices=["none", "sqrt_clip", "sqrt"])
    p.add_argument("--batch-size", type=int, default=1500)
    p.add_argument("--max-scans", type=int, default=None)
    return p.parse_args()


def load_td(tsv_gz: Path) -> dict[int, list[tuple[str, str, float]]]:
    """ordinal -> [(seq, label, ppm)] (row[5]=seq, row[6]=label, row[8]=ppm)."""
    groups: dict[int, list] = {}
    with gzip.open(tsv_gz, "rt", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd, None)
        for row in rd:
            if not row:
                continue
            groups.setdefault(int(row[0]), []).append((row[5], row[6], float(row[8])))
    return groups


def _score_avg(arrays: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(arrays[0].shape[0], dtype=np.float64)
    for s in arrays:
        lo, hi = float(s.min()), float(s.max())
        if hi - lo > 1e-12:
            acc += (s - lo) / (hi - lo)
    return acc / len(arrays)


def _raw_avg(arrays: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(arrays[0].shape[0], dtype=np.float64)
    for s in arrays:
        acc += s
    return acc / len(arrays)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = [float(x) for x in args.rel_intensities.split(",") if x.strip()]
    branches, _cols, _rel_for = p4.build_branches(rels)
    base_by_rel: dict[str, list[str]] = {}
    for r in rels:
        lvl = f"{int(round(r * 100)):03d}"
        base_by_rel[lvl] = [f"d_spec_rel{lvl}", f"f_frag_rel{lvl}",
                            f"m_by_prec_rel{lvl}"]

    cand_groups = load_td(Path(args.candidates))
    pred = p4.PrositPredictor(cache_path=Path(args.cache) if args.cache else None,
                              batch_size=args.batch_size, mods_mode="reanchor")
    _fasta, mgf = dc.resolve_species_files(Path(args.root), args.species)
    total = args.max_scans or dc.count_mgf_spectra(mgf)
    results_csv = out_dir / f"results_prositfdr_{args.species}.csv"
    header = (["ordinal", "scan_id", "charge", "n_candidates"]
              + [f"{b}_label" for b in branches]
              + [f"{b}_fused" for b in branches])
    n_processed = n_skipped = 0

    with open(results_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for ordinal, (meta, peaks) in enumerate(
            dc.progress(dc.iter_mgf_spectra(mgf, max_scans=args.max_scans),
                        total=total, desc="prosit fdr", unit="spec", mininterval=1.0)
        ):
            charge = dc.parse_charge_from_meta(meta)
            if charge <= 0 or peaks.size == 0:
                n_skipped += 1
                continue
            rows = cand_groups.get(ordinal, [])
            cands = [(seq, lab) for seq, lab, _pp in rows
                     if len(p4.strip_mods(seq)) <= p4.PROSIT_MAX_LEN]
            if not cands:
                n_skipped += 1
                continue
            seqs = [s for s, _ in cands]
            labels = [lab for _, lab in cands]
            exp = dc.transform_exp_intensity(peaks, method=args.exp_transform)
            if exp.size == 0:
                n_skipped += 1
                continue
            # ensure predictions exist for these (bb, charge)
            need_here = {(p4.strip_mods(s), charge) for s in seqs}
            pred.predict_missing(need_here)

            theos = []
            for seq in seqs:
                theo = p4.build_candidate_peaks(*_peaks_annos(pred, seq, charge),
                                                seq, "reanchor")
                if theo.size:
                    theo[:, 1] = np.sqrt(theo[:, 1])
                theos.append(theo)
            n_cand = len(seqs)
            base: dict[str, np.ndarray] = {}
            for r in rels:
                lvl = f"{int(round(r * 100)):03d}"
                for key in (f"d_spec_rel{lvl}", f"f_frag_rel{lvl}",
                            f"m_by_prec_rel{lvl}"):
                    base[key] = np.zeros(n_cand, dtype=np.float64)
            for i, theo in enumerate(theos):
                if theo.size == 0:
                    continue
                sc = p4.score_candidate(exp, theo, rels)
                for r in rels:
                    lvl = f"{int(round(r * 100)):03d}"
                    base[f"d_spec_rel{lvl}"][i] = sc[f"d_spec_rel{lvl}"]
                    base[f"f_frag_rel{lvl}"][i] = sc[f"f_frag_rel{lvl}"]
                    base[f"m_by_prec_rel{lvl}"][i] = sc[f"m_by_prec_rel{lvl}"]
            # fused per rel: hsdf (minmax d+f), hsdf_mby, dfexp(raw), dfexp_mby(raw)
            fused: dict[str, np.ndarray] = {}
            for r in rels:
                lvl = f"{int(round(r * 100)):03d}"
                d = base[f"d_spec_rel{lvl}"]; f = base[f"f_frag_rel{lvl}"]
                m = base[f"m_by_prec_rel{lvl}"]
                fused[f"hs_d{lvl}_f{lvl}"] = _score_avg([d, f])
                fused[f"hs_d{lvl}_f{lvl}_mby"] = _score_avg([d, f, m])
                fused[f"dfexp_rel{lvl}"] = _raw_avg([d, f])
                fused[f"dfexp_mby_rel{lvl}"] = _raw_avg([d, f, m])

            n_processed += 1
            row = {"ordinal": ordinal, "scan_id": meta.get("SCANS", str(ordinal)),
                   "charge": charge, "n_candidates": n_cand}
            for b in branches:
                arr = fused.get(b, base.get(b, np.zeros(n_cand)))
                best = int(np.argmax(arr))
                row[f"{b}_label"] = labels[best]
                row[f"{b}_fused"] = round(float(arr[best]), 6)
            w.writerow(row)

    print(f"[prositfdr] {args.species}: processed={n_processed} skipped={n_skipped} "
          f"branches={len(branches)} -> {results_csv}")
    return 0


def _peaks_annos(pred, seq: str, charge: int):
    ps = p4.strip_mods(seq)
    return pred.get(ps, charge)


if __name__ == "__main__":
    sys.exit(main())
