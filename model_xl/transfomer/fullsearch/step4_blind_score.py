#!/usr/bin/env python3
"""Step 4 — blind whole-library search accuracy per scoring branch.

No precursor-mass filter: each experimental spectrum (charge z) is compared to
EVERY library entry predicted at the same charge z; per branch the argmax over
the whole library is the "top-1"; search accuracy = fraction where top-1 matches
the MGF SEQ= GT (I/L-equivalent).  decoy_win = fraction where the argmax is a
decoy.  Branch family per rel level: d (peak cosine), f (explained intensity),
m_by (fragment precision), plus fusions hsdf / hsdf_mby (min-max over the per
spectrum whole-library pool) and dfexp / dfexp_mby (raw equal-weight mean).

The d component uses nearest-neighbour matching of each theoretical peak to the
experimental peaks within 0.05 Da (max-in-window approximated by nearest) —
documented approximation; run --selfcheck to compare a few spectra against the
scalar dc.peak_cosine / dc.frag_precision definitions.

Usage:
    python step4_blind_score.py --species mouse --work <fullsearch> \
        --predictor v20 [--max-gt-len 32] [--rels 0.05,0.10,0.20] [--max-scans N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from fullsearch_common import (dc, read_library_tsv, TOL, SPEC_MIN,
                               backbone_of)

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--predictor", required=True, choices=["v20", "prosit"])
    p.add_argument("--max-gt-len", type=int, default=None,
                   help="exclude spectra whose GT backbone exceeds this "
                        "(Prosit = 30; default = no limit)")
    p.add_argument("--max-lib-len", type=int, default=None,
                   help="exclude library entries whose backbone exceeds this from "
                        "the top-1 (align a full-length predictor like V20 with "
                        "Prosit's <=30 library; default = no limit)")
    p.add_argument("--rels", default="0.05,0.10,0.20")
    p.add_argument("--exp-transform", default="sqrt_clip",
                   choices=["none", "sqrt_clip", "sqrt"])
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--selfcheck", action="store_true",
                   help="compare vectorized vs scalar scoring on a few spectra")
    return p.parse_args()


def rel_lvl(r: float) -> str:
    return f"{int(round(r * 100)):03d}"


def load_lib(ddir: Path, predictor: str, charge: int) -> dict:
    npz = np.load(ddir / f"lib_{predictor}_c{charge}.npz")
    tmz = npz["tmz"].astype(np.float64)
    tint = npz["tint"].astype(np.float64)
    starts = npz["starts"]; pmax = npz["pmax"].astype(np.float64)
    label = npz["label"]
    E = int(len(starts) - 1)
    # per-peak entry id (npz entry_ids is per-entry; expand by segment lengths)
    counts = np.diff(starts)
    eid = np.repeat(np.arange(E, dtype=np.int64), counts)
    sidx = np.argsort(tmz, kind="stable")
    return {"tmz": tmz, "tint": tint, "starts": starts, "eid": eid,
            "pmax": pmax, "label": label, "blen": npz["blen"].astype(np.int32),
            "n_entries": E,
            "tmzS": tmz[sidx], "tintS": tint[sidx],
            "eidS": eid[sidx]}


def _reduce_sum(a: np.ndarray, starts: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.zeros(len(starts) - 1, dtype=np.float64)
    return np.add.reduceat(a, starts[:-1]).astype(np.float64)


def per_rel_arrays(emz, eint, lib, rel):
    """Return (d, f, m) per entry for one rel level (vectorized)."""
    tmz, tint = lib["tmz"], lib["tint"]
    pmax, starts = lib["pmax"], lib["starts"]
    eid = lib["eid"]  # per-peak entry id, 0..E-1
    E = lib["n_entries"]
    # rel <= 0 => no predicted-side intensity filter (keep every stored peak);
    # otherwise keep peaks above max(SPEC_MIN, rel * pmax_entry).
    if rel > 0:
        kept = tint > np.maximum(SPEC_MIN, rel * pmax[eid])
    else:
        kept = tint > 0.0

    # matched experimental intensity per theo peak (nearest within tol)
    idx = np.searchsorted(emz, tmz)
    idx = np.clip(idx, 0, len(emz) - 1)
    d0 = np.abs(emz[idx] - tmz)
    left = np.maximum(idx - 1, 0)
    d1 = np.abs(emz[left] - tmz)
    use0 = d0 <= d1
    near = np.where(use0, idx, left)
    mz_near = emz[near]
    matched = np.where(np.abs(mz_near - tmz) <= TOL, eint[near], 0.0)

    tkept = tint * kept
    mkept = matched * kept
    num = _reduce_sum(tkept * mkept, starts)
    sumT = _reduce_sum(tkept * tkept, starts)
    sumM = _reduce_sum(mkept * mkept, starts)
    d = num / np.sqrt(np.maximum(sumT, EPS) * np.maximum(sumM, EPS))

    # exact presence: any exp peak within tol
    lo = np.searchsorted(emz, tmz - TOL, side="left")
    hi = np.searchsorted(emz, tmz + TOL, side="right")
    has = (hi > lo)
    hask = (has & kept).astype(np.float64)
    cnt = _reduce_sum(kept.astype(np.float64), starts)
    mh = _reduce_sum(hask, starts)
    m = np.where(cnt > 0, mh / np.maximum(cnt, EPS), 0.0)

    # f: covered experimental intensity per entry (kept theo only)
    eint32 = eint.astype(np.float64)
    total = eint32.sum()
    fcov = np.zeros(E, dtype=np.float64)
    if total > 0 and kept.any():
        keep_idx = np.where(kept)[0]
        tmzS = lib["tmzS"]; eidS = lib["eidS"]; tintS = lib["tintS"]
        # restrict sorted arrays to kept theo of this rel
        # kept in sorted space: thr per entry -> use eidS to compute keptS
        if rel > 0:
            keptS = tintS > np.maximum(SPEC_MIN, rel * pmax[eidS])
        else:
            keptS = tintS > 0.0
        tmzK = tmzS[keptS]; eidK = eidS[keptS]
        for j in range(len(emz)):
            lo2 = np.searchsorted(tmzK, emz[j] - TOL, side="left")
            hi2 = np.searchsorted(tmzK, emz[j] + TOL, side="right")
            if hi2 > lo2:
                np.add.at(fcov, eidK[lo2:hi2], eint32[j])
    f = fcov / max(total, EPS)
    return d, f, m


def scalar_check(emz, eint, theo, theo_mz, theo_int, rel):
    """Reference scalar values for one entry (validation only)."""
    pmax = float(theo_int.max()) if theo_int.size else 0.0
    eff = max(SPEC_MIN, rel * pmax)
    sel = theo_int > eff
    th = np.column_stack([theo_mz[sel], theo_int[sel]])
    exp = np.column_stack([emz, eint])
    return dc.peak_cosine(exp, th, tol=TOL), dc.frag_precision(exp, th, tol=TOL)


def branch_key(base: str, r: float) -> str:
    return f"{base}_rel{rel_lvl(r)}"


def main() -> int:
    args = parse_args()
    work = Path(args.work)
    ddir = work / "data" / args.species
    odir = work / "output" / args.species
    odir.mkdir(parents=True, exist_ok=True)
    rels = [float(x) for x in args.rels.split(",") if x.strip()]

    rows = read_library_tsv(ddir / "library_annotated.tsv.gz")
    seq_by_idx = {r["idx"]: r["annoseq"] for r in rows}
    _fasta, mgf = dc.resolve_species_files(Path(args.root), args.species)
    total = args.max_scans or dc.count_mgf_spectra(mgf)

    lib_cache: dict[int, dict] = {}
    branch_names = []
    for r in rels:
        for b in ("d_spec", "f_frag", "m_by_prec", "hs_d{}_f{}", "hs_d{}_f{}_mby",
                  "dfexp", "dfexp_mby"):
            lvl = rel_lvl(r)
            key = (b.format(lvl, lvl) if "{}" in b
                   else f"{b}_rel{lvl}")
            branch_names.append(key)

    header = ["ordinal", "scan_id", "charge", "gt_seq", "n_lib"]
    for b in branch_names:
        header += [f"{b}_label"]
    results = []
    denom = 0
    branch_hit = {b: 0 for b in branch_names}
    branch_decoy = {b: 0 for b in branch_names}
    n_spectra = 0

    for ordinal, (meta, peaks) in enumerate(
        dc.progress(dc.iter_mgf_spectra(mgf, max_scans=args.max_scans),
                    total=total, desc="blind score", unit="spec", mininterval=1.0)
    ):
        n_spectra += 1
        charge = dc.parse_charge_from_meta(meta)
        gt = meta.get("SEQ", "")
        if charge <= 0 or not gt or peaks.size == 0:
            continue
        if args.max_gt_len is not None and len(backbone_of(gt)) > args.max_gt_len:
            continue  # unresolvable by this predictor
        if charge not in lib_cache:
            lib_cache[charge] = load_lib(ddir, args.predictor, charge)
        lib = lib_cache[charge]
        if lib["n_entries"] == 0:
            continue
        exp = dc.transform_exp_intensity(peaks, method=args.exp_transform)
        if exp.size == 0:
            continue
        o = np.argsort(exp[:, 0])
        emz = exp[o, 0].astype(np.float64)
        eint = exp[o, 1].astype(np.float64)
        denom += 1

        per_rel: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for r in rels:
            per_rel[rel_lvl(r)] = per_rel_arrays(emz, eint, lib, r)

        row = {"ordinal": ordinal, "scan_id": meta.get("SCANS", str(ordinal)),
               "charge": charge, "gt_seq": gt, "n_lib": lib["n_entries"]}
        lib_labels = lib["label"]  # 1 target 0 decoy
        for r in rels:
            lvl = rel_lvl(r)
            d, f, m = per_rel[lvl]
            E = lib["n_entries"]
            norm = lambda v: (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() \
                else np.zeros(E)
            hsdf = (norm(d) + norm(f)) / 2.0
            hsdf_mby = (norm(d) + norm(f) + norm(m)) / 3.0
            dfexp = (d + f) / 2.0
            dfexp_mby = (d + f + m) / 3.0
            for bkey, arr in (("d_spec", d), ("f_frag", f), ("m_by_prec", m),
                              ("hs_d{}_f{}", hsdf), ("hs_d{}_f{}_mby", hsdf_mby),
                              ("dfexp", dfexp), ("dfexp_mby", dfexp_mby)):
                if "{}" in bkey:
                    key = bkey.format(lvl, lvl)
                else:
                    key = f"{bkey}_rel{lvl}"
                if args.max_lib_len is not None:
                    arr = np.where(lib["blen"] <= args.max_lib_len, arr, -np.inf)
                best = int(np.argmax(arr))
                lab = "T" if lib_labels[best] == 1 else "D"
                row[f"{key}_label"] = lab
                if lab == "T" and dc.peptide_matches(seq_by_idx[best], gt):
                    branch_hit[key] += 1
                if lab == "D":
                    branch_decoy[key] += 1
        results.append(row)

    csv_path = odir / f"results_blind_{args.predictor}_{args.species}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    den = denom or 1
    summary = {
        "species": args.species, "predictor": args.predictor,
        "n_spectra": n_spectra, "denominator": denom,
        "max_gt_len": args.max_gt_len, "rels": rels,
        "branch_top1_acc": {b: round(branch_hit[b] / den, 6) for b in branch_names},
        "branch_decoy_win": {b: round(branch_decoy[b] / den, 6) for b in branch_names},
        "results_csv": str(csv_path),
    }
    (odir / f"summary_blind_{args.predictor}_{args.species}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[blind] {args.predictor} {args.species}: denominator={denom} spectra={n_spectra}")
    for b in branch_names:
        print(f"  {b:<24} top1_acc={summary['branch_top1_acc'][b]:.4f} "
              f"decoy_win={summary['branch_decoy_win'][b]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
