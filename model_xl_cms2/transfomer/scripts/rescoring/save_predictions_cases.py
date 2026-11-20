#!/usr/bin/env python3
"""Flatten rescoring_results.json into a full prediction CSV + three case CSVs.

Terminology
-----------
  pi / pn = PrimeNovo de novo top-1 (the beam candidate with pn_rank == 1,
            i.e. highest original_score).  This is the baseline being improved.
  v20     = our rescored top-1 (candidate with v20_rank == 1, the final fused
            ranking).  This is "our method".

Three example cases (for the paper):
  1  case1_pi_wrong_v20_fixed    pi top-1 wrong, v20 rescoring corrected it
  2  case2_pi_right_v20_broke    pi top-1 correct, v20 rescoring ranked a
                                 wrong candidate #1 (correct candidate passes
                                 5 ppm, so the error is the ranking, not the filter)
  3  case3_pi_right_v20_filtered pi top-1 correct, but its precursor mass
                                 deviates >5 ppm so the hard filter killed it
                                 ("no usable candidate within 5 ppm")

Also emits: both_correct, both_wrong, gt_absent (GT not in the beam pool).

Outputs (CSV, and XLSX if --xlsx):
  predictions_full.csv            one row per spectrum, all cases
  case1_pi_wrong_v20_fixed.csv
  case2_pi_right_v20_broke.csv
  case3_pi_right_v20_filtered.csv

Run (from v20_0701/transfomer/):
  python scripts/rescoring/save_predictions_cases.py \
      --results /path/rescoring_results.json --out-dir /path/out [--xlsx]

  # --results 也接受 HSDF 的 full_results.csv (rescore_hsdf.py --save-full-results):
  python scripts/rescoring/save_predictions_cases.py \
      --results ./hsdf_sb/full_results.csv --out-dir ./hsdf_sb/cases \
      --mgf-dir /path/to/mgf --ppm 10 --xlsx
"""

from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_TRANSFORMER_DIR = _SCRIPTS_DIR.parent
if str(_TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pandas as pd

# numpy shim 需在 pandas 之后设置, 否则 Windows 上 pandas.compat 导入会崩溃
try:
    import numpy.core as _nc, numpy.core.multiarray as _nma
    sys.modules.setdefault("numpy._core", _nc)
    sys.modules.setdefault("numpy._core.multiarray", _nma)
except Exception:
    pass

import candidate_rescoring as _cr


def _f(c, key, default=0.0):
    try:
        return float(c.get(key, default))
    except (TypeError, ValueError):
        return default


def _top_by_rank(cands, rank_key):
    return next((c for c in cands if int(c.get(rank_key, -1)) == 1), None)


def classify(scan, ppm=5.0):
    cands = scan.get("candidates", [])
    gt_rank = int(scan.get("gt_rank", 0) or 0)
    if not cands:
        return None
    if gt_rank <= 0:
        # GT not in pool
        pn = _top_by_rank(cands, "pn_rank") or max(cands, key=lambda c: _f(c, "original_score"))
        v20 = _top_by_rank(cands, "v20_rank") or cands[0]
        return {
            "case": "gt_absent", "gt_seq": "", "gt_rank": 0, "gt_hit": False,
            "pn_top1": pn.get("seq", ""), "pn_score": _f(pn, "original_score"),
            "pn_mass_ppm": _f(pn, "mass_ppm"), "pi_correct": False,
            "v20_top1": v20.get("seq", ""), "v20_score": _f(v20, "v20_score"),
            "v20_score_model": _f(v20, "v20_score_model"),
            "v20_mass_ppm": _f(v20, "mass_ppm"), "v20_correct": False,
            "n_5ppm": sum(1 for c in cands if _f(c, "mass_ppm", 999) <= ppm),
        }

    # GT candidate = the one V20 ranked at gt_rank
    gt_cand = next((c for c in cands if int(c.get("v20_rank", -1)) == gt_rank), None)
    gt_seq = gt_cand.get("seq", "") if gt_cand else ""
    pn = _top_by_rank(cands, "pn_rank") or max(cands, key=lambda c: _f(c, "original_score"))
    v20 = _top_by_rank(cands, "v20_rank") or cands[0]

    pi_correct = bool(gt_seq) and _cr._peptide_matches(str(pn.get("seq", "")), str(gt_seq))
    v20_correct = (gt_rank == 1)
    n_5ppm = sum(1 for c in cands if _f(c, "mass_ppm", 999) <= ppm)

    if pi_correct and v20_correct:
        case = "both_correct"
    elif (not pi_correct) and v20_correct:
        case = "case1_pi_wrong_v20_fixed"
    elif pi_correct and (not v20_correct):
        case = "case3_pi_right_v20_filtered" if _f(pn, "mass_ppm", 999) > ppm else "case2_pi_right_v20_broke"
    else:
        case = "both_wrong"

    return {
        "case": case, "gt_seq": gt_seq, "gt_rank": gt_rank, "gt_hit": True,
        "pn_top1": pn.get("seq", ""), "pn_score": _f(pn, "original_score"),
        "pn_mass_ppm": _f(pn, "mass_ppm"), "pi_correct": pi_correct,
        "v20_top1": v20.get("seq", ""), "v20_score": _f(v20, "v20_score"),
        "v20_score_model": _f(v20, "v20_score_model"),
        "v20_mass_ppm": _f(v20, "mass_ppm"), "v20_correct": v20_correct,
        "n_5ppm": n_5ppm,
    }


def extract_mgf_seqs(mgf_dir):
    """Return the SEQ= of every spectrum, in MGF file order (row-order aligned).

    Accepts either a directory (globs *.mgf inside) or a single .mgf file path.
    """
    p = Path(mgf_dir)
    mgf_files = [p] if p.is_file() else sorted(p.glob("*.mgf"))
    seqs = []
    for mgf_path in mgf_files:
        cur = ""
        in_block = False
        with open(mgf_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                up = s.upper()
                if up.startswith("BEGIN IONS"):
                    in_block = True
                    cur = ""
                elif up.startswith("END IONS"):
                    in_block = False
                    seqs.append(cur)
                elif in_block and s.startswith("SEQ="):
                    cur = s[4:]
    return seqs


def load_full_results_csv(path):
    """从 rescore_hsdf.py --save-full-results 生成的 full_results.csv 重建 classify() 所需的
    rescoring_results.json 结构。pn_rank/v20_rank 直接用 CSV 里的列 (无歧义);
    gt_rank = gt_match==True 那一行的 v20_rank (GT 不在池则为 0)。"""
    df = pd.read_csv(path)
    required = {"ordinal", "scan_id", "charge", "pepmass", "cand_seq",
                "pn_rank", "v20_rank", "pn_score", "fused", "mass_ppm", "gt_match"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"full_results.csv 缺少必需列: {sorted(missing)}\n"
            f"请用 rescore_hsdf.py --save-full-results 重新生成 (不要加 --skip-results-save)。")
    results = []
    for _, grp in df.groupby("ordinal", sort=True):
        grp = grp.reset_index(drop=True)
        cands = [{
            "seq": str(r["cand_seq"]),
            "original_score": float(r["pn_score"]),
            "pn_rank": int(r["pn_rank"]),
            "v20_rank": int(r["v20_rank"]),
            "v20_score": float(r["fused"]),
            "v20_score_model": 0.0,
            "mass_ppm": float(r["mass_ppm"]),
        } for _, r in grp.iterrows()]
        gm = grp[grp["gt_match"].astype(str).str.lower().isin(["true", "1"])]
        gt_rank = int(gm.iloc[0]["v20_rank"]) if len(gm) > 0 else 0
        first = grp.iloc[0]
        results.append({
            "scan_id": str(first["scan_id"]),
            "num_candidates": len(grp),
            "exp_charge": int(first["charge"]),
            "exp_pepmass": float(first.get("pepmass", 0) or 0),
            "gt_rank": gt_rank,
            "gt_hit": gt_rank > 0,
            "candidates": cands,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="rescoring_results.json 或 full_results.csv (HSDF --save-full-results 输出)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mgf-dir", default=None,
                    help="Optional: MGF dir to read the raw SEQ= GT (row-order aligned with results).")
    ap.add_argument("--xlsx", action="store_true", help="also write .xlsx")
    ap.add_argument("--ppm", type=float, default=5.0,
                    help="case3/case2 的质量阈值 (ppm): GT 质量偏差 > ppm 且 v20 排错 → "
                         "case3 (被过滤); <= ppm → case2 (排名错)。默认 5, "
                         "应与 --precursor-hard-threshold 一致。")
    args = ap.parse_args()

    results_path = Path(args.results)
    if results_path.suffix.lower() == ".csv":
        results = load_full_results_csv(results_path)
    else:
        with open(args.results) as f:
            results = json.load(f)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(results)} spectra from {results_path.name}")

    mgf_seqs = extract_mgf_seqs(args.mgf_dir) if args.mgf_dir else []
    if mgf_seqs:
        n_nonempty = sum(1 for s in mgf_seqs if s)
        if len(mgf_seqs) != len(results):
            print(f"WARNING: MGF SEQ count ({len(mgf_seqs)}) != results ({len(results)}); "
                  f"mgf_gt_seq may be misaligned.")
        print(f"MGF SEQ loaded: {len(mgf_seqs)} (row-order), non-empty SEQ: {n_nonempty}")
        if n_nonempty == 0:
            print("WARNING: no SEQ= lines found in the MGF -- is it the GT-annotated MGF?")

    rows = []
    for i, scan in enumerate(results):
        r = classify(scan, args.ppm)
        if r is None:
            continue
        rows.append({
            "scan_id": scan.get("scan_id", ""),
            "mgf_gt_seq": mgf_seqs[i] if i < len(mgf_seqs) else "",
            "gt_rank": r["gt_rank"],
            "gt_hit": r["gt_hit"],
            "pn_top1": r["pn_top1"],
            "pn_score": round(r["pn_score"], 6),
            "pn_mass_ppm": round(r["pn_mass_ppm"], 2),
            "pi_correct": r["pi_correct"],
            "v20_top1": r["v20_top1"],
            "v20_score": round(r["v20_score"], 6),
            "v20_score_model": round(r["v20_score_model"], 6),
            "v20_mass_ppm": round(r["v20_mass_ppm"], 2),
            "v20_correct": r["v20_correct"],
            "n_5ppm": r["n_5ppm"],
            "num_candidates": int(scan.get("num_candidates", 0)),
            "exp_charge": scan.get("exp_charge", 0),
            "exp_pepmass": scan.get("exp_pepmass", 0),
            "case": r["case"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "predictions_full.csv", index=False)
    if args.xlsx:
        df.to_excel(out / "predictions_full.xlsx", index=False)

    print(f"\nCase distribution (n={len(df)}):")
    for case, cnt in Counter(df["case"]).most_common():
        print(f"  {case:<32} {cnt:>6}")

    case_files = {
        "case1_pi_wrong_v20_fixed": "case1_pi_wrong_v20_fixed",
        "case2_pi_right_v20_broke": "case2_pi_right_v20_broke",
        "case3_pi_right_v20_filtered": "case3_pi_right_v20_filtered",
    }
    for case, name in case_files.items():
        sub = df[df["case"] == case]
        sub.to_csv(out / f"{name}.csv", index=False)
        if args.xlsx:
            sub.to_excel(out / f"{name}.xlsx", index=False)
        print(f"  saved {name}.csv ({len(sub)} rows)")

    print(f"\nDone. Full table + case tables in {out}")


if __name__ == "__main__":
    main()
