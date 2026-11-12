#!/usr/bin/env python3
"""Collect the two "prediction vs real spectra" cases into one tidy CSV + pivot.

For every run dir under ``--root`` (``<predictor>_<species>_<stamp>/``):

  B1  cosine_summary.csv       pred_vs_orig_mean   — prediction vs EVERY original
                                                       spectrum of that peptide (mean)
  B2  cosine_avg_summary.csv   pred_vs_avg_cosine   — prediction vs the single
                                                       AVERAGE of those originals

predictor/species come from the run dir name. Only the top-level per-run CSV is
read; the ``variants/`` copies (rel/projection matrix) are deliberately ignored.

    python collect_avg_results.py --root output [--out 汇总_预测谱vs真实谱.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from collect_results import EXPTS, parse_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="output")
    p.add_argument("--out", default="汇总_预测谱vs真实谱.csv")
    return p.parse_args()


def stats_from(rows, value_col: str, weight_col: str):
    vals = [(float(r[value_col]), int(r[weight_col])) for r in rows
            if r.get(value_col) not in ("", "nan", None)]
    if not vals:
        return None
    tot = sum(w for _, w in vals)
    return dict(groups=len(vals),
                unw=statistics.mean(v for v, _ in vals),
                median=statistics.median(v for v, _ in vals),
                wtd=(sum(v * w for v, w in vals) / tot) if tot else 0.0)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_rows = []

    for p in sorted(root.glob("*/cosine_summary.csv")):           # B1
        pred, sp = parse_run(p.parent.name)
        if sp == "?":
            continue
        st = stats_from(list(csv.DictReader(open(p, encoding="utf-8"))),
                        "pred_vs_orig_mean", "num_original")
        if st:
            out_rows.append({"predictor": pred, "species": sp, "case": "B1", **st, "path": str(p)})

    for p in sorted(root.glob("*/cosine_avg_summary.csv")):       # B2
        pred, sp = parse_run(p.parent.name)
        if sp == "?":
            continue
        st = stats_from(list(csv.DictReader(open(p, encoding="utf-8"))),
                        "pred_vs_avg_cosine", "num_original")
        if st:
            out_rows.append({"predictor": pred, "species": sp, "case": "B2", **st, "path": str(p)})

    cols = ["predictor", "species", "case", "groups", "unw", "wtd", "median", "path"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(out_rows, key=lambda r: (r["species"], r["predictor"], r["case"])):
            w.writerow({**r, "unw": round(r["unw"], 4), "wtd": round(r["wtd"], 4),
                        "median": round(r["median"], 4)})
    print(f"wrote {args.out}: {len(out_rows)} rows")

    keys = sorted({(r["predictor"], r["case"]) for r in out_rows})
    pivot = {}
    for r in out_rows:
        pivot.setdefault(r["species"], {})[(r["predictor"], r["case"])] = r["unw"]
    pivot_path = Path(args.out).with_name(Path(args.out).stem + "_pivot.csv")
    with open(pivot_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species"] + ["/".join(k) for k in keys])
        for sp in EXPTS:
            if sp in pivot:
                w.writerow([sp] + [f"{pivot[sp][k]:.4f}" if k in pivot[sp] else ""
                                   for k in keys])
    print(f"pivot -> {pivot_path.name}  (B1=逐条原谱平均, B2=平均谱)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
