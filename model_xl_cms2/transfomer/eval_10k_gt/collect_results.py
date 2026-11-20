#!/usr/bin/env python3
"""Collect every cosine_summary.csv produced by the eval matrix into one tidy CSV.

Scans a root for:
  <root>/<run>/cosine_summary.csv                     (main口径: 无过滤)
  <root>/<run>/variants/<tag>/Grouped_MGFs_rel*/cosine_summary.csv  (matrix口径)
and parses (predictor/species) from the run dir name and (projection, rel, exp, pred)
from the variant path.  Writes a tidy CSV + a species×setting pivot (unweighted mean).

Usage:
    python collect_results.py --root output [--out 汇总矩阵.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

PRED = ("v20pp", "v20raw", "prosit")          # run-dir prefixes
EXPTS = ["mouse", "human", "yeast", "bacillus", "clambacteria",
         "honeybee", "mmazei", "ricebean", "tomato"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="output")
    p.add_argument("--out", default="汇总矩阵.csv")
    return p.parse_args()


def parse_run(name: str) -> tuple[str, str]:
    # optional Prosit reference runs: prosit_byonly_<sp> / prosit_rawapi_<sp>
    for variant in ("byonly", "rawapi"):
        pre = f"prosit_{variant}_"
        if name.startswith(pre):
            rest = name[len(pre):]
            for sp in EXPTS:
                if rest.startswith(sp):
                    return f"prosit_{variant}", sp
    for pre in PRED:
        if name.startswith(pre + "_"):
            rest = name[len(pre) + 1:]
            for sp in EXPTS:
                if rest.startswith(sp):
                    return pre, sp
    return "?", "?"


def parse_tag(path: Path) -> tuple[str, str, str, str]:
    # variants/<proj>_relXXX[-XXX]_exp<exp>_pred<pred>/Grouped_MGFs_rel<lvl>
    for part in path.parts:
        m = re.match(r"^(raw|grid)_rel(\S+?)_exp(\w+?)_pred(\w+)$", part)
        if m:
            return m.group(1), m.group(2), m.group(3), m.group(4)
    return "-", "-", "-", "-"


def read_summary(p: Path):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return None
    cols = rows[0].keys()
    c_mean = "pred_vs_orig_mean" if "pred_vs_orig_mean" in cols else "prediction_vs_original_mean"
    c_oo = "orig_orig_mean" if "orig_orig_mean" in cols else "original_original_mean"
    n = len(rows)
    tot = sum(int(r["num_original"]) for r in rows)
    means = [float(r[c_mean]) for r in rows]
    oo = [float(r[c_oo]) for r in rows if r.get(c_oo) not in ("", "nan", None)]
    return dict(groups=n, spectra=tot,
                unw=statistics.mean(means), median=statistics.median(means),
                wtd=sum(float(r[c_mean]) * int(r["num_original"]) for r in rows) / tot if tot else 0.0,
                oo=statistics.mean(oo) if oo else float("nan"))


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_rows = []
    for p in sorted(root.glob("**/cosine_summary.csv")):
        st = read_summary(p)
        if not st:
            continue
        run_name = p.parent.name if "variants" not in p.parts else p.parents[3].name
        pred, sp = parse_run(run_name)
        proj, rel, exp, ptrans = parse_tag(p)
        out_rows.append({"predictor": pred, "species": sp, "projection": proj,
                         "rel": rel, "exp_transform": exp, "pred_transform": ptrans,
                         "groups": st["groups"], "spectra": st["spectra"],
                         "unw": round(st["unw"], 4), "wtd": round(st["wtd"], 4),
                         "median": round(st["median"], 4), "orig_orig": round(st["oo"], 4),
                         "path": str(p)})
    cols = ["predictor", "species", "projection", "rel", "exp_transform", "pred_transform",
            "groups", "spectra", "unw", "wtd", "median", "orig_orig", "path"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(out_rows, key=lambda r: (r["species"], r["predictor"], r["projection"], r["rel"])))
    print(f"wrote {args.out}: {len(out_rows)} rows")
    # pivot: species x (predictor|proj|rel) unweighted
    keys = sorted({(r["predictor"], r["projection"], r["rel"], r["exp_transform"], r["pred_transform"])
                   for r in out_rows})
    pivot = {}
    for r in out_rows:
        pivot.setdefault(r["species"], {})[(r["predictor"], r["projection"], r["rel"],
                                            r["exp_transform"], r["pred_transform"])] = r["unw"]
    hdr = ["species"] + ["/".join(k) for k in keys]
    with open(Path(args.out).with_name("汇总矩阵_pivot.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for sp in EXPTS:
            if sp in pivot:
                w.writerow([sp] + [pivot[sp].get(k, "") for k in keys])
    print("pivot -> 汇总矩阵_pivot.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
