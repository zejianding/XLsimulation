#!/usr/bin/env python3
"""Summarize group metrics with exact group and pooled means/standard deviations.

Pooled weights are actual comparison counts: N (direct), N*N (filtered cross),
N*(N-1)/2 (within-group). Quantiles describe group means only; pooled quantiles
are deliberately blank rather than incorrectly reusing group quantiles.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = ["domain", "source", "mode", "group_file"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate group metrics")
    rows = []
    buckets = [("N2",2,2),("N3-5",3,5),("N6-10",6,10),("N>10",11,float("inf"))]
    for domain, domain_df in df.groupby("domain"):
        for (source, mode), sub in domain_df.groupby(["source", "mode"]):
            layers = [("all",set(sub.group_file))]
            if source in ("filtered", "prosit"):
                other = domain_df[(domain_df.source == ("prosit" if source == "filtered" else "filtered")) & (domain_df["mode"] == mode) & (domain_df.valid == 1)]
                paired = set(other.group_file) & set(sub.loc[sub.valid == 1,"group_file"])
                layers.append(("paired",paired))
            else:
                # Baselines stay independent of Prosit success.
                paired = None
            for name,lo,hi in buckets:
                ids = set(sub.loc[(sub.n_spectra>=lo)&(sub.n_spectra<=hi),"group_file"])
                layers.append((name,ids))
                if paired is not None:
                    layers.append(("paired_"+name,ids & paired))
            for layer, ids in layers:
                selected = sub[sub.group_file.isin(ids)]
                good = selected[(selected.valid == 1) & np.isfinite(selected["mean"])]
                common = dict(domain=domain, source=source, mode=mode, layer=layer,
                              n_groups_expected=len(selected), n_groups=len(good), n_groups_invalid=len(selected)-len(good))
                means = good["mean"].to_numpy()
                for weighting in ("group", "pooled"):
                    n = len(means) if weighting == "group" else int(good.n_values.sum())
                    total = float(means.sum()) if weighting == "group" else float(good["sum"].sum())
                    sq = float(means@means) if weighting == "group" else float(good.sum_sq.sum())
                    avg = total/n if n else np.nan
                    std = np.sqrt(max(0., (sq-total*total/n)/(n-1))) if n>1 else np.nan
                    quantiles = np.quantile(means,[.1,.5,.9]) if len(means) and weighting == "group" else [np.nan]*3
                    rows.append(dict(common, weighting=weighting, n_units=n, mean=avg, std=std,
                                     p10=quantiles[0], median=quantiles[1], p90=quantiles[2],
                                     quantile_population="group_means" if weighting == "group" else "not_reported"))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--metrics", nargs="+", help="group_metrics.csv files from each intensity domain")
    a = p.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = [Path(x) for x in a.metrics] if a.metrics else sorted(out.glob("metrics_*/group_metrics.csv"))
    if not paths:
        raise FileNotFoundError("No group_metrics.csv inputs")
    df = summarize(paths)
    for name, data in [("summary.csv",df),("task2_summary.csv",df[df.source == "baseline"])]:
        tmp = out/(name+".partial")
        data.to_csv(tmp,index=False,float_format="%.10g")
        tmp.replace(out/name)
    print(df[(df.layer.isin(["all","paired"])) & (df.weighting == "group")].to_string(index=False))


if __name__ == "__main__":
    main()
