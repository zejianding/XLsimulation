#!/usr/bin/env python3
"""All three comparisons and within-group baselines, joined by spectrum identity.

Missing/corrupt files or ambiguous identity are fatal. Invalid spectra are retained
as NaN with explicit reasons; an aggregate is valid only when all its pairs are
valid (no silent change to a group's denominator). Detail CSVs retain every pair.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import gzip
from pathlib import Path
import numpy as np
from mgf_io import iter_blocks, headers_of, parse_block_key


def read_groups(directory):
    with (Path(directory)/"group_map.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len({r["group_file"] for r in rows}) != len(rows):
        raise ValueError("Duplicate group files")
    return rows


def load_group(mgf_path, npz_path, expected):
    with Path(mgf_path).open("rb") as fh:
        blocks = list(iter_blocks(fh))
    metas = [headers_of(b) for b in blocks]
    ids = [m.get("SPECTRUM_ID", m.get("TITLE", "")) for m in metas]
    if len(ids) != expected or len(set(ids)) != expected or any(not x for x in ids):
        raise ValueError(f"Missing/duplicate identities or incorrect count: {mgf_path}")
    with np.load(npz_path, allow_pickle=False) as z:
        emb = z["embeddings"].astype(np.float64)
        emb_ids = z["spectrum_ids"].astype(str).tolist()
        reasons = z["invalid_reasons"].astype(str).tolist() if "invalid_reasons" in z else [""]*len(emb)
    if emb.ndim != 2 or len(emb) != expected or len(emb_ids) != expected or len(set(emb_ids)) != expected or set(ids) != set(emb_ids) or len(reasons) != expected:
        raise ValueError(f"Embedding identity/count mismatch: {npz_path}")
    order = {sid:i for i,sid in enumerate(emb_ids)}
    indexes = [order[sid] for sid in ids]
    emb = emb[indexes]; reasons = [reasons[i] for i in indexes]
    norms = np.linalg.norm(emb, axis=1)
    good = np.isfinite(emb).all(axis=1) & (norms > 0) & np.array([not x for x in reasons])
    for i in range(expected):
        if not good[i] and not reasons[i]:
            reasons[i] = "nonfinite_or_zero_embedding"
    normed = np.full_like(emb, np.nan)
    normed[good] = emb[good]/norms[good,None]
    return ids, normed, reasons, blocks


def cos(a, b):
    return np.clip(a @ b.T, -1.0, 1.0)


def mean(a):
    return float(np.mean(a)) if np.size(a) else float("nan")


def complete_stats(a):
    a = np.asarray(a).ravel()
    valid = bool(len(a) and np.isfinite(a).all())
    return dict(n_values=len(a), n_valid_values=int(np.isfinite(a).sum()),
                valid=int(valid), mean=mean(a) if valid else float("nan"),
                sum=float(a.sum()) if valid else float("nan"),
                sum_sq=float(a@a) if valid else float("nan"),
                min=float(a.min()) if valid else float("nan"),
                max=float(a.max()) if valid else float("nan"))


def evaluate(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.csv")) or any(out.glob("*.csv.gz")):
        raise FileExistsError("Metrics require a fresh output directory")
    ref_dir, src_dir = Path(a.ref_grouped_dir), Path(a.src_grouped_dir)
    groups = read_groups(ref_dir)
    src_groups = {r["group_file"]:r for r in read_groups(src_dir)}
    if set(src_groups) != {r["group_file"] for r in groups}:
        raise ValueError("Real/filtered group sets differ")
    q_groups = {}
    if a.prosit_grouped_dir:
        q_groups = {r["group_file"]:r for r in read_groups(a.prosit_grouped_dir)}
        if not set(q_groups) <= set(src_groups):
            raise ValueError("Unexpected Prosit groups")
    with ExitStack() as stack:
        def writer(name, fields):
            path = out/name
            fh = stack.enter_context(gzip.open(path, "wt", encoding="utf-8", newline="") if name.endswith(".gz") else path.open("w", encoding="utf-8", newline=""))
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); return w
        base = ["group_file", "sequence", "charge", "domain"]
        direct = writer("mode1_direct.csv", base+["spectrum_id", "cos_filtered", "cos_prosit", "real_status", "filtered_status", "prosit_status"])
        averaged = writer("mode2_averaged.csv", base+["n_spectra", "cos_filtered_avg", "cos_filtered_each", "cos_prosit", "orig_vs_avg_mean"])
        avg_detail = writer("mode2_filtered_detail.csv", base+["spectrum_id", "cos_filtered_to_average", "cos_real_to_average"])
        pairwise = writer("mode3_pairwise.csv", base+["n_spectra", "pairwise_filtered", "pairwise_prosit", "filtered_offdiag"])
        cross = writer("mode3_pairs.csv.gz", base+["filtered_id", "real_id", "cosine"])
        cross_means = writer("mode3_filtered_means.csv", base+["filtered_id", "mean_cosine"])
        within = writer("task2_within_group.csv", base+["n_spectra", "n_pairs", "mean_cos_real_real", "min_cos_real_real", "max_cos_real_real", "mean_cos_filtered_filtered", "min_cos_filtered_filtered", "max_cos_filtered_filtered"])
        within_pairs = writer("task2_pairs.csv.gz", base+["spectrum_id_a", "spectrum_id_b", "cos_real_real", "cos_filtered_filtered"])
        metrics = writer("group_metrics.csv", base+["n_spectra", "source", "mode", "n_values", "n_valid_values", "valid", "mean", "sum", "sum_sq", "min", "max"])
        for count, row in enumerate(groups, 1):
            name = row["group_file"]; stem = Path(name).stem; n = int(row["n_original"])
            b = dict(group_file=name, sequence=row["sequence"], charge=int(row["charge"]), domain=a.domain)
            if n < 2 or int(src_groups[name]["n_original"]) != n:
                raise ValueError(f"Incorrect spectrum count: {name}")
            ri, er, rr_status, rb = load_group(ref_dir/name, Path(a.ref_emb_dir)/(stem+".npz"), n)
            fi, ef, ff_status, fb = load_group(src_dir/name, Path(a.src_emb_dir)/(stem+".npz"), n)
            expected_key = (row["sequence"], int(row["charge"]))
            if any(parse_block_key(x) != expected_key for x in rb+fb):
                raise ValueError(f"Group key mismatch: {name}")
            if set(ri) != set(fi):
                raise ValueError(f"Filtered/real spectrum IDs differ: {name}")
            ix = {sid:i for i,sid in enumerate(fi)}; order = [ix[sid] for sid in ri]
            ef = ef[order]; ff_status = [ff_status[i] for i in order]
            _, ar, _, arb = load_group(Path(a.avg_ref_grouped_dir)/name, Path(a.avg_ref_emb_dir)/(stem+".npz"), 1)
            _, af, _, afb = load_group(Path(a.avg_src_grouped_dir)/name, Path(a.avg_src_emb_dir)/(stem+".npz"), 1)
            if any(parse_block_key(x) != expected_key for x in arb+afb):
                raise ValueError(f"Average group key mismatch: {name}")
            q = np.full((1, er.shape[1]), np.nan); q_status = "prediction_missing"
            if name in q_groups:
                _, q, qr, qb = load_group(Path(a.prosit_grouped_dir)/name, Path(a.prosit_emb_dir)/(stem+".npz"), 1)
                if parse_block_key(qb[0]) != expected_key:
                    raise ValueError(f"Prosit group key mismatch: {name}")
                q_status = qr[0]
            m = cos(ef, er); diag = m.diagonal(); pv = cos(q, er).ravel()
            fea = cos(ef, ar).ravel(); rea = cos(er, ar).ravel()
            fav = cos(af, ar).ravel(); qav = cos(q, ar).ravel()
            iu = np.triu_indices(n, k=1); rr = cos(er, er)[iu]; ff = cos(ef, ef)[iu]
            off = m[~np.eye(n, dtype=bool)]
            values = [("filtered", "direct", diag), ("prosit", "direct", pv),
                      ("filtered", "averaged_each", fea), ("filtered", "averaged", fav),
                      ("prosit", "averaged_each", qav), ("prosit", "averaged", qav),
                      ("filtered", "pairwise", m.ravel()), ("prosit", "pairwise", pv),
                      ("baseline", "real_real", rr), ("baseline", "filtered_filtered", ff),
                      ("baseline", "orig_vs_avg", rea)]
            for source, mode, vals in values:
                metrics.writerow(dict(b, n_spectra=n, source=source, mode=mode, **complete_stats(vals)))
            for i, sid in enumerate(ri):
                direct.writerow(dict(b, spectrum_id=sid, cos_filtered=diag[i], cos_prosit=pv[i], real_status=rr_status[i], filtered_status=ff_status[i], prosit_status=q_status))
                avg_detail.writerow(dict(b, spectrum_id=sid, cos_filtered_to_average=fea[i], cos_real_to_average=rea[i]))
                cross_means.writerow(dict(b, filtered_id=sid, mean_cosine=mean(m[i])))
                for j, rid in enumerate(ri):
                    cross.writerow(dict(b, filtered_id=sid, real_id=rid, cosine=m[i,j]))
            averaged.writerow(dict(b, n_spectra=n, cos_filtered_avg=mean(fav), cos_filtered_each=mean(fea), cos_prosit=mean(qav), orig_vs_avg_mean=mean(rea)))
            pairwise.writerow(dict(b, n_spectra=n, pairwise_filtered=mean(m), pairwise_prosit=mean(pv), filtered_offdiag=mean(off)))
            rs, fs = complete_stats(rr), complete_stats(ff)
            within.writerow(dict(b, n_spectra=n, n_pairs=len(rr), mean_cos_real_real=rs["mean"], min_cos_real_real=rs["min"], max_cos_real_real=rs["max"], mean_cos_filtered_filtered=fs["mean"], min_cos_filtered_filtered=fs["min"], max_cos_filtered_filtered=fs["max"]))
            for k, (i,j) in enumerate(zip(*iu)):
                within_pairs.writerow(dict(b, spectrum_id_a=ri[i], spectrum_id_b=ri[j], cos_real_real=rr[k], cos_filtered_filtered=ff[k]))
            if count % 500 == 0:
                print(f"Compared {count}/{len(groups)} groups ({a.domain})", flush=True)
    print(f"Metrics: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["ref-grouped-dir", "ref-emb-dir", "src-grouped-dir", "src-emb-dir", "avg-ref-grouped-dir", "avg-src-grouped-dir", "avg-ref-emb-dir", "avg-src-emb-dir", "out-dir"]:
        p.add_argument("--"+name, required=True)
    p.add_argument("--prosit-grouped-dir")
    p.add_argument("--prosit-emb-dir")
    p.add_argument("--domain", choices=["none", "sqrt_clip", "sqrt"], default="none")
    p.add_argument("--save-detail", action="store_true", help="Compatibility flag; all detail is always saved")
    a = p.parse_args()
    if bool(a.prosit_grouped_dir) != bool(a.prosit_emb_dir):
        p.error("Both Prosit directories must be supplied together")
    evaluate(a)


if __name__ == "__main__":
    main()
