#!/usr/bin/env python3
"""Step 3 — batch-predict every candidate's V20 spectrum and score it vs the real one.

For each MGF spectrum with candidates (from step2), predict the (39,46,6)
intensity grid of every target/decoy candidate and compute the five spectral
similarity components (all in [0,1]):

    gc      grid cosine       (39,46,6) pred grid vs exp-projected grid
    d       spectral cosine   experimental peaks projected onto predicted peaks
    f_bidir spectral cosine   predicted peaks projected onto experimental peaks
    f_exp   explained intensity (fraction of exp intensity covered)
    mby     main b/y-ion fragment precision

The GPU prediction is the expensive part and is done ONCE; the five components
are then fused per requested branch (each branch = a comma-separated subset of
components, branches separated by ';').  Every branch's fused score is the raw
equal-weight mean of its components, so it is comparable across spectra — the
score that branch's FDR curve is ranked by.  Per spectrum each branch picks the
argmax candidate as its top-1 and records its T/D label.

--rel accepts a COMMA-SEPARATED list of predicted-side relative-intensity
filters (e.g. 0.05,0.10,0.20).  Each (branch, rel) pair becomes its own result
branch keyed ``<branch>_rel<lvl>`` (lvl = round(r*100), 3 digits), e.g.
``d_f_exp_rel020`` / ``d_f_exp_mby_rel020``, all computed in the SAME GPU
prediction pass.  gc is rel-independent; d/f_bidir/f_exp/mby are recomputed at
each rel level.

Writes:
    <out-dir>/results_step3_<species>.csv   one row per spectrum, wide per (branch,rel)
    <out-dir>/summary_step3_<species>.json  aggregate per-branch balance summary
    <out-dir>/records/record_step3_<species>.json
    <out-dir>/candidates_detail_<species>_rel<lvl>.csv.gz  per-candidate components (--save-detail)

Usage:
    python step3_score_fdr.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --candidates <fdr>/data/mouse/candidates_mouse_td.tsv.gz \
        --checkpoint /home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt \
        --out-dir <fdr>/output/mouse [--gpu 0] [--batch-size 128] [--max-scans N] \
        --rel 0.05,0.10,0.20 --branches 'd,f_exp;d,f_exp,mby'
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

import fdr_common as fc
import ds_common as dc  # noqa: F401  (path bootstrap + helpers)

DEFAULT_BRANCHES = ("d;gc;f_bidir;gc,f_exp;d,gc,f_exp;d,f_exp;d,f_bidir;"
                    "d,gc,f_bidir;mby;d,f_exp,mby")
DEFAULT_RELS = "0.05,0.10,0.20"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=fc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--candidates", required=True, help="TSV.gz from step2")
    p.add_argument("--checkpoint", default=fc.SERVER_CKPT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-mz", type=float, default=dc.DEFAULT_MIN_MZ)
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--no-packed", action="store_true")
    p.add_argument("--rel", default=DEFAULT_RELS,
                   help="comma-separated predicted-side relative-intensity filters for "
                        "d/f_bidir/f_exp/mby, e.g. 0.05,0.10,0.20 (default "
                        + DEFAULT_RELS + "; 0.20 ~ production d020, 0.05 ~ ds d005)")
    p.add_argument("--tol", type=float, default=fc.DEFAULT_TOL, help="m/z match tolerance (Da)")
    p.add_argument("--exp-transform", default="sqrt_clip", choices=["none", "sqrt_clip", "sqrt"],
                   help="map experimental intensities into the model's training domain before "
                        "scoring (default sqrt_clip = V20 training normalization)")
    p.add_argument("--exp-top-peaks", type=int, default=0,
                   help="keep only the top-N most intense experimental peaks before scoring "
                        "(0 = keep all)")
    p.add_argument("--branches", default=DEFAULT_BRANCHES,
                   help="'sem;i;colon'-separated branches; each branch is a comma-separated "
                        "subset of components (gc,d,f_bidir,f_exp,mby).  e.g. "
                        "d;gc;f_bidir;d,gc;d,f_exp;d,f_bidir;d,gc,f_bidir")
    p.add_argument("--save-detail", action="store_true",
                   help="also write one combined per-candidate component detail CSV.gz PER rel")
    p.add_argument("--save-full", action="store_true",
                   help="also write one per-candidate FULL CSV.gz PER (BRANCH, rel) "
                        "(<out-dir>/full_<branch>_rel<lvl>_<species>.csv.gz) with spectrum "
                        "info, seq/label/ppm/mass, all five component scores, the branch fused "
                        "score, within-spectrum rank, is_best and gt_match for EVERY "
                        "candidate.  Use for post-hoc score-distribution analysis.")
    return p.parse_args()


def parse_branches(s: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return [(branch_key, component_tuple), ...] for a ';'-separated spec."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        comps = tuple(c.strip() for c in part.split(",") if c.strip())
        for c in comps:
            if c not in fc.SCORE_COMPONENTS:
                raise SystemExit(f"unknown component '{c}'; choose from {fc.SCORE_COMPONENTS}")
        key = "_".join(comps)
        out.append((key, comps))
    return out


def parse_rels(s: str) -> list[float]:
    """Parse the --rel string -> sorted list of rel levels; key levels are 3-digit."""
    rels = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not rels:
        raise SystemExit("--rel must contain at least one value")
    return sorted(rels)


def rel_lvl(r: float) -> str:
    return f"{int(round(r * 100)):03d}"


def expand_branches(branch_spec: str, rels: list[float]) -> list[tuple[str, tuple[str, ...], float]]:
    """[(branch_key, comps, rel)] with branch_key suffixed by the rel level, e.g.
    ('d_f_exp_rel020', ('d', 'f_exp'), 0.20)."""
    out: list[tuple[str, tuple[str, ...], float]] = []
    for base_key, comps in parse_branches(branch_spec):
        for r in rels:
            out.append((f"{base_key}_rel{rel_lvl(r)}", comps, r))
    return out


def load_candidates_by_ordinal(tsv_gz: Path) -> dict[int, list[tuple[str, str, float]]]:
    """Read step2 TSV.gz -> {ordinal: [(seq, label, ppm), ...]}."""
    grouped: dict[int, list[tuple[str, str, float]]] = {}
    with gzip.open(tsv_gz, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            ordinal = int(row[0])
            seq = row[5]
            label = row[6]
            ppm = float(row[8])
            grouped.setdefault(ordinal, []).append((seq, label, ppm))
    return grouped


def main() -> int:
    args = parse_args()
    rels = parse_rels(args.rel)
    branches = expand_branches(args.branches, rels)
    branch_keys = [k for k, _, _ in branches]
    branch_rel = {k: r for k, _, r in branches}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[step3] device={device}")

    _fasta, mgf_path = dc.resolve_species_files(args.root, args.species)
    model, model_config, stats, _ = dc.load_checkpoint(
        Path(args.checkpoint), device, no_packed=args.no_packed
    )

    candidates_by_ordinal = load_candidates_by_ordinal(Path(args.candidates))
    print(f"[step3] loaded candidates for {len(candidates_by_ordinal)} scans; "
          f"rels={rels} branches={branch_keys} exp_transform={args.exp_transform}")

    results_csv = out_dir / f"results_step3_{args.species}.csv"
    detail_fhs: dict[str, gzip.GzipFile] = {}
    if args.save_detail:
        for r in rels:
            d_fh = gzip.open(
                out_dir / f"candidates_detail_{args.species}_rel{rel_lvl(r)}.csv.gz",
                "wt", encoding="utf-8",
            )
            d_fh.write("ordinal\tlabel\tseq\tppm\tgc\td\tf_bidir\tf_exp\tmby\n")
            detail_fhs[rel_lvl(r)] = d_fh

    full_fhs: dict[str, gzip.GzipFile] = {}
    if args.save_full:
        full_cols = ("ordinal\tscan_id\tcharge\tpepmass\trt\tgt_seq\t"
                     "cand_idx\tseq\tlabel\tppm\tneutral_mass\t"
                     "gc\td\tf_bidir\tf_exp\tmby\t"
                     "fused\trank\tis_best\tgt_match\n")
        for key in branch_keys:
            fh = gzip.open(out_dir / f"full_{key}_{args.species}.csv.gz",
                           "wt", encoding="utf-8")
            fh.write(full_cols)
            full_fhs[key] = fh

    row_fields = (["ordinal", "scan_id", "charge", "pepmass", "rt", "gt_seq",
                   "n_candidates", "n_target", "n_decoy"]
                  + [f"{k}_seq" for k in branch_keys]
                  + [f"{k}_label" for k in branch_keys]
                  + [f"{k}_fused" for k in branch_keys]
                  + [f"{k}_gt_match" for k in branch_keys])

    n_spectra = 0
    n_processed = 0
    n_skipped = 0
    branch_top1_decoy = {k: 0 for k in branch_keys}
    branch_top1_gt = {k: 0 for k in branch_keys}
    comp_idx = {c: fc.SCORE_COMPONENTS.index(c) for c in fc.SCORE_COMPONENTS}

    total_spectra = args.max_scans or dc.count_mgf_spectra(mgf_path)
    with open(results_csv, "w", encoding="utf-8", newline="") as fout_csv:
        writer = csv.DictWriter(fout_csv, fieldnames=row_fields, extrasaction="ignore")
        writer.writeheader()

        for ordinal, (meta, peaks) in enumerate(
            dc.progress(dc.iter_mgf_spectra(mgf_path, max_scans=args.max_scans),
                        total=total_spectra, desc="step3 score", unit="spec",
                        mininterval=1.0)
        ):
            n_spectra += 1
            charge = dc.parse_charge_from_meta(meta)
            pepmass = dc.parse_pepmass_from_meta(meta)
            rt = float(meta.get("RTINSECONDS", dc.DEFAULT_RT))
            gt_seq = meta.get("SEQ", "")
            cands = candidates_by_ordinal.get(ordinal, [])
            if charge <= 0 or pepmass <= 0 or not cands:
                n_skipped += 1
                continue

            seqs = [c[0] for c in cands]
            labels = [c[1] for c in cands]
            exp_peaks = fc.top_n_exp_peaks(
                dc.transform_exp_intensity(peaks, method=args.exp_transform),
                args.exp_top_peaks,
            )
            if exp_peaks.size == 0:
                n_skipped += 1
                continue

            pred, input_grids, valid_masks = fc.predict_candidate_grids(
                model, model_config, stats, device, seqs, charge, rt,
                args.min_mz, args.batch_size,
            )
            n_processed += 1

            # Five components per candidate per rel level (gc identical across rels;
            # d/f_bidir/f_exp/mby are recomputed at each rel filter).
            scores_by_rel: dict[float, np.ndarray] = {}
            for r in rels:
                mat = np.zeros((len(seqs), len(fc.SCORE_COMPONENTS)), dtype=np.float64)
                for i, seq in enumerate(seqs):
                    sc = fc.score_candidate(pred[i], input_grids[i], valid_masks[i],
                                            exp_peaks, r, args.min_mz, tol=args.tol)
                    for j, c in enumerate(fc.SCORE_COMPONENTS):
                        mat[i, j] = sc[c]
                scores_by_rel[r] = mat

            if detail_fhs:
                for r in rels:
                    fh = detail_fhs[rel_lvl(r)]
                    for i, c in enumerate(cands):
                        fh.write(f"{ordinal}\t{labels[i]}\t{seqs[i]}\t{c[2]:.4f}\t"
                                 + "\t".join(f"{scores_by_rel[r][i, comp_idx[cc]]:.6f}"
                                             for cc in fc.SCORE_COMPONENTS)
                                 + "\n")

            scan_id = meta.get("SCANS", str(ordinal))
            if args.save_full:
                masses = [fc.neutral_mass(s) for s in seqs]
                gt_matches = [bool(gt_seq) and dc.peptide_matches(s, gt_seq) for s in seqs]

            row = {
                "ordinal": ordinal,
                "scan_id": scan_id,
                "charge": charge,
                "pepmass": round(pepmass, 6),
                "rt": round(rt, 3),
                "gt_seq": gt_seq,
                "n_candidates": len(seqs),
                "n_target": sum(1 for lab in labels if lab == fc.LABEL_TARGET),
                "n_decoy": sum(1 for lab in labels if lab == fc.LABEL_DECOY),
            }
            for key, comps, r in branches:
                idxs = [comp_idx[c] for c in comps]
                fused = scores_by_rel[r][:, idxs].mean(axis=1)
                best = int(np.argmax(fused))
                row[f"{key}_seq"] = seqs[best]
                row[f"{key}_label"] = labels[best]
                row[f"{key}_fused"] = round(float(fused[best]), 6)
                gm = bool(gt_seq) and dc.peptide_matches(seqs[best], gt_seq)
                row[f"{key}_gt_match"] = int(gm)
                if labels[best] == fc.LABEL_DECOY:
                    branch_top1_decoy[key] += 1
                elif gm:
                    branch_top1_gt[key] += 1
                if args.save_full:
                    rank = np.empty(len(seqs), dtype=np.int64)
                    order = np.argsort(-fused, kind="stable")
                    rank[order] = np.arange(1, len(seqs) + 1)
                    fh = full_fhs[key]
                    for i, seq in enumerate(seqs):
                        fh.write(
                            f"{ordinal}\t{scan_id}\t{charge}\t{pepmass:.6f}\t{rt:.3f}\t{gt_seq}\t"
                            f"{i}\t{seq}\t{labels[i]}\t{cands[i][2]:.4f}\t{masses[i]:.6f}\t"
                            + "\t".join(f"{scores_by_rel[r][i, comp_idx[cc]]:.6f}"
                                        for cc in fc.SCORE_COMPONENTS)
                            + f"\t{fused[i]:.6f}\t{int(rank[i])}\t{int(rank[i] == 1)}"
                            f"\t{int(gt_matches[i])}\n"
                        )
            writer.writerow(row)

    for fh in full_fhs.values():
        fh.close()
    for fh in detail_fhs.values():
        fh.close()

    summary = {
        "species": args.species,
        "n_spectra": n_spectra,
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "rel": args.rel,
        "rel_levels": rels,
        "exp_transform": args.exp_transform,
        "exp_top_peaks": args.exp_top_peaks,
        "branches": {
            k: {
                "top1_target_rate": round((n_processed - branch_top1_decoy[k]) / n_processed, 6)
                if n_processed else 0.0,
                "top1_decoy": branch_top1_decoy[k],
                "top1_gt_match": branch_top1_gt[k],
                "top1_gt_rate": round(branch_top1_gt[k] / n_processed, 6) if n_processed else 0.0,
            }
            for k in branch_keys
        },
        "results_csv": str(results_csv),
        "save_full": args.save_full,
        "full_csvs": {key: str(out_dir / f"full_{key}_{args.species}.csv.gz")
                      for key in branch_keys} if args.save_full else {},
        **timer.summary(),
    }
    with open(out_dir / f"summary_step3_{args.species}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    record = {
        "step": 3,
        "species": args.species,
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
        "summary": summary,
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_step3_{args.species}.json", record)

    print(f"[step3] {args.species}: {n_processed} spectra scored")
    for k in branch_keys:
        s = summary["branches"][k]
        print(f"  branch {k:>16}: top1_target_rate={s['top1_target_rate']:.4f} "
              f"top1_decoy={s['top1_decoy']} top1_gt_match={s['top1_gt_match']}")
    print(f"[step3] results -> {results_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
