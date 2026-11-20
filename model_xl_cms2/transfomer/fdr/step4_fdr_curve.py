#!/usr/bin/env python3
"""Step 4 — target-decoy FDR curves per scoring branch + 1% identification counts.

Two input modes:

1) Normal (from step3 results): reads the wide per-spectrum results CSV
   (one row per spectrum, per-branch top-1 label T/D and raw fused score),
   ranks each branch by its own score and walks the target-decoy competition.

2) Recompute (--full <full_*.csv.gz>): reads a step3 per-candidate FULL CSV
   (any branch's — they all carry the 5 component scores gc/d/f_bidir/f_exp/mby
   for EVERY candidate), aggregates the per-spectrum top-1 for ANY new branch
   definition given in --branches, then runs the same FDR curves.  This costs
   seconds locally and needs NO GPU re-run, so combinations like gc,f_exp or
   d,gc,f_exp can be explored freely.

    FDR (Elias-Gygi) = cumulative_decoy / cumulative_target

For every branch it writes a separate FDR curve and reports the 1% line (largest
high-scoring prefix whose q-value <= 0.01): n_target (identifications),
n_decoy, and n_target_gt_match (target top-1s that mass-match the MGF GT).

Writes:
    <out-dir>/fdr_curve_<species>_<branch>.csv   per branch, sorted, FDR/q-values
    <out-dir>/summary_fdr_<species>.json         per-branch 1%/5%/10% counts
    <out-dir>/records/record_step4_<species>.json

Usage:
    # normal (auto-detect branches from step3 results)
    python step4_fdr_curve.py --species mouse --out-dir <fdr>/output/mouse

    # recompute any branch combo from a saved full CSV (no GPU)
    python step4_fdr_curve.py --species mouse --out-dir <fdr>/output/mouse \
        --full <fdr>/output/mouse/full_d_gc_f_bidir_mouse.csv.gz \
        --branches 'gc,f_exp;d,gc,f_exp;d,f_exp;gc;d'
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import fdr_common as fc
import ds_common as dc  # noqa: F401  (path bootstrap + helpers)

DEFAULT_RECOMPUTE_BRANCHES = "gc,f_exp;d,gc,f_exp;d,f_exp;d,f_bidir;d,gc,f_bidir;gc;d"


def _bb_len(anno: str) -> int:
    return sum(1 for ch in anno if ch.isalpha())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--results", default=None,
                   help="per-spectrum CSV from step3 (default <out-dir>/results_step3_<sp>.csv)")
    p.add_argument("--full", default=None,
                   help="per-candidate FULL CSV.gz from step3 --save-full; recompute ANY branch "
                        "combination locally without the GPU (--branches is the branch spec)")
    p.add_argument("--branches", default="",
                   help="normal mode: comma-separated branch KEYS (empty = auto-detect from "
                        "_label columns); --full mode: ';'-separated branches, each a "
                        "','-separated subset of gc,d,f_bidir,f_exp,mby")
    p.add_argument("--fdr-targets", default="0.01,0.05,0.10",
                   help="comma-separated FDR lines to report identification counts at")
    p.add_argument("--max-len", type=int, default=None,
                   help="only keep candidates with backbone <= max_len (recompute "
                        "mode only; e.g. 30 to align with Prosit).  In normal mode "
                        "this is not possible (folded per-spectrum best only).")
    p.add_argument("--max-gt-len", type=int, default=None,
                   help="normal/full mode: drop spectra whose GT backbone > max_gt_len "
                        "(<=30 口径的第一步过滤；候选池长度同质 -> 等价于逐候选过滤，"
                        "误差仅极少数 best>30 的谱)")
    return p.parse_args()


def parse_branch_spec(s: str) -> list[tuple[str, tuple[str, ...]]]:
    """'gc,f_exp;d,gc' -> [(gc_f_exp,(gc,f_exp)), (d_gc,(d,gc))]."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for part in s.split(";"):
        comps = tuple(c.strip() for c in part.split(",") if c.strip())
        if not comps:
            continue
        for c in comps:
            if c not in fc.SCORE_COMPONENTS:
                raise SystemExit(f"unknown component '{c}'; choose from {fc.SCORE_COMPONENTS}")
        out.append(("_".join(comps), comps))
    return out


def build_rows_from_full(full_path: Path, branches: list[tuple[str, tuple[str, ...]]],
                         max_len: int | None = None) -> tuple[list[dict], list[str]]:
    """Aggregate per-candidate full CSV -> per-spectrum top-1 rows for each branch.

    ``max_len`` restricts each spectrum's candidate pool to backbones <= max_len
    (e.g. 30 to align with Prosit) before the per-branch argmax / FDR ranking.
    """
    comp_keys = fc.SCORE_COMPONENTS
    comp_idx = {c: j for j, c in enumerate(comp_keys)}
    groups: dict[int, list[dict]] = defaultdict(list)
    with gzip.open(full_path, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            groups[int(r["ordinal"])].append(r)

    def keep(c) -> bool:
        if max_len is None:
            return True
        return sum(1 for ch in c["seq"] if ch.isalpha()) <= max_len

    rows: list[dict] = []
    for ordinal, cands in dc.progress(sorted(groups.items()), total=len(groups),
                                      desc="aggregate top1", unit="spec", mininterval=1.0):
        cands = [c for c in cands if keep(c)]
        if not cands:
            continue
        K = len(cands)
        scores = np.zeros((K, len(comp_keys)), dtype=np.float64)
        for i, c in enumerate(cands):
            for j, cc in enumerate(comp_keys):
                scores[i, j] = float(c[cc])
        row = {
            "ordinal": ordinal,
            "scan_id": cands[0]["scan_id"],
            "charge": int(cands[0]["charge"]),
            "pepmass": float(cands[0]["pepmass"]),
            "rt": float(cands[0]["rt"]),
            "gt_seq": cands[0]["gt_seq"],
            "n_target": sum(1 for c in cands if c["label"] == fc.LABEL_TARGET),
            "n_decoy": sum(1 for c in cands if c["label"] == fc.LABEL_DECOY),
        }
        for key, comps in branches:
            idxs = [comp_idx[c] for c in comps]
            fused = scores[:, idxs].mean(axis=1)
            best = int(np.argmax(fused))
            row[f"{key}_seq"] = cands[best]["seq"]
            row[f"{key}_label"] = cands[best]["label"]
            row[f"{key}_fused"] = float(fused[best])
            row[f"{key}_gt_match"] = int(cands[best].get("gt_match", 0) or 0)
        rows.append(row)
    return rows, [k for k, _ in branches]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    if args.full:
        full_path = Path(args.full)
        if not full_path.exists():
            raise SystemExit(f"--full CSV not found: {full_path}")
        spec = args.branches or DEFAULT_RECOMPUTE_BRANCHES
        branches = parse_branch_spec(spec)
        print(f"[step4] recompute mode: {full_path} -> branches={[k for k, _ in branches]}"
              + (f" max_len={args.max_len}" if args.max_len else ""))
        rows, branch_keys = build_rows_from_full(full_path, branches, args.max_len)
        if args.max_gt_len is not None:
            rows = [r for r in rows if _bb_len(r.get("gt_seq", "")) <= args.max_gt_len]
        header = list(rows[0]) if rows else []
    else:
        results_csv = Path(args.results) if args.results else out_dir / f"results_step3_{args.species}.csv"
        if not results_csv.exists():
            raise SystemExit(f"results CSV not found: {results_csv}")

        with open(results_csv, encoding="utf-8") as _fh:
            n_rows = max(0, sum(1 for _line in _fh) - 1)

        rows = []
        with open(results_csv, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            for r in dc.progress(reader, total=n_rows, desc="step4 read results",
                                 unit="row", mininterval=1.0):
                r["ordinal"] = int(r["ordinal"])
                r["charge"] = int(r["charge"])
                r["n_target"] = int(r["n_target"])
                r["n_decoy"] = int(r["n_decoy"])
                rows.append(r)

        if args.max_gt_len is not None:
            n0 = len(rows)
            rows = [r for r in rows if _bb_len(r.get("gt_seq", "")) <= args.max_gt_len]
            print(f"[step4] max-gt-len {args.max_gt_len}: {n0} -> {len(rows)} spectra")

        if args.branches:
            branch_keys = [b.strip() for b in args.branches.split(",") if b.strip()]
        else:
            branch_keys = [c[:-6] for c in header if c.endswith("_label")]
        if not branch_keys:
            raise SystemExit("no branch columns found in results CSV; expected *_label columns")
        for key in branch_keys:
            if f"{key}_fused" not in header or f"{key}_label" not in header:
                raise SystemExit(f"branch '{key}' missing {key}_fused/{key}_label columns in results CSV")
        for key in branch_keys:
            fcol = f"{key}_fused"
            gcol = f"{key}_gt_match"
            for r in rows:
                r[fcol] = float(r[fcol])
                if gcol in r and r[gcol] not in ("", None):
                    r[gcol] = int(r[gcol])

    fdr_targets = [float(x.strip()) for x in args.fdr_targets.split(",") if x.strip()]
    branches_out: dict[str, dict] = {}
    for key in branch_keys:
        label_col = f"{key}_label"
        fcol = f"{key}_fused"
        sorted_rows = sorted(rows, key=lambda r: (r[fcol], -r["ordinal"]), reverse=True)
        curve = fc.compute_fdr_curve(sorted_rows, label_key=label_col, score_key=fcol)
        curve_path = out_dir / f"fdr_curve_{args.species}_{key}.csv"
        fc.write_fdr_csv(curve, curve_path)
        lines = {t: fc.accepted_at_fdr(curve, t, label_key=label_col, score_key=fcol,
                                       gt_key=f"{key}_gt_match")
                 for t in fdr_targets}
        branches_out[key] = {
            "rank_col": fcol,
            "n_spectra": len(curve),
            "n_top1_target": sum(1 for r in curve if r[label_col] == fc.LABEL_TARGET),
            "n_top1_decoy": sum(1 for r in curve if r[label_col] == fc.LABEL_DECOY),
            "fdr_lines": lines,
            "fdr_curve_csv": str(curve_path),
        }

    summary = {
        "species": args.species,
        "fdr_targets": [float(t) for t in fdr_targets],
        "recompute_from": str(Path(args.full)) if args.full else None,
        "branches": branches_out,
        **timer.summary(),
    }
    with open(out_dir / f"summary_fdr_{args.species}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    dc.write_record(out_dir / "records" / f"record_step4_{args.species}.json", summary)

    # Comparison table sorted by n_target at the first (strictest) FDR line.
    t1 = fdr_targets[0] if fdr_targets else 0.01
    order = sorted(branch_keys, key=lambda k: -branches_out[k]["fdr_lines"][t1]["n_target"])

    print(f"\n===== FDR comparison ({args.species}) =====")
    print(f"spectra ranked: {len(rows)}   FDR line highlighted: {t1:.1%}")
    print(f"{'branch':>16} {'n_target@1%':>12} {'n_decoy@1%':>10} {'n_gt@1%':>9} "
          f"{'top1_tgt':>9} {'top1_decoy':>10}")
    for key in order:
        b = branches_out[key]
        l1 = b["fdr_lines"][t1]
        print(f"{key:>16} {l1['n_target']:>12} {l1['n_decoy']:>10} {l1['n_target_gt_match']:>9} "
              f"{b['n_top1_target']:>9} {b['n_top1_decoy']:>10}")
    print("\ndetail per branch (score threshold, achieved FDR) -> "
          f"{out_dir / ('summary_fdr_' + args.species + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
