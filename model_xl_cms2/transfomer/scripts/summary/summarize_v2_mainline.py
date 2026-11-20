#!/usr/bin/env python3
"""Summarize full v2 mainline rescoring outputs into a single table.

Features:
- reads per-species rescoring_summary.json
- reports GT hit/top1 metrics
- optionally computes PrimeNovo TSV top-1 from matching TSV + MGF
- writes CSV / JSON / Markdown summary tables

Typical use:

  python scripts/summarize_v2_mainline.py \
    --output-base /home/wangyuhang/seq2mass/v20/transfomer/v2_mainline_outputs \
    --out-prefix /home/wangyuhang/seq2mass/v20/transfomer/v2_mainline_summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_TRANSFORMER_DIR = _SCRIPTS_DIR.parent
if str(_TRANSFORMER_DIR) not in sys.path: sys.path.insert(0, str(_TRANSFORMER_DIR))
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

from candidate_rescoring import _peptide_matches
from mgf_row_policy import (
    MgfRowPolicy,
    build_mgf_row_catalog,
    build_occurrence_aligned_pairs,
    collect_filtered_row_order_sequences,
)


DEFAULT_SPECIES = [
    "bacillus_1",
    "bacillus_2",
    "yeast",
    "mmazei",
    "honeybee",
    "tomato",
    "ricebean",
    "clambacteria",
    "human",
    "mouse",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize v2 mainline rescoring outputs.")
    p.add_argument("--output-base", type=Path, required=True,
                   help="Directory containing per-species output dirs.")
    p.add_argument("--species", nargs="+", default=DEFAULT_SPECIES)
    p.add_argument("--out-prefix", type=Path, required=True,
                   help="Output prefix for .csv / .json / .md files.")
    p.add_argument("--skip-pn", action="store_true",
                   help="Skip PrimeNovo TSV top-1 computation.")
    return p.parse_args()


def _row_policy_from_summary(summary: dict) -> MgfRowPolicy:
    cfg = summary.get("mgf_row_policy") or {}
    return MgfRowPolicy(
        drop_seq_with_o=bool(cfg.get("drop_seq_with_o", False)),
        precursor_mismatch_mode=str(cfg.get("precursor_mismatch_mode", "off")).lower(),
        precursor_mismatch_ppm=float(cfg.get("precursor_mismatch_ppm", 50.0)),
    )


def _alignment_mode_from_summary(summary: dict) -> str:
    mode = str(summary.get("mgf_row_alignment_mode", "")).lower()
    if mode:
        return mode
    parquet_text = str(summary.get("parquet", "")).lower()
    mgf_text = str(summary.get("mgf_dir", "")).lower()
    if "mmazei" in parquet_text or "methanosarcina-mazei" in mgf_text:
        return "occurrence_aware"
    return "row_order"


def _compute_pn_top1(
    tsv_path: Path,
    mgf_dir: Path,
    row_policy: MgfRowPolicy,
    alignment_mode: str,
) -> tuple[int, int, float]:
    tsv = pd.read_csv(tsv_path, sep="\t")
    if str(alignment_mode).lower() == "occurrence_aware":
        catalog, _stats = build_mgf_row_catalog(mgf_dir, row_policy)
        pairs, _diag = build_occurrence_aligned_pairs(tsv["label"].astype(str).tolist(), catalog)
        if not pairs:
            return 0, 0, 0.0
        row_indices = [row_idx for row_idx, _entry in pairs if row_idx < len(tsv)]
        filtered_tsv = tsv.iloc[row_indices].reset_index(drop=True)
        filtered_mgf_seqs = [entry.seq for row_idx, entry in pairs if row_idx < len(tsv)]
        n = min(len(filtered_tsv), len(filtered_mgf_seqs))
        correct = 0
        for i in range(n):
            if _peptide_matches(str(filtered_tsv.iloc[i]["prediction"]), str(filtered_mgf_seqs[i])):
                correct += 1
        return correct, n, (correct / n) if n else 0.0

    keep_ordinals, mgf_seqs, _stats = collect_filtered_row_order_sequences(mgf_dir, row_policy)
    aligned_pairs = [
        (ordinal, seq)
        for ordinal, seq in zip(keep_ordinals, mgf_seqs)
        if ordinal < len(tsv)
    ]
    if not aligned_pairs:
        return 0, 0, 0.0

    filtered_tsv = tsv.iloc[[ordinal for ordinal, _seq in aligned_pairs]].reset_index(drop=True)
    filtered_mgf_seqs = [seq for _ordinal, seq in aligned_pairs]
    n = min(len(filtered_tsv), len(filtered_mgf_seqs))
    correct = 0
    for i in range(n):
        if _peptide_matches(str(filtered_tsv.iloc[i]["prediction"]), str(filtered_mgf_seqs[i])):
            correct += 1
    return correct, n, (correct / n) if n else 0.0


def _format_pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x * 100:.2f}%"


def main() -> int:
    args = parse_args()
    rows = []

    total_scans = 0
    total_processed = 0
    total_gt_hit = 0
    total_gt_top1 = 0
    total_pn_correct = 0
    total_pn_n = 0

    for sp in args.species:
        out_dir = args.output_base / sp
        summary_path = out_dir / "rescoring_summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parquet_path = Path(summary["parquet"])
        mgf_dir = Path(summary["mgf_dir"])
        tsv_path = parquet_path.with_suffix(".tsv")
        row_policy = _row_policy_from_summary(summary)
        alignment_mode = _alignment_mode_from_summary(summary)

        pn_correct = pn_n = 0
        pn_rate = None
        if (not args.skip_pn) and tsv_path.exists():
            try:
                pn_correct, pn_n, pn_rate = _compute_pn_top1(tsv_path, mgf_dir, row_policy, alignment_mode)
            except Exception:
                pn_correct, pn_n, pn_rate = 0, 0, None

        n_scans = int(summary["n_scans_in_parquet"])
        n_processed = int(summary["n_processed"])
        n_gt_hit = int(summary["n_gt_hit"])
        n_gt_top1 = int(summary["gt_top1"])
        gt_hit_rate = float(summary["gt_hit_rate"])
        gt_top1_rate = float(summary["gt_top1_rate"])
        delta_vs_pn = (gt_top1_rate - pn_rate) if pn_rate is not None else None

        rows.append({
            "species": sp,
            "parquet": str(parquet_path),
            "mgf_dir": str(mgf_dir),
            "n_scans": n_scans,
            "n_processed": n_processed,
            "n_gt_hit": n_gt_hit,
            "gt_hit_rate": gt_hit_rate,
            "n_gt_top1": n_gt_top1,
            "gt_top1_rate": gt_top1_rate,
            "pn_top1_correct": pn_correct if pn_rate is not None else None,
            "pn_top1_rate": pn_rate,
            "delta_vs_pn": delta_vs_pn,
            "mgf_alignment_mode": alignment_mode,
            "mgf_drop_seq_o": summary.get("mgf_dropped_seq_with_o"),
            "mgf_drop_precursor": summary.get("mgf_dropped_precursor_mismatch"),
            "mgf_fix_precursor": summary.get("mgf_fixed_precursor_mismatch"),
        })

        total_scans += n_scans
        total_processed += n_processed
        total_gt_hit += n_gt_hit
        total_gt_top1 += n_gt_top1
        total_pn_correct += pn_correct
        total_pn_n += pn_n

    if not rows:
        raise FileNotFoundError("No rescoring_summary.json files found for requested species.")

    total_row = {
        "species": "TOTAL",
        "parquet": "",
        "mgf_dir": "",
        "n_scans": total_scans,
        "n_processed": total_processed,
        "n_gt_hit": total_gt_hit,
        "gt_hit_rate": total_gt_hit / total_processed if total_processed else 0.0,
        "n_gt_top1": total_gt_top1,
        "gt_top1_rate": total_gt_top1 / total_processed if total_processed else 0.0,
        "pn_top1_correct": total_pn_correct if total_pn_n > 0 else None,
        "pn_top1_rate": (total_pn_correct / total_pn_n) if total_pn_n > 0 else None,
        "delta_vs_pn": ((total_gt_top1 / total_processed) - (total_pn_correct / total_pn_n))
        if total_processed > 0 and total_pn_n > 0 else None,
    }
    rows.append(total_row)

    df = pd.DataFrame(rows)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix(".csv")
    json_path = args.out_prefix.with_suffix(".json")
    md_path = args.out_prefix.with_suffix(".md")

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# V2 Mainline Summary",
        "",
        "| Species | Scans | GT hit | GT hit rate | GT top1 | GT top1 rate | PN top1 | PN top1 rate | Delta vs PN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        pn_correct_text = "" if pd.isna(row["pn_top1_correct"]) else str(int(row["pn_top1_correct"]))
        pn_rate_text = "" if pd.isna(row["pn_top1_rate"]) else _format_pct(float(row["pn_top1_rate"]))
        delta_text = ""
        if not pd.isna(row["delta_vs_pn"]):
            delta_text = f"{float(row['delta_vs_pn']) * 100:+.2f}%"
        md_lines.append(
            f"| {row['species']} | {int(row['n_scans'])} | {int(row['n_gt_hit'])} | "
            f"{_format_pct(row['gt_hit_rate'])} | {int(row['n_gt_top1'])} | "
            f"{_format_pct(row['gt_top1_rate'])} | {pn_correct_text} | {pn_rate_text} | {delta_text} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
