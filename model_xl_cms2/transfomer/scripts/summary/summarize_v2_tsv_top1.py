#!/usr/bin/env python3
"""Summarize PrimeNovo TSV top-1 accuracy for the full v2 suite."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
import sys

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_TRANSFORMER_DIR = _SCRIPTS_DIR.parent
if str(_TRANSFORMER_DIR) not in sys.path: sys.path.insert(0, str(_TRANSFORMER_DIR))
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))

from candidate_rescoring import _peptide_matches, _aa_match_prefix_suffix, _tokenize_cached
from mgf_row_policy import (
    MgfRowPolicy,
    build_mgf_row_catalog,
    build_occurrence_aligned_pairs,
    collect_filtered_row_order_sequences,
)


DEFAULT_SPECIES = [
    "mouse",
    "human",
    "honeybee",
    "tomato",
    "ricebean",
    "mmazei",
    "clambacteria",
    "bacillus_1",
    "bacillus_2",
    "yeast",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize PrimeNovo TSV top-1 on v2 species.")
    p.add_argument("--base", type=Path, required=True,
                   help="Base directory containing v2_*.tsv files.")
    p.add_argument("--mgf-root", type=Path, required=True,
                   help="Root directory of 90982 species MGF folders.")
    p.add_argument("--species", nargs="+", default=DEFAULT_SPECIES)
    p.add_argument("--out-prefix", type=Path, required=True,
                   help="Output prefix for CSV / JSON / MD.")
    p.add_argument("--mmazei-drop-seq-with-o", action="store_true",
                   help="Delete mmazei spectra whose SEQ contains illegal amino acid O before row-order alignment.")
    p.add_argument("--ricebean-precursor-mismatch-mode", type=str, default="off",
                   choices=("off", "fix", "drop"),
                   help="How to handle Vigna-mungo precursor/SEQ mass mismatches.")
    p.add_argument("--ricebean-precursor-mismatch-ppm", type=float, default=50.0,
                   help="PPM threshold for Vigna-mungo precursor mismatch handling.")
    return p.parse_args()


def _policy_for_species(species: str, args: argparse.Namespace) -> MgfRowPolicy:
    if species == "mmazei":
        return MgfRowPolicy(drop_seq_with_o=bool(args.mmazei_drop_seq_with_o))
    if species == "ricebean":
        return MgfRowPolicy(
            precursor_mismatch_mode=str(args.ricebean_precursor_mismatch_mode).lower(),
            precursor_mismatch_ppm=float(args.ricebean_precursor_mismatch_ppm),
        )
    return MgfRowPolicy()


def _alignment_mode_for_species(species: str) -> str:
    return "occurrence_aware" if species == "mmazei" else "row_order"


def _prepare_species_mgf_dir(species: str, mgf_root: Path, tmp_root: Path) -> Path:
    mapping = {
        "mouse": [mgf_root / "Mus_mouse"],
        "human": [mgf_root / "H.-sapiens"],
        "honeybee": [mgf_root / "Apis-mellifera"],
        "tomato": [mgf_root / "Solanum-lycopersicum"],
        "ricebean": [mgf_root / "Vigna-mungo"],
        "mmazei": [mgf_root / "Methanosarcina-mazei"],
        "clambacteria": [mgf_root / "Candidatus-endoloripes"],
        "bacillus_1": [mgf_root / "Bacillus_1", mgf_root / "Bacillus_2"],
        "bacillus_2": [mgf_root / "Bacillus_3", mgf_root / "Bacillus_4"],
        "yeast": [
            mgf_root / "Saccharomyces-cerevisiae-yeast_1",
            mgf_root / "Saccharomyces-cerevisiae-yeast_2",
            mgf_root / "Saccharomyces-cerevisiae-yeast_3",
            mgf_root / "Saccharomyces-cerevisiae-yeast_4",
        ],
    }
    paths = mapping[species]
    if len(paths) == 1:
        return paths[0]

    merged = tmp_root / species
    merged.mkdir(parents=True, exist_ok=True)
    existing = list(merged.glob("*.mgf"))
    if not existing:
        for src in paths:
            for mgf in sorted(src.glob("*.mgf")):
                shutil.copy2(mgf, merged / mgf.name)
    return merged


def _compute_species_top1(
    species: str,
    tsv_path: Path,
    mgf_dir: Path,
    row_policy: MgfRowPolicy,
) -> tuple[int, int, float, float, float, dict[str, int]]:
    tsv = pd.read_csv(tsv_path, sep="\t")
    alignment_mode = _alignment_mode_for_species(species)

    if alignment_mode == "occurrence_aware":
        catalog, stats = build_mgf_row_catalog(mgf_dir, row_policy)
        pairs, occurrence_diag = build_occurrence_aligned_pairs(tsv["label"].astype(str).tolist(), catalog)
        if not pairs:
            return 0, 0, 0.0, 0.0, 0.0, {
                "mgf_drop_seq_o": int(stats.dropped_seq_with_o),
                "mgf_drop_precursor": int(stats.dropped_precursor_mismatch),
                "mgf_fix_precursor": int(stats.fixed_precursor_mismatch),
                "alignment_mode": alignment_mode,
                "occurrence_missing": int(occurrence_diag["missing_labels"]),
                "occurrence_overflow": int(occurrence_diag["overflow_reuse"]),
                "occurrence_skipped_dropped": int(occurrence_diag["skipped_dropped_rows"]),
            }

        row_indices = [row_idx for row_idx, _entry in pairs if row_idx < len(tsv)]
        filtered_tsv = tsv.iloc[row_indices].reset_index(drop=True)
        filtered_mgf_seqs = [entry.seq for row_idx, entry in pairs if row_idx < len(tsv)]
        n = min(len(filtered_tsv), len(filtered_mgf_seqs))
        correct = 0
        aa_matched_total = 0; aa_truth_total = 0; aa_pred_total = 0
        for i in range(n):
            pred = str(filtered_tsv.iloc[i]["prediction"])
            truth = str(filtered_mgf_seqs[i])
            if _peptide_matches(pred, truth): correct += 1
            pt = _tokenize_cached(pred); tt = _tokenize_cached(truth)
            if pt and tt:
                m, _ = _aa_match_prefix_suffix(list(pt), list(tt))
                aa_matched_total += int(sum(m))
                aa_truth_total += len(tt); aa_pred_total += len(pt)
        aa_rec = aa_matched_total / aa_truth_total if aa_truth_total > 0 else 0.0
        aa_prec = aa_matched_total / aa_pred_total if aa_pred_total > 0 else 0.0
        return correct, n, (correct / n) if n else 0.0, aa_rec, aa_prec, {
            "mgf_drop_seq_o": int(stats.dropped_seq_with_o),
            "mgf_drop_precursor": int(stats.dropped_precursor_mismatch),
            "mgf_fix_precursor": int(stats.fixed_precursor_mismatch),
            "alignment_mode": alignment_mode,
            "occurrence_missing": int(occurrence_diag["missing_labels"]),
            "occurrence_overflow": int(occurrence_diag["overflow_reuse"]),
            "occurrence_skipped_dropped": int(occurrence_diag["skipped_dropped_rows"]),
        }

    keep_ordinals, mgf_seqs, stats = collect_filtered_row_order_sequences(mgf_dir, row_policy)
    aligned_pairs = [
        (ordinal, seq)
        for ordinal, seq in zip(keep_ordinals, mgf_seqs)
        if ordinal < len(tsv)
    ]
    if not aligned_pairs:
        return 0, 0, 0.0, 0.0, 0.0, {
            "mgf_drop_seq_o": int(stats.dropped_seq_with_o),
            "mgf_drop_precursor": int(stats.dropped_precursor_mismatch),
            "mgf_fix_precursor": int(stats.fixed_precursor_mismatch),
            "alignment_mode": alignment_mode,
            "occurrence_missing": 0,
            "occurrence_overflow": 0,
            "occurrence_skipped_dropped": 0,
        }

    filtered_tsv = tsv.iloc[[ordinal for ordinal, _seq in aligned_pairs]].reset_index(drop=True)
    filtered_mgf_seqs = [seq for _ordinal, seq in aligned_pairs]
    n = min(len(filtered_tsv), len(filtered_mgf_seqs))
    correct = 0
    aa_matched_total = 0; aa_truth_total = 0; aa_pred_total = 0
    for i in range(n):
        pred = str(filtered_tsv.iloc[i]["prediction"])
        truth = str(filtered_mgf_seqs[i])
        if _peptide_matches(pred, truth): correct += 1
        pt = _tokenize_cached(pred); tt = _tokenize_cached(truth)
        if pt and tt:
            m, _ = _aa_match_prefix_suffix(list(pt), list(tt))
            aa_matched_total += int(sum(m))
            aa_truth_total += len(tt); aa_pred_total += len(pt)
    aa_rec = aa_matched_total / aa_truth_total if aa_truth_total > 0 else 0.0
    aa_prec = aa_matched_total / aa_pred_total if aa_pred_total > 0 else 0.0
    return correct, n, (correct / n) if n else 0.0, aa_rec, aa_prec, {
        "mgf_drop_seq_o": int(stats.dropped_seq_with_o),
        "mgf_drop_precursor": int(stats.dropped_precursor_mismatch),
        "mgf_fix_precursor": int(stats.fixed_precursor_mismatch),
        "alignment_mode": alignment_mode,
        "occurrence_missing": 0,
        "occurrence_overflow": 0,
        "occurrence_skipped_dropped": 0,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    args = parse_args()
    rows = []

    total_correct = 0
    total_n = 0

    with tempfile.TemporaryDirectory(prefix="v2_tsv_mgf_", dir="/home/wangyuhang/tmp") as tmpdir:
        tmp_root = Path(tmpdir)
        for sp in args.species:
            tsv_path = args.base / f"v2_{sp}.tsv"
            if not tsv_path.exists():
                continue
            mgf_dir = _prepare_species_mgf_dir(sp, args.mgf_root, tmp_root)
            row_policy = _policy_for_species(sp, args)
            correct, n, rate, aa_rec, aa_prec, diag = _compute_species_top1(sp, tsv_path, mgf_dir, row_policy)
            rows.append({
                "species": sp,
                "tsv": str(tsv_path),
                "mgf_dir": str(mgf_dir),
                "n_rows_compared": n,
                "top1_correct": correct,
                "top1_rate": rate,
                "aa_recall": round(aa_rec, 4),
                "aa_precision": round(aa_prec, 4),
                "mgf_drop_seq_o": diag["mgf_drop_seq_o"],
                "mgf_drop_precursor": diag["mgf_drop_precursor"],
                "mgf_fix_precursor": diag["mgf_fix_precursor"],
                "alignment_mode": diag["alignment_mode"],
                "occurrence_missing": diag["occurrence_missing"],
                "occurrence_overflow": diag["occurrence_overflow"],
                "occurrence_skipped_dropped": diag["occurrence_skipped_dropped"],
            })
            total_correct += correct
            total_n += n

    rows.append({
        "species": "TOTAL",
        "tsv": "",
        "mgf_dir": "",
        "n_rows_compared": total_n,
        "top1_correct": total_correct,
        "top1_rate": (total_correct / total_n) if total_n else 0.0,
        "aa_recall": None,
        "aa_precision": None,
        "mgf_drop_seq_o": None,
        "mgf_drop_precursor": None,
        "mgf_fix_precursor": None,
        "alignment_mode": "",
        "occurrence_missing": None,
        "occurrence_overflow": None,
        "occurrence_skipped_dropped": None,
    })

    df = pd.DataFrame(rows)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix(".csv")
    json_path = args.out_prefix.with_suffix(".json")
    md_path = args.out_prefix.with_suffix(".md")

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# V2 TSV Top1 Summary",
        "",
        "| Species | Compared Rows | Top1 Correct | Top1 Rate | AA Recall | AA Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['species']} | {int(row['n_rows_compared'])} | {int(row['top1_correct'])} | {_fmt_pct(float(row['top1_rate']))} | {_fmt_pct(float(row['aa_recall']))} | {_fmt_pct(float(row['aa_precision']))} |"
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
