#!/usr/bin/env python3
"""Select peptide-specific export thresholds from replacement cosine summaries.

The selector uses prediction-vs-all-real similarity as the primary metric and,
when available, adds a cross-peptide discrimination margin. It is intended to
run after the replacement-style PiPrime post-processing workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _threshold_key(value: Any) -> str:
    threshold = _as_float(value)
    if threshold is None:
        return str(value)
    return f"{threshold:.12g}"


def _load_margin_lookup(cross_summary: Path | None) -> dict[tuple[str, str], float]:
    if cross_summary is None:
        return {}
    if not cross_summary.exists():
        raise FileNotFoundError(f"Cross summary not found: {cross_summary}")

    cross_df = pd.read_csv(cross_summary)
    lookup: dict[tuple[str, str], float] = {}
    for _, row in cross_df.iterrows():
        threshold_key = _threshold_key(row.get("threshold"))
        for col in cross_df.columns:
            if not col.endswith("_all_real_margin"):
                continue
            peptide = col[: -len("_all_real_margin")]
            margin = _as_float(row.get(col))
            if margin is not None:
                lookup[(peptide, threshold_key)] = margin
    return lookup


def select_thresholds(
    same_summary: Path,
    cross_summary: Path | None,
    score_column: str,
    margin_weight: float,
    min_margin: float | None,
    min_peaks: int | None,
    max_peaks: int | None,
    peak_target: int | None,
    peak_penalty_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not same_summary.exists():
        raise FileNotFoundError(f"Same-peptide summary not found: {same_summary}")

    same_df = pd.read_csv(same_summary)
    required = {"peptide", "threshold", "prediction_peaks", score_column}
    missing = sorted(required - set(same_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {same_summary}: {missing}")

    margin_lookup = _load_margin_lookup(cross_summary)

    candidates = []
    for _, row in same_df.iterrows():
        peptide = str(row["peptide"])
        threshold_key = _threshold_key(row["threshold"])
        threshold = _as_float(row["threshold"])
        similarity = _as_float(row.get(score_column))
        peaks = _as_float(row.get("prediction_peaks"))
        if threshold is None or similarity is None or peaks is None:
            continue

        peaks_int = int(peaks)
        if min_peaks is not None and peaks_int < min_peaks:
            continue
        if max_peaks is not None and peaks_int > max_peaks:
            continue

        margin = margin_lookup.get((peptide, threshold_key))
        if min_margin is not None and margin is not None and margin < min_margin:
            continue

        peak_penalty = 0.0
        if peak_target is not None and peak_target > 0 and peak_penalty_weight:
            peak_penalty = abs(peaks_int - peak_target) / peak_target

        margin_bonus = margin_weight * (margin if margin is not None else 0.0)
        score = similarity + margin_bonus - peak_penalty_weight * peak_penalty

        candidate = row.to_dict()
        candidate.update(
            {
                "selection_similarity": similarity,
                "selection_margin": margin,
                "selection_margin_bonus": margin_bonus,
                "selection_peak_penalty": peak_penalty_weight * peak_penalty,
                "selection_score": score,
            }
        )
        candidates.append(candidate)

    if not candidates:
        raise ValueError("No threshold candidates remained after filtering.")

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values(
        ["peptide", "selection_score", "selection_similarity", "prediction_peaks"],
        ascending=[True, False, False, True],
    )
    selected_df = cand_df.groupby("peptide", as_index=False, sort=True).head(1)
    selected_df = selected_df.sort_values("peptide").reset_index(drop=True)
    return selected_df, cand_df.reset_index(drop=True)


def write_outputs(
    selected_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = output_dir / f"{prefix}_selected_thresholds.csv"
    candidates_csv = output_dir / f"{prefix}_threshold_candidates_ranked.csv"
    selected_json = output_dir / f"{prefix}_selected_thresholds.json"
    selected_xlsx = output_dir / f"{prefix}_selected_thresholds.xlsx"

    selected_df.to_csv(selected_csv, index=False)
    candidates_df.to_csv(candidates_csv, index=False)

    records = selected_df.to_dict(orient="records")
    selected_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    with pd.ExcelWriter(selected_xlsx, engine="openpyxl") as writer:
        selected_df.to_excel(writer, sheet_name="selected", index=False)
        candidates_df.to_excel(writer, sheet_name="ranked_candidates", index=False)

    print(f"Saved: {selected_csv}")
    print(f"Saved: {candidates_csv}")
    print(f"Saved: {selected_json}")
    print(f"Saved: {selected_xlsx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select per-peptide best export thresholds from PiPrime cosine summaries."
    )
    parser.add_argument("--same-summary", required=True, type=Path)
    parser.add_argument("--cross-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="auto")
    parser.add_argument(
        "--score-column",
        default="model_prediction_vs_all_real_mean",
        help="Primary same-peptide similarity column to maximize.",
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=0.15,
        help="Weight for same-vs-cross all_real margin when cross summary is available.",
    )
    parser.add_argument("--min-margin", type=float, default=None)
    parser.add_argument("--min-peaks", type=int, default=1)
    parser.add_argument("--max-peaks", type=int, default=None)
    parser.add_argument("--peak-target", type=int, default=None)
    parser.add_argument("--peak-penalty-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.same_summary.parent
    selected_df, candidates_df = select_thresholds(
        same_summary=args.same_summary,
        cross_summary=args.cross_summary,
        score_column=args.score_column,
        margin_weight=args.margin_weight,
        min_margin=args.min_margin,
        min_peaks=args.min_peaks,
        max_peaks=args.max_peaks,
        peak_target=args.peak_target,
        peak_penalty_weight=args.peak_penalty_weight,
    )
    write_outputs(selected_df, candidates_df, output_dir, args.prefix)

    print("\nSelected thresholds:")
    columns = [
        "peptide",
        "threshold",
        "prediction_peaks",
        "selection_similarity",
        "selection_margin",
        "selection_score",
    ]
    print(selected_df[[c for c in columns if c in selected_df.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
