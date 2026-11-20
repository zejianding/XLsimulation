"""
Batch cosine similarity summary: compute prediction-vs-original cosine for all groups.

Reads paired MGF + NPZ files (matched by filename stem), groups spectra by SOURCE tag,
computes cosine similarity matrices, and outputs a summary CSV.

The expected MGF files are those produced by collect_original_with_prediction.py:
  - SOURCE=ORIGINAL     (real experimental spectra from all species)
  - SOURCE=MODEL_PREDICTION  (the predicted spectrum, exactly one per group)

The NPZ files are produced by mgf_to_embedding_batch.py (PiPrime embeddings).

Usage:
    python batch_cosine_summary.py \
        --mgf-dir ./Original_vs_Predicted/ \
        --npz-dir ./Embeddings/ \
        --output-csv cosine_summary.csv \
        --output-xlsx cosine_summary.xlsx
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = embeddings / norms
    return normalized @ normalized.T


def mean_excluding_diagonal(submatrix: np.ndarray) -> float:
    """Mean of off-diagonal elements."""
    n = submatrix.shape[0]
    if n <= 1:
        return 0.0
    return float((np.sum(submatrix) - np.trace(submatrix)) / (n * (n - 1)))


def parse_source_from_mgf(mgf_path: str) -> list[str]:
    """Extract SOURCE tag for each spectrum in an MGF file."""
    try:
        from pyteomics import mgf
    except ImportError:
        # Fallback: regex-based parsing
        return _parse_source_fallback(mgf_path)

    sources = []
    with mgf.MGF(mgf_path) as reader:
        for spec in reader:
            params = spec.get("params", {})
            source = params.get("source", params.get("SOURCE", "UNKNOWN"))
            sources.append(str(source))
    return sources


def _parse_source_fallback(mgf_path: str) -> list[str]:
    """Fallback MGF parsing without pyteomics."""
    sources = []
    current_source = "UNKNOWN"
    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s == "BEGIN IONS":
                current_source = "UNKNOWN"
            elif s.upper().startswith("SOURCE="):
                current_source = s.split("=", 1)[1].strip()
            elif s == "END IONS":
                sources.append(current_source)
    return sources


def parse_sequence_from_mgf(mgf_path: str) -> str:
    """Extract SEQ from the first spectrum in an MGF file."""
    try:
        from pyteomics import mgf
        with mgf.MGF(mgf_path) as reader:
            for spec in reader:
                params = spec.get("params", {})
                return str(params.get("seq", params.get("SEQ", "")))
    except ImportError:
        pass

    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("SEQ="):
                return s.split("=", 1)[1].strip()
    return ""


def process_one_group(mgf_path: str, npz_path: str) -> dict | None:
    """
    Compute cosine metrics for one peptide+charge group.

    Returns dict with:
      - filename, sequence, charge
      - num_original, num_prediction
      - original_original_mean
      - prediction_vs_original_mean (mean cosine of prediction vs each original)
      - prediction_vs_original_max, prediction_vs_original_min
    """
    stem = Path(mgf_path).stem

    # Load embeddings
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    n = embeddings.shape[0]

    if n == 0:
        return None

    # Get source tags from NPZ if available, otherwise fall back to MGF
    if "sources" in data:
        sources = [str(s) for s in data["sources"]]
    else:
        sources = parse_source_from_mgf(mgf_path)
    if len(sources) != n:
        print(f"  [WARNING] {stem}: sources={len(sources)} != NPZ embeddings={n}")

    source_upper = [s.upper() for s in sources]

    # Find indices
    orig_idx = [i for i, s in enumerate(source_upper) if s == "ORIGINAL"]
    pred_idx = [i for i, s in enumerate(source_upper) if s == "MODEL_PREDICTION"]

    # Also support old tags from original script
    orig_idx += [i for i, s in enumerate(source_upper) if s.startswith("ORIGINAL_")]
    pred_idx += [i for i, s in enumerate(source_upper) if s == "MODEL_PREDICTION_LAST"]

    if not orig_idx or not pred_idx:
        print(f"  [SKIP] {stem}: ORIGINAL={len(orig_idx)}, MODEL_PREDICTION={len(pred_idx)}")
        return None

    sim = cosine_similarity_matrix(embeddings)

    # Original vs Original (intra-class similarity of real spectra)
    orig_orig_mean = None
    if len(orig_idx) >= 2:
        orig_sub = sim[np.ix_(orig_idx, orig_idx)]
        orig_orig_mean = mean_excluding_diagonal(orig_sub)

    # Prediction vs each Original
    pred_orig_sub = sim[np.ix_(pred_idx, orig_idx)]  # shape: (num_pred, num_orig)
    pred_vs_orig_values = pred_orig_sub.flatten()

    # Parse charge from filename
    charge_match = re.search(r'charge(\d+)', stem)
    charge = int(charge_match.group(1)) if charge_match else 0

    seq = parse_sequence_from_mgf(mgf_path)

    return {
        "filename": Path(mgf_path).name,
        "sequence": seq,
        "charge": charge,
        "num_original": len(orig_idx),
        "num_prediction": len(pred_idx),
        "total_spectra": n,
        "original_original_mean": orig_orig_mean,
        "prediction_vs_original_mean": float(pred_vs_orig_values.mean()),
        "prediction_vs_original_std": float(pred_vs_orig_values.std()) if len(pred_vs_orig_values) > 1 else float('nan'),
        "prediction_vs_original_max": float(pred_vs_orig_values.max()),
        "prediction_vs_original_min": float(pred_vs_orig_values.min()),
    }


def process_directory(mgf_dir: str, npz_dir: str, output_csv: str,
                      output_xlsx: str = None) -> pd.DataFrame:
    """Process all MGF+NPZ pairs in the given directories."""
    mgf_dir = Path(mgf_dir)
    npz_dir = Path(npz_dir)

    mgf_files = {p.stem: str(p) for p in sorted(mgf_dir.glob("*.mgf"))}
    npz_files = {p.stem: str(p) for p in sorted(npz_dir.glob("*.npz"))}

    # Match by stem name
    common_stems = sorted(set(mgf_files.keys()) & set(npz_files.keys()))

    print(f"MGF files: {len(mgf_files)}")
    print(f"NPZ files: {len(npz_files)}")
    print(f"Matched pairs: {len(common_stems)}")

    if not common_stems:
        print("ERROR: No matching MGF+NPZ pairs found!")
        return pd.DataFrame()

    rows = []
    all_pred_vs_orig = []  # collect all prediction-vs-original scores globally
    skipped = 0
    for stem in common_stems:
        result = process_one_group(mgf_files[stem], npz_files[stem])
        if result is None:
            skipped += 1
            continue
        rows.append(result)
        # collect per-group raw scores for global mean
        if result["prediction_vs_original_mean"] is not None:
            all_pred_vs_orig.append(result["prediction_vs_original_mean"])

    df = pd.DataFrame(rows)

    if df.empty:
        print("No valid results to summarize.")
        return df

    # Sort by prediction_vs_original_mean descending
    df = df.sort_values("prediction_vs_original_mean", ascending=False)

    # Save CSV
    df.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"\nSaved summary CSV: {output_csv}")
    print(f"  groups processed: {len(df)}")
    print(f"  skipped (no match): {skipped}")

    # Compute global mean (weighted by num_original)
    global_pred_vs_orig = np.mean(all_pred_vs_orig) if all_pred_vs_orig else 0.0

    # Print overview
    print(f"\n{'='*80}")
    print("OVERVIEW")
    print(f"{'='*80}")
    print(f"  Groups: {len(df)}")
    if len(df) > 0:
        # Unweighted: each peptide group has equal weight
        unweighted = df['prediction_vs_original_mean'].mean()
        # Weighted (all-pairs): each original spectrum has equal weight
        total_orig = df['num_original'].sum()
        weighted = 0.0
        if total_orig > 0:
            weighted = (df['prediction_vs_original_mean'] * df['num_original']).sum() / total_orig
        print(f"  Prediction vs Original (unweighted):    {unweighted:.4f}  ← 每组等权")
        print(f"  Prediction vs Original (all-pairs):     {weighted:.4f}  ← 每张原始谱等权")
        print(f"  Prediction vs Original (median):         {df['prediction_vs_original_mean'].median():.4f}")
        print(f"  Original-Original (mean):               {df['original_original_mean'].dropna().mean():.4f}")
        print(f"  Avg original spectra per group:         {df['num_original'].mean():.1f}")
    print(f"{'='*80}")

    # Print top 10
    print(f"\nTop 10 groups by prediction-vs-original cosine:")
    print("-" * 70)
    for _, row in df.head(10).iterrows():
        print(f"  {row['filename']:<50s}  "
              f"cos={row['prediction_vs_original_mean']:.4f}  "
              f"n_orig={int(row['num_original'])}")

    # Save XLSX if requested
    if output_xlsx:
        with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="summary", index=False)
        print(f"\nSaved summary XLSX: {output_xlsx}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Batch cosine similarity: prediction-vs-original for all peptide groups."
    )
    parser.add_argument("--mgf-dir", required=True,
                        help="Directory containing per-group MGF files "
                             "(output of collect_original_with_prediction.py)")
    parser.add_argument("--npz-dir", required=True,
                        help="Directory containing per-group NPZ embedding files "
                             "(output of mgf_to_embedding_batch.py)")
    parser.add_argument("--output-csv", required=True,
                        help="Output summary CSV path")
    parser.add_argument("--output-xlsx", default=None,
                        help="Optional output summary XLSX path")
    args = parser.parse_args()

    process_directory(
        mgf_dir=args.mgf_dir,
        npz_dir=args.npz_dir,
        output_csv=args.output_csv,
        output_xlsx=args.output_xlsx,
    )


if __name__ == "__main__":
    main()
