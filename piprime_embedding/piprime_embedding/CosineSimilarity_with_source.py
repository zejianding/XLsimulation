import argparse
import json
import os

import numpy as np
import pandas as pd


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = embeddings / norms
    return normalized @ normalized.T


def mean_excluding_diagonal(submatrix: np.ndarray) -> float:
    n = submatrix.shape[0]
    if n <= 1:
        return 0.0
    return float((np.sum(submatrix) - np.trace(submatrix)) / (n * (n - 1)))


def _as_text_array(values, length, default):
    if values is None:
        return np.array([default] * length, dtype=np.str_)
    return np.array([str(v) for v in values], dtype=np.str_)


def process_one_npz(npz_path: str, output_excel: str):
    data = np.load(npz_path, allow_pickle=True)

    embeddings = data["embeddings"]
    n = embeddings.shape[0]
    if n == 0:
        print(f"{npz_path} has no embeddings, skipped.")
        return None

    titles = _as_text_array(data["titles"] if "titles" in data else None, n, "UNKNOWN")
    precursor_mz = (
        data["precursor_mz"].astype(np.float32)
        if "precursor_mz" in data
        else np.zeros(n, dtype=np.float32)
    )
    charges = (
        data["charges"].astype(np.int32)
        if "charges" in data
        else np.zeros(n, dtype=np.int32)
    )
    sources = _as_text_array(data["sources"] if "sources" in data else None, n, "UNKNOWN")

    sim = cosine_similarity_matrix(embeddings)

    labels = [f"{i}_{sources[i]}" for i in range(n)]
    sim_df = pd.DataFrame(sim, index=labels, columns=labels)
    meta_df = pd.DataFrame(
        {
            "index": list(range(n)),
            "label": labels,
            "title": titles,
            "source": sources,
            "precursor_mz": precursor_mz,
            "charge": charges,
        }
    )

    source_upper = [str(s).upper() for s in sources]
    orig_idx = [i for i, s in enumerate(source_upper) if s == "ORIGINAL"]
    filt_idx = [i for i, s in enumerate(source_upper) if s == "FILTERED"]
    pred_idx = [i for i, s in enumerate(source_upper) if s == "MODEL_PREDICTION"]

    summary_rows = []
    original_original_mean = None
    filtered_filtered_mean = None
    filtered_original_mean = None
    filtered_original_pair = None
    prediction_vs_original_mean = None
    prediction_vs_filtered_mean = None

    if orig_idx:
        orig_orig = sim[np.ix_(orig_idx, orig_idx)]
        original_original_mean = mean_excluding_diagonal(orig_orig)
        summary_rows.append(
            {"metric": "original_original_mean", "value": original_original_mean}
        )

    if filt_idx:
        filt_filt = sim[np.ix_(filt_idx, filt_idx)]
        filtered_filtered_mean = mean_excluding_diagonal(filt_filt)
        summary_rows.append(
            {"metric": "filtered_filtered_mean", "value": filtered_filtered_mean}
        )

    if orig_idx and filt_idx:
        cross = sim[np.ix_(filt_idx, orig_idx)]
        filtered_original_mean = float(np.mean(cross))
        summary_rows.append(
            {"metric": "filtered_original_mean", "value": filtered_original_mean}
        )

        pair_count = min(len(orig_idx), len(filt_idx))
        if pair_count > 0:
            pair_sims = [sim[filt_idx[i], orig_idx[i]] for i in range(pair_count)]
            filtered_original_pair = float(np.mean(pair_sims))
            summary_rows.append(
                {"metric": "filtered_original_pair", "value": filtered_original_pair}
            )

    if pred_idx:
        if orig_idx:
            po = sim[np.ix_(pred_idx, orig_idx)]
            prediction_vs_original_mean = float(np.mean(po))
            summary_rows.append(
                {"metric": "prediction_vs_original_mean", "value": prediction_vs_original_mean}
            )
        if filt_idx:
            pf = sim[np.ix_(pred_idx, filt_idx)]
            prediction_vs_filtered_mean = float(np.mean(pf))
            summary_rows.append(
                {"metric": "prediction_vs_filtered_mean", "value": prediction_vs_filtered_mean}
            )

    summary_df = pd.DataFrame(summary_rows)

    out_dir = os.path.dirname(os.path.abspath(output_excel))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        sim_df.to_excel(writer, sheet_name="similarity_matrix")
        meta_df.to_excel(writer, sheet_name="metadata", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"Saved: {output_excel}")

    peptide_name = os.path.splitext(os.path.basename(npz_path))[0]
    return {
        "peptide_name_and_count": peptide_name,
        "num_embeddings": int(n),
        "original_original_mean": original_original_mean,
        "filtered_filtered_mean": filtered_filtered_mean,
        "filtered_original_mean": filtered_original_mean,
        "filtered_original_pair": filtered_original_pair,
        "prediction_vs_original_mean": prediction_vs_original_mean,
        "prediction_vs_filtered_mean": prediction_vs_filtered_mean,
    }


def process_npz_dir(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {input_dir}")

    all_summary_rows = []
    results = {}
    for file_name in sorted(npz_files):
        npz_path = os.path.join(input_dir, file_name)
        out_excel = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.xlsx")
        try:
            result = process_one_npz(npz_path, out_excel)
            if result is not None:
                all_summary_rows.append(result)
                results[file_name] = result
        except Exception as exc:
            print(f"[ERROR] Failed to process {file_name}: {exc}")

    summary_excel = os.path.join(output_dir, "ALL_PEPTIDES_summary.xlsx")
    final_summary_df = pd.DataFrame(all_summary_rows)
    with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
        final_summary_df.to_excel(writer, sheet_name="summary", index=False)
    print(f"Saved summary: {summary_excel}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pi-Prime embedding cosine similarity matrices."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--npz", help="Single embedding NPZ file.")
    mode.add_argument("--input_dir", help="Directory containing embedding NPZ files.")
    parser.add_argument("--output", help="Output xlsx path for --npz mode.")
    parser.add_argument("--output_dir", help="Output directory for --input_dir mode.")
    parser.add_argument("--summary-json", help="Optional JSON summary output.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.npz:
        output_excel = args.output
        if output_excel is None:
            stem = os.path.splitext(os.path.basename(args.npz))[0]
            output_excel = f"{stem}_cosine.xlsx"
        result = process_one_npz(args.npz, output_excel)
        results = {os.path.basename(args.npz): result} if result is not None else {}
    else:
        if args.output_dir is None:
            raise ValueError("--input_dir mode requires --output_dir")
        results = process_npz_dir(args.input_dir, args.output_dir)

    if args.summary_json:
        out_dir = os.path.dirname(os.path.abspath(args.summary_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON summary: {args.summary_json}")


if __name__ == "__main__":
    main()
