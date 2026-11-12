#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# V20 side: 10k peptides (9 species) -> predicted MGF -> originals collected ->
# PiPrime embeddings -> prediction-vs-original cosine summary (details saved).
# Same step order as transfomer/run_v20_eval.sh, only TEST_NPZ/ROOT_DIR changed.
#
# Usage: bash run_v20_10k.sh <checkpoint> [gpu] [output_name]
set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint> [gpu] [output_name]}"
GPU="${2:-0}"
OUTPUT_NAME="${3:-v20_10k}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # transfomer/
REPO_ROOT="$(cd "$TF_DIR/../.." && pwd)"        # Seq2Mass/

TEST_NPZ="${TEST_NPZ:-$SCRIPT_DIR/data/v20_10k_9species.npz}"
ROOT_DIR="${ROOT_DIR:-E:/biodatasets/denovo/9species/Casanovo_preprocessed_data}"
PIPRIME_MODEL="${PIPRIME_MODEL:-$REPO_ROOT/piprime_embedding/piprime_embedding/model.ckpt}"
PIPELINE_DIR="${PIPELINE_DIR:-$SCRIPT_DIR/output}"
DEVICE="${DEVICE:-cuda:${GPU}}"
MIN_MZ="${MIN_MZ:-100}"
POSTPROC="${POSTPROC:-cluster}"           # cluster | none
BATCH_SIZE="${BATCH_SIZE:-64}"            # 6GB 卡可降到 16/8
RAW_MIN_INTENSITY="${RAW_MIN_INTENSITY:-0.01}"
MERGE_TOL="${MERGE_TOL:-0.05}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$PIPELINE_DIR/${OUTPUT_NAME}_${RUN_STAMP}"
mkdir -p "$OUT_DIR"

echo "[1/4] V20 predict -> $OUT_DIR/predicted.mgf"
_a=(--npz "$TEST_NPZ" --checkpoint "$CHECKPOINT"
    --output_mgf "$OUT_DIR/predicted.mgf" --output_summary_txt "$OUT_DIR/predicted_summary.txt"
    --batch_size "$BATCH_SIZE" --num_workers 4 --min_mz "$MIN_MZ" --merge_tol "$MERGE_TOL"
    --mz_mode max --postprocess_mode "$POSTPROC" --cluster_scene v20_pp4
    --raw_min_intensity "$RAW_MIN_INTENSITY"
    --cluster_diag_dir "$OUT_DIR/cluster_diagnostics"
    --family-aware-postprocess-prior 1 --device "$DEVICE")
[ -n "$MAX_EXAMPLES" ] && _a+=(--max_examples "$MAX_EXAMPLES")
python "$TF_DIR/predict_npz_to_mgf.py" "${_a[@]}"

echo "[2/4] collect originals per peptide"
python "$TF_DIR/collect_original_with_prediction.py" \
  --predicted-mgf "$OUT_DIR/predicted.mgf" --root-dir "$ROOT_DIR" \
  --output-dir "$OUT_DIR/Grouped_MGFs"

echo "[3/4] PiPrime embeddings"
python "$REPO_ROOT/piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py" \
  --mgf_dir "$OUT_DIR/Grouped_MGFs" --output_dir "$OUT_DIR/Embeddings" --model "$PIPRIME_MODEL"

echo "[4/4] cosine summary (细节)"
python "$TF_DIR/batch_cosine_summary.py" \
  --mgf-dir "$OUT_DIR/Grouped_MGFs" --npz-dir "$OUT_DIR/Embeddings" \
  --output-csv "$OUT_DIR/cosine_summary.csv"

echo "[5/5] per-spectrum cosine details (CSV)"
python "$SCRIPT_DIR/cosine_detail_csv.py" \
  --mgf-dir "$OUT_DIR/Grouped_MGFs" --npz-dir "$OUT_DIR/Embeddings" \
  --out-dir "$OUT_DIR"

echo "done -> $OUT_DIR/cosine_summary.csv (+ cosine_detail.csv)"
