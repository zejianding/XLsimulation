#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# Prosit side: same 10k peptides -> Prosit predicted MGF -> originals collected ->
# PiPrime embeddings -> prediction-vs-original cosine summary (details saved).
# Needs koinapy + outbound internet (Koina).
#
# Usage: bash run_prosit_10k.sh [output_name]
set -euo pipefail

OUTPUT_NAME="${1:-prosit_10k}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TF_DIR/../.." && pwd)"

TEST_NPZ="${TEST_NPZ:-$SCRIPT_DIR/data/v20_10k_9species.npz}"
ROOT_DIR="${ROOT_DIR:-E:/biodatasets/denovo/9species/Casanovo_preprocessed_data}"
PIPRIME_MODEL="${PIPRIME_MODEL:-$REPO_ROOT/piprime_embedding/piprime_embedding/model.ckpt}"
PIPELINE_DIR="${PIPELINE_DIR:-$SCRIPT_DIR/output}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$PIPELINE_DIR/${OUTPUT_NAME}_${RUN_STAMP}"
mkdir -p "$OUT_DIR"

echo "[1/4] Prosit predict (Koina) -> $OUT_DIR/predicted.mgf"
python "$SCRIPT_DIR/prosit_predict_mgf.py" --npz "$TEST_NPZ" \
  --out "$OUT_DIR/predicted.mgf" ${P4_BATCH:+--batch-size "$P4_BATCH"} ${P4_CACHE:+--cache "$P4_CACHE"} \
  $( [ "${P4_RAW:-0}" = "1" ] && echo --raw-api ) \
  $( [ "${P4_BYONLY:-0}" = "1" ] && echo --by-only )

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
