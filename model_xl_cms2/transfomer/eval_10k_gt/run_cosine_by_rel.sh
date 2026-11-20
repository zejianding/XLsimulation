#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# 在一次已完成运行（含 Grouped_MGFs）上按“口径变体”重算余弦（不重跑预测）：
#   - 预测侧 rel 过滤（0 = 不过滤）
#   - 实验谱强度变换（exp-transform，镜像 V20 打分域 sqrt_clip）
#   - 预测谱强度变换（pred-transform，如给 Prosit 预测做 sqrt 对齐）
# 每个变体目录内直接产出 Embeddings/ 与 cosine_summary.csv / cosine_detail.csv。
#
# env:
#   RUN_DIR        已完成运行目录（默认取 output/ 下最新的 v20_* 或 prosit_*）
#   RELS           逗号分隔（默认 0,0.05,0.10,0.20）
#   EXP_TRANSFORM  none|sqrt_clip|sqrt   （默认 none）
#   PRED_TRANSFORM none|sqrt|sqrt_clip   （默认 none；Prosit 对齐常用 sqrt）
#   TOP_K          预测谱峰数上限（0=不限）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$DIR/.." && pwd)"
REPO_ROOT="$(cd "$TF_DIR/../.." && pwd)"

RELS="${RELS:-0,0.05,0.10,0.20}"
EXP_TRANSFORM="${EXP_TRANSFORM:-none}"
PRED_TRANSFORM="${PRED_TRANSFORM:-none}"
TOP_K="${TOP_K:-0}"
PIPRIME_MODEL="${PIPRIME_MODEL:-$REPO_ROOT/piprime_embedding/piprime_embedding/model.ckpt}"
RUN_DIR="${RUN_DIR:-$(ls -dt "$DIR"/output/{v20*,prosit*} 2>/dev/null | head -1)}"
[ -n "$RUN_DIR" ] || { echo "need RUN_DIR=<finished run dir>"; exit 2; }
[ -d "$RUN_DIR/Grouped_MGFs" ] || { echo "no Grouped_MGFs under $RUN_DIR"; exit 2; }

VAR="$RUN_DIR/variants"
echo "=== filter: rels=$RELS exp=$EXP_TRANSFORM pred=$PRED_TRANSFORM top_k=$TOP_K"
python "$DIR/filter_group_predictions.py" --mgf-dir "$RUN_DIR/Grouped_MGFs" \
  --out-root "$VAR" --rels "$RELS" --top-k "$TOP_K" \
  --exp-transform "$EXP_TRANSFORM" --pred-transform "$PRED_TRANSFORM"

for d in $(find "$VAR" -maxdepth 2 -type d -name 'Grouped_MGFs_rel*' | sort); do
  echo "===== $d"
  python "$REPO_ROOT/piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py" \
    --mgf_dir "$d" --output_dir "$d/Embeddings" --model "$PIPRIME_MODEL"
  python "$TF_DIR/batch_cosine_summary.py" --mgf-dir "$d" --npz-dir "$d/Embeddings" \
    --output-csv "$d/cosine_summary.csv"
  python "$DIR/cosine_detail_csv.py" --mgf-dir "$d" --npz-dir "$d/Embeddings" --out-dir "$d"
done
echo "done -> $VAR/<tag>/Grouped_MGFs_rel<lvl>/{cosine_summary.csv, cosine_detail.csv}"
