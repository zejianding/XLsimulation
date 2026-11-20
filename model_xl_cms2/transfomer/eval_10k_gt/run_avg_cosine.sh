#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# B2 驱动：Grouped_MGFs → 每肽平均谱 → PiPrime 嵌入 → 预测谱 vs 平均谱 cosine。
# 产物：<run>/{Average_MGFs/, Embeddings_Avg/, cosine_avg_summary.csv, cosine_avg_detail.csv}
#
# env:
#   RUN_DIR    已完成运行目录（必填，需含 Grouped_MGFs 与 Embeddings）
#   AVG_TOL    平均谱 m/z 合并容差（默认 0.1，对齐 v20_0821 训练侧 TOL）
#   AVG_SRC    平均谱的落盘目录（默认 = RUN_DIR）。同一物种跨 predictor 时指向
#              已经算好的那个 run，配合 REUSE_AVG=1 即可避免重复平均与嵌入。
#   REUSE_AVG  1=跳过平均谱与其嵌入，复用 AVG_SRC 下的已有产物
#   OUT_DIR    cosine_avg_*.csv 的落盘目录（默认 = RUN_DIR）。RUN_DIR 与输出分离时用，
#              例如把结果集中到 /data/.../v20 而中间产物留在别处
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$DIR/.." && pwd)"
REPO_ROOT="$(cd "$TF_DIR/../.." && pwd)"
PIPRIME_MODEL="${PIPRIME_MODEL:-$REPO_ROOT/piprime_embedding/piprime_embedding/model.ckpt}"
RUN_DIR="${RUN_DIR:?need RUN_DIR=<finished run dir>}"
AVG_TOL="${AVG_TOL:-0.1}"
REUSE_AVG="${REUSE_AVG:-0}"
AVG_SRC="${AVG_SRC:-$RUN_DIR}"
OUT_DIR="${OUT_DIR:-$RUN_DIR}"
[ -d "$RUN_DIR/Grouped_MGFs" ] || { echo "no Grouped_MGFs under $RUN_DIR"; exit 2; }

AVG_DIR="$AVG_SRC/Average_MGFs"
EMB_AVG="$AVG_SRC/Embeddings_Avg"

if [ "$REUSE_AVG" = "1" ] && [ -d "$EMB_AVG" ]; then
  echo "[1-2/3] reuse existing $AVG_DIR + $EMB_AVG"
else
  echo "[1/3] average spectra -> $AVG_DIR"
  python "$DIR/grouped_to_average_mgf.py" --mgf-dir "$RUN_DIR/Grouped_MGFs" \
    --out-dir "$AVG_DIR" --tol "$AVG_TOL"
  echo "[2/3] PiPrime embeddings of the average spectra"
  python "$REPO_ROOT/piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py" \
    --mgf_dir "$AVG_DIR" --output_dir "$EMB_AVG" --model "$PIPRIME_MODEL"
fi

echo "[3/3] prediction vs average cosine"
mkdir -p "$OUT_DIR"
python "$DIR/cosine_avg_vs_pred.py" --mgf-dir "$RUN_DIR/Grouped_MGFs" \
  --npz-dir "$RUN_DIR/Embeddings" --avg-npz-dir "$EMB_AVG" --out-dir "$OUT_DIR"

echo "done -> $OUT_DIR/cosine_avg_summary.csv (+ cosine_avg_detail.csv)"
