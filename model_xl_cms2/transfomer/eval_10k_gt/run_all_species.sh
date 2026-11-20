#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# 全量重跑：九物种 × (V20pp / V20raw / Prosit-reanchor) 三个主 run + 每个 run 的矩阵。
# 产物：output/{v20pp|v20raw|prosit}_<sp>_<stamp>/{cosine_*, Grouped_MGFs/, Embeddings/, predicted.mgf,
#        variants/<raw|grid>_rel<lvl>_exp<exp>_pred<pred>/Grouped_MGFs_rel<lvl>/{cosine_summary,cosine_detail}.csv}
#
# env:
#   SRC         原谱根（9species 目录 或 v1-search）
#   CKPT        V20 权重
#   GPU         cuda 序号（默认 0）
#   SPECIES     逗号分隔（默认九物种）
#   RELS/PROJ/EXP  传给 run_matrix.sh（默认 0,0.05,0.10,0.20 / raw,grid / sqrt_clip）
#   WITH_PROSIT_REF 1=额外跑 prosit by-only 与 raw-api（默认 0）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:?need SRC=<corpus root>}"
CKPT="${CKPT:?need CKPT=<best_model.pt>}"
GPU="${GPU:-0}"
SPECIES="${SPECIES:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"
RELS="${RELS:-0,0.05,0.10,0.20}"
PROJ="${PROJ:-raw,grid}"
EXP="${EXP:-sqrt_clip}"
WITH_PROSIT_REF="${WITH_PROSIT_REF:-0}"   # 1 = 额外跑 prosit by-only / raw-api（默认关）
SKIP_PROSIT="${SKIP_PROSIT:-0}"           # 1 = 跳过 Prosit 主 run（复用已有结果）
mkdir -p "$DIR/logs"

matrix() {  # $1 = run dir
  RUN_DIR="$1" PROJ="$PROJ" RELS="$RELS" EXP="$EXP" \
    PRED="$2" bash "$DIR/run_matrix.sh" > "$DIR/logs/matrix_$(basename "$1").log" 2>&1
}

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  echo "########## $sp  ($(date '+%F %T'))"
  # --- 主 run -------------------------------------------------------------
  echo "-- [$sp] V20pp main"; sleep 1
  SPECIES=$sp SRC="$SRC" GPU="$GPU" \
    bash "$DIR/run_10k_gt_per_species.sh" > "$DIR/logs/main_v20pp_$sp.log" 2>&1
  echo "-- [$sp] V20raw main"
  POSTPROC=none MERGE_TOL=0.05 RAW_MIN_INTENSITY=0.01 SPECIES=$sp SRC="$SRC" GPU="$GPU" \
    bash "$DIR/run_10k_gt_per_species.sh" > "$DIR/logs/main_v20raw_$sp.log" 2>&1
  echo "-- [$sp] Prosit main"
  PREDICTOR=prosit SPECIES=$sp SRC="$SRC" \
    bash "$DIR/run_10k_gt_per_species.sh" > "$DIR/logs/main_prosit_$sp.log" 2>&1
  if [ "$WITH_PROSIT_REF" = "1" ]; then
    PREDICTOR=prosit P4_BYONLY=1 SPECIES=$sp SRC="$SRC" \
      bash "$DIR/run_10k_gt_per_species.sh" > "$DIR/logs/main_prosit_byonly_$sp.log" 2>&1
    PREDICTOR=prosit P4_RAW=1 SPECIES=$sp SRC="$SRC" \
      bash "$DIR/run_10k_gt_per_species.sh" > "$DIR/logs/main_prosit_rawapi_$sp.log" 2>&1
  fi
  # --- 矩阵：仅 V20（Prosit 单口径，不做投影/变体） -----------------------
  echo "-- [$sp] matrix V20pp"; echo "-- [$sp] matrix V20raw"
  matrix "$(ls -dt "$DIR"/output/v20pp_${sp}_*  | head -1)"  none \
    || echo "!!! WARN: matrix v20pp failed for $sp (see logs/)"
  matrix "$(ls -dt "$DIR"/output/v20raw_${sp}_* | head -1)"  none \
    || echo "!!! WARN: matrix v20raw failed for $sp (see logs/)"
  echo "########## $sp done ($(date '+%F %T'))"
done

echo "===== aggregating ====="
python "$DIR/collect_results.py" --root "$DIR/output" --out "$DIR/汇总矩阵.csv"
echo "===== ALL DONE: $DIR/汇总矩阵.csv (+ _pivot.csv) ====="
