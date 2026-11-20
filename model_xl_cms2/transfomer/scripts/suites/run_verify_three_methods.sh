#!/bin/bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# 三种重打分方法 × {mouse, ricebean} 功能验证脚本
#   执行顺序: 旧版 simplified (mouse, ricebean)
#            → HS-DF     (mouse, ricebean)
#            → HS-DF bidir (mouse, ricebean)
#   GPU=1 (可用 GPU 覆盖)
#
# Usage (在 transfomer 目录下或脚本任意位置):
#   bash scripts/suites/run_verify_three_methods.sh           # 全量
#   MAX_SCANS=50 bash scripts/suites/run_verify_three_methods.sh  # 冒烟
#   nohup bash scripts/suites/run_verify_three_methods.sh \
#     > verify_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ============================================================================

set -uo pipefail

# --- 可覆盖环境变量 ---
GPU="${GPU:-1}"
MAX_SCANS="${MAX_SCANS:-}"    # 留空=全量; 设值=冒烟(如 50)

# --- 服务器路径 ---
PARQUET_BASE="${PARQUET_BASE:-/home/wangyuhang/seq2mass/PI_CasanovoV2_candidates/v2}"
CKPT="${CKPT:-/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt}"
PIPRIME="${PIPRIME:-/home/wangyuhang/seq2mass/piprime_embedding/piprime_embedding/model.ckpt}"
MGF_ROOT="${MGF_ROOT:-/home/wangyuhang/datasets/90982/90982}"

# --- 定位 transfomer 目录: 脚本位于 <proj>/transfomer/scripts/suites/run_verify_three_methods.sh ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRANSFORMER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$TRANSFORMER_DIR"

PARQUET_MOUSE="$PARQUET_BASE/v2_mouse.parquet"
PARQUET_RICEBEAN="$PARQUET_BASE/v2_ricebean.parquet"
MGF_MOUSE_DIR="$MGF_ROOT/Mus_mouse/"
MGF_RICEBEAN_DIR="$MGF_ROOT/Vigna-mungo/"
MGF_MOUSE_FILE="$MGF_ROOT/Mus_mouse/peaks.db.mgf"
MGF_RICEBEAN_FILE="$MGF_ROOT/Vigna-mungo/peaks.db.mgf"

OUT_BASE="${OUT_BASE:-./verify_output}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$OUT_BASE"

MAX_ARG=()
[ -n "$MAX_SCANS" ] && MAX_ARG=(--max-scans "$MAX_SCANS")

run_step() {
  local step="$1"; shift
  log "===== START $step ====="
  if ! "$@"; then
    log "!!!!! FAIL $step (exit $?)"
    return 1
  fi
  log "===== DONE  $step ====="
}

# ---- 1. 旧版 simplified (rescore_simplified.py) ----
run_step "simplified/mouse" python scripts/rescoring/rescore_simplified.py \
  --parquet "$PARQUET_MOUSE" --mgf-dir "$MGF_MOUSE_DIR" \
  --checkpoint "$CKPT" --piprime-model "$PIPRIME" \
  --fusion-mode raw --fusion-alpha 0.5 --grid-weight 0.5 \
  --precursor-hard-threshold 5 --row-order --gpu "$GPU" \
  --output-dir "$OUT_BASE/simplified/mouse" --skip-results-save "${MAX_ARG[@]}"

run_step "simplified/ricebean" python scripts/rescoring/rescore_simplified.py \
  --parquet "$PARQUET_RICEBEAN" --mgf-dir "$MGF_RICEBEAN_DIR" \
  --checkpoint "$CKPT" --piprime-model "$PIPRIME" \
  --fusion-mode raw --fusion-alpha 0.5 --grid-weight 0.5 \
  --precursor-hard-threshold 20 \
  --precursor-mismatch-mode fix --precursor-mismatch-ppm 20 \
  --row-order --gpu "$GPU" \
  --output-dir "$OUT_BASE/simplified/ricebean" --skip-results-save "${MAX_ARG[@]}"

# ---- 2. HS-DF (rescore_hsdf.py) ----
run_step "hsdf/mouse" python scripts/rescoring/rescore_hsdf.py \
  --parquet "$PARQUET_MOUSE" --mgf "$MGF_MOUSE_FILE" \
  --checkpoint "$CKPT" \
  --rel-spec 0.20 --rel-frag 0.20 --pn-weight 0.3 \
  --precursor-hard-threshold 5 --gpu "$GPU" \
  --output-dir "$OUT_BASE/hsdf/mouse" --skip-results-save "${MAX_ARG[@]}"

run_step "hsdf/ricebean" python scripts/rescoring/rescore_hsdf.py \
  --parquet "$PARQUET_RICEBEAN" --mgf "$MGF_RICEBEAN_FILE" \
  --checkpoint "$CKPT" \
  --rel-spec 0.20 --rel-frag 0.20 --pn-weight 0.3 \
  --precursor-hard-threshold 20 \
  --precursor-mismatch-mode fix --precursor-mismatch-ppm 20 \
  --gpu "$GPU" --output-dir "$OUT_BASE/hsdf/ricebean" --skip-results-save "${MAX_ARG[@]}"

# ---- 3. HS-DF bidir (rescore_hsdf_bidir.py) ----
run_step "hsdf_bidir/mouse" python scripts/rescoring/rescore_hsdf_bidir.py \
  --parquet "$PARQUET_MOUSE" --mgf "$MGF_MOUSE_FILE" \
  --checkpoint "$CKPT" \
  --rel-spec 0.20 --rel-frag 0.20 --pn-weight 0.3 \
  --precursor-hard-threshold 5 --gpu "$GPU" \
  --output-dir "$OUT_BASE/hsdf_bidir/mouse" --skip-results-save "${MAX_ARG[@]}"

run_step "hsdf_bidir/ricebean" python scripts/rescoring/rescore_hsdf_bidir.py \
  --parquet "$PARQUET_RICEBEAN" --mgf "$MGF_RICEBEAN_FILE" \
  --checkpoint "$CKPT" \
  --rel-spec 0.20 --rel-frag 0.20 --pn-weight 0.3 \
  --precursor-hard-threshold 20 \
  --precursor-mismatch-mode fix --precursor-mismatch-ppm 20 \
  --gpu "$GPU" --output-dir "$OUT_BASE/hsdf_bidir/ricebean" --skip-results-save "${MAX_ARG[@]}"

log "===== ALL STEPS FINISHED ====="
