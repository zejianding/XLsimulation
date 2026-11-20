#!/bin/bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# HSDF (+mby) V2 全物种 重打分 suite: 每物种跑 4 种组合
#   组合 (按顺序): nofix_minmax, nofix_raw, fix_minmax, fix_raw
#   当前最优参数: hard 5ppm + pn 0.3 + rel 0.10 + use_mby + sqrtclip
#   fix: + --precursor-mismatch-mode fix --precursor-mismatch-ppm 20
#   minmax/raw: --fusion-norm
#   ricebean: hard 20ppm (57Da 错位多)
#   mmazei:   occurrence_aware 对齐 (scan 号重复)
#
# Usage:
#   cd /home/wangyuhang/seq2mass/v20_0821/transfomer
#   export CKPT=/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt
#   export BASE_V2=/home/wangyuhang/seq2mass/PI_CasanovoV2_candidates/v2
#   export MGF_ROOT=/home/wangyuhang/datasets/90982/90982
#   bash scripts/suites/run_hsdf_suite_fix_cmp.sh
#
# Optional: GPU=1 SPECIES="mouse ricebean" FORCE_RERUN=1
# Outputs:  $OUTPUT_BASE/{nofix_minmax,nofix_raw,fix_minmax,fix_raw}/<species>/summary.json
# ============================================================================

set -euo pipefail

CKPT="${CKPT:?set CKPT}"
BASE_V2="${BASE_V2:?set BASE_V2}"
MGF_ROOT="${MGF_ROOT:?set MGF_ROOT}"

GPU="${GPU:-0}"
OUTPUT_BASE="${OUTPUT_BASE:-./hsdf_suite_fix_cmp}"
TMP="${TMP:-/tmp/hsdf_mgf}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_SCANS="${MAX_SCANS:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRANSFORMER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$TRANSFORMER_DIR"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p "$OUTPUT_BASE" "$TMP"

# 多文件物种合并成单 MGF
merge_mgf() {
  local out="$TMP/$1.mgf"; shift
  if [ ! -f "$out" ]; then
    cat "$@" > "$out"; log "merged -> $out"
  fi
}
merge_mgf yeast_merged "$MGF_ROOT"/Saccharomyces-cerevisiae-yeast_{1,2,3,4}/*.mgf
merge_mgf bacillus_1_merged "$MGF_ROOT"/Bacillus_{1,2}/*.mgf
merge_mgf bacillus_2_merged "$MGF_ROOT"/Bacillus_{3,4}/*.mgf

# 通用参数 (除 ricebean 外)
COMMON="--checkpoint $CKPT --skip-results-save --pn-weight 0.3"
COMMON="$COMMON --rel-spec 0.10 --rel-frag 0.10 --rel-mby 0.10"
COMMON="$COMMON --exp-int-transform sqrtclip"

# 4 组合 (先 norm 后 raw; 先无 mby 后有 mby):
#   nofix_minmax_nomby, nofix_minmax_mby, nofix_raw_nomby, nofix_raw_mby
COMBOS=("nofix_minmax_nomby" "nofix_minmax_mby" "nofix_raw_nomby" "nofix_raw_mby")

run_quad() {
  local name="$1" parquet="$2" mgf="$3"; shift 3
  local extra=("$@")
  for combo in "${COMBOS[@]}"; do
    local out_dir="$OUTPUT_BASE/$combo/$name"
    local summary_json="$out_dir/summary.json"
    if [ ! -f "$parquet" ]; then log "[$combo/$name] SKIP: no parquet"; continue; fi
    if [ "$FORCE_RERUN" != "1" ] && [ -f "$summary_json" ]; then
      log "[$combo/$name] SKIP: existing summary"; continue; fi
    mkdir -p "$out_dir"
    log "[$combo/$name] start"
    local cmd=(python scripts/rescoring/rescore_hsdf.py
      --parquet "$parquet" --mgf "$mgf"
      --output-dir "$out_dir" --gpu "$GPU"
      --precursor-hard-threshold 5 $COMMON)
    case "$combo" in
      *minmax*) cmd+=(--fusion-norm minmax);;
      *raw*)    cmd+=(--fusion-norm raw);;
    esac
    if [[ "$combo" == *_mby ]]; then
      cmd+=(--use-mby)
    fi
    cmd+=("${extra[@]}")
    if [ -n "$MAX_SCANS" ]; then cmd+=(--max-scans "$MAX_SCANS"); fi
    "${cmd[@]}"
    log "[$combo/$name] done"
  done
}

# 普通物种 (row_order)
run_quad mouse        "$BASE_V2/v2_mouse.parquet"        "$MGF_ROOT/Mus_mouse/peaks.db.mgf"
run_quad human        "$BASE_V2/v2_human.parquet"        "$MGF_ROOT/H.-sapiens/peaks.db.mgf"
run_quad clambacteria "$BASE_V2/v2_clambacteria.parquet" "$MGF_ROOT/Candidatus-endoloripes/peaks.db.mgf"
run_quad tomato       "$BASE_V2/v2_tomato.parquet"       "$MGF_ROOT/Solanum-lycopersicum/peaks.db.mgf"
run_quad honeybee     "$BASE_V2/v2_honeybee.parquet"     "$MGF_ROOT/Apis-mellifera/peaks.db.mgf"
run_quad yeast        "$BASE_V2/v2_yeast.parquet"        "$TMP/yeast_merged.mgf"
run_quad bacillus_1   "$BASE_V2/v2_bacillus_1.parquet"   "$TMP/bacillus_1_merged.mgf"
run_quad bacillus_2   "$BASE_V2/v2_bacillus_2.parquet"   "$TMP/bacillus_2_merged.mgf"

# mmazei: occurrence_aware 对齐
run_quad mmazei "$BASE_V2/v2_mmazei.parquet" "$MGF_ROOT/Methanosarcina-mazei/peaks.db.mgf" \
  --row-alignment-mode occurrence_aware --mgf-dir "$MGF_ROOT/Methanosarcina-mazei/"

# ricebean: hard 20 (57Da 错位多), 4 种组合
for combo in "${COMBOS[@]}"; do
  out_dir="$OUTPUT_BASE/$combo/ricebean"
  summary_json="$out_dir/summary.json"
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$summary_json" ]; then
    log "[$combo/ricebean] SKIP: existing summary"; continue; fi
  mkdir -p "$out_dir"
  log "[$combo/ricebean] start (hard 20)"
  cmd=(python scripts/rescoring/rescore_hsdf.py
    --parquet "$BASE_V2/v2_ricebean.parquet" --mgf "$MGF_ROOT/Vigna-mungo/peaks.db.mgf"
    --checkpoint "$CKPT" --skip-results-save
    --output-dir "$out_dir" --gpu "$GPU"
    --precursor-hard-threshold 20 $COMMON)
  case "$combo" in
    *minmax*) cmd+=(--fusion-norm minmax);;
    *raw*)    cmd+=(--fusion-norm raw);;
  esac
  if [[ "$combo" == *_mby ]]; then
    cmd+=(--use-mby)
  fi
  if [ -n "$MAX_SCANS" ]; then cmd+=(--max-scans "$MAX_SCANS"); fi
  "${cmd[@]}"
  log "[$combo/ricebean] done"
done

log "===== HSDF suite (nofix x minmax/raw x mby/nomby) all done ====="
log "Outputs: $OUTPUT_BASE/{nofix_minmax_nomby,nofix_minmax_mby,nofix_raw_nomby,nofix_raw_mby}/<species>/summary.json"
