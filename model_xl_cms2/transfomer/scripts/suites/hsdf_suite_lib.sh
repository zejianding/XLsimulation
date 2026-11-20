#!/bin/bash
# ============================================================================
# HSDF (+mby) V2 全物种 重打分 suite 公共逻辑 (被 4 个入口脚本 source)
#   由入口脚本 export: FUSION (minmax/raw), USE_MBY (0/1)
#   当前最优参数: hard 5ppm + pn 0.3 + rel 0.10 + sqrtclip
#   ricebean: hard 20ppm; mmazei: occurrence_aware
# Usage (由入口脚本 source 后调用 run_all):
#   export CKPT=... BASE_V2=... MGF_ROOT=... FUSION=minmax USE_MBY=0
#   source hsdf_suite_lib.sh; run_all
# ============================================================================

set -euo pipefail

: "${CKPT:?set CKPT}"
: "${BASE_V2:?set BASE_V2}"
: "${MGF_ROOT:?set MGF_ROOT}"
FUSION="${FUSION:?set FUSION (minmax/raw)}"
USE_MBY="${USE_MBY:?set USE_MBY (0/1)}"

GPU="${GPU:-0}"
OUTPUT_BASE="${OUTPUT_BASE:-./hsdf_out_${FUSION}_$( [ "$USE_MBY" = "1" ] && echo mby || echo nomby )}"
TMP="${TMP:-/tmp/hsdf_mgf}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_SCANS="${MAX_SCANS:-}"
# RESCORER: 换用别的重打分脚本（如 rescore_hsdf_bidir.py）。注意 bidir 版没有 --use-mby。
RESCORER="${RESCORER:-scripts/rescoring/rescore_hsdf.py}"
# SAVE_FULL=1: 每物种额外写 full_results.csv (全量候选数据, 供 save_predictions_cases.py 案例分析)。
#   脚本 COMMON 默认带 --skip-results-save, 会挡掉 --save-full-results, 故 SAVE_FULL=1 时替换之。
SAVE_FULL="${SAVE_FULL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$TRANSFORMER_DIR"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p "$OUTPUT_BASE" "$TMP"

# 多文件物种合并成单 MGF (glob 字母序, 对应 parquet 生成顺序)
merge_mgf() {
  local out="$TMP/$1.mgf"; shift
  if [ ! -f "$out" ]; then
    cat "$@" > "$out"; log "merged -> $out"
  fi
}
merge_mgf yeast_merged "$MGF_ROOT"/Saccharomyces-cerevisiae-yeast_{1,2,3,4}/*.mgf
merge_mgf bacillus_1_merged "$MGF_ROOT"/Bacillus_{1,2}/*.mgf
merge_mgf bacillus_2_merged "$MGF_ROOT"/Bacillus_{3,4}/*.mgf

COMMON="--checkpoint $CKPT --skip-results-save --pn-weight 0.3"
COMMON="$COMMON --rel-spec 0.10 --rel-frag 0.10 --rel-mby 0.10"
COMMON="$COMMON --exp-int-transform sqrtclip"
if [ "${SAVE_FULL}" = "1" ]; then
  COMMON="${COMMON//--skip-results-save/--save-full-results}"
  log "SAVE_FULL=1: --skip-results-save -> --save-full-results"
fi

# 单物种 (row_order), 用全局 FUSION/USE_MBY
run_one() {
  local name="$1" parquet="$2" mgf="$3"; shift 3
  local extra=("$@")
  local out_dir="$OUTPUT_BASE/$name"
  if [ ! -f "$parquet" ]; then log "[$name] SKIP: no parquet"; return 0; fi
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$out_dir/summary.json" ]; then
    log "[$name] SKIP: existing summary"; return 0; fi
  mkdir -p "$out_dir"
  log "[$name] start (fusion=$FUSION, mby=$USE_MBY)"
  local cmd=(python "$RESCORER"
    --parquet "$parquet" --mgf "$mgf"
    --output-dir "$out_dir" --gpu "$GPU"
    --precursor-hard-threshold 5 $COMMON --fusion-norm "$FUSION")
  [ "$USE_MBY" = "1" ] && [ "$RESCORER" = "scripts/rescoring/rescore_hsdf.py" ] && cmd+=(--use-mby)
  cmd+=("${extra[@]}")
  if [ -n "$MAX_SCANS" ]; then cmd+=(--max-scans "$MAX_SCANS"); fi
  "${cmd[@]}"
  log "[$name] done"
}

# ricebean (hard 20, 57Da 错位多)
run_ricebean() {
  local out_dir="$OUTPUT_BASE/ricebean"
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$out_dir/summary.json" ]; then
    log "[ricebean] SKIP: existing summary"; return 0; fi
  mkdir -p "$out_dir"
  log "[ricebean] start (hard 20)"
  local cmd=(python "$RESCORER"
    --parquet "$BASE_V2/v2_ricebean.parquet" --mgf "$MGF_ROOT/Vigna-mungo/peaks.db.mgf"
    --checkpoint "$CKPT" --skip-results-save
    --output-dir "$out_dir" --gpu "$GPU"
    --precursor-hard-threshold 20 $COMMON --fusion-norm "$FUSION")
  [ "$USE_MBY" = "1" ] && [ "$RESCORER" = "scripts/rescoring/rescore_hsdf.py" ] && cmd+=(--use-mby)
  if [ -n "$MAX_SCANS" ]; then cmd+=(--max-scans "$MAX_SCANS"); fi
  "${cmd[@]}"
  log "[ricebean] done"
}

# 全部物种 (入口脚本调用)
run_all() {
  run_one mouse        "$BASE_V2/v2_mouse.parquet"        "$MGF_ROOT/Mus_mouse/peaks.db.mgf"
  run_one human        "$BASE_V2/v2_human.parquet"        "$MGF_ROOT/H.-sapiens/peaks.db.mgf"
  run_one clambacteria "$BASE_V2/v2_clambacteria.parquet" "$MGF_ROOT/Candidatus-endoloripes/peaks.db.mgf"
  run_one tomato       "$BASE_V2/v2_tomato.parquet"       "$MGF_ROOT/Solanum-lycopersicum/peaks.db.mgf"
  run_one honeybee     "$BASE_V2/v2_honeybee.parquet"     "$MGF_ROOT/Apis-mellifera/peaks.db.mgf"
  run_one yeast        "$BASE_V2/v2_yeast.parquet"        "$TMP/yeast_merged.mgf"
  run_one bacillus_1   "$BASE_V2/v2_bacillus_1.parquet"   "$TMP/bacillus_1_merged.mgf"
  run_one bacillus_2   "$BASE_V2/v2_bacillus_2.parquet"   "$TMP/bacillus_2_merged.mgf"
  # mmazei: occurrence_aware 对齐
  run_one mmazei "$BASE_V2/v2_mmazei.parquet" "$MGF_ROOT/Methanosarcina-mazei/peaks.db.mgf" \
    --row-alignment-mode occurrence_aware --mgf-dir "$MGF_ROOT/Methanosarcina-mazei/"
  run_ricebean
  log "===== all done (fusion=$FUSION, mby=$USE_MBY) ====="
  log "Outputs: $OUTPUT_BASE/<species>/summary.json"
}
