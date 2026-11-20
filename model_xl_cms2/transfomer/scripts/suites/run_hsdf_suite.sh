#!/bin/bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# HSDF (+mby) V2 全物种重打分 suite
#   Scoring: d 谱峰余弦 + f 碎片覆盖率 + mby 主b/y精度 (等权 score-avg) + pn 融合
#   默认参数: hard 5ppm + pn 0.3 + rel 0.10 + use_mby + sqrtclip
#   ricebean: hard 20ppm + fix 20 (57Da 错位修复)
#   mmazei:   occurrence_aware 对齐 (scan 号重复)
#
# Usage:
#   cd /home/wangyuhang/seq2mass/v20_0821/transfomer
#   export CKPT=/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt
#   export BASE_V2=/home/wangyuhang/seq2mass/PI_CasanovoV2_candidates/v2
#   export MGF_ROOT=/home/wangyuhang/datasets/90982/90982
#   bash scripts/suites/run_hsdf_suite.sh
#
# Optional: GPU=1 SPECIES="mouse human" FORCE_RERUN=1
# ============================================================================

set -euo pipefail

CKPT="${CKPT:?set CKPT}"
BASE_V2="${BASE_V2:?set BASE_V2}"
MGF_ROOT="${MGF_ROOT:?set MGF_ROOT}"

GPU="${GPU:-0}"
OUTPUT_BASE="${OUTPUT_BASE:-./hsdf_suite_outputs}"
TMP="${TMP:-/tmp/hsdf_mgf}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_SCANS="${MAX_SCANS:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRANSFORMER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$TRANSFORMER_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p "$OUTPUT_BASE" "$TMP"

# 多文件物种合并成单 MGF (按 glob 字母序 cat, 对应 parquet 生成顺序)
merge_mgf() {
  local out="$TMP/$1.mgf"
  shift
  if [ ! -f "$out" ]; then
    cat "$@" > "$out"
    log "merged -> $out"
  fi
}
merge_mgf yeast_merged "$MGF_ROOT"/Saccharomyces-cerevisiae-yeast_{1,2,3,4}/*.mgf
merge_mgf bacillus_1_merged "$MGF_ROOT"/Bacillus_{1,2}/*.mgf
merge_mgf bacillus_2_merged "$MGF_ROOT"/Bacillus_{3,4}/*.mgf

# 通用参数 (除 ricebean 外)
COMMON="--checkpoint $CKPT --skip-results-save"
COMMON="$COMMON --pn-weight 0.3 --rel-spec 0.10 --rel-frag 0.10 --rel-mby 0.10"
COMMON="$COMMON --use-mby --exp-int-transform sqrtclip"

run_one() {
  local name="$1" parquet="$2" mgf="$3"; shift 3
  local out_dir="$OUTPUT_BASE/$name"
  local summary_json="$out_dir/summary.json"
  if [ ! -f "$parquet" ]; then log "[$name] SKIP: missing parquet"; return 0; fi
  if [ ! -f "$mgf" ] && [ ! -d "$mgf" ]; then log "[$name] SKIP: missing mgf"; return 0; fi
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$summary_json" ]; then
    log "[$name] SKIP: existing summary"; return 0; fi
  log "[$name] start"
  local cmd=(python scripts/rescoring/rescore_hsdf.py
    --parquet "$parquet" --mgf "$mgf"
    --output-dir "$out_dir" --gpu "$GPU"
    --precursor-hard-threshold 5 $COMMON "$@")
  if [ -n "$MAX_SCANS" ]; then cmd+=(--max-scans "$MAX_SCANS"); fi
  "${cmd[@]}"
  log "[$name] done"
}

# 普通物种 (row_order)
run_one mouse        "$BASE_V2/v2_mouse.parquet"        "$MGF_ROOT/Mus_mouse/peaks.db.mgf"
run_one human        "$BASE_V2/v2_human.parquet"        "$MGF_ROOT/H.-sapiens/peaks.db.mgf"
run_one clambacteria "$BASE_V2/v2_clambacteria.parquet" "$MGF_ROOT/Candidatus-endoloripes/peaks.db.mgf"
run_one tomato       "$BASE_V2/v2_tomato.parquet"       "$MGF_ROOT/Solanum-lycopersicum/peaks.db.mgf"
run_one honeybee     "$BASE_V2/v2_honeybee.parquet"     "$MGF_ROOT/Apis-mellifera/peaks.db.mgf"
run_one yeast        "$BASE_V2/v2_yeast.parquet"        "$TMP/yeast_merged.mgf"
run_one bacillus_1   "$BASE_V2/v2_bacillus_1.parquet"   "$TMP/bacillus_1_merged.mgf"
run_one bacillus_2   "$BASE_V2/v2_bacillus_2.parquet"   "$TMP/bacillus_2_merged.mgf"

# mmazei: occurrence_aware 对齐 (scan 号重复, 需 --mgf-dir)
run_one mmazei "$BASE_V2/v2_mmazei.parquet" "$MGF_ROOT/Methanosarcina-mazei/peaks.db.mgf" \
  --row-alignment-mode occurrence_aware --mgf-dir "$MGF_ROOT/Methanosarcina-mazei/"

# ricebean: hard 20ppm + fix 20 (57Da 错位修复)
log "[ricebean] start (hard 20 + fix 20)"
python scripts/rescoring/rescore_hsdf.py \
  --parquet "$BASE_V2/v2_ricebean.parquet" \
  --mgf "$MGF_ROOT/Vigna-mungo/peaks.db.mgf" \
  --checkpoint "$CKPT" --skip-results-save \
  --output-dir "$OUTPUT_BASE/ricebean" --gpu "$GPU" \
  --precursor-hard-threshold 20 \
  --pn-weight 0.3 --rel-spec 0.10 --rel-frag 0.10 --rel-mby 0.10 \
  --use-mby --exp-int-transform sqrtclip \
  --precursor-mismatch-mode fix --precursor-mismatch-ppm 20
log "[ricebean] done"

log "===== HSDF suite all done ====="
log "Summaries under: $OUTPUT_BASE/<species>/summary.json"
