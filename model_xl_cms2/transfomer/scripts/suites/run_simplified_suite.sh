#!/bin/bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# V20_0701 simplified full-species rescoring suite (BEST MODE)
#   Scoring: 0.50·postproc_piprime + 0.50·grid_cos (2-term)
#   Filter:  hard 5ppm precursor mass cutoff
#   Fusion:  pn_fusion (alpha=0.5)
#
# Usage:
#   cd /home/wangyuhang/seq2mass/v20_0701/transfomer
#
#   # minimal:
#   export CKPT=... PIPRIME=... BASE=... MGF_ROOT=...
#   bash scripts/suites/run_simplified_suite.sh
#
#   # nohup:
#   nohup env CKPT=... PIPRIME=... BASE=... MGF_ROOT=... \
#     GPU=1 ALPHA=0.5 HARD_THRESHOLD=5 \
#     OUTPUT_BASE=./simplified_hard5_suite_outputs \
#     bash scripts/suites/run_simplified_suite.sh \
#     > nohup_simplified_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Optional env: GPU ALPHA HARD_THRESHOLD INCLUDE_MMAZEI FORCE_RERUN SAVE_DETAILED SAVE_FULL
#               SKIP_SUMMARY SPECIES MAX_SCANS OUTPUT_BASE SUMMARY_PREFIX
#   SAVE_FULL=1     每候选一行 full_results.csv（流式，内存安全；列名对齐 HSDF 的 full_results.csv）
#   SAVE_DETAILED=1 每候选一行 rescoring_diagnostics.csv（20 列，可与 SAVE_FULL 同开）
#   SKIP_SUMMARY=1  不写汇总（分批补跑时用：汇总脚本是覆盖写，会把多物种表缩成子集）
# ============================================================================

set -euo pipefail

CKPT="${CKPT:?set CKPT}"
PIPRIME="${PIPRIME:?set PIPRIME}"
BASE="${BASE:?set BASE}"
MGF_ROOT="${MGF_ROOT:?set MGF_ROOT}"

GPU="${GPU:-0}"
ALPHA="${ALPHA:-0.5}"
HARD_THRESHOLD="${HARD_THRESHOLD:-5}"
MAX_SCANS="${MAX_SCANS:-}"
OUTPUT_BASE="${OUTPUT_BASE:-./simplified_hard5_suite_outputs}"
TMP_MGF_BASE="${TMP_MGF_BASE:-/tmp/simplified_suite_mgf}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-./simplified_hard5_suite_summary}"
INCLUDE_MMAZEI="${INCLUDE_MMAZEI:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SAVE_DETAILED="${SAVE_DETAILED:-0}"
SAVE_FULL="${SAVE_FULL:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
MMAZEI_DROP_SEQ_WITH_O="${MMAZEI_DROP_SEQ_WITH_O:-1}"
RICEBEAN_PRECURSOR_MODE="${RICEBEAN_PRECURSOR_MODE:-fix}"
RICEBEAN_PRECURSOR_PPM="${RICEBEAN_PRECURSOR_PPM:-20}"
RICEBEAN_HARD_THRESHOLD="${RICEBEAN_HARD_THRESHOLD:-20}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRANSFORMER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$TRANSFORMER_DIR"

if [ -d "$BASE/v2" ]; then
  BASE_V2="$BASE/v2"
elif [ -f "$BASE/v2_mouse.parquet" ] || [ -f "$BASE/v2_mouse.tsv" ]; then
  BASE_V2="$BASE"
else
  BASE_V2="$BASE/v2"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$OUTPUT_BASE"
mkdir -p "$TMP_MGF_BASE"

declare -a SPECIES_MAP=(
  "mouse|25258|$BASE_V2/v2_mouse.parquet|$MGF_ROOT/Mus_mouse"
  "human|42296|$BASE_V2/v2_human.parquet|$MGF_ROOT/H.-sapiens"
  "clambacteria|78108|$BASE_V2/v2_clambacteria.parquet|$MGF_ROOT/Candidatus-endoloripes"
  "ricebean|106895|$BASE_V2/v2_ricebean.parquet|$MGF_ROOT/Vigna-mungo"
  "tomato|173332|$BASE_V2/v2_tomato.parquet|$MGF_ROOT/Solanum-lycopersicum"
  "honeybee|187130|$BASE_V2/v2_honeybee.parquet|$MGF_ROOT/Apis-mellifera"
  "yeast|567155|$BASE_V2/v2_yeast.parquet|$TMP_MGF_BASE/yeast"
  "bacillus_1|663731|$BASE_V2/v2_bacillus_1.parquet|$TMP_MGF_BASE/bacillus_1"
  "bacillus_2|663731|$BASE_V2/v2_bacillus_2.parquet|$TMP_MGF_BASE/bacillus_2"
)

if [ "$INCLUDE_MMAZEI" = "1" ]; then
  SPECIES_MAP=(
    "mouse|25258|$BASE_V2/v2_mouse.parquet|$MGF_ROOT/Mus_mouse"
    "human|42296|$BASE_V2/v2_human.parquet|$MGF_ROOT/H.-sapiens"
    "clambacteria|78108|$BASE_V2/v2_clambacteria.parquet|$MGF_ROOT/Candidatus-endoloripes"
    "ricebean|106895|$BASE_V2/v2_ricebean.parquet|$MGF_ROOT/Vigna-mungo"
    "tomato|173332|$BASE_V2/v2_tomato.parquet|$MGF_ROOT/Solanum-lycopersicum"
    "honeybee|187130|$BASE_V2/v2_honeybee.parquet|$MGF_ROOT/Apis-mellifera"
    "mmazei|263927|$BASE_V2/v2_mmazei.parquet|$MGF_ROOT/Methanosarcina-mazei"
    "yeast|567155|$BASE_V2/v2_yeast.parquet|$TMP_MGF_BASE/yeast"
    "bacillus_1|663731|$BASE_V2/v2_bacillus_1.parquet|$TMP_MGF_BASE/bacillus_1"
    "bacillus_2|663731|$BASE_V2/v2_bacillus_2.parquet|$TMP_MGF_BASE/bacillus_2"
  )
fi

if [ -n "${SPECIES:-}" ]; then
  declare -a FILTERED=()
  for want in $SPECIES; do
    for entry in "${SPECIES_MAP[@]}"; do
      name="${entry%%|*}"
      if [ "$name" = "$want" ]; then FILTERED+=("$entry"); break; fi
    done
  done
  SPECIES_MAP=("${FILTERED[@]}")
fi

merge_needed_mgfs() {
  log "Preparing merged MGF directories..."
  local yeast_dir="$TMP_MGF_BASE/yeast"
  local bac1_dir="$TMP_MGF_BASE/bacillus_1"
  local bac2_dir="$TMP_MGF_BASE/bacillus_2"
  mkdir -p "$yeast_dir" "$bac1_dir" "$bac2_dir"
  if ! find "$yeast_dir" -maxdepth 1 -name "*.mgf" | grep -q . 2>/dev/null; then
    cp "$MGF_ROOT"/Saccharomyces-cerevisiae-yeast_{1,2,3,4}/*.mgf "$yeast_dir"/
  fi
  if ! find "$bac1_dir" -maxdepth 1 -name "*.mgf" | grep -q . 2>/dev/null; then
    cp "$MGF_ROOT"/Bacillus_{1,2}/*.mgf "$bac1_dir"/
  fi
  if ! find "$bac2_dir" -maxdepth 1 -name "*.mgf" | grep -q . 2>/dev/null; then
    cp "$MGF_ROOT"/Bacillus_{3,4}/*.mgf "$bac2_dir"/
  fi
  log "MGF merge prep complete."
}

RESCORE_SCRIPT="scripts/rescoring/rescore_simplified.py"

run_one_species() {
  local name="$1"; local n_spectra="$2"; local parquet="$3"; local mgf_dir="$4"
  local out_dir="$OUTPUT_BASE/$name"
  local summary_json="$out_dir/rescoring_summary.json"

  if [ ! -f "$parquet" ]; then log "[$name] SKIP: missing parquet $parquet"; return 0; fi
  if [ ! -d "$mgf_dir" ]; then log "[$name] SKIP: missing mgf dir $mgf_dir"; return 0; fi
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$summary_json" ]; then
    log "[$name][$n_spectra] SKIP: existing summary at $summary_json"; return 0
  fi

  log "[$name][$n_spectra] start (alpha=$ALPHA, hard_threshold=$HARD_THRESHOLD)"
  # 保存模式：SAVE_FULL=1 -> full_results.csv(每候选一行, 流式);
  #           SAVE_DETAILED=1 -> rescoring_diagnostics.csv(每候选一行, 20 列);
  #           两个可以同时开；都不开则只留 summary.json。
  # 注意：不能用 `[ ... ] && cmd+=(...)` —— set -e 下判断为假会让整条语句返回非零直接退出。
  local save_args=()
  if [ "$SAVE_DETAILED" = "1" ]; then save_args+=(--save-diagnostics); fi
  if [ "$SAVE_FULL" = "1" ]; then save_args+=(--save-full-results); fi
  if [ "${#save_args[@]}" -eq 0 ]; then save_args+=(--skip-results-save); fi
  local cmd=(python "$RESCORE_SCRIPT"
    --parquet "$parquet" --mgf-dir "$mgf_dir"
    --checkpoint "$CKPT" --piprime-model "$PIPRIME"
    --fusion-alpha "$ALPHA" --precursor-hard-threshold "$HARD_THRESHOLD"
    --row-order --gpu "$GPU"
    --output-dir "$out_dir" "${save_args[@]}")
  if [ -n "$MAX_SCANS" ]; then cmd+=(--max-scans "$MAX_SCANS"); fi
  if [ "$name" = "mmazei" ] && [ "$MMAZEI_DROP_SEQ_WITH_O" = "1" ]; then cmd+=(--drop-seq-with-o); fi
  # ricebean: 前体质量修复 + 更宽的前体硬过滤 (57Da 错位数据, 实测最优 hard 20 + fix 20)
  if [ "$name" = "ricebean" ]; then
    cmd+=(--precursor-hard-threshold "$RICEBEAN_HARD_THRESHOLD")
    if [ "$RICEBEAN_PRECURSOR_MODE" != "off" ]; then
      cmd+=(--precursor-mismatch-mode "$RICEBEAN_PRECURSOR_MODE"
            --precursor-mismatch-ppm "$RICEBEAN_PRECURSOR_PPM")
    fi
  fi
  "${cmd[@]}"
  log "[$name][$n_spectra] done"
}

merge_needed_mgfs

log "===== Simplified suite (best mode): alpha=$ALPHA hard=$HARD_THRESHOLD ppm ====="
for entry in "${SPECIES_MAP[@]}"; do
  IFS='|' read -r name n_spectra parquet mgf_dir <<< "$entry"
  run_one_species "$name" "$n_spectra" "$parquet" "$mgf_dir"
done

if [ "$SKIP_SUMMARY" = "1" ]; then
  # 分批补跑时必须开这个：summarize_v2_mainline 是**覆盖写**，
  # 若只跑了一个子集却不跳过，会把已有的多物种汇总缩成这几行。
  log "SKIP_SUMMARY=1: 跳过汇总（分批跑时用，最后再对全部物种手工跑一次 summary）"
else
  summary_cmd=(python scripts/summary/summarize_v2_mainline.py
    --output-base "$OUTPUT_BASE" --out-prefix "$SUMMARY_PREFIX")
  if [ -n "${SPECIES:-}" ]; then summary_cmd+=(--species $SPECIES)
  elif [ "$INCLUDE_MMAZEI" != "1" ]; then
    summary_cmd+=(--species bacillus_1 bacillus_2 yeast honeybee tomato ricebean clambacteria human mouse)
  fi
  log "Building summary tables..."
  "${summary_cmd[@]}"
fi
log "All done."
