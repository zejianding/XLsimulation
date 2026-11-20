#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# fullsearch orchestrator — GT-derived library, blind whole-library top-1
# search accuracy per branch for V20 / Prosit, plus the MS-GF+ small-library leg.
#
# env:
#   WORK          fullsearch root (default = this dir)
#   SPECIES       comma list (default mouse; 'all' = 9 species)
#   PREDICTORS    comma list in {v20,prosit,msgfplus} (default all three)
#   ROOT          v1-search data root
#   CKPT          V20 checkpoint
#   GPU           CUDA index (step3_v20)
#   MAX_SCANS     cap spectra in steps 1/4 (empty = all 10000)
#   RELS          rel levels for step4 (default 0.05,0.10,0.20)
#   MSGF_TSV_STEM  per-species MS-GF+ MzidToTsv path template (see MSGF leg)
#
# Example (pilot):
#   SPECIES=mouse MAX_SCANS=50 PREDICTORS=v20,msgfplus bash run_fullsearch.sh
# Example (full all species, background):
#   SPECIES=all nohup bash run_fullsearch.sh > nohup_fullsearch.log 2>&1 &
# ============================================================================
set -euo pipefail

ALL="bacillus,clambacteria,honeybee,human,mmazei,mouse,ricebean,tomato,yeast"
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/wangyuhang/seq2mass/v1-search}"
CKPT="${CKPT:-/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt}"
SPECIES="${SPECIES:-mouse}"
[ "$SPECIES" = "all" ] && SPECIES="$ALL"
PREDICTORS="${PREDICTORS:-v20,prosit,msgfplus}"
GPU="${GPU:-0}"
MAX_SCANS="${MAX_SCANS:-}"
RELS="${RELS:-0.05,0.10,0.20}"
# MS-GF+ leg: MzidToTsv output path; default assumes run via
# fdr_full/run_msgfplus.sh with NAME=<sp> writing <msgf_out>/<sp>.tsv.
MSGF_OUT="${MSGF_OUT:-$WORK/output_msgfplus}"

log(){ echo "[$(date '+%F %T')] $*"; }
python_ds(){ python "$WORK/$1"; }

mkdir -p "$WORK/data" "$WORK/output"

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  log "===== species: $sp ====="
  mkdir -p "$WORK/data/$sp" "$WORK/output/$sp"

  log "[1] collect GT library"
  python "$WORK/step1_gt_library.py" --species "$sp" --root "$ROOT" --work "$WORK" \
    ${MAX_SCANS:+--max-scans "$MAX_SCANS"}

  log "[2] build target+decoy variant library"
  python "$WORK/step2_build_library.py" --species "$sp" --work "$WORK"

  if [[ ",$PREDICTORS," == *",v20,"* ]]; then
    log "[3v] V20 library prediction"
    python "$WORK/step3_predict_v20.py" --species "$sp" --work "$WORK" \
      --checkpoint "$CKPT" --gpu "$GPU"
    log "[4v] V20 blind whole-library scoring"
    python "$WORK/step4_blind_score.py" --species "$sp" --work "$WORK" \
      --root "$ROOT" --predictor v20 --rels "$RELS" \
      ${MAX_SCANS:+--max-scans "$MAX_SCANS"}
  fi

  if [[ ",$PREDICTORS," == *",prosit,"* ]]; then
    log "[3p] Prosit library prediction (Koina)"
    python "$WORK/step3_predict_prosit.py" --species "$sp" --work "$WORK"
    log "[4p] Prosit blind whole-library scoring (GT >30 excluded)"
    python "$WORK/step4_blind_score.py" --species "$sp" --work "$WORK" \
      --root "$ROOT" --predictor prosit --max-gt-len 30 --rels "$RELS" \
      ${MAX_SCANS:+--max-scans "$MAX_SCANS"}
  fi

  if [[ ",$PREDICTORS," == *",msgfplus,"* ]]; then
    log "[5m] MS-GF+ small-library leg (expect $MSGF_OUT/${sp}/${sp}.tsv)"
    tsv="$MSGF_OUT/$sp/${sp}.tsv"
    if [ -f "$tsv" ]; then
      python "$WORK/step5_msgfplus_acc.py" --species "$sp" --work "$WORK" \
        --root "$ROOT" --tsv "$tsv" --scan-offset 0
    else
      log "  SKIP: $tsv not found (run fdr_full/run_msgfplus.sh <sp> $WORK/data/$sp/base_plain.fasta <mzML> $MSGF_OUT/$sp)"
    fi
  fi
  log "===== done: $sp ====="
done
log "===== fullsearch all done ====="
