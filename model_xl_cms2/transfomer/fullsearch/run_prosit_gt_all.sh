#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# Prosit GT-library blind search (no precursor filter), all species.
# Prosit is intrinsically <=30: library entries with backbone>30 get empty
# spectra and spectra with GT>30 are excluded (--max-gt-len 30).  Uses the
# special Prosit handling: sqrt intensity alignment + rel 0 (no intensity
# filter) branch family.  Requires the synced ds step4_prosit_score.py
# (y-index fix + rel0) that step3_predict_prosit imports.
#
# Usage:
#   bash run_prosit_gt_all.sh                      # 9 species
#   bash run_prosit_gt_all.sh mouse,human          # subset
# env: ROOT (v1-search), RELS (default 0,0.05,0.10,0.20)
set -uo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/wangyuhang/seq2mass/v1-search}"
SPECIES="${1:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"
RELS="${RELS:-0,0.05,0.10,0.20}"
mkdir -p "$WORK/output"

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  echo "##### $sp @ $(date '+%F %T')"
  python "$WORK/step1_gt_library.py" --species "$sp" --root "$ROOT" --work "$WORK" || exit 1
  python "$WORK/step2_build_library.py" --species "$sp" --work "$WORK" || exit 1
  python "$WORK/step3_predict_prosit.py" --species "$sp" --work "$WORK" || exit 1
  python "$WORK/step4_blind_score.py" --species "$sp" --work "$WORK" \
    --root "$ROOT" --predictor prosit --max-gt-len 30 --rels "$RELS" \
    > "$WORK/output/blind_prosit_${sp}.log" 2>&1 || echo "  step4 failed $sp (see log)"
  echo "  -> $WORK/output/$sp/summary_blind_prosit_${sp}.json"
done
echo "===== ALDONE $(date '+%F %T') ====="
