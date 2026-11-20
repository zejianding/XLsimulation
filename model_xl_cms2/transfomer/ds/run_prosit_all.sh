#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# Prosit search accuracy 全物种一键（无 FDR；打分 = 我们 d/f/m_by_prec + hsdf + hsdf_mby）
# Prosit 预测走 Koina 远程，不需要本地 GPU；带增量缓存，断点重跑续传。
# env:
#   SPECIES        逗号分隔物种（默认九物种）
#   REL_INTENSITIES 过滤档（默认 0.05,0.10,0.20）
#   CANDIDATES_DIR  候选目录（默认 ds data/<sp>/；文件取 candidates_<sp>_scans.tsv.gz，
#                    缺则回退 candidates_<sp>_full.tsv.gz）
#   OUT_DIR        输出根（默认 output/prosit）
set -euo pipefail

DS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/wangyuhang/seq2mass/v1-search}"
SPECIES="${SPECIES:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"
REL="${REL_INTENSITIES:-0.05,0.10,0.20}"
CAND_ROOT="${CANDIDATES_DIR:-$DS_DIR/data}"
OUT_ROOT="${OUT_DIR:-$DS_DIR/output/prosit}"
mkdir -p "$OUT_ROOT"

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  logf="$DS_DIR/nohup_prosit_${sp}.log"
  cand="$CAND_ROOT/$sp/candidates_${sp}_scans.tsv.gz"
  [ -f "$cand" ] || cand="$CAND_ROOT/$sp/candidates_${sp}_full.tsv.gz"
  if [ ! -f "$cand" ]; then
    echo "SKIP $sp: no candidates at $cand" | tee "$logf"
    continue
  fi
  echo "===== $(date '+%F %T') Prosit: $sp (rel=$REL) ====="
  python "$DS_DIR/step4_prosit_score.py" --species "$sp" \
    --root "$ROOT" --candidates "$cand" \
    --out-dir "$OUT_ROOT/$sp" --scoring branches \
    --rel-intensities "$REL" --mods-mode reanchor \
    > "$logf" 2>&1
  echo "done $sp -> $OUT_ROOT/$sp/summary_prosit_${sp}.json (log: $logf)"
done
echo "===== ALLDONE ====="
