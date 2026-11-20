#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# Local MS-GF+ leg for fullsearch: for each species build the GT library and run
# MS-GF+ against the small GT TD FASTA (base_plain + rev_ decoys + mods), then
# parse top-1 search accuracy vs the MGF GT.
set -uo pipefail

WORK="e:/PythonProject/Seq2Mass/v20_0821/transfomer/fullsearch"
ROOTL="E:/biodatasets/denovo/v1-search"
FDRFULL="e:/PythonProject/Seq2Mass/v20_0821/transfomer/fdr_full"
SPECIES="${1:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"

log(){ echo "[$(date '+%F %T')] $*"; }

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  log "===== $sp ====="
  mkdir -p "$WORK/output_msgfplus/$sp"
  log "[1] GT library (step1)"
  python "$WORK/step1_gt_library.py" --species "$sp" --root "$ROOTL" --work "$WORK" || { log "step1 FAIL $sp"; exit 1; }
  log "[2] annotated variant library (step2)"
  python "$WORK/step2_build_library.py" --species "$sp" --work "$WORK" || { log "step2 FAIL $sp"; exit 1; }
  MZ="$(ls "$ROOTL"/v2mzml/high.${sp}.PXD*.mzML 2>/dev/null | head -1)"
  [ -n "$MZ" ] || { log "no mzML for $sp"; continue; }
  log "[3] MS-GF+ small-library search: $MZ"
  bash "$FDRFULL/run_msgfplus.sh" "$sp" \
    "$WORK/data/$sp/base_plain.fasta" "$MZ" "$WORK/output_msgfplus/$sp" \
    || { log "msgfplus FAIL $sp"; continue; }
  log "[4] parse top-1 accuracy"
  python "$WORK/step5_msgfplus_acc.py" --species "$sp" --work "$WORK" \
    --root "$ROOTL" --tsv "$WORK/output_msgfplus/$sp/$sp.tsv" --scan-offset 0 \
    || log "step5 FAIL $sp"
done
log "===== all local MS-GF+ done ====="
