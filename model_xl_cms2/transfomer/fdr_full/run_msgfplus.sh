#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# MS-GF+ 1% FDR 全自动：base 肽段 FASTA + mods JSON -> TD FASTA -> MS-GF+ 搜索 -> 1% FDR 计数。
# 可对任意数据集使用（修饰由 mods JSON 控制；指标 = 1% FDR 线上 n_target/n_decoy，不比对 GT）。
#
# env:
#   MSGFPLUS_JAR   MS-GF+ jar 路径（默认 E:/MSGFPLUS/MSGFPlus_v20240326/MSGFPlus.jar）
#   MZID2TSV       MzidToTsvConverter.exe 路径（默认同目录）
#   MODS           mods_config JSON（默认 fdr_full/mods_config.example.json）
#   THREADS        MS-GF+ 线程（默认 8）
#   INST           -inst 仪器档（默认 1）
#   MAX_SCANS      （MS-GF+ 无此选项，忽略——如需截断请先截 MGF）
#
# Usage:
#   bash run_msgfplus.sh <name> <spec_fasta> <spec_mzML_or_mgf> <out_dir>
# 例：
#   bash run_msgfplus.sh mouse for_mouse_test_incope_peptide.fasta mouse.mzML output/mouse
set -euo pipefail

MSG="${MSGFPLUS_JAR:-E:/MSGFPLUS/MSGFPlus_v20240326/MSGFPlus.jar}"
MZ2TSV="${MZID2TSV:-E:/MSGFPLUS/MSGFPlus_v20240326/MzidToTsvConverter/MzidToTsvConverter.exe}"
FDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODS="${MODS:-$FDR_DIR/mods_config.example.json}"
THREADS="${THREADS:-8}"
INST="${INST:-1}"

[ $# -ge 4 ] || { echo "usage: $0 <name> <spec_fasta> <spec_mzML> <out_dir>"; exit 1; }
NAME="$1"; SPEC_FASTA="$2"; SPEC_MS="$3"
mkdir -p "$4"
OUT="$(cd "$4" && pwd)"   # absolute (MzidToTsvConverter requires absolute paths)
LOG="$OUT/${NAME}_msgfplus.log"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== MS-GF+ 1% FDR: $NAME ==="
log "[1/4] build plain TD FASTA + engine mods file (mods=$MODS)"
python "$FDR_DIR/build_td_fasta.py" --fasta "$SPEC_FASTA" \
    --mods-config "$MODS" --out-fasta "$OUT/${NAME}_td.fasta" \
    --mods-out "$OUT/${NAME}_mods.txt"

log "[2/4] MS-GF+ search (no -tda; decoys in the rev_ TD FASTA; -mod from mods JSON)"
java -Xmx12G -jar "$MSG" -s "$SPEC_MS" -d "$OUT/${NAME}_td.fasta" \
    -o "$OUT/${NAME}.mzid" \
    -t 10ppm -ti 0,0 -m 3 -inst "$INST" -e 9 -minCharge 2 -maxCharge 4 \
    -mod "$OUT/${NAME}_mods.txt" \
    -thread "$THREADS" -allowDenseCentroidedPeaks 1 >>"$LOG" 2>&1

log "[3/4] mzid -> tsv (.exe, absolute paths)"
"$MZ2TSV" "$OUT/${NAME}.mzid" -mzid:"$OUT/${NAME}.mzid" \
    -tsv:"$OUT/${NAME}.tsv" -unroll -showDecoy >>"$LOG" 2>&1

log "[4/4] 1% FDR count (per-scan, rev_ decoy)"
python "$FDR_DIR/engine_fdr.py" --engine msgfplus --tsv "$OUT/${NAME}.tsv" \
    --decoy-token rev_ --fdr 0.01 | tee -a "$LOG"

log "done -> $OUT/${NAME}_td.fasta / ${NAME}.tsv ; log: $LOG"
