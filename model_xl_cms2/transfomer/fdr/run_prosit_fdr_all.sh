#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# Prosit 1% FDR —— 全物种一键自动流程
#
# 每个物种：step1/step2（TD 候选，缺才生成）→ prosit_fdr_score（≤30, reanchor+sqrt,
# Koina 预测+打分）→ step4_fdr_curve（(D+1)/T 计数）。
#
# env:
#   SPECIES   逗号列表（默认九物种）
#   ROOT      v1-search 根
#   RELS      rel 档（默认 0,0.05,0.10,0.20）
#   NOFDR_CACHE=1  复用同物种 no-FDR 的 Koina 缓存（../ds/output/prosit/<sp>/predictions/）
#                   否则用本目录 data/<sp>/prosit_fdr_cache.tsv.gz（独立缓存）
#   MAX_SCANS 仅调试小样
#   RETRY_N   单物种进程异常退出时重试次数（默认 0）
#
# 用法：
#   bash run_prosit_fdr_all.sh                       # 九物种
#   SPECIES=mouse NOFDR_CACHE=1 bash run_prosit_fdr_all.sh
#   nohup bash run_prosit_fdr_all.sh > nohup_prosit_fdr_all.log 2>&1 &
set -uo pipefail

FDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/wangyuhang/seq2mass/v1-search}"
SPECIES="${SPECIES:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"
RELS="${RELS:-0,0.05,0.10,0.20}"
NOFDR_CACHE="${NOFDR_CACHE:-0}"
MAX_SCANS="${MAX_SCANS:-}"
RETRY_N="${RETRY_N:-0}"
# 输出根目录（默认 = 本 fdr/ 目录）。设成 /data/... 即可把所有产物集中过去。
OUTPUT_DIR="${OUTPUT_DIR:-$FDR_DIR}"
# NOFDR_CACHE=1 时复用的 no-FDR Prosit 缓存目录（锚定代码树，不随 OUTPUT_DIR 变）
NOFDR_CACHE_DIR="${NOFDR_CACHE_DIR:-$FDR_DIR/../ds/output/prosit}"

log(){ echo "[$(date '+%F %T')] $*"; }
cd "$FDR_DIR"

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  log "===== Prosit FDR: $sp ====="
  mkdir -p "$OUTPUT_DIR/data/$sp" "$OUTPUT_DIR/output/$sp"

  # 1) TD 候选（fdr step1/2，缺才建）
  if [ ! -f "$OUTPUT_DIR/data/$sp/candidates_${sp}_td.tsv.gz" ]; then
    log "  [1] build TD candidates: $sp"
    python step1_build_td_fasta.py --species "$sp" --root "$ROOT" \
      --out-fasta "$OUTPUT_DIR/data/$sp/${sp}_td.fasta" --out-dir "$OUTPUT_DIR/output/$sp" \
      ${MAX_SCANS:+--max-peptides "$MAX_SCANS"} || { log "  step1 FAIL $sp"; continue; }
    python step2_match_td.py --species "$sp" --root "$ROOT" \
      --fasta "$OUTPUT_DIR/data/$sp/${sp}_td.fasta" \
      --out-candidates "$OUTPUT_DIR/data/$sp/candidates_${sp}_td.tsv.gz" \
      --out-dir "$OUTPUT_DIR/output/$sp" || { log "  step2 FAIL $sp"; continue; }
  else
    log "  [1] TD candidates exist: $OUTPUT_DIR/data/$sp/candidates_${sp}_td.tsv.gz"
  fi

  # 2) Prosit 打分（≤30）
  if [ "$NOFDR_CACHE" = "1" ]; then
    CACHE="$NOFDR_CACHE_DIR/$sp/predictions/prosit_cache.tsv.gz"
  else
    CACHE="$OUTPUT_DIR/data/$sp/prosit_fdr_cache.tsv.gz"
  fi
  DECOY_CACHE="${DECOY_CACHE:-$OUTPUT_DIR/data/$sp/prosit_fdr_decoy_cache.tsv.gz}"
  attempt=0
  while :; do
    log "  [2] prosit_fdr_score: $sp (cache=$CACHE decoy_cache=$DECOY_CACHE rels=$RELS)"
    python prosit_fdr_score.py --species "$sp" --root "$ROOT" \
      --candidates "$OUTPUT_DIR/data/$sp/candidates_${sp}_td.tsv.gz" \
      --out-dir "$OUTPUT_DIR/output/$sp" --cache "$CACHE" --decoy-cache "$DECOY_CACHE" \
      --rel-intensities "$RELS" \
      ${MAX_SCANS:+--max-scans "$MAX_SCANS"} && break
    attempt=$((attempt+1)); log "  [2] attempt $attempt FAILED"
    [ "$attempt" -gt "$RETRY_N" ] && break
    sleep 10
  done

  # 3) (D+1)/T 计数
  log "  [3] FDR counting: $sp"
  python step4_fdr_curve.py --species "$sp" --out-dir "$OUTPUT_DIR/output/$sp" \
    --results "$OUTPUT_DIR/output/$sp/results_prositfdr_${sp}.csv" \
    --fdr-targets 0.01,0.05,0.10 || log "  step4 FAIL $sp"

  # 4) 打印 1% 摘要（可选）
  python - "$sp" "$OUTPUT_DIR" <<'PY' || true
import json, sys
sp = sys.argv[1]
try:
    d = json.load(open(f"{sys.argv[2]}/output/{sp}/summary_fdr_{sp}.json"))
except Exception as e:
    print(f"  {sp}: no summary ({e})"); sys.exit(0)
for k, b in sorted(d.get("branches", {}).items()):
    l = (b.get("fdr_lines") or {}).get("0.01") or {}
    if l:
        print(f"  {k:<26} n_target@1%={l.get('n_target')}  n_decoy={l.get('n_decoy')}")
PY
  log "===== done: $sp ====="
done
log "===== Prosit FDR ALL DONE ====="
