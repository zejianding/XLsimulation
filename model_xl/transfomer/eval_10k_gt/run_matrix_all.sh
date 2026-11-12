#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# 对 v20pp / v20raw 两个主线 run 的九物种，批量跑「rel 强度过滤 × 投影」矩阵。
#
# 每个 run 产出 2 个变体目录（raw_ / grid_），各含 3 个 rel 档：
#   variants/raw_rel005-020_expnone_prednone/Grouped_MGFs_rel005|010|020/
#   variants/grid_rel005-020_expnone_prednone/Grouped_MGFs_rel005|010|020/
# 每个 rel 档下：Grouped_MGFs/ Embeddings/ cosine_summary.csv cosine_detail.csv
#
# 断点续跑：某 run 下已有 variants/ 就跳过；中断后重跑本脚本即可接着来。
#
# env（都有默认，可覆盖）：
#   OUT       结果根目录        (默认 /data/wangyuhang2/v20)
#   TAGS      逗号分隔的主线标签 (默认 v20pp,v20raw)
#   SPECIES   逗号分隔的物种     (默认九物种)
#   PROJ/RELS/EXP/PRED/ADD_EXPNONE  透传给 run_matrix.sh
# ============================================================================
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/data/wangyuhang2/v20}"
TAGS="${TAGS:-v20pp,v20raw}"
SPECIES="${SPECIES:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"

PROJ="${PROJ:-raw,grid}"
RELS="${RELS:-0.05,0.10,0.20}"
EXP="${EXP:-none}"          # 必须 none：默认 sqrt_clip 会让原谱被压两次、预测谱只一次
PRED="${PRED:-none}"
ADD_EXPNONE="${ADD_EXPNONE:-0}"

cd "$DIR" || exit 1
mkdir -p "$OUT/logs"

RUN_MATRIX="$DIR/run_matrix.sh"
[ -f "$RUN_MATRIX" ] || { echo "ERROR: 找不到 $RUN_MATRIX"; exit 1; }

# ---- 前置自检：确认服务器上的副本是打过补丁的版本，别在旧脚本上白跑几小时 ----
need() {  # $1=文件 $2=必须出现的一段文本 $3=说明
  grep -qF "$2" "$1" || { echo "ERROR: $1 里找不到「$2」—— 这是旧版脚本，先同步再跑"; echo "       ($3)"; exit 1; }
}
need "$DIR/filter_group_predictions.py" 'OUTDIR=' '新版会打印 OUTDIR= 给 run_matrix.sh 解析'
need "$RUN_MATRIX"                      'basename "$outdir"' '修过的改名逻辑，旧版会把 variants 移进自己子目录'
[ -f "$DIR/cosine_detail_csv.py" ] || { echo "ERROR: 缺 cosine_detail_csv.py"; exit 1; }
[ -f "$DIR/../batch_cosine_summary.py" ] || { echo "ERROR: 缺 ../batch_cosine_summary.py"; exit 1; }
command -v python >/dev/null || { echo "ERROR: python 不在 PATH"; exit 1; }
echo "前置自检通过: $(date '+%F %T')"

# ---- 完整性判据：variants 下的每个 Grouped_MGFs_rel* 都要有 summary + detail ----
# 只看目录存不存在是不够的：中途中断会留下「只有 MGF、没有 Embeddings/CSV」的空壳，
# 那种必须重跑，否则会被永久跳过。
stat_variants() {  # 输出 "rel档 summary detail"
  local v="$1/variants"
  [ -d "$v" ] || { echo "0 0 0"; return; }
  echo "$(find "$v" -maxdepth 2 -type d -name "Grouped_MGFs_rel*" 2>/dev/null | wc -l) \
$(find "$v" -maxdepth 3 -name cosine_summary.csv 2>/dev/null | wc -l) \
$(find "$v" -maxdepth 3 -name cosine_detail.csv 2>/dev/null | wc -l)"
}

n_done=0; n_skip=0; n_fail=0; n_incomplete=0
for TAG in $(echo "$TAGS" | tr ',' ' '); do
  for SP in $(echo "$SPECIES" | tr ',' ' '); do
    RUN_DIR=$(ls -dt "$OUT/output/${TAG}_${SP}_"* 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
      echo "skip $TAG/$SP: 无主线 run"; n_skip=$((n_skip+1)); continue
    fi

    read -r n_rel n_sum n_det <<< "$(stat_variants "$RUN_DIR")"
    if [ "$n_rel" -gt 0 ] && [ "$n_sum" -eq "$n_rel" ] && [ "$n_det" -eq "$n_rel" ]; then
      echo "== $TAG/$SP 已有完整 variants（$n_rel 档），跳过"; n_skip=$((n_skip+1)); continue
    fi
    if [ "$n_rel" -gt 0 ]; then
      echo "!! $TAG/$SP variants 不完整（rel档=$n_rel summary=$n_sum detail=$n_det）—— 重跑本 run"
      n_incomplete=$((n_incomplete+1))
    fi

    echo "##### $TAG/$SP matrix  $(date '+%F %T')  -> $RUN_DIR"
    if RUN_DIR="$RUN_DIR" PROJ="$PROJ" RELS="$RELS" EXP="$EXP" PRED="$PRED" \
         ADD_EXPNONE="$ADD_EXPNONE" bash "$RUN_MATRIX" \
         > "$OUT/logs/matrix_${TAG}_${SP}.log" 2>&1 ; then
      # 逐档验收：每个 Grouped_MGFs_rel* 下都要有 cosine_summary.csv + cosine_detail.csv
      n_rel=$(find "$RUN_DIR/variants" -maxdepth 2 -type d -name "Grouped_MGFs_rel*" 2>/dev/null | wc -l)
      n_sum=$(find "$RUN_DIR/variants" -maxdepth 3 -name cosine_summary.csv 2>/dev/null | wc -l)
      n_det=$(find "$RUN_DIR/variants" -maxdepth 3 -name cosine_detail.csv  2>/dev/null | wc -l)
      if [ "$n_rel" -gt 0 ] && [ "$n_sum" -eq "$n_rel" ] && [ "$n_det" -eq "$n_rel" ]; then
        echo "    OK: rel档 $n_rel 个，每档都有 summary+detail"
      else
        echo "    !! 跑完了但不完整：rel档 $n_rel，cosine_summary $n_sum，cosine_detail $n_det"
        echo "       缺的档位："
        for d in $(find "$RUN_DIR/variants" -maxdepth 2 -type d -name "Grouped_MGFs_rel*" 2>/dev/null); do
          [ -f "$d/cosine_summary.csv" ] && [ -f "$d/cosine_detail.csv" ] || echo "         $d"
        done
        echo "       日志末尾 20 行："
        tail -20 "$OUT/logs/matrix_${TAG}_${SP}.log" | sed 's/^/         /'
        n_fail=$((n_fail+1))
        continue
      fi
      n_done=$((n_done+1))
    else
      echo "    !! 失败 -> logs/matrix_${TAG}_${SP}.log（末尾 20 行：）"
      tail -20 "$OUT/logs/matrix_${TAG}_${SP}.log" | sed 's/^/       /'
      n_fail=$((n_fail+1))
    fi
  done
done

echo "===== matrix 完成: 成功 $n_done / 跳过(已完整) $n_skip / 失败 $n_fail / 曾发现不完整并重跑 $n_incomplete   $(date '+%F %T')"
[ "$n_fail" -gt 0 ] && echo "!!! 有失败项，逐个看 $OUT/logs/matrix_*.log"
exit 0
