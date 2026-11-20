#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# Supervisor for step4_prosit_score.py: restart only when PREDICTION stalls.
#
# v2: stall detection is ARMED only after the log shows "predicting N missing"
# (i.e. the big cache has finished loading and Koina prediction has started).
# During cache load / collect (which can take many minutes for a huge b/y cache
# and writes no new rows) the process is left alone, so we never kill a loader.
#
# env:
#   SPECIES BATCH RELS ROOT  (as before)
#   POLL_SEC STALL_MIN MAX_MIN
set -u

SP="${SPECIES:-mouse}"
DS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DS_DIR/output/prosit/$SP"
CACHE="$OUT/predictions/prosit_cache.tsv.gz"
ROOT="${ROOT:-/home/wangyuhang/seq2mass/v1-search}"
BATCH="${BATCH:-1500}"
RELS="${RELS:-0,0.05,0.10,0.20}"
POLL="${POLL_SEC:-120}"
STALL="$(( ${STALL_MIN:-8} * 60 ))"
MAX_RUN="$(( ${MAX_MIN:-0} * 60 ))"
LOG="$DS_DIR/nohup_prosit_${SP}.log"

mkdir -p "$OUT/predictions"
run(){
  nohup python "$DS_DIR/step4_prosit_score.py" --species "$SP" \
    --root "$ROOT" --candidates "$DS_DIR/data/$SP/candidates_${SP}_scans.tsv.gz" \
    --out-dir "$OUT" --scoring branches \
    --rel-intensities "$RELS" --mods-mode reanchor \
    --exp-transform sqrt_clip --batch-size "$BATCH" \
    > "$LOG" 2>&1 &
  echo $!
}
rows(){
  n=0
  for f in "$CACHE" "$CACHE".seg_*.tsv.gz; do
    [ -f "$f" ] && n=$(( n + $(gzip -cd "$f" 2>/dev/null | wc -l) ))
  done
  echo "$n"
}

echo "[$(date '+%F %T')] supervisor start: $SP (batch=$BATCH)"
t_start=$(date +%s)
pid=$(run)
armed=0
last_rows=-1
echo "[$(date '+%F %T')] started pid=$pid rows=$(rows) log=$LOG"

while true; do
  sleep "$POLL"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[$(date '+%F %T')] process exited (pid=$pid)"
    break
  fi
  [ "$MAX_RUN" -gt 0 ] && { sec=$(( $(date +%s) - t_start )); \
    [ "$sec" -gt "$MAX_RUN" ] && { echo "[$(date '+%F %T')] max run ${sec}s -> stop"; kill -9 "$pid"; break; }; }

  # scoring finished: just wait for exit
  if grep -q "prediction done" "$LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] prediction done, waiting for scoring ..."
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    echo "[$(date '+%F %T')] finished"
    break
  fi

  # ARM stall detection only once prediction has actually started
  if [ "$armed" = 0 ]; then
    if grep -q "predicting .* missing peptides" "$LOG" 2>/dev/null; then
      armed=1; last_rows=$(rows)
      echo "[$(date '+%F %T')] prediction started -> armed (rows=$last_rows)"
    else
      continue   # still loading cache / collecting; never kill here
    fi
  fi

  cur=$(rows)
  if [ "$cur" != "$last_rows" ]; then
    last_rows=$cur
    continue
  fi
  # no growth this poll: give a full stall window, re-check
  stalled=$STALL
  while [ "$stalled" -gt 0 ] && kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL"
    stalled=$(( stalled - POLL ))
    cur2=$(rows)
    if [ "$cur2" != "$last_rows" ]; then last_rows=$cur2; stalled=-1; fi
  done
  if [ "$stalled" -ge 0 ] && kill -0 "$pid" 2>/dev/null; then
    echo "[$(date '+%F %T')] STALL during prediction (rows=$last_rows) -> restart"
    kill -9 "$pid" 2>/dev/null; sleep 5
    pid=$(run); armed=0; last_rows=-1
    echo "[$(date '+%F %T')] restarted pid=$pid rows=$(rows)"
  fi
done
echo "[$(date '+%F %T')] supervisor done. summary: $OUT/summary_prosit_${SP}.json"
