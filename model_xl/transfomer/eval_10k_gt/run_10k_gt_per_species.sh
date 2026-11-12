#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# 逐物种跑「预测谱 vs 原谱」余弦（V20 / Prosit 通用）——物种之间完全独立：
# 每个物种用自己的肽表、自己的原谱目录，产出各自独立的 output/<pred>_<sp>_<stamp>/。
#
# env:
#   PREDICTOR   v20 | prosit               (默认 v20)
#   CKPT        V20 权重（PREDICTOR=v20 必填）
#   GPU         cuda 序号                  (默认 0)
#   SRC         原谱根：9species 目录（<sp>.mgf）或 v1-search（<sp>/*.mgf）
#   SPECIES     逗号分隔，默认九物种
#   PER_SPECIES 每物种肽数上限，0=该物种全部唯一肽（默认 0）
#   P4_CACHE    Prosit Koina 缓存（可选）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICTOR="${PREDICTOR:-v20}"
SRC="${SRC:?need SRC=<corpus root>}"
SPECIES="${SPECIES:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"
PER_SPECIES="${PER_SPECIES:-0}"
mkdir -p "$DIR/data"

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  # 找该物种的原谱入口：目录布局优先，其次单文件
  if [ -d "$SRC/$sp" ]; then ROOT_DIR="$SRC/$sp"; else ROOT_DIR="$SRC/$sp.mgf"; fi
  [ -e "$ROOT_DIR" ] || { echo "SKIP $sp: no spectra at $ROOT_DIR"; continue; }

  NPZ="$DIR/data/npz_${sp}.npz"
  echo "########## $sp  ($(date '+%F %T'))  root=$ROOT_DIR"
  # [0] 该物种的肽表（只含该物种 MGF 的 SEQ=/CHARGE= 去重）
  python "$DIR/make_peptide_npz.py" --root "$SRC" --species "$sp" \
    --per-species "$PER_SPECIES" --out "$NPZ"

  # [1..5] 预测 → 收集该物种原谱 → 嵌入 → 余弦（CSV）
  if [ "$PREDICTOR" = "v20" ]; then
    : "${CKPT:?PREDICTOR=v20 需要 CKPT=/path/best_model.pt}"
    # 输出名区分两种导出：v20pp_<sp>（cluster 后处理）/ v20raw_<sp>（无后处理）
    _tag=$([ "${POSTPROC:-cluster}" = "none" ] && echo raw || echo pp)
    TEST_NPZ="$NPZ" ROOT_DIR="$ROOT_DIR" bash "$DIR/run_v20_10k.sh" "$CKPT" "${GPU:-0}" "v20${_tag}_${sp}"
  else
    _ptag=""; [ "${P4_BYONLY:-0}" = "1" ] && _ptag="_byonly"; [ "${P4_RAW:-0}" = "1" ] && _ptag="_rawapi"
    TEST_NPZ="$NPZ" ROOT_DIR="$ROOT_DIR" P4_CACHE="${P4_CACHE:-}" bash "$DIR/run_prosit_10k.sh" "prosit${_ptag}_${sp}"
  fi
done
echo "===== per-species $PREDICTOR DONE ====="
