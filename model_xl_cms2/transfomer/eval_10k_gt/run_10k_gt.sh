#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# 1万肽「预测谱 vs 原谱」余弦对比 · 统一入口（V20 / Prosit）
#
# env:
#   PREDICTOR   v20 | prosit            (默认 v20)
#   CKPT        V20 权重（PREDICTOR=v20 必填）
#   GPU         cuda 序号               (默认 0)
#   NAME        输出名（默认 <predictor>_10k，落在 output/<NAME>_<时间戳>/）
#   TEST_NPZ    输入 NPZ（默认 data/v20_10k_9species.npz）
#   ROOT_DIR    9 物种原谱目录（默认 E:/biodatasets/denovo/9species/Casanovo_preprocessed_data）
#   P4_CACHE    Prosit 的 Koina 缓存（可选，续传）
#   MAX_EXAMPLES/P4_BATCH 等透传
#
# 用法：
#   CKPT=/path/best_model.pt bash run_10k_gt.sh                 # V20
#   PREDICTOR=prosit bash run_10k_gt.sh                         # Prosit
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICTOR="${PREDICTOR:-v20}"
NAME="${NAME:-${PREDICTOR}_10k}"

case "$PREDICTOR" in
  v20)
    : "${CKPT:?PREDICTOR=v20 需要 CKPT=/path/best_model.pt}"
    bash "$DIR/run_v20_10k.sh" "$CKPT" "${GPU:-0}" "$NAME" ;;
  prosit)
    bash "$DIR/run_prosit_10k.sh" "$NAME" ;;
  *) echo "PREDICTOR must be v20|prosit" >&2; exit 2 ;;
esac
