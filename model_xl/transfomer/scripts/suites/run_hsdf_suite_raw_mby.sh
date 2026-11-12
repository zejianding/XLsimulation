#!/bin/bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# HSDF V2 全物种重打分: fusion=raw, use_mby=1
# Usage:
#   export CKPT=/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt
#   export BASE_V2=/home/wangyuhang/seq2mass/PI_CasanovoV2_candidates/v2
#   export MGF_ROOT=/home/wangyuhang/datasets/90982/90982
#   bash scripts/suites/run_hsdf_suite_raw_mby.sh
set -euo pipefail
export FUSION=raw
export USE_MBY=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/hsdf_suite_lib.sh"
run_all
