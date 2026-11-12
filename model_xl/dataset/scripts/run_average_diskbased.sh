#!/bin/bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
set -euo pipefail

# Usage:
#   bash dataset/scripts/run_average_diskbased.sh <input_filtered.mgf> <output_averaged.mgf> [buckets]

INPUT_MGF="${1:?Usage: $0 <input_filtered.mgf> <output_averaged.mgf> [buckets]}"
OUTPUT_MGF="${2:?Usage: $0 <input_filtered.mgf> <output_averaged.mgf> [buckets]}"
BUCKETS="${3:-256}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python "${DATASET_DIR}/AverageSpectrum_diskbased.py" \
  "${INPUT_MGF}" "${OUTPUT_MGF}" \
  --tol 0.05 \
  --buckets "${BUCKETS}"
