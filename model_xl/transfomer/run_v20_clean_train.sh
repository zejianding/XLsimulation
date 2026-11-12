#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# =============================================================================
# V20 clean native baseline training entry
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V20_OUTPUT_DIR="${V20_OUTPUT_DIR:-$SCRIPT_DIR/../../runs/v20_clean_train}"
V20_TRAIN_NPZ="${V20_TRAIN_NPZ:-$SCRIPT_DIR/../../datasets/split_8_1_1/train.npz}"
V20_VAL_NPZ="${V20_VAL_NPZ:-$SCRIPT_DIR/../../datasets/split_8_1_1/val.npz}"
V20_TEST_NPZ="${V20_TEST_NPZ:-$SCRIPT_DIR/../../datasets/split_8_1_1/test.npz}"
V20_GPU="${V20_GPU:-0}"
V20_EPOCHS="${V20_EPOCHS:-10}"
V20_BATCH_SIZE="${V20_BATCH_SIZE:-64}"
V20_LR="${V20_LR:-0.001}"
V20_WEIGHT_DECAY="${V20_WEIGHT_DECAY:-0.0003}"
V20_D_MODEL="${V20_D_MODEL:-256}"
V20_N_HEADS="${V20_N_HEADS:-8}"
V20_N_LAYERS="${V20_N_LAYERS:-6}"
V20_FF_DIM="${V20_FF_DIM:-1024}"
V20_DROPOUT="${V20_DROPOUT:-0.15}"
V20_NUM_WORKERS="${V20_NUM_WORKERS:-4}"
V20_NO_PACKED="${V20_NO_PACKED:-0}"
V20_BACKBONE_WEIGHT="${V20_BACKBONE_WEIGHT:-1.0}"
V20_STRICT_PRESENCE_WEIGHT="${V20_STRICT_PRESENCE_WEIGHT:-0.30}"
V20_GROUP_FAMILY_FOCUS_WEIGHT="${V20_GROUP_FAMILY_FOCUS_WEIGHT:-0.40}"

_args=(
  --v20-output-dir "$V20_OUTPUT_DIR"
  --v20-train-npz "$V20_TRAIN_NPZ"
  --v20-val-npz "$V20_VAL_NPZ"
  --v20-test-npz "$V20_TEST_NPZ"
  --v20-gpu "$V20_GPU"
  --v20-epochs "$V20_EPOCHS"
  --v20-lr "$V20_LR"
  --v20-weight-decay "$V20_WEIGHT_DECAY"
  --v20-d-model "$V20_D_MODEL"
  --v20-n-heads "$V20_N_HEADS"
  --v20-n-layers "$V20_N_LAYERS"
  --v20-ff-dim "$V20_FF_DIM"
  --v20-dropout "$V20_DROPOUT"
  --v20-num-workers "$V20_NUM_WORKERS"
  --v20-backbone-weight "$V20_BACKBONE_WEIGHT"
  --v20-strict-presence-weight "$V20_STRICT_PRESENCE_WEIGHT"
  --v20-group-family-focus-weight "$V20_GROUP_FAMILY_FOCUS_WEIGHT"
  --v20-batch-size "$V20_BATCH_SIZE"
)

if [[ "$V20_NO_PACKED" == "1" ]]; then
  _args+=(--v20-no-packed)
fi

python "$SCRIPT_DIR/v20_clean_train.py" "${_args[@]}"
