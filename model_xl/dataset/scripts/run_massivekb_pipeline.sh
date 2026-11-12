#!/bin/bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
set -euo pipefail

# MassiveKB one-shot pipeline for v20_0701
#
# Usage:
#   cd /home/wangyuhang/seq2mass/v20_0701
#   bash dataset/scripts/run_massivekb_pipeline.sh
#
# Optional env vars:
#   DATA_DIR=/home/wangyuhang/datasets/massivekb
#   PREFIX=massivekb_82c0124b
#   INPUT_TRAIN=/custom/path/massivekb_82c0124b_train.mgf
#   INPUT_VAL=/custom/path/massivekb_82c0124b_val.mgf
#   INPUT_TEST=/custom/path/massivekb_82c0124b_test.mgf
#   NUM_WORKERS=8
#   CHUNK_SIZE=5000
#   CLEANUP=0            (默认 0：不删除任何中间/输入文件)
#   SEQ_CONSTRAINED=qk   (默认 qk：只对 Q/K 列做序列约束；off=关闭 full=所有残基列)
#   AUTO_DECOMPRESS=1
#
# Cleanup policy:
#   CLEANUP=0 (默认)  保留全部中间文件：解压后的 .mgf、*_filtered.mgf、*_averaged.mgf
#   CLEANUP=1         删除解压后的 .mgf、*_filtered.mgf、*_averaged.mgf
#                     并把 --cleanup-input 透传给 AverageSpectrum_diskbased.py
#
# Output policy:
#   train -> shard npz + merged memmap
#   val   -> single npz only
#   test  -> single npz only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-/home/wangyuhang/datasets/massivekb}"
PREFIX="${PREFIX:-massivekb_82c0124b}"

TOL_FILTER="${TOL_FILTER:-0.05}"
TOL_AVERAGE="${TOL_AVERAGE:-0.05}"
TOL_NPZ="${TOL_NPZ:-0.05}"
# 序列约束残基偏移列：off | qk（推荐）| full
#   qk   只对 Q/K 四列做约束，消除 |Q-K|=0.036 < tol 造成的互相污染（标签 -6.7%）
#   full 对所有残基列都要求 m/z 与真实 b/y 离子重合（标签 -55.5%，过于激进）
SEQ_CONSTRAINED="${SEQ_CONSTRAINED:-qk}"
CLIP_INTENSITY="${CLIP_INTENSITY:-50000.0}"
NUM_BUCKETS="${NUM_BUCKETS:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-5000}"
CLEANUP="${CLEANUP:-0}"
AUTO_DECOMPRESS="${AUTO_DECOMPRESS:-1}"

cd "${PROJECT_DIR}"

echo "=============================================="
echo "V20_0701 MassiveKB Pipeline"
echo "=============================================="
echo "PROJECT_DIR  : ${PROJECT_DIR}"
echo "DATASET_DIR  : ${DATASET_DIR}"
echo "DATA_DIR     : ${DATA_DIR}"
echo "PREFIX       : ${PREFIX}"
echo "NUM_WORKERS  : ${NUM_WORKERS}"
echo "CHUNK_SIZE   : ${CHUNK_SIZE}"
echo "CLEANUP      : ${CLEANUP}"
echo "SEQ_CONSTRAINED : ${SEQ_CONSTRAINED}"
echo "AUTO_DECOMPRESS : ${AUTO_DECOMPRESS}"
echo ""

decompress_xz_if_needed() {
    local input_xz="$1"
    local output_mgf="$2"

    if [ ! -f "${input_xz}" ]; then
        return 1
    fi
    if [ -f "${output_mgf}" ]; then
        return 0
    fi

    echo "Decompressing: ${input_xz} -> ${output_mgf}"
    python - "${input_xz}" "${output_mgf}" <<'PY'
import lzma
import shutil
import sys

src = sys.argv[1]
dst = sys.argv[2]
with lzma.open(src, "rb") as fin, open(dst, "wb") as fout:
    shutil.copyfileobj(fin, fout)
PY
}

resolve_input_paths() {
    local split="$1"
    local explicit_var=""
    case "$split" in
        train) explicit_var="${INPUT_TRAIN:-}" ;;
        val) explicit_var="${INPUT_VAL:-}" ;;
        test) explicit_var="${INPUT_TEST:-}" ;;
    esac

    if [ -n "${explicit_var}" ]; then
        echo "${explicit_var}|${explicit_var}.xz"
        return 0
    fi

    local flat_plain="${DATA_DIR}/${PREFIX}_${split}.mgf"
    local flat_xz="${DATA_DIR}/${PREFIX}_${split}.mgf.xz"
    local nested_plain="${DATA_DIR}/${split}/orig/${PREFIX}_${split}.mgf"
    local nested_xz="${DATA_DIR}/${split}/orig/${PREFIX}_${split}.mgf.xz"

    if [ -f "${nested_plain}" ] || [ -f "${nested_xz}" ]; then
        echo "${nested_plain}|${nested_xz}"
        return 0
    fi

    echo "${flat_plain}|${flat_xz}"
}

for SPLIT in val test train; do
    echo ""
    echo "=============================================="
    echo "Processing: ${SPLIT}"
    echo "=============================================="

    RESOLVED_INPUTS="$(resolve_input_paths "${SPLIT}")"
    INPUT_PLAIN="${RESOLVED_INPUTS%%|*}"
    INPUT_XZ="${RESOLVED_INPUTS#*|}"
    FILTERED="${DATA_DIR}/${PREFIX}_${SPLIT}_filtered.mgf"
    AVERAGED="${DATA_DIR}/${PREFIX}_${SPLIT}_averaged.mgf"
    OUTPUT_DIR="${DATA_DIR}/${PREFIX}_${SPLIT}_npz"
    MERGED_DIR="${DATA_DIR}/${PREFIX}_${SPLIT}_merged"
    OUTPUT_SINGLE="${DATA_DIR}/${PREFIX}_${SPLIT}.npz"

    echo "Input plain : ${INPUT_PLAIN}"
    echo "Input xz    : ${INPUT_XZ}"

    if [ "${SPLIT}" != "train" ] && [ -f "${OUTPUT_SINGLE}" ]; then
        echo "Single NPZ already exists, skip ${SPLIT}"
        continue
    fi

    if [ "${SPLIT}" = "train" ] && [ -f "${OUTPUT_DIR}/index.json" ]; then
        n=$(python -c "import json; print(json.load(open('${OUTPUT_DIR}/index.json'))['total_examples'])")
        echo "Shard NPZ already exists (${n} examples)"
        if [ ! -f "${MERGED_DIR}/.merge_done" ]; then
            echo "[merge] shards -> memmap"
            python "${DATASET_DIR}/merge_shards_to_memmap.py" "${OUTPUT_DIR}" "${MERGED_DIR}"
        else
            echo "Merged memmap already exists"
        fi
        continue
    fi

    if [ -f "${FILTERED}" ] || [ -f "${AVERAGED}" ]; then
        echo "[1/4] token filter already done"
    elif [ -f "${INPUT_PLAIN}" ]; then
        echo "[1/4] token filtering..."
        python "${DATASET_DIR}/TokenFilter_parallel.py" \
            "${INPUT_PLAIN}" "${FILTERED}" \
            --tol "${TOL_FILTER}" \
            --workers "${NUM_WORKERS}"

        if [ "${CLEANUP}" -eq 1 ] && [ -f "${INPUT_XZ}" ]; then
            rm -f "${INPUT_PLAIN}"
            echo "Removed: ${INPUT_PLAIN}"
        fi
    elif [ -f "${INPUT_XZ}" ]; then
        if [ "${AUTO_DECOMPRESS}" = "1" ]; then
            echo "[1/4] found compressed input, auto-decompressing..."
            decompress_xz_if_needed "${INPUT_XZ}" "${INPUT_PLAIN}"
            echo "[1/4] token filtering..."
            python "${DATASET_DIR}/TokenFilter_parallel.py" \
                "${INPUT_PLAIN}" "${FILTERED}" \
                --tol "${TOL_FILTER}" \
                --workers "${NUM_WORKERS}"

            if [ "${CLEANUP}" -eq 1 ]; then
                rm -f "${INPUT_PLAIN}"
                echo "Removed: ${INPUT_PLAIN}"
            fi
        else
            echo "[1/4] missing plain .mgf, decompress first:"
            echo "       xz -d ${INPUT_XZ}"
            continue
        fi
    else
        echo "[1/4] input file missing: ${INPUT_PLAIN}"
        continue
    fi

    if [ -f "${AVERAGED}" ]; then
        echo "[2/4] averaging already done"
    else
        echo "[2/4] averaging spectra..."
        _avg_args=(--tol "${TOL_AVERAGE}" --buckets "${NUM_BUCKETS}")
        if [ "${CLEANUP}" -eq 1 ]; then
            _avg_args+=(--cleanup-input)
        fi
        python "${DATASET_DIR}/AverageSpectrum_diskbased.py" \
            "${FILTERED}" "${AVERAGED}" \
            "${_avg_args[@]}"

        if [ "${CLEANUP}" -eq 1 ]; then
            rm -f "${FILTERED}"
            echo "Removed: ${FILTERED}"
        fi
    fi

    if [ -f "${AVERAGED}" ]; then
        echo "[3/4] averaged mgf -> shard npz..."
        _npz_args=(--tol "${TOL_NPZ}" --chunk-size "${CHUNK_SIZE}"
                   --clip-intensity "${CLIP_INTENSITY}" --workers "${NUM_WORKERS}")
        if [ "${SEQ_CONSTRAINED}" != "off" ]; then
            _npz_args+=(--sequence-constrained "${SEQ_CONSTRAINED}")
        fi
        python "${DATASET_DIR}/NPZBuilder_chunked.py" \
            "${AVERAGED}" "${OUTPUT_DIR}" \
            "${_npz_args[@]}"

        if [ "${SPLIT}" != "train" ]; then
            echo "[3.5/4] shard npz -> single npz..."
            python "${DATASET_DIR}/shards_to_npz.py" \
                "${OUTPUT_DIR}" "${OUTPUT_SINGLE}"
        fi

        if [ "${CLEANUP}" -eq 1 ]; then
            rm -f "${AVERAGED}"
            echo "Removed: ${AVERAGED}"
        fi
    else
        echo "[3/4] missing averaged mgf"
        continue
    fi

    if [ "${SPLIT}" != "train" ]; then
        echo "[4/4] ${SPLIT} finished with single npz: ${OUTPUT_SINGLE}"
    elif [ -f "${MERGED_DIR}/.merge_done" ]; then
        echo "[4/4] memmap already done"
    else
        echo "[4/4] shards -> memmap..."
        python "${DATASET_DIR}/merge_shards_to_memmap.py" "${OUTPUT_DIR}" "${MERGED_DIR}"
    fi

    if [ "${SPLIT}" = "train" ]; then
        echo "done: ${MERGED_DIR}"
    else
        echo "done: ${OUTPUT_SINGLE}"
    fi
done

echo ""
echo "=============================================="
echo "Pipeline finished."
echo "Recommended train command:"
echo "python /home/wangyuhang/seq2mass/v20_0701/transfomer/v20_clean_train.py \\"
echo "  --v20-output-dir /home/wangyuhang/seq2mass/runs/v20_0701_train \\"
echo "  --v20-train-npz ${DATA_DIR}/${PREFIX}_train_merged \\"
echo "  --v20-val-npz ${DATA_DIR}/${PREFIX}_val.npz \\"
echo "  --v20-test-npz ${DATA_DIR}/${PREFIX}_test.npz \\"
echo "  --v20-gpu 0"
echo "=============================================="
