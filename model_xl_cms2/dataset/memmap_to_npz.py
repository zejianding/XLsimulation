"""
将合并后的 memmap .npy 文件打包成单个 .npz 文件。
memmap 数组不会全部载入内存，np.savez_compressed 逐块压缩。

用法：
    python memmap_to_npz.py <merged_dir> <output.npz>
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm


def memmap_to_npz(merged_dir, output_npz):
    meta_path = os.path.join(merged_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {merged_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Packing {len(meta['merge_keys']) + len(meta.get('obj_merge_keys', []))} arrays → {output_npz}")

    t0 = time.time()
    arrays = {}

    # 加载 memmap 大数组
    for key in meta["merge_keys"]:
        path = os.path.join(merged_dir, f"{key}.npy")
        gb = os.path.getsize(path) / (1024**3)
        print(f"  Loading {key} ({gb:.1f} GB) as memmap...")
        arrays[key] = np.load(path, mmap_mode='r')

    # 加载 object 数组
    for key in meta.get("obj_merge_keys", []):
        path = os.path.join(merged_dir, f"{key}.npy")
        arrays[key] = np.load(path, allow_pickle=True)

    # 加载标量
    for key in meta["scalar_keys"]:
        path = os.path.join(merged_dir, f"_scalar_{key}.npy")
        if os.path.exists(path):
            arrays[key] = np.load(path, allow_pickle=True)

    # 标准化 key 名（转换回原始格式）
    key_map = {"inputs": "input_mz_matrices", "targets": "target_intensity_matrices"}
    for new_key, old_key in key_map.items():
        if new_key in arrays and old_key not in arrays:
            arrays[old_key] = arrays.pop(new_key)

    print(f"  Saving NPZ (memmap → 逐块压缩，内存峰值 ~2GB)...")
    np.savez_compressed(output_npz, **arrays)
    gb = os.path.getsize(output_npz) / (1024**3)
    print(f"  Done in {(time.time() - t0) / 60:.1f} min, {gb:.1f} GB")
    print(f"  如需分片训练: --train-npz merged_dir/")
    print(f"  如需单文件训练: --train-npz {output_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("merged_dir", help="merge_shards_to_memmap 的输出目录")
    parser.add_argument("output_npz", help="输出 .npz 文件路径")
    args = parser.parse_args()
    memmap_to_npz(args.merged_dir, args.output_npz)
