"""
将分片 NPZ 目录合并为内存映射 .npy 文件，供训练直接 memmap 加载。

用法：
    python merge_shards_to_memmap.py <shard_dir> <output_dir>
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm


def _write_npy_header(fp, shape, dtype):
    """手写 .npy v1.0 header，兼容所有 numpy 版本"""
    import struct
    fp.write(b'\x93NUMPY')
    fp.write(b'\x01\x00')  # version 1.0
    header_str = "{'descr': %s, 'fortran_order': False, 'shape': %s, }" % (
        repr(dtype.descr if dtype.names else dtype.str), repr(shape))
    header_bytes = header_str.encode('ascii')
    # 16 字节对齐
    total_header_len = len(header_bytes) + 4  # 4 = header_len field itself
    padding = (16 - total_header_len % 16) % 16
    header_bytes += b' ' * padding + b'\n'
    fp.write(struct.pack('<H', len(header_bytes)))
    fp.write(header_bytes)


def _read_npy_data_offset(fp):
    """读取 .npy header 后返回数据起始位置"""
    import struct
    if fp.read(6) != b'\x93NUMPY':
        raise ValueError("Not a .npy file")
    version = fp.read(2)
    if version == b'\x01\x00':
        header_len = struct.unpack('<H', fp.read(2))[0]
    elif version == b'\x02\x00':
        header_len = struct.unpack('<I', fp.read(4))[0]
    else:
        raise ValueError(f"Unsupported .npy version: {version!r}")
    fp.read(header_len)  # skip header
    return fp.tell()


def merge_shards_to_memmap(shard_dir, output_dir):
    with open(os.path.join(shard_dir, "index.json")) as f:
        index = json.load(f)

    total = index["total_examples"]
    shard_files = index["shard_files"]
    print(f"Merging {len(shard_files)} shards → {total:,} examples → {output_dir}/")

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    done_marker = os.path.join(output_dir, ".merge_done")
    if os.path.exists(done_marker):
        print("  Merge already done, loading memmaps...")
        result = _load_memmaps(output_dir)
        print(f"  Loaded {len(result)} arrays in {time.time() - t0:.0f}s")
        return result

    shard0 = dict(np.load(os.path.join(shard_dir, shard_files[0]), allow_pickle=True))

    # 分类：优先用 index.json 里 NPZBuilder 写的权威列表；旧数据没有该字段时
    # 退回形状启发式（shape[0] > 100）—— 后者在分片行数 <= 100 时会误判。
    per_example = index.get("per_example_keys")
    big_keys = []       # 每样本一行 → memmap
    obj_big_keys = []   # object 数组（seqs）→ 普通 .npy
    const_data = {}
    for key, val in shard0.items():
        if per_example is not None:
            if key not in per_example:
                const_data[key] = val
            elif isinstance(val, np.ndarray) and val.dtype == object:
                obj_big_keys.append(key)
            else:
                big_keys.append(key)
        elif val.ndim >= 1 and val.shape[0] > 100:
            if isinstance(val, np.ndarray) and val.dtype == object:
                obj_big_keys.append(key)
            else:
                big_keys.append(key)
        else:
            const_data[key] = val

    # 保存常量
    for key, val in const_data.items():
        if isinstance(val, np.ndarray):
            np.save(os.path.join(output_dir, f"_scalar_{key}.npy"), val)

    # 按大小排：大的先处理（一次一个，防内存压力）
    def _array_total_bytes(key):
        s = list(shard0[key].shape)
        s[0] = total
        return int(np.prod(s)) * shard0[key].dtype.itemsize

    big_keys.sort(key=_array_total_bytes, reverse=True)
    all_big_keys = big_keys + obj_big_keys

    # 逐个数组处理：创建 memmap → 逐片填充 → 刷盘
    FLUSH_EVERY = 10  # 每 N 片刷新一次，防 OS 页缓存堆积

    for key in all_big_keys:
        shape = list(shard0[key].shape)
        shape[0] = total
        dtype = shard0[key].dtype
        is_obj = (key in obj_big_keys)
        path = os.path.join(output_dir, f"{key}.npy")

        gb = np.prod(shape) * dtype.itemsize / (1024**3)
        print(f"  Merging {key} ({gb:.1f} GB)...")

        if is_obj:
            arr = np.empty(tuple(shape), dtype=dtype)
        else:
            with open(path, 'wb') as f:
                _write_npy_header(f, tuple(shape), dtype)
            with open(path, 'rb') as f:
                data_offset = _read_npy_data_offset(f)

            arr = np.memmap(path, dtype=dtype, mode='r+',
                            offset=data_offset, shape=tuple(shape))

        offset = 0
        for i, sf in enumerate(tqdm(shard_files, desc=f"    {key}", leave=False)):
            with np.load(os.path.join(shard_dir, sf), allow_pickle=True) as d:
                n = min(d[key].shape[0], total - offset)
                if n <= 0:
                    break
                arr[offset:offset + n] = d[key][:n]
                offset += n

            if not is_obj and (i + 1) % FLUSH_EVERY == 0:
                arr.flush()

        if is_obj:
            np.save(path, arr)
            del arr
        else:
            arr.flush()
            del arr

    # 元数据（把 index.json 里的数据定义字段一并带过来，便于追溯）
    meta = {
        "total_examples": total,
        "merge_keys": big_keys,
        "obj_merge_keys": obj_big_keys,
        "scalar_keys": list(const_data.keys()),
    }
    for k in ("tol", "sequence_constrained", "target_normalization", "clip_intensity",
              "per_example_keys", "const_keys"):
        if k in index:
            meta[k] = index[k]
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(done_marker, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    elapsed = time.time() - t0
    total_gb = sum(os.path.getsize(os.path.join(output_dir, f"{k}.npy"))
                   for k in all_big_keys) / (1024**3)
    print(f"  Done in {elapsed / 60:.1f} min, {total_gb:.1f} GB on disk")


def _load_memmaps(output_dir):
    with open(os.path.join(output_dir, "meta.json")) as f:
        meta = json.load(f)
    result = {}
    for key in meta["merge_keys"]:
        path = os.path.join(output_dir, f"{key}.npy")
        result[key] = np.load(path, mmap_mode='r')
    for key in meta.get("obj_merge_keys", []):
        path = os.path.join(output_dir, f"{key}.npy")
        result[key] = np.load(path, allow_pickle=True)
    for key in meta["scalar_keys"]:
        path = os.path.join(output_dir, f"_scalar_{key}.npy")
        if os.path.exists(path):
            result[key] = np.load(path, allow_pickle=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    merge_shards_to_memmap(args.shard_dir, args.output_dir)
