"""
分片 NPZ → 单个 NPZ。一次遍历分片，同时写所有数组，零中间文件。

用法：
    python shards_to_npz.py <shard_dir> <output.npz>
"""

import argparse
import json
import os
import time
import zipfile

import numpy as np
import numpy.lib.format as _npy_fmt
from tqdm import tqdm


def shards_to_npz(shard_dir, output_npz):
    with open(os.path.join(shard_dir, "index.json")) as f:
        index = json.load(f)

    total = index["total_examples"]
    shard_files = index["shard_files"]
    print(f"{len(shard_files)} shards → {output_npz} ({total:,} examples)")

    t0 = time.time()
    shard0 = dict(np.load(os.path.join(shard_dir, shard_files[0]), allow_pickle=True))

    # 分类：优先用 index.json 里 NPZBuilder 写的权威列表；旧数据没有该字段时
    # 退回形状启发式（shape[0] > 100）—— 后者在分片行数 <= 100 时会误判。
    per_example = index.get("per_example_keys")
    big_keys = []
    obj_big_keys = []
    small = {}
    for key, val in shard0.items():
        if per_example is not None:
            if key not in per_example:
                small[key] = val
            elif val.dtype == object:
                obj_big_keys.append(key)
            else:
                big_keys.append(key)
        elif val.ndim >= 1 and val.shape[0] > 100:
            if val.dtype == object:
                obj_big_keys.append(key)
            else:
                big_keys.append(key)
        else:
            small[key] = val

    with zipfile.ZipFile(output_npz, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 常量
        for key, val in small.items():
            if isinstance(val, np.ndarray):
                with zf.open(f"{key}.npy", 'w') as entry:
                    _npy_fmt.write_array(entry, val)

        # object 大数组：整体拼接后 pickle 写入（原始 object bytes 只是指针）
        for key in obj_big_keys:
            chunks = []
            written = 0
            for sf in tqdm(shard_files, desc=f"  {key}", leave=False):
                with np.load(os.path.join(shard_dir, sf), allow_pickle=True) as d:
                    arr = d[key]
                    n = min(arr.shape[0], total - written)
                    if n <= 0:
                        break
                    chunks.append(arr[:n])
                    written += n
            merged = (np.concatenate(chunks) if chunks
                      else np.empty((0,), dtype=shard0[key].dtype))
            with zf.open(f"{key}.npy", 'w', force_zip64=(merged.nbytes > (1 << 30))) as entry:
                _npy_fmt.write_array(entry, merged)

        # 数值大数组：逐个 key 跨所有分片流式写入。Python >= 3.11 的 zipfile
        # 同一时刻只允许一个写句柄，因此一次只开一个条目（每次只驻留一个数组，
        # 内存占用与原来相当）。
        for key in big_keys:
            shape = tuple([total] + list(shard0[key].shape[1:]))
            dtype = shard0[key].dtype
            # 注意用 int64：Windows 上 np.prod 默认 int32，>2 GiB 会溢出成负数
            nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            print(f"  {key}: {nbytes / (1024**3):.1f} GB  （zip64={nbytes > (1 << 31)})")

            header = _npy_fmt.header_data_from_array_1_0(np.empty(0, dtype=dtype))
            header['shape'] = shape

            written = 0
            # 条目 >2 GiB 必须开 zip64，否则 zipfile 关闭时抛 "File size too large"
            with zf.open(f"{key}.npy", 'w', force_zip64=True) as entry:
                _npy_fmt.write_array_header_2_0(entry, header)
                for sf in tqdm(shard_files, desc=f"    {key}", leave=False):
                    with np.load(os.path.join(shard_dir, sf), allow_pickle=True) as d:
                        arr = d[key]
                        n = min(arr.shape[0], total - written)
                        if n <= 0:
                            break
                        entry.write(np.ascontiguousarray(arr[:n]).tobytes())
                        written += n

    gb = os.path.getsize(output_npz) / (1024**3)
    print(f"Done: {gb:.1f} GB in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_dir")
    parser.add_argument("output_npz")
    args = parser.parse_args()
    shards_to_npz(args.shard_dir, args.output_npz)
