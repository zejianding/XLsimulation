"""
磁盘分桶版谱图平均 —— 防止大数据集 OOM。

用法：
    python AverageSpectrum_diskbased.py <input.mgf> <output.mgf> --tol 0.05 --buckets 256
    python AverageSpectrum_diskbased.py <input.mgf> <output.mgf> --cleanup-input   # 完成后删输入

断点续跑：临时目录固定，Pass1/Pass2 已完成的部分自动跳过。

输入保留策略：默认**不删除**输入 MGF，只有显式传 `--cleanup-input` 才会在散列完成后删除。
（脚本自己产生的桶文件与临时目录不受该开关影响，仍在流程内清理。）
"""

import argparse
import hashlib
import os
import shutil
import sys
import time

from AverageSpectrum_batch import (
    iter_mgf_blocks,
    parse_mgf_block,
    get_group_key,
    build_average_mgf,
)


def _bucket_for_key(key, num_buckets):
    h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(h, 16) % num_buckets


def build_average_mgf_diskbased(input_mgf, output_mgf, tol=0.05,
                                 group_by_charge=True, num_buckets=256,
                                 tmp_dir=None, cleanup_input=False):
    print(f"\nDisk-based averaging: {input_mgf} -> {output_mgf}")
    print(f"  tol={tol}, buckets={num_buckets}")

    # 固定临时目录（基于输出文件名），支持断点续跑
    if tmp_dir is None:
        tmp_dir = output_mgf + ".buckets_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    bucket_prefix = os.path.join(tmp_dir, "bucket")
    t0 = time.time()

    # ---- Pass 1: Scatter ----
    # 检查是否已有桶文件（从上次失败中恢复）
    existing_buckets = [f for f in os.listdir(tmp_dir)
                        if f.startswith("bucket_")
                        and f.endswith(".mgf")
                        and not f.endswith("_averaged.mgf")]
    if existing_buckets and os.path.exists(input_mgf):
        print("[1/3] Found existing bucket files, skipping scatter...")
        bucket_fps = {}
        for fname in existing_buckets:
            bi = int(fname.replace("bucket_", "").replace(".mgf", ""))
            bucket_fps[bi] = None  # 文件已关闭，只记录编号
        total = len(existing_buckets)
        skipped = 0
    elif not os.path.exists(input_mgf):
        if existing_buckets:
            print("[1/3] Input deleted, using existing bucket files...")
            bucket_fps = {}
            for fname in existing_buckets:
                bi = int(fname.replace("bucket_", "").replace(".mgf", ""))
                bucket_fps[bi] = None
            total = len(existing_buckets)
            skipped = 0
        else:
            raise FileNotFoundError(
                f"Input not found: {input_mgf}, and no bucket files in {tmp_dir}"
            )
    else:
        print("[1/3] Scattering spectra into buckets...")
        bucket_fps = {}
        total = 0
        skipped = 0

        for block in iter_mgf_blocks(input_mgf):
            total += 1
            item = parse_mgf_block(block)
            if item["seq"] is None:
                skipped += 1
                continue
            if group_by_charge and item["charge"] is None:
                skipped += 1
                continue

            key = get_group_key(item, group_by_charge=group_by_charge)
            bi = _bucket_for_key(key, num_buckets)

            if bi not in bucket_fps:
                path = f"{bucket_prefix}_{bi:04d}.mgf"
                bucket_fps[bi] = open(path, "w", encoding="utf-8")

            bucket_fps[bi].writelines(block)
            bucket_fps[bi].write("\n")

            if total % 50000 == 0:
                print(f"  Scattered {total} spectra...")

        for fp in bucket_fps.values():
            if fp is not None:
                fp.close()

        print(f"  Done: {total} spectra, {skipped} skipped, "
              f"{len(bucket_fps)} active buckets, "
              f"{time.time() - t0:.0f}s")

        # 散列完成后可选删除输入文件，释放磁盘空间（默认保留，需显式开启）
        if cleanup_input and os.path.exists(input_mgf):
            os.remove(input_mgf)
            print(f"  Removed input: {input_mgf}")

    # ---- Pass 2: Average each bucket ----
    t1 = time.time()
    print("[2/3] Averaging each bucket...")

    bucket_outputs = []
    for bi in sorted(bucket_fps.keys()):
        bucket_path = f"{bucket_prefix}_{bi:04d}.mgf"
        bucket_avg_path = f"{bucket_prefix}_{bi:04d}_averaged.mgf"

        # 跳过已完成的桶
        if os.path.exists(bucket_avg_path):
            print(f"  Bucket {bi:04d} — already averaged, skip")
            bucket_outputs.append(bucket_avg_path)
            if os.path.exists(bucket_path):
                os.remove(bucket_path)
            continue

        if not os.path.exists(bucket_path):
            print(f"  Bucket {bi:04d} — raw file missing, skip")
            continue

        sz_mb = os.path.getsize(bucket_path) / (1024 ** 2)
        print(f"  Bucket {bi:04d} ({sz_mb:.0f} MB)...")

        build_average_mgf(
            input_mgf=bucket_path,
            output_mgf=bucket_avg_path,
            tol=tol,
            group_by_charge=group_by_charge,
        )
        bucket_outputs.append(bucket_avg_path)

        # 桶平均完成后立即删除原始桶文件，释放空间
        if os.path.exists(bucket_path):
            os.remove(bucket_path)

    print(f"  Done: {len(bucket_outputs)} buckets, "
          f"{time.time() - t1:.0f}s")

    # ---- Pass 3: Concatenate ----
    t2 = time.time()
    print("[3/3] Concatenating bucket outputs...")

    lines = 0
    with open(output_mgf, "w", encoding="utf-8") as fout:
        for bf in bucket_outputs:
            if not os.path.exists(bf):
                continue
            with open(bf, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    lines += 1

    print(f"  Done: {lines} lines, {time.time() - t2:.0f}s")

    # ---- Cleanup ----
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"  Removed temp dir: {tmp_dir}")

    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    print(f"Output: {output_mgf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mgf")
    parser.add_argument("output_mgf")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--buckets", type=int, default=256)
    parser.add_argument("--cleanup-input", action="store_true",
                        help="散列完成后删除输入 MGF（默认保留）")
    args = parser.parse_args()

    build_average_mgf_diskbased(
        input_mgf=args.input_mgf,
        output_mgf=args.output_mgf,
        tol=args.tol,
        num_buckets=args.buckets,
        cleanup_input=args.cleanup_input,
    )
