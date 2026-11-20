"""
分片 NPZ 生成（多进程）—— 防止大数据集 OOM。

用法：
    python NPZBuilder_chunked.py <input.mgf> <output_dir> --tol 0.05 --chunk-size 5000 --workers 8

输出：
    {output_dir}/shard_00000.npz
    {output_dir}/shard_00001.npz
    ...
    {output_dir}/index.json
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from trainingExamples import (
    build_input_mz_matrix,
    build_target_intensity_matrix,
    normalize_target_intensity,
    build_position_mask,
    build_channel_mask,
    parse_sequence_with_mods,
    parse_ions_block,
    iter_mgf_blocks,
    MAX_PEPTIDE_LEN,
    MAX_CLEAVAGE_LEN,
    NUM_TOKENS,
    NUM_CHANNELS,
    CHANNEL_NAMES,
    CHARGE_TOKENS,
)

# 权威的键分类：写入 index.json 供下游（shards_to_npz / merge_shards_to_memmap /
# load_sharded_npz）使用，避免它们靠 shape[0] > 100 的形状启发式猜测——那个启发式在
# 分片行数 <= 100（如 --chunk-size 50，或极小数据集）时会把大数组误判成常量并静默丢数据。
PER_EXAMPLE_KEYS = (
    "input_mz_matrices", "target_intensity_matrices", "position_masks", "channel_masks",
    "lengths", "charges", "pepmasses", "rtinseconds", "seqs", "target_scale_values",
)
CONST_KEYS = (
    "channel_names", "tokens_charge1", "tokens_charge2", "tokens_charge3",
    "max_peptide_len", "max_cleavage_len", "target_normalization", "clip_intensity",
    "mz_unit",
)

# 全局参数（Pool 初始化时设置）
_worker_tol = 0.05
_worker_target_normalization = "sqrt_clip"
_worker_clip_intensity = 50000.0
_worker_sequence_constrained = None


def _init_worker(tol, target_normalization, clip_intensity, sequence_constrained=None):
    global _worker_tol, _worker_target_normalization, _worker_clip_intensity
    global _worker_sequence_constrained
    _worker_tol = tol
    _worker_target_normalization = target_normalization
    _worker_clip_intensity = clip_intensity
    _worker_sequence_constrained = sequence_constrained


def _process_one_block(block):
    """处理单个 spectrum block → 返回 (result_dict, skipped_reason) 或 (None, reason)"""
    seq, charge, pepmass, rtinseconds, peaks = parse_ions_block(block)

    if seq is None or charge is None or len(peaks) == 0:
        return None, "invalid"

    try:
        residues, _ = parse_sequence_with_mods(seq)
        if len(residues) > MAX_PEPTIDE_LEN:
            return None, "long"

        input_mz = build_input_mz_matrix(seq)
        target_mat = build_target_intensity_matrix(
            seq=seq, precursor_charge=charge, peaks=peaks, tol=_worker_tol,
            sequence_constrained=_worker_sequence_constrained,
        )
        target_mat, scale = normalize_target_intensity(
            target_mat, method=_worker_target_normalization,
            clip_value=_worker_clip_intensity,
        )
        pos_mask, length = build_position_mask(seq)
        chan_mask = build_channel_mask(charge)
    except Exception:
        return None, "error"

    return {
        "input_mz": input_mz,
        "target": target_mat,
        "pos_mask": pos_mask,
        "chan_mask": chan_mask,
        "length": length,
        "charge": charge,
        "pepmass": pepmass,
        "rt": rtinseconds,
        "seq": seq,
        "scale": scale,
    }, "ok"


def build_npz_chunked(input_mgf, output_dir, tol=0.05,
                       target_normalization="sqrt_clip",
                       clip_intensity=50000.0, chunk_size=5000,
                       num_workers=None, sequence_constrained=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"\nChunked NPZ: {input_mgf} -> {output_dir}/")
    print(f"  tol={tol}, chunk_size={chunk_size}, workers={num_workers}, "
          f"sequence_constrained={sequence_constrained}")

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "index.json")

    # 断点续跑
    existing = sorted([f for f in os.listdir(output_dir)
                       if f.startswith("shard_") and f.endswith(".npz")])
    if existing:
        last = existing[-1]
        start_idx = int(last.replace("shard_", "").replace(".npz", "")) + 1
        print(f"  Resuming from shard {start_idx} ({len(existing)} existing)")
    else:
        start_idx = 0

    t0 = time.time()
    total = 0
    used = 0
    shard_idx = start_idx
    skipped_invalid = 0
    skipped_long = 0
    skipped_error = 0

    # 分片累加器
    acc = [[] for _ in range(10)]

    def flush():
        nonlocal shard_idx
        if not acc[0]:
            return
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npz")
        np.savez_compressed(
            shard_path,
            input_mz_matrices=np.stack(acc[0], axis=0).astype(np.float32),
            target_intensity_matrices=np.stack(acc[1], axis=0).astype(np.float32),
            position_masks=np.stack(acc[2], axis=0).astype(np.float32),
            channel_masks=np.stack(acc[3], axis=0).astype(np.float32),
            lengths=np.array(acc[4], dtype=np.int32),
            charges=np.array(acc[5], dtype=np.int32),
            pepmasses=np.array(acc[6], dtype=np.float32),
            rtinseconds=np.array(acc[7], dtype=np.float32),
            seqs=np.array(acc[8], dtype=object),
            target_scale_values=np.array(acc[9], dtype=np.float32),
            channel_names=np.array(CHANNEL_NAMES, dtype=object),
            tokens_charge1=np.array(CHARGE_TOKENS[1], dtype=np.float32),
            tokens_charge2=np.array(CHARGE_TOKENS[2], dtype=np.float32),
            tokens_charge3=np.array(CHARGE_TOKENS[3], dtype=np.float32),
            max_peptide_len=np.array(MAX_PEPTIDE_LEN, dtype=np.int32),
            max_cleavage_len=np.array(MAX_CLEAVAGE_LEN, dtype=np.int32),
            target_normalization=np.array(target_normalization, dtype=object),
            clip_intensity=np.array(clip_intensity, dtype=np.float32),
            mz_unit=np.array("Da", dtype=object),
        )
        sz_mb = os.path.getsize(shard_path) / (1024 ** 2)
        print(f"  Wrote {shard_path} ({len(acc[0])} examples, {sz_mb:.0f} MB)")
        shard_idx += 1
        for lst in acc:
            lst.clear()

    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(tol, target_normalization, clip_intensity,
                        sequence_constrained)) as pool:
        for result, status in pool.imap_unordered(
            _process_one_block, iter_mgf_blocks(input_mgf), chunksize=20
        ):
            total += 1
            if result is None:
                if status == "invalid":
                    skipped_invalid += 1
                elif status == "long":
                    skipped_long += 1
                else:
                    skipped_error += 1
                continue

            acc[0].append(result["input_mz"])
            acc[1].append(result["target"])
            acc[2].append(result["pos_mask"])
            acc[3].append(result["chan_mask"])
            acc[4].append(result["length"])
            acc[5].append(result["charge"])
            acc[6].append(result["pepmass"])
            acc[7].append(result["rt"])
            acc[8].append(result["seq"])
            acc[9].append(result["scale"])
            used += 1

            if len(acc[0]) >= chunk_size:
                flush()

            if used % 50000 == 0:
                elapsed = time.time() - t0
                print(f"  Converted {used} examples ({used / elapsed:.0f} examples/s)...")

    flush()

    if used == 0:
        raise RuntimeError("No valid training examples built.")

    shard_files = sorted([f for f in os.listdir(output_dir)
                          if f.startswith("shard_") and f.endswith(".npz")])
    index = {
        "total_examples": used,
        "num_shards": len(shard_files),
        "shard_files": shard_files,
        "target_normalization": target_normalization,
        "clip_intensity": clip_intensity,
        "tol": tol,
        "sequence_constrained": sequence_constrained,
        "per_example_keys": list(PER_EXAMPLE_KEYS),
        "const_keys": list(CONST_KEYS),
        "max_peptide_len": MAX_PEPTIDE_LEN,
        "channel_names": CHANNEL_NAMES,
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone: {used} examples in {len(shard_files)} shards, "
          f"{elapsed / 60:.1f} min")
    print(f"  Skipped invalid={skipped_invalid}, long={skipped_long}, "
          f"error={skipped_error}")
    print(f"  Index: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mgf")
    parser.add_argument("output_dir")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--clip-intensity", type=float, default=50000.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sequence-constrained", choices=["qk", "full"], default=None,
                        help="用序列约束残基偏移列；qk=只处理 Q/K 四列(推荐)，"
                             "full=所有残基列。默认关闭（保持原行为）")
    args = parser.parse_args()

    build_npz_chunked(
        input_mgf=args.input_mgf,
        output_dir=args.output_dir,
        tol=args.tol,
        clip_intensity=args.clip_intensity,
        chunk_size=args.chunk_size,
        num_workers=args.workers,
        sequence_constrained=args.sequence_constrained,
    )
