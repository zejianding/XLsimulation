"""
多进程 Token 过滤 —— 调用原始 process_one_mgf_streaming 的逻辑，但并行化。

用法：
    python TokenFilter_parallel.py <input.mgf> <output.mgf> --tol 0.05 --workers 8
"""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from TokenTableOriginalAnalyze_massivekb_mods_precise_tokens import (
    iter_mgf_blocks,
    parse_ions_block,
    process_one_spectrum,
    raw_peak_line_map,
    rebuild_filtered_block,
)

# 全局变量，Pool 初始化时设置
_worker_tol = 0.05


def _init_worker(tol):
    global _worker_tol
    _worker_tol = tol


def _process_one_block(block):
    """处理单个 spectrum block，返回结果或 None"""
    seq, charge, headers, peaks = parse_ions_block(block)
    if seq is None or charge is None:
        return None
    try:
        kept = process_one_spectrum(seq, charge, peaks, _worker_tol)
    except Exception:
        return None
    return (headers, kept, raw_peak_line_map(block))


def process_one_mgf_parallel(input_mgf, output_mgf, tol=0.05, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"\nToken filtering (parallel): {input_mgf} -> {output_mgf}")
    print(f"  tol={tol}, workers={num_workers}")

    mgf_iter = iter_mgf_blocks  # 自动支持 .mgf 和 .mgf.xz

    total = 0
    kept = 0
    t0 = time.time()

    tmp = Path(str(output_mgf) + ".partial")
    with open(tmp, "w", encoding="utf-8") as fout:
        with Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(tol,)) as pool:
            for result in pool.imap(
                _process_one_block, mgf_iter(input_mgf), chunksize=50
            ):
                total += 1
                if result is not None:
                    headers, kept_peaks, raw_lines = result
                    fout.writelines(rebuild_filtered_block(
                        headers, kept_peaks, raw_lines))
                    fout.write("\n")
                    kept += 1

                if total % 5000 == 0:
                    elapsed = time.time() - t0
                    print(f"  Processed {total} spectra "
                          f"({total / elapsed:.0f} spectra/s), kept={kept}...")

    if total != kept:
        raise RuntimeError(f"Token filtering dropped {total - kept} spectra; output not committed")
    tmp.replace(output_mgf)
    elapsed = time.time() - t0
    print(f"Done: {total} total, {kept} kept, {elapsed / 60:.1f} min, "
          f"{total / elapsed:.0f} spectra/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mgf")
    parser.add_argument("output_mgf")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    process_one_mgf_parallel(
        input_mgf=args.input_mgf,
        output_mgf=args.output_mgf,
        tol=args.tol,
        num_workers=args.workers,
    )
