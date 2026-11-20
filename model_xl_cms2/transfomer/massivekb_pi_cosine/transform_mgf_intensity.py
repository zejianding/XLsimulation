#!/usr/bin/env python3
"""Apply a precise intensity transform to a single MGF or directory of MGFs."""
from __future__ import annotations
import argparse
import math
import shutil
from pathlib import Path
from mgf_io import iter_blocks, open_binary_stream, finish_stream


def transform_file(src, dst, transform="sqrt_clip", clip=50000.0, threads=0):
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("Input and output must differ")
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    stream, proc, size = open_binary_stream(src, threads)
    n = 0
    with stream, tmp.open("w", encoding="utf-8", newline="\n") as out:
        for block in iter_blocks(stream, read_size=size):
            lines = []
            for line in block.splitlines():
                if not line.strip() or "=" in line or line.strip() in ("BEGIN IONS", "END IONS"):
                    lines.append(line)
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid peak: {line!r}")
                mz, intensity = map(float, parts)
                if not math.isfinite(mz) or not math.isfinite(intensity) or mz < 0 or intensity < 0:
                    raise ValueError(f"Invalid peak: {line!r}")
                if transform == "none":
                    lines.append(line)
                else:
                    value = math.sqrt(intensity) if transform == "sqrt" else math.sqrt(min(intensity, clip) / clip)
                    lines.append(f"{parts[0]} {value:.12g}")
            out.write("\n".join(lines).strip("\n") + "\nEND IONS\n")
            n += 1
    finish_stream(proc)
    tmp.replace(dst)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--transform", choices=["sqrt_clip", "sqrt", "none"], default="sqrt_clip")
    p.add_argument("--clip", type=float, default=50000.0)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    files = sorted(src.glob("*.mgf")) if src.is_dir() else [src]
    if not files:
        raise ValueError("No MGF inputs")
    n = sum(transform_file(f, dst / f.name if src.is_dir() else dst,
                           a.transform, a.clip, a.threads) for f in files)
    if src.is_dir() and (src / "group_map.csv").exists():
        shutil.copyfile(src / "group_map.csv", dst / "group_map.csv")
    print(f"Transformed {n} spectra ({a.transform})")


if __name__ == "__main__":
    main()
