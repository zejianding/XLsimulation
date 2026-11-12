#!/usr/bin/env python3
"""One-time compaction of a Prosit cache: keep only clean b/y-annotation rows.

The Koina multifrag response carries many rows per peptide (neutral losses /
other ions) that step4_prosit_score never uses (build_candidate_peaks keeps only
b/y).  Streaming-filtering the legacy multi-hundred-million-row gzip shrinks it
~4-10x so subsequent loads (and the in-memory dict) become cheap.

Progress: a tqdm bar over the COMPRESSED bytes read (percentage ~ how much of the
file has been scanned; read-ahead buffering may run slightly ahead of the last
processed row).

Usage:
    python compact_prosit_cache.py \
        --cache output/prosit/mouse/predictions/prosit_cache.tsv.gz
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from pathlib import Path

_BY = re.compile(r"^[bByY]\d+(?:\^\d+)?$")
HEADER = b"peptide\tcharge\tmz\tintensity\tannotation\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", required=True, help="prosit_cache.tsv.gz path")
    p.add_argument("--dry-run", action="store_true", help="only count before/after")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.cache)
    if not src.exists():
        print(f"no cache at {src}")
        return 1
    tmp = src.with_name(src.name + ".compact.tsv.gz")
    size = os.path.getsize(src)
    try:
        from tqdm import tqdm
        bar = tqdm(total=size, unit="B", unit_scale=True,
                   desc="compact prosit cache", mininterval=0.5)
    except Exception:
        bar = None

    kept = total = 0
    raw = open(src, "rb")
    out = gzip.open(tmp, "wb")
    last = 0
    try:
        gz = gzip.GzipFile(fileobj=raw)
        out.write(HEADER)
        for bline in gz:
            cur = raw.tell()
            if bar is not None and cur > last:
                bar.update(cur - last)
                last = cur
            if bline.startswith(b"peptide\t"):
                continue
            total += 1
            if _BY.match(bline.rstrip(b"\n").split(b"\t")[-1].decode("ascii", "replace")):
                out.write(bline)
                kept += 1
    finally:
        out.close()
        raw.close()
    if bar is not None:
        bar.update(max(0, size - bar.n))  # flush read-ahead remainder
        bar.close()
    pct = kept / total * 100 if total else 0.0
    print(f"\ncache {src}: total_rows={total} b/y_rows={kept} ({pct:.1f}%)")
    if not args.dry_run:
        os.replace(tmp, src)
        print(f"replaced -> {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
