#!/usr/bin/env python3
"""Group MGF block(s) into one file per (peptide, charge), tagging each block's SOURCE.

massTest is a single flat MGF, so `collect_original_with_prediction.py` (which scans a
`<species>/peaks.db.mgf` directory layout) cannot be used. This script does the grouping
while emitting the **same filenames** it would, so every downstream eval_10k_gt script
works unchanged:

  --original-mgf massTest_filtered.mgf  --out-dir Grouped_Filtered
      every block tagged SOURCE=ORIGINAL. Feeds
      `cosine_detail_csv.py` (filtered vs filtered), `grouped_to_average_mgf.py`
      (which averages exactly the SOURCE=ORIGINAL blocks).

  --original-mgf massTest.mgf  --prediction-mgf prosit_predicted.mgf  --out-dir Grouped_Prosit
      N originals + 1 SOURCE=MODEL_PREDICTION per group. Feeds `batch_cosine_summary.py`,
      `cosine_detail_csv.py` and `cosine_avg_vs_pred.py` as-is.

Streams and appends block by block, so memory stays bounded (30k groups x ~14 spectra is
~1.4 GB of text, never held in RAM).

Usage:
  python build_groups.py --original-mgf massTest.mgf --out-dir Grouped_Real
  python build_groups.py --original-mgf massTest_filtered.mgf --out-dir Grouped_Filtered
  python build_groups.py --original-mgf massTest.mgf \
      --prediction-mgf prosit_predicted.mgf --out-dir Grouped_Prosit
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from mgf_io import (iter_blocks, make_group_filename, open_binary_stream,
                    parse_block_key, tag_block, write_block, finish_stream)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group MGF blocks into per-(peptide,charge) files")
    p.add_argument("--original-mgf", action="append", default=[],
                   help="MGF whose blocks are tagged SOURCE=ORIGINAL (repeatable)")
    p.add_argument("--prediction-mgf", default=None,
                   help="optional MGF whose blocks are tagged SOURCE=MODEL_PREDICTION")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threads", type=int, default=0,
                   help="xz decompression threads for .xz inputs; 0 = auto (min(cpu-1, 8))")
    p.add_argument("--limit", type=int, default=0, help="debug: stop after N blocks per input")
    return p.parse_args()


def stream_one(mgf: str, tag: str, out_dir: Path, seen: set, counts: Counter, threads: int,
               limit: int = 0) -> tuple[int, int]:
    """Append every block of `mgf` into its group file. Returns (n_blocks, n_no_key)."""
    stream, proc, rsz = open_binary_stream(mgf, threads)
    n_blocks = n_no_key = 0
    try:
        with stream:
            for block in iter_blocks(stream, read_size=rsz):
                n_blocks += 1
                key = parse_block_key(block)
                if key is None:
                    n_no_key += 1
                else:
                    path = out_dir / make_group_filename(key[0], key[1])
                    # first write must truncate any stale file left by an earlier run
                    mode = "a" if key in seen else "w"
                    write_block(path, tag_block(block, tag), mode)
                    seen.add(key)
                    counts[key] += 1
                if limit and n_blocks >= limit:
                    break
                if n_blocks % 100000 == 0:
                    print(f"    {n_blocks:,} blocks...", file=sys.stderr, flush=True)
    finally:
        if proc:
            try:
                stream.close()
            finally:
                finish_stream(proc, capped=bool(limit))
    print(f"  {mgf}: {n_blocks:,} blocks, {n_no_key:,} without SEQ/CHARGE",
          file=sys.stderr, flush=True)
    if n_no_key:
        raise ValueError(f"{n_no_key} blocks missing peptide/charge")
    return n_blocks, n_no_key


def main() -> int:
    args = parse_args()
    if not args.original_mgf and not args.prediction_mgf:
        raise ValueError("At least one MGF input is required")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.mgf")) or (out_dir / "group_map.csv").exists():
        raise FileExistsError("Grouping requires a fresh output directory")

    seen: set = set()                 # groups already written (so we truncate only once)
    n_orig: Counter = Counter()
    n_pred: Counter = Counter()

    print(f"[1/2] tagging SOURCE=ORIGINAL -> {out_dir}", file=sys.stderr, flush=True)
    for mgf in args.original_mgf:
        stream_one(mgf, "ORIGINAL", out_dir, seen, n_orig, args.threads, args.limit)

    if args.prediction_mgf:
        print(f"[2/2] tagging SOURCE=MODEL_PREDICTION from {args.prediction_mgf}",
              file=sys.stderr, flush=True)
        stream_one(args.prediction_mgf, "MODEL_PREDICTION", out_dir, seen, n_pred,
                   args.threads, args.limit)

    if any(n > 1 for n in n_pred.values()):
        raise ValueError("Expected at most one prediction per peptide+charge")

    # ---- report ----
    only_pred = sorted(k for k in n_pred if k not in n_orig) if args.original_mgf else []
    map_csv = out_dir / "group_map.csv"
    with open(map_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group_file", "sequence", "charge", "n_original", "n_prediction"])
        for key in sorted(set(n_orig) | set(n_pred)):
            w.writerow([make_group_filename(key[0], key[1]), key[0], key[1],
                        n_orig[key], n_pred[key]])
    if only_pred:
        (out_dir / "prediction_without_original.txt").write_text(
            "\n".join(f"{s}\t{c}" for s, c in only_pred) + "\n", encoding="utf-8")

    sizes = [c for c in n_orig.values() if c]
    print(f"\ndone: {len(n_orig):,} groups with originals, "
          f"{sum(n_orig.values()):,} original blocks, "
          f"{sum(n_pred.values()):,} prediction blocks", file=sys.stderr)
    if sizes:
        print(f"  group size: min={min(sizes)} max={max(sizes)} "
              f"mean={sum(sizes)/len(sizes):.2f}", file=sys.stderr)
    print(f"  map -> {map_csv}", file=sys.stderr)
    if only_pred:
        print(f"  WARNING: {len(only_pred):,} prediction(s) had no original spectrum "
              f"(see prediction_without_original.txt)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
