#!/usr/bin/env python3
"""Build one average spectrum per peptide from collected group MGFs (B2 左半边).

Takes the standard eval artifact ``<run>/Grouped_MGFs/<seq>_charge<z>_<hash>.mgf``
— each file holding one ``SOURCE=MODEL_PREDICTION`` block plus one block per
matched original experimental spectrum — and emits the *average of the
ORIGINALS only*, one file per group, keeping the same filename stem so the
prediction and the average stay pairable.

The prediction block is dropped BEFORE averaging: ``AverageSpectrum_batch
.parse_mgf_block`` does not read ``SOURCE=``, so feeding it a group file
directly would fold the predicted peaks into the average.

Groups with a single original are copied verbatim (peak list untouched), matching
``AverageSpectrum_batch.rebuild_single_block`` — averaging one spectrum through
``average_peaks_by_tolerance`` would chain-merge peaks 100.00/100.05 into one.

    <out-dir>/<same stem>.mgf   TITLE=<seq>|<charge>, SOURCE=AVERAGED

Usage:
    python grouped_to_average_mgf.py --mgf-dir <run>/Grouped_MGFs \
        --out-dir <run>/Average_MGFs [--tol 0.1] [--limit 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
if str(_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASET_DIR))

from AverageSpectrum_batch import average_peaks_by_tolerance  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mgf-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tol", type=float, default=0.1, help="peak m/z merge tolerance")
    p.add_argument("--limit", type=int, default=0, help="0 = all groups (debug cap)")
    return p.parse_args()


def split_blocks(text: str):
    """Yield the raw lines of each BEGIN/END IONS block."""
    cur, inside = [], False
    for raw in text.splitlines():
        s = raw.strip()
        if s == "BEGIN IONS":
            inside, cur = True, []
            continue
        if s == "END IONS":
            inside = False
            yield cur
            continue
        if inside:
            cur.append(s)


def parse_block(lines):
    """Fields we need from one block; peaks as (mz, intensity) tuples."""
    seq = charge = source = None
    pepmass = rt = None
    peaks = []
    for s in lines:
        if "=" in s and not s[0].isdigit():
            k, _, v = s.partition("=")
            k, v = k.strip().upper(), v.strip()
            if k == "SEQ":
                seq = v
            elif k == "CHARGE":
                charge = "".join(c for c in v if c.isdigit())
            elif k == "SOURCE":
                source = v
            elif k == "PEPMASS":
                try:
                    pepmass = float(v.split()[0])
                except ValueError:
                    pepmass = None
            elif k == "RTINSECONDS":
                try:
                    rt = float(v)
                except ValueError:
                    rt = None
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                peaks.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return dict(seq=seq, charge=charge, source=source or "UNKNOWN",
                pepmass=pepmass, rt=rt, peaks=peaks)


def mean_of(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    args = parse_args()
    mgf_dir, out_dir = Path(args.mgf_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_groups = n_avg = n_single = n_skipped = 0
    for mgf_path in sorted(mgf_dir.glob("*.mgf")):
        blocks = [parse_block(b) for b in split_blocks(
            mgf_path.read_text(encoding="utf-8", errors="ignore"))]
        # same rule as cosine_detail_csv.py: anything not tagged PREDICTION is an original
        orig = [b for b in blocks if "PREDICTION" not in b["source"].upper()]
        if not orig:
            n_skipped += 1
            continue

        seq = orig[0]["seq"] or ""
        charge = orig[0]["charge"] or ""
        if len(orig) == 1:
            peaks = orig[0]["peaks"]
            n_single += 1
        else:
            peaks = average_peaks_by_tolerance([b["peaks"] for b in orig], args.tol)
            n_avg += 1

        pepmass = mean_of([b["pepmass"] for b in orig])
        rt = mean_of([b["rt"] for b in orig])
        lines = ["BEGIN IONS", "SOURCE=AVERAGED", f"TITLE={seq}|{charge}"]
        if pepmass is not None:
            lines.append(f"PEPMASS={pepmass:.6f}")
        if rt is not None:
            lines.append(f"RTINSECONDS={rt:.6f}")
        if charge:
            lines.append(f"CHARGE={charge}+")
        if seq:
            lines.append(f"SEQ={seq}")
        lines.append(f"COMMENT=AVERAGE_SPECTRUM num_spectra={len(orig)} "
                     f"peaks={len(peaks)} tol={args.tol}")
        lines += [f"{mz:.6f} {it:.6f}" for mz, it in peaks]
        lines.append("END IONS")
        (out_dir / mgf_path.name).write_text("\n".join(lines) + "\n\n", encoding="utf-8")

        n_groups += 1
        if n_groups % 500 == 0:
            print(f"  {n_groups} groups ...", flush=True)
        if args.limit and n_groups >= args.limit:
            break

    print(f"done: {n_groups} averaged groups ({n_avg} multi / {n_single} single-copy), "
          f"{n_skipped} skipped (no original) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
