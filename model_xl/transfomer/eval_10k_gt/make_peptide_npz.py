#!/usr/bin/env python3
"""Build the V20 input NPZ for ~10k peptides (9 species) straight from the
ground-truth MGFs (`SEQ=` + `CHARGE=`), in the exact format
`predict_npz_to_mgf.py` expects:

    inputs (39,46,6) input-m/z grids, position_masks, channel_masks,
    charges, pepmasses, rtinseconds, seqs

No experimental peaks are used here — the spectra are only predicted later and
compared against the originals by the standard eval chain
(collect_original_with_prediction.py -> PiPrime embeddings -> cosine summary).

Usage:
    python make_peptide_npz.py --root <9species/Casanovo_preprocessed_data> \
        --out ../../datasets/v20_10k_9species.npz --per-species 1111
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # transfomer/
import predict_mgf as pm  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="dir with <species>.mgf (GT MGFs)")
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--per-species", type=int, default=0,
                   help="max unique (peptide,charge) per species; 0 = no cap "
                        "(use every unique peptide of the full per-species MGF)")
    p.add_argument("--min-len", type=int, default=7)
    p.add_argument("--max-len", type=int, default=30,
                   help="cap backbone at 30 so Prosit can score the same peptides")
    p.add_argument("--min-mz", type=float, default=100.0)
    p.add_argument("--max-total", type=int, default=0,
                   help="hard cap on the TOTAL peptide count (0 = no cap; "
                        "e.g. --per-species 1112 --max-total 10000 -> exactly 10000)")
    p.add_argument("--species", default="",
                   help="comma list to restrict species (default: all *.mgf)")
    return p.parse_args()


def bb_len(anno: str) -> int:
    return sum(1 for ch in anno if ch.isalpha())


def iter_mgf(path: Path):
    meta: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    inside = False
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            s = raw.strip()
            if s == "BEGIN IONS":
                inside, meta, peaks = True, {}, []
                continue
            if s == "END IONS":
                inside = False
                yield meta, peaks
                continue
            if not inside or not s:
                continue
            if "=" in s and not s[0].isdigit():
                k, _, v = s.partition("=")
                meta[k.strip().upper()] = v.strip()
                continue
            parts = s.split()
            if len(parts) >= 2:
                peaks.append((float(parts[0]), float(parts[1])))


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    # layouts: <root>/*.mgf (9species)  or  <root>/<sp>/*.mgf (v1-search)
    pairs = [(m.stem, m) for m in sorted(root.glob("*.mgf"))]
    if not pairs:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for m in sorted(d.glob("*.mgf")):
                pairs.append((d.name, m))
    if args.species:
        want = {s.strip() for s in args.species.split(",")}
        pairs = [(sp, m) for sp, m in pairs if sp in want]
    if not pairs:
        print(f"no *.mgf found under {root}")
        return 1

    seqs, charges = [], []
    obs_rt: list[float] = []
    obs_pmass: list[float] = []
    per_sp_counts: dict[str, int] = {}
    cap = args.per_species if args.per_species and args.per_species > 0 else None
    for sp, mgf in pairs:
        if args.max_total and len(seqs) >= args.max_total:
            break
        if cap and per_sp_counts.get(sp, 0) >= cap:
            continue                      # species quota already filled
        seen: set[tuple[str, int]] = set()
        n_sp = 0
        for meta, _pk in iter_mgf(mgf):
            if cap and per_sp_counts.get(sp, 0) + n_sp >= cap:
                break
            if args.max_total and len(seqs) + n_sp >= args.max_total:
                break
            seq = meta.get("SEQ", "")
            ch = meta.get("CHARGE", "")
            if not seq or not ch:
                continue
            chg = int("".join(c for c in ch if c.isdigit()) or 0)
            if not (2 <= chg <= 4) or not (args.min_len <= bb_len(seq) <= args.max_len):
                continue
            key = (seq, chg)
            if key in seen:
                continue
            seen.add(key)
            seqs.append(seq)
            charges.append(chg)
            # real observed context (the model consumes RT/PEPMASS as aux features)
            try:
                obs_rt.append(float(meta.get("RTINSECONDS", 0.0) or 0.0))
            except ValueError:
                obs_rt.append(0.0)
            try:
                obs_pmass.append(float(meta.get("PEPMASS", 0.0) or 0.0))
            except ValueError:
                obs_pmass.append(0.0)
            n_sp += 1
        per_sp_counts[sp] = per_sp_counts.get(sp,0) + n_sp
        print(f"  {sp}: {n_sp} peptides")

    n = len(seqs)
    inputs = np.zeros((n, pm.MAX_CLEAVAGE_LEN, pm.NUM_TOKENS, pm.NUM_CHANNELS), np.float32)
    pos_masks = np.zeros((n, pm.MAX_CLEAVAGE_LEN), np.float32)
    ch_masks = np.zeros((n, pm.NUM_CHANNELS), np.float32)
    masses = np.zeros(n, np.float32)
    for i, (seq, chg) in enumerate(zip(seqs, charges)):
        inputs[i] = pm.build_input_mz_matrix(seq, args.min_mz)
        pos_masks[i], _ = pm.build_position_mask(seq)
        ch_masks[i] = pm.build_channel_mask(chg)
        masses[i] = obs_pmass[i] if obs_pmass[i] > 0 else pm.calculate_pepmass(seq, chg)
    np.savez_compressed(
        args.out,
        inputs=inputs, position_masks=pos_masks, channel_masks=ch_masks,
        charges=np.asarray(charges, np.int64),
        pepmasses=masses,
        rtinseconds=np.asarray(obs_rt, np.float32),
        seqs=np.asarray(seqs, dtype=object),
    )
    print(f"total {n} peptides -> {args.out}  (per species: {per_sp_counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
