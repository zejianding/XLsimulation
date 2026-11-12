#!/usr/bin/env python3
"""Predicted-side intensity filtering on collected group MGFs (rel levels).

Mirrors the FDR/no-FDR convention: for the MODEL_PREDICTION block of each group,
keep peaks with intensity > max(min_abs, rel * predicted_max) of that peptide
(optionally also cap to top-K).  ORIGINAL blocks are copied verbatim, so the
metric stays "prediction vs the same originals" and only the predicted spectrum
is filtered.  Same rule is applied to V20 and Prosit runs → the rel dimension is
comparable across predictors.

Outputs one directory per rel level:
    <out-root>/Grouped_MGFs_rel000   (rel = 0 -> no filtering, keep all peaks)
    <out-root>/Grouped_MGFs_rel005/…

Usage:
    python filter_group_predictions.py --mgf-dir <run>/Grouped_MGFs \
        --out-root <run>/variants --rels 0,0.05,0.10,0.20 [--top-k 0] [--min-abs 0.01]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mgf-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--rels", default="0,0.05,0.10,0.20")
    p.add_argument("--top-k", type=int, default=0, help="0 = no cap")
    p.add_argument("--min-abs", type=float, default=0.01)
    p.add_argument("--exp-transform", default="none",
                   choices=["none", "sqrt_clip", "sqrt"],
                   help="transform ORIGINAL (experimental) peak intensities, mirroring the "
                        "V20 scoring domain: sqrt_clip = sqrt(min(x,clip))/sqrt(clip)")
    p.add_argument("--pred-transform", default="none",
                   choices=["none", "sqrt_clip", "sqrt"],
                   help="transform the MODEL_PREDICTION intensities (e.g. 'sqrt' to put a "
                        "raw Prosit prediction on the same footing as sqrt_clip experiments)")
    p.add_argument("--clip", type=float, default=50000.0)
    return p.parse_args()


def split_blocks(text: str):
    """Yield (header_lines, peak_lines) per BEGIN/END IONS block, preserving order."""
    cur, peaks, inside = [], [], False
    for raw in text.splitlines(keepends=True):
        s = raw.strip()
        if s == "BEGIN IONS":
            inside, cur, peaks = True, [raw], []
            continue
        if s == "END IONS":
            inside = False
            yield cur, peaks
            continue
        if not inside:
            continue
        if "=" in s and not s[0].isdigit():
            cur.append(raw)
        else:
            parts = s.split()
            if len(parts) >= 2:
                peaks.append((float(parts[0]), float(parts[1]), raw))


def _tf(x: float, method: str, clip: float) -> float:
    if method == "sqrt_clip":
        import math
        return math.sqrt(min(x, clip)) / math.sqrt(clip)
    if method == "sqrt":
        import math
        return math.sqrt(max(x, 0.0))
    return x


def fmt(peaks, method: str = "none", clip: float = 50000.0) -> str:
    return "".join(f"{mz:.6f} {_tf(it, method, clip):.10f}\n" for mz, it, _ in peaks)


def lvl(r: float) -> str:
    return f"{int(round(r * 100)):03d}"


def main() -> int:
    args = parse_args()
    mgf_dir = Path(args.mgf_dir)
    rels = [float(x) for x in args.rels.split(",") if x.strip()]
    tag = f"rel{lvl(rels[0])}-{lvl(rels[-1])}_exp{args.exp_transform}_pred{args.pred_transform}"
    if args.top_k:
        tag += f"_top{args.top_k}"
    out_root = Path(args.out_root)
    out_dirs = {r: out_root / tag / f"Grouped_MGFs_rel{lvl(r)}" for r in rels}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_groups = 0
    for mgf in sorted(mgf_dir.glob("*.mgf")):
        text = mgf.read_text(encoding="utf-8", errors="ignore")
        blocks = list(split_blocks(text))
        pred_i = None
        for i, (hdr, _pk) in enumerate(blocks):
            if any("MODEL_PREDICTION" in h for h in hdr):
                pred_i = i
                break
        if pred_i is None:
            continue
        for r, outdir in out_dirs.items():
            out = []
            for i, (hdr, pk) in enumerate(blocks):
                out.extend(hdr)
                if i != pred_i:            # ORIGINAL: optional exp-domain transform
                    out.append(fmt(pk, args.exp_transform, args.clip))
                else:                       # PREDICTION: optional pred transform + rel filter
                    pk2 = [(mz, _tf(it, args.pred_transform, args.clip), ln) for mz, it, ln in pk]
                    if r <= 0:
                        kept = pk2
                    else:
                        pmax = max((it for _, it, _ in pk2), default=0.0)
                        thr = max(args.min_abs, r * pmax)
                        kept = [(mz, it, ln) for mz, it, ln in pk2 if it > thr]
                        if args.top_k and len(kept) > args.top_k:
                            kept = sorted(kept, key=lambda x: -x[1])[:args.top_k]
                            kept.sort(key=lambda x: x[0])
                    out.append(fmt(kept))
                out.append("END IONS\n\n")
            (outdir / mgf.name).write_text("".join(out), encoding="utf-8")
        n_groups += 1
    print(f"filtered {n_groups} groups -> {len(rels)} variants under {out_root/tag}")
    print(f"OUTDIR={out_root/tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
