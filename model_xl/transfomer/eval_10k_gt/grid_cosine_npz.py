#!/usr/bin/env python3
"""Per-sample grid cosine between a V20 checkpoint and a training-format NPZ.

Case A of the validation chain: prediction vs the *filtered/averaged* spectrum it
was trained against, compared **directly on the 39×44×6 grid** — no peak
conversion, no PiPrime encoder. The metric is the masked cosine
(``spectrum_transformer.training.losses.masked_cosine_loss``) evaluated per
sample instead of per batch.

Why this script exists instead of ``eval_test_npz.py``: ``load_npz_data`` is
eager (``{k: raw[k] for k in raw.files}``), so a 303,625-row NPZ whose two grids
are 12.5 GB each would need ~25 GB RAM before the first forward pass. Here the
NPZ is treated as the zip archive it is: each big member is read as a sequential
``.npy`` stream in row chunks, and the small members are read whole. Peak RSS is
therefore ~(one chunk × 2 grids) + model, independent of row count.

To keep the numbers comparable with training-time eval, normalization and the
valid mask are NOT reimplemented — each chunk is fed through the real
``_build_dataset`` from ``spectrum_transformer.cli.train``.

Outputs (CSV only):
    grid_cosine_detail.csv   row_idx, sequence, charge, pepmass, rt, bb_len, cosine
    grid_cosine_summary.csv  group, n, mean, median, std   (overall + z2/z3/z4 + length buckets)

Usage:
    python grid_cosine_npz.py --npz <train-format .npz> --checkpoint <best_model.pt> \
        --out-dir <dir> [--chunk-size 4096] [--batch-size 64] [--gpu 0] [--limit 0]
"""
from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import zipfile
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_TF_DIR = _SCRIPT_DIR.parent
for _p in (str(_TF_DIR), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Checkpoints pickled under NumPy 2 use ``numpy._core``; alias it for older envs.
try:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
except Exception:
    pass

import torch  # noqa: E402
from numpy.lib import format as npformat  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from spectrum_transformer.warnings import configure_v20_warning_filters  # noqa: E402
configure_v20_warning_filters()

from spectrum_transformer.data.dataset import (  # noqa: E402
    build_cleavage_context_arrays,
)
from spectrum_transformer.models.factory import (  # noqa: E402
    build_model,
    prepare_checkpoint_for_inference,
)
from spectrum_transformer.cli.train import _build_dataset  # noqa: E402

SMALL_KEYS = ["position_masks", "channel_masks", "charges", "pepmasses",
              "rtinseconds", "lengths", "seqs"]
BIG_KEYS = ["input_mz_matrices", "target_intensity_matrices"]
BIG_ALIASES = {"input_mz_matrices": ["input_mz_matrices", "inputs"],
               "target_intensity_matrices": ["target_intensity_matrices", "targets"]}
LEN_BUCKETS = [("len<=15", 0, 15), ("len16-25", 16, 25), ("len>=26", 26, 10 ** 9)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--npz", required=True, help="training-format .npz (zip)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chunk-size", type=int, default=4096,
                   help="rows read from disk per streaming step (memory knob)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all rows (debug cap)")
    p.add_argument("--corpus", default=None,
                   help="label written into the summary. Default: inferred from the "
                        "npz path as '<parent_dir>/<stem>' so different corpora or "
                        "versions cannot be confused.")
    return p.parse_args()


class NpzRowReader:
    """Sequential row-chunk reader over one ``.npy`` member of an npz archive.

    Keeps a single open stream so chunks cost O(n) overall — reopening and
    seeking per chunk would re-decompress from the start every time.
    """

    def __init__(self, zf: zipfile.ZipFile, key: str, name: str):
        self.fp = zf.open(name, "r")
        version = npformat.read_magic(self.fp)
        if version == (1, 0):
            shape, order, dtype = npformat.read_array_header_1_0(self.fp)
        elif version in ((2, 0), (3, 0)):
            shape, order, dtype = npformat.read_array_header_2_0(self.fp)
        else:
            raise ValueError(f"{name}: unsupported .npy version {version}")
        if order:
            raise ValueError(f"{name}: Fortran-ordered, chunked row reads assume C order")
        self.key = key
        self.shape = shape
        self.dtype = np.dtype(dtype)
        trailing = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        self.row_bytes = trailing * self.dtype.itemsize
        self.pos = 0

    def read(self, n: int) -> np.ndarray:
        n = min(n, self.shape[0] - self.pos)
        if n <= 0:
            return np.empty((0,) + tuple(self.shape[1:]), dtype=self.dtype)
        buf, got = bytearray(), 0
        need = n * self.row_bytes
        while got < need:
            part = self.fp.read(need - got)
            if not part:
                raise EOFError(f"{self.key}: stream ended after {got}/{need} bytes")
            buf += part
            got += len(part)
        self.pos += n
        return np.frombuffer(bytes(buf), dtype=self.dtype).reshape((n,) + tuple(self.shape[1:]))


def pick_member(zf: zipfile.ZipFile, names) -> str:
    for n in names:
        if f"{n}.npy" in zf.namelist():
            return f"{n}.npy"
    raise KeyError(f"none of {names} found in archive")


def bb_len(seq: str) -> int:
    return sum(1 for ch in seq if ch.isalpha())


def build_model_and_stats(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config, state_dict = prepare_checkpoint_for_inference(ckpt)
    model = build_model(model_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return (model,
            np.asarray(ckpt["channel_means"], dtype=np.float32),
            np.asarray(ckpt["channel_stds"], dtype=np.float32),
            np.asarray(ckpt["aux_feature_means"], dtype=np.float32),
            np.asarray(ckpt["aux_feature_stds"], dtype=np.float32))


def summarize(detail_rows) -> list[list]:
    def agg(name, subset):
        cs = [c for _, _, _, c in subset]
        if not cs:
            return [name, 0, "nan", "nan", "nan"]
        return [name, len(cs), f"{statistics.mean(cs):.6f}",
                f"{statistics.median(cs):.6f}",
                f"{statistics.stdev(cs):.6f}" if len(cs) > 1 else "nan"]

    rows = [agg("all", detail_rows)]
    for z in sorted({r[1] for r in detail_rows}):
        rows.append(agg(f"z{z}", [r for r in detail_rows if r[1] == z]))
    for name, lo, hi in LEN_BUCKETS:
        rows.append(agg(name, [r for r in detail_rows if lo <= r[2] <= hi]))
    return rows


def main() -> int:
    args = parse_args()
    if not args.corpus:
        _npz = Path(args.npz)
        _parent = _npz.parent.name
        args.corpus = f"{_parent}/{_npz.stem}" if _parent not in ("", ".") else _npz.stem
    print(f"corpus label: {args.corpus}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "grid_cosine_detail.csv"
    summ_path = out_dir / "grid_cosine_summary.csv"

    device = torch.device(f"cuda:{args.gpu}") if (
        args.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    print(f"Device: {device}")
    model, ch_means, ch_stds, aux_means, aux_stds = build_model_and_stats(args.checkpoint, device)

    zf = zipfile.ZipFile(args.npz)
    in_name = pick_member(zf, BIG_ALIASES["input_mz_matrices"])
    tg_name = pick_member(zf, BIG_ALIASES["target_intensity_matrices"])
    print(f"NPZ: {args.npz}\n  inputs={in_name}  targets={tg_name}")

    small = {}
    for k in SMALL_KEYS:
        member = f"{k}.npy"
        if member in zf.namelist():
            small[k] = np.load(io.BytesIO(zf.read(member)), allow_pickle=(k == "seqs"))
    reader_in = NpzRowReader(zf, in_name, in_name)
    reader_tg = NpzRowReader(zf, tg_name, tg_name)
    n_total = reader_in.shape[0]
    n_run = min(args.limit, n_total) if args.limit else n_total
    print(f"  rows={n_total}  running={n_run}  chunk={args.chunk_size}  batch={args.batch_size}")

    detail_rows = []
    done = 0
    with open(detail_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_idx", "sequence", "charge", "pepmass", "rt", "bb_len", "cosine"])

        for start in range(0, n_run, args.chunk_size):
            stop = min(start + args.chunk_size, n_run)
            inputs = reader_in.read(stop - start)
            targets = reader_tg.read(stop - start)
            sl = slice(start, stop)

            raw = {"inputs": inputs, "targets": targets,
                   "position_masks": small["position_masks"][sl],
                   "channel_masks": small["channel_masks"][sl],
                   "charges": small["charges"][sl],
                   "lengths": small["lengths"][sl],
                   "seqs": small["seqs"][sl],
                   "aux_features": np.stack(
                       [small["charges"][sl], small["pepmasses"][sl],
                        small["rtinseconds"][sl]], axis=1).astype(np.float32)}
            aa, sc = build_cleavage_context_arrays(raw["seqs"], raw["lengths"])
            raw["cleavage_aa_ids"], raw["cleavage_scalar_features"] = aa, sc
            loader = DataLoader(_build_dataset(raw, ch_means, ch_stds, aux_means, aux_stds),
                                batch_size=args.batch_size, shuffle=False, num_workers=0)

            cosines = []
            with torch.no_grad():
                for batch in loader:
                    inputs_b = batch["input"].to(device)
                    target_b = batch["target"].to(device)
                    vm = batch["valid_mask"].to(device)
                    extra = {k: batch[k].to(device) for k in
                             ("cleavage_aa_ids", "cleavage_scalar_features", "charge", "length")
                             if batch.get(k) is not None}
                    out = model(inputs_b, batch["aux_features"].to(device), valid_mask=vm, **extra)
                    pred = out["prediction"] if isinstance(out, dict) else out
                    cosines.append(F.cosine_similarity(
                        (pred * vm).reshape(pred.shape[0], -1),
                        (target_b * vm).reshape(target_b.shape[0], -1),
                        dim=1, eps=1e-8).detach().cpu().numpy())

            for i, c in enumerate(np.concatenate(cosines) if cosines else []):
                idx = start + i
                seq = str(small["seqs"][idx])
                w.writerow([idx, seq, int(small["charges"][idx]),
                            f"{float(small['pepmasses'][idx]):.6f}",
                            f"{float(small['rtinseconds'][idx]):.4f}",
                            bb_len(seq), f"{float(c):.6f}"])
                detail_rows.append((idx, int(small["charges"][idx]), bb_len(seq), float(c)))
            done += stop - start
            print(f"  {done}/{n_run}  cos_mean(so far)="
                  f"{statistics.mean(r[3] for r in detail_rows):.4f}", flush=True)

    summary_rows = summarize(detail_rows)
    with open(summ_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "group", "n", "mean", "median", "std"])
        for r in summary_rows:
            w.writerow([args.corpus, *r])

    print(f"\ndone: {done} rows -> {detail_path}")
    print(f"summary -> {summ_path}  (corpus={args.corpus})")
    print(f"  overall mean cosine = {summary_rows[0][2]}  (n={summary_rows[0][1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
