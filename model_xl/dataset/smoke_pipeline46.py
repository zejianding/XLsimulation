#!/usr/bin/env python3
"""End-to-end regression test of the MGF -> NPZ and MGF -> memmap flows.

Unlike ``smoke_token46.py`` (which uses 2 spectra), this drives the whole
pipeline with **N > 100** synthetic spectra on purpose: both
``shards_to_npz.py`` and ``merge_shards_to_memmap.py`` classify arrays by
``shape[0] > 100``, so a small N silently takes the "constant" fallback and
never exercises the real merge path.

Chain:
    raw MGF -> TokenFilter_parallel -> AverageSpectrum_diskbased
            -> NPZBuilder_chunked -> {merge_shards_to_memmap, shards_to_npz}
            -> spectrum_transformer.data.dataset.load_npz_data
            -> 1 training epoch

Run from anywhere:  python dataset/smoke_pipeline46.py
"""

from __future__ import annotations

import json
import pathlib
import random
import subprocess
import sys
import tempfile

_ROOT = pathlib.Path(__file__).resolve().parent.parent  # v20_0918/
for _p in (_ROOT, _ROOT / "transfomer", _ROOT / "dataset"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import token_table as tt  # noqa: E402
import trainingExamples as te  # noqa: E402
from spectrum_transformer.data.dataset import load_npz_data  # noqa: E402

_RESULTS: list[tuple[str, bool]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


_AA = "ACDEFGHIKLMNPQRSTVWY"
_rng = random.Random(20260918)


def _make_seq() -> str:
    return "".join(_rng.choice(_AA) for _ in range(_rng.randint(8, 20)))


def _write_mgf(path: pathlib.Path, records) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i, (seq, charge) in enumerate(records):
            fh.write(
                "BEGIN IONS\nTITLE=t%d\nSEQ=%s\nCHARGE=%d\n"
                "PEPMASS=850.0\nRTINSECONDS=%d\n" % (i, seq, charge, 30 + i % 50)
            )
            for ion in te.generate_by_ions_with_mods(seq, max_charge=min(3, charge)):
                mz = float(ion["mz"]) + _rng.uniform(-0.006, 0.006)
                fh.write("%.6f %.6f\n" % (mz, 100.0 + _rng.random() * 900.0))
            fh.write("END IONS\n\n")


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_ROOT / "dataset" / script), *map(str, args)],
        capture_output=True, text=True,
    )


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="v0918_pipeline_"))
    print("workdir:", tmp)

    seqs = [_make_seq() for _ in range(120)]
    records = [(s, 2 if i % 2 == 0 else 3) for i, s in enumerate(seqs)]
    records += [records[i] for i in range(0, 60)]   # duplicates -> real averaging
    records += [(s, 2) for s in seqs[:30]]
    print(f"synthetic spectra: {len(records)}")

    raw = tmp / "raw.mgf"
    _write_mgf(raw, records)

    # 1) token filter (46-offset table, tol 0.05)
    filtered = tmp / "filtered.mgf"
    r = _run("TokenFilter_parallel.py", raw, filtered, "--tol", "0.05", "--workers", "1")
    check("TokenFilter_parallel ran", r.returncode == 0, f"exit {r.returncode}")
    if r.returncode:
        print(r.stderr[-800:])
        return 1

    # 2) average spectra.  Input is kept by default (deletion is opt-in via
    #    --cleanup-input), so it can be fed directly.
    averaged = tmp / "averaged.mgf"
    r = _run("AverageSpectrum_diskbased.py", filtered, averaged, "--tol", "0.05", "--buckets", "8")
    check("AverageSpectrum_diskbased ran", r.returncode == 0, f"exit {r.returncode}")
    if r.returncode:
        print(r.stderr[-800:])
        return 1
    check("averaging keeps its input by default", filtered.exists())

    # deletion is opt-in only
    opt_in = tmp / "opt_in.mgf"
    opt_in.write_bytes(filtered.read_bytes())
    r = _run("AverageSpectrum_diskbased.py", opt_in, tmp / "opt_out.mgf",
             "--tol", "0.05", "--buckets", "4", "--cleanup-input")
    check("--cleanup-input removes the input", r.returncode == 0 and not opt_in.exists(),
          f"exit {r.returncode}, exists={opt_in.exists()}")

    # 3) sharded NPZ build
    import NPZBuilder_chunked as nb

    shard_dir = tmp / "shards"
    nb.build_npz_chunked(str(averaged), str(shard_dir), tol=0.05, num_workers=1)

    with open(shard_dir / "index.json") as f:
        n_examples = json.load(f)["total_examples"]
    check("shards hold >100 examples (real merge path)", n_examples > 100, f"N={n_examples}")

    shard0 = sorted(shard_dir.glob("shard_*.npz"))[0]
    with np.load(shard0, allow_pickle=True) as d:
        check("shard input_mz_matrices is (N,39,46,6)",
              d["input_mz_matrices"].shape[1:] == (39, 46, 6), str(d["input_mz_matrices"].shape))
        check("shard target_intensity_matrices is (N,39,46,6)",
              d["target_intensity_matrices"].shape[1:] == (39, 46, 6), str(d["target_intensity_matrices"].shape))
        check("shard tokens_charge1/2/3 are (46,)",
              all(d[k].shape == (46,) for k in ("tokens_charge1", "tokens_charge2", "tokens_charge3")))
        check("shard tokens_charge1 == token_table[1]",
              np.allclose(d["tokens_charge1"], np.asarray(tt.CHARGE_TOKENS[1], np.float32)))

    # 4a) memmap path
    merged = tmp / "merged"
    r = _run("merge_shards_to_memmap.py", shard_dir, merged)
    check("merge_shards_to_memmap ran", r.returncode == 0, f"exit {r.returncode}")
    if r.returncode:
        print(r.stderr[-800:])
        return 1

    with open(merged / "meta.json") as f:
        meta = json.load(f)
    check("memmap merges inputs/targets as arrays",
          {"input_mz_matrices", "target_intensity_matrices"} <= set(meta["merge_keys"]))
    check("memmap keeps tokens_charge* as scalars",
          {"tokens_charge1", "tokens_charge2", "tokens_charge3"} <= set(meta["scalar_keys"]))
    check("memmap tokens_charge1 == token_table[1]",
          np.allclose(np.load(merged / "_scalar_tokens_charge1.npy"),
                      np.asarray(tt.CHARGE_TOKENS[1], np.float32)))

    ds_mem = load_npz_data(str(merged))
    check("loader(memmap) inputs is (N,39,46,6)",
          ds_mem["inputs"].shape == (n_examples, 39, 46, 6), str(ds_mem["inputs"].shape))
    check("loader(memmap) targets is (N,39,46,6)",
          ds_mem["targets"].shape == (n_examples, 39, 46, 6), str(ds_mem["targets"].shape))

    # 4b) single-npz path
    single = tmp / "single.npz"
    r = _run("shards_to_npz.py", shard_dir, single)
    check("shards_to_npz ran", r.returncode == 0, f"exit {r.returncode}")
    if r.returncode:
        print(r.stderr[-800:])
        return 1
    ds_npz = load_npz_data(str(single))
    check("loader(npz) inputs is (N,39,46,6)",
          ds_npz["inputs"].shape == (n_examples, 39, 46, 6), str(ds_npz["inputs"].shape))
    with np.load(single, allow_pickle=True) as d:
        check("npz tokens_charge1 == token_table[1]",
              np.allclose(d["tokens_charge1"], np.asarray(tt.CHARGE_TOKENS[1], np.float32)))

    # 5) both paths identical (incl. the object-dtype seqs array)
    check("memmap and npz inputs identical",
          np.array_equal(np.asarray(ds_mem["inputs"]), np.asarray(ds_npz["inputs"])))
    check("memmap and npz targets identical",
          np.array_equal(np.asarray(ds_mem["targets"]), np.asarray(ds_npz["targets"])))
    seq_mem = [str(s) for s in list(ds_mem["seqs"])]
    seq_npz = [str(s) for s in list(ds_npz["seqs"])]
    check("npz seqs round-trip as strings",
          all(len(s) >= 8 and s.isalpha() for s in seq_npz[:10]), str(seq_npz[:3]))
    check("memmap and npz seqs identical", seq_mem == seq_npz, f"n={len(seq_npz)}")

    # 6) train one epoch straight off the memmap dir
    r = subprocess.run(
        [sys.executable, str(_ROOT / "transfomer/train.py"),
         "--train-npz", str(merged), "--val-npz", str(single), "--test-npz", str(single),
         "--output-dir", str(tmp / "run"), "--epochs", "1", "--batch-size", "16",
         "--d-model", "32", "--n-heads", "4", "--n-layers", "2", "--ff-dim", "64",
         "--num-workers", "0", "--gpu", "-1"],
        capture_output=True, text=True,
    )
    check("1-epoch training on memmap dir", r.returncode == 0, f"exit {r.returncode}")
    if r.returncode:
        print("\n".join((r.stderr or "").strip().splitlines()[-25:]))

    # 7) train straight off the *shard* dir — no merge step at all
    r = subprocess.run(
        [sys.executable, str(_ROOT / "transfomer/train.py"),
         "--train-npz", str(shard_dir), "--val-npz", str(shard_dir), "--test-npz", str(single),
         "--output-dir", str(tmp / "run_shards"), "--epochs", "1", "--batch-size", "16",
         "--d-model", "32", "--n-heads", "4", "--n-layers", "2", "--ff-dim", "64",
         "--num-workers", "0", "--gpu", "-1"],
        capture_output=True, text=True,
    )
    check("1-epoch training directly on shard dir", r.returncode == 0, f"exit {r.returncode}")
    if r.returncode:
        print("\n".join((r.stderr or "").strip().splitlines()[-25:]))

    failed = [n for n, ok in _RESULTS if not ok]
    print()
    if failed:
        print(f"FAILED ({len(failed)}/{len(_RESULTS)}): " + ", ".join(failed))
        print("workdir kept for inspection:", tmp)
        return 1
    print(f"OK — all {len(_RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
