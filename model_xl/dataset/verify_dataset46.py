#!/usr/bin/env python3
"""Verify a generated dataset is a correct 39x46x6 V20 training set.

Accepts any of the three source forms the trainer accepts:
    - a single .npz file
    - a shard directory (has index.json)
    - a merged memmap directory (has meta.json)

Checks (all must pass):
    1. token axis is 46 in the grid shape
    2. stored tokens_charge1/2/3 match token_table exactly
    3. the stored input m/z grid equals the theoretical grid recomputed from
       the stored sequences with the shared 46-token table (on sampled rows)
    4. position/channel masks line up with the recorded lengths/charges
    5. reports N, unique (seq, charge) groups, and target density

Usage:
    python dataset/verify_dataset46.py <npz_or_dir> [--max-samples 64]
Exit code 0 = dataset looks correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # v20_0918/
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="single .npz / shard dir / merged memmap dir")
    ap.add_argument("--max-samples", type=int, default=64,
                    help="how many rows to recompute the grid for (default 64)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"[FAIL] path not found: {path}")
        return 1
    print(f"verifying: {path}\n")

    raw = load_npz_data(str(path))
    inputs = np.asarray(raw["inputs"])
    targets = np.asarray(raw["targets"])
    n = inputs.shape[0]

    # ---- 1. shape / token axis -------------------------------------------
    check("grid is (N, 39, 46, 6)", inputs.shape[1:] == (39, 46, 6), str(inputs.shape))
    check("targets match inputs shape", targets.shape == inputs.shape, str(targets.shape))
    if inputs.shape[1:] != (39, 46, 6):
        print("\nThis is NOT a 46-column dataset — regenerate it with the v20_0918 pipeline.")
        return 1
    check("has samples", n > 0, f"N={n}")

    # ---- 2. stored token tables ------------------------------------------
    for z in (1, 2, 3):
        key = f"tokens_charge{z}"
        if key in raw:
            stored = np.asarray(raw[key], np.float32)
            want = np.asarray(tt.CHARGE_TOKENS[z], np.float32)
            check(f"{key} == token_table[{z}]",
                  stored.shape == (46,) and np.allclose(stored, want),
                  f"shape={stored.shape}")
        else:
            check(f"{key} present", False, "missing from dataset")
    # main column offset must be exactly 0
    check("main column (idx 24) is the no-offset peak",
          abs(float(tt.CHARGE_TOKENS[1][tt.MAIN_TOKEN_IDS[0]])) < 1e-12,
          f"MAIN_TOKEN_IDS={tt.MAIN_TOKEN_IDS}")

    # ---- 3. recompute the theoretical grid and compare --------------------
    seqs = raw.get("seqs", raw.get("sequences"))
    if seqs is None:
        check("seqs present (needed to re-derive the grid)", False)
        return 1
    seqs = [str(s) for s in list(seqs)]
    k = min(int(args.max_samples), n)
    idxs = np.linspace(0, n - 1, k).astype(int)

    bad_rows = []
    for i in idxs:
        try:
            ref = te.build_input_mz_matrix(seqs[i])
        except Exception as exc:  # noqa: BLE001
            bad_rows.append((int(i), f"recompute failed: {exc}"))
            continue
        if not np.allclose(inputs[i], ref, atol=1e-4):
            d = float(np.abs(inputs[i] - ref).max())
            bad_rows.append((int(i), f"max|diff|={d:.4f}"))
    check(f"input m/z grid == theory from seqs ({k} sampled rows)",
          not bad_rows, "; ".join(f"row{r}: {m}" for r, m in bad_rows[:3]) or "exact")

    # ---- 4. masks vs recorded lengths / charges ---------------------------
    if "position_masks" in raw and "channel_masks" in raw:
        pos = np.asarray(raw["position_masks"])
        ch = np.asarray(raw["channel_masks"])
        lengths = np.asarray(raw["lengths"])
        charges = np.asarray(raw["charges"])
        check("position_masks is (N, 39)", pos.shape == (n, 39), str(pos.shape))
        check("channel_masks is (N, 6)", ch.shape == (n, 6), str(ch.shape))
        # `lengths` stores the valid-cleavage-row count (L-1), per build_position_mask.
        exp_pos = np.stack([np.arange(39) < int(L) for L in lengths]).astype(np.float32)
        check("position_masks == first `lengths` rows valid", np.array_equal(pos, exp_pos))
        exp_ch = np.stack([
            np.array([1, 1, 1, 1, 1, 1] if int(c) >= 3 else
                     ([1, 1, 0, 1, 1, 0] if int(c) == 2 else [1, 0, 0, 1, 0, 0]),
                     dtype=np.float32)
            for c in charges
        ])
        check("channel_masks matches precursor charge", np.array_equal(ch, exp_ch),
              f"charges seen: {sorted(set(int(c) for c in charges))}")

    # ---- 5. summary -------------------------------------------------------
    charges = np.asarray(raw.get("charges", np.zeros(n)))
    groups = len({(s, int(c)) for s, c in zip(seqs, charges)})
    tz = float((targets > 0).mean())
    mx = float(np.asarray(targets).max())
    print()
    print(f"  samples            : {n}")
    print(f"  unique (seq,charge): {groups}")
    if groups and n / max(groups, 1) < 1.05:
        print(f"  note               : avg spectra per group = {n / groups:.2f}")
        print("                       ~1 means each group had a single spectrum —")
        print("                       fine for a smoke test, but real training data")
        print("                       should come from averaged groups (step 2).")
    print(f"  target density     : {tz:.4f}  (fraction of non-zero cells)")
    print(f"  target max         : {mx:.4f}")

    failed = [name for name, ok in _RESULTS if not ok]
    print()
    if failed:
        print(f"FAILED ({len(failed)}/{len(_RESULTS)}): " + ", ".join(failed))
        return 1
    print(f"OK — {len(_RESULTS)}/{len(_RESULTS)} checks passed; dataset looks correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
