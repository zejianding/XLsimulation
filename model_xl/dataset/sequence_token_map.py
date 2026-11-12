"""Sequence-constrained target mask for the (39, 46, 6) grid.

Motivation
----------
``build_target_intensity_matrix`` fills a grid cell whenever an experimental
peak lies within ``tol`` of that cell's m/z.  Two cells can therefore both be
filled by ONE peak when their m/z positions are closer than ``2*tol``.  On this
vocabulary that happens for exactly one column pair — (+Q, +K) / (-K, -Q),
whose 1+ spacing is 0.036 Da (< 2*0.05) — so every Q/K residue in the sequence
also lights up the *other* residue's column (measured: 97.9% / 95.1% of the
time the correct column fires).

Since the peptide sequence is known when the training target is built, the
legitimate columns can be decided exactly: a residue-offset column at
``(row, channel)`` is legitimate **iff its m/z coincides with another real b/y
ion of the same channel** — i.e. it is one of the neighbouring ions
(``b_{r+2} = b_{r+1} + R[r+1]``, ``b_r = b_{r+1} - R[r]`` and the y-side
mirror).  Verified on real data: the sequence-predicted column fires 4.6x-13.2x
more often than the other residue columns in its row.

The base columns (main + the 7 neutral-loss / isotope satellites) are *not*
residue offsets and are always allowed.

This module is deliberately standalone: ``trainingExamples.build_target_intensity_matrix``
only calls ``allowed_target_mask`` and multiplies, so the feature can be
switched off without touching the grid logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = next(p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from token_table import (  # noqa: E402
    CHARGE_TOKENS,
    MAIN_TOKEN_IDS,
    MAX_CLEAVAGE_LEN,
    NUM_CHANNELS,
    NUM_TOKENS,
    SATELLITE_TOKEN_IDS,
)

# Main + satellites are not residue offsets: always legitimate.
_BASE_TOKENS = frozenset(MAIN_TOKEN_IDS) | frozenset(SATELLITE_TOKEN_IDS)

# Two theoretical m/z count as "the same position" within this tolerance.
# Far below the 0.036 Da Q/K gap, so Q never matches K by accident.
_MZ_EQ_TOL = 1e-4

# Columns that are ambiguous *only* because their m/z spacing is below 2*tol.
# On this vocabulary that is exactly the (+Q, +K) and (-K, -Q) pairs; the next
# smallest column gap is 0.9476 Da >> 2*tol = 0.1.
def _index_of(offset: float) -> int:
    return CHARGE_TOKENS[1].index(round(offset, 6))


_QK_TOKENS = frozenset(
    _index_of(o) for o in (128.058577540, 128.094963050, -128.058577540, -128.094963050)
)

# Modes accepted by ``allowed_target_mask``:
#   "qk"   (推荐) 只对 Q/K 四列做序列约束 —— 精准消除唯一的列间歧义，其余 42 个残基列
#          保持原行为（那些列间距 ≥ 0.9476 Da，本来就不重叠）。
#   "full" 对所有残基偏移列都要求「m/z 与同通道内某个真实 b/y 离子重合」。更严格，
#          但会连带抹掉大量「位置并非任何真实碎片」的巧合匹配（实测标签 −55%）。
MASK_MODES = ("qk", "full")

_MASK_CACHE: dict[tuple[str, str], np.ndarray] = {}


def allowed_target_mask(seq: str, mode: str = "qk") -> np.ndarray:
    """Return a ``(39, 46, 6)`` boolean mask of legitimate grid positions.

    ``True``  = the cell may be filled from the experimental spectrum.
    ``False`` = a residue-offset column the sequence does not justify (e.g. the
    ``+K`` column of a peptide whose next residue is ``Q``).
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mode must be one of {MASK_MODES}, got {mode!r}")
    key = (seq, mode)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    from trainingExamples import generate_by_ions_with_mods

    ions = generate_by_ions_with_mods(seq, max_charge=3)
    per_channel: list[list[float]] = [[] for _ in range(NUM_CHANNELS)]
    for ion in ions:
        ch = (ion["charge"] - 1) if ion["ion_type"] == "B" else (ion["charge"] + 2)
        per_channel[ch].append(float(ion["mz"]))
    pools = [np.asarray(v, dtype=np.float64) for v in per_channel]

    mask = np.zeros((MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=bool)
    for ion in ions:
        row = ion["cleavage_index"] - 1
        if row >= MAX_CLEAVAGE_LEN:
            continue
        z = ion["charge"]
        ch = (z - 1) if ion["ion_type"] == "B" else (z + 2)
        pool = pools[ch]
        base_mz = float(ion["mz"])
        for t, token in enumerate(CHARGE_TOKENS[z]):
            if t in _BASE_TOKENS:
                mask[row, t, ch] = True
            elif mode == "qk" and t not in _QK_TOKENS:
                mask[row, t, ch] = True          # 该列不歧义，保持原行为
            else:
                # 仅当该列 m/z 与同通道内某个真实 b/y 离子重合（= 相邻离子）才允许
                mz = base_mz + token
                if pool.size and float(np.min(np.abs(pool - mz))) <= _MZ_EQ_TOL:
                    mask[row, t, ch] = True

    _MASK_CACHE[key] = mask
    return mask


def apply_mask(target: np.ndarray, seq: str, mode: str = "qk") -> np.ndarray:
    """Zero out every cell that the sequence does not justify."""
    return target * allowed_target_mask(seq, mode=mode).astype(target.dtype)
