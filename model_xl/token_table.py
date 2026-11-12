"""Canonical V20 token vocabulary for the (39, 46, 6) intensity grid.

This module is the single source of truth for the mass-offset vocabulary.  It
is imported (never re-declared) by both sides of the pipeline:

  * ``dataset/``   — MGF token filtering, input-grid / target building, NPZ
  * ``transfomer/`` — inference grid building, candidate scoring, postprocess

The offset table is generated from the exact (un-rounded) residue masses at
``decimals=6``, which makes the *filter* vocabulary and the *model grid*
vocabulary the same table.  Rounding to 1 decimal — the historical 44-column
table — collapsed Q (128.058578) and K (128.094963) into one 128.1 column; at
6 decimals they stay distinct, giving 19 residue-mass pairs + 8 base offsets
= 46 columns.

Column layout (charge 1, ascending):
    0-18   residue_mass * -1   (W,Y,R,F,H,M,E,K,Q,D,N,I/L,C,T,V,P,S,A,G)
    19-23  -(H2O+CO), -(NH3+CO), -CO, -H2O, -NH3     [satellite]
    24     0.0                                        [main]
    25-26  +13C, +2x13C                               [satellite]
    27-45  residue_mass * +1   (G,A,S,P,V,T,C,I/L,N,D,Q,K,E,M,H,F,R,Y,W)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical constants (V20 mass table)
# ---------------------------------------------------------------------------
PROTON = 1.007276466812
H2O = 18.0105646837
NH3 = 17.02654910101
CO = 27.99491461957
ISOTOPE_DIFF = 1.0033548378  # 13C - 12C

AA_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.094963050,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}

# Base empirical offsets, defined at charge 1: the no-offset main peak plus the
# neutral losses and isotope peaks that sit next to it.
BASE_MASS_TOKENS = [
    0.0,
    -H2O,
    -NH3,
    -CO,
    -(H2O + CO),
    -(NH3 + CO),
    ISOTOPE_DIFF,
    2 * ISOTOPE_DIFF,
]

# ---------------------------------------------------------------------------
# Grid shape
# ---------------------------------------------------------------------------
MAX_PEPTIDE_LEN = 40
MAX_CLEAVAGE_LEN = MAX_PEPTIDE_LEN - 1  # 39
NUM_CHANNELS = 6
CHANNEL_NAMES = ("B1+", "B2+", "B3+", "Y1+", "Y2+", "Y3+")

TOKEN_DECIMALS = 6


def build_charge_tokens(decimals: int = TOKEN_DECIMALS, max_charge: int = 3) -> dict[int, list[float]]:
    """Build the per-charge offset table.

    Offsets are defined at charge 1 and divided by the fragment charge, so each
    charge table is rounded from the *exact* value independently — it is not the
    charge-1 table divided after rounding.
    """
    unique_residue_masses = sorted(set(AA_MASS.values()))
    charge1 = list(BASE_MASS_TOKENS)
    for mass in unique_residue_masses:
        charge1.append(mass)
        charge1.append(-mass)

    return {
        z: sorted({round(token / z, decimals) for token in charge1})
        for z in range(1, max_charge + 1)
    }


CHARGE_TOKENS = build_charge_tokens()
NUM_TOKENS = len(CHARGE_TOKENS[1])  # 46


def _index_of(offset: float, tokens: list[float] | None = None) -> int:
    """Resolve an exact 1+ offset to its column index in the charge-1 table."""
    if tokens is None:
        tokens = CHARGE_TOKENS[1]
    target = round(float(offset), TOKEN_DECIMALS)
    matches = [i for i, value in enumerate(tokens) if abs(value - target) < 1e-9]
    if len(matches) != 1:
        raise ValueError(f"token offset {offset} resolved to {len(matches)} columns")
    return matches[0]


# ---------------------------------------------------------------------------
# Token families — resolved positionally so they cannot drift from the table.
# ---------------------------------------------------------------------------
MAIN_TOKEN_IDS = (_index_of(0.0),)

SATELLITE_OFFSETS = (
    -(H2O + CO),
    -(NH3 + CO),
    -CO,
    -H2O,
    -NH3,
    ISOTOPE_DIFF,
    2 * ISOTOPE_DIFF,
)
SATELLITE_TOKEN_IDS = tuple(_index_of(offset) for offset in SATELLITE_OFFSETS)

RESIDUE_EVENT_TOKEN_IDS = tuple(
    i
    for i in range(NUM_TOKENS)
    if i not in MAIN_TOKEN_IDS and i not in SATELLITE_TOKEN_IDS
)

# Human-readable column semantics for the non-residue columns.
TOKEN_NAMES = {
    _index_of(0.0): "main",
    _index_of(-CO): "CO",
    _index_of(-H2O): "H2O",
    _index_of(-NH3): "NH3",
    _index_of(-(H2O + CO)): "H2O+CO",
    _index_of(-(NH3 + CO)): "NH3+CO",
    _index_of(ISOTOPE_DIFF): "13C",
    _index_of(2 * ISOTOPE_DIFF): "2x13C",
}
