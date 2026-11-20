#!/usr/bin/env python3
"""fullsearch shared helpers: sys.path bootstrap + neutral mass / backbone + IO.

Blind (no-precursor-filter) whole-library scoring needs a python that can import
ds_common / fdr_common / step2 enumerators.  This module inserts every sibling
package on sys.path.  Note: the Prosit prediction stage MUST `import pandas`
BEFORE importing anything that triggers predict_mgf (numpy alias hazard); keep it
as its own process.
"""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

_FULLSEARCH = Path(__file__).resolve().parent
_TRANSFORMER = _FULLSEARCH.parent
for _p in (_TRANSFORMER, _TRANSFORMER / "scripts", _TRANSFORMER / "ds",
           _TRANSFORMER / "fdr", _TRANSFORMER / "fdr_full"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ds_common as dc  # noqa: E402
import fdr_common as fc  # noqa: E402
from predict_mgf import AA_MASS  # noqa: E402
from step2_build_candidate_fasta import enumerate_variants, neutral_mass_with_table  # noqa: E402

MASS_TABLE = dict(AA_MASS)
FIXED_MODS = {"C": ["57.021"]}
VARIABLE_MODS = {"M": ["15.995"]}
DEFAULT_RT = dc.DEFAULT_RT
TOL = 0.05
MZ_MERGE = 0.05
SPEC_MIN = 0.01

MAX_BACKBONE_PRO = 30  # Prosit upper bound


def backbone_of(anno: str) -> str:
    residues, _deltas, _nterm = dc.parse_modified_sequence(anno)
    return "".join(residues)


def neutral_of(anno: str) -> float:
    return neutral_mass_with_table(anno, MASS_TABLE)


def read_library_tsv(path) -> list[dict]:
    """Read the annotated library rows -> [{'annoseq','backbone','label','mass','idx'}]."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            rows.append({
                "idx": int(r["idx"]),
                "annoseq": r["annoseq"],
                "backbone": r["backbone"],
                "label": r["label"],
                "mass": float(r["mass"]),
            })
    return rows
