"""DS pipeline shared helpers + path bootstrap.

Importing this module puts ``v20_0701/transfomer`` and ``.../transfomer/scripts``
on ``sys.path`` so the sibling packages/modules (``predict_mgf``,
``spectrum_transformer``, ``candidate_rescoring``) are importable, then
re-exports the constants/utilities the step scripts need.

Server layout assumption (defaults):
    V1_SEARCH_ROOT=/home/wangyuhang/seq2mass/v1-search
    CKPT=/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import lru_cache
from itertools import product
from pathlib import Path

_DS_DIR = Path(__file__).resolve().parent
_TRANSFORMER_DIR = _DS_DIR.parent
_SCRIPTS_DIR = _TRANSFORMER_DIR / "scripts"
for _p in (_TRANSFORMER_DIR, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if os.name == "nt":
    try:
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass

# IMPORTANT: keep this module pandas-free.  Importing predict_mgf registers a
# numpy._core aliasing shim, and on some numpy 1.x builds that corrupts a later
# `import pandas` (SystemError).  All ds scripts therefore avoid pandas entirely
# (csv/json/gzip only) and do NOT import candidate_rescoring.
import numpy as np  # noqa: E402

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False


def progress(iterable=None, **kwargs):
    """tqdm wrapper with a silent fallback when tqdm is unavailable."""
    if _HAS_TQDM:
        return _tqdm(iterable, **kwargs)
    return iterable

# Reuse the V20 model primitives — the authoritative mass table and parsers.
from predict_mgf import (  # noqa: E402
    AA_MASS,
    H2O,
    MAX_CLEAVAGE_LEN,
    NUM_CHANNELS,
    NUM_TOKENS,
    PROTON,
    build_input_mz_matrix,
    build_valid_mask,
    calculate_pepmass,
    load_checkpoint,
    prediction_to_mgf_peaks,
    prepare_model_input,
)
from predict_mgf import encode_cleavage_context  # noqa: E402
from spectrum_transformer.data.dataset import parse_modified_sequence  # noqa: E402

DEFAULT_MIN_MZ = 100.0
DEFAULT_RT = 1800.0

# Training target normalization (dataset/trainingExamples.py normalize_target_intensity,
# method="sqrt_clip", clip_value=50000): x -> sqrt(min(x, 50000)) / sqrt(50000).
# The model output (softplus*backbone_gain) is trained against these sqrt-normalized
# targets, so experimental intensities should be mapped into the SAME domain before
# any intensity-weighted scoring (spectral cosine, explained intensity).
EXP_CLIP_INTENSITY = 50000.0


def transform_exp_intensity(peaks: np.ndarray, method: str = "sqrt_clip",
                            clip_value: float = EXP_CLIP_INTENSITY,
                            eps: float = 1e-8) -> np.ndarray:
    """Map experimental peak intensities into the model's training domain.

    ``method`` mirrors the dataset target normalizations:
      "none"      -> unchanged (raw linear intensity)
      "sqrt_clip" -> sqrt(min(x, clip_value)) / sqrt(clip_value)   (V20 default)
      "sqrt"      -> sqrt(x)                                       (unbounded)
    The m/z column is left untouched; returns a new array.
    """
    if peaks.size == 0 or method == "none":
        return peaks
    out = peaks.copy()
    x = out[:, 1].astype(np.float32)
    if method == "sqrt_clip":
        x = np.sqrt(np.minimum(x, np.float32(clip_value))) / (np.sqrt(np.float32(clip_value)) + eps)
    elif method == "sqrt":
        x = np.sqrt(np.maximum(x, np.float32(0.0)))
    else:
        raise ValueError(f"unknown exp transform: {method}")
    out[:, 1] = x
    return out

SERVER_V1_SEARCH_ROOT = "/home/wangyuhang/seq2mass/v1-search"
SERVER_CKPT = "/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt"


# ---------------------------------------------------------------------------
# MGF parsing helpers (ported from candidate_rescoring / mgf_row_policy)
# ---------------------------------------------------------------------------

def count_mgf_spectra(mgf_path) -> int:
    """Count spectra (BEGIN IONS blocks) in an MGF file."""
    n = 0
    with open(mgf_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip() == "BEGIN IONS":
                n += 1
    return n


def parse_charge_from_meta(meta: dict) -> int:
    ch = meta.get("CHARGE", "").strip().upper()
    m = re.search(r"(\d+)", ch)
    return int(m.group(1)) if m else 0


def parse_pepmass_from_meta(meta: dict) -> float:
    pm = meta.get("PEPMASS", "").strip()
    if not pm:
        return 0.0
    parts = pm.split()
    try:
        return float(parts[0])
    except ValueError:
        return 0.0


def iter_mgf_spectra(mgf_path, max_scans=None, load_peaks=True):
    """Yield ``(meta, peaks)`` per spectrum of one MGF file, in file order.

    ``max_scans`` limits the number of blocks returned.  Meta keys are
    uppercased.  Peaks are sorted by m/z.
    """
    count = 0
    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as fh:
        block: list[str] = []
        in_block = False
        for line in fh:
            s = line.strip()
            if s == "BEGIN IONS":
                in_block = True
                block = [line]
                continue
            if not in_block:
                continue
            block.append(line)
            if s != "END IONS":
                continue

            meta: dict[str, str] = {}
            plist: list[tuple[float, float]] = []
            for bl in block:
                st = bl.strip()
                if not st:
                    continue
                if "=" in st and not st.upper().startswith(("BEGIN IONS", "END IONS")):
                    k, v = st.split("=", 1)
                    meta[k.upper()] = v.strip()
                    continue
                parts = st.split()
                if load_peaks and len(parts) == 2:
                    try:
                        plist.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
            peaks = (
                np.array(plist, dtype=np.float32)
                if plist
                else np.empty((0, 2), dtype=np.float32)
            )
            if peaks.size > 0:
                peaks = peaks[peaks[:, 0].argsort()]
            count += 1
            yield meta, peaks
            block = []
            in_block = False
            if max_scans is not None and count >= max_scans:
                return


# ---------------------------------------------------------------------------
# Mass / modification helpers
# ---------------------------------------------------------------------------

def compute_neutral_mass_annotated(annoseq: str) -> float:
    """Neutral mass (Da) of an annotated sequence, using the V20 mass table."""
    residues, deltas, nterm = parse_modified_sequence(annoseq)
    mass = sum(AA_MASS.get(aa, 0.0) + d for aa, d in zip(residues, deltas)) + nterm
    return mass + H2O


def neutral_from_mz(mz: float, charge: int) -> float:
    """Convert precursor m/z to neutral mass (Da)."""
    return mz * charge - charge * PROTON


MOD_TOKEN_RE = re.compile(r"([A-Z])([+-]\d+(?:\.\d+)?)")
NTERM_TOKEN_RE = re.compile(r"^[^A-Z]*([+-]\d+(?:\.\d+)?)")


def extract_mod_tokens(seq: str) -> tuple[list[tuple[str, float]], list[float]]:
    """Return (residue tokens, nterm tokens) from an annotated sequence."""
    res_tokens = [
        (m.group(1), float(m.group(2))) for m in MOD_TOKEN_RE.finditer(seq)
    ]
    nterm_match = NTERM_TOKEN_RE.match(seq)
    nterm = [float(nterm_match.group(1))] if nterm_match else []
    return res_tokens, nterm


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two equal-length vectors, 0.0 when either is degenerate.

    Ported verbatim from candidate_rescoring._safe_cosine (kept local so the
    ds pipeline does not need to import the pandas-heavy candidate_rescoring).
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def project_experimental_to_grid(
    exp_peaks: np.ndarray,
    input_mz: np.ndarray,
    valid_mask: np.ndarray,
    mz_tol: float = 0.05,
) -> np.ndarray:
    """Project experimental peaks onto a candidate's (39,46,6) grid.

    Ported verbatim from candidate_rescoring.project_experimental_to_grid:
    each peak maps to its nearest valid grid slot within ``mz_tol`` and the
    slot intensity is max-aggregated.
    """
    grid = np.zeros((MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS), dtype=np.float32)
    if exp_peaks.size == 0:
        return grid

    valid_rc = np.argwhere(valid_mask > 0)
    if valid_rc.size == 0:
        return grid
    slot_mz = np.array([input_mz[p, t, c] for p, t, c in valid_rc], dtype=np.float64)
    sort_idx = np.argsort(slot_mz)
    valid_rc = valid_rc[sort_idx]
    slot_mz = slot_mz[sort_idx]

    for mz, intensity in exp_peaks:
        idx = int(np.searchsorted(slot_mz, mz))
        best_dist = float("inf")
        best_i = -1
        for ni in (idx - 1, idx, idx + 1):
            if 0 <= ni < len(slot_mz):
                d = abs(float(slot_mz[ni]) - mz)
                if d <= mz_tol and d < best_dist:
                    best_dist = d
                    best_i = ni
        if best_i < 0:
            continue
        p, t, c = valid_rc[best_i]
        if float(intensity) > grid[p, t, c]:
            grid[p, t, c] = float(intensity)

    return grid


def frag_precision(exp_peaks: np.ndarray, theo_peaks: np.ndarray,
                   tol: float = 0.05) -> float:
    """Fraction of predicted (theoretical) peaks that have an experimental peak.

    This is the PRECISION-oriented counterpart of ``explained_intensity``
    (which is recall-oriented): it penalizes a candidate that predicts many
    fragments which do NOT appear in the experiment.  The true peptide's
    predicted fragments are its own (all present) -> high precision; a decoy
    that over-predicts spurious fragments -> low precision.
    """
    if theo_peaks.size == 0 or exp_peaks.size == 0:
        return 0.0
    exp_mz = exp_peaks[:, 0]
    theo_mz = theo_peaks[:, 0]
    lo = np.searchsorted(exp_mz, theo_mz - tol, side="left")
    hi = np.searchsorted(exp_mz, theo_mz + tol, side="right")
    return float((hi > lo).mean())


def peak_cosine(exp_peaks: np.ndarray, theo_peaks: np.ndarray, tol: float = 0.05) -> float:
    """Cosine between a real peak list and a theoretical peak list.

    Uses the theoretical peaks' m/z as the axis; for each theoretical peak the
    experimental intensity is the max experimental peak intensity within ``tol``.
    """
    if theo_peaks.shape[0] == 0 or exp_peaks.shape[0] == 0:
        return 0.0
    exp_mz = exp_peaks[:, 0]
    exp_int = exp_peaks[:, 1]
    theo_mz = theo_peaks[:, 0]
    lo = np.searchsorted(exp_mz, theo_mz - tol, side="left")
    hi = np.searchsorted(exp_mz, theo_mz + tol, side="right")
    matched = np.zeros(theo_mz.shape[0], dtype=np.float32)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            matched[i] = exp_int[a:b].max()
    return _safe_cosine(matched, theo_peaks[:, 1].astype(np.float32))


def explained_intensity(exp_peaks: np.ndarray, theo_peaks: np.ndarray,
                        tol: float = 0.05) -> float:
    """Fraction of experimental total intensity explained by predicted peaks.

    Each experimental peak counts toward the explained fraction if at least one
    predicted (theoretical) peak lies within ``tol`` Da of it.  The true
    peptide's predicted fragments cover most of the observed spectrum -> high
    score; a wrong peptide covers fewer -> low score.  This is a
    database-search-style fragment-matching score (as opposed to full-grid
    cosine).
    """
    if exp_peaks.size == 0 or theo_peaks.size == 0:
        return 0.0
    exp_mz = exp_peaks[:, 0]
    exp_int = exp_peaks[:, 1]
    total = float(exp_int.sum())
    if total <= 0.0:
        return 0.0
    # theo_peaks must be m/z-sorted (prediction_to_mgf_peaks output is sorted).
    theo_mz = theo_peaks[:, 0]
    lo = np.searchsorted(theo_mz, exp_mz - tol, side="left")
    hi = np.searchsorted(theo_mz, exp_mz + tol, side="right")
    matched = hi > lo
    return float(exp_int[matched].sum() / total)


# ---------------------------------------------------------------------------
# GT matching — ported verbatim from candidate_rescoring.py (V20 rescoring
# convention): AA-mass prefix/suffix matching (0.5 Da cumulative, 0.1 Da per
# token), with variant-expansion + regex fallbacks.  I/L are identical mass so
# they match naturally (I/L-equivalent); mod precision and bracket/inline
# N-term formats are normalized during tokenization.
# ---------------------------------------------------------------------------

_AA_MASSES = {
    "G": 57.021463735, "A": 71.037113805, "S": 87.032028435,
    "P": 97.052763875, "V": 99.068413945, "T": 101.047678505,
    "C+57.021": 103.009184505 + 57.02146,
    "L": 113.084064015, "I": 113.084064015, "N": 114.042927470,
    "D": 115.026943065, "Q": 128.058577540, "K": 128.094963050,
    "E": 129.042593135, "M": 131.040484645, "H": 137.058911875,
    "F": 147.068413945, "R": 156.101111050, "Y": 163.063328575,
    "W": 186.079312980,
    "+42.011": 42.010565, "+43.006": 43.005814, "-17.027": -17.026549,
    "+43.006-17.027": 43.006814 - 17.026549,
    "M+15.995": 131.040484645 + 15.994915,
    "N+0.984": 114.042927470 + 0.984016, "Q+0.984": 128.058577540 + 0.984016,
}

_PRECISION_MAP: dict[str, str] = {}
for _tok, _mass in _AA_MASSES.items():
    _m = re.match(r"^([A-Z]?)([+-]?\d+\.\d{3,})$", _tok)
    if not _m:
        continue
    _prefix = _m.group(1)
    _mod_val = float(_m.group(2))
    _rounded = round(_mod_val, 2)
    _sign = "-" if _rounded < 0 else "+"
    _abs_val = abs(_rounded)
    _int_part = int(_abs_val)
    _frac_part = int(round((_abs_val - _int_part) * 100))
    _variant = f"{_prefix}{_sign}{_int_part}.{_frac_part:02d}"
    if _variant != _tok and _variant not in _AA_MASSES:
        _PRECISION_MAP[_variant] = _tok

_TOKEN_OPTIONS: dict[str, tuple[str, ...]] = {}
for _tok, _mass in _AA_MASSES.items():
    _group = tuple(sorted(
        (other for other, other_mass in _AA_MASSES.items()
         if abs(other_mass - _mass) < 0.1),
        key=lambda item: (-len(item), item),
    ))
    _TOKEN_OPTIONS[_tok] = _group

_TOKEN_SPLIT_RE = re.compile(r"(?<=.)(?=[A-Z])")
_PROFORMA_BRACKET_RE = re.compile(r"([A-Z])(\[[^\]]+\])")
_NTERM_BRACKET_RE = re.compile(r"^(\[[^\]]+\])")


def _tokenize_peptide(peptide: str) -> list[str]:
    if not peptide:
        return []
    p = str(peptide).strip().replace(" ", "").replace("$", "")
    p = re.sub(r"\(([+-]?\d*\.?\d+)\)", r"\1", p)
    p = _PROFORMA_BRACKET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)[1:-1]}", p)
    p = _NTERM_BRACKET_RE.sub(lambda m: m.group(1)[1:-1], p)
    tokens = [tok for tok in _TOKEN_SPLIT_RE.split(p) if tok]
    normalized = []
    for tok in tokens:
        _m = re.match(r"^([A-Z])(\d+\.?\d*)$", tok)
        if _m:
            tok = f"{_m.group(1)}+{_m.group(2)}"
        elif re.match(r"^([A-Z])\.(\d+)$", tok):
            _dm = re.match(r"^([A-Z])\.(\d+)$", tok)
            tok = _dm.group(1) + "+0." + _dm.group(2)
        elif re.match(r"^([A-Z])\+\.(\d+)$", tok):
            _dm = re.match(r"^([A-Z])\+\.(\d+)$", tok)
            tok = _dm.group(1) + "+0." + _dm.group(2)
        elif re.match(r"^\d+\.?\d*$", tok):
            tok = f"+{tok}"
        elif re.match(r"^\.(\d+)$", tok):
            _dm = re.match(r"^\.(\d+)$", tok)
            tok = "+0." + _dm.group(1)
        elif re.match(r"^\+\.(\d+)$", tok):
            _dm = re.match(r"^\+\.(\d+)$", tok)
            tok = "+0." + _dm.group(1)
        elif re.match(r"^\-\.(\d+)$", tok):
            _dm = re.match(r"^\-\.(\d+)$", tok)
            tok = "-0." + _dm.group(1)
        tok = _PRECISION_MAP.get(tok, tok)
        _cm = re.match(r"^([A-Z]?)([+-]\d+\.?\d*)([+-]\d+\.?\d*)$", tok)
        if _cm:
            _p2 = _PRECISION_MAP.get(_cm.group(2), _cm.group(2))
            _p3 = _PRECISION_MAP.get(_cm.group(3), _cm.group(3))
            tok = _cm.group(1) + _p2 + _p3
        normalized.append(tok)
    return normalized


@lru_cache(maxsize=500_000)
def _tokenize_cached(sequence: str) -> tuple[str, ...]:
    return tuple(_tokenize_peptide(sequence))


def _token_mass_fallback(tok: str) -> float:
    if tok in _AA_MASSES:
        return _AA_MASSES[tok]
    m = re.match(r"^([A-Z])([+-]\d+\.?\d*)([+-]\d+\.?\d*)?$", tok)
    if m:
        base = _AA_MASSES.get(m.group(1), 0.0)
        mass = base + float(m.group(2))
        if m.group(3):
            mass += float(m.group(3))
        return mass
    m = re.match(r"^([+-]\d+\.?\d*)$", tok)
    if m:
        return float(m.group(1))
    return 0.0


def _aa_match_prefix_suffix(pep1: list[str], pep2: list[str]) -> tuple[list[bool], bool]:
    def _prefix(p1, p2):
        matches = [False] * max(len(p1), len(p2))
        i1, i2, cm1, cm2 = 0, 0, 0.0, 0.0
        while i1 < len(p1) and i2 < len(p2):
            m1, m2 = _token_mass_fallback(p1[i1]), _token_mass_fallback(p2[i2])
            if abs(cm1 + m1 - (cm2 + m2)) < 0.5:
                matches[max(i1, i2)] = abs(m1 - m2) < 0.1
                i1 += 1
                i2 += 1
                cm1 += m1
                cm2 += m2
            elif cm2 + m2 > cm1 + m1:
                i1 += 1
                cm1 += m1
            else:
                i2 += 1
                cm2 += m2
        return matches, all(matches)

    aa_matches, pep_match = _prefix(pep1, pep2)
    if pep_match:
        return aa_matches, pep_match
    i1, i2 = len(pep1) - 1, len(pep2) - 1
    try:
        i_stop = aa_matches.index(False)
    except ValueError:
        i_stop = 0
    cm1 = cm2 = 0.0
    while i1 >= i_stop and i2 >= i_stop:
        m1, m2 = _token_mass_fallback(pep1[i1]), _token_mass_fallback(pep2[i2])
        if abs(cm1 + m1 - (cm2 + m2)) < 0.5:
            aa_matches[max(i1, i2)] = abs(m1 - m2) < 0.1
            i1 -= 1
            i2 -= 1
            cm1 += m1
            cm2 += m2
        elif cm2 + m2 > cm1 + m1:
            i1 -= 1
            cm1 += m1
        else:
            i2 -= 1
            cm2 += m2
    return aa_matches, all(aa_matches)


def _build_variant_set(truth_tokens: tuple[str, ...]) -> frozenset | None:
    opts_per_pos = [_TOKEN_OPTIONS.get(t, (t,)) for t in truth_tokens]
    variant_count = 1
    for opts in opts_per_pos:
        variant_count *= len(opts)
        if variant_count > 1024:
            return None
    return frozenset("".join(tokens) for tokens in product(*opts_per_pos))


def peptide_matches(candidate: str, truth: str) -> bool:
    """V20-rescoring-convention GT match (mass-based, I/L-equivalent)."""
    cand_tokens = _tokenize_cached(candidate)
    truth_tokens = _tokenize_cached(truth)
    if not cand_tokens or not truth_tokens:
        return False
    _, m = _aa_match_prefix_suffix(list(cand_tokens), list(truth_tokens))
    if m:
        return True
    vs = _build_variant_set(truth_tokens)
    cand_str = "".join(cand_tokens)
    if vs is not None:
        return cand_str in vs
    parts = []
    for t in truth_tokens:
        opts = _TOKEN_OPTIONS.get(t, (t,))
        parts.append("(?:" + "|".join(re.escape(o) for o in opts) + ")")
    return bool(re.fullmatch("".join(parts), cand_str))


# ---------------------------------------------------------------------------
# Files / records
# ---------------------------------------------------------------------------

def resolve_species_files(root, species) -> tuple[Path, Path]:
    root = Path(root)
    fasta = root / species / f"for_{species}_test_inscope_peptide.fasta"
    mgf_candidates = sorted((root / species).glob(f"high.{species}.*.mgf"))
    if not mgf_candidates:
        raise FileNotFoundError(f"no MGF for {species} under {root / species}")
    return fasta, mgf_candidates[0]


def write_record(path, payload: dict) -> None:
    payload = dict(payload)
    payload["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload.setdefault("start_time", payload["end_time"])
    if "elapsed_sec" not in payload:
        payload["elapsed_sec"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


class StepTimer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return round(time.time() - self.start, 2)

    def summary(self) -> dict:
        return {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start)),
            "elapsed_sec": self.elapsed(),
        }
