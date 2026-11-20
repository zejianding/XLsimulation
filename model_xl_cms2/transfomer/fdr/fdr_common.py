"""FDR analysis shared helpers (target/decoy, grid prediction, scoring, FDR math).

This module mirrors the ds/ shared-helper style: it imports the stable V20
primitives from ``ds_common`` / ``predict_mgf`` and adds only the small glue the
fdr/ step scripts need.  It stays pandas-free (ds convention) and self-contained
(no import of the ds step scripts), so the fdr pipeline is an independent layer
built on the same primitives.

Main concepts
-------------
target  : a peptide from the species FASTA (C-carbamidomethylated, V20 mass table)
decoy   : the reversed target (same neutral mass, guaranteed not to exist)
          -> reversal preserves the exact total mass, so a ±ppm precursor window
             retrieves target and decoy candidates in roughly equal numbers.
scoring : per-candidate V20-predicted spectrum vs. the experimental spectrum:
            gc      grid cosine        (39,46,6) pred grid vs exp-projected grid
            d       spectral cosine    exp projected onto predicted peaks
            f_bidir spectral cosine    predicted projected onto experimental peaks
            f_exp   explained intensity fraction of exp intensity covered
            mby     main b/y-ion precision (main token)
          each in [0,1], so a raw equal-weight mean is cross-spectrum comparable.
FDR     : target-decoy competition.  Rank spectra by the top-1 fused raw score,
          cumulative decoy/target ratio (Elias-Gygi) + q-values.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_DS_DIR = Path(__file__).resolve().parent.parent / "ds"
_TRANSFORMER_DIR = _DS_DIR.parent
_SCRIPTS_DIR = _TRANSFORMER_DIR / "scripts"
for _p in (_TRANSFORMER_DIR, _SCRIPTS_DIR, _DS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if sys.platform == "win32" or sys.platform == "nt":
    try:
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass

import ds_common as dc  # noqa: E402
from predict_mgf import (  # noqa: E402
    MAX_CLEAVAGE_LEN,
    NUM_CHANNELS,
    NUM_TOKENS,
    calculate_pepmass,
    load_checkpoint,
    prediction_to_mgf_peaks,
)

# Token vocabulary — the single source of truth lives in <project_root>/token_table.py.
_TOKEN_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_TOKEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOKEN_ROOT))

from token_table import MAIN_TOKEN_IDS  # noqa: E402

SERVER_V1_SEARCH_ROOT = dc.SERVER_V1_SEARCH_ROOT
SERVER_CKPT = dc.SERVER_CKPT

# Fixed carbamidomethylation applied to every C (the only fixed mod observed in
# the 9-species MGFs).  Matches ds_common._AA_MASSES["C+57.021"] = 103.009184505 + 57.02146.
C_MOD = 57.02146

# Constants reused for grid cosine.
MAIN_TOKEN_ID = MAIN_TOKEN_IDS[0]  # token index of the main (no-offset) peak
DEFAULT_TOL = 0.05  # m/z match tolerance (Da), consistent with ds/step4

# ---------------------------------------------------------------------------
# Annotated sequence helpers (V20 format: C+57.021 on every C)
# ---------------------------------------------------------------------------


def annotate_plain(seq: str) -> str:
    """Return the V20 annotated form of a plain sequence (C -> C+57.021)."""
    return "".join("C+57.021" if aa == "C" else aa for aa in seq)


def reverse_annotated(annoseq: str) -> str:
    """Reverse an annotated sequence, keeping each mod attached to its residue.

    Returns None when the sequence carries an N-terminal modification (V20's
    annotated format cannot represent it on the C-terminus of the reversed
    peptide).  Total neutral mass is invariant under reversal.
    """
    residues, deltas, nterm = dc.parse_modified_sequence(annoseq)
    if abs(float(nterm)) > 1e-9:
        return None
    out: list[str] = []
    for aa, d in reversed(list(zip(residues, deltas))):
        out.append(aa)
        if abs(float(d)) > 1e-9:
            out.append(f"{d:+.3f}")
    return "".join(out)


def neutral_mass(annoseq: str) -> float:
    """Neutral mass (Da) of an annotated sequence (V20 AA_MASS + H2O + mods)."""
    residues, deltas, nterm = dc.parse_modified_sequence(annoseq)
    return sum(dc.AA_MASS.get(aa, 0.0) + d for aa, d in zip(residues, deltas)) + nterm + dc.H2O


def neutral_plain(seq: str) -> float:
    """Neutral mass of a plain sequence with fixed C+57.021 (equality check)."""
    return neutral_mass(annotate_plain(seq))


# ---------------------------------------------------------------------------
# Target/decoy database
# ---------------------------------------------------------------------------

LABEL_TARGET = "T"
LABEL_DECOY = "D"


def build_target_decoy(plain_seqs: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (annoseqs, labels, masses) for targets + reversed decoys.

    - targets: unique plain sequences, C-annotated
    - decoys : reversed target, C-annotated; dropped when the reversed sequence
      collides with an already-present target (palindromes / DB contains both a
      sequence and its reverse) so no decoy is ever a valid target.
    """
    targets: list[str] = []
    seen: set[str] = set()
    for s in plain_seqs:
        a = annotate_plain(s)
        if a not in seen:
            seen.add(a)
            targets.append(a)

    target_set = set(targets)
    decoys: list[str] = []
    n_collision = 0
    for a in targets:
        r = reverse_annotated(a)
        if r is None:
            continue
        if r in target_set or r in decoys:
            n_collision += 1
            continue
        decoys.append(r)

    annoseqs = targets + decoys
    labels = [LABEL_TARGET] * len(targets) + [LABEL_DECOY] * len(decoys)
    masses = [neutral_mass(a) for a in annoseqs]
    return annoseqs, labels, masses


def load_td_fasta(fasta_path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load the target+decoy FASTA -> (mass_arr float64, annoseq list, label_arr).

    Header format written by step1:
        >TD_TGT_<i> MASS=<neutral:.6f> LABEL=T SEQ=<annoseq>
        >TD_DEC_<i> MASS=<neutral:.6f> LABEL=D SEQ=<annoseq>
    """
    masses: list[float] = []
    seqs: list[str] = []
    labels: list[str] = []
    with open(fasta_path, encoding="utf-8") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                header = line[1:]
                continue
            if header is None:
                continue
            parts = header.split()
            mass = float(parts[1][5:])
            label = parts[2][6:]
            seq = line.strip()
            masses.append(mass)
            seqs.append(seq)
            labels.append(label)
            header = None
    return (
        np.asarray(masses, dtype=np.float64),
        seqs,
        np.asarray(labels, dtype=object),
    )


# ---------------------------------------------------------------------------
# Batch grid prediction (compact re-implementation of ds/step4.predict_candidate_grids)
# ---------------------------------------------------------------------------


def predict_candidate_grids(
    model,
    model_config: dict,
    stats: dict,
    device,
    candidates: list[str],
    charge: int,
    rt: float,
    min_mz: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred, input_mz_grids, valid_masks), each (K,39,46,6)."""
    import torch

    use_ctx = bool(model_config.get("use_sequence_context", True))
    expected_in = int(model_config.get("in_channels", dc.NUM_CHANNELS))
    K = len(candidates)

    input_all = np.zeros((K, dc.MAX_CLEAVAGE_LEN, dc.NUM_TOKENS, dc.NUM_CHANNELS), dtype=np.float32)
    norm_inputs = np.zeros_like(input_all)
    valid_masks = np.zeros_like(input_all)
    lengths = np.zeros(K, dtype=np.int64)
    aa_ids = np.zeros((K, dc.MAX_CLEAVAGE_LEN, 2), dtype=np.int64)
    scalar = np.zeros((K, dc.MAX_CLEAVAGE_LEN, 7), dtype=np.float32)
    pepmasses = np.zeros(K, dtype=np.float32)

    for i, seq in enumerate(candidates):
        im = dc.build_input_mz_matrix(seq, min_mz=min_mz)
        vm, cleav = dc.build_valid_mask(seq, charge, min_mz=min_mz)
        ni = dc.prepare_model_input(im, vm, stats, expected_in)[0]
        input_all[i] = im
        valid_masks[i] = vm
        norm_inputs[i] = ni
        lengths[i] = cleav + 1
        pepmasses[i] = calculate_pepmass(seq, charge)
        if use_ctx:
            a, s = dc.encode_cleavage_context(seq, int(cleav + 1), max_cleavage_len=dc.MAX_CLEAVAGE_LEN)
            aa_ids[i] = a.astype(np.int64, copy=False)
            scalar[i] = s.astype(np.float32, copy=False)

    raw_aux = np.stack([np.array([charge, pm, rt], dtype=np.float32) for pm in pepmasses])
    aux = ((raw_aux - stats["aux_feature_means"])
           / np.maximum(stats["aux_feature_stds"], 1e-6)).astype(np.float32)

    pred_all = np.zeros_like(input_all)
    for start in range(0, K, batch_size):
        end = min(start + batch_size, K)
        x = torch.from_numpy(norm_inputs[start:end]).to(device)
        fw = {
            "aux_features": torch.from_numpy(aux[start:end]).to(device),
            "valid_mask": torch.from_numpy(valid_masks[start:end]).to(device),
            "charge": torch.full((end - start,), charge, dtype=torch.long, device=device),
            "length": torch.from_numpy(lengths[start:end]).to(device),
        }
        if use_ctx:
            fw["cleavage_aa_ids"] = torch.from_numpy(aa_ids[start:end]).to(device)
            fw["cleavage_scalar_features"] = torch.from_numpy(scalar[start:end]).to(device)
        with torch.inference_mode():
            out = model(x, **fw)
        pred_all[start:end] = out["prediction"].detach().cpu().numpy().astype(np.float32)
    return pred_all, input_all, valid_masks


# ---------------------------------------------------------------------------
# Per-candidate scoring
# ---------------------------------------------------------------------------


def _main_by_peaks(input_mz, pred, valid_mask, min_intensity, min_mz, mz_merge_tol):
    """Extract main b/y-ion peaks (main token only) as a sorted (m/z, int) array."""
    im = input_mz[:, MAIN_TOKEN_ID, :]
    pr = pred[:, MAIN_TOKEN_ID, :]
    vm = valid_mask[:, MAIN_TOKEN_ID, :]
    pos = np.where((vm > 0) & (pr > min_intensity) & (im >= min_mz))
    if pos[0].size == 0:
        return np.empty((0, 2), dtype=np.float32)
    mz_vals = im[pos].astype(np.float32)
    int_vals = pr[pos].astype(np.float32)
    order = np.argsort(mz_vals)
    mz_vals = mz_vals[order]
    int_vals = int_vals[order]
    merged_mz, merged_int = [], []
    i = 0
    while i < len(mz_vals):
        j = i + 1
        while j < len(mz_vals) and (mz_vals[j] - mz_vals[i]) < mz_merge_tol:
            j += 1
        best = i + int(np.argmax(int_vals[i:j]))
        merged_mz.append(float(mz_vals[best]))
        merged_int.append(float(int_vals[best]))
        i = j
    return np.column_stack([merged_mz, merged_int])


def score_candidate(
    pred: np.ndarray,
    input_mz: np.ndarray,
    valid_mask: np.ndarray,
    exp_peaks: np.ndarray,
    rel: float,
    min_mz: float,
    tol: float = DEFAULT_TOL,
    mz_merge_tol: float = DEFAULT_TOL,
    spec_min_intensity: float = 0.01,
) -> dict[str, float]:
    """Score one candidate vs. the experimental spectrum.

    Returns raw scores (all in [0,1]):
      gc      grid cosine over the (39,46,6) pred grid vs exp-projected grid
      d       peak cosine: experimental peaks projected onto predicted peaks
      f_bidir peak cosine: predicted peaks projected onto experimental peaks
      f_exp   explained intensity (fraction of exp intensity covered by pred peaks)
      mby     main b/y-ion fragment precision
    """
    v = valid_mask > 0
    pmax = float(pred.max())

    proj = dc.project_experimental_to_grid(exp_peaks, input_mz, valid_mask, mz_tol=mz_merge_tol)
    gc = dc._safe_cosine(pred[v], proj[v])

    eff = max(spec_min_intensity, rel * pmax)
    theo = prediction_to_mgf_peaks(
        input_mz, pred, valid_mask,
        min_intensity=eff, min_mz=min_mz, mz_merge_tol=mz_merge_tol,
    )
    d = dc.peak_cosine(exp_peaks, theo, tol=tol)
    f_bidir = dc.peak_cosine(theo, exp_peaks, tol=tol)
    f_exp = dc.explained_intensity(exp_peaks, theo, tol=tol)
    by = _main_by_peaks(input_mz, pred, valid_mask, eff, min_mz, mz_merge_tol)
    mby = dc.frag_precision(exp_peaks, by, tol=tol)

    return {"gc": float(gc), "d": float(d), "f_bidir": float(f_bidir),
            "f_exp": float(f_exp), "mby": float(mby)}


SCORE_COMPONENTS = ("gc", "d", "f_bidir", "f_exp", "mby")


def fused_raw(scores: dict[str, float], components: tuple[str, ...]) -> float:
    """Equal-weight mean of the chosen raw components (all in [0,1])."""
    if not components:
        return 0.0
    return float(np.mean([scores[c] for c in components]))


def top_n_exp_peaks(peaks: np.ndarray, n: int) -> np.ndarray:
    """Keep only the top-N most intense experimental peaks (0 = keep all)."""
    if peaks.size == 0 or n <= 0 or n >= peaks.shape[0]:
        return peaks
    idx = np.argpartition(-peaks[:, 1], n - 1)[:n]
    out = peaks[idx]
    return out[out[:, 0].argsort()]


# ---------------------------------------------------------------------------
# FDR math (target-decoy competition, Elias-Gygi convention)
# ---------------------------------------------------------------------------

def compute_fdr_curve(sorted_rows: list[dict], label_key: str = "top1_label",
                      score_key: str = "top1_fused") -> list[dict]:
    """Given spectra sorted by score DESCENDING, return the FDR curve.

    ``label_key`` selects the T/D column (per-branch step4 rows carry e.g.
    ``d_label``); ``score_key`` the score column used only for the threshold
    reporting.  Output rows keep the input fields and add:
      cum_target, cum_decoy
      fdr_dt   (n_decoy + 1) / n_target       (Elias-Gygi, +1 correction — the
                                               unified FDR formula, identical to
                                               fdr_full/engine_fdr.py)
      fdr_tdt  decoy / (target + decoy)
      q_dt     q-value = suffix-min of fdr_dt (largest accepted set for a cutoff)
    Leading decoys (cum_target == 0) get fdr_dt = inf.
    """
    curve: list[dict] = []
    cum_t = cum_d = 0
    fdr_dt_arr = np.empty(len(sorted_rows), dtype=np.float64)
    for i, r in enumerate(dc.progress(sorted_rows, total=len(sorted_rows),
                                      desc="step4 FDR curve", unit="row",
                                      mininterval=1.0)):
        if r[label_key] == LABEL_DECOY:
            cum_d += 1
        else:
            cum_t += 1
        fdr_dt = ((cum_d + 1) / cum_t) if cum_t > 0 else float("inf")
        fdr_tdt = (cum_d / (cum_t + cum_d)) if (cum_t + cum_d) > 0 else float("inf")
        fdr_dt_arr[i] = fdr_dt
        out = dict(r)
        out.update(cum_target=cum_t, cum_decoy=cum_d,
                   fdr_dt=fdr_dt, fdr_tdt=fdr_tdt, q_dt=None)
        curve.append(out)

    # q-value = suffix minimum of fdr_dt (monotone from low to high confidence).
    q = np.empty(len(curve), dtype=np.float64)
    best = float("inf")
    for i in range(len(curve) - 1, -1, -1):
        best = min(best, fdr_dt_arr[i])
        q[i] = best
    for i, out in enumerate(curve):
        out["q_dt"] = float(q[i])
    return curve


def accepted_at_fdr(curve: list[dict], fdr_target: float, label_key: str = "top1_label",
                    score_key: str = "top1_fused", gt_key: str = "top1_gt_match") -> dict:
    """Largest accepted prefix (highest-confidence spectra) with q-value <= target.

    Returns the count summary for the identification set at that FDR line.
    """
    accepted = [r for r in curve if float(r["q_dt"]) <= fdr_target]
    n_t = sum(1 for r in accepted if r[label_key] == LABEL_TARGET)
    n_d = sum(1 for r in accepted if r[label_key] == LABEL_DECOY)
    n_gt = sum(1 for r in accepted
               if r[label_key] == LABEL_TARGET and bool(r.get(gt_key, False)))
    score_thr = accepted[-1][score_key] if accepted else None
    return {
        "fdr_target": fdr_target,
        "n_accepted_spectra": len(accepted),
        "n_target": n_t,
        "n_decoy": n_d,
        "n_target_gt_match": n_gt,
        "fdr_dt_achieved": round(((n_d + 1) / n_t), 6) if n_t > 0 else None,
        "score_threshold": round(float(score_thr), 6) if score_thr is not None else None,
        "top_score": round(float(curve[0][score_key]), 6) if curve else None,
    }


def write_fdr_csv(curve: list[dict], path) -> None:
    import csv

    cols = list(curve[0].keys()) if curve else ["ordinal"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(curve)
