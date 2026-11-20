#!/usr/bin/env python3
"""Step 4 — batch-predict candidate grids and score the production branch set.

For each MGF spectrum, load its candidate variants (from step3's TSV.gz), batch
predict each candidate's (39,46,6) theoretical intensity grid with the V20
checkpoint, then rank the candidates by the per-intensity single branches and
their fusions:

    d_spec_rel<R>     : spectral peak cosine (predicted peaks as the axis)
    f_frag_rel<R>     : explained intensity (fraction of exp intensity covered)
    m_by_prec_rel<R>  : main b/y-ion precision (main-token presence)
    hs_d<R>_f<R>      : score-avg fusion of d + f
    hs_d<R>_f<R>_mby  : score-avg fusion of d + f + m_by_prec   (best)

(--minimal keeps exactly this set; without it the full branch set incl. grid
cosines a/b, soft_ppm/comp_by/sat_consistency and more fusions is produced.)

For each branch the argmax candidate is the predicted sequence; accuracy vs the
MGF SEQ= ground truth uses I/L-equivalent mass matching.

Writes:
    <out-dir>/results_step4_<sp>.csv          one row per spectrum, wide
    <out-dir>/summary_step4_<sp>.json         per-branch accuracy summary
    <out-dir>/records/record_step4_<sp>.json  process record
    <out-dir>/candidates_detail_<sp>.csv.gz   per-candidate scores (--save-detail)

Usage:
    python step4_predict_score.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --candidates <ds>/data/mouse/candidates_mouse_scans.tsv.gz \
        --checkpoint /home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt \
        --out-dir <ds>/output/mouse [--gpu 0] [--batch-size 128] [--max-scans N]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ds_common as dc
from predict_mgf import generate_by_ions

# Token vocabulary — the single source of truth lives in <project_root>/token_table.py.
_TOKEN_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_TOKEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOKEN_ROOT))

from token_table import MAIN_TOKEN_IDS, TOKEN_NAMES  # noqa: E402

# Fusion definitions: (branch name suffix, csv prefix format, mode, base tags).
# Tags a/b/c/e/f expand to actual branch keys at the current rel level:
#   a = a_grid_nofilt, b = b_grid_rel<lvl>, c = c_spec_nofilt,
#   e = e_frag_nofilt, f = f_frag_rel<lvl>.
# mode "rank"  = rank-sum fusion (negated so higher = better)
# mode "score" = min-max normalized score average
FUSION_DEFS = [
    ("g_fuse",   "g{lvl}",    "rank",  ("b", "f", "c")),
    ("g_fuse5",  "g5_{lvl}",  "rank",  ("a", "b", "c", "e", "f")),
    ("h_score",  "hs_{lvl}",  "score", ("b", "f", "c")),
    ("h_score5", "hs5_{lvl}", "score", ("a", "b", "c", "e", "f")),
    ("g_bc",     "gbc{lvl}",  "rank",  ("b", "c")),
    ("g_bf",     "gbf{lvl}",  "rank",  ("b", "f")),
    ("g_cf",     "gcf{lvl}",  "rank",  ("c", "f")),
    ("hs_bc",    "hsbc{lvl}", "score", ("b", "c")),
    ("hs_bf",    "hsbf{lvl}", "score", ("b", "f")),
    ("hs_cf",    "hscf{lvl}", "score", ("c", "f")),
]


def _expand_fusion_base(tags, lvl: str) -> list[str]:
    return [
        {"a": "a_grid_nofilt", "b": f"b_grid_rel{lvl}",
         "c": "c_spec_nofilt", "e": "e_frag_nofilt",
         "f": f"f_frag_rel{lvl}"}[t]
        for t in tags
    ]


def build_branches(rel_intensities, cross_fuse: bool = False, minimal: bool = False):
    """Return (branches, cols, rel_for_branch) for the given rel filter levels.

    No-filter branches a/c/e are always present; for each relative-intensity
    level R a filtered variant of the grid/spec/frag scoring is added:
    b_grid_rel<pp>, d_spec_rel<pp>, f_frag_rel<pp> (pp = round(R*100), 3 digits).
    CSV prefix for a filtered branch is the single letter + pp (e.g. b005).
    Fusion branches follow FUSION_DEFS.  When ``cross_fuse`` is set, an
    additional grid of score-avg fusions hs_d<l1>_f<l2> (spec filter l1, frag
    filter l2) is added so the two signals can use different filter levels.

    When ``minimal`` is set, ONLY the production branch set is produced per
    level: the single branches d_spec_rel<R>, f_frag_rel<R>, m_by_prec_rel<R>,
    plus the fusions hs_d<R>_f<R> (d+f) and hs_d<R>_f<R>_mby (d+f+main-b/y).
    Everything else (a/b/c/e, soft_ppm, comp_by, sat_consistency, frag_prec,
    FUSION_DEFS, off-diagonal fusions) is dropped.
    """
    if minimal:
        # Production branch set: per-intensity single branches (d/f/m_by_prec)
        # + the d+f fusion (hs_d<R>_f<R>) + the d+f+main-b/y fusion (…_mby).
        # soft_ppm/comp_by/sat_consistency and all other fusion variants are
        # intentionally NOT produced here (they are only available in the
        # full non-minimal branch set).
        branches: list[str] = []
        cols: dict[str, tuple[str, str]] = {}
        rel_for_branch: dict[str, float] = {}
        for r in rel_intensities:
            lvl = f"{int(round(r * 100)):03d}"
            for letter, typ in (("d", "spec"), ("f", "frag"), ("m_by_prec", "")):
                key = f"{letter}_{typ}_rel{lvl}" if typ else f"m_by_prec_rel{lvl}"
                pre = f"{letter}{lvl}" if typ else f"mp{lvl}"
                branches.append(key)
                cols[key] = (pre, f"{pre}_hit")
                rel_for_branch[key] = r
            key = f"hs_d{lvl}_f{lvl}"
            branches.append(key)
            cols[key] = (f"hdf{lvl}_{lvl}", f"hdf{lvl}_{lvl}_hit")
            rel_for_branch[key] = r
            mkey = f"hs_d{lvl}_f{lvl}_mby"
            branches.append(mkey)
            cols[mkey] = (f"hdf{lvl}_{lvl}_mby", f"hdf{lvl}_{lvl}_mby_hit")
            rel_for_branch[mkey] = r
            # raw equal-weight fusions (NO min-max normalization) — the no-FDR
            # analogue of the FDR-family d_f_exp / d_f_exp_mby branches.
            rkey = f"dfexp_rel{lvl}"
            branches.append(rkey)
            cols[rkey] = (f"dfexp{lvl}", f"dfexp{lvl}_hit")
            rel_for_branch[rkey] = r
            rmk = f"dfexp_mby_rel{lvl}"
            branches.append(rmk)
            cols[rmk] = (f"dfexpmby{lvl}", f"dfexpmby{lvl}_hit")
            rel_for_branch[rmk] = r
        return tuple(branches), cols, rel_for_branch
    branches = ["a_grid_nofilt", "c_spec_nofilt", "e_frag_nofilt"]
    cols = {"a_grid_nofilt": ("a", "a_hit"),
            "c_spec_nofilt": ("c", "c_hit"),
            "e_frag_nofilt": ("e", "e_hit")}
    rel_for_branch: dict[str, float] = {}
    for r in rel_intensities:
        lvl = f"{int(round(r * 100)):03d}"
        for letter, typ in (("b", "grid"), ("d", "spec"), ("f", "frag")):
            key = f"{letter}_{typ}_rel{lvl}"
            branches.append(key)
            cols[key] = (f"{letter}{lvl}", f"{letter}{lvl}_hit")
            rel_for_branch[key] = r
        for key, pre in ((f"m_by_frag_rel{lvl}", f"mf{lvl}"),
                         (f"m_by_cos_rel{lvl}", f"mc{lvl}"),
                         (f"m_by_prec_rel{lvl}", f"mp{lvl}"),
                         (f"frag_prec_rel{lvl}", f"fp{lvl}"),
                         (f"g_by_rel{lvl}", f"gb{lvl}"),
                         (f"hs_by_rel{lvl}", f"hsb{lvl}"),
                         (f"g_dfp_rel{lvl}", f"gfp{lvl}"),
                         (f"hs_dfp_rel{lvl}", f"hfp{lvl}")):
            branches.append(key)
            cols[key] = (pre, f"{pre}_hit")
            rel_for_branch[key] = r
        for name, pre_fmt, _mode, _tags in FUSION_DEFS:
            pre = pre_fmt.format(lvl=lvl)
            key = f"{name}_rel{lvl}"
            branches.append(key)
            cols[key] = (pre, f"{pre}_hit")
            rel_for_branch[key] = r
    if cross_fuse:
        for r1 in rel_intensities:
            for r2 in rel_intensities:
                l1 = f"{int(round(r1 * 100)):03d}"
                l2 = f"{int(round(r2 * 100)):03d}"
                key = f"hs_d{l1}_f{l2}"
                branches.append(key)
                cols[key] = (f"hdf{l1}_{l2}", f"hdf{l1}_{l2}_hit")
                rel_for_branch[key] = r2
    return tuple(branches), cols, rel_for_branch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--candidates", required=True, help="TSV.gz from step3")
    p.add_argument("--checkpoint", default=dc.SERVER_CKPT)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None, help="override device (e.g. cuda:0/cpu)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-mz", type=float, default=dc.DEFAULT_MIN_MZ)
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--no-packed", action="store_true")
    p.add_argument("--save-detail", action="store_true",
                   help="also write per-candidate detail CSV.gz")
    p.add_argument("--save-predictions", action="store_true",
                   help="save per-branch best-candidate predicted grid (39,46,6) + spectrum MGF")
    p.add_argument("--save-branches", default="",
                   help="comma-separated branch keys to save predictions for (only with "
                        "--save-predictions); e.g. hs_d020_f020,hs_d010_f010,d_spec_rel020,"
                        "f_frag_rel020,d_spec_rel010,f_frag_rel010.  Empty = save all.")
    p.add_argument("--save-top-n", type=int, default=1,
                   help="how many top candidates' predictions to save per spectrum per "
                        "branch (default 1; 0 or negative = save ALL candidates, storage "
                        "scales ~N_candidates-fold)")
    p.add_argument("--pred-dtype", default="float16", choices=["float16", "float32"],
                   help="dtype for saved predicted grids")
    p.add_argument("--rel-intensities", default="0.05",
                   help="comma-separated PREDICTED-side relative-intensity filter levels, "
                        "e.g. 0.05,0.10,0.15,0.20 (experimental peaks always kept in full)")
    p.add_argument("--spec-min-intensity", type=float, default=0.01,
                   help="absolute min intensity for theoretical-peak branch (predicted side)")
    p.add_argument("--fm-tol", type=float, default=0.05,
                   help="m/z tolerance (Da) for fragment-match explained-intensity (e/f)")
    p.add_argument("--mz-merge-tol", type=float, default=0.05,
                   help="m/z merge tolerance (Da) when collapsing predicted grid cells into peaks (c/d/e/f)")
    p.add_argument("--cross-fuse", action="store_true",
                   help="add score-avg fusions hs_d<l1>_f<l2> for all spec/frag filter pairs")
    p.add_argument("--minimal", action="store_true",
                   help="only produce d_spec_rel<R>, f_frag_rel<R> and the diagonal "
                        "hs_d<R>_f<R> per level; drop all other branches/fusions")
    p.add_argument("--ppm-window", type=float, default=10.0,
                   help="ppm window used by the soft_ppm precursor-mass feature "
                        "(linear falloff from 1.0 at 0 ppm to 0 at --ppm-window)")
    p.add_argument("--fusion-norm", default="minmax", choices=["minmax", "zscore"],
                   help="normalization for score-avg fusions: minmax (robust to scale "
                        "but outlier-sensitive) or zscore (mean/std, more outlier-robust)")
    p.add_argument("--exp-transform", default="sqrt_clip", choices=["none", "sqrt_clip", "sqrt"],
                   help="map experimental peak intensities into the model's training "
                        "domain before scoring: sqrt_clip=sqrt(min(x,50000))/sqrt(50000) "
                        "(default; matches dataset target normalization), none=raw linear, "
                        "sqrt=sqrt(x). "
                        "Affects intensity-weighted branches (d spectral cosine, f explained "
                        "intensity); presence-based branches (m_by_prec/comp_by/sat) unchanged.")
    return p.parse_args()


def load_candidates_by_ordinal(tsv_gz: Path) -> dict[int, list[tuple[str, float, float]]]:
    """Read step3 TSV.gz -> {ordinal: [(seq, neutral_mass, ppm), ...]}."""
    grouped: dict[int, list[tuple[str, float, float]]] = {}
    with gzip.open(tsv_gz, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            ordinal = int(row[0])
            seq = row[6]
            mass = float(row[7])
            ppm = float(row[8])
            grouped.setdefault(ordinal, []).append((seq, mass, ppm))
    return grouped


def _build_one_candidate(args):
    """Build all model-input arrays for one candidate.  args = (seq, charge, min_mz,
    expected_in, use_ctx, stats).  Returns (input_mz, valid_mask, norm_input, length,
    aa_ids, scalar, pepmass).  Standalone so it can be run in worker processes."""
    seq, charge, min_mz, expected_in, use_ctx, stats = args
    input_mz = dc.build_input_mz_matrix(seq, min_mz=min_mz)
    valid_mask, cleavage_count = dc.build_valid_mask(seq, charge, min_mz=min_mz)
    norm_input = dc.prepare_model_input(input_mz, valid_mask, stats, expected_in)[0]
    length = cleavage_count + 1
    aa = scalar = None
    if use_ctx:
        a, s = dc.encode_cleavage_context(seq, int(length), max_cleavage_len=dc.MAX_CLEAVAGE_LEN)
        aa = a.astype(np.int64, copy=False)
        scalar = s.astype(np.float32, copy=False)
    pepmass = dc.calculate_pepmass(seq, charge)
    return input_mz, valid_mask, norm_input, length, aa, scalar, pepmass


def build_batch_arrays(
    model_config: dict,
    stats: dict,
    candidates: list[str],
    charge: int,
    rt: float,
    min_mz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all arrays for the candidates.

    Returns (input_mz, norm_inputs, valid_masks, lengths, aa_ids, scalar, aux),
    each with the first dimension = len(candidates).  Single-process grid
    building (the CPU-bound part); ``_build_one_candidate`` is factored out so
    it could be parallelized later if ever needed.
    """
    K = len(candidates)
    use_ctx = bool(model_config.get("use_sequence_context", True))
    expected_in = int(model_config.get("in_channels", dc.NUM_CHANNELS))

    input_all = np.zeros((K, dc.MAX_CLEAVAGE_LEN, dc.NUM_TOKENS, dc.NUM_CHANNELS), dtype=np.float32)
    norm_inputs = np.zeros_like(input_all)
    valid_masks = np.zeros_like(input_all)
    lengths = np.zeros(K, dtype=np.int64)
    aa_ids = np.zeros((K, dc.MAX_CLEAVAGE_LEN, 2), dtype=np.int64)
    scalar = np.zeros((K, dc.MAX_CLEAVAGE_LEN, 7), dtype=np.float32)
    pepmasses = np.zeros(K, dtype=np.float32)

    results = [_build_one_candidate(a) for a in
               ((s, charge, min_mz, expected_in, use_ctx, stats) for s in candidates)]

    for i, (im, vm, ni, ln, aa, sc_, pm) in enumerate(results):
        input_all[i] = im
        valid_masks[i] = vm
        norm_inputs[i] = ni
        lengths[i] = ln
        pepmasses[i] = pm
        if aa is not None:
            aa_ids[i] = aa
            scalar[i] = sc_

    raw_aux = np.stack(
        [np.array([charge, pm, rt], dtype=np.float32) for pm in pepmasses]
    )
    aux = (
        (raw_aux - stats["aux_feature_means"])
        / np.maximum(stats["aux_feature_stds"], 1e-6)
    ).astype(np.float32)
    return input_all, norm_inputs, valid_masks, lengths, aa_ids, scalar, aux


def predict_candidate_grids(
    model,
    model_config: dict,
    stats: dict,
    device: torch.device,
    candidates: list[str],
    charge: int,
    rt: float,
    min_mz: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred, input_mz_grids, valid_masks) each (K,39,46,6).

    Builds all candidate arrays once, then runs the GPU forward in batches of
    ``batch_size``.
    """
    K = len(candidates)
    use_ctx = bool(model_config.get("use_sequence_context", True))
    input_all, norm_inputs, valid_masks, lengths, aa_ids, scalar, aux = build_batch_arrays(
        model_config, stats, candidates, charge, rt, min_mz
    )
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


MAIN_TOKEN_ID = MAIN_TOKEN_IDS[0]  # token index of the main (no-offset) peak


def _main_by_peaks(input_mz, pred, valid_mask, min_intensity, min_mz, mz_merge_tol):
    """Extract the main b/y-ion peaks (main token only) as (m/z, intensity).

    This isolates the classic b/y fragment positions (the signal database search
    uses), keeping the model's sequence-specific choice of WHICH b/y ions are
    strong, while dropping satellite/isotope/neutral-loss tokens.
    """
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


def soft_ppm_score(ppm: float, window: float) -> float:
    """Precursor-mass soft feature: 1.0 at 0 ppm, linear falloff to 0 at ``window``."""
    return max(0.0, 1.0 - abs(ppm) / window) if window > 0 else 0.0


def _has_peak_near(exp_mz: np.ndarray, mz: float, tol: float) -> bool:
    idx = int(np.searchsorted(exp_mz, mz))
    for ni in (idx - 1, idx):
        if 0 <= ni < len(exp_mz) and abs(float(exp_mz[ni]) - mz) <= tol:
            return True
    return False


def comp_by_score(seq: str, exp_peaks: np.ndarray, tol: float) -> float:
    """Complementary b/y ion-pair consistency (structural, fragment-based).

    For each cleavage position i, count whether a b-ion at position i AND the
    complementary y-ion at position (L-i) are BOTH observed in the experimental
    spectrum (at any charge 1-3, within ``tol``).  The true peptide has coherent
    complementary pairs; a scattered decoy has fewer.  Returns the fraction of
    cleavage positions with a complete complementary pair.
    """
    if exp_peaks.size == 0:
        return 0.0
    exp_mz = np.sort(exp_peaks[:, 0].astype(np.float64))
    residues, _, _ = dc.parse_modified_sequence(seq)
    L = len(residues)
    if L < 2:
        return 0.0
    b_by_pos: dict[int, list[float]] = {}
    y_by_pos: dict[int, list[float]] = {}
    for ion in generate_by_ions(seq, max_charge=3):
        ci = int(ion["cleavage_index"])
        mz = float(ion["mz"])
        if ion["ion_type"] == "B":
            b_by_pos.setdefault(ci, []).append(mz)
        else:
            y_by_pos.setdefault(ci, []).append(mz)
    pairs = 0
    total = 0
    for i in range(1, L):
        b_obs = any(_has_peak_near(exp_mz, mz, tol) for mz in b_by_pos.get(i, []))
        y_obs = any(_has_peak_near(exp_mz, mz, tol) for mz in y_by_pos.get(L - i, []))
        total += 1
        if b_obs and y_obs:
            pairs += 1
    return pairs / total if total else 0.0


# Single-step satellites used for the consistency check: the three neutral
# losses plus the two isotope peaks.  The two combined losses (H2O+CO / NH3+CO)
# are deliberately excluded — they are not single-step satellites.
_SATELLITE_EXCLUDED = {"main", "H2O+CO", "NH3+CO"}
SATELLITE_TOKENS = {
    idx: name for idx, name in TOKEN_NAMES.items() if name not in _SATELLITE_EXCLUDED
}


def sat_consistency_score(input_mz, pred, valid_mask, exp_peaks,
                          min_intensity, tol) -> float:
    """V20-design signal: satellite consistency of observed main b/y ions.

    For each main b/y cell (main token) the model predicts strongly AND that is
    observed in the experiment, check whether the model's predicted satellite
    peaks (neutral losses NH3/H2O/CO and isotopes 13C/2x13C at the SAME
    position/channel) are ALSO observed.  The true peptide has coherent
    satellite patterns; a decoy's coincidental main ions lack them.  Returns the
    fraction of (main-ion, satellite) pairs that are both predicted and observed.
    """
    if exp_peaks.size == 0:
        return 0.0
    exp_mz = np.sort(exp_peaks[:, 0].astype(np.float64))
    pos_mask = valid_mask[:, MAIN_TOKEN_ID, :] > 0
    valid_rc = np.argwhere(pos_mask)
    confirmed = 0
    total = 0
    for pos, ch in valid_rc:
        pos, ch = int(pos), int(ch)
        if pred[pos, MAIN_TOKEN_ID, ch] <= min_intensity:
            continue  # model did not predict a main b/y here
        main_mz = float(input_mz[pos, MAIN_TOKEN_ID, ch])
        if not _has_peak_near(exp_mz, main_mz, tol):
            continue  # main ion not observed in experiment
        for sat_tok in SATELLITE_TOKENS:
            if valid_mask[pos, sat_tok, ch] <= 0:
                continue
            sat_mz = float(input_mz[pos, sat_tok, ch])
            total += 1
            if (pred[pos, sat_tok, ch] > min_intensity
                    and _has_peak_near(exp_mz, sat_mz, tol)):
                confirmed += 1
    return confirmed / total if total else 0.0


def score_candidate(
    pred: np.ndarray,
    input_mz: np.ndarray,
    valid_mask: np.ndarray,
    exp_peaks_all: np.ndarray,
    spec_min_intensity: float,
    min_mz: float,
    fm_tol: float,
    mz_merge_tol: float,
    rel_intensities: list[float],
    minimal: bool = False,
) -> dict[str, float]:
    v = valid_mask > 0
    pmax = float(pred.max())
    # The experimental spectrum is ALWAYS used in full — the relative-intensity
    # filter applies only to the PREDICTED side (removes the model's weak /
    # low-confidence predicted peaks), never to the observed peaks.
    out = {}

    if minimal:
        for r in rel_intensities:
            lvl = f"{int(round(r * 100)):03d}"
            eff = max(spec_min_intensity, r * pmax)
            theo = dc.prediction_to_mgf_peaks(
                input_mz, pred, valid_mask,
                min_intensity=eff, min_mz=min_mz, mz_merge_tol=mz_merge_tol,
            )
            out[f"d_spec_rel{lvl}"] = dc.peak_cosine(exp_peaks_all, theo, tol=fm_tol)
            out[f"f_frag_rel{lvl}"] = dc.explained_intensity(exp_peaks_all, theo, tol=fm_tol)
            by = _main_by_peaks(input_mz, pred, valid_mask, eff, min_mz, mz_merge_tol)
            out[f"m_by_prec_rel{lvl}"] = dc.frag_precision(exp_peaks_all, by, tol=fm_tol)
        return out

    proj = dc.project_experimental_to_grid(exp_peaks_all, input_mz, valid_mask, mz_tol=0.05)
    out["a_grid_nofilt"] = dc._safe_cosine(pred[v], proj[v])

    theo_lo = dc.prediction_to_mgf_peaks(
        input_mz, pred, valid_mask,
        min_intensity=spec_min_intensity, min_mz=min_mz, mz_merge_tol=mz_merge_tol,
    )
    out["c_spec_nofilt"] = dc.peak_cosine(exp_peaks_all, theo_lo, tol=fm_tol)
    out["e_frag_nofilt"] = dc.explained_intensity(exp_peaks_all, theo_lo, tol=fm_tol)

    for r in rel_intensities:
        lvl = f"{int(round(r * 100)):03d}"
        pred_f = np.where(pred >= r * pmax, pred, 0.0)
        out[f"b_grid_rel{lvl}"] = dc._safe_cosine(pred_f[v], proj[v])
        eff = max(spec_min_intensity, r * pmax)
        theo = dc.prediction_to_mgf_peaks(
            input_mz, pred, valid_mask,
            min_intensity=eff, min_mz=min_mz, mz_merge_tol=mz_merge_tol,
        )
        out[f"d_spec_rel{lvl}"] = dc.peak_cosine(exp_peaks_all, theo, tol=0.05)
        out[f"f_frag_rel{lvl}"] = dc.explained_intensity(exp_peaks_all, theo, tol=fm_tol)
        out[f"frag_prec_rel{lvl}"] = dc.frag_precision(exp_peaks_all, theo, tol=fm_tol)
        # main b/y-ion branches (main token only) — database-search-style signal
        by = _main_by_peaks(input_mz, pred, valid_mask, eff, min_mz, mz_merge_tol)
        out[f"m_by_frag_rel{lvl}"] = dc.explained_intensity(exp_peaks_all, by, tol=fm_tol)
        out[f"m_by_cos_rel{lvl}"] = dc.peak_cosine(exp_peaks_all, by, tol=fm_tol)
        out[f"m_by_prec_rel{lvl}"] = dc.frag_precision(exp_peaks_all, by, tol=fm_tol)
    return out


def main() -> int:
    args = parse_args()
    rel_intensities = [float(x.strip()) for x in args.rel_intensities.split(",") if x.strip()]
    BRANCHES, BRANCH_COLS, rel_for_branch = build_branches(
        rel_intensities, args.cross_fuse, args.minimal)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[step4] device={device}")

    _fasta, mgf_path = dc.resolve_species_files(args.root, args.species)
    model, model_config, stats, _ = dc.load_checkpoint(
        Path(args.checkpoint), device, no_packed=args.no_packed
    )
    print(f"[step4] checkpoint loaded; use_sequence_context="
          f"{model_config.get('use_sequence_context', True)}")

    candidates_by_ordinal = load_candidates_by_ordinal(Path(args.candidates))
    print(f"[step4] loaded candidates for {len(candidates_by_ordinal)} scans")

    results_csv = out_dir / f"results_step4_{args.species}.csv"
    detail_fh = None
    if args.save_detail:
        detail_fh = gzip.open(
            out_dir / f"candidates_detail_{args.species}.csv.gz",
            "wt", encoding="utf-8",
        )
        detail_fh.write("ordinal\tbranch\tcand_idx\tseq\tppm\tscore\trank\n")

    csv_columns = (
        ["ordinal", "scan_id", "charge", "pepmass", "rt", "n_candidates",
         "gt_seq", "gt_in_candidates"]
        + [f"{BRANCH_COLS[b][0]}_best_seq" for b in BRANCHES]
        + [f"{BRANCH_COLS[b][0]}_score" for b in BRANCHES]
        + [f"{BRANCH_COLS[b][0]}_gt_score" for b in BRANCHES]
        + [f"{BRANCH_COLS[b][0]}_gt_rank" for b in BRANCHES]
        + [f"{BRANCH_COLS[b][0]}_hit" for b in BRANCHES]
    )
    row_fields = list(csv_columns)

    n_spectra = 0
    n_processed = 0
    n_with_gt = 0
    n_skipped = 0
    branch_hits: dict[str, int] = {b: 0 for b in BRANCHES}
    branch_gt_rank_accum: dict[str, list[int]] = {b: [] for b in BRANCHES}
    branch_best_score_accum: dict[str, list[float]] = {b: [] for b in BRANCHES}
    branch_gt_score_accum: dict[str, list[float]] = {b: [] for b in BRANCHES}
    n_candidates_list: list[int] = []
    pred_dtype = np.dtype(args.pred_dtype)
    save_set = None
    if args.save_predictions and args.save_branches:
        save_set = set(x.strip() for x in args.save_branches.split(",") if x.strip())
    pred_holders: dict[str, list[dict]] = {}
    if args.save_predictions:
        pred_holders = {b: [] for b in BRANCHES if save_set is None or b in save_set}

    with open(results_csv, "w", encoding="utf-8", newline="") as fout_csv:
        writer = csv.DictWriter(fout_csv, fieldnames=row_fields, extrasaction="ignore")
        writer.writeheader()

        total_spectra = args.max_scans or dc.count_mgf_spectra(mgf_path)
        for ordinal, (meta, peaks) in enumerate(
            dc.progress(dc.iter_mgf_spectra(mgf_path, max_scans=args.max_scans),
                        total=total_spectra, desc="step4 score", unit="spec",
                        mininterval=1.0)
        ):
            n_spectra += 1
            charge = dc.parse_charge_from_meta(meta)
            pepmass = dc.parse_pepmass_from_meta(meta)
            rt = float(meta.get("RTINSECONDS", dc.DEFAULT_RT))
            gt_seq = meta.get("SEQ", "")
            cands = candidates_by_ordinal.get(ordinal, [])
            if charge <= 0 or pepmass <= 0 or not cands:
                n_skipped += 1
                continue

            seqs = [c[0] for c in cands]
            exp_peaks = dc.transform_exp_intensity(peaks, method=args.exp_transform)
            if exp_peaks.size == 0:
                n_skipped += 1
                continue

            pred, input_grids, valid_masks = predict_candidate_grids(
                model, model_config, stats, device, seqs, charge, rt,
                args.min_mz, args.batch_size,
            )
            n_processed += 1
            n_candidates_list.append(len(seqs))

            scores: dict[str, np.ndarray] = {}
            for i, seq in enumerate(seqs):
                sc = score_candidate(
                    pred[i], input_grids[i], valid_masks[i], exp_peaks,
                    args.spec_min_intensity, args.min_mz, args.fm_tol,
                    args.mz_merge_tol, rel_intensities, args.minimal,
                )
                if "soft_ppm" in BRANCHES:
                    sc["soft_ppm"] = soft_ppm_score(cands[i][2], args.ppm_window)
                if "comp_by" in BRANCHES:
                    sc["comp_by"] = comp_by_score(seq, exp_peaks, args.fm_tol)
                if "sat_consistency" in BRANCHES:
                    sc["sat_consistency"] = sat_consistency_score(
                        input_grids[i], pred[i], valid_masks[i], exp_peaks,
                        args.spec_min_intensity, args.fm_tol)
                for b, val in sc.items():
                    scores.setdefault(b, np.zeros(len(seqs), dtype=np.float64))[i] = val
            # ---- fusion variants per rel level (FUSION_DEFS) ----
            def _rank_sum(base):
                rs = np.zeros(len(seqs), dtype=np.float64)
                for bk in base:
                    rs += np.argsort(np.argsort(-scores[bk]))
                return -rs  # negated: higher fused = better

            def _score_avg(base):
                acc = np.zeros(len(seqs), dtype=np.float64)
                for bk in base:
                    s = scores[bk]
                    if args.fusion_norm == "zscore":
                        mu, sd = float(s.mean()), float(s.std())
                        if sd > 1e-12:
                            acc += (s - mu) / sd
                    else:
                        lo, hi = float(s.min()), float(s.max())
                        if hi - lo > 1e-12:
                            acc += (s - lo) / (hi - lo)
                return acc / len(base)

            def _raw_avg(base):
                acc = np.zeros(len(seqs), dtype=np.float64)
                for bk in base:
                    acc += scores[bk]
                return acc / len(base)

            if args.minimal:
                for r in rel_intensities:
                    lvl = f"{int(round(r * 100)):03d}"
                    df = [f"d_spec_rel{lvl}", f"f_frag_rel{lvl}"]
                    scores[f"hs_d{lvl}_f{lvl}"] = _score_avg(df)
                    scores[f"hs_d{lvl}_f{lvl}_mby"] = _score_avg(
                        df + [f"m_by_prec_rel{lvl}"])
                    scores[f"dfexp_rel{lvl}"] = _raw_avg(df)
                    scores[f"dfexp_mby_rel{lvl}"] = _raw_avg(
                        df + [f"m_by_prec_rel{lvl}"])
            else:
                for r in rel_intensities:
                    lvl = f"{int(round(r * 100)):03d}"
                    for name, _pre_fmt, mode, tags in FUSION_DEFS:
                        base = _expand_fusion_base(tags, lvl)
                        if mode == "rank":
                            scores[f"{name}_rel{lvl}"] = _rank_sum(base)
                        else:
                            scores[f"{name}_rel{lvl}"] = _score_avg(base)
                    # d + f + main-b/y fragment triple fusion
                    by_base = [f"d_spec_rel{lvl}", f"f_frag_rel{lvl}", f"m_by_frag_rel{lvl}"]
                    scores[f"g_by_rel{lvl}"] = _rank_sum(by_base)
                    scores[f"hs_by_rel{lvl}"] = _score_avg(by_base)
                    # d + f + fragment precision (recall + precision)
                    fp_base = [f"d_spec_rel{lvl}", f"f_frag_rel{lvl}", f"frag_prec_rel{lvl}"]
                    scores[f"g_dfp_rel{lvl}"] = _rank_sum(fp_base)
                    scores[f"hs_dfp_rel{lvl}"] = _score_avg(fp_base)

                if args.cross_fuse:
                    for r1 in rel_intensities:
                        for r2 in rel_intensities:
                            l1 = f"{int(round(r1 * 100)):03d}"
                            l2 = f"{int(round(r2 * 100)):03d}"
                            scores[f"hs_d{l1}_f{l2}"] = _score_avg(
                                [f"d_spec_rel{l1}", f"f_frag_rel{l2}"])

            # GT matching uses the V20-rescoring mass-matching rule (I/L equivalent).
            gt_in = bool(gt_seq) and any(dc.peptide_matches(s, gt_seq) for s in seqs)
            if gt_seq:
                n_with_gt += 1

            row = {
                "ordinal": ordinal,
                "scan_id": meta.get("SCANS", str(ordinal)),
                "charge": charge,
                "pepmass": round(pepmass, 6),
                "rt": round(rt, 3),
                "n_candidates": len(seqs),
                "gt_seq": gt_seq,
                "gt_in_candidates": int(gt_in),
            }
            for b in BRANCHES:
                prefix = BRANCH_COLS[b][0]
                order = np.argsort(-scores[b])  # descending
                best_i = int(order[0])
                best_seq = seqs[best_i]
                gt_idx = -1
                gt_score = -1.0
                if gt_seq:
                    best_gt = -float("inf")  # scores can be negative (fusion rank-sum)
                    for cand_i in range(len(seqs)):
                        if dc.peptide_matches(seqs[cand_i], gt_seq):
                            sc = float(scores[b][cand_i])
                            if sc > best_gt:
                                best_gt = sc
                                gt_idx = cand_i
                                gt_score = sc
                gt_rank = -1
                if gt_idx >= 0:
                    for rank_pos, cand_i in enumerate(order, start=1):
                        if cand_i == gt_idx:
                            gt_rank = rank_pos
                            break
                hit = int(bool(gt_seq) and dc.peptide_matches(best_seq, gt_seq))
                if hit:
                    branch_hits[b] += 1
                if gt_rank > 0:
                    branch_gt_rank_accum[b].append(gt_rank)
                if gt_idx >= 0:
                    branch_best_score_accum[b].append(float(scores[b][best_i]))
                    branch_gt_score_accum[b].append(gt_score)
                row[f"{prefix}_best_seq"] = best_seq
                row[f"{prefix}_score"] = round(float(scores[b][best_i]), 6)
                row[f"{prefix}_gt_score"] = round(gt_score, 6)
                row[f"{prefix}_gt_rank"] = gt_rank
                row[f"{prefix}_hit"] = hit

                if detail_fh is not None:
                    for rank_pos, cand_i in enumerate(order, start=1):
                        detail_fh.write(
                            f"{ordinal}\t{b}\t{cand_i}\t{seqs[cand_i]}\t"
                            f"{cands[cand_i][2]:.4f}\t{scores[b][cand_i]:.6f}\t{rank_pos}\n"
                        )

                if b in pred_holders:
                    save_indices = order[:args.save_top_n] if args.save_top_n > 0 else order
                    for _rk, cand_i in enumerate(save_indices, start=1):
                        _ig, _vm = input_grids[cand_i], valid_masks[cand_i]
                        _pmax = float(pred[cand_i].max())
                        if b in rel_for_branch:
                            _mi = max(args.spec_min_intensity,
                                      rel_for_branch[b] * _pmax)
                        else:  # no-filter branches a/c/e
                            _mi = args.spec_min_intensity
                        _pk = dc.prediction_to_mgf_peaks(
                            _ig, pred[cand_i], _vm,
                            min_intensity=_mi, min_mz=args.min_mz,
                            mz_merge_tol=args.mz_merge_tol,
                        )
                        pred_holders[b].append({
                            "ordinal": ordinal,
                            "scan_id": row["scan_id"],
                            "charge": charge,
                            "pepmass": pepmass,
                            "rt": rt,
                            "seq": seqs[cand_i],
                            "gt_seq": gt_seq,  # source MGF SEQ= (ground truth)
                            "score": float(scores[b][cand_i]),
                            "rank": int(_rk),
                            "grid": pred[cand_i].astype(pred_dtype),
                            "peaks": _pk,
                        })

            writer.writerow(row)

    if detail_fh is not None:
        detail_fh.close()

    pred_paths: dict[str, str] = {}
    if pred_holders:
        for b in BRANCHES:
            if b not in pred_holders:
                continue
            holder = pred_holders[b]
            if not holder:
                continue
            pdir = out_dir / "predictions" / b
            pdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pdir / "best_grids.npz",
                ordinals=np.array([h["ordinal"] for h in holder], dtype=np.int64),
                scan_ids=np.array([h["scan_id"] for h in holder], dtype=object),
                seqs=np.array([h["seq"] for h in holder], dtype=object),
                gt_seqs=np.array([h["gt_seq"] for h in holder], dtype=object),
                ranks=np.array([h["rank"] for h in holder], dtype=np.int64),
                charges=np.array([h["charge"] for h in holder], dtype=np.int64),
                pepmasses=np.array([h["pepmass"] for h in holder], dtype=np.float32),
                rts=np.array([h["rt"] for h in holder], dtype=np.float32),
                scores=np.array([h["score"] for h in holder], dtype=np.float32),
                grids=np.stack([h["grid"] for h in holder]).astype(pred_dtype),
            )
            with open(pdir / "predicted_spectra.mgf", "w", encoding="utf-8") as fh:
                for h in holder:
                    fh.write("BEGIN IONS\n")
                    fh.write(f"TITLE={h['scan_id']}\n")
                    fh.write(f"PEPMASS={h['pepmass']:.5f}\n")
                    fh.write(f"CHARGE={h['charge']}+\n")
                    fh.write(f"SCANS={h['scan_id']}\n")
                    fh.write(f"RTINSECONDS={h['rt']:.3f}\n")
                    fh.write(f"SEQ={h['seq']}\n")
                    fh.write(f"COMMENT=DS_RANK={h['rank']}\n")
                    for mz, inten in h["peaks"]:
                        fh.write(f"{mz:.5f} {inten:.6f}\n")
                    fh.write("END IONS\n")
            pred_paths[b] = str(pdir)
            print(f"[step4] predictions saved: {pdir} (best_grids.npz + predicted_spectra.mgf)")

    def rank_rate(branch: str, max_rank: int) -> float:
        if not branch_gt_rank_accum[branch]:
            return 0.0
        denom = n_with_gt if n_with_gt else 1
        return sum(1 for r in branch_gt_rank_accum[branch] if r <= max_rank) / denom

    summary = {
        "species": args.species,
        "n_spectra": n_spectra,
        "n_processed": n_processed,
        "n_with_gt": n_with_gt,
        "n_skipped": n_skipped,
        "mean_candidates": round(float(np.mean(n_candidates_list)), 2)
        if n_candidates_list else 0.0,
        "branches": {
            b: {
                "top1_acc": round(branch_hits[b] / n_with_gt, 6) if n_with_gt else 0.0,
                "top3_acc": round(rank_rate(b, 3), 6),
                "top10_acc": round(rank_rate(b, 10), 6),
                "hits": branch_hits[b],
                "median_best_minus_gt_score": (
                    round(float(np.median([
                        bs - gs for bs, gs in zip(
                            branch_best_score_accum[b], branch_gt_score_accum[b])
                    ])), 4) if branch_gt_score_accum[b] else None
                ),
            }
            for b in BRANCHES
        },
        "results_csv": str(results_csv),
        **timer.summary(),
    }
    with open(out_dir / f"summary_step4_{args.species}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    record = {
        "step": 4,
        "species": args.species,
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
        "batch_size": args.batch_size,
        "min_mz": args.min_mz,
        "no_packed": args.no_packed,
        "summary": summary,
        "save_predictions": args.save_predictions,
        "pred_dtype": args.pred_dtype,
        "prediction_dirs": pred_paths,
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_step4_{args.species}.json", record)

    print("[step4] done. per-branch top1_acc:")
    for b in BRANCHES:
        print(f"  {b}: {summary['branches'][b]['top1_acc']:.4f} "
              f"(top3={summary['branches'][b]['top3_acc']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
