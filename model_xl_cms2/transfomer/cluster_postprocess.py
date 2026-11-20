"""
V15.3 Cluster-level postprocessing for frozen model inference.

Implements the structural-cluster pipeline from:
  v15_3_cluster_postprocess_plan_revised.md

Core principle: no model weights, output tensors, or training losses are modified.
Only the inference-side postprocessing is restructured to convert [B,39,46,6]
prediction grids into physically plausible MGF peak lists.

Pipeline:
  1. position theoretical mask
  2. precursor-charge-aware channel mask
  3. main-peak / satellite dual-threshold filtering
  4. structural cluster construction by (pos, channel)
  5. isolated satellite removal with weak-main-peak protection
  6. cluster-level weighted scoring
  7. peptide-length + charge dual-factor dynamic cluster Top-K
  8. selected clusters expanded back to cell granularity
  9. cells mapped to m/z peak candidates
 10. final m/z neighbor merging
"""

import csv
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

# Token vocabulary — the single source of truth lives in <project_root>/token_table.py.
_TOKEN_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_TOKEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOKEN_ROOT))

from token_table import (  # noqa: E402
    MAIN_TOKEN_IDS,
    NUM_TOKENS,
    SATELLITE_TOKEN_IDS,
    TOKEN_NAMES,
)

# ---------------------------------------------------------------------------
# Token vocabulary constants
# ---------------------------------------------------------------------------
MAIN_TOKEN = MAIN_TOKEN_IDS[0]

# Satellite columns split by chemistry (indices resolved from the shared table).
NEUTRAL_LOSS_TOKENS = [i for i, n in TOKEN_NAMES.items() if n in ("CO", "H2O", "NH3")]
ISOTOPE_TOKENS = [i for i, n in TOKEN_NAMES.items() if n in ("13C", "2x13C")]
SATELLITE_TOKENS = sorted(NEUTRAL_LOSS_TOKENS + ISOTOPE_TOKENS)
# Extended set for strict mode (adds the combined neutral losses)
EXTENDED_SATELLITE_TOKENS = sorted(
    SATELLITE_TOKENS + [i for i, n in TOKEN_NAMES.items() if n in ("H2O+CO", "NH3+CO")]
)
CORE8_TOKENS = set(MAIN_TOKEN_IDS) | set(SATELLITE_TOKEN_IDS)

# Token type classification (P0-2: explicit mapping, no hardcoded index checks)
TOKEN_KIND = {}
for t in range(NUM_TOKENS):
    name = TOKEN_NAMES.get(t)
    if t in MAIN_TOKEN_IDS:            TOKEN_KIND[t] = "main"
    elif name in ("13C", "2x13C"):     TOKEN_KIND[t] = "isotope"
    elif name in ("CO", "H2O", "NH3"): TOKEN_KIND[t] = "neutral_loss"
    elif name in ("H2O+CO", "NH3+CO"): TOKEN_KIND[t] = "combined_loss"
    else:                              TOKEN_KIND[t] = "other"

# Token type prior for export_score ranking (module-level, shared by expand + rescue)
TOKEN_PRIOR = {"main": 1.25, "isotope": 1.00, "neutral_loss": 0.90,
               "combined_loss": 0.80, "other": 0.50}

AA_BASE_MASS = {
    "A": 71.037113805, "R": 156.101111050, "N": 114.042927470,
    "D": 115.026943065, "C": 103.009184505, "E": 129.042593135,
    "Q": 128.058577540, "G": 57.021463735, "H": 137.058911875,
    "I": 113.084064015, "L": 113.084064015, "K": 128.094963050,
    "M": 131.040484645, "F": 147.068413945, "P": 97.052763875,
    "S": 87.032028435, "T": 101.047678505, "W": 186.079312980,
    "Y": 163.063328575, "V": 99.068413945,
}


def estimate_mod_count_from_residues(residues=None, nterm_shift=0.0, tol=0.02):
    """Estimate modification count from deploy-time sequence parsing only."""
    count = 0
    try:
        if abs(float(nterm_shift)) > float(tol):
            count += 1
    except (TypeError, ValueError):
        pass
    if not residues:
        return count
    for aa, mass in residues:
        base = AA_BASE_MASS.get(str(aa))
        if base is None:
            continue
        try:
            if abs(float(mass) - float(base)) > float(tol):
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def load_token_profile_prior(profile_path=None, clip_min=0.75, clip_max=1.25,
                             intensity_power=0.5, nz_power=0.25):
    """Load a frozen token profile and convert it into a mild prior.

    The profile must be computed from train/validation data, not from the test
    set being evaluated.  Supported inputs:
      - JSON: {"token_prior": [NUM_TOKENS floats]} or {"tokens": [{"token_idx": ...}]}
      - CSV: a per-token distribution table from the token-distribution analysis
    """
    if not profile_path:
        return None

    def _rank_jitter_from_rows(rows):
        order = []
        for row in rows:
            tok = int(row.get("token_idx", -1))
            if tok < 0 or tok >= NUM_TOKENS:
                continue
            rank = row.get("rank_by_intensity_share", "")
            try:
                rank_val = float(rank)
            except (TypeError, ValueError):
                rank_val = None
            try:
                share = float(row.get("intensity_share", 0.0))
            except (TypeError, ValueError):
                share = 0.0
            order.append((tok, rank_val, share))
        if len(order) != NUM_TOKENS:
            return np.zeros(NUM_TOKENS, dtype=np.float32)
        if all(x[1] is not None for x in order):
            ordered = sorted(order, key=lambda x: (x[1], x[0]))
        else:
            ordered = sorted(order, key=lambda x: (-x[2], x[0]))
        jitter = np.zeros(NUM_TOKENS, dtype=np.float32)
        # Small enough to keep the prior mild, large enough to make top-N
        # deterministic and aligned with the frozen profile ranking rather
        # than with clipping/rounding artifacts.
        for pos, (tok, _rank, _share) in enumerate(ordered):
            jitter[tok] = (NUM_TOKENS - pos) * 1e-3
        return jitter

    path = str(profile_path)
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rows = obj.get("tokens") if isinstance(obj, dict) else None
        if isinstance(obj, dict) and "token_prior" in obj:
            prior = np.asarray(obj["token_prior"], dtype=np.float32)
            if prior.shape[0] != NUM_TOKENS:
                raise ValueError(f"token_prior JSON field must contain {NUM_TOKENS} values")
            prior = np.clip(prior, float(clip_min), float(clip_max))
            if isinstance(rows, list):
                prior = prior + _rank_jitter_from_rows(rows)
            return prior
        if isinstance(rows, list):
            rows = rows
        else:
            raise ValueError("Unsupported token profile JSON format")
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    intensity = np.ones(NUM_TOKENS, dtype=np.float32)
    nz_rate = np.ones(NUM_TOKENS, dtype=np.float32)
    seen = np.zeros(NUM_TOKENS, dtype=bool)
    for row in rows:
        tok = int(row.get("token_idx", -1))
        if tok < 0 or tok >= NUM_TOKENS:
            continue
        intensity[tok] = float(row.get("intensity_share", 0.0))
        nz_rate[tok] = float(row.get("nz_rate_gt0", 0.0))
        seen[tok] = True
    if not seen.all():
        missing = [str(i) for i in range(NUM_TOKENS) if not seen[i]]
        raise ValueError(f"Token profile is missing token(s): {', '.join(missing)}")

    med_int = float(np.median(intensity[intensity > 0]))
    med_nz = float(np.median(nz_rate[nz_rate > 0]))
    prior = np.power(intensity / max(med_int, 1e-12), float(intensity_power))
    prior *= np.power(nz_rate / max(med_nz, 1e-12), float(nz_power))
    prior = np.clip(prior.astype(np.float32), float(clip_min), float(clip_max))
    return prior + _rank_jitter_from_rows(rows)


def _token_kind(tok):
    return TOKEN_KIND.get(tok, "other")


def _token_prior_func(tok):
    return TOKEN_PRIOR.get(_token_kind(tok), 0.50)

# ---------------------------------------------------------------------------
# Channel definition: [B1+, B2+, B3+, Y1+, Y2+, Y3+]
# ---------------------------------------------------------------------------
CHANNEL_NAMES = ["B1+", "B2+", "B3+", "Y1+", "Y2+", "Y3+"]
N_CHANNELS = 6


# ---------------------------------------------------------------------------
# Physics-aware soft priors (applied to export_score, not hard filters)
# ---------------------------------------------------------------------------

# Amino acids frequently associated with specific neutral losses
_H2O_FAVORING_AA = set("STED")     # -H2O common for Ser/Thr/Glu/Asp
_NH3_FAVORING_AA = set("KRNQ")     # -NH3 common for Lys/Arg/Asn/Gln

# Satellite upper ratio limits (penalize if sat >> main)
_SAT_UPPER_RATIO = {
    "isotope": 1.5,          # isotope should not be >>1.5× main
    "neutral_loss": 2.0,     # loss can be up to 2× main
    "combined_loss": 1.5,    # combined loss similar to isotope
}
_SAT_UPPER_PENALTY = 0.5     # penalty multiplier when above upper ratio


def _is_b_ion(channel_idx):
    """Channels 0-2 are B ions, 3-5 are Y ions."""
    return channel_idx < 3


def _compute_neutral_loss_composition_prior(tok, pos, channel, residues, *,
                                            link_site=None, partner_residues=None):
    """Compute composition-aware prior for neutral loss tokens.

    -H2O (token 21): more common when fragment contains S/T/E/D
    -NH3 (token 22): more common when fragment contains K/R/N/Q

    Returns a prior multiplier in [0.6, 1.0]. Not a hard filter.
    """
    if residues is None or len(residues) < 2:
        return 1.0
    L = len(residues)
    if pos >= L - 1:
        return 1.0

    cleavage = pos + 1
    if _is_b_ion(channel):
        frag_set = {aa for aa, _ in residues[:cleavage]}
        carries_link = link_site is not None and cleavage >= link_site
    else:
        frag_set = {aa for aa, _ in residues[cleavage:]}
        carries_link = link_site is not None and cleavage < link_site
    if carries_link and partner_residues is not None:
        frag_set.update(aa for aa, _ in partner_residues)

    if TOKEN_NAMES.get(tok) == "H2O":  # -H2O
        if frag_set & _H2O_FAVORING_AA:
            return 1.0
        return 0.6
    elif TOKEN_NAMES.get(tok) == "NH3":  # -NH3
        if frag_set & _NH3_FAVORING_AA:
            return 1.0
        return 0.6
    return 1.0


def _compute_satellite_upper_penalty(kind, sat_score, main_score):
    """Soft penalty when satellite is abnormally stronger than main peak.

    Returns a multiplier in [0.5, 1.0]. Does NOT hard-delete.
    """
    if main_score <= 0:
        return 1.0
    ratio = sat_score / main_score
    limit = _SAT_UPPER_RATIO.get(kind, 2.0)
    if ratio > limit:
        return _SAT_UPPER_PENALTY
    return 1.0


def apply_physics_prior(tok, kind, score, main_score, pos, channel, residues=None, *,
                        link_site=None, partner_residues=None):
    """Unified physics prior for export_score calibration.

    Returns a multiplier that is multiplied into export_score.
    """
    prior = 1.0
    # Satellite upper ratio (soft penalty, not hard filter)
    if kind in ("isotope", "neutral_loss", "combined_loss"):
        prior *= _compute_satellite_upper_penalty(kind, score, main_score)
    # Neutral loss composition prior
    if kind == "neutral_loss":
        prior *= _compute_neutral_loss_composition_prior(
            tok, pos, channel, residues, link_site=link_site, partner_residues=partner_residues,
        )
    return prior


def _safe_logit_prob(x, eps=1e-6):
    x = float(np.clip(x, eps, 1.0 - eps))
    return float(np.log(x / (1.0 - x)))


def _sigmoid_prob(x):
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _compute_anchor_support_bonus(tok, kind, score, main_score, pos, channel, pred_matrix):
    if kind == "main":
        return 0.18
    sat_tokens = EXTENDED_SATELLITE_TOKENS
    sat_support = float(np.max(pred_matrix[pos, sat_tokens, channel])) if sat_tokens else 0.0
    main_support = float(main_score)
    if kind in ("isotope", "neutral_loss", "combined_loss"):
        return 0.10 + 0.25 * min(main_support, 1.0) + 0.10 * min(sat_support, 1.0)
    # residue-event family: require anchor/support to get meaningful boost
    return 0.30 * min(main_support, 1.0) + 0.20 * min(sat_support, 1.0)


def _compute_sequence_context_bonus(tok, kind, pos, channel, residues):
    if residues is None or len(residues) < 2:
        return 0.0
    if kind in ("main", "isotope", "neutral_loss", "combined_loss"):
        # sequence context mainly governs residue-event plausibility
        return 0.0
    L = len(residues)
    if pos >= L - 1:
        return 0.0
    if _is_b_ion(channel):
        fragment_aas = [aa for aa, _ in residues[:pos + 1]]
    else:
        fragment_aas = [aa for aa, _ in residues[pos + 1:]]
    frag_set = set(fragment_aas)
    bonus = 0.0
    # Small context-aware residue-event encouragement near chemically plausible losses/gains
    if frag_set & set("STED"):
        bonus += 0.06
    if frag_set & set("KRNQ"):
        bonus += 0.06
    if len(fragment_aas) <= 4:
        bonus += 0.04
    return float(bonus)


def _compute_charge_length_bonus(kind, peptide_length, precursor_charge):
    z = int(precursor_charge)
    L = int(peptide_length)
    if kind == "main":
        return 0.0
    if kind in ("isotope", "neutral_loss", "combined_loss"):
        if z == 2 and L <= 16:
            return 0.05
        if z == 3 and L <= 16:
            return 0.08
        return 0.02 if z == 3 else 0.0
    # residue-event family
    if z == 3 and L <= 16:
        return 0.16
    if z == 2 and L <= 16:
        return 0.10
    if z == 3:
        return 0.08
    return 0.03


def rerank_export_score(
    model_score,
    tok,
    kind,
    main_score,
    pos,
    channel,
    pred_matrix,
    peptide_length,
    precursor_charge,
    residues=None,
    token_profile_prior=None,
    *,
    link_site=None,
    partner_residues=None,
):
    tok_prior = float(token_profile_prior[int(tok)]) if token_profile_prior is not None else TOKEN_PRIOR.get(kind, 0.50)
    phys_prior = apply_physics_prior(
        tok, kind, model_score, main_score, pos, channel, residues,
        link_site=link_site, partner_residues=partner_residues,
    )
    base_score = float(model_score) * tok_prior * phys_prior

    anchor_bonus = _compute_anchor_support_bonus(tok, kind, model_score, main_score, pos, channel, pred_matrix)
    seq_bonus = _compute_sequence_context_bonus(tok, kind, pos, channel, residues)
    charge_len_bonus = _compute_charge_length_bonus(kind, peptide_length, precursor_charge)

    rerank_logit = (
        _safe_logit_prob(np.clip(base_score, 1e-6, 1.0 - 1e-6))
        + anchor_bonus
        + seq_bonus
        + charge_len_bonus
    )
    return _sigmoid_prob(rerank_logit)


def identity_export_score(
    model_score,
    tok,
    kind,
    main_score,
    pos,
    channel,
    pred_matrix,
    peptide_length,
    precursor_charge,
    residues=None,
    token_profile_prior=None,
    *,
    link_site=None,
    partner_residues=None,
):
    tok_prior = float(token_profile_prior[int(tok)]) if token_profile_prior is not None else TOKEN_PRIOR.get(kind, 0.50)
    phys_prior = apply_physics_prior(
        tok, kind, model_score, main_score, pos, channel, residues,
        link_site=link_site, partner_residues=partner_residues,
    )
    return float(model_score) * tok_prior * phys_prior


PP3_SUBGROUP_PEAK_BANDS = {
    "z2_short": {
        "recover_min_ratio": 0.78,
        "recover_target_ratio": 1.02,
        "recover_cap_ratio": 1.68,
        "main_budget": 5,
        "sat_budget": 12,
        "anchor_ratio": 1.06,
        "sat_scale": 0.74,
        "sat_main_ratio_scale": 0.92,
        "local_window": 1,
        "per_cluster_sat_limit": 2,
        "dense_cap_ratio": 1.62,
        "min_strong_main": 4,
    },
    "z2_long": {
        "recover_min_ratio": 0.76,
        "recover_target_ratio": 1.06,
        "recover_cap_ratio": 1.82,
        "main_budget": 4,
        "sat_budget": 10,
        "anchor_ratio": 1.04,
        "sat_scale": 0.76,
        "sat_main_ratio_scale": 0.94,
        "local_window": 1,
        "per_cluster_sat_limit": 2,
        "dense_cap_ratio": 1.76,
        "min_strong_main": 4,
    },
    "z3_short": {
        "recover_min_ratio": 0.74,
        "recover_target_ratio": 1.08,
        "recover_cap_ratio": 1.90,
        "main_budget": 6,
        "sat_budget": 14,
        "anchor_ratio": 1.00,
        "sat_scale": 0.78,
        "sat_main_ratio_scale": 0.96,
        "local_window": 1,
        "per_cluster_sat_limit": 3,
        "dense_cap_ratio": 1.86,
        "min_strong_main": 4,
    },
    "z3_long": {
        "recover_min_ratio": 0.72,
        "recover_target_ratio": 1.02,
        "recover_cap_ratio": 1.82,
        "main_budget": 3,
        "sat_budget": 8,
        "anchor_ratio": 1.08,
        "sat_scale": 0.74,
        "sat_main_ratio_scale": 0.94,
        "local_window": 1,
        "per_cluster_sat_limit": 2,
        "dense_cap_ratio": 1.74,
        "min_strong_main": 3,
    },
    "default": {
        "recover_min_ratio": 0.74,
        "recover_target_ratio": 1.02,
        "recover_cap_ratio": 1.80,
        "main_budget": 3,
        "sat_budget": 8,
        "anchor_ratio": 1.06,
        "sat_scale": 0.76,
        "sat_main_ratio_scale": 0.94,
        "local_window": 1,
        "per_cluster_sat_limit": 2,
        "dense_cap_ratio": 1.72,
        "min_strong_main": 3,
    },
}


def infer_postprocess_subgroup(peptide_length, precursor_charge):
    z = int(precursor_charge)
    L = int(peptide_length)
    if z == 2 and L <= 16:
        return "z2_short"
    if z == 2:
        return "z2_long"
    if z == 3 and L <= 16:
        return "z3_short"
    if z == 3:
        return "z3_long"
    return None


def _pp3_band_config(subgroup, params=None):
    bands = PP3_SUBGROUP_PEAK_BANDS
    if isinstance(params, dict) and isinstance(params.get("subgroup_peak_band"), dict):
        bands = params["subgroup_peak_band"]
    cfg = dict(bands.get("default", {}))
    if subgroup in bands and isinstance(bands[subgroup], dict):
        cfg.update(bands[subgroup])
    return cfg


def _pp3_target_counts(subgroup, N_by, min_peaks, target_peaks, max_peaks, params=None):
    cfg = _pp3_band_config(subgroup, params=params)
    recover_min = max(int(min_peaks), int(round(N_by * float(cfg.get("recover_min_ratio", 0.74)))))
    recover_target = max(
        recover_min,
        min(int(max_peaks), int(round(N_by * float(cfg.get("recover_target_ratio", 1.02))))),
    )
    recover_cap = max(
        recover_target,
        min(int(max_peaks), int(round(N_by * float(cfg.get("recover_cap_ratio", 1.80))))),
    )
    return cfg, recover_min, recover_target, recover_cap


# ---------------------------------------------------------------------------
# Scene parameter templates (Section 5 of the plan)
# ---------------------------------------------------------------------------
SCENE_TEMPLATES = {
    "mod": {
        "desc": "修饰 / de novo — loose, protect weak peaks",
        "t_main": 0.035,
        "t_sat": 0.015,
        "R_target": 1.18,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.10,
        "peak_cap_ratio": 3.0,
    },
    "search": {
        "desc": "常规蛋白搜库 — default balanced",
        "t_main": 0.040,
        "t_sat": 0.020,
        "R_target": 1.05,
        "channel_alpha": 0.10,
        "isolated_sat_mode": "default",
        "sat_main_ratio": 0.12,
        "peak_cap_ratio": 2.5,
    },
    "cosine_eval": {
        "desc": "PiPrime cosine evaluation — 8-token (18-25)",
        "t_main": 0.035,
        "t_sat": 0.015,
        "R_target": 1.20,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.08,
        "peak_cap_ratio": 3.0,
    },
    "cosine_eval_rich": {
        "desc": "PiPrime cosine — 14-token (18-25 + 33-36,42,43)",
        "t_main": 0.035,
        "t_sat": 0.015,
        "R_target": 1.25,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06,
        "peak_cap_ratio": 3.5,
    },
    # ── Rich ablation ──
    "rich_G1_33_36": {
        "desc": "V16 optimal: 18-25 + 33-36 (12 token), cap=2.0×N_by, z=2 relaxed",
        "t_main": 0.035, "t_sat": 0.015, "R_target": 1.25,
        "channel_alpha": 0.00, "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06, "peak_cap_ratio": 2.0,
        # z=2 relaxed thresholds: z=2 peaks are naturally weaker (higher m/z)
        "z2_t_main": 0.028, "z2_t_sat": 0.012,
    },
    "rich_G2_42_43": {
        "desc": "Ablation G2: 18-25 + 42,43 only (test 42-43 contribution)",
        "t_main": 0.035, "t_sat": 0.015, "R_target": 1.25,
        "channel_alpha": 0.00, "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06, "peak_cap_ratio": 3.5,
    },
    "v20_pp4": {
        "desc": "V20-PP4: sgr_v2 export set + post-merge expression calibration",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 1.55,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.024,
        "peak_cap_ratio": 2.90,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 7,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "sgr_mode": True,
        "sgr_generation": "v2",
        "sgr_stagea_profile_n": 16,
        "sgr_stageb_main_deficit_ratio": 0.92,
        "sgr_stageb_support_ratio": 0.82,
        "sgr_stageb_total_deficit_ratio": 1.20,
        "sgr_global_main_budget": 72,
        "sgr_global_sat_budget": 144,
        "sgr_global_tier1_budget": 144,
        "sgr_global_tier2_budget": 96,
        "pp4_mode": True,
        "pp4_score_alpha": 0.20,
        "pp4_rank_alpha": 0.08,
        "pp4_slot_alpha": 0.06,
        "pp4_tail_alpha": 0.16,
        "pp4_tail_gamma": 1.20,
        "pp4_floor": 0.82,
        "pp4_cap": 1.16,
        "pp4_energy_preserve": True,
        "pp4_subgroup_overrides": {
            "z2_short": {
                "pp4_score_alpha": 0.18,
                "pp4_rank_alpha": 0.06,
                "pp4_slot_alpha": 0.05,
                "pp4_tail_alpha": 0.20,
                "pp4_floor": 0.80,
                "pp4_cap": 1.12,
            },
            "z2_long": {
                "pp4_score_alpha": 0.19,
                "pp4_rank_alpha": 0.07,
                "pp4_slot_alpha": 0.05,
                "pp4_tail_alpha": 0.18,
                "pp4_floor": 0.81,
                "pp4_cap": 1.14,
            },
            "z3_short": {
                "pp4_score_alpha": 0.22,
                "pp4_rank_alpha": 0.09,
                "pp4_slot_alpha": 0.06,
                "pp4_tail_alpha": 0.14,
                "pp4_floor": 0.84,
                "pp4_cap": 1.18,
            },
            "z3_long": {
                "pp4_score_alpha": 0.21,
                "pp4_rank_alpha": 0.08,
                "pp4_slot_alpha": 0.06,
                "pp4_tail_alpha": 0.15,
                "pp4_floor": 0.83,
                "pp4_cap": 1.17,
            },
        },
    },
    # ── Extended ablation ──
    "abl_C1_no18_19": {
        "desc": "Ablation C1: 20-25 + 33-36,42,43 (no combined neutral loss 18-19)",
        "t_main": 0.035, "t_sat": 0.015, "R_target": 1.25,
        "channel_alpha": 0.00, "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06, "peak_cap_ratio": 3.5,
    },
    "abl_N1_neg_residue": {
        "desc": "Ablation N1: 18-25 + negative residue tokens 0-17",
        "t_main": 0.040, "t_sat": 0.020, "R_target": 1.10,
        "channel_alpha": 0.05, "isolated_sat_mode": "default",
        "sat_main_ratio": 0.10, "peak_cap_ratio": 2.5,
    },
    "abl_R2p2": {
        "desc": "Ablation R2.2: rich with cap=2.2×N_by",
        "t_main": 0.035, "t_sat": 0.015, "R_target": 1.25,
        "channel_alpha": 0.00, "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06, "peak_cap_ratio": 2.2,
    },
    "abl_R2p0": {
        "desc": "Ablation R2.0: rich with cap=2.0×N_by",
        "t_main": 0.035, "t_sat": 0.015, "R_target": 1.25,
        "channel_alpha": 0.00, "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06, "peak_cap_ratio": 2.0,
    },
    # ── Extended residue gain: add 26-32 (Gly/Ala/Ser/Pro/Val/Thr/Cys) ──
    "abl_rich_26_32": {
        "desc": "Ablation: 18-25 + 26-32 (small residue gains only)",
        "t_main": 0.035, "t_sat": 0.015, "R_target": 1.25,
        "channel_alpha": 0.00, "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06, "peak_cap_ratio": 3.5,
    },
    "abl_rich_26_36": {
        "desc": "Ablation: 18-25 + 26-36 (all small/medium residue gains)",
        "t_main": 0.035, "t_sat": 0.015, "R_target": 1.25,
        "channel_alpha": 0.00, "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.06, "peak_cap_ratio": 3.5,
    },
    "full44_diag": {
        "desc": "Diagnostic: full 46-token export (upper-limit test only, NOT for search)",
        "t_main": 0.030,
        "t_sat": 0.010,
        "R_target": 1.50,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.05,
        "peak_cap_ratio": 4.0,
    },
    "profile44": {
        "desc": "Profile-calibrated 46-token export (train/val prior, constrained)",
        "t_main": 0.016,
        "t_sat": 0.0045,
        "R_target": 2.20,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.03,
        "peak_cap_ratio": 3.00,
        "max_sat_per_cluster": 6,
        "max_tokens_per_cluster": 5,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": False,
    },
    "profile44_tiered": {
        "desc": "Profile-calibrated 46-token export with per-cluster token-tier budget",
        "t_main": 0.016,
        "t_sat": 0.0045,
        "R_target": 2.00,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.035,
        "peak_cap_ratio": 2.60,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 6,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": False,
        "token_tiered_budget": True,
        "profile_top_n": 16,
        "core_token_budget": 3,
        "profile_extra_budget": 2,
        "low_prior_budget": 1,
        "low_prior_strong_factor": 2.0,
        "state_guard": True,
    },
    "profile_top14": {
        "desc": "Profile top-14 stable-token export (train/val prior whitelist)",
        "t_main": 0.018,
        "t_sat": 0.0055,
        "R_target": 1.85,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.035,
        "peak_cap_ratio": 2.55,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 5,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "profile_top_n": 14,
    },
    "profile_top16": {
        "desc": "Profile top-N stable-token export (train/val prior whitelist)",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 1.95,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.032,
        "peak_cap_ratio": 2.75,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 6,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "profile_top_n": 16,
    },
    "profile_top20": {
        "desc": "Profile top-20 stable-token export (train/val prior whitelist)",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 2.10,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.030,
        "peak_cap_ratio": 3.10,
        "max_sat_per_cluster": 6,
        "max_tokens_per_cluster": 6,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "profile_top_n": 20,
    },
    "profile_tier": {
        "desc": "Profile-tier export: core + high profile + score-gated plateau",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 1.90,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.04,
        "peak_cap_ratio": 2.80,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 5,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "token_tiered_budget": True,
        "profile_top_n": 22,
        "profile_high_n": 14,
        "profile_plateau_n": 22,
        "core_token_budget": 3,
        "profile_extra_budget": 2,
        "plateau_budget": 1,
        "plateau_strong_factor": 1.35,
        "low_prior_budget": 0,
    },
    "profile_custom": {
        "desc": "Validation-selected custom-token export (non-leaking whitelist)",
        "t_main": 0.020,
        "t_sat": 0.006,
        "R_target": 1.75,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.04,
        "peak_cap_ratio": 2.60,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 5,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "export_tokens": None,
    },
    "cape_v1": {
        "desc": "CAPE v1: conditional adaptive profile exportor",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 1.95,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.035,
        "peak_cap_ratio": 2.85,
        "max_sat_per_cluster": 6,
        "max_tokens_per_cluster": 7,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "cape_mode": True,
        "cape_tier1_n": 8,
        "cape_tier2_n": 8,
    },
    "cape_v2": {
        "desc": "CAPE v2: subgroup-first sparse recovery exportor",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 2.10,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.030,
        "peak_cap_ratio": 3.00,
        "max_sat_per_cluster": 7,
        "max_tokens_per_cluster": 8,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "cape_mode": True,
        "cape_generation": "v2",
        "cape_tier1_n": 10,
        "cape_tier2_n": 10,
    },
    "cape_v3": {
        "desc": "CAPE v3: full-tier subgroup recovery exportor",
        "t_main": 0.016,
        "t_sat": 0.0045,
        "R_target": 2.60,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.025,
        "peak_cap_ratio": 4.20,
        "max_sat_per_cluster": 9,
        "max_tokens_per_cluster": 12,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "cape_mode": True,
        "cape_generation": "v3",
        "cape_tier1_n": 16,
        "cape_tier2_n": 20,
    },
    "sgr_v1": {
        "desc": "SGR v1: spectrum-guided lower-bound recovery",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 1.95,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.028,
        "peak_cap_ratio": 3.40,
        "max_sat_per_cluster": 6,
        "max_tokens_per_cluster": 10,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "sgr_mode": True,
        "sgr_generation": "v1",
        "sgr_stagea_profile_n": 20,
        "sgr_stageb_main_deficit_ratio": 0.78,
        "sgr_stageb_total_deficit_ratio": 0.92,
        "sgr_global_main_budget": 48,
        "sgr_global_sat_budget": 84,
        "sgr_global_tier1_budget": 96,
        "sgr_global_tier2_budget": 72,
    },
    "sgr_v2": {
        "desc": "SGR v2: position-centric subgroup recovery",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 1.55,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.024,
        "peak_cap_ratio": 2.90,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 7,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "sgr_mode": True,
        "sgr_generation": "v2",
        "sgr_stagea_profile_n": 16,
        "sgr_stageb_main_deficit_ratio": 0.92,
        "sgr_stageb_support_ratio": 0.82,
        "sgr_stageb_total_deficit_ratio": 1.20,
        "sgr_global_main_budget": 72,
        "sgr_global_sat_budget": 144,
        "sgr_global_tier1_budget": 144,
        "sgr_global_tier2_budget": 96,
    },
    "tcr_v1": {
        "desc": "TCR v1: theory-constrained reconstruction",
        "t_main": 0.018,
        "t_sat": 0.005,
        "R_target": 1.35,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.020,
        "peak_cap_ratio": 2.20,
        "max_sat_per_cluster": 4,
        "max_tokens_per_cluster": 5,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "tcr_mode": True,
        "tcr_generation": "v1",
        "tcr_anchor_main_ratio": 0.82,
        "tcr_anchor_support_ratio": 0.55,
        "tcr_main_floor_ratio": 0.72,
        "tcr_support_floor_ratio": 0.90,
        "tcr_family_cap_main": 1,
        "tcr_family_cap_support": 3,
        "tcr_global_main_budget": 96,
        "tcr_global_support_budget": 192,
    },
    "tcr_v2": {
        "desc": "TCR v2: denser theory-constrained reconstruction",
        "t_main": 0.017,
        "t_sat": 0.0045,
        "R_target": 1.70,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.018,
        "peak_cap_ratio": 2.90,
        "max_sat_per_cluster": 5,
        "max_tokens_per_cluster": 6,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "tcr_mode": True,
        "tcr_generation": "v2",
        "tcr_anchor_main_ratio": 0.68,
        "tcr_anchor_support_ratio": 0.28,
        "tcr_main_floor_ratio": 0.90,
        "tcr_support_floor_ratio": 1.35,
        "tcr_family_cap_main": 1,
        "tcr_family_cap_support": 5,
        "tcr_global_main_budget": 144,
        "tcr_global_support_budget": 320,
    },
    "tcr_v3": {
        "desc": "TCR v3: anchor-floor dense reconstruction",
        "t_main": 0.016,
        "t_sat": 0.0040,
        "R_target": 2.00,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.016,
        "peak_cap_ratio": 3.20,
        "max_sat_per_cluster": 6,
        "max_tokens_per_cluster": 8,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "tcr_mode": True,
        "tcr_generation": "v3",
        "tcr_anchor_main_ratio": 0.58,
        "tcr_anchor_support_ratio": 0.10,
        "tcr_anchor_floor_ratio": 0.92,
        "tcr_main_floor_ratio": 1.00,
        "tcr_support_floor_ratio": 1.60,
        "tcr_support_budget_ratio": 1.45,
        "tcr_family_cap_main": 1,
        "tcr_family_cap_support": 7,
        "tcr_global_main_budget": 192,
        "tcr_global_support_budget": 480,
    },
    "tcr_v4": {
        "desc": "TCR v4: anchor + position backfill reconstruction",
        "t_main": 0.015,
        "t_sat": 0.0035,
        "R_target": 2.30,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.014,
        "peak_cap_ratio": 3.60,
        "max_sat_per_cluster": 7,
        "max_tokens_per_cluster": 10,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "tcr_mode": True,
        "tcr_generation": "v4",
        "tcr_anchor_main_ratio": 0.48,
        "tcr_anchor_support_ratio": 0.05,
        "tcr_anchor_floor_ratio": 1.00,
        "tcr_main_floor_ratio": 1.10,
        "tcr_support_floor_ratio": 1.90,
        "tcr_support_budget_ratio": 1.70,
        "tcr_position_backfill_ratio": 0.65,
        "tcr_proposal_main_ratio": 0.85,
        "tcr_proposal_support_ratio": 0.80,
        "tcr_family_cap_main": 1,
        "tcr_family_cap_support": 9,
        "tcr_global_main_budget": 256,
        "tcr_global_support_budget": 640,
    },
    "tcr_v5": {
        "desc": "TCR v5: coverage-first lower-bound reconstruction",
        "t_main": 0.015,
        "t_sat": 0.0035,
        "R_target": 2.45,
        "channel_alpha": 0.00,
        "isolated_sat_mode": "loose",
        "sat_main_ratio": 0.013,
        "peak_cap_ratio": 3.85,
        "max_sat_per_cluster": 7,
        "max_tokens_per_cluster": 10,
        "cluster_anchor_mode": "max_token",
        "cluster_score_mode": "profile_max",
        "require_main_for_sat": False,
        "dense_guard": True,
        "tcr_mode": True,
        "tcr_generation": "v5",
        "tcr_anchor_main_ratio": 0.46,
        "tcr_anchor_support_ratio": 0.05,
        "tcr_anchor_floor_ratio": 1.00,
        "tcr_main_floor_ratio": 1.08,
        "tcr_support_floor_ratio": 1.95,
        "tcr_support_budget_ratio": 1.90,
        "tcr_position_backfill_ratio": 0.72,
        "tcr_proposal_main_ratio": 0.82,
        "tcr_proposal_support_ratio": 0.78,
        "tcr_family_cap_main": 1,
        "tcr_family_cap_support": 10,
        "tcr_global_main_budget": 288,
        "tcr_global_support_budget": 768,
        "tcr_position_support_floor": 2,
        "tcr_weak_position_bonus_ratio": 0.22,
        "tcr_weak_support_bonus_ratio": 0.32,
        "tcr_weak_backfill_bonus_ratio": 0.10,
    },
    "quant": {
        "desc": "临床定量大样本 — strict, low FP",
        "t_main": 0.060,
        "t_sat": 0.030,
        "R_target": 0.82,
        "channel_alpha": 0.20,
        "isolated_sat_mode": "strict",
        "sat_main_ratio": 0.15,
        "peak_cap_ratio": 1.5,
    },
}


def resolve_scene_params(scene=None, overrides=None):
    """Resolve scene template with optional per-key overrides.

    Args:
        scene: one of 'mod', 'search', 'quant', or None (uses 'search' defaults).
        overrides: optional dict of parameter overrides.

    Returns:
        dict of resolved parameters.
    """
    template = SCENE_TEMPLATES.get(scene, SCENE_TEMPLATES["search"]).copy()
    if overrides:
        template.update(overrides)
    return template


def _resolve_subgroup_postprocess_params(p, peptide_length, precursor_charge):
    subgroup_overrides = p.get("subgroup_overrides")
    if not isinstance(subgroup_overrides, dict):
        return p
    subgroup = infer_postprocess_subgroup(peptide_length, precursor_charge)
    if subgroup is None:
        return p
    local = subgroup_overrides.get(subgroup)
    if not isinstance(local, dict):
        return p
    merged = dict(p)
    merged.update(local)
    merged["resolved_subgroup"] = subgroup
    return merged


# ---------------------------------------------------------------------------
# Peptide length binning
# ---------------------------------------------------------------------------
def get_length_bin(L):
    """Map peptide length to bin index."""
    if L <= 8:
        return 0  # bin 1
    elif L <= 15:
        return 1  # bin 2
    elif L <= 25:
        return 2  # bin 3
    else:
        return 3  # bin 4


LENGTH_BIN_EDGES = [0, 8, 15, 25, float("inf")]


# ---------------------------------------------------------------------------
# Default N_true_mean table: (length_bin, charge) -> expected true cluster count.
# These are placeholders — should be calibrated from training/validation set
# statistics.  Values are rough estimates from typical tryptic peptide spectra.
# ---------------------------------------------------------------------------
# fmt: off
DEFAULT_N_TRUE_MEAN = {
    # length_bin: {charge: mean_cluster_count}
    0: {1: 18, 2: 22, 3: 24, 4: 20},   # L <= 8
    1: {1: 28, 2: 36, 3: 40, 4: 34},   # 9-15
    2: {1: 40, 2: 52, 3: 58, 4: 48},   # 16-25
    3: {1: 52, 2: 68, 3: 76, 4: 62},   # L > 25
}
# fmt: on


def load_n_true_mean_from_stats(stats_path=None):
    """Load calibrated N_true_mean from a JSON stats file.

    Expected format: {"length_bin_0": {"1": 18.5, "2": 22.3, ...}, ...}
    If path is None or file missing, returns DEFAULT_N_TRUE_MEAN.
    """
    if stats_path is None:
        return DEFAULT_N_TRUE_MEAN
    try:
        import json
        with open(stats_path, "r") as f:
            raw = json.load(f)
        # Convert string keys to int
        result = {}
        for lb_str, charge_dict in raw.items():
            lb = int(lb_str.split("_")[-1]) if "_" in lb_str else int(lb_str)
            result[lb] = {int(c): float(v) for c, v in charge_dict.items()}
        return result
    except Exception:
        return DEFAULT_N_TRUE_MEAN


# ---------------------------------------------------------------------------
# 3.1 Position theoretical mask
# ---------------------------------------------------------------------------
def apply_position_mask(pred_matrix, peptide_length):
    """Zero out padding positions (pos >= L-1).

    Args:
        pred_matrix: (39, 46, 6) float array
        peptide_length: int, number of amino acid residues

    Returns:
        masked matrix (same shape)
    """
    L = max(int(peptide_length), 1)
    masked = pred_matrix.copy()
    # Valid cleavage positions: 0, 1, ..., L-2
    # Padding: pos >= L-1
    if L - 1 < pred_matrix.shape[0]:
        masked[L - 1:, :, :] = 0.0
    return masked


# ---------------------------------------------------------------------------
# 3.2 Charge-aware channel mask
# ---------------------------------------------------------------------------
def build_charge_channel_mask(precursor_charge):
    """Return boolean array of shape (6,) indicating which channels are valid
    given the precursor charge.

    Fragment charge cannot exceed precursor charge:
      charge 1: only b1(0), y1(3)
      charge 2: b1(0), b2(1), y1(3), y2(4)
      charge >=3: all channels
    """
    mask = np.zeros(N_CHANNELS, dtype=bool)
    try:
        z = int(precursor_charge)
    except (TypeError, ValueError):
        mask[:] = True
        return mask

    if z == 1:
        mask[[0, 3]] = True
    elif z == 2:
        mask[[0, 1, 3, 4]] = True
    else:
        mask[:] = True
    return mask


def apply_charge_mask(pred_matrix, precursor_charge):
    """Zero out channels where fragment charge > precursor charge.

    Args:
        pred_matrix: (39, 46, 6) float array
        precursor_charge: int

    Returns:
        masked matrix (same shape)
    """
    ch_mask = build_charge_channel_mask(precursor_charge)
    masked = pred_matrix.copy()
    masked[:, :, ~ch_mask] = 0.0
    return masked


# ---------------------------------------------------------------------------
# 3.3 & 3.4 Main-peak / satellite dual-threshold filtering
# ---------------------------------------------------------------------------
def get_adaptive_t_main(peptide_length, base_t_main=0.04):
    """Peptide-length adaptive main peak threshold (Section 3.4).

    L <= 8:  0.035
    9-24:    0.040
    L >= 25: 0.045
    """
    L = max(int(peptide_length), 1)
    # Profile-calibrated scenes intentionally use much lower base thresholds.
    # Do not snap them back to the old 0.04/0.045 legacy floor, otherwise z=2
    # and short/mid-length peptides collapse back to sparse 8-token behavior.
    if float(base_t_main) < 0.030:
        if L <= 8:
            return float(base_t_main) * 0.85
        elif L <= 12:
            return float(base_t_main) * 0.95
        elif L <= 24:
            return float(base_t_main)
        else:
            return float(base_t_main) * 1.10
    if L <= 8:
        return min(base_t_main, 0.035)
    elif L <= 24:
        return 0.040
    else:
        return max(base_t_main, 0.045)


def get_t_sat(t_main, min_t_sat=0.01):
    """Satellite threshold derived from main peak threshold."""
    return max(min_t_sat, 0.5 * t_main)


def build_cluster_mask(pred_matrix, t_main=0.04, t_weak_main=0.015,
                       t_sat=0.018, isolated_sat_mode="default",
                       cluster_anchor_mode="main", token_profile_prior=None):
    """Apply main-peak / satellite dual-threshold filtering with weak-main-peak
    protection and isolated satellite removal.

    This operates at the (pos, channel) structural-cluster level.

    Args:
        pred_matrix: (39, 46, 6) float array
        t_main: main peak threshold
        t_weak_main: weak main peak protection threshold
        t_sat: satellite threshold
        isolated_sat_mode: 'loose', 'default', or 'strict'

    Returns:
        filtered_matrix: (39, 46, 6) with isolated satellites zeroed
        cluster_info: dict with per-cluster diagnostics
            - main_scores: (39, 6) main peak (main token) scores
            - cluster_masks: (39, 6) bool — is cluster active
            - weak_main: (39, 6) bool — weak main peak flag
            - isolated_sat_removed: (39, 6) bool — where satellites were removed
    """
    H, W, C = pred_matrix.shape
    filtered = pred_matrix.copy()

    # Extract main peak scores: (H, C)
    main_scores = pred_matrix[:, MAIN_TOKEN, :]
    if token_profile_prior is None:
        token_profile_prior = np.ones(W, dtype=np.float32)
    token_profile_prior = np.asarray(token_profile_prior, dtype=np.float32).reshape(1, W, 1)

    if str(cluster_anchor_mode or "main") == "max_token":
        anchor_scores = (pred_matrix * token_profile_prior).max(axis=1)
    else:
        anchor_scores = main_scores

    # Per-cluster active mask: main_score for legacy scenes, max calibrated
    # token score for profile44.  This avoids deleting real non-main-token
    # clusters before token-level ranking can evaluate them.
    cluster_active = anchor_scores >= t_weak_main
    weak_main = (anchor_scores >= t_weak_main) & (anchor_scores < t_main)
    strong_main = main_scores >= t_main

    # For strict mode, extend satellite set
    if isolated_sat_mode == "strict":
        sat_tokens = EXTENDED_SATELLITE_TOKENS
    else:
        sat_tokens = SATELLITE_TOKENS

    # Remove isolated satellites: where no main peak, zero out satellite tokens
    isolated_sat_removed = np.zeros((H, C), dtype=bool)
    for ch in range(C):
        for pos in range(H):
            if not cluster_active[pos, ch]:
                # Check if there are any satellite responses at this cluster
                sat_response = pred_matrix[pos, sat_tokens, ch].max()
                if sat_response > 0:
                    # No main peak — remove all satellites
                    filtered[pos, sat_tokens, ch] = 0.0
                    isolated_sat_removed[pos, ch] = True

    # Apply satellite threshold for active clusters
    for ch in range(C):
        for pos in range(H):
            if cluster_active[pos, ch]:
                # For satellite tokens, apply t_sat threshold
                sat_mask = pred_matrix[pos, sat_tokens, ch] >= t_sat
                filtered[pos, sat_tokens, ch] = np.where(
                    sat_mask, filtered[pos, sat_tokens, ch], 0.0
                )

    # For weak main peaks (between t_weak_main and t_main), apply stricter
    # satellite threshold but don't remove
    if isolated_sat_mode != "loose":
        for ch in range(C):
            for pos in range(H):
                if weak_main[pos, ch]:
                    # Tighten satellite threshold slightly for weak mains
                    sat_mask = pred_matrix[pos, sat_tokens, ch] >= (t_sat * 1.25)
                    filtered[pos, sat_tokens, ch] = np.where(
                        sat_mask, filtered[pos, sat_tokens, ch], 0.0
                    )

    cluster_info = {
        "main_scores": main_scores,
        "anchor_scores": anchor_scores,
        "cluster_active": cluster_active,
        "weak_main": weak_main,
        "strong_main": strong_main,
        "isolated_sat_removed": isolated_sat_removed,
        "n_clusters_active": int(cluster_active.sum()),
        "n_weak_main": int(weak_main.sum()),
        "n_isolated_sat_removed": int(isolated_sat_removed.sum()),
    }
    return filtered, cluster_info


# ---------------------------------------------------------------------------
# 4.1-4.3 Structural cluster construction & weighted scoring
# ---------------------------------------------------------------------------
def compute_cluster_scores(pred_matrix, cluster_active):
    """Compute per-cluster weighted scores.

    cluster_score =
        1.00 * main_score
      + 0.30 * max(isotope_score)
      + 0.35 * max(neutral_loss_score)
      + 0.15 * sum(top2_satellite_score)

    Weak main peaks (< t_main) get a 0.75 coefficient.

    Args:
        pred_matrix: (39, 46, 6) float array
        cluster_active: (39, 6) bool array

    Returns:
        cluster_scores: (39, 6) float array
        score_components: dict with breakdown
    """
    H, W, C = pred_matrix.shape
    cluster_scores = np.zeros((H, C), dtype=np.float32)

    for ch in range(C):
        for pos in range(H):
            if not cluster_active[pos, ch]:
                continue

            main_score = float(pred_matrix[pos, MAIN_TOKEN, ch])
            isotope_scores = pred_matrix[pos, ISOTOPE_TOKENS, ch]
            neutral_loss_scores = pred_matrix[pos, NEUTRAL_LOSS_TOKENS, ch]

            max_isotope = float(np.max(isotope_scores)) if len(isotope_scores) > 0 else 0.0
            max_neutral = float(np.max(neutral_loss_scores)) if len(neutral_loss_scores) > 0 else 0.0

            # Top-2 satellite scores
            all_sat = np.concatenate([isotope_scores, neutral_loss_scores])
            top2_sat = float(np.sum(np.sort(all_sat)[-2:])) if len(all_sat) >= 2 else float(np.sum(all_sat))

            score = 1.00 * main_score + 0.30 * max_isotope + 0.35 * max_neutral + 0.15 * top2_sat
            cluster_scores[pos, ch] = score

    return cluster_scores


# Token export policy: which tokens to include per scene.
# search/quant: neutral losses + main + isotope (6 satellite/main columns)
# mod: adds the combined neutral losses (the full 8-column base block)
# Residue loss/gain columns are internal fragments — never exported.
# (Scene names keep their historical *_44 labels; the tables are 46-wide now.)
_EXPORT_TOKENS = {
    "search":           [21, 22, 23, 24, 25, 26],
    "quant":            [21, 22, 23, 24, 25, 26],
    "mod":              [19, 20, 21, 22, 23, 24, 25, 26],
    "cosine_eval":      [19, 20, 21, 22, 23, 24, 25, 26],
    "cosine_eval_rich": [19, 20, 21, 22, 23, 24, 25, 26,
                         34, 35, 36, 37, 43, 44],
    # ── Rich ablation (G1-G5) ──
    "rich_G1_33_36":    [19, 20, 21, 22, 23, 24, 25, 26, 34, 35, 36, 37],
    "rich_G2_42_43":    [19, 20, 21, 22, 23, 24, 25, 26, 43, 44],
    "full44_diag":      list(range(NUM_TOKENS)),
    "profile44":        list(range(NUM_TOKENS)),
    "profile44_tiered": list(range(NUM_TOKENS)),
    "profile_top14":    list(range(NUM_TOKENS)),
    "profile_top16":    list(range(NUM_TOKENS)),
    "profile_top20":    list(range(NUM_TOKENS)),
    "profile_tier":     list(range(NUM_TOKENS)),
    "profile_custom":   list(range(NUM_TOKENS)),
    "cape_v1":          list(range(NUM_TOKENS)),
    "cape_v2":          list(range(NUM_TOKENS)),
    "cape_v3":          list(range(NUM_TOKENS)),
    "sgr_v1":           list(range(NUM_TOKENS)),
    "sgr_v2":           list(range(NUM_TOKENS)),
    "v20_pp4":          list(range(NUM_TOKENS)),
    "tcr_v1":           list(range(NUM_TOKENS)),
    "tcr_v2":           list(range(NUM_TOKENS)),
    "tcr_v3":           list(range(NUM_TOKENS)),
    "tcr_v4":           list(range(NUM_TOKENS)),
    "tcr_v5":           list(range(NUM_TOKENS)),
    # ── Extended ablation tokens ──
    "abl_C1_no18_19":   [21, 22, 23, 24, 25, 26, 34, 35, 36, 37, 43, 44],
    "abl_N1_neg_residue": list(range(0, 27)),   # residue losses + base block
    "abl_R2p2":         [19, 20, 21, 22, 23, 24, 25, 26, 34, 35, 36, 37, 43, 44],
    "abl_R2p0":         [19, 20, 21, 22, 23, 24, 25, 26, 34, 35, 36, 37, 43, 44],
    "abl_rich_26_32":   [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    "abl_rich_26_36":   [19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
}


def build_cape_token_tiers(token_profile_prior=None, core_tokens=None,
                           tier1_n=8, tier2_n=8):
    """Build CAPE token tiers from a frozen token profile.

    Tier 0 is always core8.
    Tier 1 and Tier 2 are taken from the highest-priority non-core tokens.
    """
    if core_tokens is None:
        core_tokens = sorted(CORE8_TOKENS)
    tier0 = list(sorted(set(int(t) for t in core_tokens)))
    if token_profile_prior is None:
        # Conservative non-leaking fallback:
        # extend from physically interpretable support families only.
        fallback_order = [25, 26, 21, 22, 23, 19, 20]
        ranked = [int(t) for t in fallback_order if int(t) not in tier0]
    else:
        prior = np.asarray(token_profile_prior, dtype=np.float32)
        ranked = [int(t) for t in np.argsort(-prior) if int(t) not in tier0]
    tier1 = ranked[:max(0, int(tier1_n))]
    tier2 = ranked[max(0, int(tier1_n)):max(0, int(tier1_n)) + max(0, int(tier2_n))]
    return {
        "tier0": tier0,
        "tier1": tier1,
        "tier2": tier2,
    }


def resolve_cape_export_tokens(token_profile_prior=None, core_tokens=None,
                               tier1_n=8, tier2_n=8):
    tiers = build_cape_token_tiers(
        token_profile_prior=token_profile_prior,
        core_tokens=core_tokens,
        tier1_n=tier1_n,
        tier2_n=tier2_n,
    )
    tokens = sorted(set(tiers["tier0"] + tiers["tier1"] + tiers["tier2"]))
    return tokens, tiers


def compute_cape_cluster_budget(peptide_length, precursor_charge, mod_count_est,
                                terminal_mod_est, main_score, cluster_anchor,
                                tier1_budget=2, tier2_budget=1,
                                generation="v1"):
    """Condition-aware per-cluster CAPE budget.

    The budget is intentionally explicit and interpretable:
      - weak/sparse chronic groups get more room;
      - strong anchors get access to more profile tokens;
      - high-mod/terminal-mod cases can open extra slots.
    """
    L = int(peptide_length)
    z = int(precursor_charge)
    anchor = float(cluster_anchor)
    main = float(main_score)

    generation = str(generation)
    max_total = 5
    reserve_tier1 = int(tier1_budget)
    reserve_tier2 = int(tier2_budget)
    tail_budget = 0

    if z == 2 and L <= 16:
        max_total += 2
        reserve_tier1 += 2
        reserve_tier2 += 1
    elif z == 2 or L <= 12:
        max_total += 1
        reserve_tier1 += 1
        reserve_tier2 += 1

    if int(mod_count_est) >= 2 or bool(terminal_mod_est):
        max_total += 1
        reserve_tier2 += 2
    elif int(mod_count_est) >= 1:
        reserve_tier2 += 1

    if anchor >= 0.025 or main >= 0.020:
        reserve_tier1 += 1
    if anchor >= 0.040 or main >= 0.030:
        reserve_tier2 += 1

    if generation == "v2":
        max_total = min(max_total + 1, 10)
    elif generation == "v3":
        max_total = min(max_total + 3, 14)
        reserve_tier1 += 2
        reserve_tier2 += 2
        if z == 2 and L <= 16:
            reserve_tier1 += 2
            reserve_tier2 += 2
        elif z == 2 or L <= 12:
            reserve_tier1 += 1
            reserve_tier2 += 1
        if int(mod_count_est) >= 1 or bool(terminal_mod_est):
            reserve_tier2 += 1
    else:
        max_total = min(max_total, 8)
    reserve_tier1 = min(reserve_tier1, max_total - 1)
    reserve_tier2 = min(reserve_tier2, max(0, max_total - 1 - reserve_tier1))
    # Keep at least 1 slot for the main token and 1 spill slot for either
    # extra core or a strong tail/profile token if capacity allows.
    spill_budget = 1 if max_total >= 5 else 0
    core_budget = max(1, max_total - reserve_tier1 - reserve_tier2 - spill_budget)
    if core_budget + reserve_tier1 + reserve_tier2 > max_total:
        overflow = core_budget + reserve_tier1 + reserve_tier2 - max_total
        core_budget = max(1, core_budget - overflow)
    tail_budget = max(0, max_total - core_budget - reserve_tier1 - reserve_tier2)
    return {
        "max_total": int(max_total),
        "core_budget": int(max(1, core_budget)),
        "tier1_budget": int(max(0, reserve_tier1)),
        "tier2_budget": int(max(0, reserve_tier2)),
        "tail_budget": int(max(0, tail_budget)),
    }


def select_cape_token_items(token_items, token_profile_prior=None,
                            peptide_length=0, precursor_charge=2,
                            mod_count_est=0, terminal_mod_est=False,
                            main_score=0.0, cluster_anchor=0.0,
                            core_tokens=None, tier1_n=8, tier2_n=8,
                            generation="v1"):
    """Condition-aware token activation for CAPE.

    This replaces global whitelist search with an explicit, local controller.
    """
    if core_tokens is None:
        core_tokens = sorted(CORE8_TOKENS)
    by_tok = {int(tok): float(score) for tok, score in token_items}
    if not by_tok:
        return []

    tiers = build_cape_token_tiers(
        token_profile_prior=token_profile_prior,
        core_tokens=core_tokens,
        tier1_n=tier1_n,
        tier2_n=tier2_n,
    )
    tier0 = [t for t in tiers["tier0"] if t in by_tok]
    tier1 = [t for t in tiers["tier1"] if t in by_tok]
    tier2 = [t for t in tiers["tier2"] if t in by_tok]
    other = [t for t in by_tok if t not in set(tier0 + tier1 + tier2)]

    def _prior(tok):
        if token_profile_prior is not None:
            return float(np.asarray(token_profile_prior, dtype=np.float32)[int(tok)])
        return _token_prior_func(int(tok))

    def _rank(tokens):
        return sorted(tokens, key=lambda t: by_tok[t] * _prior(t), reverse=True)

    budget = compute_cape_cluster_budget(
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
        terminal_mod_est=terminal_mod_est,
        main_score=main_score,
        cluster_anchor=cluster_anchor,
        generation=generation,
    )

    selected = []
    used = set()

    def _add(tok_list, limit, min_score=None):
        if len(selected) >= budget["max_total"]:
            return
        limit = min(int(limit), max(0, budget["max_total"] - len(selected)))
        if limit <= 0:
            return
        added = 0
        for tok in _rank(tok_list):
            if len(selected) >= budget["max_total"]:
                break
            if tok in used:
                continue
            if min_score is not None and by_tok[tok] * _prior(tok) < float(min_score):
                continue
            selected.append((tok, by_tok[tok]))
            used.add(tok)
            added += 1
            if added >= limit or len(selected) >= budget["max_total"]:
                break

    chronic_sparse_group = (
        (int(precursor_charge) == 2 and int(peptide_length) <= 16)
        or int(peptide_length) <= 12
        or int(mod_count_est) >= 2
        or bool(terminal_mod_est)
    )
    mod_group = int(mod_count_est) >= 1 or bool(terminal_mod_est)
    very_sparse_group = (
        (int(precursor_charge) == 2 and int(peptide_length) <= 12)
        or int(peptide_length) <= 8
        or int(mod_count_est) >= 2
    )
    rescue_priority_group = (
        (int(precursor_charge) == 2 and int(peptide_length) <= 16)
        or mod_group
    )

    core_budget = int(budget["core_budget"])
    # Tier 0: keep main first, then a bounded number of core satellites so that
    # CAPE can reserve room for Tier 1 / Tier 2 instead of letting core8
    # consume the whole budget.
    _add([MAIN_TOKEN] if MAIN_TOKEN in tier0 else [], 1)
    _add([t for t in tier0 if t != MAIN_TOKEN], max(0, core_budget - len(selected)))
    # Tier 1: enabled by anchor strength and chronic sparse groups.
    tier1_gate = max(0.0035, 0.45 * max(float(main_score), float(cluster_anchor), 0.0))
    if chronic_sparse_group:
        tier1_gate *= 0.80
    if very_sparse_group:
        tier1_gate *= 0.85
    _add(tier1, budget["tier1_budget"], min_score=tier1_gate)
    # Tier 2: only open when evidence or subgroup need is stronger.
    tier2_gate = max(0.0045, 0.65 * max(float(main_score), float(cluster_anchor), 0.0))
    if chronic_sparse_group:
        tier2_gate *= 0.80
    if mod_group:
        tier2_gate *= 0.90
    if very_sparse_group:
        tier2_gate *= 0.85
    _add(tier2, budget["tier2_budget"], min_score=tier2_gate)
    # Refill from remaining core tokens if tier reservations were not fully
    # used, then allow a small tail budget for very strong non-tier evidence.
    remaining_core = [t for t in tier0 if t != MAIN_TOKEN and t not in used]
    if len(selected) < budget["max_total"] and remaining_core:
        _add(remaining_core, budget["max_total"] - len(selected))
    if len(selected) < budget["max_total"]:
        tail_gate = max(0.008, 0.90 * max(float(main_score), float(cluster_anchor), 0.0))
        tail_limit = min(budget["max_total"] - len(selected), max(0, budget.get("tail_budget", 0)))
        if rescue_priority_group and tail_limit <= 0 and len(selected) < budget["max_total"]:
            # Chronic sparse groups may borrow a single spill slot from the
            # remaining global headroom when strong tail evidence exists.
            tail_limit = 1
        _add(other, tail_limit, min_score=tail_gate)

    budget["used_total"] = int(len(selected))
    budget["used_tier0"] = int(sum(1 for tok, _ in selected if tok in set(tier0)))
    budget["used_tier1"] = int(sum(1 for tok, _ in selected if tok in set(tier1)))
    budget["used_tier2"] = int(sum(1 for tok, _ in selected if tok in set(tier2)))
    budget["used_tail"] = int(max(0, len(selected) - budget["used_tier0"] - budget["used_tier1"] - budget["used_tier2"]))
    return selected, tiers, budget


def compute_cape_evidence_cap(N_by, peptide_length, precursor_charge,
                              mod_count_est, terminal_mod_est,
                              min_peaks, mid_main_count, strong_main_count):
    """CAPE-native evidence cap controller."""
    L = int(peptide_length)
    z = int(precursor_charge)
    evidence_cap = int(round(14 + 1.85 * mid_main_count + 2.75 * strong_main_count))
    if z == 2:
        evidence_cap += int(round(N_by * 0.08))
    if L <= 12:
        evidence_cap += int(round(N_by * 0.06))
    if int(mod_count_est) >= 1:
        evidence_cap += int(round(N_by * 0.10))
    if int(mod_count_est) >= 2:
        evidence_cap += int(round(N_by * 0.12))
    if bool(terminal_mod_est):
        evidence_cap += int(round(N_by * 0.08))

    if z == 2 and L <= 16 and int(mod_count_est) <= 1:
        z2_cap_ratio = 1.60 if L <= 12 else 1.75
        z2_floor = 60 if L <= 8 else 75
        evidence_cap = min(evidence_cap, int(max(min_peaks, z2_floor, N_by * z2_cap_ratio)))
    elif z == 2:
        evidence_cap = min(evidence_cap, int(max(80, N_by * 2.15)))
    elif L <= 16:
        evidence_cap = min(evidence_cap, int(max(150, N_by * 2.20)))

    return max(int(min_peaks), int(evidence_cap))


def compute_cape_sparse_support(peptide_length, precursor_charge,
                                mod_count_est, terminal_mod_est,
                                mid_main_count, strong_main_count):
    """Whether CAPE should allow sparse rescue for the current spectrum."""
    L = int(peptide_length)
    z = int(precursor_charge)
    sparse_mid_need = max(3, min(L, 16) // 2)
    sparse_strong_need = max(2, min(L, 16) // 4)
    supports = (
        mid_main_count >= sparse_mid_need
        or strong_main_count >= sparse_strong_need
        or (z == 2 and mid_main_count >= 3)
        or (L <= 12 and mid_main_count >= 3)
        or (int(mod_count_est) >= 1 and strong_main_count >= 2)
        or (bool(terminal_mod_est) and strong_main_count >= 2)
    )
    return bool(supports), int(sparse_mid_need), int(sparse_strong_need)


def compute_cape_dense_guard_start(peptide_length, precursor_charge, mod_count_est):
    """CAPE-native dense-control trigger ratio."""
    L = int(peptide_length)
    z = int(precursor_charge)
    if z == 2 and L <= 16:
        return 1.85 if int(mod_count_est) <= 1 else 2.10
    if z == 2:
        return 2.10
    if L <= 16:
        return 2.10
    return 2.30


def compute_cape_retry_budgets(peptide_length, mod_count_est, terminal_mod_est):
    """CAPE-native retry budgets for sparse recovery."""
    L = int(peptide_length)
    core_budget = 4 if L <= 12 else 3
    tier1_budget = 3 if L <= 12 else 2
    tier2_budget = 2 if (int(mod_count_est) >= 1 or bool(terminal_mod_est) or L <= 12) else 1
    max_tokens = 8 if L <= 8 else (7 if L <= 12 else 6)
    max_sat = 7 if L <= 8 else (6 if L <= 12 else 5)
    return {
        "core_budget": int(core_budget),
        "tier1_budget": int(tier1_budget),
        "tier2_budget": int(tier2_budget),
        "max_tokens": int(max_tokens),
        "max_sat": int(max_sat),
    }


def should_apply_cape_dense_cap(peptide_length, precursor_charge, mod_count_est,
                                over_dense_ratio, dense_guard_start,
                                strong_main_count, mid_main_count):
    """CAPE-native dense-cap trigger."""
    L = int(peptide_length)
    z = int(precursor_charge)
    weak_anchor = (
        strong_main_count < max(6, L // 2)
        and mid_main_count < max(12, L)
    )
    chronic_overfill = (
        z == 2
        and L <= 16
        and int(mod_count_est) <= 1
        and float(over_dense_ratio) >= float(dense_guard_start)
    )
    return bool(float(over_dense_ratio) >= float(dense_guard_start) and (weak_anchor or chronic_overfill)), bool(weak_anchor), bool(chronic_overfill)


def build_cape_state_summary(N_by, peptide_length, precursor_charge,
                             mod_count_est, terminal_mod_est,
                             min_peaks, max_peaks,
                             strong_main_count, mid_main_count,
                             over_dense_ratio,
                             generation="v1"):
    """Compute CAPE-native state-control summary in one place."""
    evidence_cap = compute_cape_evidence_cap(
        N_by=N_by,
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
        terminal_mod_est=terminal_mod_est,
        min_peaks=min_peaks,
        mid_main_count=mid_main_count,
        strong_main_count=strong_main_count,
    )
    evidence_supports_sparse, sparse_mid_need, sparse_strong_need = compute_cape_sparse_support(
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
        terminal_mod_est=terminal_mod_est,
        mid_main_count=mid_main_count,
        strong_main_count=strong_main_count,
    )
    dense_guard_start = compute_cape_dense_guard_start(
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
    )
    if str(generation) == "v2":
        dense_guard_start += 0.20  # delay dense suppression; recover first
    elif str(generation) == "v3":
        dense_guard_start += 0.45  # recovery first, dense suppression much later
    apply_dense_cap, weak_anchor, chronic_overfill = should_apply_cape_dense_cap(
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
        over_dense_ratio=over_dense_ratio,
        dense_guard_start=dense_guard_start,
        strong_main_count=strong_main_count,
        mid_main_count=mid_main_count,
    )
    retry = compute_cape_retry_budgets(
        peptide_length=peptide_length,
        mod_count_est=mod_count_est,
        terminal_mod_est=terminal_mod_est,
    )
    if str(generation) == "v2":
        retry["core_budget"] = max(retry["core_budget"], 5)
        retry["tier1_budget"] = max(retry["tier1_budget"], 4)
        retry["tier2_budget"] = max(retry["tier2_budget"], 3)
        retry["max_tokens"] = max(retry["max_tokens"], 9)
        retry["max_sat"] = max(retry["max_sat"], 8)
    elif str(generation) == "v3":
        retry["core_budget"] = max(retry["core_budget"], 6)
        retry["tier1_budget"] = max(retry["tier1_budget"], 6)
        retry["tier2_budget"] = max(retry["tier2_budget"], 5)
        retry["max_tokens"] = max(retry["max_tokens"], 12)
        retry["max_sat"] = max(retry["max_sat"], 10)
    summary = {
        "generation": str(generation),
        "evidence_cap": int(min(max(evidence_cap, min_peaks), max_peaks)),
        "evidence_supports_sparse": bool(evidence_supports_sparse),
        "sparse_mid_need": int(sparse_mid_need),
        "sparse_strong_need": int(sparse_strong_need),
        "dense_guard_start": float(dense_guard_start),
        "apply_dense_cap": bool(apply_dense_cap),
        "weak_anchor_for_dense": bool(weak_anchor),
        "chronic_overfill": bool(chronic_overfill),
        "retry_core_budget": int(retry["core_budget"]),
        "retry_tier1_budget": int(retry["tier1_budget"]),
        "retry_tier2_budget": int(retry["tier2_budget"]),
        "retry_max_tokens": int(retry["max_tokens"]),
        "retry_max_sat": int(retry["max_sat"]),
    }
    return summary


def build_sgr_state_summary(cells, pred_matrix, peptide_length, precursor_charge,
                            mod_count_est, terminal_mod_est,
                            min_peaks, target_peaks,
                            main_deficit_ratio=0.50,
                            support_deficit_ratio=0.70,
                            total_deficit_ratio=1.00):
    """Estimate whole-spectrum deficit for SGR spectrum-first recovery."""
    L = int(peptide_length)
    z = int(precursor_charge)
    main_positions = {
        int(c["pos"])
        for c in cells
        if c.get("is_main", False) and "pos" in c
    }
    covered_positions = {
        int(c["pos"])
        for c in cells
        if "pos" in c
    }
    n_main = len(main_positions)
    n_total = len(cells)
    n_covered_positions = len(covered_positions)
    support_positions = {
        int(c["pos"])
        for c in cells
        if not c.get("is_main", False) and "pos" in c
    }
    n_support_positions = len(support_positions)
    n_by = max(L - 1, 0) * 2 * min(z, 3)
    max_positions = max(L - 1, 0)
    main_target = max(1, min(max_positions, int(round(max_positions * float(main_deficit_ratio)))))
    position_target = max(1, min(max_positions, int(round(max_positions * min(float(total_deficit_ratio), 1.35)))))
    support_target = max(1, min(max_positions, int(round(max_positions * float(support_deficit_ratio)))))
    total_target = max(int(min_peaks), int(round(target_peaks * float(total_deficit_ratio))))
    subgroup_z2_short = bool(z == 2 and L <= 16)
    subgroup_short = bool(L <= 12)
    subgroup_mod = bool(int(mod_count_est) >= 1 or bool(terminal_mod_est))
    main_deficit = max(0, main_target - n_main)
    position_deficit = max(0, position_target - n_covered_positions)
    support_deficit = max(0, support_target - n_support_positions)
    total_deficit = max(position_deficit, support_deficit, max(0, total_target - n_total))
    return {
        "n_by": int(n_by),
        "n_main": int(n_main),
        "n_total": int(n_total),
        "n_covered_positions": int(n_covered_positions),
        "n_support_positions": int(n_support_positions),
        "max_positions": int(max_positions),
        "main_target": int(main_target),
        "position_target": int(position_target),
        "support_target": int(support_target),
        "total_target": int(total_target),
        "main_deficit": int(main_deficit),
        "position_deficit": int(position_deficit),
        "support_deficit": int(support_deficit),
        "total_deficit": int(total_deficit),
        "subgroup_z2_short": bool(subgroup_z2_short),
        "subgroup_short": bool(subgroup_short),
        "subgroup_mod": bool(subgroup_mod),
    }


def run_sgr_global_recovery(cells, pred_matrix, input_mz, token_profile_prior,
                            peptide_length, precursor_charge,
                            mod_count_est, terminal_mod_est,
                            min_peaks, target_peaks, max_peaks,
                            t_main_rescue, t_sat_rescue,
                            residues=None,
                            sgr_generation="v1",
                            sgr_main_deficit_ratio=0.50,
                            sgr_support_deficit_ratio=0.70,
                            sgr_total_deficit_ratio=1.00,
                            sgr_global_main_budget=48,
                            sgr_global_sat_budget=96,
                            sgr_global_tier1_budget=96,
                            sgr_global_tier2_budget=72, *, score_builder=rerank_export_score):
    """Stage-B/C SGR hook: spectrum-first global recovery skeleton.

    Current version focuses on restoring the lower bound:
    - recover missing main peaks globally;
    - then recover strong core/profile satellites globally;
    - defer finer redistribution rules to later SGR iterations.
    """
    sgr_state = build_sgr_state_summary(
        cells=cells,
        pred_matrix=pred_matrix,
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
        terminal_mod_est=terminal_mod_est,
        min_peaks=min_peaks,
        target_peaks=target_peaks,
        main_deficit_ratio=sgr_main_deficit_ratio,
        support_deficit_ratio=sgr_support_deficit_ratio,
        total_deficit_ratio=sgr_total_deficit_ratio,
    )
    diag = {"sgr_state": dict(sgr_state)}
    if sgr_state["total_deficit"] <= 0 and sgr_state["main_deficit"] <= 0:
        diag["sgr_added_main"] = 0
        diag["sgr_added_sat"] = 0
        diag["sgr_added_tier1"] = 0
        diag["sgr_added_tier2"] = 0
        diag["sgr_main_candidate_pool"] = 0
        diag["sgr_sat_candidate_pool"] = 0
        diag["sgr_tier1_candidate_pool"] = 0
        diag["sgr_tier2_candidate_pool"] = 0
        diag["sgr_main_deficit"] = int(sgr_state["main_deficit"])
        diag["sgr_position_deficit"] = int(sgr_state["position_deficit"])
        diag["sgr_support_deficit"] = int(sgr_state["support_deficit"])
        diag["sgr_total_deficit"] = int(sgr_state["total_deficit"])
        diag["n_main_rescued"] = 0
        diag["n_sat_rescued"] = 0
        return cells, diag

    ch_mask = build_charge_channel_mask(precursor_charge)
    existing_main = {
        (int(c["pos"]) - 1, CHANNEL_NAMES.index(c["channel"]))
        for c in cells
        if c.get("is_main", False) and c.get("channel") in CHANNEL_NAMES
    }
    existing_pairs = {
        (int(c["pos"]) - 1, int(c["token_idx"]), CHANNEL_NAMES.index(c["channel"]))
        for c in cells
        if c.get("channel") in CHANNEL_NAMES
    }

    def _prior(tok):
        if token_profile_prior is not None:
            return float(token_profile_prior[int(tok)])
        return _token_prior_func(int(tok))

    main_candidates = []
    sat_candidates = []
    tier1_candidates = []
    tier2_candidates = []
    L = int(peptide_length)
    z = int(precursor_charge)
    sgr_generation = str(sgr_generation)
    for pos in range(pred_matrix.shape[0]):
        if pos >= L - 1:
            continue
        for ch in range(N_CHANNELS):
            if not ch_mask[ch]:
                continue
            main_score = float(pred_matrix[pos, MAIN_TOKEN, ch])
            if (pos, ch) not in existing_main:
                if main_score >= t_main_rescue:
                    main_candidates.append({
                        "mz": float(input_mz[pos, MAIN_TOKEN, ch]),
                        "intensity": main_score,
                        "norm_intensity": main_score,
                        "support_score": 1.0,
                    "export_score": score_builder(
                        model_score=main_score,
                        tok=MAIN_TOKEN,
                        kind="main",
                        main_score=main_score,
                        pos=pos,
                        channel=ch,
                        pred_matrix=pred_matrix,
                        peptide_length=peptide_length,
                        precursor_charge=precursor_charge,
                        residues=residues,
                        token_profile_prior=token_profile_prior,
                    ),
                        "pos": int(pos + 1),
                        "token_idx": MAIN_TOKEN,
                        "channel": CHANNEL_NAMES[ch],
                        "is_main": True,
                        "_token_kind": "main",
                        "is_rescue": True,
                    })
            sat_tokens = [t for t in sorted(CORE8_TOKENS) if t != MAIN_TOKEN]
            tier1_tokens = [33, 34, 35, 20, 29, 27]
            tier2_tokens = [37, 13, 30, 36, 42, 43, 28]
            if z == 2 and L <= 16:
                tier1_tokens = [33, 34, 35, 20, 29, 27, 37, 13]
                tier2_tokens = [30, 36, 42, 43, 28]
            elif int(mod_count_est) >= 1 or bool(terminal_mod_est):
                tier1_tokens = [33, 34, 35, 20, 29, 27]
                tier2_tokens = [37, 13, 30, 36]
            seen = set()
            sat_tokens = [tok for tok in sat_tokens if not (tok in seen or seen.add(tok))]
            for tok in sat_tokens:
                key = (pos, int(tok), ch)
                if key in existing_pairs:
                    continue
                score = float(pred_matrix[pos, tok, ch])
                if score < t_sat_rescue:
                    continue
                sat_candidates.append({
                    "mz": float(input_mz[pos, tok, ch]),
                    "intensity": score,
                    "norm_intensity": score,
                    "support_score": 1.0,
                    "export_score": score_builder(
                        model_score=score,
                        tok=tok,
                        kind=TOKEN_KIND.get(int(tok), "other"),
                        main_score=main_score,
                        pos=pos,
                        channel=ch,
                        pred_matrix=pred_matrix,
                        peptide_length=peptide_length,
                        precursor_charge=precursor_charge,
                        residues=residues,
                        token_profile_prior=token_profile_prior,
                    ),
                    "pos": int(pos + 1),
                    "token_idx": int(tok),
                    "channel": CHANNEL_NAMES[ch],
                    "is_main": False,
                    "_token_kind": TOKEN_KIND.get(int(tok), "other"),
                    "is_rescue": True,
                })
            for tok in tier1_tokens:
                key = (pos, int(tok), ch)
                if key in existing_pairs:
                    continue
                score = float(pred_matrix[pos, tok, ch])
                if score < t_sat_rescue:
                    continue
                tier1_candidates.append({
                    "mz": float(input_mz[pos, tok, ch]),
                    "intensity": score,
                    "norm_intensity": score,
                    "support_score": 1.0,
                    "export_score": score_builder(
                        model_score=score,
                        tok=tok,
                        kind=TOKEN_KIND.get(int(tok), "other"),
                        main_score=main_score,
                        pos=pos,
                        channel=ch,
                        pred_matrix=pred_matrix,
                        peptide_length=peptide_length,
                        precursor_charge=precursor_charge,
                        residues=residues,
                        token_profile_prior=token_profile_prior,
                    ),
                    "pos": int(pos + 1),
                    "token_idx": int(tok),
                    "channel": CHANNEL_NAMES[ch],
                    "is_main": False,
                    "_token_kind": TOKEN_KIND.get(int(tok), "other"),
                    "is_rescue": True,
                })
            if sgr_generation == "v2":
                for tok in tier2_tokens:
                    key = (pos, int(tok), ch)
                    if key in existing_pairs:
                        continue
                    score = float(pred_matrix[pos, tok, ch])
                    if score < max(0.0035, t_sat_rescue * 0.85):
                        continue
                    tier2_candidates.append({
                        "mz": float(input_mz[pos, tok, ch]),
                        "intensity": score,
                        "norm_intensity": score,
                        "support_score": 1.0,
                        "export_score": score_builder(
                            model_score=score,
                            tok=tok,
                            kind=TOKEN_KIND.get(int(tok), "other"),
                            main_score=main_score,
                            pos=pos,
                            channel=ch,
                            pred_matrix=pred_matrix,
                            peptide_length=peptide_length,
                            precursor_charge=precursor_charge,
                            residues=residues,
                            token_profile_prior=token_profile_prior,
                        ),
                        "pos": int(pos + 1),
                        "token_idx": int(tok),
                        "channel": CHANNEL_NAMES[ch],
                        "is_main": False,
                        "_token_kind": TOKEN_KIND.get(int(tok), "other"),
                        "is_rescue": True,
                    })

    main_candidates.sort(key=lambda c: -float(c["export_score"]))
    sat_candidates.sort(key=lambda c: -float(c["export_score"]))
    tier1_candidates.sort(key=lambda c: -float(c["export_score"]))
    tier2_candidates.sort(key=lambda c: -float(c["export_score"]))
    main_budget = min(max(0, sgr_state["main_deficit"] * 4), int(sgr_global_main_budget))
    sat_budget = min(max(0, sgr_state["support_deficit"] * 6), int(sgr_global_sat_budget))
    tier1_budget = min(max(0, max(sgr_state["position_deficit"], sgr_state["support_deficit"]) * 6), int(sgr_global_tier1_budget))
    tier2_budget = min(max(0, max(sgr_state["position_deficit"], sgr_state["support_deficit"]) * 4), int(sgr_global_tier2_budget))
    if sgr_state["subgroup_z2_short"]:
        main_budget = min(max(main_budget, 18), int(max(72, sgr_global_main_budget)))
        sat_budget = min(max(sat_budget, 48), int(max(180, sgr_global_sat_budget)))
        tier1_budget = min(max(tier1_budget, 64), int(max(220, sgr_global_tier1_budget)))
        tier2_budget = min(max(tier2_budget, 32), int(max(128, sgr_global_tier2_budget)))
    elif sgr_state["subgroup_short"] or sgr_state["subgroup_mod"]:
        main_budget = min(max(main_budget, 12), int(max(60, sgr_global_main_budget)))
        sat_budget = min(max(sat_budget, 32), int(max(144, sgr_global_sat_budget)))
        tier1_budget = min(max(tier1_budget, 40), int(max(160, sgr_global_tier1_budget)))
        tier2_budget = min(max(tier2_budget, 20), int(max(96, sgr_global_tier2_budget)))

    new_cells = list(cells)
    new_cells.extend(main_candidates[:main_budget])
    new_cells.extend(sat_candidates[:sat_budget])
    new_cells.extend(tier1_candidates[:tier1_budget])
    if sgr_generation == "v2":
        new_cells.extend(tier2_candidates[:tier2_budget])
    if len(new_cells) > int(max_peaks):
        new_cells = sorted(new_cells, key=lambda c: (not c.get("is_main", False), -float(c.get("export_score", 0.0))))[:int(max_peaks)]

    diag["sgr_added_main"] = int(min(main_budget, len(main_candidates)))
    diag["sgr_added_sat"] = int(min(sat_budget, len(sat_candidates)))
    diag["sgr_added_tier1"] = int(min(tier1_budget, len(tier1_candidates)))
    diag["sgr_added_tier2"] = int(min(tier2_budget, len(tier2_candidates)))
    diag["sgr_main_candidate_pool"] = int(len(main_candidates))
    diag["sgr_sat_candidate_pool"] = int(len(sat_candidates))
    diag["sgr_tier1_candidate_pool"] = int(len(tier1_candidates))
    diag["sgr_tier2_candidate_pool"] = int(len(tier2_candidates))
    diag["sgr_main_deficit"] = int(sgr_state["main_deficit"])
    diag["sgr_position_deficit"] = int(sgr_state["position_deficit"])
    diag["sgr_support_deficit"] = int(sgr_state["support_deficit"])
    diag["sgr_total_deficit"] = int(sgr_state["total_deficit"])
    diag["n_main_rescued"] = int(min(main_budget, len(main_candidates)))
    diag["n_sat_rescued"] = int(
        min(sat_budget, len(sat_candidates))
        + min(tier1_budget, len(tier1_candidates))
        + min(tier2_budget, len(tier2_candidates))
    )
    return new_cells, diag


def extract_tcr_anchor_positions(filtered, token_profile_prior, peptide_length,
                                 precursor_charge, t_main, t_sat,
                                 anchor_main_ratio=0.82, anchor_support_ratio=0.55):
    """Extract sparse theory anchors for TCR reconstruction.

    Returns anchor records keyed by cleavage position and channel.
    """
    L = int(peptide_length)
    ch_mask = build_charge_channel_mask(precursor_charge)
    anchors = []
    for pos in range(filtered.shape[0]):
        if pos >= L - 1:
            continue
        for ch in range(N_CHANNELS):
            if not ch_mask[ch]:
                continue
            main_score = float(filtered[pos, MAIN_TOKEN, ch])
            if main_score < float(t_main) * float(anchor_main_ratio):
                continue
            support_score = float(filtered[pos, SATELLITE_TOKENS, ch].max()) if len(SATELLITE_TOKENS) else 0.0
            if support_score < float(t_sat) * float(anchor_support_ratio):
                continue
            anchor_score = main_score + 0.35 * support_score
            anchors.append({
                "pos": int(pos + 1),
                "channel_idx": int(ch),
                "channel": CHANNEL_NAMES[ch],
                "main_score": float(main_score),
                "support_score": float(support_score),
                "anchor_score": float(anchor_score),
            })
    anchors.sort(key=lambda a: -float(a["anchor_score"]))
    return anchors


def build_tcr_family_proposals(filtered, input_mz, anchors, token_profile_prior,
                               peptide_length, precursor_charge,
                               t_main, t_sat, tcr_generation="v1",
                               proposal_main_ratio=1.00,
                               proposal_support_ratio=1.00, *, residues=None,
                               score_builder=rerank_export_score):
    """Build family proposals around each active theory anchor."""
    proposals = []
    L = int(peptide_length)
    z = int(precursor_charge)

    def _prior(tok):
        if token_profile_prior is not None:
            return float(token_profile_prior[int(tok)])
        return _token_prior_func(int(tok))

    def _profile_extra_tokens(limit=14):
        if token_profile_prior is None:
            return []
        prior = np.asarray(token_profile_prior, dtype=np.float32)
        ranked = [int(t) for t in np.argsort(-prior) if int(t) not in CORE8_TOKENS and int(t) != MAIN_TOKEN]
        return ranked[:max(0, int(limit))]

    for anchor in anchors:
        pos0 = int(anchor["pos"]) - 1
        ch = int(anchor["channel_idx"])
        if pos0 < 0 or pos0 >= filtered.shape[0] or pos0 >= L - 1:
            continue
        main_score = float(filtered[pos0, MAIN_TOKEN, ch])
        if main_score >= float(t_main) * float(proposal_main_ratio):
            proposals.append({
                "mz": float(input_mz[pos0, MAIN_TOKEN, ch]),
                "intensity": main_score,
                "norm_intensity": main_score,
                "support_score": 1.0,
                "export_score": score_builder(
                    model_score=main_score,
                    tok=MAIN_TOKEN,
                    kind="main",
                    main_score=main_score,
                    pos=pos0,
                    channel=ch,
                    pred_matrix=filtered,
                    peptide_length=peptide_length,
                    precursor_charge=precursor_charge,
                    residues=residues,
                    token_profile_prior=token_profile_prior,
                ),
                "pos": int(pos0 + 1),
                "token_idx": MAIN_TOKEN,
                "channel": CHANNEL_NAMES[ch],
                "is_main": True,
                "_token_kind": "main",
                "tcr_family": "main",
            })
        family_tokens = [25, 26, 21, 22, 23]
        extra_limit = 0
        if z == 2 and L <= 16:
            extra_limit = 10
        elif z == 2 or L <= 12:
            extra_limit = 6
        elif str(tcr_generation) in ("v3", "v4", "v5"):
            extra_limit = 8
        if str(tcr_generation) in ("v3", "v4", "v5"):
            extra_limit = max(extra_limit, 12)
        family_tokens.extend(_profile_extra_tokens(limit=extra_limit))
        seen = set()
        family_tokens = [tok for tok in family_tokens if not (tok in seen or seen.add(tok)) and tok != MAIN_TOKEN]
        for tok in family_tokens:
            score = float(filtered[pos0, tok, ch])
            if score < float(t_sat) * float(proposal_support_ratio):
                continue
            proposals.append({
                "mz": float(input_mz[pos0, tok, ch]),
                "intensity": score,
                "norm_intensity": score,
                "support_score": 1.0,
                "export_score": score_builder(
                    model_score=score,
                    tok=tok,
                    kind=TOKEN_KIND.get(int(tok), "other"),
                    main_score=main_score,
                    pos=pos0,
                    channel=ch,
                    pred_matrix=filtered,
                    peptide_length=peptide_length,
                    precursor_charge=precursor_charge,
                    residues=residues,
                    token_profile_prior=token_profile_prior,
                ),
                "pos": int(pos0 + 1),
                "token_idx": int(tok),
                "channel": CHANNEL_NAMES[ch],
                "is_main": False,
                "_token_kind": TOKEN_KIND.get(int(tok), "other"),
                "tcr_family": TOKEN_KIND.get(int(tok), "other"),
            })
    proposals.sort(key=lambda c: (not c.get("is_main", False), -float(c.get("export_score", 0.0))))
    return proposals


def reconstruct_tcr_cells(filtered, input_mz, token_profile_prior,
                          peptide_length, precursor_charge,
                          min_peaks, target_peaks, max_peaks,
                          t_main, t_sat,
                          anchor_main_ratio=0.82, anchor_support_ratio=0.55,
                          anchor_floor_ratio=0.75,
                          main_floor_ratio=0.72, support_floor_ratio=0.90,
                          support_budget_ratio=0.90,
                          position_backfill_ratio=0.0,
                          proposal_main_ratio=1.00,
                          proposal_support_ratio=1.00,
                          family_cap_main=1, family_cap_support=3,
                          global_main_budget=96, global_support_budget=192,
                          position_support_floor=1,
                          weak_position_bonus_ratio=0.0,
                          weak_support_bonus_ratio=0.0,
                          weak_backfill_bonus_ratio=0.0,
                          mod_count_est=0,
                          terminal_mod_est=False,
                          tcr_generation="v1", *, residues=None,
                          score_builder=rerank_export_score):
    """Run TCR sparse anchor extraction + constrained reconstruction."""
    L = int(peptide_length)
    z = int(precursor_charge)
    subgroup_z2 = (z == 2)
    subgroup_short = (L <= 12)
    subgroup_z2_short = (z == 2 and L <= 16)
    subgroup_mod = bool(int(mod_count_est) >= 1)
    subgroup_terminal = bool(terminal_mod_est)
    weak_group = False
    if str(tcr_generation) == "v5":
        weak_group = subgroup_z2 or subgroup_short or subgroup_z2_short or subgroup_mod or subgroup_terminal
    position_backfill_ratio_eff = float(position_backfill_ratio)
    if weak_group:
        position_backfill_ratio_eff = min(0.95, position_backfill_ratio_eff + float(weak_backfill_bonus_ratio))

    anchors = extract_tcr_anchor_positions(
        filtered=filtered,
        token_profile_prior=token_profile_prior,
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        t_main=t_main,
        t_sat=t_sat,
        anchor_main_ratio=anchor_main_ratio,
        anchor_support_ratio=anchor_support_ratio,
    )
    proposals = build_tcr_family_proposals(
        filtered=filtered,
        input_mz=input_mz,
        anchors=anchors,
        token_profile_prior=token_profile_prior,
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        t_main=t_main,
        t_sat=t_sat,
        tcr_generation=tcr_generation,
        proposal_main_ratio=proposal_main_ratio,
        proposal_support_ratio=proposal_support_ratio,
        residues=residues,
        score_builder=score_builder,
    )
    if float(position_backfill_ratio_eff) > 0:
        ch_mask = build_charge_channel_mask(precursor_charge)
        anchor_positions = {(int(a["pos"]), int(a["channel_idx"])) for a in anchors}
        backfill = []
        for pos in range(filtered.shape[0]):
            if pos >= L - 1:
                continue
            for ch in range(N_CHANNELS):
                if not ch_mask[ch]:
                    continue
                if (int(pos + 1), int(ch)) in anchor_positions:
                    continue
                main_score = float(filtered[pos, MAIN_TOKEN, ch])
                support_score = float(filtered[pos, SATELLITE_TOKENS, ch].max()) if len(SATELLITE_TOKENS) else 0.0
                if main_score < float(t_main) * float(position_backfill_ratio_eff):
                    continue
                backfill.append({
                    "pos": int(pos + 1),
                    "channel_idx": int(ch),
                    "channel": CHANNEL_NAMES[ch],
                    "main_score": float(main_score),
                    "support_score": float(support_score),
                    "anchor_score": float(main_score + 0.20 * support_score),
                })
        backfill.sort(key=lambda a: -float(a["anchor_score"]))
        if backfill:
            proposals.extend(
                build_tcr_family_proposals(
                    filtered=filtered,
                    input_mz=input_mz,
                    anchors=backfill,
                    token_profile_prior=token_profile_prior,
                    peptide_length=peptide_length,
                    precursor_charge=precursor_charge,
                    t_main=t_main,
                    t_sat=t_sat,
                    tcr_generation=tcr_generation,
                    proposal_main_ratio=max(0.75, float(proposal_main_ratio) * 0.90),
                    proposal_support_ratio=max(0.65, float(proposal_support_ratio) * 0.85),
                )
            )
            proposals.sort(key=lambda c: (not c.get("is_main", False), -float(c.get("export_score", 0.0))))
    max_positions = max(L - 1, 0)
    anchor_floor = max(1, min(max_positions, int(round(max_positions * float(anchor_floor_ratio)))))
    if len(anchors) < anchor_floor:
        fallback_anchors = []
        ch_mask = build_charge_channel_mask(precursor_charge)
        for pos in range(filtered.shape[0]):
            if pos >= L - 1:
                continue
            for ch in range(N_CHANNELS):
                if not ch_mask[ch]:
                    continue
                main_score = float(filtered[pos, MAIN_TOKEN, ch])
                support_score = float(filtered[pos, SATELLITE_TOKENS, ch].max()) if len(SATELLITE_TOKENS) else 0.0
                anchor_score = main_score + 0.25 * support_score
                fallback_anchors.append({
                    "pos": int(pos + 1),
                    "channel_idx": int(ch),
                    "channel": CHANNEL_NAMES[ch],
                    "main_score": float(main_score),
                    "support_score": float(support_score),
                    "anchor_score": float(anchor_score),
                })
        fallback_anchors.sort(key=lambda a: -float(a["anchor_score"]))
        used = {(a["pos"], a["channel_idx"]) for a in anchors}
        for a in fallback_anchors:
            key = (a["pos"], a["channel_idx"])
            if key in used:
                continue
            anchors.append(a)
            used.add(key)
            if len(anchors) >= anchor_floor:
                break
        proposals = build_tcr_family_proposals(
            filtered=filtered,
            input_mz=input_mz,
            anchors=anchors,
            token_profile_prior=token_profile_prior,
            peptide_length=peptide_length,
            precursor_charge=precursor_charge,
            t_main=t_main,
            t_sat=t_sat,
            tcr_generation=tcr_generation,
        )
    main_floor = max(1, min(max_positions, int(round(max_positions * float(main_floor_ratio)))))
    support_floor = max(1, min(max_positions, int(round(max_positions * float(support_floor_ratio)))))
    if weak_group:
        main_floor = min(max_positions, max(main_floor, int(round(main_floor * (1.0 + float(weak_position_bonus_ratio))))))
        support_floor = min(max_positions, max(support_floor, int(round(support_floor * (1.0 + float(weak_support_bonus_ratio))))))
    main_budget = min(int(global_main_budget), max(main_floor, int(round(target_peaks * 0.25))))
    support_budget = min(int(global_support_budget), max(support_floor, int(round(target_peaks * float(support_budget_ratio)))))
    if weak_group:
        main_budget = min(int(global_main_budget), max(main_budget, int(round(main_budget * (1.0 + float(weak_position_bonus_ratio))))))
        support_budget = min(int(global_support_budget), max(support_budget, int(round(support_budget * (1.0 + float(weak_support_bonus_ratio))))))
    selected = []
    used_main_positions = set()
    used_support_per_pos = {}
    support_selected = 0
    phase_main_added = 0
    phase_support_floor_added = 0
    phase_weak_recovery_added = 0
    phase_support_fill_added = 0

    def _phase_cell(cell, phase_name):
        out = dict(cell)
        out["_tcr_phase"] = str(phase_name)
        return out

    main_cells = [c for c in proposals if c.get("is_main", False)]
    support_cells = [c for c in proposals if not c.get("is_main", False)]

    main_by_pos = {}
    for cell in main_cells:
        pos = int(cell["pos"])
        if pos not in main_by_pos:
            main_by_pos[pos] = cell

    support_by_pos = {}
    for cell in support_cells:
        pos = int(cell["pos"])
        support_by_pos.setdefault(pos, []).append(cell)

    ordered_positions = []
    seen_positions = set()
    for anchor in anchors:
        pos = int(anchor["pos"])
        if pos not in seen_positions:
            ordered_positions.append(pos)
            seen_positions.add(pos)
    for pos, cell in sorted(main_by_pos.items(), key=lambda kv: -float(kv[1].get("export_score", 0.0))):
        if pos not in seen_positions:
            ordered_positions.append(pos)
            seen_positions.add(pos)

    for pos in ordered_positions:
        if pos in used_main_positions:
            continue
        cell = main_by_pos.get(pos)
        if cell is None:
            continue
        if len(used_main_positions) >= main_budget:
            break
        selected.append(_phase_cell(cell, "main_floor"))
        used_main_positions.add(pos)
        phase_main_added += 1

    target_main_floor = min(len(used_main_positions), main_floor) if len(used_main_positions) < main_floor else main_floor
    if len(used_main_positions) < main_floor:
        for pos, cell in sorted(main_by_pos.items(), key=lambda kv: -float(kv[1].get("export_score", 0.0))):
            if pos in used_main_positions:
                continue
            if len(used_main_positions) >= main_floor or len(used_main_positions) >= main_budget:
                break
            selected.append(_phase_cell(cell, "main_floor"))
            used_main_positions.add(pos)
            phase_main_added += 1

    support_floor_positions = list(used_main_positions)
    if weak_group:
        support_floor_positions = sorted(
            support_floor_positions,
            key=lambda pos: (
                -(len(support_by_pos.get(pos, [])) > 0),
                -float(main_by_pos.get(pos, {}).get("export_score", 0.0)) if pos in main_by_pos else 0.0,
            ),
        )
    per_position_support_floor = max(1, int(position_support_floor))
    if weak_group:
        per_position_support_floor = min(int(family_cap_support), max(per_position_support_floor, int(round(per_position_support_floor * (1.0 + float(weak_support_bonus_ratio))))))
    for pos in support_floor_positions:
        candidates = support_by_pos.get(pos, [])
        if not candidates:
            continue
        for cell in candidates:
            if support_selected >= support_budget:
                break
            per_pos = used_support_per_pos.get(pos, 0)
            if per_pos >= int(family_cap_support) or per_pos >= per_position_support_floor:
                break
            selected.append(_phase_cell(cell, "support_floor"))
            used_support_per_pos[pos] = per_pos + 1
            support_selected += 1
            phase_support_floor_added += 1
        if support_selected >= support_budget:
            break

    if weak_group:
        weak_support_target = min(
            int(family_cap_support),
            max(
                per_position_support_floor + 1,
                int(round(per_position_support_floor * (1.0 + float(weak_support_bonus_ratio)))),
            ),
        )
        weak_positions = sorted(
            list(used_main_positions),
            key=lambda pos: (
                used_support_per_pos.get(pos, 0),
                -float(main_by_pos.get(pos, {}).get("export_score", 0.0)) if pos in main_by_pos else 0.0,
            ),
        )
        for pos in weak_positions:
            candidates = support_by_pos.get(pos, [])
            if not candidates:
                continue
            start_idx = used_support_per_pos.get(pos, 0)
            for cell in candidates[start_idx:]:
                if support_selected >= support_budget:
                    break
                per_pos = used_support_per_pos.get(pos, 0)
                if per_pos >= int(family_cap_support) or per_pos >= weak_support_target:
                    break
                selected.append(_phase_cell(cell, "weak_recovery"))
                used_support_per_pos[pos] = per_pos + 1
                support_selected += 1
                phase_weak_recovery_added += 1
            if support_selected >= support_budget:
                break

    support_fill_candidates = []
    for pos, cells_at_pos in support_by_pos.items():
        if pos not in used_main_positions:
            continue
        for rank_idx, cell in enumerate(cells_at_pos):
            per_pos = used_support_per_pos.get(pos, 0)
            if per_pos >= int(family_cap_support):
                break
            if rank_idx < per_pos:
                continue
            coverage_penalty = 0.18 * float(used_support_per_pos.get(pos, 0))
            if pos not in used_main_positions:
                coverage_penalty += 0.10
            boosted = dict(cell)
            boosted["_coverage_score"] = float(cell.get("export_score", 0.0)) - coverage_penalty
            support_fill_candidates.append(boosted)
    support_fill_candidates.sort(key=lambda c: -float(c.get("_coverage_score", c.get("export_score", 0.0))))

    for cell in support_fill_candidates:
        if support_selected >= support_budget:
            break
        pos = int(cell["pos"])
        per_pos = used_support_per_pos.get(pos, 0)
        if per_pos >= int(family_cap_support):
            continue
        selected.append(_phase_cell(cell, "support_fill"))
        used_support_per_pos[pos] = per_pos + 1
        support_selected += 1
        phase_support_fill_added += 1
    if len(selected) > int(max_peaks):
        _phase_rank = {
            "main_floor": 0,
            "support_floor": 1,
            "weak_recovery": 2,
            "support_fill": 3,
        }
        selected = sorted(
            selected,
            key=lambda c: (
                _phase_rank.get(str(c.get("_tcr_phase", "support_fill")), 9),
                not c.get("is_main", False),
                -float(c.get("export_score", 0.0)),
            ),
        )[:int(max_peaks)]
    diag = {
        "tcr_anchor_count": int(len(anchors)),
        "tcr_proposal_count": int(len(proposals)),
        "tcr_selected_main_positions": int(len(used_main_positions)),
        "tcr_selected_cells": int(len(selected)),
        "tcr_anchor_floor": int(anchor_floor),
        "tcr_backfill_enabled": int(float(position_backfill_ratio) > 0),
        "tcr_main_floor": int(main_floor),
        "tcr_support_floor": int(support_floor),
        "tcr_main_budget": int(main_budget),
        "tcr_support_budget": int(support_budget),
        "tcr_support_selected": int(support_selected),
        "tcr_phase_main_added": int(phase_main_added),
        "tcr_phase_support_floor_added": int(phase_support_floor_added),
        "tcr_phase_weak_recovery_added": int(phase_weak_recovery_added),
        "tcr_phase_support_fill_added": int(phase_support_fill_added),
        "tcr_position_support_floor": int(per_position_support_floor),
        "tcr_weak_group_bias": int(bool(weak_group)),
        "tcr_weak_group_z2": int(bool(subgroup_z2)),
        "tcr_weak_group_short": int(bool(subgroup_short)),
        "tcr_weak_group_z2_short": int(bool(subgroup_z2_short)),
    }
    return selected, diag


def run_cape_state_controller(cells, selected, filtered, input_mz,
                              token_profile_prior,
                              peptide_length, precursor_charge,
                              mod_count_est, terminal_mod_est,
                              length_adaptive_t_main, t_sat_val,
                              sat_main_ratio_val, max_sat_per_cluster_val,
                              max_tokens_per_cluster_val,
                              require_main_for_sat_val,
                              p, residues,
                              N_by, min_peaks, target_peaks, max_peaks,
                              strong_main_count, mid_main_count, *, score_builder=rerank_export_score):
    """Run CAPE-native dense/sparse/rescue control.

    Returns:
      cells, expand_stats_or_none, cape_diag
    """
    cape_diag = {}
    cape_state = build_cape_state_summary(
        N_by=N_by,
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
        terminal_mod_est=terminal_mod_est,
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        strong_main_count=strong_main_count,
        mid_main_count=mid_main_count,
        over_dense_ratio=len(cells) / max(N_by, 1),
        generation=str(p.get("cape_generation", "v1")),
    )
    evidence_cap = int(cape_state["evidence_cap"])
    cape_diag["cape_state"] = dict(cape_state)
    cape_adjustment = "none"
    cape_allow_rescue = False
    z = int(precursor_charge)
    L = int(peptide_length)
    generation = str(cape_state.get("generation", "v1"))
    target_retry_ratio = 0.82
    if z == 2 and L <= 16:
        target_retry_ratio = 0.95
    elif z == 2 or L <= 12 or int(mod_count_est) >= 1 or bool(terminal_mod_est):
        target_retry_ratio = 0.90
    if generation == "v3":
        target_retry_ratio = 1.10
        if z == 2 and L <= 16:
            target_retry_ratio = 1.25
        elif z == 2 or L <= 12 or int(mod_count_est) >= 1 or bool(terminal_mod_est):
            target_retry_ratio = 1.18
    retry_entry_need = max(int(min_peaks), int(target_peaks * target_retry_ratio))
    cape_diag["cape_retry_entry_need"] = int(retry_entry_need)

    if len(cells) > evidence_cap and bool(cape_state["apply_dense_cap"]):
        cells = sorted(cells, key=lambda c: -float(c.get("export_score", 0.0)))[:evidence_cap]
        cape_adjustment = "dense_cap"
    elif len(cells) < retry_entry_need and bool(cape_state["evidence_supports_sparse"]):
        retry_max_sat = max(max_sat_per_cluster_val, int(cape_state["retry_max_sat"]))
        retry_max_tokens = max(
            int(max_tokens_per_cluster_val) if max_tokens_per_cluster_val is not None else 0,
            int(cape_state["retry_max_tokens"]),
        )
        retry_core_budget = int(cape_state["retry_core_budget"])
        retry_profile_budget = int(cape_state["retry_tier1_budget"])
        retry_low_budget = int(cape_state["retry_tier2_budget"])
        retry_cells, retry_stats = expand_clusters_to_cells(
            selected, filtered, input_mz,
            t_main=length_adaptive_t_main,
            t_sat=t_sat_val,
            sat_main_ratio=sat_main_ratio_val,
            max_sat_per_cluster=retry_max_sat,
            token_profile_prior=token_profile_prior,
            score_builder=score_builder,
            max_tokens_per_cluster=retry_max_tokens,
            require_main_for_sat=require_main_for_sat_val,
            token_tiered_budget=False,
            profile_top_n=int(p.get("profile_top_n", 16)),
            core_token_budget=retry_core_budget,
            profile_extra_budget=retry_profile_budget,
            low_prior_budget=retry_low_budget,
            low_prior_strong_factor=float(p.get("low_prior_strong_factor", 2.0)),
            cape_mode=True,
            peptide_length=peptide_length,
            precursor_charge=precursor_charge,
            mod_count_est=mod_count_est,
            terminal_mod_est=terminal_mod_est,
            residues=residues,
        )
        if len(retry_cells) > len(cells):
            cells = retry_cells
            cape_diag["retry_expand_stats"] = retry_stats
        cape_adjustment = "sparse_retry"
        cape_allow_rescue = len(cells) < retry_entry_need
    else:
        if len(cells) < retry_entry_need:
            cape_adjustment = "low_evidence_no_rescue"

    cape_diag["cape_adjustment"] = str(cape_adjustment)
    cape_diag["cape_allow_rescue"] = bool(cape_allow_rescue)
    cape_diag["cape_evidence_cap"] = int(evidence_cap)
    cape_diag["cape_sparse_retry"] = str(cape_adjustment).startswith("sparse_retry")
    return cells, cape_diag


def get_export_tokens(scene="search", token_profile_prior=None, profile_top_n=16,
                      profile_plateau_n=None, custom_export_tokens=None):
    """Return export-token whitelist for a scene.

    profile_topN is data-profile driven but non-leaking: the profile must come
    from train/val, and core8 is always retained.
    """
    base = list(_EXPORT_TOKENS.get(scene, _EXPORT_TOKENS["search"]))
    if scene == "cape_v1":
        tokens, _tiers = resolve_cape_export_tokens(
            token_profile_prior=token_profile_prior,
            core_tokens=sorted(CORE8_TOKENS),
            tier1_n=8,
            tier2_n=8,
        )
        return tokens
    if scene == "cape_v2":
        tokens, _tiers = resolve_cape_export_tokens(
            token_profile_prior=token_profile_prior,
            core_tokens=sorted(CORE8_TOKENS),
            tier1_n=10,
            tier2_n=10,
        )
        return tokens
    if scene == "cape_v3":
        tokens, _tiers = resolve_cape_export_tokens(
            token_profile_prior=token_profile_prior,
            core_tokens=sorted(CORE8_TOKENS),
            tier1_n=16,
            tier2_n=20,
        )
        return tokens
    if scene == "profile_custom":
        tokens = list(CORE8_TOKENS)
        if custom_export_tokens is not None:
            for tok in custom_export_tokens:
                try:
                    tok_i = int(tok)
                except (TypeError, ValueError):
                    continue
                if 0 <= tok_i < NUM_TOKENS and tok_i not in tokens:
                    tokens.append(tok_i)
        return sorted(tokens)
    if scene in ("profile_top14", "profile_top16", "profile_top20", "profile_tier"):
        if scene == "profile_tier":
            top_n = int(profile_plateau_n if profile_plateau_n is not None else profile_top_n)
        else:
            top_n = int(profile_top_n)
        top_n = max(top_n, len(CORE8_TOKENS))
        if token_profile_prior is not None:
            prior = np.asarray(token_profile_prior, dtype=np.float32)
            if prior.shape[0] == NUM_TOKENS:
                ranked = [int(x) for x in np.argsort(-prior)]
                tokens = list(CORE8_TOKENS)
                for tok in ranked:
                    if tok not in tokens:
                        tokens.append(tok)
                    if len(tokens) >= top_n:
                        break
                return sorted(tokens)
        # Conservative non-leaking fallback:
        # only use core and directly interpretable support-family tokens.
        fallback_order = [24, 25, 26, 22, 23, 21, 19, 20]
        tokens = list(CORE8_TOKENS)
        for tok in fallback_order:
            if tok not in tokens:
                tokens.append(tok)
            if len(tokens) >= top_n:
                break
        return sorted(tokens)
    return base


def build_clusters(pred_matrix, cluster_active, cluster_scores, scene="search",
                   token_profile_prior=None, profile_top_n=16,
                   profile_plateau_n=None, custom_export_tokens=None):
    """Build structured cluster list from prediction matrix.

    Each cluster = (pos, channel) with its constituent tokens.
    Only physically relevant tokens are included per scene policy.

    Returns:
        list of dicts: [{pos, channel, main_score, cluster_score, tokens: {token_idx: score}}, ...]
    """
    H, W, C = pred_matrix.shape
    export_tokens = set(get_export_tokens(
        scene,
        token_profile_prior=token_profile_prior,
        profile_top_n=profile_top_n,
        profile_plateau_n=profile_plateau_n,
        custom_export_tokens=custom_export_tokens,
    ))
    clusters = []
    for ch in range(C):
        for pos in range(H):
            if not cluster_active[pos, ch]:
                continue
            tokens = {}
            for tok in range(W):
                if tok not in export_tokens:
                    continue
                val = float(pred_matrix[pos, tok, ch])
                if val > 0:
                    tokens[tok] = val
            clusters.append({
                "pos": int(pos),
                "channel": int(ch),
                "channel_name": CHANNEL_NAMES[ch],
                "main_score": float(pred_matrix[pos, MAIN_TOKEN, ch]),
                "cluster_score": float(cluster_scores[pos, ch]),
                "tokens": tokens,
            })
    return clusters


# ---------------------------------------------------------------------------
# 4.5 Peptide-length + charge dual-factor dynamic cluster Top-K
# ---------------------------------------------------------------------------
def compute_dynamic_k(peptide_length, precursor_charge, R_target=1.02,
                      n_true_mean=None, min_k=None, max_k=None):
    """Compute the target number of clusters to export.

    K_cluster = round(R_target * N_true_mean[(length_bin, charge)])
    clamped to [min_k, max_k]

    Args:
        peptide_length: int
        precursor_charge: int
        R_target: target R value (predicted/true cluster ratio)
        n_true_mean: dict of (length_bin, charge) -> mean cluster count
        min_k: optional override for minimum clusters
        max_k: optional override for maximum clusters

    Returns:
        K_cluster: int
    """
    L = max(int(peptide_length), 1)
    lb = get_length_bin(L)
    try:
        z = int(precursor_charge)
    except (TypeError, ValueError):
        z = 2

    if n_true_mean is None:
        n_true_mean = DEFAULT_N_TRUE_MEAN

    N_true = n_true_mean.get(lb, {}).get(z, None)
    if N_true is None:
        # Fallback: use charge 2 value, or estimate from length
        N_true = n_true_mean.get(lb, {}).get(2, L * 3.5)

    K = int(round(R_target * N_true))

    # Compute bounds
    if min_k is None:
        min_k = max(4, L // 2)
    if max_k is None:
        max_k = min(6 * (L - 1), 160)

    # Theory cap: K cannot exceed theoretical maximum clusters
    # N_theory_max = (L-1) × 2 × min(z,3)
    theory_max = (L - 1) * 2 * min(z, 3)
    if theory_max > 0:
        max_k = min(max_k, theory_max)

    K = max(min_k, min(K, max_k))
    return K


def select_top_clusters(clusters, K, channel_alpha=0.10, scene="search"):
    """Select top-K clusters by cluster_score, with optional low-confidence
    channel suppression.

    Args:
        clusters: list of cluster dicts from build_clusters()
        K: target cluster count
        channel_alpha: low-confidence channel suppression factor (0=off)
        scene: scene identifier for permissive modes

    Returns:
        selected_clusters: list of selected cluster dicts
    """
    if len(clusters) <= K:
        return clusters

    # Sort by cluster_score descending
    sorted_clusters = sorted(clusters, key=lambda c: c["cluster_score"], reverse=True)

    if channel_alpha <= 0 or scene == "mod":
        return sorted_clusters[:K]

    # Apply channel suppression: within same pos, suppress channels
    # whose cluster_score is below alpha * max_cluster_score_at_pos
    pos_max = {}
    for c in sorted_clusters:
        p = c["pos"]
        if p not in pos_max or c["cluster_score"] > pos_max[p]:
            pos_max[p] = c["cluster_score"]

    # Penalize low-confidence channels
    for c in sorted_clusters:
        p = c["pos"]
        pmax = pos_max.get(p, c["cluster_score"])
        if pmax > 0 and c["cluster_score"] < channel_alpha * pmax:
            c["cluster_score"] *= 0.5  # Downweight but don't remove

    # Re-sort after penalty
    sorted_clusters = sorted(sorted_clusters, key=lambda c: c["cluster_score"], reverse=True)
    return sorted_clusters[:K]


# ---------------------------------------------------------------------------
# 4.6 Cluster expansion back to cell granularity
# ---------------------------------------------------------------------------
def expand_clusters_to_cells(selected_clusters, pred_matrix, input_mz,
                             t_main=0.04, t_sat=0.018, sat_main_ratio=0.15,
                             max_sat_per_cluster=2, residues=None,
                             token_profile_prior=None,
                             score_builder=None,
                             max_tokens_per_cluster=None,
                             require_main_for_sat=True,
                             token_tiered_budget=False,
                             profile_top_n=16,
                             profile_high_n=None,
                             profile_plateau_n=None,
                             core_token_budget=3,
                             profile_extra_budget=2,
                             plateau_budget=0,
                             plateau_strong_factor=1.35,
                             low_prior_budget=1,
                             low_prior_strong_factor=2.0,
                             cape_mode=False,
                             peptide_length=0,
                             precursor_charge=2,
                             mod_count_est=0,
                             terminal_mod_est=False,
                             cape_tier1_n=8,
                             cape_tier2_n=8,
                             cape_generation="v1"):
    if score_builder is None:
        score_builder = rerank_export_score
    """Expand selected clusters back to individual cell predictions.

    Token filtering is done at build_clusters() time via EXPORT_TOKENS per scene.
    This function applies intensity thresholds and satellite-to-main ratio constraint.

    Rules:
      - Main peak (main token): must pass t_main
      - Satellite (tokens 18-22, 24-25): must satisfy ALL of:
          1. main_score >= t_main  (main peak itself is exportable)
          2. score >= t_sat
          3. score >= main_score × sat_main_ratio
      - This prevents orphan satellites when the main peak is below export threshold

    Args:
        selected_clusters: list of selected cluster dicts
        pred_matrix: (39, 46, 6) float array
        input_mz: (39, 46, 6) float array of theoretical m/z values
        t_main: main peak threshold
        t_sat: satellite threshold
        sat_main_ratio: minimum satellite/main score ratio (0 = disabled)

    Returns:
        cells: list of cell dicts
        stats: dict with rejection and per-token-type counts
    """
    cells = []
    stats = {
        "n_main_exported": 0, "n_loss_exported": 0, "n_isotope_exported": 0,
        "n_combined_exported": 0, "n_other_exported": 0,
        "n_rejected_main_thresh": 0, "n_rejected_sat_thresh": 0,
        "n_rejected_sat_ratio": 0, "n_rejected_sat_orphan": 0,
        "n_rejected_tier_budget": 0,
        "cape_tier0_exported": 0, "cape_tier1_exported": 0, "cape_tier2_exported": 0,
        "cape_tail_exported": 0, "cape_budget_total": 0, "cape_budget_used": 0,
        "cape_pre_rescue_budget_used": 0,
        "n_capped_main": 0, "n_capped_isotope": 0, "n_capped_loss": 0,
        "n_capped_combined": 0, "n_capped_other": 0,
    }

    def _token_kind(tok): return TOKEN_KIND.get(tok, "other")
    if token_profile_prior is not None:
        token_profile_prior = np.asarray(token_profile_prior, dtype=np.float32)
        if token_profile_prior.shape[0] != NUM_TOKENS:
            raise ValueError(f"token_profile_prior must contain {NUM_TOKENS} values")

    def _token_prior(tok):
        if token_profile_prior is not None:
            return float(token_profile_prior[int(tok)])
        return TOKEN_PRIOR.get(_token_kind(tok), 0.50)

    profile_high_n = int(profile_high_n if profile_high_n is not None else profile_top_n)
    profile_plateau_n = int(profile_plateau_n if profile_plateau_n is not None else profile_top_n)
    profile_high_n = max(0, profile_high_n)
    profile_plateau_n = max(profile_high_n, profile_plateau_n)

    if token_profile_prior is not None:
        _profile_order = list(np.argsort(-token_profile_prior))
        profile_top_tokens = set(int(t) for t in _profile_order[:max(0, int(profile_top_n))])
        profile_high_tokens = set(int(t) for t in _profile_order[:profile_high_n])
        profile_plateau_tokens = set(int(t) for t in _profile_order[profile_high_n:profile_plateau_n])
    else:
        profile_top_tokens = set()
        profile_high_tokens = set()
        profile_plateau_tokens = set()

    def _score_token_item(kv):
        tok, score = kv
        return float(score) * _token_prior(int(tok))

    def _select_tiered_items(token_items, max_total):
        """Select per-cluster tokens with stable tier budgets.

        This avoids full44 treating every offset as an equal competitor.  The
        second tier is derived from the frozen train/val token profile, so this
        remains independent of the test MGF distribution.
        """
        if max_total is None or int(max_total) <= 0:
            return token_items
        max_total = int(max_total)
        by_tok = {int(tok): float(score) for tok, score in token_items}
        selected = []
        used = set()

        def add_tokens(candidates, limit, strong_factor=None):
            nonlocal selected, used
            limit = max(0, min(int(limit), max_total - len(selected)))
            if limit <= 0:
                return
            ranked = sorted(
                [(tok, by_tok[tok]) for tok in candidates if tok in by_tok and tok not in used],
                key=_score_token_item,
                reverse=True,
            )
            added = 0
            for tok, score in ranked:
                if strong_factor is not None:
                    if score * _token_prior(tok) < t_sat * float(strong_factor):
                        continue
                selected.append((tok, score))
                used.add(tok)
                added += 1
                if added >= limit or len(selected) >= max_total:
                    break

        add_tokens([MAIN_TOKEN], 1)
        add_tokens(
            [t for t in CORE8_TOKENS if t != MAIN_TOKEN],
            core_token_budget,
        )
        add_tokens(
            [t for t in profile_high_tokens if t not in CORE8_TOKENS and t != MAIN_TOKEN],
            profile_extra_budget,
        )
        add_tokens(
            [t for t in profile_plateau_tokens if t not in CORE8_TOKENS and t != MAIN_TOKEN],
            plateau_budget,
            strong_factor=plateau_strong_factor,
        )
        # Fill remaining room only with core/profile-top tokens.  This protects
        # short and z=2 spectra from being filled by many low-prior full44 cells.
        add_tokens(
            [t for t in list(CORE8_TOKENS) + list(profile_high_tokens) if t != MAIN_TOKEN],
            max_total,
        )
        add_tokens(
            [t for t in by_tok if t not in CORE8_TOKENS and t not in profile_top_tokens and t != MAIN_TOKEN],
            low_prior_budget,
            strong_factor=low_prior_strong_factor,
        )

        stats["n_rejected_tier_budget"] += max(0, len(token_items) - len(selected))
        return selected

    for cl in selected_clusters:
        pos = cl["pos"]
        ch = cl["channel"]
        main_score = cl["main_score"]
        n_sat_exported_this_cluster = 0
        token_items = list(cl["tokens"].items())
        cluster_anchor = max((float(score) for _tok, score in token_items), default=0.0)
        if cape_mode:
            token_items, _cape_tiers, _cape_budget = select_cape_token_items(
                token_items,
                token_profile_prior=token_profile_prior,
                peptide_length=peptide_length,
                precursor_charge=precursor_charge,
                mod_count_est=mod_count_est,
                terminal_mod_est=terminal_mod_est,
                main_score=main_score,
                cluster_anchor=cluster_anchor,
                core_tokens=sorted(CORE8_TOKENS),
                tier1_n=int(cape_tier1_n),
                tier2_n=int(cape_tier2_n),
                generation=str(cape_generation),
            )
            stats["n_rejected_tier_budget"] += max(0, len(cl["tokens"]) - len(token_items))
            stats["cape_budget_total"] += int(_cape_budget.get("max_total", len(token_items)))
            stats["cape_budget_used"] += int(_cape_budget.get("used_total", len(token_items)))
            stats["cape_pre_rescue_budget_used"] += int(_cape_budget.get("used_total", len(token_items)))
            _cape_tier0 = set(int(t) for t in _cape_tiers["tier0"])
            _cape_tier1 = set(int(t) for t in _cape_tiers["tier1"])
            _cape_tier2 = set(int(t) for t in _cape_tiers["tier2"])
        elif max_tokens_per_cluster is not None and int(max_tokens_per_cluster) > 0:
            if token_tiered_budget:
                token_items = _select_tiered_items(token_items, int(max_tokens_per_cluster))
            else:
                token_items = sorted(
                    token_items,
                    key=lambda kv: float(kv[1]) * _token_prior(int(kv[0])),
                    reverse=True,
                )[:int(max_tokens_per_cluster)]
        for tok, score in token_items:
            tok = int(tok)
            kind = _token_kind(tok)
            tok_prior = max(_token_prior(tok), 1e-6)
            t_main_tok = t_main / tok_prior
            t_sat_tok = t_sat / tok_prior

            if kind == "main":
                if score < t_main_tok:
                    stats["n_rejected_main_thresh"] += 1
                    continue
                stats["n_main_exported"] += 1
            elif kind in ("isotope", "neutral_loss", "combined_loss"):
                if require_main_for_sat and main_score < t_main:
                    stats["n_rejected_sat_orphan"] += 1
                    continue
                if score < t_sat_tok:
                    stats["n_rejected_sat_thresh"] += 1
                    continue
                if (
                    sat_main_ratio > 0
                    and main_score >= t_main
                    and score < main_score * sat_main_ratio
                ):
                    stats["n_rejected_sat_ratio"] += 1
                    continue
                # max_sat_per_cluster: limit satellites per cluster (P2-7)
                if n_sat_exported_this_cluster >= max_sat_per_cluster:
                    continue
                n_sat_exported_this_cluster += 1
                if kind == "isotope":      stats["n_isotope_exported"] += 1
                elif kind == "neutral_loss": stats["n_loss_exported"] += 1
                else:                       stats["n_combined_exported"] += 1
            else:
                if score < t_sat_tok:
                    stats["n_rejected_sat_thresh"] += 1
                    continue
                stats["n_other_exported"] += 1

            if cape_mode:
                if tok in _cape_tier0:
                    stats["cape_tier0_exported"] += 1
                elif tok in _cape_tier1:
                    stats["cape_tier1_exported"] += 1
                elif tok in _cape_tier2:
                    stats["cape_tier2_exported"] += 1
                else:
                    stats["cape_tail_exported"] += 1

            mz = float(input_mz[pos, tok, ch])
            cells.append({
                "mz": mz,
                "intensity": float(score),
                "norm_intensity": float(score),
                "support_score": 1.0,
                "export_score": score_builder(
                    model_score=float(score),
                    tok=tok,
                    kind=kind,
                    main_score=main_score,
                    pos=pos,
                    channel=ch,
                    pred_matrix=pred_matrix,
                    peptide_length=peptide_length,
                    precursor_charge=precursor_charge,
                    residues=residues,
                    token_profile_prior=token_profile_prior,
                ),
                "pos": int(pos + 1),
                "token_idx": int(tok),
                "channel": CHANNEL_NAMES[ch],
                "is_main": (kind == "main"),
                "_token_kind": kind,
            })
    return cells, stats


# ---------------------------------------------------------------------------
# Top-level pipeline orchestrator
# ---------------------------------------------------------------------------
def run_cluster_postprocess(pred_matrix, input_mz, peptide_length, precursor_charge,
                            scene="search", params=None, n_true_mean=None,
                            return_diagnostics=True, residues=None, nterm_shift=0.0, *,
                            link_site=None, partner_residues=None):
    """Run the full cluster-level postprocessing pipeline.

    This is the main entry point. It runs steps 1-10 from Section 2.2.

    Args:
        pred_matrix: (39, 46, 6) float array — raw model prediction
        input_mz: (39, 46, 6) float array — theoretical m/z grid
        peptide_length: int — number of amino acid residues
        precursor_charge: int
        scene: 'mod', 'search', or 'quant'
        params: optional dict of parameter overrides
        n_true_mean: optional calibrated N_true_mean table
        return_diagnostics: if True, return diagnostics dict
        residues: local-chain (amino acid, modified residue mass) pairs
        link_site: optional 1-based linked residue on the local chain
        partner_residues: partner-chain residue pairs, supplied together with link_site

    Returns:
        cells: list of cell candidate dicts (for merge_overlapping_peaks)
        diagnostics: dict with per-step statistics (if return_diagnostics)
    """
    # Resolve parameters
    p = resolve_scene_params(scene, params)
    p = _resolve_subgroup_postprocess_params(p, peptide_length=peptide_length, precursor_charge=precursor_charge)
    export_score_mode = str(p.get("export_score_mode", "family_rerank")).lower()
    rerank_score = rerank_export_score
    identity_score = identity_export_score
    if link_site is not None or partner_residues is not None:
        if residues is None or partner_residues is None or link_site is None or not 1 <= link_site <= len(residues):
            raise ValueError("XL postprocessing requires both chains' residues and a valid 1-based link site")
        rerank_score = partial(rerank_score, link_site=link_site, partner_residues=partner_residues)
        identity_score = partial(identity_score, link_site=link_site, partner_residues=partner_residues)
    score_builder = rerank_score if export_score_mode == "family_rerank" else identity_score

    length_adaptive_t_main = get_adaptive_t_main(peptide_length, p["t_main"])
    t_sat_val = get_t_sat(length_adaptive_t_main, min_t_sat=p["t_sat"])
    mod_count_est = estimate_mod_count_from_residues(residues, nterm_shift=nterm_shift)
    try:
        terminal_mod_est = abs(float(nterm_shift)) > 0.02
    except (TypeError, ValueError):
        terminal_mod_est = False

    # z=2 short peptide relaxed thresholds: ratio data shows z=2,≤16 peaks are too sparse
    if int(precursor_charge) == 2 and peptide_length <= 16 and scene != "profile44_tiered":
        z2_tm = p.get("z2_t_main", p["t_main"] * 0.80)
        z2_ts = p.get("z2_t_sat", p["t_sat"] * 0.80)
        length_adaptive_t_main = get_adaptive_t_main(peptide_length, z2_tm)
        t_sat_val = get_t_sat(length_adaptive_t_main, min_t_sat=z2_ts)

    diag_params = {
        k: v for k, v in p.items()
        if k != "token_profile_prior"
    }
    diag = {"scene": scene, "params": diag_params,
            "length_adaptive_t_main": length_adaptive_t_main,
            "t_sat": t_sat_val,
            "mod_count_est": int(mod_count_est),
            "terminal_mod_est": bool(terminal_mod_est)}
    if "resolved_subgroup" in p:
        diag["resolved_subgroup"] = str(p["resolved_subgroup"])
    token_profile_prior = p.get("token_profile_prior")
    if token_profile_prior is not None:
        token_profile_prior = np.asarray(token_profile_prior, dtype=np.float32)
        if token_profile_prior.shape[0] != NUM_TOKENS:
            raise ValueError(f"token_profile_prior must contain {NUM_TOKENS} values")
        diag["token_profile_prior_min"] = float(token_profile_prior.min())
        diag["token_profile_prior_max"] = float(token_profile_prior.max())
    export_tokens = get_export_tokens(
        scene,
        token_profile_prior=token_profile_prior,
        profile_top_n=int(p.get("profile_top_n", 16)),
        profile_plateau_n=p.get("profile_plateau_n"),
        custom_export_tokens=p.get("export_tokens"),
    )
    cape_tiers = None
    if scene in ("cape_v1", "cape_v2", "cape_v3"):
        _cape_tokens, cape_tiers = resolve_cape_export_tokens(
            token_profile_prior=token_profile_prior,
            core_tokens=sorted(CORE8_TOKENS),
            tier1_n=int(p.get("cape_tier1_n", 8)),
            tier2_n=int(p.get("cape_tier2_n", 8)),
        )
        export_tokens = _cape_tokens
    diag["export_token_count"] = int(len(export_tokens))
    diag["export_tokens"] = [int(t) for t in export_tokens]
    if cape_tiers is not None:
        diag["cape_tier0_tokens"] = [int(t) for t in cape_tiers["tier0"]]
        diag["cape_tier1_tokens"] = [int(t) for t in cape_tiers["tier1"]]
        diag["cape_tier2_tokens"] = [int(t) for t in cape_tiers["tier2"]]

    # Step 1: Position theoretical mask
    masked = apply_position_mask(pred_matrix, peptide_length)
    diag["n_nonzero_before_position_mask"] = int((pred_matrix > 0).sum())
    diag["n_nonzero_after_position_mask"] = int((masked > 0).sum())

    # Step 2: Charge-aware channel mask
    masked = apply_charge_mask(masked, precursor_charge)
    if scene in ("profile_top14", "profile_top16", "profile_top20", "profile_tier", "profile_custom", "cape_v1", "cape_v2", "cape_v3", "sgr_v1", "sgr_v2", "v20_pp4", "tcr_v1", "tcr_v2", "tcr_v3", "tcr_v4", "tcr_v5"):
        token_mask = np.zeros(masked.shape[1], dtype=bool)
        token_mask[np.asarray(export_tokens, dtype=int)] = True
        masked = masked.copy()
        masked[:, ~token_mask, :] = 0.0
        diag["n_nonzero_after_export_token_mask"] = int((masked > 0).sum())
    diag["n_nonzero_after_charge_mask"] = int((masked > 0).sum())
    diag["charge_mask"] = build_charge_channel_mask(precursor_charge).tolist()
    diag["n_charge_invalid_removed"] = (
        diag["n_nonzero_after_position_mask"] - diag["n_nonzero_after_charge_mask"]
    )

    # Step 3: Main-peak / satellite dual-threshold + isolated satellite removal
    t_weak_main_val = max(0.015, length_adaptive_t_main * 0.375)
    filtered, cluster_info = build_cluster_mask(
        masked, t_main=length_adaptive_t_main,
        t_weak_main=t_weak_main_val,
        t_sat=t_sat_val,
        isolated_sat_mode=p["isolated_sat_mode"],
        cluster_anchor_mode=p.get("cluster_anchor_mode", "main"),
        token_profile_prior=token_profile_prior,
    )
    diag.update(cluster_info)
    diag["n_nonzero_after_dual_threshold"] = int((filtered > 0).sum())

    # Step 4: Build structural clusters
    if p.get("cluster_score_mode") == "profile_max":
        _prior = token_profile_prior
        if _prior is None:
            _prior = np.ones(filtered.shape[1], dtype=np.float32)
        _prior = np.asarray(_prior, dtype=np.float32).reshape(1, filtered.shape[1], 1)
        cluster_scores = (filtered * _prior).max(axis=1).astype(np.float32)
        cluster_scores = np.where(cluster_info["cluster_active"], cluster_scores, 0.0)
    else:
        cluster_scores = compute_cluster_scores(filtered, cluster_info["cluster_active"])
    clusters = build_clusters(
        filtered,
        cluster_info["cluster_active"],
        cluster_scores,
        scene=scene,
        token_profile_prior=token_profile_prior,
        profile_top_n=int(p.get("profile_top_n", 16)),
        profile_plateau_n=p.get("profile_plateau_n"),
        custom_export_tokens=p.get("export_tokens"),
    )
    diag["n_clusters_built"] = len(clusters)

    # Step 5: Already done in build_cluster_mask (isolated satellite removal)

    # Step 6: Cluster-level weighted scoring already done

    # Step 7: Dynamic Top-K
    K = compute_dynamic_k(
        peptide_length, precursor_charge,
        R_target=p["R_target"],
        n_true_mean=n_true_mean,
    )
    diag["K_target"] = K
    diag["n_clusters_before_topk"] = len(clusters)

    selected = select_top_clusters(
        clusters, K,
        channel_alpha=p["channel_alpha"],
        scene=scene,
    )
    diag["n_clusters_selected"] = len(selected)

    # Step 8: Expand to cells (token filtering already done in build_clusters)
    L = peptide_length
    z = int(precursor_charge)
    sat_main_ratio_val = p.get("sat_main_ratio", 0.15)
    _msc = {"search": 1, "cosine_eval": 2, "cosine_eval_rich": 3,
            "full44_diag": 4, "profile44_tiered": 5}
    max_sat_per_cluster_val = _msc.get(scene, 3 if scene == "mod" else 2)
    if p.get("max_sat_per_cluster") is not None:
        max_sat_per_cluster_val = int(p["max_sat_per_cluster"])
    max_tokens_per_cluster_val = p.get("max_tokens_per_cluster")
    if max_tokens_per_cluster_val is not None:
        max_tokens_per_cluster_val = int(max_tokens_per_cluster_val)
    require_main_for_sat_val = bool(p.get("require_main_for_sat", True))
    # Short peptides: fewer b/y ions → each ion has more room for satellites
    if L <= 8:
        max_sat_per_cluster_val = max(max_sat_per_cluster_val, 5)
    elif L <= 12:
        max_sat_per_cluster_val = max(max_sat_per_cluster_val, 4)
    cells, expand_stats = expand_clusters_to_cells(
        selected, filtered, input_mz,
        t_main=length_adaptive_t_main,
        t_sat=t_sat_val,
        sat_main_ratio=sat_main_ratio_val,
        max_sat_per_cluster=max_sat_per_cluster_val,
        token_profile_prior=token_profile_prior,
        score_builder=score_builder,
        max_tokens_per_cluster=max_tokens_per_cluster_val,
        require_main_for_sat=require_main_for_sat_val,
        token_tiered_budget=bool(p.get("token_tiered_budget", False)),
        profile_top_n=int(p.get("profile_top_n", 16)),
        profile_high_n=p.get("profile_high_n"),
        profile_plateau_n=p.get("profile_plateau_n"),
        core_token_budget=int(p.get("core_token_budget", 3)),
        profile_extra_budget=int(p.get("profile_extra_budget", 2)),
        plateau_budget=int(p.get("plateau_budget", 0)),
        plateau_strong_factor=float(p.get("plateau_strong_factor", 1.35)),
        low_prior_budget=int(p.get("low_prior_budget", 1)),
        low_prior_strong_factor=float(p.get("low_prior_strong_factor", 2.0)),
        cape_mode=bool(p.get("cape_mode", False)),
        peptide_length=peptide_length,
        precursor_charge=precursor_charge,
        mod_count_est=mod_count_est,
        terminal_mod_est=terminal_mod_est,
        residues=residues,
        cape_tier1_n=int(p.get("cape_tier1_n", 8)),
        cape_tier2_n=int(p.get("cape_tier2_n", 8)),
        cape_generation=str(p.get("cape_generation", "v1")),
    )

    if scene in ("tcr_v1", "tcr_v2", "tcr_v3", "tcr_v4", "tcr_v5"):
        cells, _tcr_diag = reconstruct_tcr_cells(
            filtered=filtered,
            input_mz=input_mz,
            token_profile_prior=token_profile_prior,
            peptide_length=peptide_length,
            precursor_charge=precursor_charge,
            min_peaks=max(1, int(max((peptide_length - 1) * 2 * min(z, 3), 1) * 0.55)),
            target_peaks=max(1, int(max((peptide_length - 1) * 2 * min(z, 3), 1) * p.get("R_target", 1.35))),
            max_peaks=max(40, int(max((peptide_length - 1) * 2 * min(z, 3), 1) * p.get("peak_cap_ratio", 2.20))),
            t_main=length_adaptive_t_main,
            t_sat=t_sat_val,
            anchor_main_ratio=float(p.get("tcr_anchor_main_ratio", 0.82)),
            anchor_support_ratio=float(p.get("tcr_anchor_support_ratio", 0.55)),
            anchor_floor_ratio=float(p.get("tcr_anchor_floor_ratio", 0.75)),
            main_floor_ratio=float(p.get("tcr_main_floor_ratio", 0.72)),
            support_floor_ratio=float(p.get("tcr_support_floor_ratio", 0.90)),
            support_budget_ratio=float(p.get("tcr_support_budget_ratio", 0.90)),
            position_backfill_ratio=float(p.get("tcr_position_backfill_ratio", 0.0)),
            proposal_main_ratio=float(p.get("tcr_proposal_main_ratio", 1.00)),
            proposal_support_ratio=float(p.get("tcr_proposal_support_ratio", 1.00)),
            family_cap_main=int(p.get("tcr_family_cap_main", 1)),
            family_cap_support=int(p.get("tcr_family_cap_support", 3)),
            global_main_budget=int(p.get("tcr_global_main_budget", 96)),
            global_support_budget=int(p.get("tcr_global_support_budget", 192)),
            position_support_floor=int(p.get("tcr_position_support_floor", 1)),
            weak_position_bonus_ratio=float(p.get("tcr_weak_position_bonus_ratio", 0.0)),
            weak_support_bonus_ratio=float(p.get("tcr_weak_support_bonus_ratio", 0.0)),
            weak_backfill_bonus_ratio=float(p.get("tcr_weak_backfill_bonus_ratio", 0.0)),
            mod_count_est=int(mod_count_est),
            terminal_mod_est=bool(terminal_mod_est),
            tcr_generation=str(p.get("tcr_generation", "v1")),
            residues=residues if link_site is not None else None,
            score_builder=rerank_score,
        )
        expand_stats["n_main_exported"] = sum(1 for c in cells if c.get("is_main", False))
        expand_stats["n_isotope_exported"] = sum(1 for c in cells if c.get("_token_kind") == "isotope")
        expand_stats["n_loss_exported"] = sum(1 for c in cells if c.get("_token_kind") == "neutral_loss")
        expand_stats["n_combined_exported"] = sum(1 for c in cells if c.get("_token_kind") == "combined_loss")
        expand_stats["n_other_exported"] = sum(1 for c in cells if c.get("_token_kind") == "other")
        diag.update(_tcr_diag)
    diag["n_cells_exported"] = len(cells)
    diag["n_main_cells"] = expand_stats["n_main_exported"]
    diag["n_loss_cells"] = expand_stats["n_loss_exported"]
    diag["n_isotope_cells"] = expand_stats["n_isotope_exported"]
    diag["n_combined_cells"] = expand_stats.get("n_combined_exported", 0)
    diag["n_rejected_sat_orphan"] = expand_stats["n_rejected_sat_orphan"]
    diag["n_rejected_sat_ratio"] = expand_stats["n_rejected_sat_ratio"]
    diag["n_rejected_main_thresh"] = expand_stats["n_rejected_main_thresh"]
    diag["n_rejected_tier_budget"] = expand_stats.get("n_rejected_tier_budget", 0)
    if scene in ("cape_v1", "cape_v2", "cape_v3"):
        diag["cape_tier0_exported"] = expand_stats.get("cape_tier0_exported", 0)
        diag["cape_tier1_exported"] = expand_stats.get("cape_tier1_exported", 0)
        diag["cape_tier2_exported"] = expand_stats.get("cape_tier2_exported", 0)
        diag["cape_tail_exported"] = expand_stats.get("cape_tail_exported", 0)
        diag["cape_budget_total"] = expand_stats.get("cape_budget_total", 0)
        diag["cape_budget_used"] = expand_stats.get("cape_budget_used", 0)
        diag["cape_pre_rescue_budget_used"] = expand_stats.get("cape_pre_rescue_budget_used", 0)

    # ── Theory-budget peak control ──
    N_by = max(L - 1, 0) * 2 * min(z, 3)
    diag["N_by"] = N_by

    # Define budget per scene based on N_by (theoretical b/y cluster count)
    if scene == "search":
        min_peaks = int(N_by * 0.35)
        target_peaks = int(N_by * 0.80)
        max_peaks = min(int(N_by * 1.50), 450)
    elif scene == "cosine_eval":
        min_peaks = int(N_by * 0.50)
        target_peaks = int(N_by * 1.10)
        max_peaks = min(int(N_by * 2.00), 600)
    elif scene in ("cosine_eval_rich", "rich_G1_33_36", "rich_G2_42_43",
                  "abl_C1_no18_19", "abl_rich_26_32", "abl_rich_26_36",
                  "abl_R2p2", "abl_R2p0"):
        _cap_r = p.get("peak_cap_ratio", 2.5)
        min_peaks = int(N_by * 0.55)
        target_peaks = int(N_by * 1.30)
        max_peaks = min(int(N_by * _cap_r), 700)
    elif scene == "profile44":
        _cap_r = p.get("peak_cap_ratio", 3.0)
        min_peaks = int(N_by * 0.65)
        target_peaks = int(N_by * 1.75)
        max_peaks = min(int(N_by * _cap_r), 750)
    elif scene == "profile44_tiered":
        _cap_r = p.get("peak_cap_ratio", 2.6)
        min_peaks = int(N_by * 0.55)
        target_peaks = int(N_by * 1.35)
        max_peaks = min(int(N_by * _cap_r), 650)
    elif scene in ("profile_top14", "profile_top16", "profile_top20", "profile_tier", "profile_custom", "cape_v1", "cape_v2", "cape_v3", "sgr_v1", "sgr_v2", "v20_pp4"):
        _cap_r = p.get(
            "peak_cap_ratio",
            (
                2.4 if scene == "profile_top14"
                else 2.6 if scene == "profile_top16"
                else 2.8 if scene == "profile_tier"
                else 2.90 if scene in ("sgr_v2", "v20_pp4")
                else 3.40 if scene == "sgr_v1"
                else 4.20 if scene == "cape_v3"
                else 3.00 if scene == "cape_v2"
                else 2.85 if scene == "cape_v1"
                else 2.6 if scene == "profile_custom"
                else 3.0
            ),
        )
        min_peaks = int(N_by * 0.55)
        _target_r = (
            1.20 if scene == "profile_top14"
            else 1.30 if scene == "profile_top16"
            else 1.35 if scene == "profile_tier"
            else 1.75 if scene in ("sgr_v2", "v20_pp4")
            else 1.55 if scene == "sgr_v1"
            else 2.10 if scene == "cape_v3"
            else 1.55 if scene == "cape_v2"
            else 1.40 if scene == "cape_v1"
            else 1.30 if scene == "profile_custom"
            else 1.45
        )
        target_peaks = int(N_by * _target_r)
        max_peaks = min(int(N_by * _cap_r), 700)
    elif scene == "full44_diag":
        min_peaks = int(N_by * 0.60)
        target_peaks = int(N_by * 1.50)
        max_peaks = min(int(N_by * 3.00), 1000)
    else:  # mod, quant
        min_peaks = int(N_by * 0.40)
        target_peaks = int(N_by * 1.00)
        max_peaks = min(int(N_by * 2.50), 600)
    diag["min_peaks"] = min_peaks
    diag["target_peaks"] = target_peaks

    # profile44 sparse handling is centralized in the budget state machine below.
    # Do not densify charge-2 spectra by default: many z=2 originals genuinely
    # have only tens to ~200 peaks.

    # Cap floor = physical ceiling minus safety margin.
    # Physical ceiling: N_by × (1 + max_sat_per_cluster) cells per spectrum.
    # Short peptides need near-ceiling cap because every b/y ion matters.
    # L<=8: N_by×6 = 120 max, cap at 110 (92% of ceiling)
    # L<=12: N_by×5 = 180 max with max_sat=4, cap at 160 (89%)
    if scene == "profile44_tiered" and z == 2:
        if L <= 8:
            max_peaks = max(max_peaks, 60)
        elif L <= 12:
            max_peaks = max(max_peaks, 90)
    else:
        if L <= 8:
            max_peaks = max(max_peaks, 110)
        elif L <= 12:
            max_peaks = max(max_peaks, 160)

    diag["peak_cap_max"] = max_peaks

    def _refresh_expand_diag():
        diag["n_cells_exported"] = len(cells)
        diag["n_main_cells"] = expand_stats["n_main_exported"]
        diag["n_loss_cells"] = expand_stats["n_loss_exported"]
        diag["n_isotope_cells"] = expand_stats["n_isotope_exported"]
        diag["n_combined_cells"] = expand_stats.get("n_combined_exported", 0)
        diag["n_rejected_tier_budget"] = expand_stats.get("n_rejected_tier_budget", 0)
        diag["n_rejected_sat_orphan"] = expand_stats["n_rejected_sat_orphan"]
        diag["n_rejected_sat_ratio"] = expand_stats["n_rejected_sat_ratio"]
        diag["n_rejected_main_thresh"] = expand_stats["n_rejected_main_thresh"]
        if _is_cape:
            diag["cape_tier0_exported"] = expand_stats.get("cape_tier0_exported", 0)
            diag["cape_tier1_exported"] = expand_stats.get("cape_tier1_exported", 0)
            diag["cape_tier2_exported"] = expand_stats.get("cape_tier2_exported", 0)
            diag["cape_tail_exported"] = expand_stats.get("cape_tail_exported", 0)
            diag["cape_budget_total"] = expand_stats.get("cape_budget_total", 0)
            diag["cape_budget_used"] = expand_stats.get("cape_budget_used", 0)

    def _conditional_local_recovery_pp3(cells_local):
        subgroup = str(diag.get("resolved_subgroup") or infer_postprocess_subgroup(peptide_length, precursor_charge) or "none")
        cfg, recover_min, recover_target, recover_cap = _pp3_target_counts(
            subgroup,
            N_by=N_by,
            min_peaks=min_peaks,
            target_peaks=target_peaks,
            max_peaks=max_peaks,
            params=p,
        )
        pp3_diag = {
            "pp3_subgroup": subgroup,
            "pp3_recover_min": int(recover_min),
            "pp3_recover_target": int(recover_target),
            "pp3_recover_cap": int(recover_cap),
            "pp3_triggered": False,
            "pp3_main_added": 0,
            "pp3_sat_added": 0,
            "pp3_candidates_main": 0,
            "pp3_candidates_sat": 0,
            "pp3_anchor_gate_pass": False,
            "pp3_reason": "not_pp3_scene",
        }
        if not bool(p.get("pp3_mode", False)):
            return cells_local, pp3_diag
        if len(cells_local) >= recover_min or N_by <= 0:
            pp3_diag["pp3_reason"] = "not_under_dense" if len(cells_local) >= recover_min else "no_theory_budget"
            return cells_local, pp3_diag

        strong_main_thr_local = max(float(length_adaptive_t_main), float(length_adaptive_t_main) * 1.25)
        strong_main_count_local = sum(
            1 for c in cells_local
            if c.get("is_main", False) and float(c.get("intensity", 0.0)) >= strong_main_thr_local
        )
        min_strong_main_local = int(cfg.get("min_strong_main", 3))
        if strong_main_count_local < min_strong_main_local:
            pp3_diag["pp3_reason"] = "weak_anchor"
            return cells_local, pp3_diag
        pp3_diag["pp3_anchor_gate_pass"] = True

        channel_to_idx = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}
        occupied_slots = {(int(c["pos"]), str(c["channel"]), int(c["token_idx"])) for c in cells_local}
        existing_main_keys = {
            (int(c["pos"]) - 1, channel_to_idx.get(str(c["channel"]), -1))
            for c in cells_local
            if c.get("is_main", False)
        }
        existing_main_keys = {key for key in existing_main_keys if key[1] >= 0}
        main_positions_by_channel = {}
        for pos0, ch in existing_main_keys:
            main_positions_by_channel.setdefault(ch, set()).add(pos0)

        local_window = int(p.get("pp3_recovery_local_window", cfg.get("local_window", 1)))
        global_budget = min(
            max(0, recover_target - len(cells_local)),
            max(0, recover_cap - len(cells_local)),
            int(p.get("pp3_recovery_budget", cfg.get("main_budget", 3) + cfg.get("sat_budget", 8))),
        )
        if global_budget <= 0:
            pp3_diag["pp3_reason"] = "no_budget"
            return cells_local, pp3_diag

        relaxed_anchor_ratio = float(p.get("pp3_recovery_anchor_ratio", cfg.get("anchor_ratio", 1.06)))
        sat_scale = float(cfg.get("sat_scale", 0.76))
        sat_ratio_scale = float(cfg.get("sat_main_ratio_scale", 0.94))
        per_cluster_sat_limit = int(cfg.get("per_cluster_sat_limit", 2))
        main_budget = int(cfg.get("main_budget", 3))
        sat_budget = int(cfg.get("sat_budget", 8))

        def _token_prior(tok):
            if token_profile_prior is not None:
                return float(token_profile_prior[int(tok)])
            return _token_prior_func(int(tok))

        main_candidates = []
        sat_candidates = []
        for cl in selected:
            pos0 = int(cl["pos"])
            ch = int(cl["channel"])
            if ch < 0 or ch >= len(CHANNEL_NAMES):
                continue
            main_score = float(cl.get("main_score", 0.0))
            cluster_anchor = max(
                (float(score) * _token_prior(int(tok)) for tok, score in cl.get("tokens", {}).items()),
                default=0.0,
            )
            nearest_main_dist = None
            if ch in main_positions_by_channel and main_positions_by_channel[ch]:
                nearest_main_dist = min(abs(pos0 - p0) for p0 in main_positions_by_channel[ch])

            main_slot = (pos0 + 1, CHANNEL_NAMES[ch], MAIN_TOKEN)
            if (
                main_slot not in occupied_slots
                and nearest_main_dist is not None
                and nearest_main_dist <= local_window
                and main_score >= float(length_adaptive_t_main) * relaxed_anchor_ratio
                and cluster_anchor >= float(length_adaptive_t_main) * relaxed_anchor_ratio
            ):
                main_candidates.append(
                    {
                        "mz": float(input_mz[pos0, MAIN_TOKEN, ch]),
                        "intensity": main_score,
                        "norm_intensity": main_score,
                        "support_score": 1.0,
                        "export_score": score_builder(
                            model_score=main_score,
                            tok=MAIN_TOKEN,
                            kind="main",
                            main_score=main_score,
                            pos=pos0,
                            channel=ch,
                            pred_matrix=pred_matrix,
                            peptide_length=peptide_length,
                            precursor_charge=precursor_charge,
                            residues=residues,
                            token_profile_prior=token_profile_prior,
                        ),
                        "pos": int(pos0 + 1),
                        "token_idx": MAIN_TOKEN,
                        "channel": CHANNEL_NAMES[ch],
                        "is_main": True,
                        "is_rescue": False,
                        "is_pp3_recovery": True,
                        "_pp3_cluster": (pos0, ch),
                    }
                )

        recovered_main_keys = set(existing_main_keys)
        for cand in sorted(main_candidates, key=lambda c: -float(c.get("export_score", 0.0)))[:main_budget]:
            slot = (int(cand["pos"]), str(cand["channel"]), int(cand["token_idx"]))
            if slot in occupied_slots:
                continue
            cells_local.append(cand)
            occupied_slots.add(slot)
            ch = channel_to_idx[str(cand["channel"])]
            recovered_main_keys.add((int(cand["pos"]) - 1, ch))
            main_positions_by_channel.setdefault(ch, set()).add(int(cand["pos"]) - 1)
            pp3_diag["pp3_main_added"] += 1

        for cl in selected:
            pos0 = int(cl["pos"])
            ch = int(cl["channel"])
            if (pos0, ch) not in recovered_main_keys:
                continue
            main_score = float(cl.get("main_score", 0.0))
            cluster_added = 0
            token_items = sorted(
                cl.get("tokens", {}).items(),
                key=lambda kv: float(kv[1]) * _token_prior(int(kv[0])),
                reverse=True,
            )
            for tok, score in token_items:
                tok = int(tok)
                if tok == MAIN_TOKEN:
                    continue
                kind = TOKEN_KIND.get(tok, "other")
                if kind not in ("isotope", "neutral_loss", "combined_loss"):
                    continue
                slot = (pos0 + 1, CHANNEL_NAMES[ch], tok)
                if slot in occupied_slots:
                    continue
                tok_prior = max(_token_prior(tok), 1e-6)
                t_sat_tok = float(t_sat_val) * sat_scale / tok_prior
                if float(score) < t_sat_tok:
                    continue
                if sat_main_ratio_val > 0 and float(score) < main_score * sat_main_ratio_val * sat_ratio_scale:
                    continue
                sat_candidates.append(
                    {
                        "mz": float(input_mz[pos0, tok, ch]),
                        "intensity": float(score),
                        "norm_intensity": float(score),
                        "support_score": float(score) / max(main_score, 1e-6),
                        "export_score": score_builder(
                            model_score=float(score),
                            tok=tok,
                            kind=kind,
                            main_score=main_score,
                            pos=pos0,
                            channel=ch,
                            pred_matrix=pred_matrix,
                            peptide_length=peptide_length,
                            precursor_charge=precursor_charge,
                            residues=residues,
                            token_profile_prior=token_profile_prior,
                        ),
                        "pos": int(pos0 + 1),
                        "token_idx": tok,
                        "channel": CHANNEL_NAMES[ch],
                        "is_main": False,
                        "is_rescue": False,
                        "is_pp3_recovery": True,
                        "_pp3_cluster": (pos0, ch),
                    }
                )
                cluster_added += 1
                if cluster_added >= per_cluster_sat_limit:
                    break

        pp3_diag["pp3_candidates_main"] = len(main_candidates)
        pp3_diag["pp3_candidates_sat"] = len(sat_candidates)
        sat_limit = min(sat_budget, max(0, global_budget - pp3_diag["pp3_main_added"]))
        sat_added_by_cluster = {}
        for cand in sorted(sat_candidates, key=lambda c: -float(c.get("export_score", 0.0))):
            if pp3_diag["pp3_sat_added"] >= sat_limit:
                break
            slot = (int(cand["pos"]), str(cand["channel"]), int(cand["token_idx"]))
            if slot in occupied_slots:
                continue
            cluster_key = cand.get("_pp3_cluster")
            if sat_added_by_cluster.get(cluster_key, 0) >= per_cluster_sat_limit:
                continue
            cells_local.append(cand)
            occupied_slots.add(slot)
            sat_added_by_cluster[cluster_key] = sat_added_by_cluster.get(cluster_key, 0) + 1
            pp3_diag["pp3_sat_added"] += 1

        pp3_diag["pp3_triggered"] = bool(pp3_diag["pp3_main_added"] or pp3_diag["pp3_sat_added"])
        pp3_diag["pp3_reason"] = "recovered" if pp3_diag["pp3_triggered"] else "no_local_candidates"
        return cells_local, pp3_diag

    def _post_recovery_density_cap_pp3(cells_local):
        subgroup = str(diag.get("resolved_subgroup") or infer_postprocess_subgroup(peptide_length, precursor_charge) or "none")
        cfg, _, _, recover_cap = _pp3_target_counts(
            subgroup,
            N_by=N_by,
            min_peaks=min_peaks,
            target_peaks=target_peaks,
            max_peaks=max_peaks,
            params=p,
        )
        cap_ratio = float(cfg.get("dense_cap_ratio", cfg.get("recover_cap_ratio", 1.72)))
        cap = min(int(max_peaks), int(round(N_by * cap_ratio))) if N_by > 0 else int(max_peaks)
        cap = max(int(min_peaks), min(int(recover_cap), cap))
        if terminal_mod_est and subgroup in ("z2_short", "z2_long", "z3_short"):
            cap = max(int(min_peaks), cap - 4)
        cap_diag = {
            "pp3_density_cap": int(cap),
            "pp3_density_cap_applied": False,
            "pp3_density_cells_before": int(len(cells_local)),
            "pp3_density_cells_after": int(len(cells_local)),
        }
        if len(cells_local) <= cap:
            return cells_local, cap_diag
        capped = sorted(
            cells_local,
            key=lambda c: (not c.get("is_main", False), -float(c.get("export_score", 0.0))),
        )[:cap]
        cap_diag["pp3_density_cap_applied"] = True
        cap_diag["pp3_density_cells_after"] = int(len(capped))
        return capped, cap_diag

    # Budget state machine.  Only one adjustment route is allowed:
    # dense_cap, sparse_retry, low_evidence_no_rescue, or none.
    dense_guard_applied = False
    dense_guard_cap = None
    strong_main_thr = max(0.030, length_adaptive_t_main * 1.50)
    mid_main_thr = max(length_adaptive_t_main, 0.012)
    strong_main_count = sum(
        1 for c in cells
        if c.get("is_main", False) and float(c.get("intensity", 0.0)) >= strong_main_thr
    )
    mid_main_count = sum(
        1 for c in cells
        if c.get("is_main", False) and float(c.get("intensity", 0.0)) >= mid_main_thr
    )
    diag["dense_guard_strong_main_count"] = int(strong_main_count)
    diag["dense_guard_mid_main_count"] = int(mid_main_count)
    diag["dense_guard_strong_main_thr"] = float(strong_main_thr)
    diag["dense_guard_mid_main_thr"] = float(mid_main_thr)

    evidence_cap = max_peaks
    profile44_adjustment = "none"
    profile44_allow_rescue = False
    evidence_supports_sparse = False
    _is_cape = (scene in ("cape_v1", "cape_v2", "cape_v3"))
    _is_sgr = (scene in ("sgr_v1", "sgr_v2", "v20_pp4"))
    cape_adjustment = "none"
    cape_allow_rescue = False
    cape_evidence_supports_sparse = False

    profile_state_guard_on = (
        (scene == "profile44" and p.get("dense_guard", True))
        or (scene == "profile44_tiered" and p.get("state_guard", True))
        or (scene == "profile_custom" and p.get("dense_guard", False))
        or (_is_cape and p.get("dense_guard", True))
    )
    diag["profile_state_guard_on"] = bool(profile_state_guard_on)

    if _is_cape and profile_state_guard_on:
        _cape_cells, _cape_ctrl = run_cape_state_controller(
            cells=cells,
            selected=selected,
            filtered=filtered,
            input_mz=input_mz,
            token_profile_prior=token_profile_prior,
            peptide_length=peptide_length,
            precursor_charge=precursor_charge,
            mod_count_est=mod_count_est,
            terminal_mod_est=terminal_mod_est,
            length_adaptive_t_main=length_adaptive_t_main,
            t_sat_val=t_sat_val,
            sat_main_ratio_val=sat_main_ratio_val,
            max_sat_per_cluster_val=max_sat_per_cluster_val,
            max_tokens_per_cluster_val=max_tokens_per_cluster_val,
            require_main_for_sat_val=require_main_for_sat_val,
            p=p,
            residues=residues,
            N_by=N_by,
            min_peaks=min_peaks,
            target_peaks=target_peaks,
            max_peaks=max_peaks,
            strong_main_count=strong_main_count,
            mid_main_count=mid_main_count,
            score_builder=rerank_score,
        )
        cells = _cape_cells
        cape_state = dict(_cape_ctrl.get("cape_state", {}))
        cape_adjustment = str(_cape_ctrl.get("cape_adjustment", "none"))
        cape_allow_rescue = bool(_cape_ctrl.get("cape_allow_rescue", False))
        cape_evidence_supports_sparse = bool(cape_state.get("evidence_supports_sparse", False))
        evidence_cap = int(_cape_ctrl.get("cape_evidence_cap", max_peaks))
        dense_guard_applied = cape_adjustment == "dense_cap"
        dense_guard_cap = evidence_cap if dense_guard_applied else None
        profile44_adjustment = cape_adjustment
        profile44_allow_rescue = cape_allow_rescue
        evidence_supports_sparse = cape_evidence_supports_sparse
        diag["cape_state"] = cape_state
        if "retry_expand_stats" in _cape_ctrl:
            expand_stats = _cape_ctrl["retry_expand_stats"]
            _refresh_expand_diag()

    if _is_sgr:
        sgr_t_main_rescue = max(0.010, length_adaptive_t_main * (0.52 if (z == 2 and L <= 16) else 0.58))
        sgr_t_sat_rescue = max(0.0040, t_sat_val * (0.60 if (z == 2 and L <= 16) else 0.68))
        _sgr_cells, _sgr_diag = run_sgr_global_recovery(
            cells=cells,
            pred_matrix=pred_matrix,
            input_mz=input_mz,
            token_profile_prior=token_profile_prior,
            peptide_length=peptide_length,
            precursor_charge=precursor_charge,
            mod_count_est=mod_count_est,
            terminal_mod_est=terminal_mod_est,
            residues=residues,
            min_peaks=min_peaks,
            target_peaks=target_peaks,
            max_peaks=max_peaks,
            t_main_rescue=sgr_t_main_rescue,
            t_sat_rescue=sgr_t_sat_rescue,
            sgr_generation=str(p.get("sgr_generation", "v1")),
            sgr_main_deficit_ratio=float(p.get("sgr_stageb_main_deficit_ratio", 0.50)),
            sgr_total_deficit_ratio=float(p.get("sgr_stageb_total_deficit_ratio", 1.00)),
            sgr_global_main_budget=int(p.get("sgr_global_main_budget", 48)),
            sgr_global_sat_budget=int(p.get("sgr_global_sat_budget", 96)),
            sgr_global_tier1_budget=int(p.get("sgr_global_tier1_budget", 96)),
            sgr_global_tier2_budget=int(p.get("sgr_global_tier2_budget", 72)),
            score_builder=rerank_score,
        )
        cells = _sgr_cells
        diag["sgr_t_main_rescue"] = float(sgr_t_main_rescue)
        diag["sgr_t_sat_rescue"] = float(sgr_t_sat_rescue)
        diag.update(_sgr_diag)
    if profile_state_guard_on and not _is_cape:
        cape_state = None
        if scene == "profile44_tiered":
            evidence_cap = int(round(12 + 1.45 * mid_main_count + 2.15 * strong_main_count))
            if mod_count_est >= 1:
                evidence_cap += int(round(N_by * 0.12))
            if mod_count_est >= 2:
                evidence_cap += int(round(N_by * 0.18))
            if terminal_mod_est:
                evidence_cap += int(round(N_by * 0.10))
        elif scene == "profile_custom":
            # profile_custom already prunes token eligibility before cluster
            # activation, so it benefits most from a lightweight evidence cap:
            # suppress only clearly over-dense spectra without inheriting the
            # heavier profile44_tiered retry/state logic.
            evidence_cap = int(round(16 + 1.75 * mid_main_count + 2.45 * strong_main_count))
            if mod_count_est >= 1:
                evidence_cap += int(round(N_by * 0.10))
            if mod_count_est >= 2:
                evidence_cap += int(round(N_by * 0.15))
            if terminal_mod_est:
                evidence_cap += int(round(N_by * 0.08))
        else:
            evidence_cap = int(round(18 + 2.2 * mid_main_count + 2.8 * strong_main_count))
        if z == 2:
            if L <= 16:
                if scene in ("profile44_tiered", "profile_custom") and mod_count_est <= 1:
                    z2_cap_ratio = 1.60 if L <= 12 else 1.75
                    z2_floor = 60 if L <= 8 else 75
                    z2_lowmod_ceiling = int(max(min_peaks, z2_floor, N_by * z2_cap_ratio))
                    diag["dense_guard_z2_lowmod_ceiling"] = int(z2_lowmod_ceiling)
                    evidence_cap = min(evidence_cap, z2_lowmod_ceiling)
                else:
                    z2_cap_ratio = 2.35 if scene == "profile_custom" else 2.55
                    evidence_cap = min(evidence_cap, int(max(80, N_by * z2_cap_ratio)))
            elif L <= 24:
                z2_cap_ratio = (
                    2.25
                    if scene in ("profile44_tiered", "profile_custom") and mod_count_est <= 1
                    else 2.60
                )
                evidence_cap = min(evidence_cap, int(max(120, N_by * z2_cap_ratio)))
        elif L <= 16:
            short_cap_ratio = 2.35 if scene == "profile_custom" else 2.60
            evidence_cap = min(evidence_cap, int(max(150, N_by * short_cap_ratio)))

        evidence_cap = max(min_peaks, evidence_cap)

        if scene == "profile44_tiered":
            sparse_mid_need = max(4, min(L, 20) // (3 if mod_count_est >= 2 else 2))
            sparse_strong_need = max(2, min(L, 20) // (4 if mod_count_est >= 2 else 3))
            evidence_supports_sparse = (
                mid_main_count >= sparse_mid_need
                or strong_main_count >= sparse_strong_need
                or (mod_count_est >= 2 and mid_main_count >= 4)
                or (terminal_mod_est and strong_main_count >= 2)
            )
        elif scene == "profile_custom":
            sparse_mid_need = max(4, min(L, 16) // 2)
            sparse_strong_need = max(2, min(L, 16) // 3)
            evidence_supports_sparse = (
                mid_main_count >= sparse_mid_need
                or strong_main_count >= sparse_strong_need
                or (mod_count_est >= 2 and mid_main_count >= 4)
            )
        else:
            sparse_mid_need = max(4, min(L, 16) // 2)
            sparse_strong_need = max(2, min(L, 16) // 3)
            evidence_supports_sparse = (
                mid_main_count >= sparse_mid_need
                or strong_main_count >= sparse_strong_need
            )
        diag["profile44_sparse_mid_need"] = int(sparse_mid_need)
        diag["profile44_sparse_strong_need"] = int(sparse_strong_need)

        over_dense_ratio = len(cells) / max(N_by, 1)
        if scene == "profile44_tiered" and z == 2 and L <= 16:
            dense_guard_start = 2.05 if mod_count_est <= 1 else 2.35
        elif scene == "profile_custom" and z == 2 and L <= 16:
            dense_guard_start = 1.95 if mod_count_est <= 1 else 2.20
        elif scene == "profile_custom" and z == 2:
            dense_guard_start = 2.20
        elif scene == "profile_custom":
            dense_guard_start = 2.20 if L <= 16 else 2.35
        elif scene == "profile44_tiered" and z == 2:
            dense_guard_start = 2.30
        elif z == 2 and L <= 16:
            dense_guard_start = 3.20
        elif z == 2:
            dense_guard_start = 3.00
        else:
            dense_guard_start = 2.55 if scene == "profile44_tiered" else 2.80
        weak_anchor_for_dense = (
            strong_main_count < max(6, L // 2)
            and mid_main_count < max(12, L)
        )
        short_z2_lowmod_overfill = (
            scene in ("profile44_tiered", "profile_custom", "cape_v1", "cape_v2", "cape_v3")
            and z == 2
            and L <= 16
            and mod_count_est <= 1
            and over_dense_ratio >= dense_guard_start
        )
        diag["dense_guard_over_dense_ratio"] = float(over_dense_ratio)
        diag["dense_guard_start_ratio"] = float(dense_guard_start)
        diag["dense_guard_weak_anchor"] = bool(weak_anchor_for_dense)
        diag["dense_guard_short_z2_lowmod_overfill"] = bool(short_z2_lowmod_overfill)

        if (
            len(cells) > evidence_cap
            and (over_dense_ratio >= dense_guard_start and (weak_anchor_for_dense or short_z2_lowmod_overfill))
        ):
            cells = sorted(cells, key=lambda c: -float(c.get("export_score", 0.0)))[:evidence_cap]
            dense_guard_applied = True
            dense_guard_cap = evidence_cap
            profile44_adjustment = "dense_cap"
        elif len(cells) < min_peaks and scene not in ("profile_custom",):
            if evidence_supports_sparse:
                retry_max_sat = max(max_sat_per_cluster_val, 6 if L <= 12 else 5)
                if max_tokens_per_cluster_val is None:
                    retry_max_tokens = 6 if L <= 12 else 5
                else:
                    retry_max_tokens = max(max_tokens_per_cluster_val, 7 if L <= 8 else 6)
                retry_core_budget = int(p.get("core_token_budget", 3))
                retry_profile_budget = int(p.get("profile_extra_budget", 2))
                retry_low_budget = int(p.get("low_prior_budget", 1))
                if scene == "profile44_tiered":
                    retry_core_budget += 1
                    retry_profile_budget += 1
                    if mod_count_est >= 2 or terminal_mod_est:
                        retry_low_budget += 1
                retry_cells, retry_stats = expand_clusters_to_cells(
                    selected, filtered, input_mz,
                    t_main=length_adaptive_t_main,
                    t_sat=t_sat_val,
                    sat_main_ratio=sat_main_ratio_val,
                    max_sat_per_cluster=retry_max_sat,
                    token_profile_prior=token_profile_prior,
                    score_builder=rerank_score,
                    max_tokens_per_cluster=retry_max_tokens,
                    require_main_for_sat=require_main_for_sat_val,
                    token_tiered_budget=bool(p.get("token_tiered_budget", False)),
                    profile_top_n=int(p.get("profile_top_n", 16)),
                    core_token_budget=retry_core_budget,
                    profile_extra_budget=retry_profile_budget,
                    low_prior_budget=retry_low_budget,
                    low_prior_strong_factor=float(p.get("low_prior_strong_factor", 2.0)),
                    cape_mode=bool(p.get("cape_mode", False)),
                    peptide_length=peptide_length,
                    precursor_charge=precursor_charge,
                    mod_count_est=mod_count_est,
                    terminal_mod_est=terminal_mod_est,
                    residues=residues,
                    cape_tier1_n=int(p.get("cape_tier1_n", 8)),
                    cape_tier2_n=int(p.get("cape_tier2_n", 8)),
                    cape_generation=str(p.get("cape_generation", "v1")),
                )
                diag["profile44_sparse_retry_before"] = len(cells)
                diag["profile44_sparse_retry_after"] = len(retry_cells)
                diag["profile44_sparse_retry_max_tokens"] = retry_max_tokens
                diag["profile44_sparse_retry_max_sat"] = retry_max_sat
                diag["profile44_sparse_retry_core_budget"] = int(retry_core_budget)
                diag["profile44_sparse_retry_profile_budget"] = int(retry_profile_budget)
                diag["profile44_sparse_retry_low_budget"] = int(retry_low_budget)
                if len(retry_cells) > len(cells):
                    cells = retry_cells
                    expand_stats = retry_stats
                    _refresh_expand_diag()
                profile44_adjustment = "sparse_retry"
                profile44_allow_rescue = len(cells) < min_peaks
            else:
                profile44_adjustment = "low_evidence_no_rescue"

        if (
            len(cells) > evidence_cap
            and profile44_adjustment != "dense_cap"
            and (len(cells) / max(N_by, 1)) >= dense_guard_start
            and (weak_anchor_for_dense or short_z2_lowmod_overfill)
        ):
            diag["dense_guard_over_dense_ratio_after_retry"] = float(len(cells) / max(N_by, 1))
            cells = sorted(cells, key=lambda c: -float(c.get("export_score", 0.0)))[:evidence_cap]
            dense_guard_applied = True
            dense_guard_cap = evidence_cap
            if profile44_adjustment == "sparse_retry":
                profile44_adjustment = "sparse_retry_capped"
                profile44_allow_rescue = False
            else:
                profile44_adjustment = "dense_cap"

    diag["profile44_adjustment"] = profile44_adjustment
    diag["profile44_allow_rescue"] = bool(profile44_allow_rescue)
    diag["profile44_evidence_supports_sparse"] = bool(evidence_supports_sparse)
    diag["profile44_evidence_cap"] = int(evidence_cap)
    diag["profile44_sparse_retry"] = profile44_adjustment.startswith("sparse_retry")
    diag["dense_guard_applied"] = bool(dense_guard_applied)
    diag["dense_guard_cap"] = int(dense_guard_cap) if dense_guard_cap is not None else 0
    diag["dense_guard_cells_after"] = len(cells)
    if _is_cape:
        if cape_adjustment == "none":
            cape_adjustment = str(profile44_adjustment)
        cape_allow_rescue = bool(cape_allow_rescue or profile44_allow_rescue)
        cape_evidence_supports_sparse = bool(cape_evidence_supports_sparse or evidence_supports_sparse)
        diag["cape_adjustment"] = str(cape_adjustment)
        diag["cape_allow_rescue"] = bool(cape_allow_rescue)
        diag["cape_evidence_supports_sparse"] = bool(cape_evidence_supports_sparse)
        diag["cape_evidence_cap"] = int(evidence_cap)
        diag["cape_sparse_retry"] = str(cape_adjustment).startswith("sparse_retry")
        diag["cape_dense_guard_applied"] = bool(dense_guard_applied)
        diag["cape_dense_guard_cap"] = int(dense_guard_cap) if dense_guard_cap is not None else 0
        diag["cape_dense_guard_cells_after"] = len(cells)
        diag["cape_final_budget_used"] = len(cells)

    # Step 9: Hard cap with main-first tiebreaker
    n_before_cap = len(cells)
    if len(cells) > max_peaks:
        # Sort: main peaks first, then by export_score (improvement 4)
        cells_sorted = sorted(cells, key=lambda c: (not c.get("is_main", False), -c["export_score"]))
        capped_cells = cells_sorted[max_peaks:]
        for cc in capped_cells:
            k = cc.get("_token_kind", "other")
            if k == "main":           expand_stats["n_capped_main"] += 1
            elif k == "isotope":      expand_stats["n_capped_isotope"] += 1
            elif k == "neutral_loss": expand_stats["n_capped_loss"] += 1
            elif k == "combined_loss": expand_stats["n_capped_combined"] += 1
            else:                     expand_stats["n_capped_other"] += 1
        cells = cells_sorted[:max_peaks]
    diag["n_cells_after_cap"] = len(cells)
    diag["n_cells_capped"] = n_before_cap - len(cells)
    diag["n_cells_before_cap"] = n_before_cap
    diag["n_cells_before_rescue_cap"] = len(cells)
    diag["n_cells_exported"] = len(cells)
    for k in ["n_capped_main", "n_capped_isotope", "n_capped_loss", "n_capped_combined", "n_capped_other"]:
        diag[k] = expand_stats.get(k, 0)

    # Step 10: Theory-budget rescue (triggers on final_peaks < min_peaks, not just main count)
    _is_profile_family = scene in (
        "profile_top14",
        "profile_top16",
        "profile_top20",
        "profile_tier",
        "profile_custom",
        "cape_v1",
        "cape_v2",
        "cape_v3",
    )
    if scene in ("profile44", "profile44_tiered"):
        t_main_rescue = max(0.010, length_adaptive_t_main * 0.55)
    elif _is_profile_family:
        # profile-top/profile-tier scenes use much lower calibrated thresholds.
        # Keeping the old 0.03 rescue floor makes rescue stricter than normal
        # export and effectively disables recovery on z=2 / short spectra.
        rescue_scale = 0.60
        if z == 2 and L <= 16:
            rescue_scale = 0.50
        elif z == 2 or L <= 12:
            rescue_scale = 0.55
        if mod_count_est >= 2 or terminal_mod_est:
            rescue_scale *= 0.95
        t_main_rescue = max(0.010, length_adaptive_t_main * rescue_scale)
    else:
        t_main_rescue = max(0.030, length_adaptive_t_main * 0.65)
    diag["t_main_rescue"] = t_main_rescue
    _is_cosine_eval = (scene == "cosine_eval")
    _is_profile44 = (scene in ("profile44", "profile44_tiered"))
    n_total_before_rescue = len(cells)
    diag["n_before_rescue"] = n_total_before_rescue

    _profile44_rescue_allowed = (not _is_profile44) or bool(diag.get("profile44_allow_rescue", False))
    if _is_profile_family:
        profile_rescue_ratio = 0.82
        if z == 2 and L <= 16:
            profile_rescue_ratio = 0.92
        elif z == 2 or L <= 12 or mod_count_est >= 2 or terminal_mod_est:
            profile_rescue_ratio = 0.88
        sat_rescue_need = max(min_peaks, int(target_peaks * profile_rescue_ratio))
    else:
        sat_rescue_need = int(target_peaks * 0.65) if _is_cosine_eval else min_peaks
    rescue_entry_need = sat_rescue_need if _is_profile_family else min_peaks
    diag["rescue_entry_need"] = int(rescue_entry_need)

    if n_total_before_rescue < rescue_entry_need and _profile44_rescue_allowed:
        rescue_candidates = []
        export_token_set = set(_EXPORT_TOKENS.get(scene, _EXPORT_TOKENS["search"]))
        def _rescue_prior(tok):
            if token_profile_prior is not None:
                return float(token_profile_prior[int(tok)])
            return _token_prior_func(tok)
        # Phase 1: main-token candidates (all scenes)
        for pos in range(pred_matrix.shape[0]):
            if pos >= L - 1:
                continue
            for ch in range(N_CHANNELS):
                if not build_charge_channel_mask(precursor_charge)[ch]:
                    continue
                score = float(pred_matrix[pos, MAIN_TOKEN, ch])
                if score < t_main_rescue:
                    continue
                if any(c["pos"] == pos + 1 and c["channel"] == CHANNEL_NAMES[ch] and c.get("is_main") for c in cells):
                    continue
                mz = float(input_mz[pos, MAIN_TOKEN, ch])
                rescue_candidates.append({
                    "mz": mz, "intensity": score, "norm_intensity": score,
                    "support_score": 1.0,
                    "export_score": rerank_score(
                        model_score=score,
                        tok=MAIN_TOKEN,
                        kind="main",
                        main_score=score,
                        pos=pos,
                        channel=ch,
                        pred_matrix=pred_matrix,
                        peptide_length=peptide_length,
                        precursor_charge=precursor_charge,
                        residues=residues,
                        token_profile_prior=token_profile_prior,
                    ),
                    "pos": int(pos + 1), "token_idx": MAIN_TOKEN,
                    "channel": CHANNEL_NAMES[ch], "is_main": True, "is_rescue": True,
                })
        # Phase 2: satellite rescue for cosine/profile modes, when still below target.
        # profile44 uses the frozen token prior and full token set, so this remains
        # model-output based and does not use test MGF peak counts.
        _sat_rescue_trigger = (
            (_is_cosine_eval or _is_profile44 or _is_profile_family)
            and _profile44_rescue_allowed
            and n_total_before_rescue < sat_rescue_need
        )
        if _sat_rescue_trigger:
            if _is_profile44:
                t_sat_rescue = max(0.006, t_sat_val * 0.55)
            elif _is_profile_family:
                sat_scale = 0.78
                if z == 2 and L <= 16:
                    sat_scale = 0.62
                elif z == 2 or L <= 12:
                    sat_scale = 0.70
                if mod_count_est >= 2 or terminal_mod_est:
                    sat_scale *= 0.95
                t_sat_rescue = max(0.0045, t_sat_val * sat_scale)
            else:
                t_sat_rescue = max(0.012, t_sat_val * 0.55)
            diag["sat_rescue_trigger"] = True
            diag["t_sat_rescue"] = t_sat_rescue
            if scene == "profile44_tiered" and token_profile_prior is not None:
                _profile_order = list(np.argsort(-np.asarray(token_profile_prior, dtype=np.float32)))
                _stable_tokens = set(CORE8_TOKENS)
                _stable_tokens.update(int(t) for t in _profile_order[:int(p.get("profile_top_n", 16))])
                sat_tokens = [t for t in _profile_order if t in _stable_tokens and t != MAIN_TOKEN]
            elif _is_cape:
                _cape_tokens, _cape_tiers = resolve_cape_export_tokens(
                    token_profile_prior=token_profile_prior,
                    core_tokens=sorted(CORE8_TOKENS),
                    tier1_n=int(p.get("cape_tier1_n", 8)),
                    tier2_n=int(p.get("cape_tier2_n", 8)),
                )
                _tier1 = [int(t) for t in _cape_tiers["tier1"] if int(t) != MAIN_TOKEN]
                _tier2 = [int(t) for t in _cape_tiers["tier2"] if int(t) != MAIN_TOKEN]
                _core_sat = [t for t in sorted(CORE8_TOKENS) if int(t) != MAIN_TOKEN]
                sat_tokens = []
                for group in (_core_sat, _tier1, _tier2):
                    for tok in group:
                        if tok not in sat_tokens:
                            sat_tokens.append(tok)
            elif _is_profile_family:
                sat_tokens = sorted(
                    [int(t) for t in export_token_set if int(t) != MAIN_TOKEN],
                    key=lambda tok: _rescue_prior(tok),
                    reverse=True,
                )
            elif _is_profile44:
                sat_tokens = [t for t in export_token_set if t != MAIN_TOKEN]
            else:
                sat_tokens = [t for t in SATELLITE_TOKENS + [19, 20] if t in export_token_set]
            _sat_added_this_spectrum = 0
            _sat_per_cluster = {}  # (pos,ch) -> count
            _profile_deficit = max(0, min_peaks - n_total_before_rescue)
            if _is_cape:
                _sat_limit = min(72, max(12, target_peaks - n_total_before_rescue))
                if z == 2 and L <= 16:
                    _sat_limit = min(84, max(_sat_limit, int(target_peaks - n_total_before_rescue)))
                _sat_per_cluster_limit = 3 if (z == 2 and L <= 16) or mod_count_est >= 1 or terminal_mod_est else 2
            elif _is_profile_family:
                _sat_limit = min(48, max(10, target_peaks - n_total_before_rescue))
                if z == 2 and L <= 16:
                    _sat_limit = min(60, max(_sat_limit, int(target_peaks - n_total_before_rescue)))
                _sat_per_cluster_limit = 2 if len(export_token_set) <= 14 else 3
            else:
                _sat_limit = min(30, max(8, _profile_deficit)) if _is_profile44 else 20
                _sat_per_cluster_limit = 2 if _is_profile44 else 1
            for pos in range(pred_matrix.shape[0]):
                if pos >= L - 1:
                    continue
                for ch in range(N_CHANNELS):
                    ch_mask = build_charge_channel_mask(precursor_charge)
                    if not ch_mask[ch]:
                        continue
                    main_score = float(pred_matrix[pos, MAIN_TOKEN, ch])
                    if _is_profile_family:
                        cluster_anchor = 0.0
                        for tok in export_token_set:
                            cluster_anchor = max(
                                cluster_anchor,
                                float(pred_matrix[pos, int(tok), ch]) * _rescue_prior(int(tok)),
                            )
                        if cluster_anchor < max(t_sat_rescue, t_main_rescue * 0.85):
                            continue
                    else:
                        if main_score < t_main_rescue:
                            continue
                    for tok in sat_tokens:
                        if _sat_added_this_spectrum >= _sat_limit:  # per-spectrum limit
                            break
                        score = float(pred_matrix[pos, tok, ch])
                        if score < t_sat_rescue:
                            continue
                        if any(c["pos"] == pos + 1 and c["channel"] == CHANNEL_NAMES[ch] and c["token_idx"] == tok for c in cells):
                            continue
                        _ck = (pos, ch)
                        if _sat_per_cluster.get(_ck, 0) >= _sat_per_cluster_limit:  # per-cluster limit
                            continue
                        mz = float(input_mz[pos, tok, ch])
                        rescue_candidates.append({
                            "mz": mz, "intensity": score, "norm_intensity": score,
                            "support_score": 1.0,
                            "export_score": rerank_score(
                                model_score=score,
                                tok=tok,
                                kind=TOKEN_KIND.get(int(tok), "other"),
                                main_score=float(pred_matrix[pos, MAIN_TOKEN, ch]),
                                pos=pos,
                                channel=ch,
                                pred_matrix=pred_matrix,
                                peptide_length=peptide_length,
                                precursor_charge=precursor_charge,
                                residues=residues,
                                token_profile_prior=token_profile_prior,
                            ),
                            "pos": int(pos + 1), "token_idx": tok,
                            "channel": CHANNEL_NAMES[ch], "is_main": False, "is_rescue": True,
                        })
                        _sat_added_this_spectrum += 1
                        _sat_per_cluster[_ck] = _sat_per_cluster.get(_ck, 0) + 1
            diag["n_sat_rescued"] = _sat_added_this_spectrum
        else:
            diag["sat_rescue_trigger"] = False
            diag["n_sat_rescued"] = 0

        rescue_candidates.sort(key=lambda c: (-c.get("is_main", False), -c["export_score"]))
        rescue_goal = min_peaks if _is_profile44 else target_peaks
        needed = max(0, rescue_goal - n_total_before_rescue)
        n_added = min(needed, len(rescue_candidates))
        cells = cells + rescue_candidates[:n_added]
        if len(cells) > max_peaks:
            cells_sorted = sorted(cells, key=lambda c: (not c.get("is_main", False), -c["export_score"]))
            cells = cells_sorted[:max_peaks]
        diag["n_rescue_candidates"] = len(rescue_candidates)
        diag["n_main_rescued"] = n_added
        diag["n_cells_exported"] = len(cells)
    else:
        diag["sat_rescue_trigger"] = False
        diag["n_main_rescued"] = 0
        diag["n_sat_rescued"] = 0
        diag["n_rescue_candidates"] = 0

    # Final assertion + theory_ratio
    diag["final_peaks_ok"] = (len(cells) <= max_peaks)
    diag["theory_ratio"] = len(cells) / max(N_by, 1)
    diag["n_cells_exported"] = len(cells)

    if return_diagnostics:
        return cells, diag
    return cells


# ---------------------------------------------------------------------------
# Ablation framework helpers
# ---------------------------------------------------------------------------
ABLATION_STEPS = [
    ("A0", "baseline", "Native postprocessing baseline"),
    ("A1", "position_mask", "A0 + position theoretical mask"),
    ("A2", "charge_mask", "A1 + charge-aware channel mask"),
    ("A3", "isolated_sat", "A2 + isolated satellite removal + weak main peak protection"),
    ("A4", "dual_threshold", "A3 + main-peak / satellite dual-threshold adaptive"),
    ("A5", "cluster_topk", "A4 + cluster-level Top-K R-value control"),
    ("A6", "merge_tol", "A5 + merge_tol ablation (0.05 / 0.08 / 0.10)"),
    ("A7", "scene_templates", "A5 + three scene parameter templates"),
]


def get_ablation_config(step_id):
    """Return the configuration for a given ablation step.

    Args:
        step_id: one of 'A0'-'A7'

    Returns:
        dict with enabled flags
    """
    configs = {
        "A0": {"position_mask": False, "charge_mask": False,
               "isolated_sat": False, "dual_threshold": False,
               "cluster_topk": False, "scene": "search"},
        "A1": {"position_mask": True, "charge_mask": False,
               "isolated_sat": False, "dual_threshold": False,
               "cluster_topk": False, "scene": "search"},
        "A2": {"position_mask": True, "charge_mask": True,
               "isolated_sat": False, "dual_threshold": False,
               "cluster_topk": False, "scene": "search"},
        "A3": {"position_mask": True, "charge_mask": True,
               "isolated_sat": True, "dual_threshold": False,
               "cluster_topk": False, "scene": "search"},
        "A4": {"position_mask": True, "charge_mask": True,
               "isolated_sat": True, "dual_threshold": True,
               "cluster_topk": False, "scene": "search"},
        "A5": {"position_mask": True, "charge_mask": True,
               "isolated_sat": True, "dual_threshold": True,
               "cluster_topk": True, "scene": "search"},
        "A6": {"position_mask": True, "charge_mask": True,
               "isolated_sat": True, "dual_threshold": True,
               "cluster_topk": True, "scene": "search"},
        "A7": {"position_mask": True, "charge_mask": True,
               "isolated_sat": True, "dual_threshold": True,
               "cluster_topk": True, "scene": "search"},
    }
    return configs.get(step_id, configs["A5"])


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------
def summarize_diagnostics(all_diags):
    """Aggregate per-spectrum diagnostics into summary statistics.

    Args:
        all_diags: list of per-spectrum diagnostics dicts

    Returns:
        dict of aggregate statistics
    """
    if not all_diags:
        return {}

    keys_numeric = [
        "n_nonzero_before_position_mask",
        "n_nonzero_after_position_mask",
        "n_nonzero_after_charge_mask",
        "n_nonzero_after_dual_threshold",
        "n_charge_invalid_removed",
        "n_clusters_active",
        "n_weak_main",
        "n_isolated_sat_removed",
        "n_clusters_built",
        "n_clusters_before_topk",
        "K_target",
        "n_clusters_selected",
        "n_cells_exported",
        "n_main_cells",
        "n_sat_cells",
        "cape_tier0_exported",
        "cape_tier1_exported",
        "cape_tier2_exported",
        "cape_tail_exported",
        "cape_budget_total",
        "cape_budget_used",
        "cape_pre_rescue_budget_used",
        "cape_final_budget_used",
        "sgr_added_main",
        "sgr_added_sat",
        "sgr_added_tier1",
        "sgr_added_tier2",
        "sgr_main_candidate_pool",
        "sgr_sat_candidate_pool",
        "sgr_tier1_candidate_pool",
        "sgr_tier2_candidate_pool",
        "sgr_main_deficit",
        "sgr_position_deficit",
        "sgr_total_deficit",
        "n_main_rescued",
        "n_sat_rescued",
    ]

    summary = {}
    for key in keys_numeric:
        values = [d.get(key, 0) for d in all_diags if key in d]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_sum"] = int(np.sum(values))

    adjustment_values = [str(d.get("profile44_adjustment", "")) for d in all_diags]
    adjustment_values = [v for v in adjustment_values if v]
    if adjustment_values:
        total = len(adjustment_values)
        for name in sorted(set(adjustment_values)):
            count = sum(1 for v in adjustment_values if v == name)
            summary[f"profile44_adjustment_{name}_count"] = count
            summary[f"profile44_adjustment_{name}_pct"] = 100.0 * count / max(total, 1)

    cape_adjustment_values = [str(d.get("cape_adjustment", "")) for d in all_diags]
    cape_adjustment_values = [v for v in cape_adjustment_values if v]
    if cape_adjustment_values:
        total = len(cape_adjustment_values)
        for name in sorted(set(cape_adjustment_values)):
            count = sum(1 for v in cape_adjustment_values if v == name)
            summary[f"cape_adjustment_{name}_count"] = count
            summary[f"cape_adjustment_{name}_pct"] = 100.0 * count / max(total, 1)

    # Total deletions
    if "n_nonzero_before_position_mask_sum" in summary and "n_cells_exported_sum" in summary:
        summary["total_cells_removed"] = (
            summary["n_nonzero_before_position_mask_sum"]
            - summary["n_cells_exported_sum"]
        )

    return summary
