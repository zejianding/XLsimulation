#!/usr/bin/env python3
"""V20_0701 simplified rescoring: 0.50·postproc + 0.50·grid + hard-5ppm filter.

Best mode (PR@1=0.6018 on mouse): hard filter 5 ppm + 0.50·postproc_piprime
+ 0.50·grid_cosine + pn_fusion α=0.5. Physical meaning: discard candidates
exceeding instrument mass accuracy (5 ppm for high-res MS), no per-dataset
tuning needed.

Run from v20_0701/transfomer/:
  python scripts/rescoring/rescore_simplified.py --parquet ... --mgf-dir ... \\
      --checkpoint ... --piprime-model ... --output-dir ... --gpu 0 --row-order \\
      --fusion-alpha 0.5 --skip-results-save
"""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_TRANSFORMER_DIR = _SCRIPTS_DIR.parent
if str(_TRANSFORMER_DIR) not in sys.path: sys.path.insert(0, str(_TRANSFORMER_DIR))
if str(_SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import numpy.core as _nc, numpy.core.multiarray as _nma
    sys.modules.setdefault("numpy._core", _nc); sys.modules.setdefault("numpy._core.multiarray", _nma)
except Exception: pass
if os.name == "nt":
    try: import pathlib; pathlib.PosixPath = pathlib.WindowsPath
    except Exception: pass

import candidate_rescoring as _cr


def _gc_similarity(pv, fv, mode="cosine", topk=None, min_intens=0.0):
    """Compute grid similarity with optional metric.

    topk: keep only the top-K predicted ions (by predicted intensity). The
    low-intensity predicted ions are "predicted but not observed", and only
    drag the cosine denominator up; dropping them focuses on the confident
    ions. None = keep all (current behavior).

    min_intens: relative intensity threshold in [0,1). A cell is dropped only
    when its PREDICTED intensity is below min_intens * pred_max AND it has no
    projected experimental peak (filt == 0) — i.e. pure model noise. Cells with
    a real experimental peak (filt > 0) are always kept regardless of predicted
    intensity; the experimental side is never filtered by its own base-peak
    ratio. 0.0 = no filtering.

    实验峰强度空间对齐由 --exp-int-transform 在投影前统一处理 (不再在 gc 分支内做
    sqrt/log 局部变换, 也不再做 pred 侧 restore)。
    """
    pv = np.asarray(pv, dtype=np.float32)
    fv = np.asarray(fv, dtype=np.float32)
    if min_intens > 0:
        # 只过滤「pred 弱 且 filt 无峰」的纯噪声格子：
        #   - pred 弱 + filt 无峰  → 模型噪声, 丢弃
        #   - pred 弱 + filt 有峰  → 真实实验峰, 保留(不丢实验证据)
        #   - pred 强(不管 filt)   → 保留
        # 实验谱(filt)本身不做相对强度过滤。
        pmax = float(pv.max()) if pv.size else 0.0
        if pmax > 0:
            keep = ~((pv < min_intens * pmax) & (fv == 0))
            pv = pv[keep]
            fv = fv[keep]
    if topk is not None and topk > 0 and topk < len(pv):
        idx = np.argpartition(pv, -topk)[-topk:]
        pv = pv[idx]
        fv = fv[idx]
    if mode == "jaccard":
        pb = pv > 0.0
        fb = fv > 0.0
        inter = int(np.logical_and(pb, fb).sum())
        union = int(np.logical_or(pb, fb).sum())
        return float(inter / union) if union > 0 else 0.0
    if mode == "support":
        mask = (pv > 0.0) & (fv > 0.0)
        if int(mask.sum()) == 0:
            return 0.0
        return _cr._safe_cosine(pv[mask], fv[mask])
    return _cr._safe_cosine(pv, fv)  # cosine


def _simplified_score(pred_grid, filt_grid, valid_mask, input_mz,
                       precursor_mz, precursor_charge, min_mz, grid_weight=0.5,
                       gc_mode="cosine", gc_topk=None,
                       gc_min_intens=0.0, pp_min_intens=0.005):
    # pp 路径的 pred 强度过滤: 相对基峰阈值 -> 绝对阈值 (prediction_to_mgf_peaks
    # 的 min_intensity 是绝对强度)。0 = 不过滤。与 gc_min_intens 一样只对 pred 去噪,
    # filt 侧在 score_postprocessed_piprime_cosine 内部恒用 min_intensity=0.0。
    pp_abs_intens = float(pred_grid.max()) * pp_min_intens if pp_min_intens > 0 else 0.0
    pp = _cr.score_postprocessed_piprime_cosine(
        pred_grid, filt_grid, valid_mask, input_mz,
        precursor_mz, precursor_charge,
        min_mz=min_mz, min_intensity=pp_abs_intens, mz_merge_tol=0.05)
    valid = valid_mask > 0
    pv = pred_grid[valid].astype(np.float32); fv = filt_grid[valid].astype(np.float32)
    if pv.size == 0:
        return 0.0, {"grid_cosine": 0.0, "piprime_score": pp}
    gc = _gc_similarity(pv, fv, mode=gc_mode, topk=gc_topk,
                        min_intens=gc_min_intens)
    score = max(0.0, (1.0 - grid_weight) * float(pp) + grid_weight * float(gc))
    return score, {"grid_cosine": float(gc), "piprime_score": float(pp)}


def _patched_rescore(scan_id, candidates, original_scores, model, model_config,
                     stats, device, exp_peaks, exp_charge, exp_pepmass, exp_rt,
                     args, ground_truth=None):
    mz_tol = float(args.mz_tol); min_mz = float(args.min_mz)
    grid_weight = float(getattr(args, "grid_weight", 0.5))
    gc_mode = str(getattr(args, "gc_mode", "cosine"))
    gc_topk = int(getattr(args, "gc_topk", 0)) if getattr(args, "gc_topk", 0) else None
    gc_min_intens = float(getattr(args, "gc_min_intens", 0.0))
    # 实验峰强度变换 (投影前, 影响 pp/gc 全部分支): 对齐模型训练空间
    if str(getattr(args, "exp_int_transform", "off")) == "sqrtclip":
        _arr = np.array(exp_peaks, dtype=np.float32)
        if _arr.size:
            _arr[:, 1] = np.sqrt(np.clip(_arr[:, 1], 0, 50000)) / np.sqrt(50000)
        exp_peaks = _arr
    pp_min_intens = float(getattr(args, "pp_min_intens", 0.005))
    fuse_mode = str(getattr(args, "fuse_mode", "weighted")).lower()
    unit = str(getattr(args, "mass_threshold_unit", "ppm")).lower()
    ppm_deviations = np.array([_cr.compute_mass_ppm_deviation(s, exp_pepmass, exp_charge)
                                for s in candidates])
    mass_da = np.array([_cr.compute_mass_da_deviation(s, exp_pepmass, exp_charge)
                        for s in candidates])
    filter_dev = mass_da if unit == "da" else ppm_deviations
    hard_pass = filter_dev <= args.precursor_hard_threshold
    rescue = str(getattr(args, "mass_rescue", "off"))
    filter_after = str(getattr(args, "filter_mode", "before")) == "after"
    pn_best = float(max(original_scores)) if original_scores else 0.0

    pred_grids, input_grids, valid_masks = _cr.batch_predict_candidates(
        model=model, model_config=model_config, candidates=candidates,
        precursor_charge=exp_charge, pepmass=exp_pepmass, rt=exp_rt,
        stats=stats, device=device, min_mz=min_mz)

    results = []
    for i, (seq, pg, ig, vm, ppm_d, passes_h) in enumerate(
        zip(candidates, pred_grids, input_grids, valid_masks, ppm_deviations, hard_pass)):
        # filter_after (HSDF 式): 先对所有候选重打分; before: 只对质量通过的候选打分
        if filter_after or passes_h:
            fg = _cr.project_experimental_to_grid(exp_peaks, ig, vm, mz_tol=mz_tol)
            bs, diag = _simplified_score(pg, fg, vm, ig,
                                         float(exp_pepmass), int(exp_charge), min_mz,
                                         grid_weight=grid_weight,
                                         gc_mode=gc_mode,
                                         gc_topk=gc_topk, gc_min_intens=gc_min_intens,
                                         pp_min_intens=pp_min_intens)
            pp_val = diag["piprime_score"]; n_filt = int((fg > 0).sum())
        else:
            bs, diag = 0.0, {"grid_cosine": 0.0, "piprime_score": 0.0}
            pp_val, n_filt = 0.0, 0
        final_score = bs
        results.append({"seq": seq,
            "original_score": original_scores[i] if i < len(original_scores) else np.nan,
            "piprime_score": round(float(pp_val), 6),
            "v20_score_model": round(float(final_score), 6),
            "v20_rescue_score": 0.0,  # 仅在 rescue_applied 的谱上补算 (见下方)
            "v20_score": round(float(final_score), 6),
            "grid_cosine": round(float(diag.get("grid_cosine", 0)), 6),
            "mass_ppm": round(float(ppm_d), 2), "n_filt_slots": n_filt,
            "mass_da": round(float(mass_da[i]), 4),
            # after 模式全部候选都重打分并参与融合, 故该标志置 True; 排除由下方 post-fusion 过滤完成
            "passes_hard_filter": True if filter_after else bool(passes_h)})

    _cr._annotate_candidate_context(results)

    # --fuse-mode score-avg: DS 同款, 各分量(pp/gc)在本谱候选集内 minmax 归一化
    # 后加权平均 (对齐 ds 的 hs_d020_f020)。weighted(默认) = _simplified_score 的裸加权。
    if fuse_mode == "score-avg":
        pp_arr = np.array([r["piprime_score"] for r in results], dtype=np.float64)
        gc_arr = np.array([r["grid_cosine"] for r in results], dtype=np.float64)

        def _mm01(a):
            lo, hi = float(a.min()), float(a.max())
            return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

        fused = (1.0 - grid_weight) * _mm01(pp_arr) + grid_weight * _mm01(gc_arr)
        for r, v in zip(results, fused):
            r["v20_score_model"] = round(float(v), 6)

    alpha = float(getattr(args, "fusion_alpha", 0.7))
    if getattr(args, "fusion_mode", "norm") == "raw":
        _cr._apply_fusion_raw(results, alpha)
    else:
        _cr._apply_fusion(results, "pn_fusion", alpha)

    # hard filter: after fusion, force deviation>threshold to bottom (α-independent)
    if filter_after:
        # HSDF 式: 有候选通过阈值 → 只保留通过者; 全部超标 → 全量兜底 (不过滤)
        devs_arr = np.array([r["mass_da"] if unit == "da" else r["mass_ppm"]
                             for r in results])
        pass_any = devs_arr <= args.precursor_hard_threshold
        if pass_any.any():
            for r, ok in zip(results, pass_any):
                if not ok:
                    r["v20_score"] = -1e10
    else:
        for r in results:
            dev = r["mass_da"] if unit == "da" else r["mass_ppm"]
            if dev > args.precursor_hard_threshold:
                r["v20_score"] = -1e10

    results.sort(key=lambda r: r["v20_score"], reverse=True)
    for rank, r in enumerate(results, start=1): r["v20_rank"] = rank
    best = results[0]

    # --mass-rescue: 触发式救援。rescue_applied = 融合 top-1 被质量硬过滤淘汰
    # (整谱候选全部质量超标)。仅这种谱才补算救援分 (V20 补算含 piprime embedding
    # 模型前向, 昂贵), 正常谱零额外开销。v20_alone = 纯 V20 模型分重排
    # (mass-blind); fused_all = 无硬过滤融合分 (α·V20 + (1−α)·PN)。
    best_rescue_seq = None
    rescue_applied = best["v20_score"] <= -1e9
    if rescue != "off" and rescue_applied:
        for i, (pg, ig, vm) in enumerate(zip(pred_grids, input_grids, valid_masks)):
            fg = _cr.project_experimental_to_grid(exp_peaks, ig, vm, mz_tol=mz_tol)
            bs, _ = _simplified_score(pg, fg, vm, ig,
                                      float(exp_pepmass), int(exp_charge), min_mz,
                                      grid_weight=grid_weight,
                                      gc_mode=gc_mode,
                                      gc_topk=gc_topk, gc_min_intens=gc_min_intens,
                                      pp_min_intens=pp_min_intens)
            results[i]["v20_rescue_score"] = round(float(bs), 6)
        if rescue == "fused_all":
            if getattr(args, "fusion_mode", "norm") == "raw":
                for r in results:
                    v20 = r["v20_rescue_score"]
                    pn = max(0.0, min(1.0, float(r["original_score"])))
                    r["v20_score_all"] = round(float(alpha * v20 + (1 - alpha) * pn), 6)
            else:
                v20_raw = _cr._safe_score_array([r["v20_rescue_score"] for r in results])
                pn_raw = _cr._safe_score_array([r["original_score"] for r in results])
                v20_norm = _cr._minmax_normalize(v20_raw)
                pn_norm = _cr._minmax_normalize(pn_raw)
                for idx, r in enumerate(results):
                    r["v20_score_all"] = round(
                        float(alpha * v20_norm[idx] + (1 - alpha) * pn_norm[idx]), 6)
            rescue_sorted = sorted(results, key=lambda r: r["v20_score_all"], reverse=True)
        else:
            rescue_sorted = sorted(results, key=lambda r: r["v20_rescue_score"], reverse=True)
        best_rescue_seq = rescue_sorted[0]["seq"]
        # rescue 结果直接作为 best (部署口径): summary 的 GT top-1/AA 统计即部署结果。
        # best_score 用 rescue 排序分 (fused_all 的 v20_score_all / v20_alone 的 v20_rescue_score)。
        best = rescue_sorted[0]
        best["v20_score"] = round(float(
            best.get("v20_score_all", best.get("v20_rescue_score", 0.0))), 6)

    gt_rank = 0
    aa_matched = 0; aa_truth_len = 0; aa_pred_len = 0
    if ground_truth is not None:
        for r in results:
            if _cr._peptide_matches(r["seq"], ground_truth):
                gt_rank = r["v20_rank"]; break
        best_tokens = _cr._tokenize_cached(best["seq"])
        truth_tokens = _cr._tokenize_cached(ground_truth)
        if best_tokens and truth_tokens:
            matches, _ = _cr._aa_match_prefix_suffix(list(best_tokens), list(truth_tokens))
            aa_matched = int(sum(matches))
            aa_truth_len = len(truth_tokens); aa_pred_len = len(best_tokens)
    return {"scan_id": scan_id, "num_candidates": len(candidates),
            "exp_charge": exp_charge, "exp_pepmass": round(exp_pepmass, 4),
            "n_exp_peaks": int(exp_peaks.shape[0]),
            "best_seq": best["seq"], "best_score": best["v20_score"],
            "best_original_score": best["original_score"],
            "best_mass_ppm": best["mass_ppm"],
            "best_rescue_seq": best_rescue_seq,
            "rescue_applied": rescue_applied,
            "gt_rank": gt_rank, "gt_hit": gt_rank > 0,
            "aa_matched": aa_matched, "aa_truth_len": aa_truth_len, "aa_pred_len": aa_pred_len,
            "candidates": results}

_cr.rescore_scan = _patched_rescore

if __name__ == "__main__":
    for flag in ("--score-method", "--base-score-method"):
        if flag in sys.argv:
            idx = sys.argv.index(flag); sys.argv.pop(idx); sys.argv.pop(idx)
    sys.argv += ["--score-method", "pn_fusion", "--base-score-method", "piprime_cosine"]
    if "--precursor-hard-threshold" not in sys.argv:
        unit = "ppm"
        if "--mass-threshold-unit" in sys.argv:
            unit = sys.argv[sys.argv.index("--mass-threshold-unit") + 1]
        default = "0.1" if unit.lower() == "da" else "5"
        sys.argv += ["--precursor-hard-threshold", default]
    if "--fusion-alpha" not in sys.argv:
        sys.argv += ["--fusion-alpha", "0.5"]
    if "--grid-weight" not in sys.argv:
        sys.argv += ["--grid-weight", "0.5"]
    _cr.main()
