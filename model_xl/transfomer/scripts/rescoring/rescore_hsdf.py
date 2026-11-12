#!/usr/bin/env python3
"""HS-DF 重打分: 用 DS 的 hs_d*_f* 逻辑对候选池重打分。

核心 (与 ds/step4_predict_score.py 的 hs_d020_f020 一致):
  d  = peak_cosine(实验峰, 预测峰theo)          # 谱峰余弦: 以预测峰为轴, 实验峰±tol 取 max 强度
  f  = explained_intensity(实验峰, 预测峰theo)  # 碎片覆盖率: 实验强度被预测峰覆盖的比例
  fused = score-avg(d, f)                        # 各分支在谱内候选集 minmax 归一化后平均

独立于 rescore_simplified 的多分支加权方案, 不复用其打分分支。
候选来源 = PN 候选池 parquet (row-order 与 MGF 逐位对齐)。

用法 (服务器):
  python scripts/rescoring/rescore_hsdf.py \
      --parquet .../v2_ricebean.parquet \
      --mgf .../Vigna-mungo/peaks.db.mgf \
      --checkpoint .../best_model.pt \
      --output-dir ./hsdf_ricebean --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent
_TRANSFORMER_DIR = _SCRIPTS_DIR.parent
_DS_DIR = _TRANSFORMER_DIR / "ds"
for _p in (_TRANSFORMER_DIR, _SCRIPTS_DIR, _DS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ds_common import (  # noqa: E402
    explained_intensity,
    frag_precision,
    iter_mgf_spectra,
    parse_charge_from_meta,
    parse_pepmass_from_meta,
    peak_cosine,
    peptide_matches,
)
from predict_mgf import (  # noqa: E402
    calculate_pepmass,
    load_checkpoint,
    prediction_to_mgf_peaks,
)
from candidate_rescoring import (  # noqa: E402
    _aa_match_prefix_suffix,
    _tokenize_cached,
    batch_predict_candidates,
    load_spectrum_by_offset,
)

# Token vocabulary — the single source of truth lives in <project_root>/token_table.py.
_TOKEN_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_TOKEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOKEN_ROOT))

from token_table import MAIN_TOKEN_IDS  # noqa: E402
from mgf_row_policy import (  # noqa: E402
    MgfRowPolicy,
    build_mgf_row_catalog,
    build_occurrence_aligned_pairs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HS-DF rescoring: d(spectrum cosine) + f(fragment coverage), score-avg fused."
    )
    p.add_argument("--parquet", type=Path, required=True, help="PN 候选池 parquet")
    p.add_argument("--mgf", type=Path, required=True, help="MGF 文件 (含 SEQ=GT 和实验峰)")
    p.add_argument("--mgf-dir", type=Path, default=None,
                   help="MGF 目录 (仅 occurrence_aware 对齐模式使用, 对齐 V20 的 mmazei 处理)")
    p.add_argument("--row-alignment-mode", type=str, default="row_order",
                   choices=["row_order", "occurrence_aware"],
                   help="parquet 与 MGF 的对齐方式: "
                        "'row_order' = 行序对齐 (默认, 优先); "
                        "'occurrence_aware' = 按 scan 值 + 出现次数匹配 (用于 scan 号重复的 "
                        "数据集如 mmazei, 需指定 --mgf-dir)。")
    p.add_argument("--checkpoint", type=Path, required=True, help="V20 checkpoint")
    p.add_argument("--output-dir", type=Path, default=Path("./hsdf_output"))
    p.add_argument("--rel-spec", type=float, default=0.20,
                   help="d 谱峰余弦的 pred 相对强度过滤档位 (如 0.20 对应 ds 的 d020)")
    p.add_argument("--rel-frag", type=float, default=0.20,
                   help="f 碎片覆盖率的 pred 相对强度过滤档位 (如 0.20 对应 f020)")
    p.add_argument("--tol", type=float, default=0.05, help="m/z 匹配容差 (Da)")
    p.add_argument("--min-mz", type=float, default=100.0)
    p.add_argument("--max-scans", type=int, default=None, help="调试: 只处理前 N 张谱")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--precursor-mismatch-mode", type=str, default="off",
                   help="off / fix / drop")
    p.add_argument("--precursor-mismatch-ppm", type=float, default=50.0)
    p.add_argument("--precursor-hard-threshold", type=float, default=10.0,
                   help="前体质量硬过滤阈值 (默认 10 ppm)。候选理论质量与 (fix 后) PEPMASS 偏差超过该值则从融合中排除, 对齐 DS 的 ±10ppm 候选筛选。0 = 不过滤。")
    p.add_argument("--hard-ladder", type=str, default=None,
                   help="前体质量硬过滤的阶梯阈值列表 (逗号分隔, 如 '5,20')。"
                        "逐级放宽: 先用 5ppm 过滤, 若全部超标则用 20ppm, 若仍全部超标则全量兜底。"
                        "默认 None = 用 --precursor-hard-threshold 单值 (当前行为)。")
    p.add_argument("--pn-weight", type=float, default=0.0,
                   help="PN 分融合权重 (0~1)。final = (1-w)*(d+f score-avg) + w*minmax(PN分)。"
                        "0 = 纯 d/f (默认); 0.5 = d/f 与 PN 分各半 (借用 PN 分, 对齐 V20 的 PN 融合)。")
    p.add_argument("--mass-threshold-unit", type=str, default="ppm",
                   choices=["ppm", "da"], help="质量过滤单位 (默认 ppm)")
    p.add_argument("--save-diagnostics", action="store_true",
                   help="额外保存逐谱 CSV (best_seq, d/f 分数, gt_rank)")
    p.add_argument("--save-full-results", action="store_true",
                   help="额外保存每候选一行的全量结果 CSV (d/f/mby/pn/融合分/质量/命中), "
                        "供事后分析所有候选的打分")
    p.add_argument("--save-per-spectrum", action="store_true",
                   help="额外保存每谱一行的结果 CSV: GT(SEQ) / PN top1(序列+分+命中) / "
                        "HSDF top1(序列+d/f分) / gt_rank / hit。每谱一条, 对应全部质谱。")
    p.add_argument("--skip-results-save", action="store_true",
                   help="跳过所有结果 CSV (diagnostics/full/per_spectrum), 只保留 "
                        "summary.json (对齐 V20 重打分的 --skip-results-save)。")
    p.add_argument("--fusion-norm", type=str, default="minmax",
                   choices=["minmax", "zscore", "raw"],
                   help="d/f/mby/pn 融合时的归一化方式: "
                        "'minmax' = 各分量谱内 minmax 归一化到 [0,1] 后等权融合 (默认); "
                        "'zscore' = 各分量谱内 z-score (x-mean)/std, 对离群更鲁棒; "
                        "'raw' = 直接用原始分数 (保留绝对尺度, 分数可跨谱比, 但 pn 可能主导)。")
    p.add_argument("--use-mby", action="store_true",
                   help="启用主 b/y 离子碎片精度 (m_by_prec) 并等权融入 d/f: "
                        "fused = (minmax(d)+minmax(f)+minmax(mby))/3 (对齐 DS 的 "
                        "hs_d020_f020_mby)。frag_precision = 预测的主 b/y 峰在主峰 token "
                        "实验谱命中的比例, precision 方向。不加 = 不启用 (d/f 等权)。")
    p.add_argument("--exp-int-transform", type=str, default="off",
                   choices=["off", "sqrtclip"],
                   help="实验峰强度变换 (用于 d/f 打分): "
                        "'off' = 用原始 MGF 强度 (默认); "
                        "'sqrtclip' = sqrt(clip(强度,0,50000))/sqrt(50000), "
                        "和模型训练的强度空间对齐, 压缩动态范围使弱峰也有贡献。")
    p.add_argument("--rel-mby", type=float, default=0.20,
                   help="m_by_prec 的主 b/y 峰提取 pred 相对强度档位 (默认 0.20)。"
                        "V2 PN 候选的主峰可能更弱, 可试 0.05/0.10 保留更多主峰。")
    return p.parse_args()


def _score_d_f(pred, input_mz, valid_mask, exp_peaks, rel_spec, rel_frag, min_mz, tol):
    """返回 (d, f)。d 用 rel_spec 的 theo, f 用 rel_frag 的 theo (可同档位共用)。"""
    pmax = float(pred.max())
    theo_spec = prediction_to_mgf_peaks(
        input_mz, pred, valid_mask,
        min_intensity=max(0.01, rel_spec * pmax), min_mz=min_mz, mz_merge_tol=tol,
    )
    d = peak_cosine(exp_peaks, theo_spec, tol=tol)
    if abs(rel_spec - rel_frag) < 1e-9:
        f = explained_intensity(exp_peaks, theo_spec, tol=tol)
    else:
        theo_frag = prediction_to_mgf_peaks(
            input_mz, pred, valid_mask,
            min_intensity=max(0.01, rel_frag * pmax), min_mz=min_mz, mz_merge_tol=tol,
        )
        f = explained_intensity(exp_peaks, theo_frag, tol=tol)
    return d, f


_MAIN_TOKEN_ID = MAIN_TOKEN_IDS[0]  # token index of the main (no-offset) peak


def _main_by_peaks(input_mz, pred, valid_mask, min_intensity, min_mz, mz_merge_tol):
    """提取主 b/y 离子峰 (main token only), 丢弃 satellite/isotope/neutral-loss。"""
    im = input_mz[:, _MAIN_TOKEN_ID, :]
    pr = pred[:, _MAIN_TOKEN_ID, :]
    vm = valid_mask[:, _MAIN_TOKEN_ID, :]
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
        merged_mz.append(mz_vals[best])
        merged_int.append(int_vals[best])
        i = j
    return np.column_stack([np.asarray(merged_mz, dtype=np.float32),
                            np.asarray(merged_int, dtype=np.float32)])


def _aligned_spectra(args, df):
    """根据 row-alignment-mode 生成 (row_idx, meta, peaks) 序列。"""
    if args.row_alignment_mode == "occurrence_aware":
        if not args.mgf_dir:
            raise ValueError("occurrence_aware 模式需要 --mgf-dir")
        catalog, _ = build_mgf_row_catalog(Path(args.mgf_dir), MgfRowPolicy())
        pairs, _ = build_occurrence_aligned_pairs(df["scan_id"].tolist(), catalog)
        for row_idx, entry in pairs:
            meta, peaks = load_spectrum_by_offset(entry.mgf_path, entry.byte_offset)
            yield row_idx, meta, peaks
    else:
        for i, (meta, peaks) in enumerate(iter_mgf_spectra(args.mgf, max_scans=args.max_scans)):
            yield i, meta, peaks


def main() -> int:
    args = parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_config, stats, _ = load_checkpoint(args.checkpoint, device)
    in_ch = int(model_config.get("in_channels", 6))
    print(f"  in_channels={in_ch}, d_model={model_config.get('d_model', '?')}, "
          f"seq_context={model_config.get('use_sequence_context', True)}")

    cols = ["scan_id", "candidates"] + (
        ["scores"] if (args.pn_weight > 0 or args.save_per_spectrum or args.save_full_results) else [])
    df = pd.read_parquet(args.parquet, columns=cols)
    print(f"Loading parquet: {args.parquet}")
    print(f"  rows={len(df)}, columns={list(df.columns)}")

    print(f"Loading MGF spectra from: {args.mgf}")
    print(f"  rel_spec={args.rel_spec}, rel_frag={args.rel_frag}, tol={args.tol}, "
          f"hard_threshold={args.precursor_hard_threshold}{args.mass_threshold_unit}")
    print(f"  use_mby={int(bool(args.use_mby))}"
          + (f"   rel_mby={args.rel_mby}" if args.use_mby else "   (mby 分支未启用)"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_total = n_top1 = 0
    n_gt_inpool = 0
    n_aa_matched = n_aa_truth = n_aa_pred = 0
    n_fix = 0
    t0 = time.time()
    rows = []
    full_rows = []
    per_spec_rows = []
    _total = min(len(df), args.max_scans) if args.max_scans else len(df)
    for row_idx, meta, peaks in tqdm(
            _aligned_spectra(args, df), total=_total,
            desc="rescoring scans", dynamic_ncols=True):
        gt = meta.get("SEQ", "")
        ch = parse_charge_from_meta(meta)
        pm = parse_pepmass_from_meta(meta)
        if not gt or ch <= 0 or pm <= 0:
            continue
        cands = [str(c) for c in df["candidates"].iloc[row_idx]]

        # 可选 fix: 用 GT 理论质量覆盖 PEPMASS (ricebean 57Da 错位)
        if args.precursor_mismatch_mode == "fix":
            theo_mass = calculate_pepmass(gt, ch)
            if theo_mass > 0 and abs(theo_mass - pm) / pm * 1e6 > args.precursor_mismatch_ppm:
                pm = theo_mass
                n_fix += 1

        pred_grids, input_grids, valid_masks = batch_predict_candidates(
            model=model, model_config=model_config, candidates=cands,
            precursor_charge=ch, pepmass=pm,
            rt=float(meta.get("RTINSECONDS", "1800") or 1800.0),
            stats=stats, device=device, min_mz=args.min_mz,
        )

        # 实验峰强度变换 (对齐模型训练的 sqrt(clip(.,50000))/sqrt(50000) 空间)
        exp_for_scoring = peaks
        if args.exp_int_transform == "sqrtclip":
            exp_for_scoring = np.array(peaks, dtype=np.float32)
            exp_for_scoring[:, 1] = (np.sqrt(np.clip(exp_for_scoring[:, 1], 0, 50000))
                                     / np.sqrt(50000))

        d_arr = np.zeros(len(cands), dtype=np.float64)
        f_arr = np.zeros(len(cands), dtype=np.float64)
        mby_arr = np.zeros(len(cands), dtype=np.float64)
        for j, (pg, ig, vm) in enumerate(zip(pred_grids, input_grids, valid_masks)):
            d_arr[j], f_arr[j] = _score_d_f(
                pg, ig, vm, exp_for_scoring, args.rel_spec, args.rel_frag,
                args.min_mz, args.tol)
            if args.use_mby:
                _pmax = float(pg.max())
                by = _main_by_peaks(ig, pg, vm,
                                    max(0.01, args.rel_mby * _pmax),
                                    args.min_mz, args.tol)
                mby_arr[j] = frag_precision(exp_for_scoring, by, tol=args.tol)

        def _mm(a):
            lo, hi = float(a.min()), float(a.max())
            return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

        def _zs(a):
            mu, sd = float(a.mean()), float(a.std())
            return (a - mu) / sd if sd > 1e-9 else np.zeros_like(a)

        if args.fusion_norm == "minmax":
            _n = _mm
        elif args.fusion_norm == "zscore":
            _n = _zs
        else:
            _n = lambda a: a.astype(np.float64)  # raw

        # 等权 score-avg 融合: d/f/mby (对齐 DS 的 hs_d020_f020_mby)。
        comps = [_n(d_arr), _n(f_arr)]
        if args.use_mby:
            comps.append(_n(mby_arr))
        fused = np.mean(comps, axis=0)
        # PN 分 (pn_weight>0 时融合用; save_per_spectrum 时保存 PN top1 参考用)。
        if args.pn_weight > 0 or args.save_per_spectrum or args.save_full_results:
            pn = np.array([float(s) for s in df["scores"].iloc[row_idx]], dtype=np.float64)
        else:
            pn = np.zeros(len(cands), dtype=np.float64)
        if args.pn_weight > 0:
            fused = (1.0 - args.pn_weight) * fused + args.pn_weight * _n(pn)

        # 前体质量硬过滤 (阶梯型): 逐级放宽阈值, 找到第一个有候选通过的档。
        #   e.g. hard_ladder="5,20": 先用 5ppm 过滤, 若全部超标则用 20ppm, 若仍全部超标则全量兜底。
        #   单值 hard_threshold = 当前行为 (5ppm 或全量 fallback)。
        has_hf = args.precursor_hard_threshold > 0 or bool(args.hard_ladder)
        if has_hf or args.save_full_results:
            devs = np.array([
                abs(calculate_pepmass(c, ch) - pm) / pm * 1e6
                if args.mass_threshold_unit == "ppm"
                else abs(calculate_pepmass(c, ch) - pm) * ch
                for c in cands
            ], dtype=np.float64)
        if has_hf:
            if args.hard_ladder:
                th_list = [float(x) for x in args.hard_ladder.split(",") if x.strip()]
            else:
                th_list = [args.precursor_hard_threshold]
            for th in th_list:
                hard_pass = devs <= th
                if hard_pass.any():
                    fused[~hard_pass] = -np.inf
                    break
            # 全部档都超标 -> 全量兜底 (不过滤)

        best_idx = int(np.argmax(fused))
        best = cands[best_idx]
        hit = bool(peptide_matches(best, gt))

        n_total += 1
        if any(peptide_matches(c, gt) for c in cands):
            n_gt_inpool += 1
        if hit:
            n_top1 += 1

        # AA micro (对齐 V20 的 AA recall/precision 口径)
        _bt = _tokenize_cached(best)
        _tt = _tokenize_cached(gt)
        if _bt and _tt:
            _matches, _ = _aa_match_prefix_suffix(list(_bt), list(_tt))
            n_aa_matched += int(sum(_matches))
            n_aa_truth += len(_tt)
            n_aa_pred += len(_bt)

        if (args.save_diagnostics or args.save_per_spectrum) and not args.skip_results_save:
            gt_rank = 0
            for k, cand in enumerate(cands):
                if peptide_matches(cand, gt):
                    gt_rank = k + 1
                    break
            if args.save_diagnostics and not args.skip_results_save:
                rows.append({
                    "ordinal": row_idx, "scan_id": meta.get("TITLE", ""), "charge": ch,
                    "pepmass": round(pm, 4), "gt_seq": gt, "best_seq": best,
                    "d_score": round(float(d_arr[best_idx]), 6),
                    "f_score": round(float(f_arr[best_idx]), 6),
                    "gt_rank": gt_rank, "hit": hit,
                })
            if args.save_per_spectrum and not args.skip_results_save:
                # raw_fused: 不归一化的融合分 (用原始 d/f/mby/pn), 分数可跨谱比较, 低置信谱自然低
                comps_raw = [float(d_arr[best_idx]), float(f_arr[best_idx])]
                if args.use_mby:
                    comps_raw.append(float(mby_arr[best_idx]))
                raw_fused = sum(comps_raw) / len(comps_raw)
                if args.pn_weight > 0:
                    raw_fused = (1.0 - args.pn_weight) * raw_fused + args.pn_weight * float(pn[best_idx])
                per_spec_rows.append({
                    "ordinal": row_idx, "scan_id": meta.get("TITLE", ""), "charge": ch,
                    "pepmass": round(pm, 4),
                    "gt_seq": gt,
                    "pn_top1_seq": cands[0],
                    "pn_top1_score": round(float(pn[0]), 6),
                    "pn_hit": bool(peptide_matches(cands[0], gt)),
                    "hsdf_top1_seq": best,
                    "d_score": round(float(d_arr[best_idx]), 6),
                    "f_score": round(float(f_arr[best_idx]), 6),
                    "best_pn_score": round(float(pn[best_idx]), 6),
                    "raw_fused": round(raw_fused, 6),
                    "gt_rank": gt_rank, "hit": hit,
                })
        if args.save_full_results and not args.skip_results_save:
            # 含案例分析所需全部字段 (pn_rank/v20_rank/mass_ppm/mass_da), 供 save_predictions_cases.py 直接消费
            _theo = [calculate_pepmass(c, ch) for c in cands]
            _vr = np.zeros(len(cands), dtype=int)
            for _rk, _j in enumerate(np.argsort(-fused, kind="stable"), start=1):
                _vr[_j] = _rk
            _pr = np.zeros(len(cands), dtype=int)
            for _rk, _j in enumerate(np.argsort(-pn, kind="stable"), start=1):
                _pr[_j] = _rk
            for j, c in enumerate(cands):
                _fv = float(fused[j])
                if not np.isfinite(_fv):
                    _fv = -1e10  # 硬过滤排除 (对齐旧 V20 的 -1e10)
                full_rows.append({
                    "ordinal": row_idx, "scan_id": meta.get("TITLE", ""), "charge": ch,
                    "pepmass": round(float(pm), 4),
                    "cand_seq": c,
                    "pn_rank": int(_pr[j]),
                    "v20_rank": int(_vr[j]),
                    "d_score": round(float(d_arr[j]), 6),
                    "f_score": round(float(f_arr[j]), 6),
                    "mby_score": round(float(mby_arr[j]), 6) if args.use_mby else 0.0,
                    "pn_score": round(float(pn[j]), 6),
                    "fused": round(_fv, 6),
                    "mass_ppm": round(float(abs(_theo[j] - pm) / pm * 1e6), 2),
                    "mass_da": round(float(abs(_theo[j] - pm) * ch), 4),
                    "passes_hard": bool(np.isfinite(float(fused[j]))),
                    "is_best": bool(j == best_idx),
                    "gt_match": bool(peptide_matches(c, gt)),
                })

    top1_rate = n_top1 / n_total if n_total else 0.0
    gt_hit_rate = n_gt_inpool / n_total if n_total else 0.0
    aa_recall = n_aa_matched / n_aa_truth if n_aa_truth else 0.0
    aa_precision = n_aa_matched / n_aa_pred if n_aa_pred else 0.0

    print("\n===== Summary =====")
    print(f"Scans processed:   {n_total}/{_total}")
    print("Missing in MGF:    0")
    print("Alignment mode:    row_order")
    print(f"MGF policy:        keep={n_total}/{_total}, drop_O=0, drop_prec=0, fix_prec={n_fix}")
    print(f"GT hit rate:       {gt_hit_rate:.4f} ({n_gt_inpool}/{n_total})")
    print(f"GT top-1 rate:     {top1_rate:.4f} ({n_top1}/{n_total})")
    print(f"AA recall (micro): {aa_recall:.4f} ({n_aa_matched}/{n_aa_truth})")
    print(f"AA precision (micro): {aa_precision:.4f} ({n_aa_matched}/{n_aa_pred})")

    summary = {
        "n_scans": n_total, "n_top1": n_top1, "top1_rate": round(top1_rate, 6),
        "gt_hit_rate": round(gt_hit_rate, 6),
        "aa_recall_micro": round(aa_recall, 6),
        "aa_precision_micro": round(aa_precision, 6),
        "fix_precursor": n_fix,
        "rel_spec": args.rel_spec, "rel_frag": args.rel_frag, "tol": args.tol,
        "precursor_hard_threshold": args.precursor_hard_threshold,
        "pn_weight": args.pn_weight,
        "precursor_mismatch_mode": args.precursor_mismatch_mode,
        "use_mby": bool(args.use_mby), "rel_mby": args.rel_mby,
        # 记录本次实际用的输入，避免产物与权重/数据对不上（旧版不放这几个键）。
        "checkpoint": str(args.checkpoint),
        "parquet": str(args.parquet),
        "mgf": str(args.mgf),
        "tol_da": args.tol,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to: {args.output_dir / 'summary.json'}")

    if args.save_diagnostics and not args.skip_results_save:
        import csv
        cols = ["ordinal", "scan_id", "charge", "pepmass", "gt_seq", "best_seq",
                "d_score", "f_score", "gt_rank", "hit"]
        with (args.output_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"Diagnostics saved to: {args.output_dir / 'diagnostics.csv'}")

    if args.save_full_results and not args.skip_results_save:
        import csv
        cols = ["ordinal", "scan_id", "charge", "pepmass", "cand_seq",
                "pn_rank", "v20_rank", "d_score", "f_score", "mby_score",
                "pn_score", "fused", "mass_ppm", "mass_da", "passes_hard",
                "is_best", "gt_match"]
        with (args.output_dir / "full_results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(full_rows)
        print(f"Full results saved to: {args.output_dir / 'full_results.csv'} "
              f"({len(full_rows)} rows)")

    if args.save_per_spectrum and not args.skip_results_save:
        import csv
        cols = ["ordinal", "scan_id", "charge", "pepmass", "gt_seq",
                "pn_top1_seq", "pn_top1_score", "pn_hit",
                "hsdf_top1_seq", "d_score", "f_score", "best_pn_score",
                "raw_fused", "gt_rank", "hit"]
        with (args.output_dir / "per_spectrum.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(per_spec_rows)
        print(f"Per-spectrum saved to: {args.output_dir / 'per_spectrum.csv'} "
              f"({len(per_spec_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
