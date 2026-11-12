#!/usr/bin/env python3
"""从肽段序列 + 前体电荷预测理论谱（峰列表）。

复用已训练的 V20 checkpoint 权重：
  序列 → in-silico 裂解 m/z 网格 (39,46,6) → Transformer 前向 → 预测强度网格
       → 按 m/z 合并容差展平成 (m/z, intensity) 峰列表。

用法:
    python predict_theoretical_spectrum.py \
        --sequence QLEEAEEEAQR --charge 2 \
        --checkpoint runs/v200701_evidence_v2_massivekb/best_model.pt \
        --mz-merge-tol 0.05 \
        --output out.tsv \
        --save-grid pred_grid.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 脚本与 predict_mgf.py 同级，保证能 import spectrum_transformer / predict_mgf
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_mgf import (  # noqa: E402
    NUM_CHANNELS,
    build_input_mz_matrix,
    build_valid_mask,
    calculate_pepmass,
    load_checkpoint,
    predict_spectrum,
    prediction_to_mgf_peaks,
    prepare_model_input,
    write_mgf,
)
from spectrum_transformer.config import DEFAULT_MIN_VALID_MZ  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从序列 + 电荷预测理论谱（峰列表）。"
    )
    p.add_argument("--sequence", required=True,
                   help="肽段序列，支持修饰写法如 M+15.995 / M(+15.99)")
    p.add_argument("--charge", type=int, default=2, help="前体电荷 (默认 2)")
    p.add_argument("--pepmass", type=float, default=None,
                   help="前体 m/z；默认用序列理论质量 calculate_pepmass(seq, charge)")
    p.add_argument("--rt", type=float, default=1800.0, help="保留时间 RTINSECONDS (默认 1800)")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="V20 checkpoint 权重路径")
    p.add_argument("--mz-merge-tol", type=float, default=0.05,
                   help="m/z 合并容差 (Da, 默认 0.05)")
    p.add_argument("--min-intensity", type=float, default=0.01,
                   help="最小预测强度阈值 (默认 0.01)")
    p.add_argument("--rel-intensity", type=float, default=0.0,
                   help="相对基峰强度阈值 (0~1)，过滤毛噪声峰；0=不过滤。如 0.05 保留 >=5% 基峰的峰")
    p.add_argument("--min-mz", type=float, default=DEFAULT_MIN_VALID_MZ,
                   help="最小输出 m/z (默认 100)")
    p.add_argument("--output", type=Path, default=None,
                   help="峰列表输出文件；.tsv/.txt=两列 m/z intensity，.mgf=MGF 格式")
    p.add_argument("--save-grid", type=Path, default=None,
                   help="额外保存预测网格 prediction (39,46,6) 为 .npy")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--no-packed", action="store_true")
    return p.parse_args()


def _write_peaks_tsv(path: Path, peaks: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("m/z\tintensity\n")
        for mz, intens in peaks:
            f.write(f"{float(mz):.5f}\t{float(intens):.6f}\n")


def main() -> int:
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    model, model_config, stats, _ = load_checkpoint(
        args.checkpoint, device, no_packed=args.no_packed
    )

    pepmass = (
        args.pepmass
        if args.pepmass is not None
        else calculate_pepmass(args.sequence, args.charge)
    )

    # 1) in-silico 裂解 -> 输入 m/z 网格 + valid mask
    input_mz = build_input_mz_matrix(args.sequence, min_mz=args.min_mz)
    valid_mask, cleavage_count = build_valid_mask(
        args.sequence, args.charge, min_mz=args.min_mz
    )
    model_input = prepare_model_input(
        input_mz,
        valid_mask,
        stats,
        expected_in_channels=int(model_config.get("in_channels", NUM_CHANNELS)),
    )

    # 2) Transformer 前向 -> 预测强度网格 prediction (39, 46, 6)
    raw_aux = np.array([[float(args.charge), pepmass, args.rt]], dtype=np.float32)
    aux_norm = (
        (raw_aux - stats["aux_feature_means"])
        / np.maximum(stats["aux_feature_stds"], 1e-6)
    ).astype(np.float32)[0]

    prediction = predict_spectrum(
        model, model_input, valid_mask, aux_norm,
        args.sequence, args.charge, cleavage_count, device,
    )

    # 3) 网格 -> 峰列表（按 m/z 合并容差）
    eff_min_intensity = args.min_intensity
    if args.rel_intensity > 0:
        eff_min_intensity = max(
            args.min_intensity,
            args.rel_intensity * float(prediction.max()),
        )
    peaks = prediction_to_mgf_peaks(
        input_mz,
        prediction,
        valid_mask,
        min_intensity=eff_min_intensity,
        min_mz=args.min_mz,
        mz_merge_tol=args.mz_merge_tol,
    )

    # 4) 输出
    print(f"序列: {args.sequence}  电荷: {args.charge}+  前体 m/z: {pepmass:.5f}")
    print(f"预测网格 shape: {prediction.shape}  峰数: {len(peaks)}  "
          f"(m/z 合并容差 {args.mz_merge_tol} Da, min_intensity {args.min_intensity})")
    print(f"{'m/z':>12}  {'intensity':>10}")
    for mz, intens in peaks:
        print(f"{float(mz):>12.5f}  {float(intens):>10.6f}")

    if args.save_grid is not None:
        np.save(args.save_grid, prediction)
        print(f"\n预测网格已保存: {args.save_grid}  shape={prediction.shape}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".mgf":
            write_mgf(
                args.output, [peaks], args.sequence, args.charge, pepmass,
                "MODEL_PREDICTION", [args.rt],
            )
        else:
            _write_peaks_tsv(args.output, peaks)
        print(f"峰列表已写入: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
