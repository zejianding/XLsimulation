#!/usr/bin/env python3
"""打印一个已生成数据集的内容，用于人工核对。

支持三种形态：单个 .npz / 分片目录（含 index.json）/ memmap 目录（含 meta.json）。

用法：
    python dataset/inspect_npz.py <path> [--samples 5] [--mz-check 60] [--no-mz-check]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent  # v20_0918/
for _p in (_ROOT, _ROOT / "transfomer", _ROOT / "dataset"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

# 终端编码不一致（Windows GBK / Linux UTF-8）时避免 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import token_table as tt  # noqa: E402
from spectrum_transformer.data.dataset import load_npz_data  # noqa: E402


def _source_kind(path: pathlib.Path) -> str:
    if path.is_dir():
        if (path / "meta.json").exists():
            return "memmap 目录"
        if (path / "index.json").exists():
            return "分片目录"
        return "目录（未识别）"
    return "单个 .npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--samples", type=int, default=5, help="打印前 N 个样本")
    ap.add_argument("--mz-check", type=int, default=60, help="抽样校验 m/z 精度的样本数")
    ap.add_argument("--no-mz-check", action="store_true")
    args = ap.parse_args()

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"路径不存在: {path}")
        return 1

    print("=" * 84)
    print(f"数据集检查: {path}")
    print(f"来源类型  : {_source_kind(path)}")
    print("=" * 84)

    raw = load_npz_data(str(path))
    inputs = np.asarray(raw["inputs"])
    targets = np.asarray(raw["targets"])
    N = inputs.shape[0]

    # ---------------------------------------------------------------- 形状
    print("\n【网格形状】")
    print(f"  inputs            : {inputs.shape}  {inputs.dtype}")
    print(f"  targets           : {targets.shape}  {targets.dtype}")
    shape_ok = inputs.shape[1:] == (tt.MAX_CLEAVAGE_LEN, tt.NUM_TOKENS, tt.NUM_CHANNELS)
    print(f"  期望              : (N, {tt.MAX_CLEAVAGE_LEN}, {tt.NUM_TOKENS}, {tt.NUM_CHANNELS})")
    print(f"  裂解轴 = 39       : {'OK' if inputs.shape[1] == 39 else 'FAIL'}")
    print(f"  偏移轴 = 46       : {'OK' if inputs.shape[2] == tt.NUM_TOKENS else 'FAIL'}  "
          f"(实测 {inputs.shape[2]}, 若为 44 说明还是旧表)")
    print(f"  通道轴 = 6        : {'OK' if inputs.shape[3] == 6 else 'FAIL'}")
    print(f"  总体              : {'[OK] 是 39x46x6' if shape_ok else '[FAIL] 不是 39x46x6'}")

    # ------------------------------------------------------------ token 表
    print("\n【token 表】（存储在 NPZ 内，应与 token_table 一致）")
    all_tok_ok = True
    for z in (1, 2, 3):
        key = f"tokens_charge{z}"
        if key not in raw:
            print(f"  {key:16s}: 缺失")
            all_tok_ok = False
            continue
        stored = np.asarray(raw[key], dtype=np.float64)
        expect = np.asarray(tt.CHARGE_TOKENS[z], dtype=np.float64)
        ok = stored.shape == (46,) and np.allclose(stored, expect)
        all_tok_ok &= bool(ok)
        print(f"  {key:16s}: shape={stored.shape}  与 token_table 一致={ok}")
    c1 = tt.CHARGE_TOKENS[1]
    print(f"  charge-1 前 5 项  : {[f'{v:+.6f}' for v in c1[:5]]}")
    print(f"  charge-1 Q/K 两列 : idx37={c1[37]:+.6f} (+Q)   idx38={c1[38]:+.6f} (+K)")
    print(f"  主列 idx={tt.MAIN_TOKEN_IDS[0]}        : offset={c1[tt.MAIN_TOKEN_IDS[0]]:+.6f}")
    print(f"  全部 6 位小数     : {'OK' if all_tok_ok else 'FAIL'}")

    # ---------------------------------------------------------------- 样本
    n_show = max(0, min(args.samples, N))
    print(f"\n【前 {n_show} 个样本】")
    seqs = raw.get("seqs")
    charges = np.asarray(raw.get("charges", np.zeros(N)))
    lengths = np.asarray(raw.get("lengths", np.zeros(N)))
    pepmasses = np.asarray(raw.get("pepmasses", np.zeros(N)))
    for i in range(n_show):
        s = str(seqs[i]) if seqs is not None else "?"
        print(f"  [{i}] seq={s:<28s} z={int(charges[i])} len={int(lengths[i]):>2d} "
              f"pepmass={float(pepmasses[i]):>9.4f}  非零目标格={int((targets[i] > 0).sum())}")

    # ---------------------------------------------------------------- 统计
    nz = int((targets > 0).sum())
    per = targets.reshape(N, -1)
    print("\n【统计】")
    print(f"  样本数 N          : {N}")
    print(f"  目标密度(非零占比) : {nz / targets.size:.4f}")
    print(f"  每样本非零格       : 均值 {per.astype(bool).sum(axis=1).mean():.1f}  "
          f"中位 {np.median(per.astype(bool).sum(axis=1)):.0f}  "
          f"最大 {per.astype(bool).sum(axis=1).max()}")
    print(f"  目标最大值         : {float(targets.max()):.4f}   (sqrt_clip 归一化后应 <= 1)")
    keys = [k for k in ("per_example_keys", "sequence_constrained", "tol", "target_normalization")]
    meta = {}
    for mf in ("index.json", "meta.json"):
        p = path / mf if path.is_dir() else None
        if p and p.exists():
            meta.update(json.loads(p.read_text(encoding="utf-8")))
    if meta:
        print("  元数据            : " + "  ".join(
            f"{k}={meta[k]}" for k in keys if k in meta) or "  (无)")
    if seqs is not None:
        combos = list(zip([str(s) for s in seqs], [int(c) for c in charges]))
        u = len(set(combos))
        print(f"  唯一 (seq,charge)  : {u}  {'= N，无重复 OK' if u == N else f'< N，有 {N-u} 条重复'}")
        us = len(set(str(s) for s in seqs))
        print(f"  唯一 seq           : {us}   (同一序列多条谱 = 不同电荷)")

    # ---------------------------------------------------------- m/z 精度
    if not args.no_mz_check and seqs is not None:
        import trainingExamples as te
        print(f"\n【m/z 精度：独立重算（抽样 {min(args.mz_check, N)} 个样本）】")
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=min(args.mz_check, N), replace=False)
        per_z = {z: [0, 0, 0.0] for z in (1, 2, 3)}   # [命中, 总数, 最大偏差]
        for i in idx:
            for ion in te.generate_by_ions_with_mods(str(seqs[i]), max_charge=3):
                r = ion["cleavage_index"] - 1
                if r >= 39:
                    continue
                z = ion["charge"]
                ch = (z - 1) if ion["ion_type"] == "B" else (z + 2)
                for t, tok in enumerate(tt.CHARGE_TOKENS[z]):
                    stored = float(inputs[i, r, t, ch])
                    err = abs((stored - float(ion["mz"])) - tok)
                    per_z[z][1] += 1
                    if err < 1e-6 * max(1.0, abs(stored)):
                        per_z[z][0] += 1
                    per_z[z][2] = max(per_z[z][2], err)
        for z in (1, 2, 3):
            hit, tot, mx = per_z[z]
            if tot == 0:
                continue
            print(f"  z={z}: {hit}/{tot} 命中该电荷的 6 位小数表  最大|偏差|={mx:.2e}")
        print("  说明：偏差量级 = float32 在相应 m/z 处的舍入（约 1e-4 @ m/z 2000），非表误差。")

    print("\n" + "=" * 84)
    print("结论: " + ("[OK] 是 39x46x6 数据集" if shape_ok else "[FAIL] 形状不是 39x46x6"))
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
