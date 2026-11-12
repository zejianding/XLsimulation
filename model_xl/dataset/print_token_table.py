#!/usr/bin/env python3
"""打印 V20 的 46 token 表与质量精度，并核对「模型侧」与「数据集生成侧」用的是同一张表。

用法：
    python dataset/print_token_table.py                 # 只打印两侧用的表
    python dataset/print_token_table.py <数据集路径>     # 额外对比 NPZ 里存的 tokens_charge*
        <路径> 可为 单个 .npz / 分片目录 / memmap 目录
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent  # v20_0918/
for _p in (_ROOT, _ROOT / "transfomer", _ROOT / "dataset"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def token_meaning(idx: int, tt) -> str:
    name = tt.TOKEN_NAMES.get(idx)
    if name:
        return name
    v = tt.CHARGE_TOKENS[1][idx]
    for aa, m in tt.AA_MASS.items():
        if abs(m - abs(v)) < 1e-6 and not (aa == "L" and "L" in tt.TOKEN_NAMES.values()):
            return ("+" if v > 0 else "-") + aa
    for aa, m in tt.AA_MASS.items():
        if abs(m - abs(v)) < 1e-6:
            return ("+" if v > 0 else "-") + aa
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None, help="可选：已生成的数据集路径")
    args = ap.parse_args()

    import token_table as tt            # 唯一来源

    rule("① 唯一来源：token_table.py")
    print(f"  文件路径        : {pathlib.Path(tt.__file__).resolve()}")
    print(f"  TOKEN_DECIMALS  : {tt.TOKEN_DECIMALS}          <- token 偏移的小数位数")
    print(f"  NUM_TOKENS      : {tt.NUM_TOKENS}")
    print(f"  CHARGE_TOKENS   : " + ", ".join(f"z={z}: {len(v)} 个" for z, v in tt.CHARGE_TOKENS.items()))
    print(f"  MAIN_TOKEN_IDS       : {tt.MAIN_TOKEN_IDS}")
    print(f"  SATELLITE_TOKEN_IDS  : {tt.SATELLITE_TOKEN_IDS}")
    print(f"  RESIDUE_EVENT 列数   : {len(tt.RESIDUE_EVENT_TOKEN_IDS)}")

    # ------------------------------------------------------------ 模型侧
    rule("② 模型侧：用的就是这张表")
    import predict_mgf as pm
    from spectrum_transformer import config as cfg
    from spectrum_transformer.models.factory import build_model, create_model_config

    print(f"  {'模块':<28s} {'NUM_TOKENS':>10s}  {'CHARGE_TOKENS':>14s}")
    for label, mod in (("predict_mgf", pm),
                       ("spectrum_transformer.config", cfg),
                       ("spectrum_transformer.models.transformer", __import__(
                           "spectrum_transformer.models.transformer", fromlist=["x"]))):
        n = getattr(mod, "NUM_TOKENS", None)
        ct = getattr(mod, "CHARGE_TOKENS", None)
        ct_desc = "同一对象" if ct is tt.CHARGE_TOKENS else ("（该模块不导入，未持有）" if ct is None else "值相同/否")
        print(f"  {label:<28s} {str(n):>10s}  {ct_desc:>14s}")
    print(f"  config.MAIN_TOKEN_IDS={cfg.MAIN_TOKEN_IDS}   "
          f"config.SATELLITE_TOKEN_IDS={cfg.SATELLITE_TOKEN_IDS}")

    model = build_model(create_model_config()).eval()
    print("\n  模型内与 token 轴相关的张量：")
    print(f"    packed_pos_w.weight      : {tuple(model.packed_pos_w.weight.shape)}  "
          f"(第 0 维 = token 列数，应为 {tt.NUM_TOKENS})")
    print(f"    packed_pos_h.weight      : {tuple(model.packed_pos_h.weight.shape)}  (裂解行数，应为 39)")
    print(f"    token_family_index       : {tuple(model.token_family_index.shape)}")
    fi = model.token_family_index.cpu().numpy()
    fam = {0: "residue_event", 1: "satellite", 2: "main"}
    print(f"    main 家族(2)所在列        : {[int(i) for i in np.where(fi == 2)[0]]}")
    print(f"    satellite 家族(1)所在列   : {[int(i) for i in np.where(fi == 1)[0]]}")
    print(f"    pos_encoding.embedding   : {tuple(model.pos_encoding.embedding.shape)}  "
          f"(max_len = 39*46 = {39*46})")

    # --------------------------------------------------- 数据集生成侧
    rule("③ 数据集生成侧：用的也是这张表")
    import trainingExamples as te
    import TokenTableOriginalAnalyze_massivekb_mods_precise_tokens as flt
    for label, mod in (("trainingExamples", te), ("TokenFilter 过滤模块", flt)):
        n = getattr(mod, "NUM_TOKENS", None)
        n_desc = f"{n} (一致={n == tt.NUM_TOKENS})" if n is not None else "未定义（该模块只 import CHARGE_TOKENS）"
        print(f"  {label:22s} NUM_TOKENS={n_desc:<44s} "
              f"CHARGE_TOKENS 是同一对象={getattr(mod,'CHARGE_TOKENS',None) is tt.CHARGE_TOKENS}")
    print(f"  NPZBuilder 会把这些写进 NPZ: tokens_charge1/2/3 = CHARGE_TOKENS[1/2/3]  (各 {tt.NUM_TOKENS} 个)")

    # ------------------------------------------------------ 两侧逐值对比
    rule("④ 两侧逐值对比")
    print(f"  token_table.CHARGE_TOKENS is predict_mgf.CHARGE_TOKENS   : {pm.CHARGE_TOKENS is tt.CHARGE_TOKENS}")
    print(f"  token_table.CHARGE_TOKENS is trainingExamples.CHARGE_TOKENS: {te.CHARGE_TOKENS is tt.CHARGE_TOKENS}")
    print(f"  token_table.AA_MASS       is trainingExamples.AA_MASS      : {te.AA_MASS is tt.AA_MASS}")
    print(f"  token_table.AA_MASS       is predict_mgf.AA_MASS           : {pm.AA_MASS is tt.AA_MASS}")
    ok = all(list(pm.CHARGE_TOKENS[z]) == list(te.CHARGE_TOKENS[z]) for z in (1, 2, 3))
    print(f"  三张电荷表逐值相等                                        : {ok}")

    # ----------------------------------------------------- AA 质量精度
    rule("⑤ 氨基酸质量精度（token 表由它生成）")
    print(f"  {'AA':>3s} {'质量(AA_MASS 原始)':>22s} {'6 位小数':>14s} {'若只保留 1 位小数':>18s}")
    for aa in sorted(tt.AA_MASS, key=lambda a: tt.AA_MASS[a]):
        m = tt.AA_MASS[aa]
        print(f"  {aa:>3s} {m:>22.9f} {round(m,6):>14.6f} {round(m,1):>18.1f}")
    print("\n  token 生成方式：1+ 偏移 = round(残基质量, 6)；z 电荷表 = round(1+偏移 / z, 6)")
    print(f"  TOKEN_DECIMALS = {tt.TOKEN_DECIMALS}  ->  每个 token 都是 6 位小数")

    # ------------------------------------------------------- 完整 46 表
    rule("⑥ 46 个 token 完整表（z=1 / z=2 / z=3）")
    c1, c2, c3 = tt.CHARGE_TOKENS[1], tt.CHARGE_TOKENS[2], tt.CHARGE_TOKENS[3]
    print(f"  {'idx':>3s} {'z=1':>14s} {'z=2':>13s} {'z=3':>13s}  含义")
    for i in range(tt.NUM_TOKENS):
        mark = ""
        if i in tt.MAIN_TOKEN_IDS:
            mark = "   <-- MAIN"
        elif i in tt.SATELLITE_TOKEN_IDS:
            mark = "   <-- satellite"
        print(f"  {i:>3d} {c1[i]:>+14.6f} {c2[i]:>+13.6f} {c3[i]:>+13.6f}  {token_meaning(i, tt):<12s}{mark}")

    # ---------------------------------------------------------- 精度检查
    rule("⑦ 精度检查")
    bad = [(z, i, v) for z in (1, 2, 3) for i, v in enumerate(tt.CHARGE_TOKENS[z])
           if abs(v - round(v, tt.TOKEN_DECIMALS)) > 1e-12]
    print(f"  所有 token 都是 {tt.TOKEN_DECIMALS} 位小数（无多出位数）: {not bad}")
    for z in (1, 2, 3):
        a = tt.CHARGE_TOKENS[z]
        one_dp = sorted({round(v, 1) for v in a})
        diff = sum(1 for v in a if min(abs(v - o) for o in one_dp) > 0.005)
        print(f"  z={z}: 6dp 有 {len(a)} 列，1dp 只有 {len(one_dp)} 列；"
              f"两表不同的 token 数 = {diff}（若用 1 位小数会丢/错这些）")

    # ------------------------------------------------- 与数据集内表的对比
    if args.dataset:
        rule("⑧ 与已生成数据集里存的 token 表对比")
        from spectrum_transformer.data.dataset import load_npz_data
        raw = load_npz_data(args.dataset)
        for z in (1, 2, 3):
            k = f"tokens_charge{z}"
            if k not in raw:
                print(f"  {k}: 数据集中不存在")
                continue
            stored = np.asarray(raw[k], dtype=np.float64)
            expect = np.asarray(tt.CHARGE_TOKENS[z], dtype=np.float64)
            print(f"  {k:16s} shape={stored.shape}  == token_table: {np.allclose(stored, expect, atol=1e-6)}")
        g = np.asarray(raw["inputs"])
        print(f"  inputs: {g.shape}   "
              f"{'[OK] 是 39x46x6' if g.shape[1:] == (39, 46, 6) else '[FAIL] 不是 39x46x6'}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
