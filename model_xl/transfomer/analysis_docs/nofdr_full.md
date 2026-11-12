# 工作一：无 FDR 全量 search accuracy（候选池内排名，≤30 公平口径）

## 目标
对每个物种约 1 万张实验谱，比较四个方法在"候选池内排名"下的 top1/top3/top10 命中率：我们的 **V20**、公开 **Prosit**、搜索引擎 **MS-GF+ / MSFragger**。由于 Prosit 只收 ≤30aa，所有方法都对齐到 ≤30 口径。

## 输入 / 上游公共（ds step1–3）
```
step1_scan_mgf_mods.py         扫 MGF 修饰 → configs/mods_config.json (C+57 / M+15.995≤3)
step2_build_candidate_fasta.py 枚举修饰变体 → data/<sp>/variants_<sp>.fasta
step3_match_by_mass.py         前体 ±10ppm 逐谱匹配 → data/<sp>/candidates_<sp>_scans.tsv.gz
```
产物 = **无 decoy** 的每谱候选池（DB 全库 in-scope 变体，质量窗内几十条）。

## 打分三路（分支同一套）
```
V20    : step4_predict_score.py   minimal; rel 0.05/0.10/0.20; sqrt_clip
         分支：d_spec/f_frag/m_by_prec + hsdf(=hs_d f,minmax) + hsdf_mby(+m_by)
               + dfexp/dfexp_mby(raw 均值)   → results_step4_<sp>.csv + summary_step4_<sp>.json
Prosit : step4_prosit_score.py    reanchor + sqrt 强度对齐 + rel0族(不过滤)
         分支同上 + rel000 族                   → results_prosit_<sp>.csv + summary_prosit_<sp>.json
引擎   : MS-GF+ / MSFragger 各自搜索 → parse 脚本 → results csv（≤32）
```
- 实验谱强度统一 sqrt_clip；Prosit 是唯一需要"特殊处理"的（√ 预测强度、rel=0 不过滤、y 离子 C 端编号）。
- Prosit 天然 ≤30（>30 候选剔除、GT>30 谱跳过）；**V20/引擎需事后把 GT>30 过滤掉**。

## ≤30 对齐与汇总
```
results_len30_accuracy.py   过滤 results_step4_*.csv / results_prosit_*.csv 到 GT≤30（分母=GT≤30 且在候选）
engine_top1_len30.py        引擎 PSM → ≤30 top1
→ len30/summary_len30_engines.tsv / summary_len30_v20.tsv → summary_len30_merged_top1.tsv（5 方法）
```
分母核对：各方法 n≤30 应一致（mouse 9984、human 9936 …）。

## 产物与落盘
- 每谱 csv + summary json（每分支 top1/3/10、hits）
- ≤30 过滤后 json / 汇总 tsv

## 状态
- 引擎 ≤30：完成（本地 18/18）
- V20 新跑（含 dfexp/dfmby）九物种：结果已跑，≤30 汇总待补
- Prosit no-FDR（候选池 47.8万/物种）：逐物种续跑中（mouse 已出），跑完需做 GT≤30 过滤
- 主表拼装：待 prosit 补齐
