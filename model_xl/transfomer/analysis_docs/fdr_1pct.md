# 工作二：1% FDR（target-decoy 竞争，统一 (D+1)/T）

## 目标
证明鉴定量不是靠库大碰运气：每谱候选池掺入反转诱饵（decoy），所有谱按分数降序，用 **(decoy+1)/target** 卡 1% 线，报告线上 target 数/decoy 数（不比对 GT）。四方法统一同一公式。

## 库 / 候选（target+decoy）
```
V20 侧 (fdr/):
  step1_build_td_fasta.py   in-scope 变体 target + 每 target 反转 decoy（碰撞剔除）→ data/<sp>/<sp>_td.fasta
  step2_match_td.py         ±10ppm 逐谱匹配 → data/<sp>/candidates_<sp>_td.tsv.gz（T/D）
引擎侧 (fdr_full/):
  build_td_fasta.py         plain target+decoy（引擎搜索时用 mods 文件展开可变修饰）
  MS-GF+ 搜 TD → MzidToTsv → engine_fdr（rev_ 前缀）
```

## 三（四）方法打分与计数
```
V20 1% FDR：fdr/step3_score_fdr.py（d/f_exp/mby；raw df/dfmby；rel 0.05/0.10/0.20 一次预测多档）
            → results_step3 / full_*.csv.gz
            step4_fdr_curve.py  → summary_fdr_<sp>.json（1%/5%/10% n_target/n_decoy）
MS-GF+  ：run_msgfplus.sh + engine_fdr.py  → n_target/n_decoy@1%
MSFragger：FragPipe 手动搜（decoy_prefix=rev_）→ engine_fdr --engine msfragger
Prosit  ：prosit_fdr_score.py（见本目录副本 / fdr/prosit_fdr_score.py）
            读 candidates_<sp>_td.tsv.gz；候选≤30；reanchor+sqrt；
            每谱每分支取 argmax 候选记 T/D + fused → results_prositfdr_<sp>.csv
           step4_fdr_curve.py --results …  → summary_fdr_<sp>.json（含 Prosit 分支）
```

## 口径/注意
- 公式统一 **(D+1)/T**（fdr_common 与 engine_fdr 均已改为）。
- Prosit-FDR **天然 ≤30**（>30 候选剔除）；**MS-GF+/V20/MSFragger 的 FDR 目前是 ≤32 全库口径**，要和 Prosit 比需额外做 ≤30（后续过滤）。
- decoy 反转保留质量；回文/反库相撞的 decoy 被剔除，保证 decoy 永不等同 target。

## 产物
- 每物种：results/曲线 csv + summary_fdr_<sp>.json；汇总 `fdr_full/output/summary_msgfplus_1pct.tsv`（MS-GF+ 已完成）
- 待：V20 df/dfmby 三档 9 物种、Prosit FDR、MSFragger FDR 收尾

## 状态
- MS-GF+ 1% FDR：完成（9 物种）
- V20 df/dfmby 三档：服务器跑批中（含并发 output 目录踩踏教训：勿多进程同物种重叠）
- Prosit FDR：脚本就绪（analysis_docs/prosit_fdr_score.py 副本），待跑
- MSFragger FDR：待 FragPipe 重搜 + engine_fdr
