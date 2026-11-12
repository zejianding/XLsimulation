# 工作三：GT 全库盲搜（fullsearch，无前体过滤最严对照）

## 目标
最严对照：库 = 该批谱的**真值(GT)肽段子集**（去重 ~4–6k）+ 修饰变体 + 反转诱饵（共 ~1.8 万/电荷）；每张实验谱**不做前体质量过滤**，与**同电荷全库**逐条比对取 top1 vs GT，并统计 decoy_win（top1 被诱饵抢走比例）。

## 流程（fullsearch/，0821）
```
建库：
  step1_gt_library.py       MGF→去重 GT backbone → base_plain.fasta + scans_meta.json
  step2_build_library.py    GT+修饰变体(target)+反转 decoy → library_annotated.tsv.gz

库谱预测（三路）：
  V20   : step3_predict_v20.py    → lib_v20_c{2,3,4}.npz（逐变体×电荷, GPU）
  Prosit: step3_predict_prosit.py → lib_prosit_c{2,3,4}.npz（骨架≤30×电荷；Koina；y 索引+sqrt）
  MS-GF+: run_msgfplus_local.sh（本地参考，小库 plain TD）

盲搜打分：
  step4_blind_score.py      每谱 vs 同电荷全库；d/f/m_by + hsdf/hsdf_mby + dfexp/dfexp_mby；
                             全库 argmax=top1 → results_blind_<pred>_<sp>.csv
                             + summary_blind json（top1_acc/decoy_win；rel 0/0.05/0.10/0.20）
  step5_msgfplus_acc.py     MS-GF+ 小库 top1 vs GT
  诊断：prosit_blind_diag.py（单谱定位）
编排：run_fullsearch.sh（v20/prosit/msgfplus 可选）；run_prosit_gt_all.sh（prosit 全物种）
```

## 公平对齐（关键）
- Prosit 天然 ≤30；**V20 要加 `--max-gt-len 30 --max-lib-len 30`**（GT>30 谱不进分母、>30aa 条目不参与 top1），否则 V20 能赢在 Prosit 根本比不了的 >30 条目上。
- 分支/rel/强度处理对齐同 no-FDR；Prosit 另需 sqrt + rel000 族。

## 途中发现/修复（表示问题）
- Prosit y 离子编号从 C 端数 → 修正 `ion_lookup`（半数碎片曾放错位）；
- Prosit 预测强度偏斜 → sqrt 对齐；rel 过滤过狠 → 保留全部 b/y（rel=0）；
- 结论：Prosit 位置(presence)强（f ~0.75 no-filter）、强度加权(d)弱——真实差异。

## 产物
- `output/<sp>/summary_blind_{v20,prosit}_<sp>.json`（top1_acc / decoy_win，各分支×rel）
- MS-GF+ 参考：`output/<sp>/summary_msgfplus_<sp>.json`

## 状态
- MS-GF+ 参考：完成（9 物种）
- Prosit 盲搜：bacillus/mouse/human 已出，其余串行跑（快，~0.5–1h/物种）
- V20 盲搜 ≤30：需逐物种（GPU 预测分钟级 + CPU 打分数小时/物种），范围待定
