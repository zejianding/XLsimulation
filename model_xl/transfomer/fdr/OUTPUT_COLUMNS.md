# FDR 流水线输出文件与列属性说明

本文档逐文件说明 FDR 分析（`transfomer/fdr/`）保留的每个 CSV / JSON 文件里各列（字段）的含义，以及它们之间怎么互相串起来。

所有文件都放在 `output/<species>/` 下（物种占位为 `<sp>`）。

## 文件清单

| 文件 | 由哪步生成 | 粒度 | 作用 |
|---|---|---|---|
| `<sp>_td.fasta` | step1 | 每条序列一条记录 | target+decoy 数据库（模型打分的候选来源） |
| `candidates_<sp>_td.tsv.gz` | step2 | 每候选一行 | 每条谱按前体质量匹配到的候选，带 T/D 标签 |
| `candidates_detail_<sp>.csv.gz` | step3（`--save-detail`） | 每候选一行 | 所有候选的 5 个分量分（合并一个文件） |
| `full_<分支>_<sp>.csv.gz` | step3（`SAVE_FULL=1`，默认开） | 每候选一行 | **全量结果**：每个分支一个文件，含谱信息、序列、全部分量分、该分支融合分/排名 |
| `results_step3_<sp>.csv` | step3 | 每谱一行 | 每个分支的 top1（序列/标签/分数/是否命中真值），宽表 |
| `fdr_curve_<sp>_<分支>.csv` | step4 | 每谱一行 | 该分支按分数降序 + 累计 FDR / q 值 |
| `summary_step3_<sp>.json` | step3 | — | step3 的聚合统计 |
| `summary_fdr_<sp>.json` | step4 | — | 各分支 1%/5%/10% 线的鉴定数 |
| `summary_fdr_all_species.json` | run_fdr.sh 汇总 | — | 全物种 1% 线对比 |
| `records/record_stepN_<sp>.json` | 各步 | — | 运行记录（参数、耗时、自检） |

**串联关系**：`ordinal`（谱在 MGF 中的顺序号，0 起）把不同文件里的"同一张谱"连起来；`seq`（候选注释序列）+ `label`（T/D）把"同一个候选"连起来；`full_*.csv.gz` 与 `results_step3` 之间通过 `ordinal` 关联，且 full 文件里 `rank=1`（`is_best=1`）的那行就是 `results_step3` 里该分支的 top1。

---

## 1. `full_<分支>_<sp>.csv.gz` —— 逐候选全量结果（核心文件）

每个分支（如 `d`、`gc`、`f_exp`、`d_gc_f_exp`……）一个文件；每张谱的每个候选一行。8 个默认分支的文件里 **gc/d/f_bidir/f_exp/mby 这 5 列是相同**的（重复存储），只有 `fused/rank/is_best` 随分支不同。

| 列 | 含义 |
|---|---|
| `ordinal` | 谱序号（MGF 文件顺序，0 起），全文件唯一的谱 ID |
| `scan_id` | 原始谱编号（MGF 的 `SCANS`，可能重复；跨文件定位谱要用 `ordinal`） |
| `charge` | 前体电荷数 |
| `pepmass` | 前体质荷比 m/z（MGF `PEPMASS`） |
| `rt` | 保留时间（秒，MGF `RTINSECONDS`，缺省 1800） |
| `gt_seq` | 该谱的真值肽段（MGF `SEQ=`），可为空 |
| `cand_idx` | 该候选在本谱候选池内的序号（0 起） |
| `seq` | 候选肽段**注释序列**（如 `C+57.021`、`M+15.995`），即喂给 V20 模型的序列 |
| `label` | `T` = target（来自 FASTA 的真实肽段）；`D` = decoy（反转的 target，保证不存在） |
| `ppm` | 候选中性质量相对前体中性的偏差（ppm），来自 step2 质量匹配 |
| `neutral_mass` | 候选的中性质量（Da） |
| `gc` | **网格余弦**：预测的 (39,44,6) 强度网格 vs 实验峰投到网格后的余弦，[0,1]，衡量整张谱形状 |
| `d` | **谱余弦（真实谱→预测峰）**：实验峰强度取到预测峰位置上与预测强度算余弦，惩罚"预测了但实验没有"，[0,1] |
| `f_bidir` | **谱余弦（预测峰→真实谱）**：预测峰强度取到实验峰位置上与实验强度算余弦，惩罚"实验有但预测没覆盖"，[0,1] |
| `f_exp` | **explained intensity**：实验总强度中被预测峰覆盖的比例，recall 方向，[0,1] |
| `mby` | **主 b/y 离子精度**：预测的主 b/y 峰（token 24，主列）在实验谱中被命中的比例，[0,1] |
| `fused` | **该分支的融合分** = 分支所选分量的等权平均。单分量分支（如 `gc`）= 该分量本身 |
| `rank` | 该候选在本谱内按 `fused` 降序的排名（1 = 最优/top1） |
| `is_best` | 1 = 本谱该分支的 top1（`rank==1`） |
| `gt_match` | 1 = 该候选与 `gt_seq` 质量匹配（I/L 等价、前缀后缀 AA 质量对齐，见 `ds_common.peptide_matches`） |

> 用途示例：查 `label=T` vs `label=D` 的 `gc` 分数分布；查每条谱 GT 候选的 `rank` 分布；用 `seq`+`gt_match` 找"模型选错/选对"的候选做案例分析。任何新的分量组合都可由本文件秒算（`step4_fdr_curve.py --full ... --branches ...`）。

---

## 2. `candidates_<sp>_td.tsv.gz` —— step2 质量匹配结果

| 列 | 含义 |
|---|---|
| `ordinal` | 谱序号 |
| `scan_id` | 谱编号 |
| `charge` | 前体电荷 |
| `pepmass` | 前体质荷比 |
| `rt` | 保留时间 |
| `seq` | 候选注释序列 |
| `label` | T / D |
| `neutral_mass` | 候选中性质量 |
| `ppm` | 候选质量相对前体的 ppm 偏差 |

此文件是 step3 的输入；`mean_candidates`、`gt_in_targets` 等质量指标记录在 `records/record_step2_<sp>.json`。

---

## 3. `results_step3_<sp>.csv` —— 每谱 top1 宽表（step3 输出）

| 列 | 含义 |
|---|---|
| `ordinal` / `scan_id` / `charge` / `pepmass` / `rt` / `gt_seq` | 谱信息（同全量文件） |
| `n_candidates` | 本谱候选总数 |
| `n_target` | 其中 target 数 |
| `n_decoy` | 其中 decoy 数 |
| 对每个分支 `B`（如 `d_gc_f_exp`）： | |
| `B_seq` | 该分支的 top1 序列 |
| `B_label` | 该分支 top1 的 T/D 标签 |
| `B_fused` | 该分支 top1 的**原始融合分**（跨谱可比，FDR 排序用它） |
| `B_gt_match` | 该分支 top1 是否匹配真值（1/0） |

---

## 4. `fdr_curve_<sp>_<分支>.csv` —— 单分支 FDR 曲线（step4 输出）

每行 = 一张谱，已按该分支 `B_fused` **降序**排列。除谱信息 + 该分支的 top1 列（`B_seq`/`B_label`/`B_fused`/`B_gt_match`，普通模式下也会带上其他分支的列，属冗余无碍）外，追加：

| 列 | 含义 |
|---|---|
| `cum_target` | 到本行为止累计的 target top1 数 |
| `cum_decoy` | 到本行为止累计的 decoy top1 数 |
| `fdr_dt` | 到本行为止的 FDR = `cum_decoy / cum_target`（Elias–Gygi 惯例） |
| `fdr_tdt` | 另一种口径 = `cum_decoy / (cum_target + cum_decoy)` |
| `q_dt` | **q 值** = 分数不高于本行时能达到的最小 FDR（后缀最小值）。**1% FDR 线 = `q_dt ≤ 0.01` 的最大前缀** |

> 1% 线鉴定集导出：`awk -F, 'NR==1 || $q_dt列<=0.01' fdr_curve_<sp>_<分支>.csv`

---

## 5. `summary_fdr_<sp>.json` —— FDR 汇总（step4）

```json
{
  "species": "mouse",
  "fdr_targets": [0.01, 0.05, 0.10],
  "branches": {
    "d_gc_f_exp": {
      "rank_col": "d_gc_f_exp_fused",
      "n_spectra": 10000,
      "n_top1_target": 9934,
      "n_top1_decoy": 66,
      "fdr_lines": {
        "0.01": {
          "fdr_target": 0.01,
          "n_accepted_spectra": 10000,
          "n_target": 9934,          // 1% 线下的 target 鉴定数（对比主角）
          "n_decoy": 66,             // 线内 decoy（应≈n_target×1%）
          "n_target_gt_match": 9828, // 其中真正匹配真值的数（最诚实指标）
          "fdr_dt_achieved": 0.00664,// 线内实际 FDR
          "score_threshold": 0.113,  // 该线的分数阈值
          "top_score": 0.945
        }
      },
      "fdr_curve_csv": "output/mouse/fdr_curve_mouse_d_gc_f_exp.csv"
    }
  }
}
```

---

## 6. 其它产物简述

- **`summary_step3_<sp>.json`**：step3 聚合。`branches.<分支>` 下有 `top1_target_rate`（top1 是 target 的比例）、`top1_decoy`、`top1_gt_match`、`top1_gt_rate`；还有 `results_csv`、`full_csvs`（各分支 full 文件路径）、`rel`/`exp_transform` 等打分参数。
- **`<sp>_td.fasta`**：step1 输出。每记录两行：
  ```
  >TD_TGT_<i> MASS=<中性质量:.6f> LABEL=T SEQ=<注释序列>
  <注释序列>
  ```
  `TGT`/`DEC` 区分 target/decoy。step2 据此建质量索引。
- **`candidates_detail_<sp>.csv.gz`**（`--save-detail`）：`ordinal  label  seq  ppm  gc  d  f_bidir  f_exp  mby`，一个文件含所有候选分量，不带 fused/rank。
- **`summary_fdr_all_species.json`**：`run_fdr.sh` 末尾汇总。每物种列出 `d_gc_f_exp`/`d_f_exp`/`d_gc_f_bidir`/`gc` 四个分支的 `n_target`/`n_gt`（1% 线）以及该物种的 `best_branch` 和对应计数。

---

## 几个关键概念的提醒

- **target vs decoy**：decoy = target 的反转（质量完全相同）。`label` 是 FDR 统计的唯一依据——top1 是 decoy 就意味着这条谱"找错了"（必然假）。
- **为什么 FDR 排序用 `fused`（原始分）而不是谱内 minmax**：原始分跨谱可比，才能把不同谱按可信度统一排队；谱内归一化会破坏可比性。
- **`n_target` vs `n_target_gt_match`**：`n_target` 是"1% 线上的 target 鉴定数"（可能含错误 target）；`n_target_gt_match` 才保证匹配上了真值，对外报告/对比建议用后者。
