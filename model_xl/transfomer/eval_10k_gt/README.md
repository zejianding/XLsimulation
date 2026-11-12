# eval_10k_gt · 1 万肽「预测谱 vs 原谱」余弦对比（V20 / Prosit）

对 9 物种的 9999 条肽（7–30aa，含修饰）各预测一张谱，与该肽在原始数据里的**全部**质谱算余弦，
逐肽取**平均**；所有产物都是 **CSV（无 JSON）**，且**全程流式、内存有界**。

## 流程（与 `transfomer/run_v20_eval.sh` 顺序一致）
```
[0] make_peptide_npz.py        9物种 MGF 的 SEQ=/CHARGE= -> 输入 NPZ（V20 输入网格）
[1] V20: predict_npz_to_mgf.py       Prosit: prosit_predict_mgf.py（Koina，先批量预测再去重写 MGF）
[2] collect_original_with_prediction.py   单遍扫全部原谱，逐块「追加」写每组文件（不驻留内存）
[3] mgf_to_embedding_batch.py             PiPrime 编码（逐组，需 GPU/cupy）
[4] batch_cosine_summary.py               每肽汇总 CSV（原有）
[5] cosine_detail_csv.py                  逐谱细节 + 每肽汇总（本次新增，CSV）
```

## 内存安全设计（为什么不爆）
- **[2] 收集**：目标肽集合（9999）常驻；每张原谱**逐块读入→命中即 append 写盘**，不缓存原谱块；只有计数与映射在内存。
- **[3] 嵌入**：逐组（每组=1 肽的全部原谱）编码，模型只加载一次。
- **[5] 细节/汇总**：逐组读该组的 MGF + NPZ（每组很小），算完即写 CSV 行、即释放；不累积嵌入矩阵。
- **Prosit 预测**：先去重 (骨架,电荷) 批量调 Koina（缓存落少量分片），再逐肽写 MGF。
- 需要控制峰值内存时可用 `--limit N`（只处理前 N 组，调试用）。

## 产物（每次运行 `runs/<name>_<stamp>/`，全部 CSV/文本）
| 文件 | 内容 |
|---|---|
| `predicted.mgf` | 预测谱（V20 或 Prosit），格式同 run_v20_eval |
| `Grouped_MGFs/<seq>_charge<z>_<hash>.mgf` | 该肽全部原谱（`SOURCE=ORIGINAL`，`COMMENT=species:<sp>`）+ 1 条预测谱（`SOURCE=MODEL_PREDICTION`） |
| `Grouped_MGFs/group_map.csv` | 肽 ↔ 组文件 ↔ 原谱数 ↔ 各物种计数 ↔ 无匹配清单 |
| `Embeddings/<stem>.npz` | PiPrime 嵌入（与 MGF 谱序一致） |
| `cosine_summary.csv` | **每肽一行**：`num_original, num_prediction, pred_vs_orig_mean/std/min/max, orig_orig_mean` |
| `cosine_detail.csv` | **每张原谱一行**：`group_file, sequence, charge, idx_in_group, source_tag, species, spectrum_id, cosine_to_prediction` |

`pred_vs_orig_mean` 就是"预测谱与该肽所有原谱余弦的平均"；细节表保留每张原谱的单独相似度。

## 服务器命令
```bash
# 前置：cupy（PrimeNovo 需要）、koinapy（Prosit 需要）、PiPrime ckpt、9species 原谱目录
cd ~/seq2mass/v20_0701/transfomer/eval_10k_gt
TEST_NPZ=~/v20_10k_9species.npz ROOT_DIR=~/datasets/9species/Casanovo_preprocessed_data \
  bash run_v20_10k.sh /home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt 0 v20_10k
bash run_prosit_10k.sh prosit_10k          # P4_CACHE=<tsv.gz> 可复用 Koina 缓存
```
读结果：两个 `runs/*/cosine_summary.csv`（逐肽）+ `cosine_detail.csv`（逐谱）；按 `sequence` 即可对齐 V20↔Prosit。

## 本地已验证
`[1]`（smoke 权重 3 肽）与 `[2]`（1,526,282 张原谱中匹配 40 张，3 组）通过；
`[5]` 用假嵌入验证（3 组 / 40 行细节 + 逐肽汇总）；`[3]` 需服务器 cuPy。
