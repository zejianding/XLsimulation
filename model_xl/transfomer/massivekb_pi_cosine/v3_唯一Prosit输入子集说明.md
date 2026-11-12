# massiveTest v3：30,000 个唯一 Prosit 输入的完整子集

## 目标与 v2 的区别

v2 按原始带修饰 `SEQ＋charge` 抽样，不同修饰形式可能去修饰后映射到同一个 `(backbone, charge)`，且存在超过预测长度限制的组合。v2 已完成的相似度结果继续保留。

v3 按用户确认的规则生成独立实验队列：

1. **去修饰序列＋电荷必须有 30,000 个唯一键**；同一骨架的不同电荷允许分别入选。
2. 对同一个键存在多个原始修饰组合时，只随机选一个原始组合作为代表，不把不同修饰组的真实谱合并。
3. 本次严格要求原始组合至少有 3 条真实谱，保留该组合在全源中的所有成员。
4. 使用与 Prosit 预测相同的 `parse_modified_sequence` 解析骨架。筛选骨架长度 2–30、标准氨基酸字符及正整数电荷。
5. 在线生成候选的 Prosit 谱，仅接受成功导出非空预测谱的组合；失败候选记录原因，从后续随机候选补齐，直到真实组合、唯一 Prosit 输入、成功预测谱三个数量都恰好为 30,000。

该规则条件化于 Prosit 预测成功，不能把 v3 与 v2 的分数差解释为同样本上的算法提升。预测谱成功生成不保证其 PI 嵌入一定有效，后续仍需检查编码和相似度的有效组数。

## 抽样实现

- 入口：`sample_massivekb_test.py`。
- 新辅助模块：`prosit_sampling.py`。
- 复用预测出口：`../eval_10k_gt/prosit_predict_mgf.py`。
- 校验与打包：`package_verified_sample.py`。
- `run_pipeline.py` 读取新元数据时额外核验 backbone＋charge 唯一性和预测覆盖。

候选原始组先按键排序；每个 Prosit 输入在符合谱数条件的修饰形式中均匀随机选一个代表，再打乱唯一键候选顺序。种子为 `20261012`。过滤后候选数量不足或者最终成功预测数不足时，不提交一个不足 30,000 组的正式样本。

处理顺序：全源计数 → 筛选并去重候选 → 分批在线验证/补选 → 全源再次扫描提取全部真实谱 → 核对计数与身份 → 提交样本和预测谱 → 独立校验及 ZIP 打包。

`--unique-prosit-input` 仅保证候选键唯一及本地准入规则；`--verify-prosit` 隐含该参数，并要求实际预测导出成功。本次使用后者。验证模式要求 `min_spectra=prefer_spectra`，此次两者均为 3。

## 本地生成命令

本次生成与打包已经完成，以下命令用于复现。重新生成时使用新的输出目录，不要覆盖本次样本。

```powershell
$repo = 'E:/PythonProject/Seq2Mass'
$out = 'E:/biodatasets/denovo/massivekb/massiveTest_v3_prosit_unique30k'
$env:PATH = 'E:/Git/mingw64/bin;' + $env:PATH
$env:XZ_DEFAULTS = '--memlimit-mt-decompress=2GiB'
$env:PYTHONIOENCODING = 'utf-8'
# 本机本次为预测安装的临时依赖目录；其他环境使用已配置的 koinapy 即可。
$env:PYTHONPATH = 'C:/Users/admin/AppData/Local/Temp/seq2mass_koina_sampling_deps'

python -u "$repo/v20_0918/transfomer/massivekb_pi_cosine/sample_massivekb_test.py" `
  --mgf-xz E:/biodatasets/denovo/massivekb/massivekb_82c0124b_train.mgf.xz `
  --out-dir "$out" --n-groups 30000 `
  --min-spectra 3 --prefer-spectra 3 --seed 20261012 `
  --threads 8 --verify-prosit --progress-every 1000000

# 仅在上一条成功结束后执行；ZIP 已存在时拒绝覆盖。
python "$repo/v20_0918/transfomer/massivekb_pi_cosine/package_verified_sample.py" `
  --sample-dir "$out"
```

本地为本次运行在临时目录安装了 koinapy/Triton 等依赖，没有替换原 Python 环境的 NumPy、PyTorch 等包。预测出口预先导入可用的 koinapy，避免项目旧 NumPy 兼容别名影响新版客户端的原生扩展导入。

## 交付内容

| 文件 | 内容 |
|---|---|
| `massTest.mgf` | 30,000 个入选原始组合的全部真实谱 |
| `massTest_groups.tsv` | 原始组合计数清单 |
| `massTest_meta.json` | 全源信息、抽样参数、唯一输入池、预测验证统计 |
| `prosit_predicted.mgf` | 与入选组严格一一对应的 30,000 条已验证预测谱 |
| `prosit_input_map.tsv` | 原始 sequence、去修饰 backbone、charge、真实谱数 |
| `sampling_prosit_rejected.csv` | 定样前被排除的预测失败候选；不属于最终入选组 |
| `validation_report.json` | 独立校验结果、数量、代码哈希和文件 SHA256 |
| `README_server.md` | 校验与服务器续跑命令 |
| `SHA256SUMS.txt` | LF 换行的校验清单 |

`prediction_batches/` 和其下 Koina 缓存保留在本地作为生成过程记录，不放入上传包。打包脚本会检查 ZIP CRC，并在 ZIP 旁生成 `.zip.sha256`。

## 上传后的服务器分析

同步本次更新后的项目源码，尤其新增的 `prosit_sampling.py` 和更新的 `run_pipeline.py`。解压到 `/data/wangyuhang2/datasets/kb30000/` 后执行：

```bash
conda activate s2m
REPO=/home/wangyuhang/seq2mass
SAMPLE=/data/wangyuhang2/datasets/kb30000/massiveTest_v3_prosit_unique30k
OUT=/data/massiveTest_v3_unique30k_analysis
MODEL="$REPO/piprime_embedding/piprime_embedding/model.ckpt"

(cd "$SAMPLE" && sha256sum -c SHA256SUMS.txt)
mkdir -p "$OUT"
set -o pipefail
python -u "$REPO/v20_0918/transfomer/massivekb_pi_cosine/run_pipeline.py" \
  --sample-dir "$SAMPLE" --out-dir "$OUT" --model "$MODEL" \
  --prosit-mgf "$SAMPLE/prosit_predicted.mgf" \
  --n-groups 30000 --min-spectra 3 --prefer-spectra 3 --seed 20261012 \
  --domains none,sqrt_clip --workers 8 --gpu 0 --stop-after full \
  2>&1 | tee "$OUT/full_$(date +%Y%m%d_%H%M%S).log"
```

校验失败时先处理错误，不继续分析。`--prosit-mgf` 导入本次验证成功的预测，避免再次请求服务导致覆盖发生变化。不要沿用 v2 的 `--min-spectra 2` 或输出目录。其余过滤、平均谱、PI 编码和比较口径与 [全流程操作说明](全流程操作说明.md) 一致。

## 本次生成状态

2026-10-12：正式样本、在线预测、独立校验和上传包均已完成。17 项回归测试通过，包括修饰去重、不同电荷保留和失败候选补选；在线两组合小样及正式样本均通过完整流程验证。

| 项目 | 最终值 |
|---|---:|
| 全源谱数 | 28,508,636 |
| 全源原始组合数 | 2,029,550 |
| 符合条件的唯一 backbone＋charge 候选 | 806,543 |
| 入选原始组合数 | 30,000 |
| 入选唯一 backbone＋charge 数 | 30,000 |
| 去修饰序列本身的种类数 | 29,608（允许同骨架不同电荷） |
| 入选真实谱数 | 679,895 |
| 每组谱数范围 | 3–100 |
| 每组平均谱数 | 22.6631667 |
| 在线验证候选数 | 30,000 |
| 成功导出的 Prosit 谱数 | 30,000 |
| 预测失败/补选数 | 0 |
| 抽样种子 | 20261012 |

独立检查通过：真实谱 ID 唯一且按来源顺序，MGF/组清单/元数据计数一致，去修饰请求键没有碰撞，预测侧每个入选组恰好一条谱，原始组全部成员已提取。校验清单采用 LF 换行，ZIP CRC 检查通过。

本地目录：`E:/biodatasets/denovo/massivekb/massiveTest_v3_prosit_unique30k/`。

上传包：`E:/biodatasets/denovo/massivekb/massiveTest_v3_prosit_unique30k.zip`，共 1,540,411,785 字节，包含前述 9 个文件。

ZIP SHA-256：

```text
3c668db67d40f2eafc193987149024412d1b785ff4f9593f4dba7582d72433be
```

本次只完成 v3 数据集及其预测谱生成，尚未执行 v3 的过滤、PI 编码和相似度分析。v2 的既有结果保持原样。
