# Case study：肽段＋电荷 → 预测 grid → 模拟质谱

此目录使用本版本的 Seq2Mass 模型和 `token_table.py`。权重由 `--checkpoint` 指定，不下载权重、不重新训练，也不需要原始 MGF、Prosit 或 PI encoder。原目录与 `v20_0918_casestudy` 副本提供同一套代码。

## 1. Windows CMD 直接预测

在已有 PyTorch、NumPy、tqdm 的环境中执行。以下是 **CMD** 语法，`^` 为续行符，行尾不要加空格。允许重复使用同一输出目录，重跑时直接覆盖本命令生成的同名结果文件；若要保留不同参数的结果，请使用不同目录。

```bat
cd /d E:\PythonProject\Seq2Mass
set "CKPT=C:\Users\admin\Desktop\v20_0918_39466_最新数据\npz-grid-cosine\massivekb-test-npz\v200918_massivekb46\best_model.pt"

python v20_0918\transfomer\casestudy\case_study.py predict ^
  --peptide QLEEAEEEAQR --charge 2 ^
  --checkpoint "%CKPT%" ^
  --out-dir E:\biodatasets\denovo\case_study\QLEEAEEEAQR_z2
```

使用副本时，把脚本路径中的 `v20_0918` 换成 `v20_0918_casestudy`，权重仍可使用同一个文件。服务器上将路径改成服务器实际路径；Bash 的续行符是 `\`。

常用参数：

| 参数 | 默认行为 / 用途 |
|---|---|
| `--peptide`、`--charge` | 必填：带可选修饰的肽段、正整数前体电荷 |
| `--checkpoint` | 必填：含模型配置、参数和归一化统计的 `best_model.pt`；不是 PI 的 `model.ckpt` |
| `--out-dir` | 必填：本次结果目录；不存在则创建，已存在则覆盖同名结果文件 |
| `--rt` | 可选，单位秒；未提供时取 checkpoint 的训练 RT 均值，本次权重为 3917.633789 秒。模型使用 RT，因此比较同一 case 时应固定 RT；不是预测得到的保留时间 |
| `--pepmass` | 可选，前体 m/z；默认由序列、修饰和电荷计算，不是中性分子质量 |
| `--min-mz` | 默认 100；构造网格与有效掩码时使用，修改它需要重新预测 |
| `--min-intensity` | 默认 0.01；只输出预测强度 **严格大于** 此值的有效单元；阈值始终处于模型输出空间 |
| `--mz-merge-tol` | 默认 0.05 Da；输出峰的合并窗口 |
| `--intensity-space` | `model`（默认）：保留模型强度；`relative100`：强度平方后将保留峰的基峰归一到 100，供绘图 |
| `--device` | 默认有 CUDA 则用 `cuda:0`，否则 CPU；可指定 `cpu`、`cuda:1` 等 |
| `--threads` | 默认 4；PyTorch CPU 线程数 |

输入范围：20 种标准氨基酸的大写序列，长度 **2–40**；支持 `M+15.995`、`C(+57.021)`、`M[+15.995]` 和开头的 `+42.011` 等带符号数值质量增量。示例：`--peptide "AC+57.021M(+15.995)K" --charge 3`。不会自动给所有 C 加固定修饰；需要时在序列中明确填写。不支持 `UNIMOD:35`、氧化文字标签、带侧翼残基的序列或未知氨基酸；遇到这些输入会报错，不会静默丢弃。

前体电荷可为正整数，但碎片通道只有 1+、2+、3+，且不超过前体电荷。本模型的序列上下文电荷嵌入把大于 7 的前体电荷归到 7 档（连续电荷特征仍使用输入值）；能计算不代表模型在任意电荷或任意修饰上都经过验证。

## 2. 具体处理过程

1. 解析肽段及修饰，计算各断裂位点的 b/y 离子 m/z，并加上 46 种 token 质量偏移。
2. 得到 `39×46×6` 的 m/z 网格。根据实际长度、电荷和最低 m/z 构造有效掩码。
3. 使用 checkpoint 的统计量归一化输入；将序列上下文、前体电荷、前体 m/z 和 RT 一起送入模型。使用权重保存的 packed 配置，严格核对加载参数，输出同形状强度网格。
4. 保留有效且强度超过阈值的单元，展开成峰，按 m/z 升序排列。每个合并窗口以当前最小 m/z 为起点，纳入与起点相差 **小于** 0.05 Da 的峰；取其中最强单元的 m/z 和强度作为代表。强度不相加，也不做连续链式扩窗。
5. 写入一条模拟谱的 MGF、峰表和网格；记录权重哈希及实际参数。

这里采用 `predict_mgf.py` 的直接阈值＋m/z 合并规则。项目原有 `scripts/predict_spectrum.py`、`predict_npz_to_mgf.py` 还使用 `cluster_postprocess` 的场景规则，结果不保证相同；复现旧的 cluster 实验应继续用对应旧入口。本脚本便于明确追溯每个输出峰所对应的预测网格。

## 3. 输出文件与绘图取用

| 文件 | 内容与用途 |
|---|---|
| `grid.npz` | 压缩的 NumPy 数组包；包含 `input_mz`、`prediction`、`valid_mask` 三个 `float32[39,46,6]` 数组，及字符串 `metadata_json`。这是后续重新导出 MGF 的完整输入 |
| `prediction_grid.npy` | 模型原始输出 `float32[39,46,6]`，方便直接画热图；单独这个文件没有 m/z、掩码和肽段信息，不能独立恢复完整 MGF |
| `grid.csv` | 10,764 行，包含全部网格单元（含无效和补零位置），每行一个单元，列含义见下表 |
| `prediction.mgf` | 一条模拟谱，带 `SEQ`、`CHARGE`、`PEPMASS`、`RTINSECONDS` 以及 m/z–强度峰表 |
| `peaks.tsv` | 两列 `mz`、`intensity`，与 MGF 的峰一致；画棒状质谱图优先读这个文件 |
| `metadata.json` | 本次序列、电荷、RT、m/z 下限、checkpoint 路径及 SHA256、模型配置、归一化统计、代码哈希、导出阈值、强度空间和峰数 |

`grid.csv` 列说明：

| 列 | 含义 |
|---|---|
| `cleavage_index` | 从 N 端开始的断裂位置，1–39；只有前 L−1 行对应实际断裂，L 是肽段长度 |
| `ion_ordinal` | 对 b 通道为断裂位置 i，对 y 通道为 L−i；补零行留空 |
| `token_index` | token 列号，**0–45**；主峰（偏移为 0）是第 24 号 |
| `offset_mz` | 此 token 按对应碎片电荷缩放后的 m/z 偏移；负值表示减去质量，正值表示增加质量 |
| `channel` | `B1+`、`B2+`、`B3+`、`Y1+`、`Y2+`、`Y3+`，按此顺序排列 |
| `mz` | 对应网格单元的候选 m/z；低于最低 m/z 或补零位置可能为 0 |
| `predicted_intensity` | 未经导出阈值筛选、峰合并或绘图缩放的模型输出 |
| `valid` | 1：长度、电荷、m/z 均有效；0：不参与导出。不能只凭 `mz>0` 判断有效性 |

**强度口径：**网格永远保存模型原始输出。项目默认训练目标是 `sqrt(min(I,50000))/sqrt(50000)`，因此它不是仪器原始计数；本次 checkpoint 未保存目标变换标签，不能从权重文件本身独立确认该预处理。默认 MGF 也保留模型空间强度。选择 `relative100` 时执行 `100×(s/max(s))²`（s 为保留峰的模型强度），表达的是按平方逆变换约定得到的相对绘图强度，不能恢复被截断的原始强度。绘制真实谱与预测谱对照时，两边需采用一致的强度口径。模型预测也可能略大于 1，本脚本不会擅自截断。

热图可读取 `prediction_grid.npy` 并按通道选取 `grid[:, :, 0]`（B1+）；用 `valid_mask` 遮住无效位置。每个通道是 `39×46`，实际肽段只需展示前 L−1 行。完整网格保留固定形状以对应模型。

## 4. 从已有 grid 重新导出

不加载模型权重，也不重新推理；仍需当前 Python 依赖环境。可以输出到新目录，也可以原目录重新导出，仅覆盖 `prediction.mgf`、`peaks.tsv` 和 `metadata.json`。原有网格文件保留，`grid.npz` 内的元数据记录生成该网格时的参数；最新导出参数以 `metadata.json` 为准。`predict` 重跑则更新全部六个结果文件。其他文件不受影响。

```bat
python v20_0918\transfomer\casestudy\case_study.py export ^
  --grid E:\biodatasets\denovo\case_study\QLEEAEEEAQR_z2\grid.npz ^
  --out-dir E:\biodatasets\denovo\case_study\QLEEAEEEAQR_z2_plot ^
  --min-intensity 0.03 --mz-merge-tol 0.05 --intensity-space relative100
```

`export` 的三个导出参数默认值仍为 `0.01 / 0.05 / model`，**不会自动继承**先前自定义阈值；请明确填写需要的设置。改肽段、电荷、RT 或 `--min-mz` 时，需要运行 `predict`。

## 5. 验证与已有示例

```bat
python -m unittest discover -s v20_0918\transfomer\casestudy -p test_case_study.py -v
```

`examples/QLEEAEEEAQR_z2/` 为实际 0918 权重在 CPU 上生成的示例：默认 RT、默认导出参数，438 个峰。权重 SHA256 为 `2a166978012814cff3808e33d6283fae46ad5d679f2008715b8cd76c89c5efc1`。若要保留此示例，自己的预测请使用其他目录。

如需迁移到服务器，复制整个版本目录及单独的 checkpoint。仅复制 `casestudy` 目录不够：它依赖同版本的 `predict_mgf.py`、`spectrum_transformer/` 和根目录 `token_table.py`；本功能不依赖另一个 0918 副本。
