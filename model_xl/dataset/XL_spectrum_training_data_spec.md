# XL Spectrum 数据构建与 model-xl 实现方案

本方案更新于 2026-11-12。model-xl 的训练和导出采用六个碎片离子通道 `[b1+, b2+, b3+, y1+, y2+, y3+]`，目标数据接口为六通道展开 m/z 格式（`schema_version=2`）。前体电荷 `Z` 仍按鉴定表记录，用于模型输入、路由和损失分组。

现有 `xl_spectrum_training_data` 是从原始谱构建的六通道 v2 完整产物。当前构建器的 `verify` 支持校验 v1、v2 和 v3。双肽 Dataset、联合 Transformer、PPT 第 5 页的损失及训练入口已实现，并通过真实小批次前向、反向和 checkpoint 重载验证。全量训练尚待启动，XL PP4 导出尚待实现。

训练样本的单位是一个交联肽对的平均谱组。每张实验谱先按该肽对的理论格点筛峰，同组谱图再生成一张平均谱，最后将平均谱重新匹配到 A/B 双链网格生成标签。XL 构建入口是 [build_xl_spectrum_dataset.py](build_xl_spectrum_dataset.py)，双链网格使用 [V20 token 表](../token_table.py)。

## 参考实现与本轮边界

原始模型以 `E:/hard_cityu/XLsearch/model/model/v20_0918` 为准，结构与损失参考 [framwork.pptx 第 3–5 页](../framwork.pptx)。XL 入口使用独立的双肽 Dataset、模型和损失模块。

| 部分 | 原版依据 | 本方案处理 |
|---|---|---|
| 理论输入持久化 | [trainingExamples.py](../../model/model/v20_0918/dataset/trainingExamples.py) 保存展开的单肽 `(39,46,6)` 原始 m/z | XL 改为直接保存 `(2,39,46,6)`；保留 XL 现有的 `float64` 精度 |
| 标准化与有效性 | [data/dataset.py](../../model/model/v20_0918/transfomer/spectrum_transformer/data/dataset.py) | 保留训练集统计、运行时 m/z 下限、无效输入与目标置零的阶段划分 |
| 网络主体 | [models/transformer.py](../../model/model/v20_0918/transfomer/spectrum_transformer/models/transformer.py) | 保留原版模块和运算，增加双肽轴、交联特征与肽标识；两肽联合注意力 |
| 训练目标 | [training/losses.py](../../model/model/v20_0918/transfomer/spectrum_transformer/training/losses.py) | 以原版实际执行公式为基准，进行维度适配；电荷权重等语义问题单列讨论 |
| 输出谱 | [cluster_postprocess.py](../../model/model/v20_0918/transfomer/cluster_postprocess.py)、[predict_npz_to_mgf.py](../../model/model/v20_0918/transfomer/predict_npz_to_mgf.py) | 保留原版 PP4 流程及 max 合峰规则，明确双链适配边界 |

六通道方案保留离线展开理论 m/z 的接口。A/B 分组顺序、原始 split、跨条件平均、`full` 标签约束、`sqrt_clip(50000)`、RT 输入、两侧 residue 求平均、三电荷分支、主峰局部极大值规则和 PP4 沿用下文约定。XL 共享主峰 BCE 和幅度损失对 Z2、Z3、Z≥4 使用相同的样本权重 1.0。

## 输入与划分

当前输入目录为 `D:/XL/prosit-xl`。鉴定表提供序列、交联位点、交联剂及前体电荷。实验谱表提供前体 m/z、保留时间及用于重匹配的质心峰表。

| 分区 | 鉴定表 | 实验谱表 |
|---|---|---|
| Train | `nms2_train.parquet` | `train_centroid.parquet` |
| Validation | `nms2_val.parquet` | `val_centroid.parquet` |
| Test | `nms2_holdout.parquet` | `test_centroid.parquet` |

每个分区的实验谱表有 `N` 行，对应鉴定表的 `2N` 行。以 0 基行号计，实验谱第 `i` 行对应鉴定表的 `i` 和 `i+N` 两条 A/B 镜像。构建前核对互换后的序列及位点，以及 RAW 名、扫描号、交联剂和电荷。成员谱标识为 `(dataset, raw_file, scan_number)`。此批数据的交联剂是 BS3，成桥净质量取 `138.06808` Da。

按两条链的裸序列、普通修饰位置及质量增量、交联位点对链排序，所有链相关字段随之重排。组键为 `(sequence_A, link_site_A, sequence_B, link_site_B, crosslinker, precursor_charge)`。序列按统一的数字修饰格式表示。镜像注释归入同组，每组按谱图标识去重。

沿用六个输入文件的原始分区，全部 CSM 参与各自分区的分组和平均。同组成员可跨碰撞能量生成一张平均谱。构建报告记录分区间共享实验谱数量。跨分区扫描的处理策略尚待讨论；未决定前，格式迁移保持现有分组成员不变，不静默删除扫描或重新划分。

### 已发布数据与电荷分布

现有六通道 v2 产物的 train/val/test 分别写出 17,474/2,046/4,070 个平均谱组，均有有效正标签。下表按当前 v2 产物统计，同一肽对的不同前体电荷各计一组。随后对共享原始扫描的逐峰核查来自此前的六通道 v1 数据，当前 v2 构建报告的共享扫描计数也是 16/7/1。

| 前体电荷 | train | val | test | train 占比 |
|---|---:|---:|---:|---:|
| 2+ | 18 | 1 | 21 | 0.10% |
| 3+ | 5,428 | 471 | 1,170 | 31.06% |
| 4+ | 8,847 | 1,137 | 2,320 | 50.63% |
| 5+ | 2,929 | 411 | 534 | 16.76% |
| 6+ | 252 | 26 | 25 | 1.44% |
| 合计 | 17,474 | 2,046 | 4,070 | 100% |

训练集 `Z>=4` 共 12,028 组，占 68.83%；验证集 `Z=2` 仅 1 组。三组之间没有相同 `group_id`，也没有忽略电荷后的相同交联肽对（序列、位点、交联剂相同）。带修饰的单链序列仅 train/test 共享 1 条。三组均无“非空但有效正标签为零”的记录。

构建报告中的共享原始扫描数为 train/val=16、train/test=7、val/test=1，按 `(dataset,raw_file,scan_number)` 判断。进一步核对原始 centroid/nms2 后，确认共有 24 个唯一跨分区扫描；全部对应峰表的 m/z 和强度数组逐元素相同，前体 m/z 与 RT 也相同，故不是仅文件名或扫描号碰巧相同。

这 24 对来源关联均具有不同的规范肽对注释，忽略电荷后仍不同；13 对的电荷注释也不同，11 对相同。相应 group_id 都存在于生成的 `.npy` 中。由此可见，按 peptide/肽对分组确实隔离了分子身份，但没有保证训练和评估的观测谱独立；同一观测可能通过不同注释参与不同分区平均谱。筛峰与平均会改变信号贡献，因而不能直接量化它对最终指标的影响。成因是合理的多重鉴定还是注释冲突，仍需额外鉴定证据，不能由本次核查断定。

本次源表核查还确认：没有三方共享标识，24 个唯一扫描在两个分区各出现一条 centroid 记录，共 48 条来源记录。两个具体例子如下，`@数字` 表示 1 基交联位点：

| 来源扫描 | train 注释 | val 注释 | 原始峰表与筛峰核对 |
|---|---|---|---|
| `PXD019926 / HEK293_DSS_FAIMS_405060_Fr1 / scan 26454` | `TSMSKMFVK@5 + VGSTKVPMTSGVK@5`，Z4，组内 4 张谱 | `IKVFGPGIEGK@2 + VKVEPAVDTSR@2`，Z4，组内 10 张谱 | 两边原始 2,134 峰完全相同；按各自肽对候选筛峰后分别 1,060/1,179 峰，其中 769 峰来自相同观测 |
| `PXD016554 / IParfentev_030718_Ecoli_XL_DSS1_F23 / scan 12826` | `AKVALDVFVSR@2 + IHKYIGSYTAR@3`，Z4，组内 3 张谱 | `HLHIPLQSGSNTVLKR@15 + KYTMEFFQER@1`，Z5，组内 5 张谱 | 两边原始 1,213 峰完全相同；筛后分别 831/837 峰，其中 633 峰来自相同观测 |

上述筛峰交集按完全相同的 `(m/z,intensity)` 计算，使用构建器的候选格点 ±0.05 m/z 条件。这些结果尚未经过组内平均、`full` 和 `sqrt_clip`；它们证明肽对筛峰没有完全隔离观测信号，不代表两边最终训练 target 相同。

## 保存字段与坐标

`G = (2,39,46,6)`，轴顺序为 A/B、断裂、token、离子通道。断裂索引 `h = i−1` 对应第 `i` 个残基之后的断裂。六个碎片通道固定为 `[b1+, b2+, b3+, y1+, y2+, y3+]`。表中的形状均为目标六通道格式的单样本形状。

| 保存字段 | 形状 | dtype | 定义 |
|---|---|---|---|
| `group_id` | `()` | UTF-8 字符串 | 由规范化组键确定的组标识，在分区内唯一 |
| `n_spectra` | `()` | `uint32` | 组内去重后的实验谱数，也是平均强度的分母 |
| `sequence` | `(2,)` | UTF-8 字符串 | 按规范 A/B 顺序排列的两条带修饰序列 |
| `lengths` | `(2,)` | `uint8` | 两条链的残基数 `L`，有效断裂数为 `L−1` |
| `link_sites` | `(2,)` | `uint8` | 各链交联位点，按残基 **1 基**计数 |
| `crosslinker` | `()` | UTF-8 字符串 | 交联剂名称 |
| `linker_mass` | `()` | `float64` | 所用交联剂的成桥净质量，单位 Da |
| `precursor_charge` | `()` | `uint8` | 组内共有的前体电荷 `Z` |
| `precursor_mz` | `()` | `float64` | 组内实验谱的前体 m/z 均值 |
| `rt_seconds` | `()` | `float64` | 组内实验谱的保留时间均值，单位秒 |
| `input_theo_mz` | `G` | `float64` | 含 XL 质量、已展开 46 个 token 偏移的原始理论 m/z。长度或电荷无效的位置填 0；候选位置保留原值，包括低于 100 的值 |
| `target` | `G` | `float32` | 平均谱重新匹配并经 `full` 约束及 `sqrt_clip` 归一化后的强度标签 |

每个分区写一个结构化 NumPy 数组文件 `train.npy`、`val.npy` 或 `test.npy`，数组的首维为组数。字符串采用定宽 UTF-8 字节串，三个文件使用同一 dtype，可用 `np.load(path, mmap_mode="r", allow_pickle=False)` 按组读取。

`meta.json` 记录 `target_normalization="sqrt_clip"`、`clip_intensity=50000`、`sequence_constrained="full"`、各阶段 m/z 容差、训练集标准化统计和各分区的构建计数。报告记录正标签中达到裁剪上限的比例。`group_members.parquet` 记录分区、组标识、成员谱标识、centroid 行及两条鉴定行。序列编码、修饰数组、全局特征和 mask 在读取时派生，不保存标准化后的网络输入。

六通道展开格式使用以下最小元数据，同时按实际 dtype 更新 `record_bytes`，并将 `grid_shape` 设为 `[2,39,46,6]`。

```json
{
  "schema_version": 2,
  "input_representation": "expanded_raw_mz",
  "input_shape": [2, 39, 46, 6],
  "input_dtype": "float64",
  "main_token_id": 24,
  "channel_names": ["B1+", "B2+", "B3+", "Y1+", "Y2+", "Y3+"]
}
```

可记录 token 表内容校验值，用于确认离线构建、Dataset、checkpoint 与推理使用相同的词表及顺序。标准化统计、token 定义、数据 schema 和最终采用的训练/导出设置均应随 checkpoint 保存。

### 格式版本、迁移与空间

原版 V20 保存展开的 `float32` m/z；XL 本方案继续使用当前构建器的 `float64`，以保留 `full` 约束 `1e-4` 质量比较的精度。标准化完成后才转为模型需要的 `float32`。

| 项目 | v1：基础 m/z | v2：展开 m/z |
|---|---:|---:|
| 输入形状 | `(2,39,6)` | `(2,39,46,6)` |
| 输入字节数/组 | 3,744 | 172,224 |
| 整条记录字节数/组 | 90,128 | 258,608 |
| 按 23,590 组估算的记录总量 | 2.126 GB | 6.101 GB |

按当前字符串宽度、`align=True` 和组数估算，v2 的 train/val/test 分别约 4.519/0.529/1.053 GB，合计约 5.682 GiB；不含 `.npy` 文件头、meta 和 provenance。输入字段增大 46 倍，整条记录约增大 2.869 倍。并存 v1/v2 需要约 7.662 GiB 的记录空间。若后续修改扫描处理或分组策略，组数和体积需重算。

当前构建器的 `build` 输出 v2，`verify` 兼容 v1/v2/v3，并接受未记录 `channel_names` 的历史 v2。v3 的 `input_theo_mz` 和 `target` 均为 `(2,39,46,8)`。六通道 XL Dataset 直接读取 v2，兼容 v1 时只展开一次。无 `schema_version` 的文件只有在字段、dtype 和 `(2,39,6)` 基础形状严格匹配时才识别为 v1。遇到 v3、未知版本或版本与形状冲突时，应报出具体的格式不匹配。

当前六通道产物由原始数据重建。构建时从筛峰阶段就只使用六通道，再重新平均、匹配并写出 v2。历史 v3 的 4+ 格点参与了筛峰及后续平均，直接取前六列可能保留由 4+ 候选影响的标签。若可取得原始六通道 v1，则可逐组展开 `input_theo_mz`，原样复制其余字段、`target` 和 provenance，以获得等价的 v2。两条路径均写入独立目录并验证。

若后续更改分组成员，则重新计算平均谱、前体 m/z/RT 均值、`n_spectra`、标签、统计和 provenance。共享扫描处理见文末待讨论项。

### 序列约定

保存现有解析器支持的数字修饰格式。鉴定表编码 `2`、`21` 转成 `C[+57.021464]`、`M[+15.994915]`，编码 `22` 按 K 的残基质量处理，并与 1 基交联位点核对。残基修饰跟在所属残基后，交联位置由 `link_sites` 指定，交联质量在计算碎片时加入。序列按实际长度存储。离线构建器和 Dataset 共用解析函数及质量常数。 [现有序列解析代码](../transfomer/spectrum_transformer/data/dataset.py)

例如 `sequence = ["AC[+57.021464]KPEPTR", "AKM[+15.994915]NLGK"]` 对应 `lengths = [8, 7]` 和 `link_sites = [3, 2]`。解析后校验残基数与 `lengths` 一致，以及交联位点符合各链长度和所用交联剂的位点约定。40 个残基和 39 个断裂位置是张量容量。

## 离线构建

### 理论碎片 m/z

令 $m_{p,r}$ 为链 $p$ 第 $r$ 个残基计入普通修饰后的质量。计算碎片时，N 端修饰质量计入首个残基。单链中性质量为 $M_p=\sum_r m_{p,r}+M_{\mathrm{H_2O}}$。对链 $p$，设 $q$ 为另一条链，$s_p$ 为 1 基交联位置，$\Delta$ 为 `linker_mass`。第 $i$ 个残基之后断裂时，两侧碎片的中性质量为

$$
F_b=\sum_{r\le i}m_{p,r}+\mathbf 1[s_p\le i](M_q+\Delta)
$$

$$
F_y=\sum_{r>i}m_{p,r}+M_{\mathrm{H_2O}}+\mathbf 1[s_p>i](M_q+\Delta).
$$

携带交联位点的一侧加上完整伙伴肽和一份净桥质量。对碎片电荷 $z$，基础 m/z 为 $F/z+m_{\mathrm{proton}}$，临时记为 `base[p,i-1,c]`，形状为 `(2,39,6)`。普通 b 与 b-xl 按位点共享 b 通道，y 同理；不新增离子通道，也不再次给普通修饰数组添加交联质量。

离线基础碎片计算、偏移展开、筛峰和重新匹配共用计算函数及 token 表；运行时展开仅用于显式 v1 兼容路径。令 `offset[t,c] = CHARGE_TOKENS[z_c][t]`，其值已按碎片电荷换算。

```text
g[p,h,t,c] = base[p,h,c] + offset[t,c]
candidate = (h < lengths[p] - 1) & (z_c <= min(3, precursor_charge)) & isfinite(g)
input_theo_mz = where(candidate, g, 0.0)  # 保存 float64 的完整 G 网格
valid_mask = candidate & (g >= 100)
```

离线筛峰和重新匹配使用 `candidate`。Dataset 直接读取 v2 完整网格，应用长度、电荷、有限性和 m/z 下限生成 `valid_mask`，**不得再次添加 token 偏移**。长度和电荷条件必须独立检查，不能用 `m/z!=0` 代替。质量与匹配计算使用 `float64`。

持久化时不做标准化，不按目标匹配结果筛格点，不把 `full` 序列约束并入输入 mask，也不将候选中 `<100`、零或负的展开值提前裁掉。`token=24` 的偏移为零，故 `base=input_theo_mz[:,:,24,:]` 可完整恢复基础碎片，供质量守恒及 `full` 校验使用。补齐及超前体电荷位置则所有 token 一律保存为零。

### 筛峰

读取每个 CSM 对应实验谱的峰表，保留 m/z、强度均有限且为正的峰。实验峰落在 A/B 任一候选格点的 **±0.05 m/z** 窗口内即保留。筛后峰按 `(m/z, 强度)` 去重并按 m/z 排序。 [现有筛峰实现](TokenTableOriginalAnalyze_massivekb_mods_precise_tokens.py#L360)

### 分组平均

合并同组所有筛后峰并按 m/z 排序。每个峰簇以当前最小 m/z 为固定锚点，收集 `[anchor, anchor + 0.05]` 内的峰。每张实验谱在该簇内取最高强度，缺峰贡献 0。簇的输出 m/z 为锚点，输出强度为各谱贡献之和除以 `n_spectra`。单张谱筛空仍计入分母。单谱组使用该谱的筛后峰。前体 m/z 和保留时间分别取组内均值。平均峰表为空的组只计入构建报告。 [现有平均谱实现](AverageSpectrum_batch.py#L184)

### 重新匹配与强度标签

1. 对每个候选格点 `g`，在平均峰表的 **±0.05 m/z** 窗口内取最高峰强度 $I_{\mathrm{avg}}$。未匹配的格点和长度或电荷无效的格点赋 0。
2. 对匹配结果应用 **`full` 序列约束**，对应现有脚本的 `SEQ_CONSTRAINED=full`。主峰和卫星 token 使用窗口匹配结果。每个残基偏移格点的 `g` 必须与同一条链、同一离子通道内某个候选断裂位置的基础 b/y m/z 相差至多 `1e-4`，其余残基偏移格点的强度置 0。基础离子池使用含交联质量的 `base`，或 v2 中 `input_theo_mz[:,:,24,:]`。 [现有序列约束](sequence_token_map.py#L77)
3. 对两条链使用同一个裁剪上限 50,000。对约束后的强度计算 $\sqrt{\min(I_{\mathrm{avg}},50000)/50000}$，将结果写为 `float32`。 [现有强度归一化](trainingExamples.py#L443)

## Dataset 与 collate_fn

`Dataset` 读取持久化字段，并用共享解析函数把 `sequence` 转为临时的 `residue_ids (2,40), int64` 和 `mod_delta (2,40), float32`。残基沿用字母编码，补齐为 0。按各链残基数切出每个断裂位置的左右残基，构造 `cleavage_aa_ids (2,39,2), int64` 和 `cleavage_scalar_features (2,39,11), float32`。填充位置对应特征置 0。

原 7 维按顺序为 $i/L$、$(L-i)/L$、左右残基普通修饰标记、左右修饰质量的 $\tanh(\delta/50)$、N 端修饰质量的 $\tanh(\delta_{\mathrm N}/50)$。新增 4 维为 $(i+0.5-s)/L$、$\mathbf1[s\le i]$、$\mathbf1[s=i]$、$\mathbf1[s=i+1]$。

Dataset 使用 `np.load(..., mmap_mode="r", allow_pickle=False)` 按组读结构化数组。v2 的 `collate_fn` 直接堆叠完整理论 m/z、标准化输入，并按下表准备模型输入和训练监督；仅显式 v1 兼容路径做偏移展开。张量堆叠后最前面加 batch 维 `B`，每个 batch 样本始终是一对肽的一张 XL 平均谱。

| 运行时字段 | 单样本形状 | dtype | 计算规则 |
|---|---|---|---|
| `input_theo_mz` | `G` | `float64` | 磁盘中的原始 m/z；留给输出、质量校验，不送入标准化特征投影 |
| `input` | `G` | `float32` | 对已展开的有效格点按通道标准化，无效格点置 0 |
| `valid_mask` | `G` | `bool` | 按理论格点的长度、电荷及 m/z 条件生成 |
| `aux_features` | `(3,)` | `float32` | 拼接并逐维标准化 `[precursor_charge, precursor_mz, rt_seconds]` |
| `precursor_charge`、`lengths` | `()`、`(2,)` | `int64` | 转为 embedding 与电荷路由使用的整数 |
| `target` | `G` | `float32` | 读取离线强度标签，`valid_mask` 以外置 0 |
| `main_anchor_target`、`main_anchor_mask` | 各 `G` | `float32`、`bool` | mask 取 `valid_mask` 中的 `t=24` 基础峰格点。标签在 `target >= 0.03` 且为同链、同通道相邻断裂内的局部极大值时取 1，其余取 0 |

六个 m/z 通道的均值和标准差从 train 的理论有效格点统计，三个全局输入的统计量也从 train 计算。val、test 和推理共用这些值。注意力和谱图强度损失使用 `valid_mask`，cosine 按样本展平 A/B 两链计算。主峰标签门槛 `0.03` 对应归一化前的平均谱峰强度约 45。

非空平均谱组均写出样本行。全局谱图损失和 cosine 覆盖这些样本。共享主峰 BCE 和幅度损失使用 Z≥2 的有效格点，幅度项只监督达到阈值的局部主峰。额外的分组损失使用各自的电荷与短链条件。无有效正标签样本的 cosine 记为 0。按 split 报告总组数、平均峰表为空的组数、非空但无有效正标签的组数和有有效正标签的组数。

## Model 与谱图输出

### 第三页：双肽共享特征编码

记 `D=256`、`Dctx=64`。将两条肽的局部特征分别送入同一套参数；只在局部编码计算时允许临时 reshape 为 `B*2`，进入 Transformer 之前必须恢复每个肽对的整体。

| 分支 | 输入 | 运算与输出 |
|---|---|---|
| m/z | 标准化 `input (B,2,39,46,6)` | 共享 `Linear(6,256)`，输出 `(B,2,39,46,256)` |
| 断裂两侧残基 | `cleavage_aa_ids (B,2,39,2)` | 沿用原版字母编码和 `Embedding(28,64)`；左右求平均得到 `(B,2,39,64)` |
| 断裂标量 | `cleavage_scalar_features (B,2,39,11)` | `Linear(11,64)`，与残基表示 concat 为 128 维，再用 `Linear(128,256)`；沿 46 个 token 广播 |
| 电荷与链长 | `precursor_charge (B,)`、`lengths (B,2)` | 沿用 `Embedding(8,64)` 和 `Embedding(41,64)`，每链 concat 后 `Linear(128,256)`；沿断裂和 token 广播 |
| token family | 46 个 token 各自的 family ID | `Embedding(3,256)`，乘原版可学习 `family_context_gain`，初值 0.10；广播至两肽和所有断裂 |
| 全局条件 | `aux_features (B,3)` | `Linear(3,256)` 后以 `(B,1,1,1,256)` 广播到整张 XL 谱 |

将上述分支相加得到 `X (B,2,39,46,256)`。`aux_features` 顺序保持 `[precursor_charge,precursor_mz,rt_seconds]`，使用实际前体 m/z，不把中性质量当作该字段。两侧残基求平均是原版行为；本轮不改为 concat。A/B 使用各自的序列、修饰、长度和交联位点，伙伴肽质量已经体现在原始理论 m/z 中。

第三页的“token family `(46)`”表示长度为 46 的 family ID 数组，其取值只有 0/1/2；不表示 46 个 family。图中的可学习参数按原版 family embedding 与标量 gain 实现，不额外添加另一份重复的 46×256 token 参数表；token identity 由第四页的 `t` embedding 表达。

### 第四页：联合 pack、注意力与 unpack

1. `cell_valid=valid_mask.any(dim=-1)`，形状 `(B,2,39,46)`。保留原版物理有效性定义，`full` 仅约束离线标签；暂不将它并入网络 mask。
2. 在每个样本内部按 `(p,h,t)` 展平。可用 `k=((p*39)+h)*46+t` 保存 scatter 索引。每张谱 `N_i=N_A,i+N_B,i`，批内 `N_max=max_i N_i`；PPT 的 `N_max=N_A+N_B` 应按此区分样本与批次。
3. 对有效格点 pack 成 `(B,N_max,256)`，另存 `p/h/t`、scatter 索引以及 `(B,N_max)` 的 `key_padding_mask`，其中 `True` 表示补齐位置。
4. 加上 `Embedding(2,256)[p] + Embedding(39,256)[h] + Embedding(46,256)[t]`。保留原版 h/t 编码，新增 p 编码；编码使用原来的链内断裂位置，不使用 pack 后重新排序的位置。
5. 执行 5 层共享 Transformer Encoder，两条肽之间允许 attention。每层沿用原版 `d_model=256,nhead=8,dim_feedforward=1024,dropout=0.15,activation=GELU` 及其归一化顺序。
6. 按共同前体电荷选择一层 `Z=2 / Z=3 / Z>=4` Encoder，计算 `H + alpha_Z*(H_Z-H)`，接原版连续两个 LayerNorm。`alpha_2/alpha_3/alpha_4plus` 初值分别为 `0.08/0.10/0.01`。当前方案保留三分支；可按电荷子批次执行以节省计算，但须与原版“各分支计算后选择”的数学一致。
7. 依据 scatter 索引还原 `(B,2,39,46,256)`。补齐 token 不得散射成有效信号，还原后再次用 `cell_valid` 归零。校验全无效输入并明确报错或返回零，不让全 padding 的 attention 产生 NaN。

A/B 共享编码和预测头参数，但 `Embedding(2,256)` 会区分肽身份，因此本模型不宣称天然具备 A/B 交换等变性。训练和推理统一采用数据构建时的规范排序；交换输入链后，必须先重新规范排序，才能比较输出一致性。

### 第四页：强度头与主峰辅助头

三个独立 family 强度头均为 `LayerNorm(256) -> Linear(256,256) -> GELU -> Linear(256,6)`，各自输出 `(B,2,39,46,6)`。按 token family 在相应格点选择 head logits；family 索引广播为 `(1,1,1,46,1)`，两条肽共用这三个头。

沿用原版 `prediction = g*softplus(logits)`，`g` 初值 1.05，再乘 `valid_mask`。原版 g 是无约束标量；本轮为对齐原版保留，正值参数化是否加入见待讨论项。main/satellite/residue 三个末层 bias 初值均为 -1.00。

主峰辅助头为 `LayerNorm(256) -> Linear(256,128) -> GELU -> Linear(128,6)`，末层 bias 初值 -0.85，输出同形状的 `main_anchor_logits` 和 `sigmoid(logits)*valid_mask`。该概率只用于辅助监督和诊断，**不乘入最终预测强度**，与原版 forward 一致。

主峰辅助 mask 只覆盖 `token=24` 的基础峰。原版 `build_main_peak_mask` 在单肽 `(h,t)` 网格上做 3×3 max-pool；传入仅包含 token24 的 mask 后，等价于同链同通道相邻断裂基础峰的局部极大值。XL 中将 `(B,2,39,46,6)` 临时 reshape 为 `(B*2,39,46,6)` 调用此单肽函数，再还原双肽维。不得将 A/B 沿断裂轴拼成 78 行再做局部池化。阈值继续使用 `0.03`，不将其误称为谱内基峰的 3%。

模型至少返回 `prediction`、`backbone_pred`、`detail_pred`、`detail_logits`、`main_anchor_logits`、`main_anchor_prob`，保留原版训练所需的同义输出。原始理论 m/z 随样本单独传递，不从标准化输入反推导出质量。

### 训练损失与原版对齐

原版实际执行的总损失如下，不能只按参数名推断其作用：

```text
L = 0.20 * global_recon
  + 0.90 * z2_recon
  + 1.20 * z3_recon
  + 1.30 * z3_main_cosine
  + 1.10 * z3_short_recon
  + 0.80 * main_peak_amp
  + 0.35 * family_weighted_recon
  + 0.45 * anchor_z23
  + 0.65 * z3_main_anchor
  + 0.15 * density_gap_penalty
```

`global_recon=L1+cosine_loss`；z2/z3 的重建项也由 L1 与 cosine 组成。family 加权 L1 中 residue/satellite 权重为 1/3，main 权重为 Z3 时 11、其他电荷时 8。详细正负样本权重和归一化分母沿用原版 [losses.py](../../model/model/v20_0918/transfomer/spectrum_transformer/training/losses.py)。

保留原版运算时，双肽适配必须满足：

- 全谱 cosine 使用 `reshape(B,-1)`，A/B 联合得到每张 XL 谱一个分数；不能当成 `2B` 张独立谱平均 cosine，也不分别归一化 A/B 强度。
- L1 对两条肽所有有效格点共同加权求和，并除以对应有效权重之和。按样本求和的维度由原版 `(1,2,3)` 改为 `(1,2,3,4)`。
- family 索引改为 `mask[...,token_ids,:]`；电荷样本权重广播为 `(B,1,1,1,1)`。局部池化只在链内运行。
- 原版 BCE 使用主峰局部极大值标签、概率 clamp 和正负样本权重；本轮不自动替换标签定义。若改用 logits 版本，应保持同一标签、同一权重和分母，作为数值实现调整单独验证。
- 首版按每条链的 `length<=16` 判定，形成 `(B,2)` 的短链 mask。Z3 短链重建和额外 anchor BCE 使用选中链的有效格点，cosine 按 XL 样本展平。没有有效短链格点的样本权重为 0。
- 原版 `amp_sample_weight` 对 Z2/Z3/其他电荷分别为 1.2/2.2/0。XL 首轮对 Z2/Z3/Z≥4 均设为 1.0，作用于共享 `anchor_z23` 和 `main_peak_amp`；原有 `anchor_z23` 名称用于兼容日志。`z3_main_anchor` 仍按 Z3 短链项单独计算，短链口径见上一条。

原版 `v20_backbone_weight` 没有进入上述 total，`v20_group_family_focus_weight` 只用于日志，`strict_presence` 只进入日志项 `amp_strict`。若保留原版，不能在迁移时把这些参数悄悄接入总损失。原版 density 项使用硬阈值计数，保留其报告数值时，也不能将它解释为有连续梯度的密度约束。

### 训练配置、评估与 checkpoint

| 设置 | 原版参考值 | XL 方案 |
|---|---|---|
| 优化器 | AdamW | 保留 |
| 学习率 / weight decay | `1e-3 / 3e-4` | 作为首轮参考，不因格式迁移修改 |
| epochs / seed | `20 / 42` | 保留参考值，最终记录实际设置 |
| batch size / workers | `64 / 4` | batch 64 仅为原版参考；按双肽有效 token 数实测显存后确定 |
| 网络 | D=256，8 heads，5+1 层，FF=1024，dropout=0.15 | 保留 |
| scheduler / AMP / 梯度裁剪 | 原版实际训练循环未启用 | 不因配置中出现名称就声称启用；后续优化单独记录 |

验证每个样本的完整 XL 网格 cosine、masked MAE、正标签 MAE、主峰辅助指标及分 family 指标，并按 Z2/Z3/Z4/Z5/Z6 报告样本数与表现。谱图诊断包括预测与标签的峰密度差、平均有效格点强度差、峰存在 precision/recall/F1 和 V20 强度分段召回率。按链以 `L<=16` 分长短链，再按前体电荷报告 cosine 和 masked MAE。整张 XL 谱的主指标按肽对计算。零正标签样本保留并报告，cosine 记为 0。

XL 入口以验证集 A/B 整谱 cosine 的逐样本均值选模，cosine 相同时取 masked MAE 较低的 checkpoint。验证集只有 1 个 Z2 样本，因此分电荷结果同时报告样本数。原版 `compute_v20_selection_score` 可用于对照分析。

选模和超参数调整只使用 train/val；test 在训练方案及 checkpoint 确定后评估。各组指标按真实样本数汇总，不对不等大小 batch 的均值再次等权平均。checkpoint 保存模型参数、11 维特征顺序、token 表、通道顺序、数据 schema、训练统计、排序规则、全部实际生效的 loss/选模/导出设置；旧单肽 checkpoint 不是可直接完整加载的 XL checkpoint，预训练迁移若需要应另列权重映射。

### 训练后谱图导出

谱级输出以原版 `v20_pp4` 完整导出流程为兼容基线，设置 `postprocess_mode="cluster"`、`cluster_scene="v20_pp4"` 和 `family_aware_postprocess_prior=True`。输入为未标准化、含交联质量的理论 m/z 网格和模型直接预测的 `sqrt_clip` 空间强度。沿用现有 XL 方案的以下步骤；改为整谱统一校准属于待讨论项，不在本轮擅自替换。
   - A/B 分别处理，每条链先按自身 `valid_mask` 将无效格点的预测强度置 0，再使用本链残基数、带普通修饰的序列及共同前体电荷执行 PP4。token family 与模型共用 V20 token 表，沿用原版对 `main`、`satellite`、`residue_event` 及卫星子类型的筛选、重排、峰数控制和补峰规则。阈值及预算采用 PP4 随长度、电荷和 token 类别调整的规则。 [原版 PP4](../transfomer/cluster_postprocess.py#L3298)
   - 每条链的候选按 m/z 从小到大排序，以每簇首个 m/z 为固定锚点，将与锚点相差不超过 0.05 Da 的候选归入该簇。每簇保留强度最高格点的理论 m/z 和强度，同时保留簇内最高 `export_score` 和格点数 `num_slots`。随后按本链长度、共同前体电荷和原版 PP4 参数，对该链合峰结果执行一次强度校准。 [原版合峰与校准](../transfomer/predict_npz_to_mgf.py#L309)
   - 将 A/B 校准后的峰表合并，再按上述固定锚点 0.05 Da 规则联合合峰，保留每簇最强峰的理论 m/z 和校准后强度，得到完整 XL 预测谱。最终强度保持在 PP4 校准后的 `sqrt_clip` 空间。

导出必须直接使用传入的 XL m/z，不能重新调用仅输入单链序列的线性肽 m/z 构建函数覆盖它。已核对原版 `run_cluster_postprocess` 直接从传入的 `input_mz` 取质量，不会自行按序列重算；可复用该调用方式。原版 `predict_theoretical_spectrum.py` 则会通过单肽 `build_input_mz_matrix` 重建网格，该 CLI 不能直接用于 XL。需要序列质量支持关系的其他辅助逻辑，必须检查它是否支持包含伙伴肽和交联剂的基础离子；后续实现时为跨交联位置的断裂建立专门验证案例。残基修饰参数仍只描述普通修饰，不伪造一个含伙伴肽质量的普通修饰。

用于结果对比时，另外计算不经 PP4 校准、仅按相同 0.05 Da 规则进行 max 合峰的谱级基线。网格中的多个 token 可以对应同一 m/z，联合合峰沿用最大值而不是累加。网格 cosine 与合峰谱 cosine 分开报告；预测和参考均明确使用 `sqrt_clip` 空间，不能把校准后的强度标为原始计数。这里的“完整 XL 预测谱”指 A/B 在当前六通道和 46-token 字典覆盖范围内的联合谱，不代表覆盖实验谱所有峰。

## 数据构建代码与实施状态

当前构建脚本已实现六通道重建、展开 m/z 的保存和 v1/v2/v3 校验。独立 v1 迁移入口和 XL Dataset/collate 尚待实施。

| 位置 | 修改内容 | 保持的行为 |
|---|---|---|
| `record_dtype` | 六通道重建使用现有 v2 dtype，`input_theo_mz` 为 `(2,39,46,6)`、`<f8`，`target` 为同形状的 `<f4` | 记录字段类型及顺序沿用 v2 |
| `build_theoretical_mz` / `expand_grid` | 计算并展开六通道 base；筛峰、匹配、统计与保存共用同一网格 | XL 质量公式、token 顺序及 float64 精度 |
| `build_split` | 写入 `where(candidate,grid,0)`，不再把 base 写入记录 | 筛峰、平均、full、sqrt_clip、低 m/z 标签保留规则不变 |
| `build_dataset` | `build` 生成六通道 v2 产物，写出对应 schema、形状、通道顺序及统计 | train 统计仅从 train 有效理论格点累计 |
| `verify_dataset` | 沿用现有 v2 校验路径，核对六通道完整网格及 token24 基础切片 | 成员对应、target 约束、分区计数及统计核验 |
| v1 迁移入口（待实现） | 流式读取旧记录，只展开 input 并复制其他字段到独立 v2 目录 | 逐行保留 group_id、target 和 provenance；不重跑源谱处理 |
| XL Dataset / collate | 显式识别 v1/v2；v2 不再展开，先计算 mask 再标准化 | 原版 6 通道统计、3 维 aux、字母编码和局部特征规则 |

训练通道统计仍对全部有效 `(A/B,h,t)` 格点按最后一维六通道汇总，并以 float64 累加；旧构建器本来就是按展开网格统计，纯格式迁移不应改变 count/mean/std。val/test 不参与统计。不得因为输入字段变大而按“每条谱先算均值再等权平均”替代原来的格点累计。

## model-xl 代码实施顺序

1. **数据接口**：接入现有六通道 v2 产物的读取器。若从六通道 v1 迁移，确认同一旧样本在两条读取路径产生相同的模型输入、mask 和 target。
2. **局部编码**：复用原版 parser、7 维特征和 embedding；加 4 维交联特征，完成两个肽共享参数与正确广播。
3. **联合 backbone**：扩展 pack/unpack、p/h/t 编码和前体电荷路由。以完整肽对为 batch 单位，验证 padding 与 scatter。
4. **输出与监督**：复用三个 family 头、辅助头、主峰标签和实际 loss 公式，按双肽轴处理 mask 和归约。共享主峰损失使用等权方案，Z3 短链按链筛选。
5. **训练与评估**：使用 AdamW。终端显示 batch loss、cosine、峰密度差和耗时，并显示验证和测试进度。每轮指标写入 `training_history.csv`，按验证集整谱 cosine 保存最佳 `best.pt`。训练结束后重载 `best.pt` 评估 test，将最佳验证和最终测试指标写入 `run_summary.json`。
6. **导出**：将 v2 原始 XL m/z 传入 PP4，按兼容路线处理两肽并联合 max 合峰；同时报告不经 PP4 的谱级对照。

模型入口是 [`spectrum_transformer.cli.train_xl`](../transfomer/spectrum_transformer/cli/train_xl.py)，数据、模型和损失分别位于 [`xl_dataset.py`](../transfomer/spectrum_transformer/data/xl_dataset.py)、[`xl_transformer.py`](../transfomer/spectrum_transformer/models/xl_transformer.py) 和 [`xl_losses.py`](../transfomer/spectrum_transformer/training/xl_losses.py)。从 `model_xl/transfomer` 用包含 PyTorch 和 NumPy 的环境执行：

```powershell
& 'E:/hard_cityu/XLsearch/XLsimulation/Scripts/python.exe' -B -m spectrum_transformer.cli.train_xl --smoke --batch-size 5 --device cuda
```

真实 v2 smoke 按种子 42 读取 Z2–Z6 各一组，检查 `(5,2,39,46,6)` 输出、有效 mask、有限 loss 和梯度，执行一次 AdamW 更新，并核对内存 checkpoint 重载后的预测。RTX A4000 上本批次峰值 CUDA 分配约 0.94 GiB，正式训练的 batch size 仍需按实际样本长度测量。另在真实小样上验证了一轮训练与验证、末批不足 `batch_size` 时的逐样本汇总及 `best.pt` 保存。

## 构建与校验命令

在包含 NumPy 和 PyArrow 的 Python 环境中，从本目录运行 [双链构建器](build_xl_spectrum_dataset.py)。六通道完整构建写入独立目录：

```powershell
python -B build_xl_spectrum_dataset.py build --input-dir D:/XL/prosit-xl --output-dir ./xl_spectrum_training_data_v2
python -B build_xl_spectrum_dataset.py verify --output-dir ./xl_spectrum_training_data_v2
```

小样构建可用 `--limit-groups` 限制各分区组数，并指定单独的输出目录。已发布的 v2 可用 `verify --output-dir ./xl_spectrum_training_data` 校验。v1 纯格式迁移命令随对应入口实现后补充。

本次 9 项回归测试通过，覆盖六通道构建、Z4/Z5/Z6 前体、仅命中 4+ 格点的峰在平均前被筛除、历史版本校验和错误格式拒绝。真实数据 train/val/test 各 32 组均通过构建及校验。

此前的六通道 v1/v2 存储验证结果如下。

- 当时的 5 项数据存储回归测试通过，覆盖独立理论质量/强度预期、空谱平均、空组丢弃、低/负 m/z、补齐与超电荷位置、v1 兼容，以及错误元数据和损坏网格拒绝。[当前回归测试](test_build_xl_spectrum_dataset.py)
- 在同一 Python 环境下用修改前后的构建器分别构建 train/val/test 各 32 组，共 96 组真实数据。除输入字段形状和格式元数据外，target、所有其他记录字段、split 报告、训练统计和 provenance 完全一致；v2 网格与 v1 展开结果逐元素一致。
- 同时对照当时完整 v1 产物中的这 96 组，target 和所有非输入字段逐元素一致。理论 m/z 独立重算与既有文件的最大差值为 `9.09e-13`。同一环境下的修改前后对比逐元素一致。

## 数据与模型验收

- 按 split 核对分组和计数。`group_id` 能回查组键和成员谱，`n_spectra` 等于组内不同实验谱的数量。总组数等于空组、非空零匹配组和有效监督组数之和。每条写出的记录符合字段表的形状、dtype 及 A/B 顺序。
- 同一断裂的中性 b/y 质量之和为 $M_A+M_B+\Delta$。用首端、中间和末端的交联位点分别核验。
- 离线与运行时在同一格点得到相同的理论 m/z。补齐位置及超电荷位置的目标为 0，Dataset 对 m/z 低于 100 的格点屏蔽输入和目标。
- v2 的 `input_theo_mz` 和 `target` 均为 `(2,39,46,6)`，通道顺序为 `[b1+, b2+, b3+, y1+, y2+, y3+]`，dtype 分别为 float64 和 float32。token24 与原 base 一致，candidate 外全零。覆盖负偏移导致的非正值及 `<100` 基础峰，验证它们在磁盘保留、在 runtime 屏蔽，且不破坏基础质量恢复。
- m/z 下限逐展开格点判断：若 base<100，而某个正偏移后的格点>=100，该格点仍有效；不能按基础峰大小把整行 46 个 token 一起遮掉。
- 从原始谱重建时，核对筛峰候选仅来自六个碎片通道，并覆盖 Z4/Z5/Z6 前体。若走 v1 纯格式迁移路径，样本数、group_id、所有非输入字段和 target 逐元素相同，v1/v2 的 valid_mask、标准化 float32 输入、runtime target、辅助标签与训练统计相同。读取器拒绝八通道 v3 及版本与形状冲突的数据，v2 直接使用原始展开 m/z。
- 用两张实验谱核验平均强度。一个峰簇只有一张谱贡献强度 1000 时，平均峰强度为 500，匹配后通过序列约束的标签为 0.1。强度超过 50,000 的匹配峰经裁剪后的标签为 1。
- 按 token 类别核验 `full` 约束。残基偏移格点通过约束需在同链同通道找到基础离子，主峰和卫星 token 按 m/z 窗口的匹配结果赋值。
- 批内 pack/unpack 保持有效格点的链、断裂和 token 位置。主峰监督仅覆盖基础 token，模型注意力使用由理论格点得到的 mask。
- 用异长肽、不同电荷及不同有效 token 数构造混合 batch，验证 scatter 后有效格点对应值不变、padding 不影响结果；主峰局部池化不跨 A/B 边界。
- 在共享单肽模块、相同权重与关闭 dropout 的条件下，局部 embedding、family readout、主峰标签和对应损失应与原版逐链调用一致；联合 attention 是有意新增的双肽运算，不要求整网输出与两个独立单肽模型相同。
- 完整 XL cosine 必须为每个肽对一个数。测试修改 A/B 相对幅度会改变整体指标。构造有合格主峰的 Z4/Z5/Z6 样本，验证共享主峰 BCE 和幅度损失的梯度。
- 在小样本上验证一次前向、反向和参数更新：loss/梯度有限、所有输出形状正确；再验证小数据拟合与 checkpoint 重载的一致性。显存实测后确定 batch size。
- PP4 始终接收 XL raw m/z。对含交联位点的断裂及跨链/跨 token 重合峰验证质量不丢失、合峰不求和；输出与参考采用一致的强度空间和匹配容差。

## 待讨论项与当前状态

六通道接口和共享主峰损失的等权配置已确认。下列事项仍待确定，会影响后续训练、选模或数据处理。

| 项目 | 已确认事实与影响 | 建议及待决定内容 |
|---|---|---|
| 三电荷分支 | Z2 train 仅 18 组，分支数据少 | 当前保留 PPT 与原版结构；不因该统计自动删分支。是否后续共享分支以消融实验判断 |
| 共享原始扫描 | 24 张相同观测被不同肽对注释分配至不同 split；肽对无交集不能保证谱图独立 | 可在平均前从所有涉及 split 排除这些扫描；或在确认注释有效后，使同一扫描的全部关联组只属于同一 split，并复查肽对隔离；或维持来源划分并报告局限。未决定前不删除、不移动 |
| full support 与 runtime mask | full 在离线标签上置零，原版 runtime mask 仍只按长度、电荷和 m/z | 当前保留。并入 support 可减少无效 token，但改变网络输入和损失定义，应另做消融 |
| 归一化、CE 与 RT | 当前六通道 v2 的 train 正标签有 12.94% 达到 clip 上限；未保存 CE，RT 是组均值 | 当前保留 sqrt_clip(50000)、跨条件平均和 RT。若要预测指定 CE 或恢复绝对高峰差异，需要另立数据版本，不能靠存储展开解决 |
| 局部 residue、主峰定义与 g | 原版采用左右平均、链内网格局部极大值、无约束标量 g | 当前保留。concat、出现概率标签、正值 g 都是独立算法改动，不随本轮迁移引入 |
| PP4 双链校准 | 分链校准可能影响 A/B 相对强度，原版先验来自单肽 | 当前保留现有分链 PP4 再联合 max 合峰路线；增加无 PP4 的联合谱对照。联合校准及峰预算如何定义另行讨论 |
