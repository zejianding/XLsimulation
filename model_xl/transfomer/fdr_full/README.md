# fdr_full：可复用 1% FDR 全流程（MS-GF+ / MSFragger / V20-df·dfmby）

> 对任意数据集计算"**1% FDR 线上的识别数**"：输入 = 基础肽段 FASTA + 谱 + 手动 mods JSON，
> 输出 = 在 `(n_decoy+1)/n_target ≤ 1%` 前缀上的 **n_target / n_decoy**（论文公式，**不比对 GT**，
> 未来数据集可能没有真值注释）。
>
> 修饰可配（mods_config JSON），覆盖"其它可变修饰"的数据集。

## 三种方法的定位

| 方法 | 谁跑搜索 | 脚本/命令 | 输出 |
|---|---|---|---|
| **MS-GF+** | 本目录自动 | `run_msgfplus.sh`（建 TD → MS-GF+ → 计数） | n_target / n_decoy @1%FDR |
| **MSFragger** | 你 FragPipe 手动 | `engine_fdr.py`（对 FragPipe tsv 计数） | n_target / n_decoy @1%FDR |
| **V20 `df` / `df_mby`** | 复用 `../fdr/` | `run_v20_fdr.sh` → `fdr/run_fdr.sh`（step1-4） | 每分支 n_target / n_decoy @1%FDR |

共同口径：
- TD 库 = in-scope 肽段 FASTA 的全修饰变体（targets）+ 每条反转（decoys，`rev_` 前缀）；
- 反转保持中性质量 → ±ppm 前体窗口对 T/D 近似等量召回；
- FDR 按论文公式 `FDR_i = min_{j≤i}(ndecoy_j+1)/ntarget_j`，取 ≤1% 的最大高分前缀；
- `n_target` = 该前缀里通过谱数（**每谱只取最优 PSM**，MS-GF+ tsv 先按谱去重再计）。

## 目录文件

```
fdr_full/
  mods_config.example.json  手动修饰配置示例
  build_td_fasta.py         (引擎用) 基础肽段 FASTA + mods JSON → TD FASTA（rev_ decoy）
  engine_fdr.py             通用 1% FDR 计数（MS-GF+ tsv / MSFragger tsv，GT 无关）
  run_msgfplus.sh           MS-GF+ 全自动（建 TD → 搜索 → 转换 → 计数）
  run_v20_fdr.sh            V20 df/dfmby：调 ../fdr/run_fdr.sh
  data/  output/            运行时产物
```

## 修饰配置（新数据集改这里）

`mods_config.example.json`：
```json
{"fixed": {"C": [57.021]},
 "variable": {"M": [15.995]},
 "max_variable_mods": 3}
```
- `fixed`：残基必加（支持 `NTERM`）；`variable`：可选，各残基独立，全部可变位点 ≤ `max_variable_mods`。
- 新修饰直接加：如 `"S": [79.966]`（磷酸化）、`"N": [0.984]`。数值用 V20 质量表口径的单同位素位移。

## MS-GF+（自动，一条命令）

```bash
cd E:/PythonProject/Seq2Mass/v20_0821/transfomer/fdr_full
# <spec_mzML>：mzML 或 MGF；<out_dir>：输出目录（TD fasta/tsv/计数）
bash run_msgfplus.sh <name> <for_<x>_peptide.fasta> <spec.mzML> <out_dir>
# 例（v1-search mouse）：
bash run_msgfplus.sh mouse \
    E:/biodatasets/denovo/v1-search/mouse/for_mouse_test_inscope_peptide.fasta \
    E:/biodatasets/denovo/v1-search/v2mzml/high.mouse.PXD004948_len7_32_mod_within_scope_10000.mzML \
    output/mouse
```
env：`MSGFPLUS_JAR`、`MZID2TSV`、`MODS`（默认 example JSON）、`THREADS`、`INST`(默认1)。
每物种 4 步：建 TD → MS-GF+（`-ti 0,0 -inst 1 -e 9 -minCharge 2 -maxCharge 4`，不加 `-tda`）→
.exe 转 tsv → `engine_fdr.py` 计数。

手动等价命令（便于逐物种核对）：
```bash
MSG=E:/MSGFPLUS/MSGFPlus_v20240326/MSGFPlus.jar
CONV=E:/MSGFPLUS/MSGFPlus_v20240326/MzidToTsvConverter/MzidToTsvConverter.exe
python build_td_fasta.py --fasta for_mouse_test_incope_peptide.fasta \
    --mods-config mods_config.example.json --out-fasta output/mouse_td.fasta
java -Xmx12G -jar "$MSG" -s <spec.mzML> -d output/mouse_td.fasta \
    -o output/mouse.mzid -t 10ppm -ti 0,0 -m 3 -inst 1 -e 9 \
    -minCharge 2 -maxCharge 4 -thread 8 -allowDenseCentroidedPeaks 1
"$CONV" output/mouse.mzid -mzid:output/mouse.mzid -tsv:output/mouse.tsv -unroll -showDecoy
python engine_fdr.py --engine msgfplus --tsv output/mouse.tsv \
    --decoy-token rev_ --fdr 0.01
```

## MSFragger（你 FragPipe 手动搜，我计数）

1. FragPipe 搜索：数据库 = `build_td_fasta.py` 产出的 `<name>_td.fasta`，`decoy_prefix = rev_`，
   酶切 nocleavage、Isotope Error=0、±10ppm、关校准（产物 tsv）。
2. 计数：
```bash
python engine_fdr.py --engine msfragger --tsv <fragpipe>.tsv \
    --decoy-token rev_ --fdr 0.01
```

## V20 `df` / `df_mby`（复用 transfomer/fdr/）

```bash
# 服务器 GPU（v1-search 布局）
cd E:/PythonProject/Seq2Mass/v20_0821/transfomer/fdr_full
bash run_v20_fdr.sh                    # mouse, BRANCHES=d,f_exp;d,f_exp,mby
SPECIES=all bash run_v20_fdr.sh
# 或直接：
cd ../fdr && BRANCHES='d,f_exp;d,f_exp,mby' bash run_fdr.sh
```
- 分支语义：`d,f_exp`=df；`d,f_exp,mby`=df+主b/y（对齐 DS hsdf_mby 组分）；融合=分量**原始等权平均**
  （不做谱内 min-max，跨谱可比，可直接做 FDR 排序——这正是 1% FDR 的要求）。
- 新数据集/其它可变修饰：V20 侧用 `fdr/step1_build_td_fasta.py --mods-config <json>` 生成带 LABEL 的
  TD 库（该脚本已支持 `--mods-config`）；或先用本目录 `build_td_fasta.py` 预生成库。

## 输出解读

`engine_fdr.py` 打印（以已测 v1-search mouse 为例）：
```
MSFragger mouse:  n_target=9933  n_decoy=5   (expect <= 4.49)
MS-GF+   mouse:  n_target=9924  n_decoy=54   (SpecEValue <= 2.2e-04, 9978 spectra)
```
- `n_decoy ≈ 1%×n_target` 属正常（+1 修正下略高）；
- 不同方法/分支比较：**n_target 越大越好**（同 FDR 线、同 decoy 集、同公式）。

## 注意

- MGF 输入 MS-GF+ 可能把密集 centroid 谱误判 profile 跳过 → 用 mzML + `-allowDenseCentroidedPeaks 1`
  （脚本已带）；
- `MzidToTsvConverter.exe` 只认绝对路径（run_msgfplus.sh 已处理）；
- `engine_fdr.py` 按谱去重（`-unroll` 的 I/L 重复行只留最优），保证跨方法计数口径一致。
