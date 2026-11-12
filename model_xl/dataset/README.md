# Dataset

这里只保留 MassiveKB 数据集生成主流程。

## 默认服务器路径

```bash
cd /home/wangyuhang/seq2mass/v20_0701

DATA_DIR=/home/wangyuhang/datasets/massivekb
PREFIX=massivekb_82c0124b
```

## 输入路径支持

脚本支持两类原始布局。

### 1. 平铺布局

```text
/home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train.mgf
/home/wangyuhang/datasets/massivekb/massivekb_82c0124b_val.mgf
/home/wangyuhang/datasets/massivekb/massivekb_82c0124b_test.mgf
```

或压缩版：

```text
/home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train.mgf.xz
/home/wangyuhang/datasets/massivekb/massivekb_82c0124b_val.mgf.xz
/home/wangyuhang/datasets/massivekb/massivekb_82c0124b_test.mgf.xz
```

### 2. 分层布局

```text
/home/wangyuhang/datasets/massivekb/train/orig/massivekb_82c0124b_train.mgf
/home/wangyuhang/datasets/massivekb/val/orig/massivekb_82c0124b_val.mgf
/home/wangyuhang/datasets/massivekb/test/orig/massivekb_82c0124b_test.mgf
```

或压缩版：

```text
/home/wangyuhang/datasets/massivekb/train/orig/massivekb_82c0124b_train.mgf.xz
/home/wangyuhang/datasets/massivekb/val/orig/massivekb_82c0124b_val.mgf.xz
/home/wangyuhang/datasets/massivekb/test/orig/massivekb_82c0124b_test.mgf.xz
```

当前一键脚本会优先识别分层布局。

## 1. 一次性执行整条流程

直接运行：

```bash
cd /home/wangyuhang/seq2mass/v20_0701
bash dataset/scripts/run_massivekb_pipeline.sh
```

默认行为：

```text
val -> ${PREFIX}_val.npz
test -> ${PREFIX}_test.npz
train -> ${PREFIX}_train_merged
```

默认支持：

- 已解压 `.mgf`
- 压缩包 `.mgf.xz`

如果只有 `.mgf.xz`，脚本会自动解压后继续处理。

## 2. 一次性脚本允许上传的参数

这些参数都可以在运行时覆盖：

```bash
DATA_DIR=/home/wangyuhang/datasets/massivekb \
PREFIX=massivekb_82c0124b \
INPUT_TRAIN=/home/wangyuhang/datasets/massivekb/train/orig/massivekb_82c0124b_train.mgf \
INPUT_VAL=/home/wangyuhang/datasets/massivekb/val/orig/massivekb_82c0124b_val.mgf \
INPUT_TEST=/home/wangyuhang/datasets/massivekb/test/orig/massivekb_82c0124b_test.mgf \
TOL_FILTER=0.05 \
TOL_AVERAGE=0.1 \
TOL_NPZ=0.1 \
CLIP_INTENSITY=50000 \
NUM_BUCKETS=256 \
NUM_WORKERS=8 \
CHUNK_SIZE=5000 \
CLEANUP=1 \
AUTO_DECOMPRESS=1 \
bash dataset/scripts/run_massivekb_pipeline.sh
```

参数说明：

- `DATA_DIR`
  - MassiveKB 数据目录
- `PREFIX`
  - 文件前缀
- `INPUT_TRAIN`
  - 显式指定 train 原始 mgf 路径
- `INPUT_VAL`
  - 显式指定 val 原始 mgf 路径
- `INPUT_TEST`
  - 显式指定 test 原始 mgf 路径
- `TOL_FILTER`
  - Token 过滤容差
- `TOL_AVERAGE`
  - 平均谱聚合容差
- `TOL_NPZ`
  - 构建 target matrix 的匹配容差
- `CLIP_INTENSITY`
  - 强度裁剪上限
- `NUM_BUCKETS`
  - 平均谱分桶数
- `NUM_WORKERS`
  - 多进程 worker 数
- `CHUNK_SIZE`
  - 分片大小
- `CLEANUP`
  - 是否删除中间文件，`1` 为删除
- `AUTO_DECOMPRESS`
  - 是否自动解压 `.mgf.xz`

## 3. 分步执行流程

### Step 1: Token 过滤

```bash
python dataset/TokenFilter_parallel.py \
  /home/wangyuhang/datasets/massivekb/train/orig/massivekb_82c0124b_train.mgf \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_filtered.mgf \
  --tol 0.05 \
  --workers 8
```

### Step 2: 平均化谱图

```bash
python dataset/AverageSpectrum_diskbased.py \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_filtered.mgf \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_averaged.mgf \
  --tol 0.1 \
  --buckets 256
```

或者：

```bash
bash dataset/scripts/run_average_diskbased.sh \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_filtered.mgf \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_averaged.mgf \
  256
```

### Step 3: MGF 转分片 NPZ

```bash
python dataset/NPZBuilder_chunked.py \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_averaged.mgf \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_npz \
  --tol 0.1 \
  --chunk-size 5000 \
  --clip-intensity 50000 \
  --workers 8
```

### Step 4: 产物规则

当前推荐规则：

```text
train -> merged memmap
val   -> single npz
test  -> single npz
```

### Step 5: train 合并为 memmap

```bash
python dataset/merge_shards_to_memmap.py \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_npz \
  /home/wangyuhang/datasets/massivekb/massivekb_82c0124b_train_merged
```

## 4. 推荐训练输入

```text
train -> ${PREFIX}_train_merged
val   -> ${PREFIX}_val.npz
test  -> ${PREFIX}_test.npz
```
