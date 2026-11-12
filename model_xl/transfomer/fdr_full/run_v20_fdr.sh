#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# V20 方法（df / df_mby 分支）1% FDR —— 复用 transfomer/fdr/ 现有管线。
# V20 的 TD 库/匹配/逐候选预测/分支打分/FDR 曲线全部由 fdr/step1-4 完成；
# 本脚本只是带正确 BRANCHES 与 env 调用它。
#
# BRANCHES 语义（fdr/step3）：';' 分隔多个分支，每个是 gc,d,f_bidir,f_exp,mby 的逗号子集。
#   d,f_exp     = df 分支
#   d,f_exp,mby = df + main-b/y 分支（对齐 DS hsdf_mby 的组分）
# 融合 = 分量原始等权平均（无谱内归一化）——跨谱可比，可直接做 FDR 排序。
#
# 说明：现有 fdr/step1 依赖 v1-search 布局（root/<sp>/for_<sp>_...fasta + MGF），新数据集
# 用 fdr/step1 --mods-config 生成 TD 库或 NO_MOX=1 快模式；其它 env 直通 fdr/run_fdr.sh。
#
# Usage（服务器 GPU）：
#   bash run_v20_fdr.sh                    # mouse, df+dfmby 分支
#   SPECIES=mouse MAX_SCANS=300 bash run_v20_fdr.sh
#   SPECIES=all BRANCHES='d,f_exp;d,f_exp,mby' bash run_v20_fdr.sh
set -euo pipefail

FDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../fdr" && pwd)"
SPECIES="${SPECIES:-mouse}"
BRANCHES="${BRANCHES:-d,f_exp;d,f_exp,mby}"

echo "[run_v20_fdr] SPECIES=$SPECIES BRANCHES=$BRANCHES"
echo "[run_v20_fdr] calling fdr/run_fdr.sh (env passthrough: CKPT/MAX_SCANS/PPM/"
echo "              REL/EXP_TRANSFORM/NO_MOX/MAX_VAR_MODS/FDR_TARGETS/...)"
export SPECIES BRANCHES
bash "$FDR_DIR/run_fdr.sh"
