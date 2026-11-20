#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# 把矩阵产出的 PiPrime 余弦 CSV 归档到新目录，按 物种 / 后处理 / 投影 / rel档 分层。
#
# 源：<OUT>/output/<v20pp|v20raw>_<sp>_<stamp>/variants/<proj>_rel005-020_expnone_prednone/Grouped_MGFs_rel<lvl>/
# 目标：<DST>/<sp>/<pp|raw>/<noproj|grid>/rel<lvl>/{cosine_summary.csv,cosine_detail.csv}
#
# 命名说明：
#   · 后处理层  pp / raw       ← 来自目录名 v20pp / v20raw（是否做簇后处理）
#   · 投影层    noproj / grid  ← 来自 variants 里的 raw / grid 前缀（是否投影到 39×46×6 网格）
#     故意不叫 raw，否则会出现 .../raw/raw/ 这种"raw 指两件事"的路径
#   · rel 档单独一层：同一 (物种,后处理,投影) 下有 005/010/020 三档，不分开会重名
#
# env：OUT（源根，默认 /data/wangyuhang2/v20）  DST（目标根，默认 /data/wangyuhang2/primecosine）
#      SPECIES（默认九物种）  MOVE=1 用 mv 代替 cp（默认 cp，保守）
# ============================================================================
set -uo pipefail

OUT="${OUT:-/data/wangyuhang2/v20}"
DST="${DST:-/data/wangyuhang2/primecosine}"
SPECIES="${SPECIES:-mouse human yeast bacillus clambacteria honeybee mmazei ricebean tomato}"
MOVE="${MOVE:-0}"
REL_TAG="rel005-020_expnone_prednone"

[ -d "$OUT/output" ] || { echo "ERROR: 找不到 $OUT/output"; exit 1; }
mkdir -p "$DST"

n_cp=0; n_miss=0
for SP in $SPECIES; do
  for TAG in v20pp v20raw; do
    RUN=$(ls -dt "$OUT/output/${TAG}_${SP}_"* 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then echo "skip $SP/$TAG: 无 run 目录"; n_miss=$((n_miss+1)); continue; fi
    POST="${TAG#v20}"                       # v20pp->pp, v20raw->raw
    for PROJ in raw grid; do
      PDIR=$([ "$PROJ" = "grid" ] && echo grid || echo noproj)
      for REL in 005 010 020; do
        src="$RUN/variants/${PROJ}_${REL_TAG}/Grouped_MGFs_rel${REL}"
        dst="$DST/$SP/$POST/$PDIR/rel$REL"
        if [ ! -d "$src" ]; then
          echo "  缺源: $SP/$POST/$PDIR/rel$REL"; n_miss=$((n_miss+1)); continue
        fi
        mkdir -p "$dst"
        for f in cosine_summary.csv cosine_detail.csv; do
          if [ -f "$src/$f" ]; then
            if [ "$MOVE" = "1" ]; then mv -f "$src/$f" "$dst/$f"; else cp -f "$src/$f" "$dst/$f"; fi
            n_cp=$((n_cp+1))
          else
            echo "  缺文件: $SP/$POST/$PDIR/rel$REL/$f"; n_miss=$((n_miss+1))
          fi
        done
      done
    done
  done
done

echo "===== 归档完成: 复制 $n_cp 个文件（应为 216 = 9物种×2后处理×2投影×3档×2文件），缺失 $n_miss 处"
echo "目标根: $DST"
[ "$n_miss" -gt 0 ] && echo "!!! 有缺失，说明源里还有没跑完的档，先补跑再归档"

# 概览
echo
echo "===== 目录概览 ====="
find "$DST" -name "cosine_*.csv" | wc -l | sed 's/^/  文件总数: /'
du -sh "$DST" | sed 's/^/  占用: /'
