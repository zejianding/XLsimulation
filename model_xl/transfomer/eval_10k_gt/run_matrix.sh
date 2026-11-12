#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# 结果矩阵驱动器：对一次已完成运行（含 Grouped_MGFs）批量产出多口径余弦结果。
# 轴：投影{raw,grid} × rel{0,0.05,0.10,0.20} × exp{sqrt_clip,(+none参考)} × pred{V20:none, Prosit:sqrt}
# 命名：variants/<proj>_<relTag>_exp<exp>_pred<pred>/Grouped_MGFs_rel<lvl>/{cosine_summary,cosine_detail}.csv
#
# env:
#   RUN_DIR     已完成运行目录（必填，含 Grouped_MGFs）
#   PROJ        raw,grid（默认 raw,grid；grid=先把原谱投影到 V20 网格再编码）
#   RELS        0,0.05,0.10,0.20
#   EXP         sqrt_clip | none（默认 sqrt_clip，对齐 FDR 域）
#   PRED        none（V20）| sqrt（Prosit 建议）
#   ADD_EXPNONE 1=再加 exp=none & rel=0 的“纯无处理”参考
#   TOP_K       预测谱峰数上限（0=不限）
#   GRID_PROJ_PRED 1=grid 档把预测谱也吸附到同一网格（默认 1，两侧对称）；
#                  0=只投影原谱、预测谱保持原样（两侧不同网格）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$DIR/.." && pwd)"
REPO_ROOT="$(cd "$TF_DIR/../.." && pwd)"
PIPRIME_MODEL="${PIPRIME_MODEL:-$REPO_ROOT/piprime_embedding/piprime_embedding/model.ckpt}"
PROJ="${PROJ:-raw,grid}"
RELS="${RELS:-0,0.05,0.10,0.20}"
EXP="${EXP:-sqrt_clip}"
PRED="${PRED:-none}"
ADD_EXPNONE="${ADD_EXPNONE:-1}"
TOP_K="${TOP_K:-0}"
RUN_DIR="${RUN_DIR:?need RUN_DIR=<finished run dir>}"
[ -d "$RUN_DIR/Grouped_MGFs" ] || { echo "no Grouped_MGFs under $RUN_DIR"; exit 2; }
VAR="$RUN_DIR/variants"; mkdir -p "$VAR"

embed_cosine() {   # $1 = dir holding Grouped_MGFs_rel*
  local d="$1"
  echo "===== $d"
  python "$REPO_ROOT/piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py" \
    --mgf_dir "$d" --output_dir "$d/Embeddings" --model "$PIPRIME_MODEL"
  python "$TF_DIR/batch_cosine_summary.py" --mgf-dir "$d" --npz-dir "$d/Embeddings" \
    --output-csv "$d/cosine_summary.csv"
  python "$DIR/cosine_detail_csv.py" --mgf-dir "$d" --npz-dir "$d/Embeddings" --out-dir "$d"
}

make_variants() {  # $1=src dir  $2=proj tag  $3=rels  $4=exp  $5=pred
  local src="$1" proj="$2" rels="$3" exp="$4" pred="$5"
  local outdir
  outdir=$(python "$DIR/filter_group_predictions.py" --mgf-dir "$src" --out-root "$VAR" \
             --rels "$rels" --top-k "$TOP_K" --exp-transform "$exp" --pred-transform "$pred" \
           | sed -n 's/^OUTDIR=//p')
  [ -n "$outdir" ] || { echo "filter_group_predictions produced no OUTDIR"; return 1; }
  local tagged="$VAR/${proj}_$(basename "$outdir")"
  [ -d "$tagged" ] && rm -rf "$tagged"
  mv "$outdir" "$tagged"
  for d in "$tagged"/Grouped_MGFs_rel*; do embed_cosine "$d"; done
}

for proj in $(echo "$PROJ" | tr ',' ' '); do
  if [ "$proj" = "grid" ]; then
    SRCDIR="$RUN_DIR/Grouped_MGFs_projected"
    # 每次都调用；脚本自己按 manifest 判断是复用还是重建。原来这里的
    # `[ ! -d "$SRCDIR" ]` 会把上一次中断留下的半成品当成完成品复用，
    # 导致 grid 那一档在残缺谱集上算完却不报错。
    echo "=== projecting spectra to V20 grid (GRID_PROJ_PRED=${GRID_PROJ_PRED:-1})"
    python "$TF_DIR/project_grouped_mgfs_to_v20_grid.py" \
      --input-dir "$RUN_DIR/Grouped_MGFs" --output-dir "$SRCDIR" \
      --min-mz 100 --match-tol 0.08 --aggregate max \
      $( [ "${GRID_PROJ_PRED:-1}" = "1" ] && echo --project-prediction )
  else
    SRCDIR="$RUN_DIR/Grouped_MGFs"
  fi
  make_variants "$SRCDIR" "$proj" "$RELS" "$EXP" "$PRED"
  [ "$ADD_EXPNONE" = "1" ] && make_variants "$SRCDIR" "$proj" "0" "none" "none"
done
echo "===== matrix done: $VAR"
