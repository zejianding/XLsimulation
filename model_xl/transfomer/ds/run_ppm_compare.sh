#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# Compare PPM10 vs PPM5 on the same spectra with the NEW 6-branch scoring.
#
# Reuses the existing variants FASTA from step2 (ppm-independent).  For each
# ppm value runs step3 (match) + step4 (score, 6 branches) into separate
# output dirs output/<sp>/ppm<PPM10> and output/<sp>/ppm<PPM5>, then prints a
# side-by-side comparison of top1_acc and median_best_minus_gt_score.
#
# Env overrides:
#   V1_SEARCH_ROOT    data root (default /home/wangyuhang/seq2mass/v1-search)
#   CKPT              checkpoint (default runs/v200701_evidence_v2_massivekb/best_model.pt)
#   SPECIES           (default mouse)
#   MAX_SCANS         (default 300)
#   PPM10 / PPM5      (default 10 / 5)
#   SAVE_PREDICTIONS  "1" to also save predicted grids+spectra (default 1)
#   REL_INTENSITIES   comma-separated predicted-side filter levels for step4
#                     (default "0.05"; e.g. "0.05,0.10,0.15,0.20")
#   GPU / DEVICE / BATCH_SIZE
#
# Usage (server):
#   bash run_ppm_compare.sh
# ============================================================================
set -euo pipefail

V1_SEARCH_ROOT="${V1_SEARCH_ROOT:-/home/wangyuhang/seq2mass/v1-search}"
CKPT="${CKPT:-/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt}"
DS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECIES="${SPECIES:-mouse}"
MAX_SCANS="${MAX_SCANS:-300}"
PPM10="${PPM10:-10}"
PPM5="${PPM5:-5}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
REL_INTENSITIES="${REL_INTENSITIES:-0.05}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"

FASTA="$DS_DIR/data/$SPECIES/variants_${SPECIES}.fasta"
[ -f "$FASTA" ] || { echo "ERROR: variants FASTA missing: $FASTA (run step2 first)" >&2; exit 1; }

SP_ARGS=()
if [ "$SAVE_PREDICTIONS" = "1" ]; then
  SP_ARGS+=(--save-predictions)
fi

run_one() {
  local ppm="$1" out="$2"
  echo ""
  echo "===== PPM=$ppm ====="
  python "$DS_DIR/step3_match_by_mass.py" \
    --species "$SPECIES" --root "$V1_SEARCH_ROOT" \
    --fasta "$FASTA" \
    --out-candidates "$DS_DIR/data/$SPECIES/candidates_${SPECIES}_ppm${ppm}.tsv.gz" \
    --out-dir "$DS_DIR/output/$SPECIES" \
    --ppm "$ppm" ${MAX_SCANS:+--max-scans "$MAX_SCANS"}
  python "$DS_DIR/step4_predict_score.py" \
    --species "$SPECIES" --root "$V1_SEARCH_ROOT" \
    --candidates "$DS_DIR/data/$SPECIES/candidates_${SPECIES}_ppm${ppm}.tsv.gz" \
    --checkpoint "$CKPT" \
    --out-dir "$out" \
    --batch-size "$BATCH_SIZE" --gpu "$GPU" \
    --rel-intensities "$REL_INTENSITIES" \
    ${DEVICE:+--device "$DEVICE"} \
    ${MAX_SCANS:+--max-scans "$MAX_SCANS"} \
    "${SP_ARGS[@]}"
}

mkdir -p "$DS_DIR/output/$SPECIES"
run_one "$PPM10" "$DS_DIR/output/$SPECIES/ppm$PPM10"
run_one "$PPM5"  "$DS_DIR/output/$SPECIES/ppm$PPM5"

echo ""
echo "===== comparison: top1_acc and best_minus_gt score gap ====="
python - "$DS_DIR/output/$SPECIES" "$SPECIES" "$PPM10" "$PPM5" <<'PY'
import json, sys

base, sp, p10, p5 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


def load(p):
    try:
        with open(f"{base}/ppm{p}/summary_step4_{sp}.json", encoding="utf-8") as fh:
            return json.load(fh)["branches"]
    except Exception as exc:
        return {"__error__": str(exc)}


b10, b5 = load(p10), load(p5)
print(f"{'branch':<16}{'ppm'+p10+' top1':>12}{'ppm'+p5+' top1':>12}"
      f"{'ppm'+p10+' gap':>12}{'ppm'+p5+' gap':>12}")
for b in b10:
    if b == "__error__":
        continue
    g10 = b10[b].get("median_best_minus_gt_score")
    b5b = b5.get(b, {})
    g5 = b5b.get("median_best_minus_gt_score")
    t5 = b5b.get("top1_acc", 0.0)
    print(f"{b:<16}{b10[b]['top1_acc']:>12.4f}{t5:>12.4f}"
          f"{('' if g10 is None else f'{g10:.4f}'):>12}"
          f"{('' if g5 is None else f'{g5:.4f}'):>12}")
PY
