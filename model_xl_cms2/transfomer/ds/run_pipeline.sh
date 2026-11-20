#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# DS pipeline one-shot runner: step1 -> step2 -> step3 -> step4 for species list.
#
# All env vars have server-friendly defaults; override freely.
#
#   V1_SEARCH_ROOT  v1-search data root (default /home/wangyuhang/seq2mass/v1-search)
#   CKPT            V20 checkpoint   (default /home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt)
#   OUTPUT_DIR      output root      (default = this ds/ dir)
#   SPECIES         comma-separated species list (default mouse; 'all' = 9 species)
#   GPU             CUDA device index (default 0)
#   DEVICE          explicit device override, e.g. cpu / cuda:1 (empty = auto)
#   MAX_SCANS       cap spectra processed in step3/step4 (empty = all 10000)
#   PPM             mass window ppm for step3 (default 10)
#   PPM_OFFSET      systematic ppm re-centering for step3 (default 0)
#   BATCH_SIZE      step4 prediction batch size (default 128)
#   NOPACKED        set to 1 to pass --no-packed to step4
#   SAVE_PREDICTIONS  set to 1 to save per-branch best-candidate predicted
#                     grid (39,46,6) + spectrum MGF under output/<sp>/predictions/
#   PRED_DTYPE      saved-grid dtype (default float16; float32 for full precision)
#   REL_INTENSITIES step4 predicted-side filter levels (default 0.05;
#                     production "0.15,0.20")
#   MINIMAL         set to 1 (default) to run the trimmed production branch
#                     set only: d_spec/f_frag/m_by_prec per level + hs_d<R>_f<R>
#                     + hs_d<R>_f<R>_mby.  All other branches are dropped.
#   CROSS_FUSE      set to 1 to add hs_d<l1>_f<l2> cross-level fusions (non-minimal)
#   FUSION_NORM     minmax (default) or zscore for score-avg fusions
#   EXP_TRANSFORM   map experimental intensities into the model's training
#                     domain: sqrt_clip (default, V20 training
#                     sqrt(min(x,50000))/sqrt(50000)), none (raw), sqrt
#   SKIP_STEP1      set to 1 to reuse an existing configs/mods_config.json
#
# Example (full mouse run on server GPU):
#   cd /home/wangyuhang/seq2mass/v20_0701/transfomer/ds
#   bash run_pipeline.sh
#
# Example (mouse pilot, 300 spectra, CPU):
#   SPECIES=mouse MAX_SCANS=300 DEVICE=cpu bash run_pipeline.sh
# ============================================================================
set -euo pipefail

V1_SEARCH_ROOT="${V1_SEARCH_ROOT:-/home/wangyuhang/seq2mass/v1-search}"
CKPT="${CKPT:-/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt}"
DS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$DS_DIR}"
ALL_SPECIES="bacillus,clambacteria,honeybee,human,mmazei,mouse,ricebean,tomato,yeast"
SPECIES="${SPECIES:-mouse}"
[ "$SPECIES" = "all" ] && SPECIES="$ALL_SPECIES"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-}"
MAX_SCANS="${MAX_SCANS:-}"
PPM="${PPM:-10}"
PPM_OFFSET="${PPM_OFFSET:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NOPACKED="${NOPACKED:-0}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
PRED_DTYPE="${PRED_DTYPE:-float16}"
SAVE_BRANCHES="${SAVE_BRANCHES:-hs_d020_f020_mby}"
REL_INTENSITIES="${REL_INTENSITIES:-0.05}"
CROSS_FUSE="${CROSS_FUSE:-0}"
FUSION_NORM="${FUSION_NORM:-minmax}"
MINIMAL="${MINIMAL:-1}"
SKIP_STEP1="${SKIP_STEP1:-0}"
EXP_TRANSFORM="${EXP_TRANSFORM:-sqrt_clip}"

CONFIG_DIR="$OUTPUT_DIR/configs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[ -d "$V1_SEARCH_ROOT" ] || { echo "ERROR: V1_SEARCH_ROOT not found: $V1_SEARCH_ROOT" >&2; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: CKPT not found: $CKPT" >&2; exit 1; }
command -v python >/dev/null || { echo "ERROR: python not on PATH" >&2; exit 1; }

mkdir -p "$CONFIG_DIR"

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  log "===== species: $sp ====="
  mkdir -p "$OUTPUT_DIR/output/$sp" "$OUTPUT_DIR/data/$sp"

  if [ "$SKIP_STEP1" != "1" ] || [ ! -f "$CONFIG_DIR/mods_config.json" ]; then
    log "[step1] scan MGF modifications: $sp"
    python "$DS_DIR/step1_scan_mgf_mods.py" \
      --species "$sp" --root "$V1_SEARCH_ROOT" \
      --out-dir "$OUTPUT_DIR/output/$sp" --config-dir "$CONFIG_DIR"
  else
    log "[step1] SKIP (mods_config.json exists)"
  fi

  log "[step2] build candidate variant FASTA: $sp"
  python "$DS_DIR/step2_build_candidate_fasta.py" \
    --species "$sp" --root "$V1_SEARCH_ROOT" \
    --mods-config "$CONFIG_DIR/mods_config.json" \
    --out-fasta "$OUTPUT_DIR/data/$sp/variants_${sp}.fasta" \
    --out-dir "$OUTPUT_DIR/output/$sp"

  log "[step3] match MGF spectra to candidates by mass: $sp"
  python "$DS_DIR/step3_match_by_mass.py" \
    --species "$sp" --root "$V1_SEARCH_ROOT" \
    --fasta "$OUTPUT_DIR/data/$sp/variants_${sp}.fasta" \
    --out-candidates "$OUTPUT_DIR/data/$sp/candidates_${sp}_scans.tsv.gz" \
    --out-dir "$OUTPUT_DIR/output/$sp" \
    --ppm "$PPM" ${PPM_OFFSET:+--ppm-offset "$PPM_OFFSET"} \
    ${MAX_SCANS:+--max-scans "$MAX_SCANS"}

  log "[step4] predict + score (4 branches): $sp"
  python "$DS_DIR/step4_predict_score.py" \
    --species "$sp" --root "$V1_SEARCH_ROOT" \
    --candidates "$OUTPUT_DIR/data/$sp/candidates_${sp}_scans.tsv.gz" \
    --checkpoint "$CKPT" \
    --out-dir "$OUTPUT_DIR/output/$sp" \
    --batch-size "$BATCH_SIZE" --gpu "$GPU" \
    --rel-intensities "$REL_INTENSITIES" --fusion-norm "$FUSION_NORM" \
    ${EXP_TRANSFORM:+--exp-transform "$EXP_TRANSFORM"} \
    $( [ "$MINIMAL" = "1" ] && echo --minimal ) \
    $( [ "$CROSS_FUSE" = "1" ] && echo --cross-fuse ) \
    ${DEVICE:+--device "$DEVICE"} \
    ${MAX_SCANS:+--max-scans "$MAX_SCANS"} \
    $( [ "$NOPACKED" = "1" ] && echo --no-packed ) \
    $( [ "$SAVE_PREDICTIONS" = "1" ] && echo --save-predictions --pred-dtype "$PRED_DTYPE" --save-branches "$SAVE_BRANCHES" )
done

log "===== aggregate summary ====="
python - "$OUTPUT_DIR" "$SPECIES" <<'PY'
import json, sys

out_root, species_csv = sys.argv[1], sys.argv[2]
result = {}
for sp in species_csv.split(","):
    f = f"{out_root}/output/{sp}/summary_step4_{sp}.json"
    try:
        with open(f, encoding="utf-8") as fh:
            result[sp] = json.load(fh)
    except Exception as exc:
        result[sp] = {"error": str(exc)}
out_path = f"{out_root}/output/summary_all_species.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)
print(f"aggregate -> {out_path}")
for sp, d in result.items():
    if "error" in d:
        print(f"  {sp}: {d['error']}")
        continue
    acc = {b: d.get("branches", {}).get(b, {}).get("top1_acc")
           for b in ("d_spec_rel020", "f_frag_rel020", "m_by_prec_rel020",
                     "hs_d020_f020", "hs_d020_f020_mby")}
    print(f"  {sp}: d={acc['d_spec_rel020']} f={acc['f_frag_rel020']} "
          f"m_by_prec={acc['m_by_prec_rel020']} hsdf={acc['hs_d020_f020']} "
          f"hsdf_mby={acc['hs_d020_f020_mby']}")
PY

log "===== all done ====="
