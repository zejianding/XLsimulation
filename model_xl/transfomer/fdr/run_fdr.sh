#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# FDR analysis one-shot runner: step1 (target+decoy FASTA) -> step2 (mass match)
# -> step3 (predict + gc/d/bidir scoring) -> step4 (FDR curve at 1%).
#
# Runs one species or a comma-separated list of species (or ALL of them), then
# aggregates every species' 1% FDR identification counts into one comparison.
#
# All env vars have server-friendly defaults; override freely.
#
#   V1_SEARCH_ROOT  v1-search data root (default /home/wangyuhang/seq2mass/v1-search)
#   CKPT            V20 checkpoint   (default /home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt)
#   OUTPUT_DIR      output root      (default = this fdr/ dir)
#   SPECIES         species or comma-separated species list (default mouse;
#                   'all' = bacillus,clambacteria,honeybee,human,mmazei,mouse,
#                   ricebean,tomato,yeast)
#   GPU             CUDA device index (default 0)
#   DEVICE          explicit device override, e.g. cpu / cuda:1 (empty = auto)
#   MAX_SCANS       cap spectra processed in step2/step3 per species (empty = all)
#   PPM             mass window ppm (default 10)
#   PPM_OFFSET      systematic ppm re-centering, applied to every species
#                   (default 0; ricebean had ~+3.7ppm bias — run it separately
#                   with PPM_OFFSET=<-gt_ppm.median> if needed)
#   BATCH_SIZE      step3 prediction batch size (default 128)
#   NOPACKED        set to 1 to pass --no-packed to step3
#   VARIANT_FASTA   optional prebuilt step2 variants FASTA to use as the target
#                   set verbatim (empty = build targets from the species FASTA)
#   NO_MOX          set to 1 to build targets with plain C+57.021 only (no M-ox
#                   variants) — fast iteration; default 0 = full M-ox variants
#   MAX_VAR_MODS    max M oxidation sites per peptide when enumerating (default 3)
#   REL             comma-separated predicted-side relative-intensity filters
#                   (default 0.05,0.10,0.20).  Each (branch, rel) becomes its own
#                   FDR branch keyed <branch>_rel<lvl> (e.g. d_f_exp_rel020), all
#                   from the SAME GPU prediction pass.
#   BRANCHES        ';'-separated scoring branches; each is a comma-separated
#                   subset of gc,d,f_bidir,f_exp,mby (default includes d_gc_f_exp,
#                   the current best branch).  All branches share the SAME GPU
#                   prediction, so adding branches costs no extra model forward.
#   EXP_TRANSFORM   none | sqrt_clip (default) | sqrt
#   EXP_TOP_PEAKS   keep top-N experimental peaks before scoring (default 0 = all)
#   FDR_TARGETS     FDR lines to report (default 0.01,0.05,0.10)
#   SAVE_DETAIL     set to 1 to write a combined per-candidate component CSV in step3
#   SAVE_FULL       default 1 = always write one per-candidate FULL CSV.gz PER BRANCH
#                   (full_<branch>_<sp>.csv.gz: spectrum info + seq/label/ppm/mass
#                   + all 5 component scores + branch fused + rank + is_best + gt_match)
#                   -> enables instant recompute of any new branch via
#                   step4_fdr_curve.py --full ... --branches '...'
#                   (set to 0 to skip; ~40-60 MB gz per branch per species)
#
# Examples (server GPU):
#   cd /home/wangyuhang/seq2mass/v20_0821/transfomer/fdr
#   bash run_fdr.sh                       # mouse only
#   SPECIES=all bash run_fdr.sh           # all 9 species (~22h GPU step3 total)
#   MAX_SCANS=300 SPECIES=all bash run_fdr.sh   # fast pilot across all species
#   BRANCHES=d,gc,f_exp bash run_fdr.sh   # just the production branch
# ============================================================================
set -euo pipefail

ALL_SPECIES="bacillus,clambacteria,honeybee,human,mmazei,mouse,ricebean,tomato,yeast"

V1_SEARCH_ROOT="${V1_SEARCH_ROOT:-/home/wangyuhang/seq2mass/v1-search}"
CKPT="${CKPT:-/home/wangyuhang/seq2mass/runs/v200701_evidence_v2_massivekb/best_model.pt}"
FDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$FDR_DIR}"
SPECIES="${SPECIES:-mouse}"
[ "$SPECIES" = "all" ] && SPECIES="$ALL_SPECIES"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-}"
MAX_SCANS="${MAX_SCANS:-}"
PPM="${PPM:-10}"
PPM_OFFSET="${PPM_OFFSET:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NOPACKED="${NOPACKED:-0}"
VARIANT_FASTA="${VARIANT_FASTA:-}"
NO_MOX="${NO_MOX:-0}"
MAX_VAR_MODS="${MAX_VAR_MODS:-3}"
REL="${REL:-0.05,0.10,0.20}"
BRANCHES="${BRANCHES:-d;gc;f_bidir;gc,f_exp;d,gc,f_exp;d,f_exp;d,f_bidir;d,gc,f_bidir;mby;d,f_exp,mby}"
EXP_TRANSFORM="${EXP_TRANSFORM:-sqrt_clip}"
EXP_TOP_PEAKS="${EXP_TOP_PEAKS:-0}"
FDR_TARGETS="${FDR_TARGETS:-0.01,0.05,0.10}"
SAVE_DETAIL="${SAVE_DETAIL:-0}"
SAVE_FULL="${SAVE_FULL:-1}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[ -d "$V1_SEARCH_ROOT" ] || { echo "ERROR: V1_SEARCH_ROOT not found: $V1_SEARCH_ROOT" >&2; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: CKPT not found: $CKPT" >&2; exit 1; }
command -v python >/dev/null || { echo "ERROR: python not on PATH" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR/data" "$OUTPUT_DIR/output"

for sp in $(echo "$SPECIES" | tr ',' ' '); do
  mkdir -p "$OUTPUT_DIR/data/$sp" "$OUTPUT_DIR/output/$sp"
  log "===== species: $sp ====="

  log "[step1] build target+decoy FASTA: $sp"
  python "$FDR_DIR/step1_build_td_fasta.py" \
    --species "$sp" --root "$V1_SEARCH_ROOT" \
    --out-fasta "$OUTPUT_DIR/data/$sp/${sp}_td.fasta" \
    --out-dir "$OUTPUT_DIR/output/$sp" \
    --max-variable-mods "$MAX_VAR_MODS" \
    ${VARIANT_FASTA:+--variants-fasta "$VARIANT_FASTA"} \
    $( [ "$NO_MOX" = "1" ] && echo --no-mox )

  log "[step2] match MGF spectra to target/decoy candidates by mass: $sp"
  python "$FDR_DIR/step2_match_td.py" \
    --species "$sp" --root "$V1_SEARCH_ROOT" \
    --fasta "$OUTPUT_DIR/data/$sp/${sp}_td.fasta" \
    --out-candidates "$OUTPUT_DIR/data/$sp/candidates_${sp}_td.tsv.gz" \
    --out-dir "$OUTPUT_DIR/output/$sp" \
    --ppm "$PPM" ${PPM_OFFSET:+--ppm-offset "$PPM_OFFSET"} \
    ${MAX_SCANS:+--max-scans "$MAX_SCANS"}

  log "[step3] predict + score (branches): $sp"
  python "$FDR_DIR/step3_score_fdr.py" \
    --species "$sp" --root "$V1_SEARCH_ROOT" \
    --candidates "$OUTPUT_DIR/data/$sp/candidates_${sp}_td.tsv.gz" \
    --checkpoint "$CKPT" \
    --out-dir "$OUTPUT_DIR/output/$sp" \
    --batch-size "$BATCH_SIZE" --gpu "$GPU" \
    --rel "$REL" --branches "$BRANCHES" \
    --exp-transform "$EXP_TRANSFORM" --exp-top-peaks "$EXP_TOP_PEAKS" \
    ${DEVICE:+--device "$DEVICE"} \
    ${MAX_SCANS:+--max-scans "$MAX_SCANS"} \
    $( [ "$NOPACKED" = "1" ] && echo --no-packed ) \
    $( [ "$SAVE_DETAIL" = "1" ] && echo --save-detail ) \
    $( [ "$SAVE_FULL" = "1" ] && echo --save-full )

  log "[step4] FDR curves per branch: $sp"
  python "$FDR_DIR/step4_fdr_curve.py" \
    --species "$sp" --out-dir "$OUTPUT_DIR/output/$sp" \
    --fdr-targets "$FDR_TARGETS"

  log "===== done: $sp ====="
done

log "===== aggregate across species ====="
python - "$OUTPUT_DIR" "$SPECIES" "$FDR_TARGETS" <<'PY'
import json, sys

out_root, species_csv, targets = sys.argv[1], sys.argv[2], sys.argv[3]
t1 = float(targets.split(",")[0])
result = {}
for sp in species_csv.split(","):
    f = f"{out_root}/output/{sp}/summary_fdr_{sp}.json"
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as exc:
        result[sp] = {"error": str(exc)}
        continue
    brs = d.get("branches", {}) or {}
    row = {"n_spectra": None, "branches": {}}
    if brs:
        row["n_spectra"] = next(iter(brs.values())).get("n_spectra")
        for k, b in brs.items():
            line = (b.get("fdr_lines") or {}).get(f"{t1:.2f}") or (b.get("fdr_lines") or {}).get("0.01") or {}
            row["branches"][k] = {"n_target": line.get("n_target"),
                                  "n_gt": line.get("n_target_gt_match")}
    best_k, best_n = None, -1
    for k, v in row["branches"].items():
        if v.get("n_target") is not None and v["n_target"] > best_n:
            best_k, best_n = k, v["n_target"]
    row["best_branch"] = best_k
    row["best_n_target"] = best_n
    row["best_n_gt"] = row["branches"][best_k]["n_gt"] if best_k else None
    result[sp] = row

out_path = f"{out_root}/output/summary_fdr_all_species.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, ensure_ascii=False, indent=2)
print(f"aggregate -> {out_path}")

def fmt(v):
    return f"{v}" if v is not None else "-"

print(f"\n{'species':>14} {'spectra':>8}  best branch @{t1:.1%} (n_target/n_gt)")
for sp, row in result.items():
    if "error" in row:
        print(f"{sp:>14}  ERROR {row['error']}")
        continue
    bb = row.get("best_branch", "-")
    print(f"{sp:>14} {str(row.get('n_spectra')):>8}  {bb} "
          f"{fmt(row.get('best_n_target'))}/{fmt(row.get('best_n_gt'))}")
print("\nper-branch detail (all <branch>_rel<lvl> keys at 1% line) -> summary_fdr_all_species.json")
PY

log "===== all done ====="
