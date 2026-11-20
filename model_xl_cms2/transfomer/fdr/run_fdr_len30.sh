#!/usr/bin/env bash
# CRLF 自愈：从 Windows 拷来的脚本可能带 \r，会让下面的 set 行报 "invalid option name"。
# 检测到自己含 \r 就先转成 LF 再重新执行一次。行尾的 " # CRLF" 用来吃掉可能存在的 \r，勿删。
if grep -q $'\r' "${BASH_SOURCE[0]}"; then sed -i 's/\r$//' "${BASH_SOURCE[0]}"; exec bash "${BASH_SOURCE[0]}" "$@"; fi  # CRLF
# ============================================================================
# ≤30aa 口径的 1%FDR —— 只重跑 step4，不碰 step1/2/3，不需要 GPU。
#
# 为什么只跑 step4 就够（不需要 --full / --max-len）：
#   一个谱的候选池是长度同质的：step2 的 ±10ppm 窗 vs 相邻残基至少 57.02Da(Gly)，
#   57Da 对应几万 ppm，远超 10ppm。所以 GT 骨架 ≤30 的谱，池里也全是 ≤30 的候选；
#   反转 decoy 长度不变，同样 ≤30。于是 step4_fdr_curve.py --max-gt-len 30 的
#   "只按 gt_seq 骨架长度筛行" 就等价于完整的 ≤30aa 口径。
#   （v20_0918/transfomer/fdr 自带的 V20 说明.md 也是这个结论。）
#
# 产物与全量结果**完全分开存放**，绝不覆盖 output/ 里的全量 summary_fdr 与 fdr_curve：
#   <OUTPUT_DIR>/output_len30/<sp>/summary_fdr_<sp>.json
#   <OUTPUT_DIR>/output_len30/<sp>/fdr_curve_<sp>_<branch>.csv
#   <OUTPUT_DIR>/output_len30/summary_fdr_all_species_len30.json
#   <OUTPUT_DIR>/output_len30/fdr_len30_九物种汇总.csv      长表：全量 vs ≤30 并列
#
# env（都有默认）：
#   OUTPUT_DIR   与 run_fdr.sh 同一个根（默认 = 本 fdr/ 目录，需与上次跑全量时一致）
#   SPECIES      逗号分隔（默认九物种）
#   MAX_GT_LEN   骨架长度上限（默认 30）
#   FDR_TARGETS  默认 0.01,0.05,0.10
#   FORCE_RERUN  1 = 已有 ≤30 summary 也重算
# ============================================================================
set -uo pipefail

FDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$FDR_DIR}"
SPECIES="${SPECIES:-mouse,human,yeast,bacillus,clambacteria,honeybee,mmazei,ricebean,tomato}"
MAX_GT_LEN="${MAX_GT_LEN:-30}"
FDR_TARGETS="${FDR_TARGETS:-0.01,0.05,0.10}"
FORCE_RERUN="${FORCE_RERUN:-0}"

OUT_LEN="$OUTPUT_DIR/output_len30"
log() { echo "[$(date '+%F%T')] $*"; }

[ -f "$FDR_DIR/step4_fdr_curve.py" ] || { echo "ERROR: 找不到 $FDR_DIR/step4_fdr_curve.py"; exit 1; }
[ -d "$OUTPUT_DIR/output" ] || { echo "ERROR: 找不到 $OUTPUT_DIR/output —— OUTPUT_DIR 要和跑全量时一致"; exit 1; }
mkdir -p "$OUT_LEN"

n_ok=0; n_skip=0; n_fail=0
for sp in $(echo "$SPECIES" | tr ',' ' '); do
  res="$OUTPUT_DIR/output/$sp/results_step3_${sp}.csv"
  if [ ! -f "$res" ]; then
    echo "skip $sp: 找不到 $res"; n_skip=$((n_skip+1)); continue
  fi
  sjson="$OUT_LEN/$sp/summary_fdr_${sp}.json"
  if [ "$FORCE_RERUN" != "1" ] && [ -f "$sjson" ]; then
    log "$sp 已有 ≤$MAX_GT_LEN summary，跳过"; n_skip=$((n_skip+1)); continue
  fi
  mkdir -p "$OUT_LEN/$sp"
  log "[$sp] step4 (max-gt-len=$MAX_GT_LEN)"
  if python "$FDR_DIR/step4_fdr_curve.py" \
       --species "$sp" \
       --results "$res" \
       --out-dir "$OUT_LEN/$sp" \
       --max-gt-len "$MAX_GT_LEN" \
       --fdr-targets "$FDR_TARGETS" \
       > "$OUT_LEN/$sp/step4_len30.log" 2>&1 ; then
    n_ok=$((n_ok+1))
  else
    echo "  !! 失败 -> $OUT_LEN/$sp/step4_len30.log（末尾 15 行）"
    tail -15 "$OUT_LEN/$sp/step4_len30.log" | sed 's/^/    /'
    n_fail=$((n_fail+1))
  fi
done

log "===== 汇总：全量 vs ≤$MAX_GT_LEN ====="
python - "$OUTPUT_DIR" "$OUT_LEN" "$SPECIES" "$FDR_TARGETS" "$MAX_GT_LEN" <<'PY'
import csv, json, sys
from pathlib import Path

out_root, out_len, species_csv, targets, max_gt = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
species = [s.strip() for s in species_csv.split(",") if s.strip()]
t1 = float(targets.split(",")[0])
tk = f"{t1:.2f}"


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


rows, agg = [], {}
for sp in species:
    full_p = out_root / "output" / sp / f"summary_fdr_{sp}.json"
    le_p = out_len / sp / f"summary_fdr_{sp}.json"
    if not le_p.exists():
        rows.append({"species": sp, "scope": "le", "error": "no summary"})
        continue
    le = load(le_p)
    full = load(full_p) if full_p.exists() else {}
    brs = le.get("branches", {}) or {}
    if not brs:
        rows.append({"species": sp, "scope": "le", "error": le.get("error", "no branches")})
        continue
    n_sp_le = next(iter(brs.values())).get("n_spectra")
    n_sp_full = next(iter((full.get("branches") or {}).values()), {}).get("n_spectra")
    best_k, best_n = None, -1
    for k, b in brs.items():
        line = (b.get("fdr_lines") or {}).get(tk, {})
        n = line.get("n_target")
        for scope, src, ns in (("full", full.get("branches", {}).get(k), n_sp_full),
                               ("le", b, n_sp_le)):
            ln = (src.get("fdr_lines") or {}).get(tk, {}) if src else {}
            rows.append({"species": sp, "scope": scope, "branch": k,
                         "fdr_target": tk, "n_spectra": ns,
                         "n_target": ln.get("n_target"), "n_decoy": ln.get("n_decoy"),
                         "n_target_gt_match": ln.get("n_target_gt_match"),
                         "fdr_dt_achieved": ln.get("fdr_dt_achieved")})
        if n is not None and n > best_n:
            best_k, best_n = k, n
    agg[sp] = {"n_spectra_le": n_sp_le, "n_spectra_full": n_sp_full,
               "best_branch": best_k, "best_n_target": best_n,
               "best_n_gt": (brs[best_k]["fdr_lines"][tk].get("n_target_gt_match")
                             if best_k else None),
               "branches": {k: {"n_target": (b.get("fdr_lines") or {}).get(tk, {}).get("n_target"),
                                "n_decoy": (b.get("fdr_lines") or {}).get(tk, {}).get("n_decoy"),
                                "n_target_gt_match": (b.get("fdr_lines") or {}).get(tk, {}).get("n_target_gt_match"),
                                "n_spectra": n_sp_le}
                            for k, b in brs.items()}}

(out_len / "summary_fdr_all_species_len30.json").write_text(
    json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

cols = ["species", "scope", "branch", "fdr_target", "n_spectra",
        "n_target", "n_decoy", "n_target_gt_match", "fdr_dt_achieved"]
with (out_len / "fdr_len30_九物种汇总.csv").open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        if "branch" in r:
            w.writerow(r)

print(f"\n{'species':>14} {'n<=%s' % max_gt:>7} {'n_all':>7}  best branch @{t1:.1%} (n_target / n_gt)")
for sp in species:
    a = agg.get(sp)
    if not a:
        print(f"{sp:>14}  (无结果)"); continue
    print(f"{sp:>14} {a['n_spectra_le']:>7} {a['n_spectra_full']:>7}  "
          f"{a['best_branch']} {a['best_n_target']}/{a['best_n_gt']}")

print(f"\n主要分支 n_target @{t1:.1%}   全量 -> <={max_gt}aa（差）")
KEY = ["d_f_exp_mby_rel005", "d_f_exp_mby_rel010", "d_f_exp_mby_rel020",
       "d_f_exp_rel005", "d_f_exp_rel010", "d_f_exp_rel020"]
full_all = {sp: (load(out_root / "output" / sp / f"summary_fdr_{sp}.json").get("branches") or {})
            for sp in species}
for k in KEY:
    cells = []
    for sp in species:
        le = (agg.get(sp, {}).get("branches") or {}).get(k, {}).get("n_target")
        fu = ((full_all.get(sp) or {}).get(k, {}).get("fdr_lines") or {}).get(tk, {}).get("n_target")
        cells.append("-" if le is None else (f"{le}" if fu is None else f"{fu}->{le}({le-fu:+d})"))
    print(f"  {k:>20}  " + "  ".join(cells))
print(f"\n汇总 JSON -> {out_len/'summary_fdr_all_species_len30.json'}")
print(f"长表 CSV -> {out_len/'fdr_len30_九物种汇总.csv'}")
PY

log "===== 完成: 成功 $n_ok / 跳过 $n_skip / 失败 $n_fail ====="
log "全量结果未改动：$OUTPUT_DIR/output/ ；≤${MAX_GT_LEN}aa 在 $OUT_LEN/"
exit 0
