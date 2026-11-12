#!/usr/bin/env bash
# Thin Bash launcher. Python owns stage validation, caching and atomic commits.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT_DIR:?Set a NEW output root for the corrected pipeline}"
if [[ -n "${MAX_SCANS:-}" ]]; then
  echo "MAX_SCANS is removed. Use LIMIT_GROUPS to retain complete groups." >&2
  exit 2
fi
args=(--out-dir "$OUT_DIR" --n-groups "${N_GROUPS:-30000}"
      --min-spectra "${MIN_SPECTRA:-2}" --prefer-spectra "${PREFER_SPECTRA:-3}"
      --seed "${SEED:-20261010}" --threads "${THREADS:-0}" --workers "${WORKERS:-8}"
      --gpu "${GPU:-0}" --tol-filter "${TOL_FILTER:-0.05}" --tol-average "${TOL_AVERAGE:-0.05}"
      --domains "${INTENSITY_TRANSFORM:-none,sqrt_clip}" --limit-groups "${LIMIT_GROUPS:-0}"
      --stop-after "${STOP_AFTER:-full}")
if [[ -n "${SAMPLE_DIR:-}" ]]; then args+=(--sample-dir "$SAMPLE_DIR");
elif [[ "${SKIP_SAMPLE:-0}" == 1 ]]; then
  echo "SKIP_SAMPLE=1 requires SAMPLE_DIR with a validated schema-v2 sample." >&2; exit 2
else args+=(--train-mgf-xz "${TRAIN_MGF_XZ:?Set TRAIN_MGF_XZ or SAMPLE_DIR}"); fi
if [[ -n "${PIPRIME_MODEL:-}" ]]; then args+=(--model "$PIPRIME_MODEL"); fi
if [[ -n "${PROSIT_CACHE:-}" ]]; then args+=(--prosit-cache "$PROSIT_CACHE"); fi
if [[ -n "${PROSIT_MGF:-}" ]]; then args+=(--prosit-mgf "$PROSIT_MGF"); fi
if [[ "${SKIP_PROSIT:-0}" == 1 ]]; then args+=(--skip-prosit); fi
if [[ "${FORCE:-0}" == 1 ]]; then args+=(--force); fi
export PYTHONIOENCODING=utf-8
exec "${PYTHON:-python}" "$HERE/run_pipeline.py" "${args[@]}" "$@"
