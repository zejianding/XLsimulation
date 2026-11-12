#!/usr/bin/env python3
"""Prosit-based DS search accuracy.

Predicts each candidate's fragment spectrum with Prosit (Koina
`Prosit_2025_intensity_MultiFrag`), keeps only b/y ions using Prosit's own
`annotation` column (each b/y fragment's intensity is placed at its
mod-corrected theoretical m/z), and scores each candidate against the
experimental spectrum.  Two scoring modes (--scoring):

  branches (default) - the SAME branch machinery as the V20 pipeline:
      d_spec/f_frag/m_by_prec per rel level + hs_d<lvl>_f<lvl> + hs_d<lvl>_f<lvl>_mby,
      so the predictor comparison is fair (identical evaluation harness).  Downstream
      users filter to the d/f/hsdf columns they care about.
  cosine             - a single spectral cosine between Prosit's b/y peaks and
      the experimental peaks.

Experimental intensities are mapped with --exp-transform (default sqrt_clip,
matching the V20 training domain); common data (step3 candidates TSV, MGF,
SEQ= ground truth, mass matching) is reused as-is.

Modification handling (--mods-mode):
  reanchor (default) - Prosit only accepts unmodified sequences, so we predict
      the backbone, use Prosit's `annotation` to identify each b/y fragment,
      and place its intensity at the MOD-corrected theoretical m/z.
  native             - pass the annotated sequence straight to Prosit (only if
                       the Koina model supports modified sequences).
  auto               - probe the schema; native if supported, else reanchor.

Outputs:
  <out-dir>/results_prosit_<sp>.csv
  <out-dir>/summary_prosit_<sp>.json
  <out-dir>/predictions/prosit_cache.tsv.gz   all Koina predictions (resumable)
  <out-dir>/records/record_prosit_<sp>.json

Usage:
    python step4_prosit_score.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --candidates data/mouse/candidates_mouse_full.tsv.gz \
        --out-dir output/prosit/mouse --max-scans 300 --scoring branches \
        --rel-intensities "0.05,0.10,0.20" --mods-mode reanchor
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np

# pandas MUST be imported before ds_common: importing predict_mgf registers a
# numpy._core alias that breaks a later `import pandas` on some numpy 1.x builds.
import pandas as pd

import ds_common as dc
from predict_mgf import generate_by_ions

PROSIT_MODEL = "Prosit_2025_intensity_MultiFrag"
PROSIT_HOST = "koina.wilhelmlab.org:443"
PROSIT_MAX_LEN = 30    # multifrag25 input 'intseq' is [-1,30]; longer backbones make
                       # Koina fail deterministically ("all input arrays must have
                       # the same shape" / unexpected shape) -> skip them up front.
MZ_TOL = 0.05          # experimental matching tolerance (Da), same as the DS pipeline


# ---------------------------------------------------------------------------
# sequence / ion helpers
# ---------------------------------------------------------------------------

def strip_mods(annoseq: str) -> str:
    """Plain AA backbone of an annotated sequence (mod annotations removed)."""
    residues, _, _ = dc.parse_modified_sequence(annoseq)
    return "".join(residues)


def ion_lookup(annoseq: str) -> dict[tuple, float]:
    """{(ion_type, fragment_index, charge): m/z} keyed the way Prosit annotates.

    generate_by_ions reports b/y both under the N-side cleavage index ``i``.
    Prosit labels b fragments ``b<i>`` (N-side) but y fragments ``y<k>`` from the
    C-terminus with ``k = n - i``, so y keys must be remapped or every y peak is
    placed at a wrong (mirrored) m/z.
    """
    ions = generate_by_ions(annoseq, max_charge=3)
    n = max(int(ion["cleavage_index"]) for ion in ions
            if ion["ion_type"] == "B") + 1 if ions else 0
    out: dict[tuple, float] = {}
    for ion in ions:
        i = int(ion["cleavage_index"])
        z = int(ion["charge"])
        idx = i if ion["ion_type"] == "B" else (n - i)
        out[(ion["ion_type"], idx, z)] = float(ion["mz"])
    return out


def parse_annotation(anno: str):
    """Prosit annotation 'b10' / 'y10^2' -> ('B'/'Y', cleavage_index, charge) or None.

    Only b/y are kept; other ion types (a, c, x, z, ...) return None.
    """
    m = re.match(r"^([a-zA-Z])(\d+)(?:\^(\d+))?$", str(anno).strip())
    if not m:
        return None
    typ = m.group(1).upper()
    if typ not in ("B", "Y"):
        return None
    return (typ, int(m.group(2)), int(m.group(3) or 1))


def merge_peaks(peaks: np.ndarray, mz_merge_tol: float = 0.05) -> np.ndarray:
    """Collapse peaks within ``mz_merge_tol`` Da keeping the strongest one."""
    if peaks.size == 0:
        return peaks
    p = peaks[np.argsort(peaks[:, 0])]
    merged: list[np.ndarray] = []
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and p[j, 0] - p[i, 0] < mz_merge_tol:
            j += 1
        best = i + int(np.argmax(p[i:j, 1]))
        merged.append(p[best])
        i = j
    return np.array(merged, dtype=np.float32)


def build_candidate_peaks(prosit_peaks: np.ndarray, annos, annotated: str,
                          mods_mode: str, tol: float = 0.1) -> np.ndarray:
    """Turn Prosit's raw fragment peaks into the candidate's b/y peak list.

    Preferred path: use Prosit's ``annotation`` column (e.g. 'b10', 'y10^2') to
    identify each fragment as a b/y ion, then place its intensity at the
    MOD-corrected theoretical m/z (handles C+57.021 / M+15.995 exactly).
    Fallback (no annotation): match by m/z to the unmodified theoretical b/y
    positions (reanchor) or annotated positions (native) within ``tol``.
    """
    if prosit_peaks is None or prosit_peaks.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    lookup = ion_lookup(annotated)  # mod-corrected theoretical b/y positions
    out: list[tuple[float, float]] = []
    if annos is not None and len(annos) == len(prosit_peaks):
        for (mz, inten), anno in zip(prosit_peaks, annos):
            lbl = parse_annotation(anno)
            if lbl is None:
                continue
            mod_mz = lookup.get(lbl)
            if mod_mz is None:
                continue
            out.append((mod_mz, float(inten)))
        return merge_peaks(np.array(sorted(out), dtype=np.float32))

    # fallback: m/z matching (no annotation available)
    if mods_mode == "native":
        match_lookup = lookup
    else:
        backbone = strip_mods(annotated)
        match_lookup = ion_lookup(backbone)
    items = sorted(match_lookup.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    ion_mz = np.array([v for _, v in items], dtype=np.float64)
    best = match_peaks_to_ions(prosit_peaks[:, 0], prosit_peaks[:, 1], ion_mz, labels, tol)
    out = [(lookup[label], inten) for label, inten in best.items()]
    return merge_peaks(np.array(sorted(out), dtype=np.float32))


def match_peaks_to_ions(peak_mz, peak_int, ion_mz, ion_labels, tol: float = 0.1) -> dict:
    """Match predicted peaks to the nearest theoretical ion within ``tol``.

    ``ion_mz`` must be ascending (labels aligned).  Each ion keeps the maximum
    predicted intensity among the peaks that match it.  Returns {label: intensity}.
    """
    best: dict = {}
    for mz, inten in zip(peak_mz, peak_int):
        idx = bisect_left(ion_mz, float(mz))
        for cand in (idx - 1, idx):
            if 0 <= cand < len(ion_mz):
                d = abs(ion_mz[cand] - float(mz))
                if d <= tol:
                    label = ion_labels[cand]
                    cur = best.get(label)
                    if cur is None or float(inten) > cur[1]:
                        best[label] = (d, float(inten))
    return {k: v[1] for k, v in best.items()}


# ---------------------------------------------------------------------------
# scoring (defined here for Prosit, not the V20 branch machinery)
# ---------------------------------------------------------------------------

def spectral_cosine(exp: np.ndarray, theo: np.ndarray, tol: float = MZ_TOL) -> float:
    """Cosine between experimental peaks and Prosit's predicted b/y peaks.

    The predicted peaks define the axis; for each predicted peak the experimental
    intensity is the max observed peak within ``tol`` (0 if absent).  Cosine of
    the two vectors.  Missing predicted peaks contribute 0 and do not help.
    """
    if theo.size == 0 or exp.size == 0:
        return 0.0
    exp_mz = exp[:, 0]
    exp_int = exp[:, 1]
    theo_mz = theo[:, 0]
    lo = np.searchsorted(exp_mz, theo_mz - tol, side="left")
    hi = np.searchsorted(exp_mz, theo_mz + tol, side="right")
    matched = np.zeros(theo_mz.shape[0], dtype=np.float32)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            matched[i] = exp_int[a:b].max()
    norm_a = float(np.linalg.norm(matched))
    norm_b = float(np.linalg.norm(theo[:, 1]))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(matched, theo[:, 1]) / (norm_a * norm_b))


def build_branches(rel_intensities: list[float]):
    """Full minimal branch set (same as step4_predict_score.py): per rel level
    d/f/m_by_prec + hs_d<lvl>_f<lvl> + hs_d<lvl>_f<lvl>_mby.  Downstream users
    filter to the d/f/hsdf columns they care about."""
    branches: list[str] = []
    cols: dict[str, tuple[str, str]] = {}
    rel_for_branch: dict[str, float] = {}
    for r in rel_intensities:
        lvl = f"{int(round(r * 100)):03d}"
        for letter, typ in (("d", "spec"), ("f", "frag"), ("m_by_prec", "")):
            key = f"{letter}_{typ}_rel{lvl}" if typ else f"m_by_prec_rel{lvl}"
            pre = f"{letter}{lvl}" if typ else f"mp{lvl}"
            branches.append(key)
            cols[key] = (pre, f"{pre}_hit")
            rel_for_branch[key] = r
        key = f"hs_d{lvl}_f{lvl}"
        branches.append(key)
        cols[key] = (f"hdf{lvl}_{lvl}", f"hdf{lvl}_{lvl}_hit")
        rel_for_branch[key] = r
        mkey = f"hs_d{lvl}_f{lvl}_mby"
        branches.append(mkey)
        cols[mkey] = (f"hdf{lvl}_{lvl}_mby", f"hdf{lvl}_{lvl}_mby_hit")
        rel_for_branch[mkey] = r
        # raw equal-weight fusions (NO min-max) — mirror the V20 dfexp / dfexp_mby
        rkey = f"dfexp_rel{lvl}"
        branches.append(rkey)
        cols[rkey] = (f"dfexp{lvl}", f"dfexp{lvl}_hit")
        rel_for_branch[rkey] = r
        rmk = f"dfexp_mby_rel{lvl}"
        branches.append(rmk)
        cols[rmk] = (f"dfexpmby{lvl}", f"dfexpmby{lvl}_hit")
        rel_for_branch[rmk] = r
    return tuple(branches), cols, rel_for_branch


def score_candidate(exp: np.ndarray, theo: np.ndarray, rel_intensities: list[float],
                    spec_min_intensity: float = 0.01, tol: float = MZ_TOL) -> dict[str, float]:
    """d/f/m_by_prec per rel level for a Prosit b/y peak list (V20-identical branches).

    rel <= 0 means NO predicted-side intensity filter (keep every positive peak);
    this is the branch Prosit needs because its intensities are poorly calibrated
    but its fragment positions are reliable.  ``theo`` intensities are expected to
    have been sqrt-transformed (same domain as the sqrt_clip experimental peaks).
    """
    out: dict[str, float] = {}
    pmax = float(theo[:, 1].max()) if theo.size else 0.0
    for r in rel_intensities:
        lvl = f"{int(round(r * 100)):03d}"
        if r > 0:
            eff = max(spec_min_intensity, r * pmax)
            tf = theo[theo[:, 1] > eff]
        else:
            tf = theo[theo[:, 1] > 0.0]
        out[f"d_spec_rel{lvl}"] = dc.peak_cosine(exp, tf, tol)
        out[f"f_frag_rel{lvl}"] = dc.explained_intensity(exp, tf, tol)
        out[f"m_by_prec_rel{lvl}"] = dc.frag_precision(exp, tf, tol)
    return out


# ---------------------------------------------------------------------------
# Koina prediction with disk cache
# ---------------------------------------------------------------------------

def _sorted_peaks(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    return arr[np.argsort(arr[:, 0])]


class PrositPredictor:
    def __init__(self, cache_path, batch_size: int = 5000, nce: float | None = None,
                 mods_mode: str = "reanchor"):
        self.batch_size = batch_size
        self.nce = nce
        # v1 caches may contain misaligned annotations and cannot be repaired reliably.
        self.cache_path = Path(str(cache_path) + ".aligned-v2") if cache_path else None
        self.failures = {}
        # cache[(peptide, charge)] = (peaks (M,2) [mz, intensity] sorted, annos list|None)
        self.cache: dict[tuple[str, int], tuple[np.ndarray, list | None]] = {}
        self._model = None
        # incremental persistence: newly predicted (peptide,charge) rows are buffered
        # here and flushed as a small per-save SEGMENT file (never a full rewrite of
        # a multi-GB cache).  self._seg_idx tracks the last written segment.
        self._pending: list[tuple[str, int, np.ndarray, list | None]] = []
        self._seg_idx = 0
        self._load_cache()
        self.mods_mode = mods_mode
        if mods_mode == "auto":
            self.mods_mode = self._detect_mode()
            print(f"[prosit] mods-mode auto -> {self.mods_mode}")

    def _get_model(self):
        if self._model is None:
            from koinapy import Koina  # lazy so the module imports without koinapy
            self._model = Koina(PROSIT_MODEL, PROSIT_HOST)
        return self._model

    def _detect_mode(self) -> str:
        try:
            m = self._get_model()
            for attr in ("inputs", "model_config", "input_schema"):
                obj = getattr(m, attr, None)
                if obj is None:
                    continue
                s = str(obj).lower()
                if any(k in s for k in ("modified", "modifications", "mod_peptide", "ptm")):
                    print(f"[prosit] schema hint '{attr}' mentions modifications -> native")
                    return "native"
            return "reanchor"
        except Exception as exc:  # pragma: no cover
            print(f"[prosit] schema probe failed ({exc}); fallback reanchor")
            return "reanchor"

    def _cache_files(self) -> list[Path]:
        """Legacy monolithic cache + all incremental segment files, sorted."""
        if not self.cache_path:
            return []
        files = []
        if self.cache_path.exists():
            files.append(self.cache_path)
        files += sorted(self.cache_path.parent.glob(f"{self.cache_path.name}.seg_*.tsv.gz"))
        return files

    def _load_cache(self):
        import time
        rows: dict[tuple[str, int], list[tuple[float, float, str]]] = {}
        is_tty = sys.stderr.isatty()
        files = self._cache_files()
        t_start = time.time()
        for fi, path in enumerate(files):
            size = path.stat().st_size
            bar = None
            if is_tty:
                try:
                    from tqdm import tqdm
                    bar = tqdm(total=size, unit="B", unit_scale=True,
                               desc=f"[prosit] load cache {fi + 1}/{len(files)}",
                               mininterval=0.5)
                except Exception:
                    bar = None
            raw = open(path, "rb")
            last = cur = 0
            n_rows = 0
            try:
                gz = gzip.GzipFile(fileobj=raw)
                for bline in gz:
                    cur = raw.tell()
                    if bar is not None and cur > last:
                        bar.update(cur - last)
                        last = cur
                    if bline.startswith(b"peptide\t"):
                        continue  # header (legacy file or any segment)
                    parts = bline.rstrip(b"\r\n").split(b"\t")
                    if len(parts) < 4:
                        continue
                    try:
                        key = (parts[0].decode("utf-8", "replace"), int(parts[1]))
                    except ValueError:
                        continue
                    anno = (parts[4].decode("utf-8", "replace")
                            if len(parts) > 4 else "")
                    rows.setdefault(key, []).append(
                        (float(parts[2]), float(parts[3]), anno))
                    n_rows += 1
                    if bar is None and n_rows % 1_000_000 == 0:
                        el = max(time.time() - t_start, 1e-9)
                        print(f"[prosit] cache load {path.name}: "
                              f"{n_rows / 1e6:.0f}M rows, "
                              f"{int(n_rows / el)} rows/s, "
                              f"{100.0 * cur / max(size, 1):.0f}% bytes", flush=True)
            finally:
                raw.close()
            if bar is not None:
                bar.update(max(0, size - bar.n))
                bar.close()
            if bar is None and n_rows:
                print(f"[prosit] cache load done {path.name}: "
                      f"{n_rows / 1e6:.1f}M rows in {time.time() - t_start:.0f}s", flush=True)
        for k, vals in rows.items():
            vals.sort(key=lambda row: row[0])
            peaks = np.array([[mz, it] for mz, it, _ in vals], dtype=np.float32)
            annos = [a for _, _, a in vals] if vals and vals[0][2] else None
            self.cache[k] = (peaks, annos)
        # restore the segment counter for future writes
        if self.cache_path:
            pat = f"{self.cache_path.name}.seg_"
            idxs = [int(p.name.rsplit("seg_", 1)[1].split(".tsv.gz")[0])
                    for p in self.cache_path.parent.glob(pat + "*.tsv.gz")]
            self._seg_idx = max(idxs) if idxs else 0
        if self.cache:
            print(f"[prosit] loaded {len(self.cache)} predicted peptides from cache")

    def save_cache(self):
        """Flush buffered rows as one small segment gzip (append-style)."""
        if not self.cache_path or not self._pending:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._seg_idx += 1
        seg = self.cache_path.with_name(
            f"{self.cache_path.name}.seg_{self._seg_idx:06d}.tsv.gz")
        tmp = seg.with_name(seg.name + ".partial")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write("peptide\tcharge\tmz\tintensity\tannotation\n")
            for pep, ch, arr, annos in self._pending:
                for i, (mz, it) in enumerate(arr):
                    anno = annos[i] if annos else ""
                    fh.write(f"{pep}\t{ch}\t{mz:.7f}\t{it:.10f}\t{anno}\n")
        tmp.replace(seg)
        self._pending = []

    def _predict_with_retry(self, model, df: pd.DataFrame,
                            attempts: int = 4, backoff: float = 5.0) -> pd.DataFrame:
        """Koina gRPC calls can be reset by the server (RST_STREAM); retry with
        exponential backoff so transient network failures do not kill the run."""
        for attempt in range(1, attempts + 1):
            try:
                return model.predict(df)
            except Exception as exc:
                print(f"[prosit] predict chunk failed (attempt {attempt}/{attempts}): "
                      f"{type(exc).__name__}: {exc}")
                if attempt == attempts:
                    raise
                time.sleep(backoff * (2 ** (attempt - 1)))

    def predict_missing(self, needed: set[tuple[str, int]]):
        missing = needed - set(self.cache)
        if not missing:
            print(f"[prosit] all {len(needed)} peptides already cached")
            return
        print(f"[prosit] predicting {len(missing)} missing peptides "
              f"(cache has {len(self.cache)}), batch_size={self.batch_size}")
        model = self._get_model()
        items = sorted(missing)
        n_chunks = (len(items) + self.batch_size - 1) // self.batch_size
        n_dropped = 0
        for ci in range(n_chunks):
            chunk = items[ci * self.batch_size:(ci + 1) * self.batch_size]
            n_dropped += self._predict_robust(model, chunk)
            print(f"[prosit] chunk {ci + 1}/{n_chunks} done ({len(self.cache)} in cache)")
            self.save_cache()  # incremental, survives interruption
        print(f"[prosit] prediction done; cache now has {len(self.cache)}"
              + (f"; {n_dropped} peptides skipped (Koina-hostile)" if n_dropped else ""))

    def _predict_robust(self, model, items: list[tuple[str, int]]) -> int:
        """Predict ``items`` recursively, isolating Koina-hostile rows.

        Some inputs make the Koina server fail deterministically (e.g. a shape /
        ValueError inside the model), which fails the WHOLE batch no matter how
        many times we retry it.  Retry a few times for transient resets, then
        bisect and recurse; single rows that still fail after leaf retries are
        dropped (returned as the drop count) instead of killing the run.
        """
        if not items:
            return 0
        df = pd.DataFrame({
            "peptide_sequences": [p for p, _ in items],
            "precursor_charges": [c for _, c in items],
            "fragmentation_types": ["HCD"] * len(items),
        })
        if self.nce is not None:
            df["collision_energy"] = float(self.nce)
        try:
            pred = self._predict_with_retry(model, df)
            self._absorb(pred)
            return 0
        except Exception as exc:
            if len(items) == 1:
                # leaf: retry a few times before dropping (transient resets happen)
                for attempt in range(1, 4):
                    time.sleep(5.0 * attempt)
                    try:
                        pred = model.predict(df)
                        self._absorb(pred)
                        return 0
                    except Exception as exc2:
                        if attempt == 3:
                            self.failures[items[0]] = f"{type(exc2).__name__}: {exc2}"
                            print(f"[prosit] DROP peptide {items[0]}: "
                                  f"{type(exc2).__name__}: {exc2}")
                return 1
            mid = len(items) // 2
            print(f"[prosit] batch {len(items)} failed ({type(exc).__name__}); "
                  f"bisecting -> {mid}/{len(items) - mid}")
            time.sleep(2.0)
            return self._predict_robust(model, items[:mid]) + \
                   self._predict_robust(model, items[mid:])

    def _absorb(self, pred: pd.DataFrame):
        for col in ("mz", "intensities"):
            if col not in pred.columns:
                raise ValueError(f"Koina output missing column {col!r}; got {list(pred.columns)}")
        has_anno = "annotation" in pred.columns
        for key, g in pred.groupby(["peptide_sequences", "precursor_charges"], sort=False):
            pep, ch = str(key[0]), int(key[1])
            mz = g["mz"].to_numpy(dtype=np.float32)
            it = g["intensities"].to_numpy(dtype=np.float32)
            annos = None
            if has_anno:
                annos = [a.decode("utf-8") if isinstance(a, bytes) else str(a)
                         for a in g["annotation"].tolist()]
                # keep only clean b/y rows (the only ones build_candidate_peaks uses)
                keep = [i for i, a in enumerate(annos) if parse_annotation(a) is not None]
                if len(keep) < len(annos):
                    mz = mz[keep]
                    it = it[keep]
                    annos = [annos[i] for i in keep]
            order = np.argsort(mz, kind="stable")
            arr = np.column_stack([mz, it])[order]
            if annos is not None:
                annos = [annos[int(i)] for i in order]
            was_new = (pep, ch) not in self.cache
            self.cache[(pep, ch)] = (arr, annos)
            if was_new:
                self._pending.append((pep, ch, arr, annos))

    def get(self, peptide: str, charge: int) -> tuple[np.ndarray, list | None]:
        return self.cache.get((peptide, charge),
                              (np.empty((0, 2), dtype=np.float32), None))


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def load_candidates_by_ordinal(tsv_gz: Path) -> dict[int, list[tuple[str, float, float]]]:
    grouped: dict[int, list[tuple[str, float, float]]] = {}
    with gzip.open(tsv_gz, "rt", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd, None)
        for row in rd:
            if not row:
                continue
            grouped.setdefault(int(row[0]), []).append((row[6], float(row[7]), float(row[8])))
    return grouped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--candidates", required=True, help="step3 TSV.gz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--scoring", default="branches", choices=["branches", "cosine"],
                   help="branches = same d/f/m_by_prec + hsdf/hsdf_mby machinery as the "
                        "V20 pipeline (fair predictor comparison); cosine = single "
                        "spectral cosine only")
    p.add_argument("--rel-intensities", default="0,0.05,0.10,0.20",
                   help="predicted-side rel filter levels; rel 0 = no intensity "
                        "filter (Prosit's intensity-insensitive branch); "
                        "e.g. 0,0.05,0.10,0.20")
    p.add_argument("--mods-mode", default="reanchor",
                   choices=["auto", "native", "reanchor"])
    p.add_argument("--batch-size", type=int, default=2000,
                   help="Koina prediction chunk size (peptides per call); smaller "
                        "batches reduce response size / RST_STREAM risk for remote Koina")
    p.add_argument("--nce", type=float, default=None,
                   help="collision energy, only if the model requires it")
    p.add_argument("--exp-transform", default="sqrt_clip",
                   choices=["none", "sqrt_clip", "sqrt"],
                   help="map experimental peak intensities into the V20 training domain "
                        "before scoring (default sqrt_clip = sqrt(min(x,50000))/sqrt(50000), "
                        "the same preprocessing the V20 pipeline uses so the d/f comparison "
                        "is fair; none = raw).  Presence-based branches unchanged.")
    p.add_argument("--selfcheck", action="store_true",
                   help="connect Koina, print model schema, predict one peptide, then exit")
    return p.parse_args()


def selfcheck(args) -> int:
    predictor = PrositPredictor(cache_path=None, mods_mode=args.mods_mode)
    model = predictor._get_model()
    print(f"\n[prosit] model: {PROSIT_MODEL} @ {PROSIT_HOST}")
    print(f"[prosit] resolved mods-mode: {predictor.mods_mode}")
    for attr in ("inputs", "outputs", "model_config", "input_schema", "output_schema"):
        obj = getattr(model, attr, None)
        if obj is not None:
            print(f"[prosit] {attr} = {obj}")
    test_pep = "ASIQEIDISSNK"
    for charge in (2, 3):
        df = pd.DataFrame({
            "peptide_sequences": [test_pep],
            "precursor_charges": [charge],
            "fragmentation_types": ["HCD"],
        })
        try:
            pred = model.predict(df)
            print(f"\n[prosit] predict {test_pep} charge={charge}: "
                  f"rows={len(pred)}, columns={list(pred.columns)}")
            print(pred.head())
        except Exception as exc:
            print(f"\n[prosit] predict charge={charge} FAILED: {exc}")
    return 0


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        return selfcheck(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    candidates_by_ordinal = load_candidates_by_ordinal(Path(args.candidates))
    print(f"[prosit] loaded candidates for {len(candidates_by_ordinal)} scans")

    predictor = PrositPredictor(out_dir / "predictions" / "prosit_cache.tsv.gz",
                                batch_size=args.batch_size, nce=args.nce,
                                mods_mode=args.mods_mode)
    mods_mode = predictor.mods_mode

    mgf = dc.resolve_species_files(Path(args.root), args.species)[1]
    total = args.max_scans or dc.count_mgf_spectra(mgf)

    # pass 1: collect spectra + needed (prosit_seq, charge)
    specs = []
    needed: set[tuple[str, int]] = set()
    n_skipped = 0
    n_too_long = 0      # candidates dropped because backbone > PROSIT_MAX_LEN
    n_gt_too_long = 0   # of those, ones that mass-match the GT (unresolvable by Prosit)
    for ordinal, (meta, peaks) in enumerate(
        dc.progress(dc.iter_mgf_spectra(mgf, max_scans=args.max_scans),
                    total=total, desc="prosit collect", unit="spec", mininterval=1.0)
    ):
        charge = dc.parse_charge_from_meta(meta)
        pepmass = dc.parse_pepmass_from_meta(meta)
        rt = float(meta.get("RTINSECONDS", dc.DEFAULT_RT))
        gt_seq = meta.get("SEQ", "")
        cands = candidates_by_ordinal.get(ordinal, [])
        if charge <= 0 or pepmass <= 0 or not cands:
            n_skipped += 1
            continue
        if peaks.size == 0:
            n_skipped += 1
            continue
        exp = dc.transform_exp_intensity(peaks, method=args.exp_transform).astype(np.float32)
        seqs = []
        for c in cands:
            s = c[0]
            ps = s if mods_mode == "native" else strip_mods(s)
            if len(ps) > PROSIT_MAX_LEN:
                n_too_long += 1
                if gt_seq and dc.peptide_matches(s, gt_seq):
                    n_gt_too_long += 1
                continue
            seqs.append(s)
        if not seqs:
            n_skipped += 1
            continue
        specs.append((ordinal, meta, charge, pepmass, rt, gt_seq, exp, seqs))
        for s in seqs:
            ps = s if mods_mode == "native" else strip_mods(s)
            needed.add((ps, charge))
    if n_too_long:
        print(f"[prosit] skipped {n_too_long} candidates with backbone > "
              f"{PROSIT_MAX_LEN} aa (Prosit limit); {n_gt_too_long} of them were GT "
              f"(those spectra are unresolvable by Prosit)")

    predictor.predict_missing(needed)

    # ---- scoring setup ----
    if args.scoring == "branches":
        rel_intensities = [float(x.strip()) for x in args.rel_intensities.split(",") if x.strip()]
        BRANCHES, BRANCH_COLS, _rel_for_branch = build_branches(rel_intensities)
        csv_columns = (
            ["ordinal", "scan_id", "charge", "pepmass", "rt", "n_candidates",
             "gt_seq", "gt_in_candidates"]
            + [f"{BRANCH_COLS[b][0]}_best_seq" for b in BRANCHES]
            + [f"{BRANCH_COLS[b][0]}_score" for b in BRANCHES]
            + [f"{BRANCH_COLS[b][0]}_gt_score" for b in BRANCHES]
            + [f"{BRANCH_COLS[b][0]}_gt_rank" for b in BRANCHES]
            + [f"{BRANCH_COLS[b][0]}_hit" for b in BRANCHES]
        )
    else:  # cosine
        BRANCHES = ["cosine"]
        BRANCH_COLS = {"cosine": ("cos", "cos_hit")}
        csv_columns = ["ordinal", "scan_id", "charge", "pepmass", "rt", "n_candidates",
                       "gt_seq", "gt_in_candidates",
                       "cos_best_seq", "cos_score", "cos_gt_score", "cos_gt_rank", "cos_hit"]
    results_csv = out_dir / f"results_prosit_{args.species}.csv"

    n_processed = 0
    n_with_gt = 0
    branch_hits: dict[str, int] = {b: 0 for b in BRANCHES}
    branch_gt_rank_accum: dict[str, list[int]] = {b: [] for b in BRANCHES}
    n_candidates_list: list[int] = []

    with open(results_csv, "w", encoding="utf-8", newline="") as fout_csv:
        writer = csv.DictWriter(fout_csv, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for ordinal, meta, charge, pepmass, rt, gt_seq, exp, seqs in dc.progress(
                specs, desc="prosit score", unit="spec", mininterval=1.0):
            n_processed += 1
            n_candidates_list.append(len(seqs))

            # per-candidate Prosit b/y peaks (sqrt-transform intensities to the
            # same sqrt_clip domain as the experimental peaks; Prosit raw
            # intensities are skewed, which breaks rel filtering and cosine)
            theos = []
            for seq in seqs:
                ps = seq if mods_mode == "native" else strip_mods(seq)
                peaks, annos = predictor.get(ps, charge)
                theo = build_candidate_peaks(peaks, annos, seq, mods_mode)
                if theo.size:
                    theo[:, 1] = np.sqrt(theo[:, 1])
                theos.append(theo)

            gt_in = bool(gt_seq) and any(dc.peptide_matches(s, gt_seq) for s in seqs)
            if gt_seq:
                n_with_gt += 1

            # per-branch scores
            scores: dict[str, np.ndarray] = {}
            if args.scoring == "branches":
                for i, theo in enumerate(theos):
                    sc = score_candidate(exp, theo, rel_intensities)
                    for b, val in sc.items():
                        scores.setdefault(b, np.zeros(len(seqs), dtype=np.float64))[i] = val

                def _score_avg(base):
                    acc = np.zeros(len(seqs), dtype=np.float64)
                    for bk in base:
                        s = scores[bk]
                        lo, hi = float(s.min()), float(s.max())
                        if hi - lo > 1e-12:
                            acc += (s - lo) / (hi - lo)
                    return acc / len(base)

                def _raw_avg(base):
                    acc = np.zeros(len(seqs), dtype=np.float64)
                    for bk in base:
                        acc += scores[bk]
                    return acc / len(base)

                for r in rel_intensities:
                    lvl = f"{int(round(r * 100)):03d}"
                    df_b = [f"d_spec_rel{lvl}", f"f_frag_rel{lvl}"]
                    scores[f"hs_d{lvl}_f{lvl}"] = _score_avg(df_b)
                    scores[f"hs_d{lvl}_f{lvl}_mby"] = _score_avg(
                        df_b + [f"m_by_prec_rel{lvl}"])
                    scores[f"dfexp_rel{lvl}"] = _raw_avg(df_b)
                    scores[f"dfexp_mby_rel{lvl}"] = _raw_avg(
                        df_b + [f"m_by_prec_rel{lvl}"])
            else:  # cosine
                scores["cosine"] = np.array(
                    [spectral_cosine(exp, theo) for theo in theos], dtype=np.float64)

            row = {
                "ordinal": ordinal,
                "scan_id": meta.get("SCANS", str(ordinal)),
                "charge": charge,
                "pepmass": round(pepmass, 6),
                "rt": round(rt, 3),
                "n_candidates": len(seqs),
                "gt_seq": gt_seq,
                "gt_in_candidates": int(gt_in),
            }
            for b in BRANCHES:
                prefix = BRANCH_COLS[b][0]
                order = np.argsort(-scores[b])
                best_i = int(order[0])
                best_seq = seqs[best_i]
                gt_idx = -1
                gt_score = -1.0
                if gt_seq:
                    best_gt = -float("inf")
                    for cand_i in range(len(seqs)):
                        if dc.peptide_matches(seqs[cand_i], gt_seq):
                            sc = float(scores[b][cand_i])
                            if sc > best_gt:
                                best_gt = sc
                                gt_idx = cand_i
                                gt_score = sc
                gt_rank = -1
                if gt_idx >= 0:
                    for rank_pos, cand_i in enumerate(order, start=1):
                        if cand_i == gt_idx:
                            gt_rank = rank_pos
                            break
                hit = int(bool(gt_seq) and dc.peptide_matches(best_seq, gt_seq))
                if hit:
                    branch_hits[b] += 1
                if gt_rank > 0:
                    branch_gt_rank_accum[b].append(gt_rank)
                row[f"{prefix}_best_seq"] = best_seq
                row[f"{prefix}_score"] = round(float(scores[b][best_i]), 6)
                row[f"{prefix}_gt_score"] = round(gt_score, 6)
                row[f"{prefix}_gt_rank"] = gt_rank
                row[f"{prefix}_hit"] = hit
            writer.writerow(row)

    predictor.save_cache()

    # ---- summary ----
    def rank_rate(branch: str, max_rank: int) -> float:
        denom = n_with_gt if n_with_gt else 1
        return sum(1 for r in branch_gt_rank_accum[branch] if r <= max_rank) / denom

    summary = {
        "species": args.species,
        "predictor": "prosit",
        "prosit_model": PROSIT_MODEL,
        "scoring": args.scoring,
        "mods_mode": mods_mode,
        "exp_transform": args.exp_transform,
        "n_processed": n_processed,
        "n_with_gt": n_with_gt,
        "n_skipped": n_skipped,
        "mean_candidates": round(float(np.mean(n_candidates_list)), 2) if n_candidates_list else 0.0,
        "branches": {
            b: {
                "top1_acc": round(branch_hits[b] / n_with_gt, 6) if n_with_gt else 0.0,
                "top3_acc": round(rank_rate(b, 3), 6),
                "top10_acc": round(rank_rate(b, 10), 6),
                "hits": branch_hits[b],
            }
            for b in BRANCHES
        },
        "results_csv": str(results_csv),
        **timer.summary(),
    }
    with open(out_dir / f"summary_prosit_{args.species}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    record = {
        "step": 4,
        "predictor": "prosit",
        "species": args.species,
        "scoring": args.scoring,
        "rel_intensities": args.rel_intensities if args.scoring == "branches" else None,
        "mods_mode": mods_mode,
        "batch_size": args.batch_size,
        "nce": args.nce,
        "summary": summary,
        "cache_path": str(out_dir / "predictions" / "prosit_cache.tsv.gz"),
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_prosit_{args.species}.json", record)

    print(f"\n[prosit] done. search accuracy (scoring={args.scoring}):")
    for b in BRANCHES:
        print(f"  {b:<24} top1={summary['branches'][b]['top1_acc']:.4f} "
              f"top3={summary['branches'][b]['top3_acc']:.4f} "
              f"top10={summary['branches'][b]['top10_acc']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
