"""Cache Model XL cosine scores and select each Human Test winner by F."""

import argparse
from itertools import islice
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from pyarrow import csv, parquet as pq
from tqdm import tqdm


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model_xl"
TRANSFORMER = MODEL_ROOT / "transfomer"
DATA_PARENT = Path(r"D:\XL\prosit-xl\打分测试\human_test_log12_top100000_v1")
DEFAULT_DATA = DATA_PARENT / "human_test_log12_top100000_v1"
DEFAULT_OUTPUT = DATA_PARENT / "model_xl_cosine_per_spectrum"
DEFAULT_CHECKPOINT = TRANSFORMER / "runs" / "xl_presence_rank010_valcosine_seed42_epochs20_20261116_113448" / "best.pt"

sys.path[:0] = [str(TRANSFORMER), str(MODEL_ROOT / "dataset")]
from build_xl_spectrum_dataset import GroupKey, NUM_CHANNELS, build_theoretical_mz
from run_xl_candidate_rescore import (
    COLUMNS, CSV_OPTIONS, HEADER, Forms, load_spectra, report, score_spectrum, verify_csv,
)
from spectrum_transformer.warnings import configure_v20_warning_filters


def modified_sequence(form):
    residues = list(form["sequence"])
    for mod in form["modifications"]:
        residues[mod["position_0based"]] += f"[{mod['mass_delta_da']:+.6f}]"
    return "".join(residues)


def nearest_ions(mz, grid):
    """Return the closest theoretical ion and absolute ppm error for each peak."""
    flat = grid.ravel()
    ions = np.flatnonzero(flat > 0)
    masses, first = np.unique(flat[ions], return_index=True)
    ions = ions[first]
    # On either side of a peak, relative error grows with distance from it.
    right = np.searchsorted(masses, mz)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(masses) - 1)
    left_ppm = 1e6 * np.abs(mz - masses[left]) / masses[left]
    right_ppm = 1e6 * np.abs(mz - masses[right]) / masses[right]
    return ions[np.where(right_ppm < left_ppm, right, left)], np.minimum(left_ppm, right_ppm)


def fragment_features(spectrum, form_a, form_b, site_a, site_b, mz, intensity):
    key = GroupKey(
        modified_sequence(form_a), site_a + 1, modified_sequence(form_b), site_b + 1,
        spectrum["crosslinker"], spectrum["precursor_charge"],
    )
    grid, lengths = build_theoretical_mz(key)
    matches = [nearest_ions(mz, chain) for chain in grid]
    explained = [ppm <= 20 for _, ppm in matches]
    shared = explained[0] & explained[1]
    total_intensity = intensity.sum()
    coverage, fractions = [], []
    n5 = missed = consecutive = 0
    for chain, (form, site) in enumerate(((form_a, site_a), (form_b, site_b))):
        ions, ppm = matches[chain]
        independent = explained[chain] & ~shared
        coverage.append(len(np.unique(ions[independent] // NUM_CHANNELS)) / (int(lengths[chain]) - 1))
        fractions.append(intensity[independent].sum() / total_intensity if total_intensity else 0.0)
        exact = independent & (ppm <= 5)
        n5 += int(exact.sum())
        supported = np.zeros(grid[chain].size, dtype=bool)
        supported[ions[exact]] = True
        supported = supported.reshape(grid[chain].shape)
        consecutive += int(np.any(supported[:-1] & supported[1:], axis=1).sum())
        sequence = form["sequence"]
        missed += sum(i != site and aa in "KR" and next_aa != "P"
                      for i, (aa, next_aa) in enumerate(zip(sequence, sequence[1:])))
    return min(coverage), min(fractions), n5, missed, consecutive


class FinalRescorer:
    def __init__(self, data):
        self.forms = pq.read_table(data / "peptidoforms.parquet",
                                   columns=["form_id", "sequence", "modifications", "neutral_mass_da"])
        self.form_ids = self.forms["form_id"].to_numpy()
        masses = pq.read_table(data / "test" / "spectra.parquet", columns=["query_id", "precursor_neutral_mass"])
        self.precursor_masses = dict(zip(masses["query_id"].to_pylist(), masses["precursor_neutral_mass"].to_pylist()))

    def top_candidate(self, spectrum, scores, candidates):
        query = spectrum["query_id"]
        cosine = scores["cosine"].to_numpy()
        search_rank = scores["candidate_rank"].to_numpy()
        valid = np.flatnonzero(np.isfinite(cosine))
        ranked = valid[np.lexsort((search_rank[valid], -cosine[valid]))]
        cosine_rank = np.zeros(len(cosine), dtype=np.int32)
        cosine_rank[ranked] = np.arange(1, len(ranked) + 1)
        pool = valid[(search_rank[valid] <= 100) | (cosine_rank[valid] <= 100)]
        shortlist = scores.take(pool)
        metadata = pq.read_table(candidates / f"{query}.parquet",
                                 columns=["rank", "form_a_id", "form_b_id", "isotope_error"])
        if (not metadata["rank"].equals(scores["candidate_rank"])
                or any(not metadata[key].equals(scores[key]) for key in ("form_a_id", "form_b_id"))):
            raise ValueError(f"Candidate metadata does not match the cosine CSV for {query}")
        metadata = metadata.take(pool)

        form_ids = np.unique(np.concatenate([shortlist[key].to_numpy() for key in ("form_a_id", "form_b_id")]))
        form_rows = np.searchsorted(self.form_ids, form_ids)
        if (form_rows >= len(self.form_ids)).any() or not np.array_equal(self.form_ids[form_rows], form_ids):
            raise ValueError(f"Candidate references an unknown peptidoform for {query}")
        forms = {row["form_id"]: row for row in self.forms.take(form_rows).to_pylist()}
        mz = np.asarray(spectrum["mz"], dtype=np.float64)
        intensity = np.asarray(spectrum["intensity"], dtype=np.float64)
        keep = np.isfinite(mz) & (mz > 0) & np.isfinite(intensity) & (intensity > 0)
        mz, intensity = mz[keep], np.sqrt(intensity[keep])

        s4 = np.empty(len(pool), dtype=np.float64)
        s8 = np.empty_like(s4)
        pairs = []
        for i, (row, isotope) in enumerate(zip(shortlist.to_pylist(), metadata["isotope_error"].to_pylist())):
            a, b = row["form_a_id"], row["form_b_id"]
            pairs.append((min(a, b), max(a, b)))
            form_a, form_b = forms[a], forms[b]
            f_weak, j_weak, n5, missed, consecutive = fragment_features(
                spectrum, form_a, form_b, row["site_a_0based"], row["site_b_0based"], mz, intensity,
            )
            pair_mass = form_a["neutral_mass_da"] + form_b["neutral_mass_da"] + spectrum["crosslink_mass_da"]
            corrected_mass = self.precursor_masses[query] - isotope * 1.0033548378
            d_ppm = 1e6 * abs(corrected_mass - pair_mass) / pair_mass
            s4[i] = (-np.log(row["candidate_rank"] * cosine_rank[pool[i]]) + 8 * f_weak
                     - 0.5 * d_ppm + 64 * j_weak)
            s8[i] = s4[i] + (40 * row["cosine"] + n5 - 4 * missed + 0.5 * consecutive)

        by_s8 = np.lexsort((search_rank[pool], -s8))
        best = by_s8[0]
        second = next((i for i in by_s8[1:] if pairs[i] != pairs[best]), None)
        # A pool with a single distinct pair has no competition margin.
        final_scores = s4
        if second is not None:
            competitor = np.array([s8[second] if pair == pairs[best] else s8[best] for pair in pairs])
            final_scores = s4 + (s8 - competitor)
        winner = np.lexsort((search_rank[pool], -final_scores))[0]
        row = pool[winner]
        return {
            "top1_rank": int(search_rank[row]), "top1_cosine": float(cosine[row]),
            "top1_f": float(final_scores[winner]), "top1_is_gt": bool(scores["is_gt"][int(row)].as_py()),
        }


def save_spectrum_csv(path, spectrum, gt, write_rows):
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", prefix=path.stem + ".", suffix=".part", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(HEADER)
            write_rows(f)
            f.flush()
            os.fsync(f.fileno())
        checked = verify_csv(temporary, [(spectrum, gt)])
        os.replace(temporary, path)
        return checked
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_combined_csv(source, output, spectra, preceding_rows):
    """Reuse complete blocks from a run with the same data and checkpoint."""
    imported = 0
    with source.open("rb") as f:
        if f.readline() != HEADER:
            raise ValueError(f"Unexpected combined CSV header: {source}")
        remaining_rows = islice(f, preceding_rows, None)
        for spectrum, gt in spectra:
            row_count = spectrum["candidate_count"]
            rows = list(islice(remaining_rows, row_count))
            if len(rows) < row_count or not rows[-1].endswith(b"\n"):
                if rows:
                    report("ignored_partial_import", rows=len(rows))
                break
            query = spectrum["query_id"]
            if not rows[0].startswith(f"{query},1,".encode("ascii")) or not rows[-1].startswith(
                f"{query},{row_count},".encode("ascii")
            ):
                raise ValueError(f"Combined CSV does not match the source order at {query}")
            path = output / f"{query}.csv"
            if not path.exists():
                save_spectrum_csv(path, spectrum, gt, lambda stream: stream.writelines(rows))
                imported += 1
    report("imported", spectra=imported, source=str(source))


def make_scorer(checkpoint):
    import torch
    from xl_rescore_inference import XLRescoreInference

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Model XL scoring script")
    return XLRescoreInference(checkpoint, "cuda")


def main(*, final_ranking=True):
    description = __doc__ if final_ranking else "Score the Human Test candidates with Model XL cosine."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Source Parquet data package")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model XL best.pt")
    parser.add_argument("--output-dir", type=Path, help="Directory for <query_id>.csv files (default: dataset-specific)")
    parser.add_argument("--batch-size", type=int, default=32, help="Maximum CUDA batch size (default: %(default)s)")
    parser.add_argument("--start-spectrum", type=int, default=1, help="1-based starting position among GT-hit spectra")
    parser.add_argument("--limit-spectra", type=int, help="Maximum number of spectra from --start-spectrum")
    parser.add_argument("--import-combined", type=Path, help="Reuse a combined CSV from the same data and checkpoint")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_spectra is not None and args.limit_spectra < 1):
        parser.error("Use --batch-size and --limit-spectra values of at least 1")

    start = perf_counter()
    eligible = load_spectra(args.data)
    if not 1 <= args.start_spectrum <= len(eligible):
        parser.error(f"Use --start-spectrum between 1 and {len(eligible)}")
    first = args.start_spectrum - 1
    stop = first + args.limit_spectra if args.limit_spectra is not None else None
    spectra = eligible[first:stop]
    last_spectrum = first + len(spectra)
    if args.output_dir is None:
        data = args.data.resolve()
        args.output_dir = (DEFAULT_OUTPUT if data == DEFAULT_DATA.resolve()
                           else data.with_name(f"{data.name}_model_xl_cosine_per_spectrum"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.import_combined is not None:
        preceding_rows = sum(spec["candidate_count"] for spec, _ in eligible[:first])
        import_combined_csv(args.import_combined, args.output_dir, spectra, preceding_rows)

    candidates = args.data / "test" / "candidates"
    scorer = None
    rescorer = FinalRescorer(args.data) if final_ranking else None
    reused = hits = unsupported = candidate_total = 0
    interactive = sys.stdout.isatty()
    if interactive:
        configure_v20_warning_filters()
    bar_options = dict(file=sys.stdout, disable=not interactive, dynamic_ncols=True)
    last_timing = ""
    with (
        tqdm(desc="Spectrum", position=0, bar_format="{desc}", **bar_options) as spectrum_status,
        tqdm(total=spectra[0][0]["candidate_count"], desc="Candidate", position=1, leave=False, ascii=True,
             bar_format="{desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
             **bar_options) as candidate_bar,
    ):
        for ordinal, (spectrum, gt) in enumerate(spectra, args.start_spectrum):
            query = spectrum["query_id"]
            scan = spectrum["scan_number"]
            row_count = spectrum["candidate_count"]
            label = f"Spectrum {ordinal}/{last_spectrum} (scan {scan})"
            path = args.output_dir / f"{query}.csv"
            cached = path.exists()
            if cached:
                if interactive:
                    spectrum_status.set_description_str(f"{label} | Checking saved CSV{last_timing}")
                checked = verify_csv(path, [(spectrum, gt)])
                if rescorer is not None:
                    result = csv.read_csv(path, convert_options=csv.ConvertOptions(column_types=COLUMNS))
                reused += 1
            else:
                if interactive:
                    spectrum_status.set_description_str(f"{label}{last_timing}")
                    candidate_bar.set_postfix_str("", refresh=False)
                    candidate_bar.reset(total=row_count)
                if scorer is None:
                    forms = Forms(args.data / "peptidoforms.parquet")
                    scorer = make_scorer(args.checkpoint)
                started = perf_counter()
                if not interactive:
                    report("spectrum_started", ordinal=ordinal, query_id=query, scan_number=scan, batch_size=args.batch_size)

                def show_progress(completed_candidates, completed_combinations, total_combinations):
                    sites = f"KxK {completed_combinations:,}/{total_combinations:,}"
                    if interactive:
                        candidate_bar.n = completed_candidates
                        candidate_bar.set_postfix_str(sites, refresh=True)
                    else:
                        print(f"{label} | Candidate {completed_candidates:,}/{row_count:,} | {sites}", flush=True)

                result, stats = score_spectrum(
                    scorer, forms, spectrum, gt, candidates, args.batch_size,
                    on_progress=show_progress, progress_seconds=1 if interactive else 5,
                )
                checked = save_spectrum_csv(
                    path, spectrum, gt, lambda stream: csv.write_csv(result, stream, write_options=CSV_OPTIONS),
                )
                if checked["top1_hits"] != stats["top1_is_gt"] or checked["unsupported"] != stats["unsupported"]:
                    raise ValueError(f"Written CSV does not match the scored result: {path}")

            top = rescorer.top_candidate(spectrum, result, candidates) if rescorer is not None else {}
            if not cached:
                seconds = perf_counter() - started
                if interactive:
                    last_timing = f" | Last saved scan {scan} in {seconds:.1f}s"
                    spectrum_status.set_description_str(f"{label}{last_timing}")
                else:
                    report("spectrum_completed", ordinal=ordinal, seconds=round(seconds, 3), **(stats | top))
            hits += top.get("top1_is_gt", checked["top1_hits"])
            unsupported += checked["unsupported"]
            candidate_total += checked["candidates"]
        if interactive:
            spectrum_status.set_description_str(f"Spectra saved {len(spectra)}/{len(spectra)}{last_timing}")

    report("complete", spectra=len(spectra), reused=reused, candidates=candidate_total, unsupported=unsupported,
           top1_hits=hits, top1_fraction=hits / len(spectra), seconds=round(perf_counter() - start, 3))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
