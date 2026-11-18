"""Score the Human Test candidates with Model XL and save one CSV per spectrum."""

import argparse
from itertools import islice
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

from pyarrow import csv
from tqdm import tqdm


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model_xl"
TRANSFORMER = MODEL_ROOT / "transfomer"
DATA_PARENT = Path(r"D:\XL\prosit-xl\打分测试\human_test_log12_top100000_v1")
DEFAULT_DATA = DATA_PARENT / "human_test_log12_top100000_v1"
DEFAULT_OUTPUT = DATA_PARENT / "model_xl_cosine_per_spectrum"
DEFAULT_CHECKPOINT = TRANSFORMER / "runs" / "xl_presence_rank010_valcosine_seed42_epochs20_20261116_113448" / "best.pt"

sys.path[:0] = [str(TRANSFORMER), str(MODEL_ROOT / "dataset")]
from run_xl_candidate_rescore import (
    CSV_OPTIONS, HEADER, ROWS, SPECTRA, UNSUPPORTED, Forms, load_spectra, report, score_spectrum, verify_csv,
)
from spectrum_transformer.warnings import configure_v20_warning_filters


def save_spectrum_csv(path, write_rows):
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", prefix=path.stem + ".", suffix=".part", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(HEADER)
            write_rows(f)
            f.flush()
            os.fsync(f.fileno())
        checked = verify_csv(temporary, [path.stem])
        os.replace(temporary, path)
        return checked
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_combined_csv(source, output, spectra, start_spectrum=1):
    """Reuse complete blocks from the previous run with the same checkpoint."""
    imported = 0
    with source.open("rb") as f:
        if f.readline() != HEADER:
            raise ValueError(f"Unexpected combined CSV header: {source}")
        remaining_rows = islice(f, (start_spectrum - 1) * ROWS, None)
        for spectrum, _ in spectra:
            rows = list(islice(remaining_rows, ROWS))
            if len(rows) < ROWS or not rows[-1].endswith(b"\n"):
                if rows:
                    report("ignored_partial_import", rows=len(rows))
                break
            query = spectrum["query_id"]
            if not rows[0].startswith(f"{query},1,".encode("ascii")) or not rows[-1].startswith(
                f"{query},{ROWS},".encode("ascii")
            ):
                raise ValueError(f"Combined CSV does not match the source order at {query}")
            path = output / f"{query}.csv"
            if not path.exists():
                save_spectrum_csv(path, lambda stream: stream.writelines(rows))
                imported += 1
    report("imported", spectra=imported, source=str(source))


def make_scorer(checkpoint):
    import torch
    from xl_rescore_inference import XLRescoreInference

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Model XL scoring script")
    return XLRescoreInference(checkpoint, "cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Source Parquet data package")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model XL best.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for <query_id>.csv files")
    parser.add_argument("--batch-size", type=int, default=32, help="Maximum CUDA batch size (default: %(default)s)")
    parser.add_argument("--start-spectrum", type=int, default=1, help="1-based starting position among GT-hit spectra")
    parser.add_argument("--limit-spectra", type=int, default=SPECTRA, help="Maximum number of spectra from --start-spectrum")
    parser.add_argument("--import-combined", type=Path, help="Reuse an earlier combined CSV from the same scoring run")
    args = parser.parse_args()
    if args.batch_size < 1 or not 1 <= args.limit_spectra <= SPECTRA:
        parser.error(f"Use --batch-size >= 1 and --limit-spectra between 1 and {SPECTRA}")
    if not 1 <= args.start_spectrum <= SPECTRA:
        parser.error(f"Use --start-spectrum between 1 and {SPECTRA}")

    start = perf_counter()
    first = args.start_spectrum - 1
    spectra = load_spectra(args.data)[first:first + args.limit_spectra]
    last_spectrum = first + len(spectra)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.import_combined is not None:
        import_combined_csv(args.import_combined, args.output_dir, spectra, args.start_spectrum)

    candidates = args.data / "test" / "candidates"
    scorer = None
    reused = hits = unsupported = 0
    interactive = sys.stdout.isatty()
    if interactive:
        configure_v20_warning_filters()
    bar_options = dict(file=sys.stdout, disable=not interactive, dynamic_ncols=True)
    last_timing = ""
    with (
        tqdm(desc="Spectrum", position=0, bar_format="{desc}", **bar_options) as spectrum_status,
        tqdm(total=ROWS, desc="Candidate", position=1, leave=False, ascii=True,
             bar_format="{desc} {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
             **bar_options) as candidate_bar,
    ):
        for ordinal, (spectrum, gt) in enumerate(spectra, args.start_spectrum):
            query = spectrum["query_id"]
            scan = spectrum["scan_number"]
            label = f"Spectrum {ordinal}/{last_spectrum} (scan {scan})"
            path = args.output_dir / f"{query}.csv"
            if path.exists():
                if interactive:
                    spectrum_status.set_description_str(f"{label} | Checking saved CSV{last_timing}")
                checked = verify_csv(path, [query])
                reused += 1
            else:
                if interactive:
                    spectrum_status.set_description_str(f"{label}{last_timing}")
                    candidate_bar.set_postfix_str("", refresh=False)
                    candidate_bar.reset()
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
                        print(f"{label} | Candidate {completed_candidates:,}/{ROWS:,} | {sites}", flush=True)

                result, stats = score_spectrum(
                    scorer, forms, spectrum, gt, candidates, args.batch_size,
                    on_progress=show_progress, progress_seconds=1 if interactive else 5,
                )
                checked = save_spectrum_csv(path, lambda stream: csv.write_csv(result, stream, write_options=CSV_OPTIONS))
                if checked["top1_hits"] != stats["top1_is_gt"] or checked["unsupported"] != stats["unsupported"]:
                    raise ValueError(f"Written CSV does not match the scored result: {path}")
                seconds = perf_counter() - started
                if interactive:
                    last_timing = f" | Last saved scan {scan} in {seconds:.1f}s"
                    spectrum_status.set_description_str(f"{label}{last_timing}")
                else:
                    report("spectrum_completed", ordinal=ordinal, seconds=round(seconds, 3), **stats)
            hits += checked["top1_hits"]
            unsupported += checked["unsupported"]
        if interactive:
            spectrum_status.set_description_str(f"Spectra saved {len(spectra)}/{len(spectra)}{last_timing}")

    if len(spectra) == SPECTRA and unsupported != UNSUPPORTED:
        raise ValueError(f"Expected {UNSUPPORTED} rows containing U, found {unsupported}")
    report("complete", spectra=len(spectra), reused=reused, candidates=len(spectra) * ROWS, unsupported=unsupported,
           top1_hits=hits, top1_fraction=hits / len(spectra), seconds=round(perf_counter() - start, 3))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
