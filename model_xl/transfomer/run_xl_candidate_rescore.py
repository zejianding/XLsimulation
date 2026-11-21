"""Rescore Human Test candidates with the Model XL test cosine."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq


MAX_LENGTH = 40
COLUMNS = {
    "query_id": pa.string(),
    "candidate_rank": pa.int32(),
    "form_a_id": pa.int32(),
    "form_b_id": pa.int32(),
    "site_a_0based": pa.int8(),
    "site_b_0based": pa.int8(),
    "cosine": pa.float32(),
    "status": pa.string(),
    "is_gt": pa.uint8(),
}
HEADER = (",".join(COLUMNS) + "\n").encode("ascii")
CSV_OPTIONS = csv.WriteOptions(include_header=False, quoting_style="none")


def report(event, **details):
    print(json.dumps({"event": event, **details}), flush=True)


class Forms:
    def __init__(self, path):
        file = pq.ParquetFile(path)
        count = file.metadata.num_rows
        self.ids = np.empty(count, dtype=np.int32)
        self.aa = np.zeros((count, MAX_LENGTH), dtype=np.uint8)
        self.lengths = np.empty(count, dtype=np.uint8)
        self.oxidation = np.empty(count, dtype=np.uint64)
        self.cammask = np.empty(count, dtype=np.uint64)
        self.unsupported = np.empty(count, dtype=bool)
        self.k_counts = np.empty(count, dtype=np.uint8)
        allowed_sites = []
        allowed_ascii = np.zeros(256, dtype=bool)
        allowed_ascii[[0, *b"ACDEFGHIKLMNPQRSTVWYU"]] = True
        position = np.arange(MAX_LENGTH)
        bit = np.left_shift(np.uint64(1), position.astype(np.uint64))
        columns = ["form_id", "sequence", "length", "oxidation_mask", "modifications", "k_positions_0based"]
        use_allowed = "allowed_k_positions_0based" in file.schema_arrow.names
        if use_allowed:
            columns.append("allowed_k_positions_0based")

        offset = 0
        for batch in file.iter_batches(columns=columns, batch_size=65_536):
            end = offset + len(batch)
            sl = slice(offset, end)
            if any(batch[name].null_count for name in columns):
                raise ValueError("Peptidoform input contains null fields")
            lengths = batch["length"].to_numpy().astype(np.int64)
            self.ids[sl] = batch["form_id"].to_numpy()
            self.lengths[sl] = lengths
            ox = batch["oxidation_mask"].to_numpy()
            self.oxidation[sl] = ox
            if (lengths < 2).any() or (lengths > MAX_LENGTH).any():
                raise ValueError("Peptidoform length outside the Model XL grid")

            seq = batch["sequence"]
            _, offsets_buf, data_buf = seq.buffers()
            offsets = np.frombuffer(offsets_buf, dtype=np.int32)[seq.offset:seq.offset + len(seq) + 1]
            if not np.array_equal(np.diff(offsets), lengths):
                raise ValueError("Peptidoform sequence length mismatch")
            characters = np.frombuffer(data_buf, dtype=np.uint8)
            present = position < lengths[:, None]
            aa = self.aa[sl]
            aa[present] = characters[(offsets[:-1, None] + position)[present]]
            if not allowed_ascii[aa].all():
                raise ValueError("Unexpected residue outside the 20 standard amino acids and U")
            self.unsupported[sl] = (aa == ord("U")).any(axis=1)
            cammask = np.bitwise_or.reduce(np.where(aa == ord("C"), bit, np.uint64(0)), axis=1)
            self.cammask[sl] = cammask

            mods = batch["modifications"]
            mod_offsets = mods.offsets.to_numpy()
            values = mods.values.slice(int(mod_offsets[0]), int(mod_offsets[-1] - mod_offsets[0]))
            parents = np.repeat(np.arange(len(batch)), np.diff(mod_offsets))
            sites = values.field("position_0based").to_numpy()
            masses = values.field("mass_delta_da").to_numpy()
            cam = (masses == 57.021464) & (aa[parents, sites] == ord("C"))
            oxid = (masses == 15.994915) & (aa[parents, sites] == ord("M"))
            if (sites < 0).any() or (sites >= MAX_LENGTH).any() or not (cam | oxid).all():
                raise ValueError("Unexpected modification outside fixed C CAM and variable M oxidation")
            expected_mods = cammask | ox
            actual_mods = np.zeros(len(batch), dtype=np.uint64)
            np.bitwise_or.at(actual_mods, parents, bit[sites])
            if not np.array_equal(expected_mods, actual_mods) or len(np.unique(parents * MAX_LENGTH + sites)) != len(sites):
                raise ValueError("Modification list disagrees with the fixed CAM or oxidation mask")

            k = batch["k_positions_0based"]
            k_offsets = k.offsets.to_numpy()
            k_description = k.values.slice(int(k_offsets[0]), int(k_offsets[-1] - k_offsets[0])).to_numpy()
            k_rows, k_positions = np.nonzero(aa == ord("K"))
            if not np.array_equal(k_positions, k_description) or not np.array_equal(
                np.bincount(k_rows, minlength=len(batch)), np.diff(k_offsets)
            ):
                raise ValueError("K positions disagree with the peptidoform sequence")
            if use_allowed:
                allowed = batch["allowed_k_positions_0based"]
                k_offsets = allowed.offsets.to_numpy()
                k_positions = allowed.values.slice(int(k_offsets[0]), int(k_offsets[-1] - k_offsets[0])).to_numpy()
                k_rows = np.repeat(np.arange(len(batch)), np.diff(k_offsets))
                if ((k_positions < 0).any() or (k_positions >= MAX_LENGTH).any()
                        or not np.all(aa[k_rows, k_positions] == ord("K"))
                        or np.any((k_rows[1:] == k_rows[:-1]) & (k_positions[1:] <= k_positions[:-1]))):
                    raise ValueError("Allowed K positions must be an ordered subset of the peptidoform's K sites")
            self.k_counts[sl] = np.diff(k_offsets)
            allowed_sites.append(k_positions.astype(np.uint8) + 1)
            offset = end

        if offset != count or np.any(self.ids[1:] <= self.ids[:-1]):
            raise ValueError("Expected strictly increasing, complete peptidoform IDs")
        self.k_sites = np.concatenate(allowed_sites)
        self.k_offsets = np.cumsum(self.k_counts, dtype=np.int64) - self.k_counts
        self.sequence_order = self.aa.view(f"S{MAX_LENGTH}").ravel()

    def indices(self, form_ids):
        idx = np.searchsorted(self.ids, form_ids)
        if (idx >= len(self.ids)).any() or not np.array_equal(self.ids[idx], form_ids):
            raise ValueError("Candidate references an unknown peptidoform")
        return idx

    def canonical_swap(self, a, b, sa, sb):
        seq_a, seq_b = self.sequence_order[a], self.sequence_order[b]
        swap = seq_a > seq_b
        same_sequence = seq_a == seq_b
        same_a, same_b = a[same_sequence], b[same_sequence]
        ma = self.cammask[same_a] | self.oxidation[same_a]
        mb = self.cammask[same_b] | self.oxidation[same_b]
        diff = ma ^ mb
        first = diff & -diff
        later = ~(first | (first - np.uint64(1)))
        # Compare the ordered lists of modified positions, with a shorter prefix first.
        swap[same_sequence] = np.where(
            diff == 0,
            sa[same_sequence] > sb[same_sequence],
            np.where((ma & first) != 0, (mb & later) == 0, (ma & later) != 0),
        )
        return swap


def load_spectra(data):
    test = data / "test"
    truth = {
        gt["query_id"]: gt
        for gt in pq.ParquetFile(test / "ground_truth.parquet").read().to_pylist()
        if (gt["candidate_rank"] or 0) > 0
    }
    columns = ["query_id", "scan_number", "precursor_charge", "precursor_mz", "retention_time_seconds",
               "mz", "intensity", "crosslinker", "crosslink_mass_da", "candidate_count"]
    spectra = pq.ParquetFile(test / "spectra.parquet").read(columns=columns).to_pylist()
    selected = [(spec, truth[spec["query_id"]]) for spec in spectra if spec["query_id"] in truth]
    if not selected:
        raise ValueError("No spectra have ground truth in the candidate lists")
    if len(selected) != len(truth) or len({s["query_id"] for s, _ in selected}) != len(truth):
        raise ValueError("Expected exactly one spectrum per included ground-truth query")
    for spec, gt in selected:
        if spec["candidate_count"] is None or gt["candidate_rank"] > spec["candidate_count"]:
            raise ValueError(f"Ground-truth rank exceeds the candidate count for {spec['query_id']}")
    return selected


def score_spectrum(scorer, forms, spec, gt, candidate_dir, batch_size, *, on_progress=None, progress_seconds=60):
    query = spec["query_id"]
    charge, mz, rt = (spec[k] for k in ("precursor_charge", "precursor_mz", "retention_time_seconds"))
    if spec["crosslinker"] != "BS3" or spec["crosslink_mass_da"] != 138.06808:
        raise ValueError(f"Unexpected crosslinker for {query}")
    table = pq.ParquetFile(candidate_dir / f"{query}.parquet").read(columns=["rank", "form_a_id", "form_b_id"])
    rank, fa, fb = (table[key].to_numpy() for key in ("rank", "form_a_id", "form_b_id"))
    row_count = spec["candidate_count"]
    if not np.array_equal(rank, np.arange(1, row_count + 1)):
        raise ValueError(f"Expected {row_count} ordered ranks for {query}")
    ga, gb = gt["form_id_a"], gt["form_id_b"]
    is_gt = ((fa == ga) & (fb == gb)) | ((fa == gb) & (fb == ga))
    if is_gt.sum() != 1 or not is_gt[gt["candidate_rank"] - 1]:
        raise ValueError(f"Ground truth candidate does not match recorded rank for {query}")
    a, b = forms.indices(fa), forms.indices(fb)
    unsupported = forms.unsupported[a] | forms.unsupported[b]
    unsupported_count = int(unsupported.sum())
    if (unsupported & is_gt).any():
        raise ValueError(f"Ground truth unexpectedly contains U for {query}")

    pairs = np.flatnonzero(~unsupported)
    pairs = pairs[np.argsort(forms.lengths[a[pairs]] + forms.lengths[b[pairs]], kind="stable")]
    ka, kb = forms.k_counts[a[pairs]].astype(np.int64), forms.k_counts[b[pairs]].astype(np.int64)
    counts = ka * kb
    if np.any(counts == 0):
        raise ValueError(f"Candidate is missing a K site for {query}")
    ends = np.cumsum(counts)
    starts = ends - counts
    hyp = np.repeat(pairs, counts)
    combo = np.arange(len(hyp)) - np.repeat(starts, counts)
    b_width = forms.k_counts[b[hyp]]
    ha, hb = a[hyp], b[hyp]
    sa = forms.k_sites[forms.k_offsets[ha] + combo // b_width]
    sb = forms.k_sites[forms.k_offsets[hb] + combo % b_width]
    swap = forms.canonical_swap(ha, hb, sa, sb)
    steps = np.stack((np.where(swap, hb, ha), np.where(swap, ha, hb)), axis=1)
    sites = np.stack((np.where(swap, sb, sa), np.where(swap, sa, sb)), axis=1)
    peak_state = scorer.prepare_spectrum(spec["mz"], spec["intensity"])
    scores = np.empty(len(hyp), dtype=np.float32)
    bit = np.left_shift(np.uint64(1), np.arange(MAX_LENGTH, dtype=np.uint64))
    last_report = perf_counter()
    if on_progress is not None:
        on_progress(unsupported_count, 0, len(hyp))
    for start in range(0, len(hyp), batch_size):
        stop = min(start + batch_size, len(hyp))
        ids = steps[start:stop]
        lengths = forms.lengths[ids]
        width = int(lengths.max())
        aa = np.ascontiguousarray(forms.aa[ids, :width])
        ox = (forms.oxidation[ids, None] & bit[:width]) != 0
        scores[start:stop] = scorer.score(aa, ox, lengths, sites[start:stop], charge, mz, rt, peak_state)
        now = perf_counter()
        if now - last_report >= progress_seconds or (on_progress is not None and stop == len(hyp)):
            if on_progress is None:
                report("spectrum_progress", query_id=query, completed_combinations=stop, combinations=len(hyp))
            else:
                completed = unsupported_count + int(np.searchsorted(ends, stop, side="right"))
                on_progress(completed, stop, len(hyp))
            last_report = now
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1.000001).any():
        raise ValueError(f"Invalid cosine values for {query}")

    # Site combinations are enumerated in ascending (A site, B site) order.
    best = np.maximum.reduceat(scores, starts)
    best_rows = np.minimum.reduceat(np.where(scores == np.repeat(best, counts), np.arange(len(hyp)), len(hyp)), starts)
    if (best_rows >= len(hyp)).any():
        raise ValueError(f"Could not recover maximizing sites for {query}")
    cosine = np.full(row_count, np.nan, dtype=np.float32)
    cosine[pairs] = best
    best_a = np.zeros(row_count, dtype=np.int8)
    best_b = np.zeros(row_count, dtype=np.int8)
    best_a[pairs], best_b[pairs] = sa[best_rows] - 1, sb[best_rows] - 1
    top = int(np.nanargmax(cosine))
    result = pa.Table.from_arrays([
        pa.repeat(query, row_count), pa.array(rank), pa.array(fa), pa.array(fb),
        pa.array(best_a, mask=unsupported), pa.array(best_b, mask=unsupported),
        pa.array(cosine, mask=unsupported), pa.array(np.where(unsupported, "unsupported_residue", "ok")),
        pa.array(is_gt.astype(np.uint8)),
    ], schema=pa.schema(COLUMNS))
    return result, {
        "query_id": query, "scan_number": spec["scan_number"], "combinations": len(hyp),
        "unsupported": unsupported_count, "top1_rank": int(rank[top]),
        "top1_cosine": float(cosine[top]), "top1_is_gt": bool(is_gt[top]),
    }


def resume_output(stream, spectra):
    query_ids = [spec["query_id"] for spec, _ in spectra]
    stream.seek(0, os.SEEK_END)
    end = stream.tell()
    if end == 0:
        stream.write(HEADER)
        stream.flush()
        os.fsync(stream.fileno())
        return 0
    stream.seek(0)
    if stream.readline() != HEADER:
        raise ValueError("Existing CSV header does not match this run")
    if end == len(HEADER):
        return 0

    tail_start = max(len(HEADER), end - 16 * 1024 * 1024)
    stream.seek(tail_start)
    tail = stream.read()
    last_nl = tail.rfind(b"\n")
    if last_nl < 0 and tail_start == len(HEADER):
        prefix = f"{query_ids[0]},1,".encode("ascii")
        if tail.startswith(prefix) or prefix.startswith(tail):
            stream.truncate(len(HEADER))
            stream.seek(len(HEADER))
            report("recovered_partial_csv", discarded_bytes=len(tail))
            return 0
    previous_nl = tail.rfind(b"\n", 0, last_nl)
    if last_nl < 0 or (previous_nl < 0 and tail_start != len(HEADER)):
        raise ValueError("CSV tail has no recoverable row boundary")
    last_line = tail[previous_nl + 1:last_nl].split(b",")
    if len(last_line) != len(COLUMNS):
        raise ValueError("CSV tail does not match the expected CSV row format")
    query = last_line[0].decode("ascii")
    ordinal = query_ids.index(query)
    rank = int(last_line[1])
    row_count = spectra[ordinal][0]["candidate_count"]
    if not 1 <= rank <= row_count:
        raise ValueError(f"Existing CSV rank exceeds the candidate count for {query}")
    if rank == row_count and last_line[-1] in (b"0", b"1"):
        offset = tail_start + last_nl + 1
        completed = ordinal + 1
    else:
        first_row_prefix = f"{query},1,".encode("ascii")
        marker = b"\n" + first_row_prefix
        scan_end = tail_start + last_nl
        while True:
            scan_start = max(len(HEADER) - 1, scan_end - 16 * 1024 * 1024)
            stream.seek(scan_start)
            index = stream.read(scan_end - scan_start).rfind(marker)
            if index >= 0:
                offset = scan_start + index + 1
                break
            if scan_start == len(HEADER) - 1:
                raise ValueError("Could not identify the start of an incomplete spectrum block")
            scan_end = scan_start + len(marker) - 1
        completed = ordinal
    if offset != end:
        stream.truncate(offset)
        report("recovered_partial_csv", discarded_bytes=end - offset)
    stream.seek(offset)
    return completed


def verify_csv(output, spectra):
    reader = csv.open_csv(
        output, read_options=csv.ReadOptions(block_size=16 * 1024 * 1024),
        convert_options=csv.ConvertOptions(column_types=COLUMNS),
    )
    ordinal = position = hit = unsupported = candidates = 0
    top_score, top_is_gt = -math.inf, False
    for batch in reader:
        ranks = batch["candidate_rank"].to_numpy()
        scores = batch["cosine"].to_numpy(zero_copy_only=False)
        is_gt = batch["is_gt"].to_numpy()
        is_ok = pc.equal(batch["status"], "ok").to_numpy(zero_copy_only=False)
        blank = np.isnan(scores)
        if not np.array_equal(is_ok, ~blank) or not pc.all(pc.is_in(
            batch["status"], value_set=pa.array(["ok", "unsupported_residue"])
        )).as_py():
            raise ValueError("CSV score and status disagree")
        if not np.array_equal(blank, batch["site_a_0based"].is_null().to_numpy(zero_copy_only=False)) or not np.array_equal(
            blank, batch["site_b_0based"].is_null().to_numpy(zero_copy_only=False)
        ):
            raise ValueError("CSV score and site nullability disagree")
        if (is_gt > 1).any() or ((is_gt != 0) & blank).any():
            raise ValueError("CSV has invalid Ground Truth flags")
        unsupported += int(blank.sum())
        start = 0
        while start != len(batch):
            if ordinal >= len(spectra):
                raise ValueError("CSV has more spectra than expected")
            spec, gt = spectra[ordinal]
            row_count = spec["candidate_count"]
            length = min(len(batch) - start, row_count - position)
            stop = start + length
            if not np.array_equal(ranks[start:stop], np.arange(position + 1, position + length + 1)):
                raise ValueError("CSV rank sequence is invalid")
            if not pc.all(pc.equal(batch["query_id"].slice(start, length), spec["query_id"])).as_py():
                raise ValueError("CSV spectrum sequence is invalid")
            if not np.array_equal(is_gt[start:stop] != 0, ranks[start:stop] == gt["candidate_rank"]):
                raise ValueError("CSV ground-truth rank does not match the data package")
            slice_scores = np.where(blank[start:stop], -np.inf, scores[start:stop])
            top = int(np.argmax(slice_scores))
            if slice_scores[top] > top_score:
                top_score, top_is_gt = float(slice_scores[top]), bool(is_gt[start + top])
            position += length
            start = stop
            if position == row_count:
                if not math.isfinite(top_score):
                    raise ValueError("CSV spectrum is missing a valid score")
                hit += int(top_is_gt)
                ordinal += 1
                candidates += row_count
                position = 0
                top_score, top_is_gt = -math.inf, False
    if position or ordinal != len(spectra):
        raise ValueError("CSV spectrum or candidate totals are invalid")
    return {"spectra": ordinal, "candidates": candidates, "unsupported": unsupported,
            "top1_hits": hit, "top1_fraction": hit / ordinal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-spectra", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_spectra is not None and args.limit_spectra < 1):
        parser.error("Invalid batch size or spectrum limit")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
    from xl_rescore_inference import XLRescoreInference

    start = perf_counter()
    spectra = load_spectra(args.data)[:args.limit_spectra]
    mode = "r+b" if args.output.exists() else "x+b"
    with args.output.open(mode) as output:
        first = resume_output(output, spectra)
        if first != len(spectra):
            forms = Forms(args.data / "peptidoforms.parquet")
            scorer = XLRescoreInference(args.checkpoint, "cuda")
            report("ready", completed=first, seconds=round(perf_counter() - start, 3))
            for index in range(first, len(spectra)):
                time = perf_counter()
                spec, gt = spectra[index]
                report("spectrum_started", ordinal=index + 1, query_id=spec["query_id"], scan_number=spec["scan_number"])
                result, stats = score_spectrum(
                    scorer, forms, spec, gt, args.data / "test" / "candidates", args.batch_size
                )
                committed = output.tell()
                try:
                    csv.write_csv(result, output, write_options=CSV_OPTIONS)
                    output.flush()
                    os.fsync(output.fileno())
                except BaseException:
                    output.flush()
                    output.truncate(committed)
                    raise
                report("spectrum_completed", ordinal=index + 1, seconds=round(perf_counter() - time, 3), **stats)
    report("complete", seconds=round(perf_counter() - start, 3), **verify_csv(args.output, spectra))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
