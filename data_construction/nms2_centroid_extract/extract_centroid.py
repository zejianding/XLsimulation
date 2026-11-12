"""Extract complete MS2 centroid spectra in Prosit-XL peptide-pair order.

Source data are read only. Each source split's first half defines the output
rows; the second half must be the exact A/B-swapped counterpart. A RAW scan
can therefore appear more than once in the output, with its original public
charge for each peptide-pair record.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import threading
import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\XL\prosit-xl")
PWSH = Path(r"C:\Users\zejiading2\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe")
LIBRARY = Path(r"E:\FragPipe\FragPipe-24.0\tools\MSFragger-4.4.1\ext\thermo")
HEADER = struct.Struct("<iiddiB")
MAGIC = b"NMS2C001"
SPLITS = {"train": "train", "val": "val", "test": "holdout"}
FIELDS = [
    pa.field("raw_file", pa.string(), nullable=False),
    pa.field("scan_number", pa.int64(), nullable=False),
    pa.field("mz", pa.list_(pa.float64()), nullable=False),
    pa.field("intensity", pa.list_(pa.float64()), nullable=False),
    pa.field("precursor_mz", pa.float64(), nullable=False),
    pa.field("precursor_charge", pa.int16(), nullable=False),
    pa.field("retention_time_seconds", pa.float64(), nullable=False),
]


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_plan():
    rows = {}
    source_hashes = {}
    stats = {}
    required = defaultdict(set)
    for target, original in SPLITS.items():
        h5_path = SOURCE / f"nms2_{original}.h5"
        parquet_path = SOURCE / f"nms2_{original}.parquet"
        for path in (h5_path, parquet_path):
            source_hashes[str(path)] = digest_file(path)
        with h5py.File(h5_path, "r") as data:
            names_bytes = data["raw_file"][...]
            scans_float = data["scan_number"][...]
            onehot = data["precursor_charge_onehot"][...]
            count = len(names_bytes)
            if count % 2:
                raise ValueError(f"Odd number of source rows: {h5_path}")
            half = count // 2
            if not np.all(np.isfinite(scans_float)) or not np.all(scans_float == np.floor(scans_float)):
                raise ValueError(f"Non-integer scan numbers: {h5_path}")
            if not np.all((onehot == 0) | (onehot == 1)) or not np.all(onehot.sum(axis=1) == 1):
                raise ValueError(f"Invalid charge one-hot: {h5_path}")
            for field in ("raw_file", "scan_number", "precursor_charge_onehot"):
                if not np.array_equal(data[field][:half], data[field][half:]):
                    raise ValueError(f"Source half pairing failed: {target}/{field}")
            for left, right in (("sequence_integer_1", "sequence_integer_2"),
                                ("sequence_integer_2", "sequence_integer_1"),
                                ("alpha_position_crosslinker", "beta_position_crosslinker"),
                                ("beta_position_crosslinker", "alpha_position_crosslinker")):
                if not np.array_equal(data[left][:half], data[right][half:]):
                    raise ValueError(f"Source A/B swap failed: {target}/{left}")
            public = pq.read_table(parquet_path, columns=["raw_file", "scan_number", "precursor_charge_onehot"])
            if public.num_rows != count:
                raise ValueError(f"H5/Parquet row count mismatch: {target}")
            for field, expected in (("raw_file", names_bytes), ("scan_number", scans_float), ("precursor_charge_onehot", onehot)):
                actual = np.asarray(public[field].to_pylist())
                if not np.array_equal(actual, expected):
                    raise ValueError(f"H5/Parquet mismatch: {target}/{field}")
            names = [x.decode("utf-8") for x in names_bytes[:half]]
            scans = scans_float[:half].astype(np.int64)
            charges = onehot[:half].argmax(axis=1).astype(np.int16) + 1
            rows[target] = [(name, int(scan), int(charge)) for name, scan, charge in zip(names, scans, charges)]
            charge_sets = defaultdict(set)
            for name, scan, charge in rows[target]:
                required[name].add(scan)
                charge_sets[(name, scan)].add(charge)
            stats[target] = {
                "source_rows": count,
                "output_rows": half,
                "unique_raw_scans": len(charge_sets),
                "charge_conflicting_scans": sum(len(x) > 1 for x in charge_sets.values()),
            }
    inventory = defaultdict(list)
    for path in (SOURCE / "NMS2_raw").rglob("*.raw"):
        inventory[path.stem.casefold()].append(path)
    jobs = []
    raw_snapshot = {}
    for name in sorted(required):
        matches = inventory.get(name.casefold(), [])
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one RAW for {name}: {matches}")
        path = matches[0]
        file_stat = path.stat()
        raw_snapshot[str(path)] = {"size": file_stat.st_size, "mtime_ns": file_stat.st_mtime_ns}
        jobs.append({"raw_index": len(jobs), "raw_file": name, "path": str(path), "scans": sorted(required[name])})
    keys = {split: {(name, scan) for name, scan, _ in data} for split, data in rows.items()}
    overlaps = {f"{a}-{b}": len(keys[a] & keys[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    plan = {"files": jobs, "source_sha256": source_hashes, "source_stats": stats,
            "raw_snapshot": raw_snapshot, "cross_split_shared_scans": overlaps,
            "unique_scans": sum(len(job["scans"]) for job in jobs)}
    return rows, plan


def read_exact(stream, length):
    chunks = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"RAW helper stream ended with {remaining}/{length} bytes missing")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def records_from_raw(plan, job_path, stderr_path):
    write_json(job_path, {"files": plan["files"]})
    command = [str(PWSH), "-NoLogo", "-NoProfile", "-NonInteractive", "-File",
               str(ROOT / "read_centroid.ps1"), "-JobPath", str(job_path),
               "-LibraryDirectory", str(LIBRARY)]
    errors = []
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW)

    def consume_errors():
        done = 0
        with stderr_path.open("w", encoding="utf-8") as log_file:
            for line in iter(process.stderr.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                log_file.write(text + "\n")
                log_file.flush()
                errors.append(text)
                if len(errors) > 8:
                    errors.pop(0)
                done += 1
                if done % 20 == 0:
                    log("raw_progress", raw_files_finished=done, raw_files_total=len(plan["files"]))

    drain = threading.Thread(target=consume_errors, daemon=True)
    drain.start()
    try:
        if read_exact(process.stdout, 8) != MAGIC:
            raise ValueError("Unexpected RAW helper stream signature")
        while True:
            first = read_exact(process.stdout, 4)
            if struct.unpack("<i", first)[0] == -1:
                break
            header = first + read_exact(process.stdout, HEADER.size - 4)
            raw_index, scan, precursor, rt, count, method = HEADER.unpack(header)
            if not 0 <= raw_index < len(plan["files"]) or count < 0 or method not in (0, 1, 2):
                raise ValueError("Invalid RAW helper record header")
            body = read_exact(process.stdout, count * 16)
            mz = np.frombuffer(body, dtype="<f8", count=count)
            intensity = np.frombuffer(body, dtype="<f8", count=count, offset=count * 8)
            if not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensity)):
                raise ValueError(f"Nonfinite peaks at RAW index {raw_index} scan {scan}")
            if not np.isfinite(precursor) or precursor <= 0 or not np.isfinite(rt) or rt < 0:
                raise ValueError(f"Invalid scan metadata at RAW index {raw_index} scan {scan}")
            yield header, body, (raw_index, scan, precursor, rt, count, method), mz, intensity
        extra = process.stdout.read()
        code = process.wait()
        drain.join()
        if extra or code:
            raise RuntimeError(f"RAW helper failed (exit={code}, trailing_bytes={len(extra)}): {errors}")
    except BaseException as exc:
        process.kill()
        process.wait()
        drain.join(timeout=5)
        raise RuntimeError(f"RAW extraction failed: {exc}; helper stderr: {errors}") from exc
    finally:
        process.stdout.close()
        process.stderr.close()


def schema_for(split, plan):
    metadata = {
        "source_split": SPLITS[split],
        "row_mapping": "Output row i corresponds to source H5/Parquet rows i and i + source_row_count/2 (zero-based). Different peptide pairs sharing a scan retain separate output rows, even with identical charges.",
        "charge_source": "Prosit-XL precursor_charge_onehot, argmax + 1; never replaced with the RAW charge",
        "precursor_mz_source": "Thermo RAW scan filter GetReaction(0).PrecursorMass (reaction/isolation precursor m/z)",
        "retention_time_source": "Thermo RAW ScanStatistics.StartTime * 60, seconds",
        "peak_source": "Stored centroid: GetSegmentedScanFromScanNumber. Profile: existing GetCentroidStream(false), else Thermo Scan.ToCentroid PreferredMasses/PreferredIntensities.",
        "peak_processing": "No peak filtering, normalization, truncation, isotope removal, sorting, padding, or precision reduction. Float64 m/z and intensity arrays preserve reader output.",
        "raw_reader_version": "5.0.0.93",
        "source_sha256": json.dumps(plan["source_sha256"], sort_keys=True),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return pa.schema(FIELDS, metadata={k.encode(): v.encode() for k, v in metadata.items()})


def smoke(rows, plan):
    # One small sample from each PRIDE dataset exercises multiple instruments.
    selected = []
    seen_projects = set()
    for job in plan["files"]:
        project = Path(job["path"]).parent.name
        if project not in seen_projects:
            selected.append({**job, "raw_index": len(selected), "scans": job["scans"][:2]})
            seen_projects.add(project)
    sample_plan = {**plan, "files": selected}
    count = 0
    modes = Counter()
    sample_rows = []
    for _, _, record, mz, intensity in records_from_raw(sample_plan, ROOT / "smoke_jobs.json", ROOT / "smoke_reader.log"):
        index, scan, precursor, rt, _, method = record
        name = selected[index]["raw_file"]
        source_charge = next(charge for records in rows.values() for raw, s, charge in records if raw == name and s == scan)
        sample_rows.append({"raw_file": name, "scan_number": scan, "mz": mz, "intensity": intensity,
                            "precursor_mz": precursor, "precursor_charge": source_charge, "retention_time_seconds": rt})
        count += 1
        modes[str(method)] += 1
    if count != sum(len(job["scans"]) for job in selected):
        raise ValueError("Smoke test scan count mismatch")
    table = pa.Table.from_pylist(sample_rows, schema=schema_for("train", plan))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    restored = pq.read_table(pa.BufferReader(sink.getvalue()))
    if not table.equals(restored):
        raise ValueError("Smoke test Parquet roundtrip changed data")
    log("smoke_passed", spectra=count, projects=sorted(seen_projects), centroid_methods=dict(modes), fields=table.column_names)


def extract(rows, plan, output_dir):
    output_dir = output_dir.resolve()
    if output_dir != SOURCE.resolve():
        raise ValueError("This extraction is restricted to the intended dataset directory")
    outputs = {split: output_dir / f"{split}_centroid.parquet" for split in SPLITS}
    partials = {split: path.with_suffix(".parquet.partial") for split, path in outputs.items()}
    for path in [*outputs.values(), *partials.values()]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    spool_path = ROOT / "centroid_peaks.bin"
    if spool_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction spool: {spool_path}")
    write_json(ROOT / "input_audit.json", plan)
    locations = {}
    modes = Counter()
    total_peaks = 0
    empty_spectra = 0
    unsorted_spectra = 0
    negative_intensity_spectra = 0
    expected = {(job["raw_file"], scan) for job in plan["files"] for scan in job["scans"]}
    start = time.monotonic()
    with spool_path.open("xb") as spool:
        for header, body, record, mz, intensity in records_from_raw(plan, ROOT / "raw_jobs.json", ROOT / "raw_reader.log"):
            raw_index, scan, precursor, rt, count, method = record
            key = (plan["files"][raw_index]["raw_file"], scan)
            if key not in expected or key in locations:
                raise ValueError(f"Unexpected or duplicated helper spectrum: {key}")
            locations[key] = (spool.tell(), len(header) + len(body), hashlib.sha256(header + body).digest())
            spool.write(header)
            spool.write(body)
            modes[str(method)] += 1
            total_peaks += count
            empty_spectra += count == 0
            unsorted_spectra += bool(np.any(np.diff(mz) < 0))
            negative_intensity_spectra += bool(np.any(intensity < 0))
            if len(locations) % 5000 == 0:
                log("spectra_read", count=len(locations), expected=len(expected), total_peaks=total_peaks,
                    elapsed_seconds=round(time.monotonic() - start, 1))
    if set(locations) != expected:
        raise ValueError("Not all requested RAW scans were extracted")
    log("all_spectra_read", spectra=len(locations), peaks=total_peaks, centroid_methods=dict(modes))
    write_json(ROOT / "spool_index.json", [{"raw_file": k[0], "scan": k[1], "offset": v[0], "bytes": v[1], "sha256": v[2].hex()} for k, v in locations.items()])
    split_peaks = {}
    with spool_path.open("rb") as spool:
        for split, public_rows in rows.items():
            schema = schema_for(split, plan)
            batch = []
            peak_count = 0
            with pq.ParquetWriter(partials[split], schema, compression="zstd", compression_level=3,
                                  use_dictionary=["raw_file"], write_statistics=True) as writer:
                for name, scan, charge in public_rows:
                    offset, length, checksum = locations[(name, scan)]
                    spool.seek(offset)
                    payload = read_exact(spool, length)
                    if hashlib.sha256(payload).digest() != checksum:
                        raise ValueError("Extraction spool checksum mismatch")
                    _, stored_scan, precursor, rt, count, _ = HEADER.unpack(payload[:HEADER.size])
                    if stored_scan != scan:
                        raise ValueError("Extraction spool scan mismatch")
                    mz = np.frombuffer(payload, dtype="<f8", count=count, offset=HEADER.size)
                    intensity = np.frombuffer(payload, dtype="<f8", count=count, offset=HEADER.size + count * 8)
                    batch.append({"raw_file": name, "scan_number": scan, "mz": mz, "intensity": intensity,
                                  "precursor_mz": precursor, "precursor_charge": charge, "retention_time_seconds": rt})
                    peak_count += count
                    if len(batch) >= 512:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            split_peaks[split] = peak_count
            log("parquet_written", split=split, rows=len(public_rows), peaks=peak_count,
                bytes=partials[split].stat().st_size)
    validation = validate_outputs(rows, plan, partials, locations, spool_path)
    verify_legacy_sample(locations, spool_path)
    for path, original_hash in plan["source_sha256"].items():
        if digest_file(Path(path)) != original_hash:
            raise ValueError(f"Source dataset changed during extraction: {path}")
    for path, before in plan["raw_snapshot"].items():
        after = Path(path).stat()
        if (after.st_size, after.st_mtime_ns) != (before["size"], before["mtime_ns"]):
            raise ValueError(f"Source RAW changed during extraction: {path}")
    report = {"status": "validated", "source_stats": plan["source_stats"],
              "unique_raw_files": len(plan["files"]), "unique_raw_scans": len(locations),
              "unique_scan_peaks": total_peaks, "centroid_methods": dict(modes),
              "centroid_method_names": {"0": "native stored centroid", "1": "stored centroid stream for profile", "2": "vendor profile conversion"},
              "empty_spectra": empty_spectra, "unsorted_spectra": unsorted_spectra,
              "negative_intensity_spectra": negative_intensity_spectra,
              "cross_split_shared_scans": plan["cross_split_shared_scans"],
              "validation": validation, "source_files_unchanged": True,
              "peak_counts_per_split": split_peaks, "elapsed_seconds": round(time.monotonic() - start, 1)}
    for split, partial in partials.items():
        partial.rename(outputs[split])
    report["outputs"] = {split: {"path": str(path), "bytes": path.stat().st_size,
                                  "rows": len(rows[split]), "sha256": digest_file(path)} for split, path in outputs.items()}
    write_json(ROOT / "extraction_report.json", report)
    log("complete", **report)


def validate_outputs(rows, plan, paths, locations, spool_path):
    result = {}
    with spool_path.open("rb") as spool:
        for split, path in paths.items():
            file = pq.ParquetFile(path)
            if file.schema_arrow.names != [field.name for field in FIELDS]:
                raise ValueError(f"Unexpected output columns: {split}")
            if not file.schema_arrow.remove_metadata().equals(pa.schema(FIELDS)):
                raise ValueError(f"Unexpected output field types: {split}")
            index = 0
            for batch in file.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    expected = rows[split][index]
                    if (row["raw_file"], row["scan_number"], row["precursor_charge"]) != expected:
                        raise ValueError(f"Public row order/charge changed: {split}/{index}")
                    offset, length, checksum = locations[(expected[0], expected[1])]
                    spool.seek(offset)
                    header = read_exact(spool, HEADER.size)
                    raw_index, scan, precursor, rt, count, method = HEADER.unpack(header)
                    if len(row["mz"]) != count or len(row["intensity"]) != count:
                        raise ValueError(f"Peak count changed: {split}/{index}")
                    reconstructed = HEADER.pack(raw_index, scan, row["precursor_mz"], row["retention_time_seconds"], count, method)
                    reconstructed += np.asarray(row["mz"], dtype="<f8").tobytes()
                    reconstructed += np.asarray(row["intensity"], dtype="<f8").tobytes()
                    if hashlib.sha256(reconstructed).digest() != checksum:
                        raise ValueError(f"Peak or metadata values changed during Parquet roundtrip: {split}/{index}")
                    index += 1
            if index != len(rows[split]):
                raise ValueError(f"Output row count mismatch: {split}")
            result[split] = {"rows_checked": index, "public_order_and_charge_exact": True,
                             "all_peak_arrays_and_raw_metadata_exact": True}
            log("validation_passed", split=split, **result[split])
    return result


def verify_legacy_sample(locations, spool_path):
    path = ROOT.parents[1] / "验证真实谱和过滤谱的相似度" / "selected_raw_peaks.json"
    if not path.exists():
        raise FileNotFoundError(f"Independent reference sample is missing: {path}")
    reference = json.loads(path.read_text(encoding="utf-8-sig"))
    name = Path(reference["raw_path"]).stem
    checked = 0
    with spool_path.open("rb") as spool:
        for row in reference["spectra"]:
            offset, length, _ = locations[(name, row["scan_number"])]
            spool.seek(offset)
            payload = read_exact(spool, length)
            _, _, _, _, count, method = HEADER.unpack(payload[:HEADER.size])
            mz = np.frombuffer(payload, dtype="<f8", count=count, offset=HEADER.size)
            intensity = np.frombuffer(payload, dtype="<f8", count=count, offset=HEADER.size + count * 8)
            if method != 0 or not np.array_equal(mz, row["mz"]) or not np.array_equal(intensity, row["intensity"]):
                raise ValueError(f"Independent existing RAW reference mismatch: {name}/{row['scan_number']}")
            checked += 1
    log("independent_reference_passed", spectra=checked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "extract"), required=True)
    args = parser.parse_args()
    rows, plan = load_plan()
    log("input_audit_passed", splits=plan["source_stats"], raw_files=len(plan["files"]),
        unique_scans=plan["unique_scans"], cross_split_shared_scans=plan["cross_split_shared_scans"])
    if args.mode == "smoke":
        smoke(rows, plan)
    else:
        extract(rows, plan, SOURCE)


if __name__ == "__main__":
    main()
