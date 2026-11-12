"""Build and validate the structured NumPy datasets for XL spectrum training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, islice
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from AverageSpectrum_batch import average_peaks_by_tolerance
from trainingExamples import (
    H2O,
    MAX_CLEAVAGE_LEN,
    MAX_PEPTIDE_LEN,
    NUM_TOKENS,
    generate_by_ions_with_mods,
    normalize_target_intensity,
    parse_sequence_with_mods,
)
from sequence_token_map import _MZ_EQ_TOL
from token_table import MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, build_charge_tokens


SPLITS = {"train": "train", "val": "val", "test": "holdout"}
LINKER_MASSES = {"BS3": 138.06808}
PEAK_TOLERANCE = 0.05
MIN_VALID_MZ = 100.0
CLIP_INTENSITY = 50000.0
CHANNEL_NAMES = ("B1+", "B2+", "B3+", "Y1+", "Y2+", "Y3+")
# Retain the wider layout to verify previously built v3 datasets.
V3_CHANNEL_NAMES = (*CHANNEL_NAMES, "B4+", "Y4+")
NUM_CHANNELS = len(CHANNEL_NAMES)
CHANNEL_INDEX = {name: index for index, name in enumerate(V3_CHANNEL_NAMES)}
CHANNEL_CHARGES = np.array([int(name[1:-1]) for name in V3_CHANNEL_NAMES])
B_CHANNELS = np.array([name.startswith("B") for name in V3_CHANNEL_NAMES])
CHARGE_TOKENS = build_charge_tokens(max_charge=int(CHANNEL_CHARGES.max()))
TOKEN_OFFSETS = np.column_stack([CHARGE_TOKENS[z] for z in CHANNEL_CHARGES])
RESIDUE_TOKENS = np.array(RESIDUE_EVENT_TOKEN_IDS)
BASE_SHAPE = (2, MAX_CLEAVAGE_LEN, NUM_CHANNELS)
GRID_SHAPE = (2, MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS)
SCHEMA_VERSION = 2
MAIN_TOKEN_ID = MAIN_TOKEN_IDS[0]

RESIDUE_CODES = dict(enumerate("ACDEFGHIKLMNPQRSTVWY", start=1))
RESIDUE_CODES.update({21: "M", 22: "K", 24: "C"})
RESIDUE_MODS = {2: 57.021464, 21: 15.994915}
MEMBER_SCHEMA = pa.schema([
    ("split", pa.string()),
    ("group_id", pa.string()),
    ("dataset", pa.string()),
    ("raw_file", pa.string()),
    ("scan_number", pa.int64()),
    ("centroid_row", pa.int64()),
    ("nms2_row", pa.int64()),
    ("nms2_mirror_row", pa.int64()),
])


class GroupKey(NamedTuple):
    sequence_a: str
    site_a: int
    sequence_b: str
    site_b: int
    crosslinker: str
    charge: int


SpectrumID = tuple[str, str, int]
GroupMembers = dict[SpectrumID, list[int]]


@dataclass(frozen=True, order=True)
class Chain:
    bare_sequence: str
    modifications: tuple[tuple[int, float], ...]
    site: int
    sequence: str


@dataclass
class SplitData:
    name: str
    annotation_path: Path
    centroid_path: Path
    precursor_mz: np.ndarray
    rt_seconds: np.ndarray
    spectrum_ids: set[SpectrumID]
    groups: dict[GroupKey, GroupMembers]


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=32768)
def decode_chain(codes, site):
    length = next((i for i, code in enumerate(codes) if code == 0), len(codes))
    require(2 <= length <= MAX_PEPTIDE_LEN and not any(codes[length:]),
            f"Invalid peptide length or padding: {codes}")
    codes = codes[:length]
    require(1 <= site <= length and codes[site - 1] == 22 and codes.count(22) == 1,
            f"Expected a single linked K at site {site}: {codes}")
    require(all(code in RESIDUE_CODES for code in codes), f"Unknown NMS2 residue code: {codes}")
    residues = [RESIDUE_CODES[code] for code in codes]
    modifications = tuple((i, RESIDUE_MODS[code]) for i, code in enumerate(codes, start=1)
                          if code in RESIDUE_MODS)
    tokens = residues.copy()
    for i, mass in modifications:
        tokens[i - 1] += f"[{mass:+.6f}]"
    return Chain("".join(residues), modifications, int(site), "".join(tokens))


def integer_column(values, label):
    source = np.asarray(values)
    require(np.isfinite(source).all() and np.equal(source, np.trunc(source)).all(),
            f"{label} must contain finite integers")
    return source.astype(np.int64)


def read_split(input_dir, split):
    annotation_path = input_dir / f"nms2_{SPLITS[split]}.parquet"
    centroid_path = input_dir / f"{split}_centroid.parquet"
    seq_fields = ("sequence_integer_1", "sequence_integer_2")
    site_fields = ("alpha_position_crosslinker", "beta_position_crosslinker")
    identity_fields = ("raw_file", "scan_number", "crosslinker", "precursor_charge_onehot")
    ann = pq.read_table(annotation_path, columns=[*seq_fields, *site_fields, *identity_fields]).to_pydict()
    meta = pq.read_table(centroid_path, columns=[
        "dataset", "raw_file", "scan_number", "precursor_charge", "precursor_mz", "retention_time_seconds",
    ]).to_pydict()
    n = len(meta["scan_number"])
    require(len(ann["raw_file"]) == 2 * n, f"{split}: expected two annotation rows per centroid row")
    for fields in (seq_fields, site_fields):
        a, b = (ann[field] for field in fields)
        require(a[:n] == b[n:] and b[:n] == a[n:], f"{split}: A/B mirror mismatch in {fields}")
    for field in identity_fields:
        require(ann[field][:n] == ann[field][n:], f"{split}: mirror mismatch in {field}")

    onehot = np.asarray(ann["precursor_charge_onehot"][:n])
    require(onehot.ndim == 2 and np.isin(onehot, (0, 1)).all() and (onehot.sum(axis=1) == 1).all(),
            f"{split}: invalid precursor charge one-hot")
    charges = onehot.argmax(axis=1) + 1
    scans = integer_column(ann["scan_number"][:n], f"{split} scan_number")
    require([value.decode("utf-8") for value in ann["raw_file"][:n]] == meta["raw_file"]
            and np.array_equal(scans, meta["scan_number"])
            and np.array_equal(charges, meta["precursor_charge"]),
            f"{split}: annotation and centroid row alignment mismatch")

    precursor_mz = np.asarray(meta["precursor_mz"], dtype=np.float64)
    rt_seconds = np.asarray(meta["retention_time_seconds"], dtype=np.float64)
    require(np.isfinite(precursor_mz).all() and (precursor_mz > 0).all() and np.isfinite(rt_seconds).all(),
            f"{split}: invalid precursor m/z or retention time")
    codes_a, codes_b = (integer_column(ann[field][:n], f"{split} {field}") for field in seq_fields)
    sites_a, sites_b = (integer_column(ann[field][:n], f"{split} {field}") for field in site_fields)
    spectrum_ids = list(zip(meta["dataset"], meta["raw_file"], meta["scan_number"]))
    groups = defaultdict(dict)
    for row, (sid, linker, z) in enumerate(zip(spectrum_ids, ann["crosslinker"], charges)):
        linker = linker.decode("utf-8")
        require(linker in LINKER_MASSES, f"{split}: unsupported crosslinker {linker}")
        a, b = sorted((
            decode_chain(tuple(codes_a[row]), sites_a[row]),
            decode_chain(tuple(codes_b[row]), sites_b[row]),
        ))
        key = GroupKey(a.sequence, a.site, b.sequence, b.site, linker, int(z))
        groups[key].setdefault(sid, []).append(row)
    return SplitData(split, annotation_path, centroid_path, precursor_mz, rt_seconds,
                     set(spectrum_ids), dict(groups))


def group_id(key):
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=16384)
def linear_peptide(sequence, num_channels):
    residues, nterm_shift = parse_sequence_with_mods(sequence)
    neutral_mass = sum(mass for _, mass in residues) + nterm_shift + H2O
    base = np.zeros((len(residues) - 1, num_channels), dtype=np.float64)
    max_fragment_charge = int(CHANNEL_CHARGES[:num_channels].max())
    for ion in generate_by_ions_with_mods(sequence, max_charge=max_fragment_charge):
        channel = CHANNEL_INDEX[f"{ion['ion_type']}{ion['charge']}+"]
        base[ion["cleavage_index"] - 1, channel] = ion["mz"]
    return base, neutral_mass


def build_theoretical_mz(key, num_channels=NUM_CHANNELS):
    linear_a, mass_a = linear_peptide(key.sequence_a, num_channels)
    linear_b, mass_b = linear_peptide(key.sequence_b, num_channels)
    linker_mass = LINKER_MASSES[key.crosslinker]
    charges = CHANNEL_CHARGES[:num_channels]
    base = np.zeros((*BASE_SHAPE[:-1], num_channels), dtype=np.float64)
    for p, (linear, site, partner_mass) in enumerate((
        (linear_a, key.site_a, mass_b), (linear_b, key.site_b, mass_a),
    )):
        cleavages = np.arange(1, len(linear) + 1)
        has_link = np.where(B_CHANNELS[:num_channels], cleavages[:, None] >= site,
                            cleavages[:, None] < site)
        base[p, :len(linear)] = linear + has_link * (partner_mass + linker_mass) / charges
    base[..., charges > key.charge] = 0.0
    lengths = np.array([len(linear_a) + 1, len(linear_b) + 1], dtype=np.uint8)
    return base, lengths


def expand_grid(base, lengths, precursor_charge):
    num_channels = base.shape[-1]
    grid = base[:, :, None, :] + TOKEN_OFFSETS[None, None, :, :num_channels]
    candidate = ((np.arange(MAX_CLEAVAGE_LEN)[None, :, None, None] < lengths[:, None, None, None] - 1)
                 & (CHANNEL_CHARGES[:num_channels] <= precursor_charge)[None, None, None, :]
                 & np.isfinite(grid))
    return grid, candidate


def read_peaks(split):
    table = pq.read_table(split.centroid_path, columns=["mz", "intensity"]).combine_chunks()
    mz, intensity = (table[name].chunk(0) for name in ("mz", "intensity"))
    offsets = mz.offsets.to_numpy()
    require(not mz.null_count and not intensity.null_count
            and len(mz) == len(split.precursor_mz)
            and np.array_equal(offsets, intensity.offsets.to_numpy()),
            f"{split.name}: centroid peak arrays are misaligned")
    return offsets, mz.values.to_numpy(zero_copy_only=False), intensity.values.to_numpy(zero_copy_only=False)


def filter_spectrum(mz, intensity, sorted_grid):
    valid = np.isfinite(mz) & (mz > 0) & np.isfinite(intensity) & (intensity > 0)
    if not valid.all():
        mz, intensity = mz[valid], intensity[valid]
    if np.any(mz[1:] < mz[:-1]):
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
    index = np.searchsorted(sorted_grid, mz)
    left = sorted_grid[np.maximum(index - 1, 0)]
    right = sorted_grid[np.minimum(index, sorted_grid.size - 1)]
    kept = (((index > 0) & (mz <= left + PEAK_TOLERANCE))
            | ((index < sorted_grid.size) & (mz >= right - PEAK_TOLERANCE)))
    mz, intensity = mz[kept], intensity[kept]
    if np.any(mz[1:] == mz[:-1]):
        mz, intensity = np.unique(np.column_stack((mz, intensity)), axis=0).T
    return list(zip(mz, intensity))


def match_average_peaks(peaks, grid, candidate):
    mz, intensity = np.asarray(peaks, dtype=np.float64).T
    cells = grid[candidate]
    start = np.searchsorted(mz, cells - PEAK_TOLERANCE, side="left")
    stop = np.searchsorted(mz, cells + PEAK_TOLERANCE, side="right")
    width = stop - start
    present = width > 0
    maxima = np.zeros(cells.shape, dtype=np.float64)
    maxima[present] = intensity[start[present]]
    for shift in range(1, int(width.max(initial=0))):
        active = width > shift
        maxima[active] = np.maximum(maxima[active], intensity[start[active] + shift])
    target = np.zeros(grid.shape, dtype=np.float64)
    target[candidate] = maxima
    return target


def full_support(base, lengths, grid, matched):
    kept = matched.copy()
    for p, length in enumerate(lengths):
        for channel in range(base.shape[-1]):
            rows, local_tokens = np.nonzero(matched[p, :, :, channel][:, RESIDUE_TOKENS])
            if not rows.size:
                continue
            tokens = RESIDUE_TOKENS[local_tokens]
            pool = np.sort(base[p, :length - 1, channel])
            values = grid[p, rows, tokens, channel]
            index = np.searchsorted(pool, values)
            left = pool[np.maximum(index - 1, 0)]
            right = pool[np.minimum(index, pool.size - 1)]
            kept[p, rows, tokens, channel] = np.minimum(np.abs(values - left), np.abs(values - right)) <= _MZ_EQ_TOL
    return kept


def record_dtype(sequence_bytes, crosslinker_bytes, schema_version=SCHEMA_VERSION):
    require(type(schema_version) is int and schema_version in (1, 2, 3),
            f"Unsupported dataset schema version: {schema_version!r}")
    num_channels = len(V3_CHANNEL_NAMES) if schema_version == 3 else NUM_CHANNELS
    grid_shape = (*GRID_SHAPE[:-1], num_channels)
    input_shape = (*BASE_SHAPE[:-1], num_channels) if schema_version == 1 else grid_shape
    return np.dtype([
        ("group_id", "S64"), ("n_spectra", "<u4"), ("sequence", f"S{sequence_bytes}", (2,)),
        ("lengths", "u1", (2,)), ("link_sites", "u1", (2,)), ("crosslinker", f"S{crosslinker_bytes}"),
        ("linker_mass", "<f8"), ("precursor_charge", "u1"), ("precursor_mz", "<f8"),
        ("rt_seconds", "<f8"), ("input_theo_mz", "<f8", input_shape), ("target", "<f4", grid_shape),
    ], align=True)


class FeatureStats:
    def __init__(self, width):
        self.count = np.zeros(width, dtype=np.int64)
        self.total = np.zeros(width, dtype=np.float64)
        self.square_total = np.zeros(width, dtype=np.float64)

    def update(self, values, valid):
        axes = tuple(range(values.ndim - 1))
        selected = np.where(valid, values, 0.0)
        self.count += np.sum(valid, axis=axes)
        self.total += np.sum(selected, axis=axes)
        self.square_total += np.sum(np.square(selected), axis=axes)

    def result(self):
        count = np.maximum(self.count, 1)
        mean = self.total / count
        std = np.sqrt(np.maximum(self.square_total / count - mean ** 2, 0.0))
        return {"count": self.count.tolist(), "mean": mean.tolist(),
                "std": np.where(std > 1e-6, std, 1.0).tolist()}


def compact_records(path, count):
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    compact = path.with_suffix(".compact.npy")
    if count:
        destination = np.lib.format.open_memmap(compact, mode="w+", dtype=source.dtype, shape=(count,))
        for start in range(0, count, 256):
            destination[start:start + 256] = source[start:min(start + 256, count)]
        destination.flush()
        del destination
    else:
        np.save(compact, np.empty(0, dtype=source.dtype), allow_pickle=False)
    del source
    os.replace(compact, path)


def build_split(split, output_dir, dtype, limit, provenance, channel_stats, aux_stats):
    groups = list(islice(split.groups.items(), limit))
    report = {
        "input_csms": len(split.precursor_mz), "source_groups": len(split.groups),
        "selected_groups": len(groups), "written_groups": 0, "empty_groups": 0,
        "valid_positive_groups": 0, "nonempty_zero_positive_groups": 0, "empty_spectra_after_filter": 0,
        "positive_cells_before_full": 0, "positive_cells": 0, "saturated_positive_cells": 0,
        "groups_with_saturation": 0, "member_csm_rows": 0,
    }
    path = output_dir / f"{split.name}.npy"
    if not groups:
        np.save(path, np.empty(0, dtype=dtype), allow_pickle=False)
        report["saturation_fraction"] = 0.0
        return report
    offsets, mz, intensity = read_peaks(split)
    records = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(groups),))
    try:
        for i, (key, members) in enumerate(groups, start=1):
            base, lengths = build_theoretical_mz(key)
            grid, candidate = expand_grid(base, lengths, key.charge)
            sorted_grid = np.sort(grid[candidate])
            rows = [indices[0] for indices in members.values()]
            filtered = [filter_spectrum(mz[offsets[row]:offsets[row + 1]],
                                        intensity[offsets[row]:offsets[row + 1]], sorted_grid) for row in rows]
            report["empty_spectra_after_filter"] += sum(not peaks for peaks in filtered)
            peaks = filtered[0] if len(filtered) == 1 else average_peaks_by_tolerance(filtered, tol=PEAK_TOLERANCE)
            if not peaks:
                report["empty_groups"] += 1
                continue

            target = match_average_peaks(peaks, grid, candidate)
            valid = candidate & (grid >= MIN_VALID_MZ)
            report["positive_cells_before_full"] += int(np.count_nonzero((target > 0) & valid))
            target[~full_support(base, lengths, grid, target > 0)] = 0.0
            target, _ = normalize_target_intensity(target, method="sqrt_clip", clip_value=CLIP_INTENSITY)
            positive = int(np.count_nonzero((target > 0) & valid))
            saturated = int(np.count_nonzero((target == 1) & valid))
            report["positive_cells"] += positive
            report["saturated_positive_cells"] += saturated
            report["groups_with_saturation"] += bool(saturated)
            report["valid_positive_groups"] += bool(positive)
            report["nonempty_zero_positive_groups"] += not positive

            # Every nonempty average spectrum is retained, including zero-positive targets.
            mz_mean = float(np.mean(split.precursor_mz[rows]))
            rt_mean = float(np.mean(split.rt_seconds[rows]))
            if split.name == "train":
                channel_stats.update(grid, valid)
                aux = np.array([[key.charge, mz_mean, rt_mean]])
                aux_stats.update(aux, np.ones(aux.shape, dtype=bool))

            identifier = group_id(key)
            # Store raw expanded m/z, including candidates below the runtime m/z cutoff.
            input_theo_mz = np.where(candidate, grid, 0.0)
            records[report["written_groups"]] = (
                identifier.encode("ascii"), len(rows),
                (key.sequence_a.encode("utf-8"), key.sequence_b.encode("utf-8")),
                lengths, (key.site_a, key.site_b), key.crosslinker.encode("utf-8"),
                LINKER_MASSES[key.crosslinker], key.charge, mz_mean, rt_mean, input_theo_mz, target,
            )
            report["written_groups"] += 1
            for (dataset, raw_file, scan), source_rows in members.items():
                for row in source_rows:
                    provenance.append({
                        "split": split.name, "group_id": identifier, "dataset": dataset, "raw_file": raw_file,
                        "scan_number": scan, "centroid_row": row, "nms2_row": row,
                        "nms2_mirror_row": row + len(split.precursor_mz),
                    })
                    report["member_csm_rows"] += 1
            if i % 2000 == 0:
                print(f"{split.name}: processed {i:,}/{len(groups):,} groups", flush=True)
    finally:
        records.flush()
        del records

    if report["written_groups"] < len(groups):
        compact_records(path, report["written_groups"])
    report["saturation_fraction"] = report["saturated_positive_cells"] / max(report["positive_cells"], 1)
    print(f"{split.name}: {report['written_groups']:,} written, {report['empty_groups']} empty, "
          f"{report['valid_positive_groups']:,} with valid positive labels, "
          f"{report['nonempty_zero_positive_groups']} without valid positive labels", flush=True)
    return report


def verify_dataset(directory):
    metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    schema_version = metadata.get("schema_version", 1)
    dtype = record_dtype(**metadata["string_widths"], schema_version=schema_version)
    grid_shape = dtype["target"].shape
    num_channels = grid_shape[-1]
    expected_input = {
        "input_representation": "base_raw_mz" if schema_version == 1 else "expanded_raw_mz",
        "input_shape": list(dtype["input_theo_mz"].shape),
        "input_dtype": "float64",
        "main_token_id": MAIN_TOKEN_ID,
    }
    if schema_version == 3 or "channel_names" in metadata:
        expected_input["channel_names"] = list(V3_CHANNEL_NAMES[:num_channels])
    for name, expected in expected_input.items():
        default = expected if schema_version == 1 else None
        require(metadata.get(name, default) == expected,
                f"Unexpected {name} for dataset schema v{schema_version}")
    require(set(metadata["splits"]) == set(SPLITS) and metadata["record_bytes"] == dtype.itemsize
            and tuple(metadata["fields"]) == dtype.names and tuple(metadata["grid_shape"]) == grid_shape,
            "Unexpected structured dataset schema in meta.json")
    require((metadata["target_normalization"], metadata["clip_intensity"], metadata["sequence_constrained"])
            == ("sqrt_clip", CLIP_INTENSITY, "full")
            and metadata["full_mz_tolerance"] == _MZ_EQ_TOL and metadata["min_valid_mz"] == MIN_VALID_MZ
            and metadata["mz_tolerances"] == dict.fromkeys(("filter", "average", "match"), PEAK_TOLERANCE),
            "Unexpected build parameters in meta.json")
    members = pq.read_table(directory / "group_members.parquet")
    require(members.schema.equals(MEMBER_SCHEMA), "Unexpected group_members.parquet schema")
    by_group = defaultdict(set)
    member_rows = defaultdict(set)
    for item in members.to_pylist():
        split = item["split"]
        group = item["group_id"]
        row = item["centroid_row"]
        input_csms = metadata["splits"][split]["input_csms"]
        require(0 <= row < input_csms and item["nms2_row"] == row
                and item["nms2_mirror_row"] == row + input_csms and row not in member_rows[split],
                f"{split}: invalid source rows for group {group}")
        member_rows[split].add(row)
        by_group[split, group].add((item["dataset"], item["raw_file"], item["scan_number"]))

    channel_stats = FeatureStats(num_channels)
    aux_stats = FeatureStats(3)
    for split, report in metadata["splits"].items():
        rows = np.load(directory / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        require(rows.dtype == dtype and rows.shape == (report["written_groups"],), f"{split}: wrong layout")
        require(len(member_rows[split]) == report["member_csm_rows"], f"{split}: wrong provenance count")
        require(report["selected_groups"] == len(rows) + report["empty_groups"], f"{split}: wrong group counts")
        positive = saturated = valid_positive_groups = saturated_groups = 0
        for row in rows:
            key = GroupKey(row["sequence"][0].decode("utf-8"), int(row["link_sites"][0]),
                           row["sequence"][1].decode("utf-8"), int(row["link_sites"][1]),
                           row["crosslinker"].decode("utf-8"), int(row["precursor_charge"]))
            identifier = row["group_id"].decode("ascii")
            spectra = by_group.pop((split, identifier), set())
            require(identifier == group_id(key) and len(spectra) == row["n_spectra"] and spectra,
                    f"{split}: invalid group or member count for {identifier}")
            require(row["linker_mass"] == LINKER_MASSES[key.crosslinker]
                    and np.isfinite(row["precursor_mz"]) and row["precursor_mz"] > 0
                    and np.isfinite(row["rt_seconds"]),
                    f"{split}: invalid group metadata for {identifier}")
            base, lengths = build_theoretical_mz(key, num_channels)
            grid, candidate = expand_grid(base, lengths, key.charge)
            expected_mz = base if schema_version == 1 else np.where(candidate, grid, 0.0)
            require(np.array_equal(row["lengths"], lengths)
                    and np.array_equal(row["input_theo_mz"], expected_mz),
                    f"{split}: invalid theoretical m/z for {identifier}")
            target = row["target"]
            require(np.isfinite(target).all() and (target >= 0).all() and (target <= 1).all(),
                    f"{split}: invalid label values for {identifier}")
            matched = target > 0
            require(not (matched & ~candidate).any()
                    and np.array_equal(full_support(base, lengths, grid, matched), matched),
                    f"{split}: labels violate the candidate or full mask for {identifier}")
            valid = candidate & (grid >= MIN_VALID_MZ)
            if split == "train":
                channel_stats.update(grid, valid)
                aux = np.array([[key.charge, row["precursor_mz"], row["rt_seconds"]]])
                aux_stats.update(aux, np.ones(aux.shape, dtype=bool))
            n_positive = int(np.count_nonzero(matched & valid))
            n_saturated = int(np.count_nonzero((target == 1) & valid))
            positive += n_positive
            saturated += n_saturated
            valid_positive_groups += bool(n_positive)
            saturated_groups += bool(n_saturated)
        # Older metadata called the positive-label group count "supervised_groups".
        reported_positive_groups = report.get("valid_positive_groups", report.get("supervised_groups"))
        require((positive, saturated, valid_positive_groups, saturated_groups, len(rows) - valid_positive_groups) ==
                (report["positive_cells"], report["saturated_positive_cells"], reported_positive_groups,
                 report["groups_with_saturation"], report["nonempty_zero_positive_groups"]),
                f"{split}: label counts differ from meta.json")
        require(report["positive_cells_before_full"] >= positive
                and report["saturation_fraction"] == saturated / max(positive, 1),
                f"{split}: invalid label statistics")
        print(f"{split}: validated {len(rows):,} records and {positive:,} valid positive labels", flush=True)
    require(not by_group, "Provenance contains groups with no dataset row")
    for name, stats in (("mz_channels", channel_stats), ("aux_features", aux_stats)):
        for statistic, values in stats.result().items():
            np.testing.assert_allclose(metadata["train_statistics"][name][statistic], values,
                                       rtol=1e-10, atol=1e-10, err_msg=f"Invalid train {name} {statistic}")


def build_dataset(input_dir, output_dir, limit):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    splits = [read_split(input_dir, split) for split in SPLITS]
    widths = {
        "sequence_bytes": max(len(sequence.encode("utf-8")) for data in splits for key in data.groups
                              for sequence in (key.sequence_a, key.sequence_b)),
        "crosslinker_bytes": max(len(key.crosslinker.encode("utf-8")) for data in splits for key in data.groups),
    }
    dtype = record_dtype(**widths)
    channel_stats = FeatureStats(NUM_CHANNELS)
    aux_stats = FeatureStats(3)
    provenance = []
    meta = {
        "input_dir": str(input_dir), "string_widths": widths, "record_bytes": dtype.itemsize,
        "schema_version": SCHEMA_VERSION, "input_representation": "expanded_raw_mz",
        "input_shape": GRID_SHAPE, "input_dtype": "float64", "main_token_id": MAIN_TOKEN_ID,
        "channel_names": CHANNEL_NAMES,
        "fields": list(dtype.names), "grid_shape": GRID_SHAPE, "limit_groups_per_split": limit,
        "target_normalization": "sqrt_clip", "clip_intensity": CLIP_INTENSITY,
        "sequence_constrained": "full", "full_mz_tolerance": _MZ_EQ_TOL, "min_valid_mz": MIN_VALID_MZ,
        "mz_tolerances": dict.fromkeys(("filter", "average", "match"), PEAK_TOLERANCE),
        "shared_spectra_in_source_splits": {
            f"{a.name}_{b.name}": len(a.spectrum_ids & b.spectrum_ids) for a, b in combinations(splits, 2)
        },
        "splits": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temp_dir:
        staging = Path(temp_dir) / "dataset"
        staging.mkdir()
        for split in splits:
            report = build_split(split, staging, dtype, limit, provenance, channel_stats, aux_stats)
            report.update({"annotation_file": split.annotation_path.name, "centroid_file": split.centroid_path.name})
            meta["splits"][split.name] = report
        meta["train_statistics"] = {
            "mz_channels": channel_stats.result(), "aux_features": aux_stats.result(),
            "aux_order": ["precursor_charge", "precursor_mz", "rt_seconds"],
        }
        pq.write_table(pa.Table.from_pylist(provenance, schema=MEMBER_SCHEMA), staging / "group_members.parquet")
        (staging / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_dataset(staging)
        staging.rename(output_dir)
    print(f"Published dataset: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build and validate a six-channel XL dataset (schema v2)")
    build.add_argument("--input-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--limit-groups", type=int, help="limit groups per split for smoke validation")
    verify = commands.add_parser("verify", help="validate an existing XL dataset")
    verify.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        verify_dataset(args.output_dir)
    else:
        if args.limit_groups is not None and args.limit_groups <= 0:
            parser.error("--limit-groups must be positive")
        build_dataset(args.input_dir, args.output_dir, args.limit_groups)


if __name__ == "__main__":
    main()
