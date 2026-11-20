"""Build and verify cross-NCE cleavable-MS2 average spectra from centroid Parquet files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import zlib
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
    MAX_CLEAVAGE_LEN,
    MAX_PEPTIDE_LEN,
    NUM_TOKENS,
    generate_by_ions_with_mods,
    normalize_target_intensity,
)
from sequence_token_map import _MZ_EQ_TOL
from token_table import MAIN_TOKEN_IDS, RESIDUE_EVENT_TOKEN_IDS, build_charge_tokens


SPLITS = {"train": "train", "val": "val", "test": "holdout"}
# The six-channel rescoring helper also imports BS3 and the per-arm offset table.
LINKER_MASSES = {"BS3": 138.06808, "DSSO": 158.003765, "DSBU": 196.084792}
STUB_MASSES = {"DSSO": (54.010565, 85.982635), "DSBU": (85.052764, 111.032028)}
LINK_SITE_CODES = {"DSSO": 22, "DSBU": 23}
PEAK_TOLERANCE = 0.05
MIN_VALID_MZ = 100.0
CLIP_INTENSITY = 50000.0
ION_NAMES = ("B1+", "B2+", "B3+", "Y1+", "Y2+", "Y3+")
NUM_CHANNELS = len(ION_NAMES)  # b/y channels per arm
CHANNEL_NAMES = tuple(f"{state}_{ion}" for state in ("plain_or_short", "long") for ion in ION_NAMES)
CHANNEL_INDEX = {name: index for index, name in enumerate(ION_NAMES)}
ION_CHARGES = np.array([1, 2, 3, 1, 2, 3])
CHANNEL_CHARGES = np.tile(ION_CHARGES, 2)
CHARGE_TOKENS = build_charge_tokens(max_charge=3)
TOKEN_OFFSETS = np.column_stack([CHARGE_TOKENS[z] for z in ION_CHARGES])
CMS2_OFFSETS = np.tile(TOKEN_OFFSETS, (1, 2))
RESIDUE_TOKENS = np.array(RESIDUE_EVENT_TOKEN_IDS)
BASE_SHAPE = (2, MAX_CLEAVAGE_LEN, len(CHANNEL_NAMES))
GRID_SHAPE = (2, MAX_CLEAVAGE_LEN, NUM_TOKENS, len(CHANNEL_NAMES))
SCHEMA_VERSION = 4
MAIN_TOKEN_ID = MAIN_TOKEN_IDS[0]

RESIDUE_CODES = dict(enumerate("ACDEFGHIKLMNPQRSTVWY", start=1))
RESIDUE_CODES.update({21: "M", 22: "K", 23: "K", 24: "C"})
RESIDUE_MODS = {2: 57.021464, 21: 15.994915}
MEMBER_SCHEMA = pa.schema([
    ("split", pa.string()),
    ("group_id", pa.string()),
    ("dataset", pa.string()),
    ("raw_file", pa.string()),
    ("scan_number", pa.int64()),
    ("centroid_row", pa.int64()),
    ("cms2_source_rows", pa.list_(pa.int64())),
    ("source_nce", pa.float64()),
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
    source_rows: list[list[int]]
    source_nce: np.ndarray


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=32768)
def decode_chain(codes, site, linker):
    length = next((i for i, code in enumerate(codes) if code == 0), len(codes))
    require(2 <= length <= MAX_PEPTIDE_LEN and not any(codes[length:]),
            f"Invalid peptide length or padding: {codes}")
    codes = codes[:length]
    require(1 <= site <= length and codes[site - 1] == LINK_SITE_CODES[linker]
            and codes.count(22) + codes.count(23) == 1,
            f"Expected a single linked K at site {site}: {codes}")
    require(all(code in RESIDUE_CODES for code in codes), f"Unknown cMS2 residue code: {codes}")
    residues = [RESIDUE_CODES[code] for code in codes]
    modifications = tuple((i, RESIDUE_MODS[code]) for i, code in enumerate(codes, start=1)
                          if code in RESIDUE_MODS)
    tokens = residues.copy()
    for i, mass in modifications:
        tokens[i - 1] += f"[{mass:+.6f}]"
    return Chain("".join(residues), modifications, int(site), "".join(tokens))


def annotation_matrix(table, name):
    column = table[name].combine_chunks()
    widths = np.diff(column.offsets.to_numpy())
    require(len(widths) > 0 and widths[0] > 0 and np.all(widths == widths[0])
            and not column.null_count and not column.values.null_count, f"Invalid array in {name}")
    return column.values.to_numpy(zero_copy_only=False).reshape(len(column), int(widths[0]))


def read_split(input_dir, split):
    annotation_path = input_dir / f"cms2_{SPLITS[split]}.parquet"
    centroid_path = input_dir / f"cms2_{split}_centroid.parquet"
    seq_fields = ("sequence_integer_1", "sequence_integer_2")
    site_fields = ("alpha_position_crosslinker", "beta_position_crosslinker")
    ann = pq.read_table(annotation_path, columns=[
        *seq_fields, *site_fields, "raw_file", "scan_number", "crosslinker",
        "precursor_charge_onehot", "collision_energy_aligned_normed",
    ])
    cent = pq.read_table(centroid_path, columns=[
        "dataset", "raw_file", "scan_number", "precursor_charge", "precursor_mz", "retention_time_seconds",
    ])
    meta = cent.schema.metadata or {}
    require(meta.get(b"source_annotation_file") == annotation_path.name.encode()
            and meta.get(b"source_split") == SPLITS[split].encode(), f"{split}: source metadata mismatch")
    digest = hashlib.sha256()
    with annotation_path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    require(digest.hexdigest().encode() == meta.get(b"source_annotation_sha256"),
            f"{split}: annotation file differs from the centroid source")
    mapping = np.frombuffer(zlib.decompress(base64.b64decode(meta[b"source_row_to_output_row_b64"],
                                                            validate=True)), dtype="<u4")
    require(len(mapping) == len(ann) and np.array_equal(np.unique(mapping), np.arange(len(cent))),
            f"{split}: invalid or incomplete annotation-to-centroid mapping")
    ids = list(zip(cent["dataset"].to_pylist(), cent["raw_file"].to_pylist(), cent["scan_number"].to_pylist()))
    charges = cent["precursor_charge"].to_numpy()
    onehot = annotation_matrix(ann, "precursor_charge_onehot")
    count = onehot.sum(axis=1)
    require(onehot.shape[1] == 6 and np.isin(onehot, (0, 1)).all() and (count <= 1).all()
            and np.array_equal(np.where(count == 0, 7, onehot.argmax(axis=1) + 1), charges[mapping]),
            f"{split}: invalid precursor charge or 7+ fallback")
    scans = annotation_matrix(ann, "scan_number").ravel()
    require(np.array_equal(scans, cent["scan_number"].to_numpy()[mapping]), f"{split}: scan number mismatch")
    public_names = [(name + ".pkl").encode()[:50].decode().casefold() for _, name, _ in ids]
    require(all(n.casefold() == public_names[i] for n, i in zip(ann["raw_file"].to_pylist(), mapping)),
            f"{split}: RAW filename mismatch")
    mz = cent["precursor_mz"].to_numpy()
    rt = cent["retention_time_seconds"].to_numpy()
    energy = ann["collision_energy_aligned_normed"].to_numpy()
    require(np.all(np.isfinite(mz) & (mz > 0)) and np.all(np.isfinite(rt) & (rt >= 0))
            and np.isfinite(energy).all(), f"{split}: invalid scan metadata")

    code_a, code_b = (annotation_matrix(ann, k) for k in seq_fields)
    for codes in (code_a, code_b):
        require(np.isin(codes, [0, *RESIDUE_CODES]).all(), f"{split}: unknown residue code")
    code_a, code_b = code_a.astype(np.uint8), code_b.astype(np.uint8)
    site_a, site_b = (annotation_matrix(ann, k).ravel() for k in site_fields)
    linkers = annotation_matrix(ann, "crosslinker").ravel()
    groups = defaultdict(dict)
    source_rows = [[] for _ in ids]
    centroid_keys = [None] * len(ids)
    centroid_nce = np.empty(len(ids), dtype=np.float64)
    for source, row in enumerate(mapping):
        linker = linkers[source]
        require(linker in STUB_MASSES, f"{split}: unsupported linker {linker}")
        a, b = sorted((decode_chain(code_a[source].tobytes(), int(site_a[source]), linker),
                       decode_chain(code_b[source].tobytes(), int(site_b[source]), linker)))
        key = GroupKey(a.sequence, a.site, b.sequence, b.site, linker, int(charges[row]))
        if not source_rows[row]:
            groups[key].setdefault(ids[row], []).append(int(row))
            centroid_keys[row] = key
            centroid_nce[row] = energy[source]
        else:
            require(key == centroid_keys[row] and centroid_nce[row] == energy[source],
                    f"{split}: inconsistent annotations mapped to centroid row {row}")
        source_rows[row].append(source)
    return SplitData(split, annotation_path, centroid_path, mz, rt, set(ids), dict(groups),
                     source_rows, centroid_nce)


def group_id(key):
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=16384)
def linear_peptide(sequence):
    ions = list(generate_by_ions_with_mods(sequence, max_charge=3))
    length = max(ion["cleavage_index"] for ion in ions)
    base = np.zeros((length, NUM_CHANNELS), dtype=np.float64)
    for ion in ions:
        channel = CHANNEL_INDEX[f"{ion['ion_type']}{ion['charge']}+"]
        base[ion["cleavage_index"] - 1, channel] = ion["mz"]
    return base


def build_theoretical_mz(key):
    short_mass, long_mass = STUB_MASSES[key.crosslinker]
    base = np.zeros(BASE_SHAPE, dtype=np.float64)
    # -1 = impossible, 0 = plain, 1 = short, 2 = long.
    states = np.full(BASE_SHAPE, -1, dtype=np.int8)
    lengths = []
    for p, (seq, site) in enumerate(((key.sequence_a, key.site_a), (key.sequence_b, key.site_b))):
        linear = linear_peptide(seq)
        n = len(linear)
        lengths.append(n + 1)
        cuts = np.arange(1, n + 1)[:, None]
        contains = np.where(np.arange(NUM_CHANNELS) < 3, cuts >= site, cuts < site)
        states[p, :n, :NUM_CHANNELS] = contains
        states[p, :n, NUM_CHANNELS:] = np.where(contains, 2, -1)
        base[p, :n, :NUM_CHANNELS] = linear + contains * (short_mass / ION_CHARGES)
        base[p, :n, NUM_CHANNELS:] = np.where(contains, linear + long_mass / ION_CHARGES, 0)
    states[..., CHANNEL_CHARGES > key.charge] = -1
    base[states < 0] = 0
    return base, states, np.array(lengths, dtype=np.uint8)


def expand_grid(base, states):
    grid = base[:, :, None, :] + CMS2_OFFSETS
    valid = (states[:, :, None, :] >= 0) & np.isfinite(grid) & (grid >= MIN_VALID_MZ)
    return np.where(valid, grid, 0.0), valid


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
    if not sorted_grid.size:
        return []
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


def full_support(base, states, grid, matched):
    kept = matched.copy()
    for p in range(2):
        for channel in range(base.shape[-1]):
            rows, columns = np.nonzero(matched[p, :, :, channel][:, RESIDUE_TOKENS])
            tokens = RESIDUE_TOKENS[columns]
            ion_states = states[p, :, channel]
            for state in np.unique(ion_states[rows]):
                select = ion_states[rows] == state
                r, t = rows[select], tokens[select]
                pool = np.sort(base[p, ion_states == state, channel])
                values = grid[p, r, t, channel]
                idx = pool.searchsorted(values)
                left = pool[np.maximum(idx - 1, 0)]
                right = pool[np.minimum(idx, len(pool) - 1)]
                kept[p, r, t, channel] = np.minimum(abs(values - left), abs(values - right)) <= _MZ_EQ_TOL
    return kept


def record_dtype(sequence_bytes, crosslinker_bytes):
    return np.dtype([
        ("group_id", "S64"), ("n_spectra", "<u4"), ("sequence", f"S{sequence_bytes}", (2,)),
        ("lengths", "u1", (2,)), ("link_sites", "u1", (2,)), ("crosslinker", f"S{crosslinker_bytes}"),
        ("linker_mass", "<f8"), ("precursor_charge", "u1"), ("precursor_mz", "<f8"),
        ("rt_seconds", "<f8"), ("input_theo_mz", "<f8", GRID_SHAPE), ("target", "<f4", GRID_SHAPE),
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


def project_group(key, rows, offsets, mz, intensity):
    base, states, lengths = build_theoretical_mz(key)
    grid, valid = expand_grid(base, states)
    sorted_grid = np.sort(grid[valid])
    filtered = [filter_spectrum(mz[offsets[r]:offsets[r + 1]], intensity[offsets[r]:offsets[r + 1]],
                                sorted_grid) for r in rows]
    missing = sum(not peaks for peaks in filtered)
    peaks = filtered[0] if len(filtered) == 1 else average_peaks_by_tolerance(filtered, tol=PEAK_TOLERANCE)
    if not peaks:
        return None, missing
    target = match_average_peaks(peaks, grid, valid)
    before_full = np.count_nonzero(target)
    target[~full_support(base, states, grid, target > 0)] = 0.0
    target, _ = normalize_target_intensity(target, method="sqrt_clip", clip_value=CLIP_INTENSITY)
    return (lengths, grid, valid, target, before_full), missing


def build_split(split, output_dir, dtype, limit, provenance, channel_stats, aux_stats):
    groups = list(islice(split.groups.items(), limit))
    report = {
        "input_csms": len(split.precursor_mz), "source_groups": len(split.groups),
        "selected_groups": len(groups), "written_groups": 0, "empty_groups": 0,
        "valid_positive_groups": 0, "nonempty_zero_positive_groups": 0, "empty_spectra_after_filter": 0,
        "positive_cells_before_full": 0, "positive_cells": 0, "saturated_positive_cells": 0,
        "groups_with_saturation": 0, "member_csm_rows": 0, "mixed_nce_groups": 0,
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
            rows = [indices[0] for indices in members.values()]
            projected, missing = project_group(key, rows, offsets, mz, intensity)
            report["empty_spectra_after_filter"] += missing
            if projected is None:
                report["empty_groups"] += 1
                continue
            lengths, grid, valid, target, before_full = projected
            report["positive_cells_before_full"] += before_full
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
            report["mixed_nce_groups"] += int(np.unique(split.source_nce[rows]).size > 1)
            records[report["written_groups"]] = (
                identifier.encode("ascii"), len(rows),
                (key.sequence_a.encode("utf-8"), key.sequence_b.encode("utf-8")),
                lengths, (key.site_a, key.site_b), key.crosslinker.encode("utf-8"),
                LINKER_MASSES[key.crosslinker], key.charge, mz_mean, rt_mean, grid, target,
            )
            report["written_groups"] += 1
            for (dataset, raw_file, scan), source_rows in members.items():
                for row in source_rows:
                    provenance.append({
                        "split": split.name, "group_id": identifier, "dataset": dataset, "raw_file": raw_file,
                        "scan_number": scan, "centroid_row": row, "cms2_source_rows": split.source_rows[row],
                        "source_nce": float(split.source_nce[row]),
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


def verify_dataset(directory, sources=None):
    """Compare all records, memberships, statistics and labels with the input Parquets."""
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    dtype = record_dtype(**meta["string_widths"])
    expected = {
        "schema_version": SCHEMA_VERSION, "spectrum_type": "cms2",
        "input_representation": "expanded_raw_mz", "input_dtype": "float64",
        "grid_shape": list(GRID_SHAPE), "input_shape": list(GRID_SHAPE), "fields": list(dtype.names),
        "record_bytes": dtype.itemsize, "channel_names": list(CHANNEL_NAMES),
        "stub_masses_da": {k: list(v) for k, v in STUB_MASSES.items()},
        "group_by": list(GroupKey._fields), "average_across_nce": True,
        "main_token_id": MAIN_TOKEN_ID, "target_normalization": "sqrt_clip",
        "clip_intensity": CLIP_INTENSITY, "min_valid_mz": MIN_VALID_MZ,
        "sequence_constrained": "full_same_state", "full_mz_tolerance": _MZ_EQ_TOL,
        "mz_tolerances": dict.fromkeys(("filter", "average", "match"), PEAK_TOLERANCE),
    }
    for name, value in expected.items():
        require(meta.get(name) == value, f"Invalid cMS2 metadata: {name}")
    require(set(meta["splits"]) == set(SPLITS), "Invalid split names")
    require(meta["train_statistics"]["aux_order"] == ["precursor_charge", "precursor_mz", "rt_seconds"],
            "Invalid auxiliary features")
    if sources is None:
        sources = [read_split(Path(meta["input_dir"]), name) for name in SPLITS]
    overlap = {f"{a.name}_{b.name}": len(a.spectrum_ids & b.spectrum_ids) for a, b in combinations(sources, 2)}
    require(overlap == meta["shared_spectra_in_source_splits"], "Incorrect split overlap counts")
    members = pq.read_table(directory / "group_members.parquet")
    require(members.schema.equals(MEMBER_SCHEMA), "Invalid membership schema")
    by_centroid = {}
    for item in members.to_pylist():
        index = (item["split"], item["centroid_row"])
        require(index not in by_centroid, f"Duplicate centroid membership {index}")
        by_centroid[index] = item

    mz_stats, aux_stats = FeatureStats(len(CHANNEL_NAMES)), FeatureStats(3)
    for source in sources:
        split = source.name
        report = meta["splits"][split]
        records = np.load(directory / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        groups = list(islice(source.groups.items(), meta["limit_groups_per_split"]))
        require(records.dtype == dtype and records.shape == (report["written_groups"],)
                and report["input_csms"] == len(source.precursor_mz)
                and report["source_groups"] == len(source.groups) and report["selected_groups"] == len(groups),
                f"{split}: invalid layout or input counts")
        require(report["annotation_file"] == source.annotation_path.name
                and report["centroid_file"] == source.centroid_path.name, f"{split}: invalid sources")
        offsets, mz, intensity = read_peaks(source)
        written = member_count = empty_scans = positive = saturated = before_full = 0
        with_positive = with_saturation = mixed = 0
        for key, group in groups:
            rows = [indices[0] for indices in group.values()]
            projected, missing = project_group(key, rows, offsets, mz, intensity)
            empty_scans += missing
            if projected is None:
                continue
            lengths, grid, valid, target, n_before = projected
            require(written < len(records), f"{split}: missing record")
            row = records[written]
            identifier = group_id(key)
            mean_mz, mean_rt = np.mean(source.precursor_mz[rows]), np.mean(source.rt_seconds[rows])
            require(row["group_id"].decode() == identifier and row["n_spectra"] == len(rows)
                    and row["sequence"].tolist() == [key.sequence_a.encode(), key.sequence_b.encode()]
                    and row["link_sites"].tolist() == [key.site_a, key.site_b]
                    and row["crosslinker"].decode() == key.crosslinker and row["precursor_charge"] == key.charge
                    and row["linker_mass"] == LINKER_MASSES[key.crosslinker]
                    and row["precursor_mz"] == mean_mz and row["rt_seconds"] == mean_rt
                    and np.array_equal(row["lengths"], lengths), f"{split}/{identifier}: group mismatch")
            require(np.array_equal(row["input_theo_mz"], grid) and np.array_equal(row["target"], target),
                    f"{split}/{identifier}: grid or target differs from source reconstruction")
            require(np.isfinite(target).all() and np.all((target >= 0) & (target <= 1))
                    and not target[~valid].any(), f"{split}/{identifier}: invalid intensities")
            for sid, indices in group.items():
                for idx in indices:
                    item = by_centroid.pop((split, idx), None)
                    require(item is not None and item["group_id"] == identifier
                            and (item["dataset"], item["raw_file"], item["scan_number"]) == sid
                            and item["cms2_source_rows"] == source.source_rows[idx]
                            and item["source_nce"] == source.source_nce[idx], f"{split}/{idx}: provenance mismatch")
                    member_count += 1
            if split == "train":
                mz_stats.update(grid, valid)
                aux = np.array([[key.charge, mean_mz, mean_rt]])
                aux_stats.update(aux, np.ones(aux.shape, dtype=bool))
            pos, sat = np.count_nonzero(target), np.count_nonzero(target == 1)
            written += 1
            positive += pos
            saturated += sat
            before_full += n_before
            with_positive += int(pos > 0)
            with_saturation += int(sat > 0)
            mixed += int(np.unique(source.source_nce[rows]).size > 1)
        counts = {"written_groups": written, "empty_groups": len(groups) - written,
                  "empty_spectra_after_filter": empty_scans, "member_csm_rows": member_count,
                  "positive_cells_before_full": before_full, "positive_cells": positive,
                  "saturated_positive_cells": saturated, "valid_positive_groups": with_positive,
                  "nonempty_zero_positive_groups": written - with_positive, "groups_with_saturation": with_saturation,
                  "saturation_fraction": saturated / max(positive, 1), "mixed_nce_groups": mixed}
        require(all(report[k] == v for k, v in counts.items()), f"{split}: incorrect build counters")
        print(f"{split}: verified {written:,} records against centroid peaks, {mixed:,} mixed-NCE groups", flush=True)
    require(not by_centroid, "Unreferenced provenance rows")
    for name, stats in (("mz_channels", mz_stats), ("aux_features", aux_stats)):
        for field, value in stats.result().items():
            np.testing.assert_allclose(meta["train_statistics"][name][field], value, rtol=1e-10, atol=1e-10)


def build_dataset(input_dir, output_dir, limit):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    splits = [read_split(input_dir, split) for split in SPLITS]
    widths = {
        "sequence_bytes": max(len(sequence.encode("utf-8")) for data in splits for key in data.groups
                              for sequence in (key.sequence_a, key.sequence_b)),
        "crosslinker_bytes": max(len(key.crosslinker.encode("utf-8")) for data in splits for key in data.groups),
    }
    dtype = record_dtype(**widths)
    channel_stats = FeatureStats(len(CHANNEL_NAMES))
    aux_stats = FeatureStats(3)
    provenance = []
    meta = {
        "input_dir": str(input_dir), "string_widths": widths, "record_bytes": dtype.itemsize,
        "schema_version": SCHEMA_VERSION, "spectrum_type": "cms2", "stub_masses_da": STUB_MASSES,
        "group_by": list(GroupKey._fields), "average_across_nce": True, "input_representation": "expanded_raw_mz",
        "input_shape": GRID_SHAPE, "input_dtype": "float64", "main_token_id": MAIN_TOKEN_ID,
        "channel_names": CHANNEL_NAMES,
        "fields": list(dtype.names), "grid_shape": GRID_SHAPE, "limit_groups_per_split": limit,
        "target_normalization": "sqrt_clip", "clip_intensity": CLIP_INTENSITY,
        "sequence_constrained": "full_same_state", "full_mz_tolerance": _MZ_EQ_TOL, "min_valid_mz": MIN_VALID_MZ,
        "mz_tolerances": dict.fromkeys(("filter", "average", "match"), PEAK_TOLERANCE),
        "shared_spectra_in_source_splits": {
            f"{a.name}_{b.name}": len(a.spectrum_ids & b.spectrum_ids) for a, b in combinations(splits, 2)
        },
        "splits": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".cms2-building"
    staging.mkdir()  # Exclusive build lock on the destination filesystem.
    published = []
    try:
        require(all(p == staging for p in output_dir.iterdir()), "Output directory is no longer empty")
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
        verify_dataset(staging, splits)
        # Publish the metadata last. Hard links fail if a destination already exists.
        for name in ("train.npy", "val.npy", "test.npy", "group_members.parquet", "meta.json"):
            destination = output_dir / name
            os.link(staging / name, destination)
            published.append(destination)
    except BaseException:
        for path in published:
            if path.exists() and os.path.samefile(path, staging / path.name):
                path.unlink()
        raise
    finally:
        shutil.rmtree(staging)
    print(f"Published dataset: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build and validate a 12-channel cMS2 dataset (schema v4)")
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
