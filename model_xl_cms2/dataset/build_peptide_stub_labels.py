"""Build separate intact-peptide short/long intensity labels for an existing cMS2 v4 split."""

import argparse
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

import build_xl_spectrum_dataset as v4
from token_table import H2O, PROTON
from trainingExamples import normalize_target_intensity, parse_sequence_with_mods


CHARGES = np.array([1, 2, 3])
SHAPE = (2, 2, len(CHARGES))
LABEL_DTYPE = np.dtype([
    ("group_id", "S64"), ("mz", "<f8", SHAPE),
    ("target", "<f4", SHAPE), ("valid_mask", "?", SHAPE),
])


@lru_cache(maxsize=32768)
def neutral_mass(sequence):
    residues, nterm_shift = parse_sequence_with_mods(sequence)
    return sum(mass for _, mass in residues) + nterm_shift + H2O


def theoretical_grid(key):
    masses = np.array([neutral_mass(key.sequence_a), neutral_mass(key.sequence_b)])
    stub = np.array(v4.STUB_MASSES[key.crosslinker])
    mz = (masses[:, None, None] + stub[None, :, None] + CHARGES * PROTON) / CHARGES
    valid = (CHARGES <= key.charge) & (mz >= v4.MIN_VALID_MZ)
    return np.where(valid, mz, 0.0), valid


def project(rows, offsets, observed_mz, intensities, grid, valid):
    candidates = np.sort(grid[valid])
    spectra = []
    for row in rows:
        segment = slice(offsets[row], offsets[row + 1])
        spectra.append(v4.filter_spectrum(observed_mz[segment], intensities[segment], candidates))
    peaks = spectra[0] if len(spectra) == 1 else v4.average_peaks_by_tolerance(spectra, tol=v4.PEAK_TOLERANCE)
    if not peaks:
        return np.zeros(SHAPE, dtype=np.float32)
    raw = v4.match_average_peaks(peaks, grid, valid)
    target, _ = normalize_target_intensity(raw, method="sqrt_clip", clip_value=v4.CLIP_INTENSITY)
    return target


def build_split(name, dataset, raw_dir, staging, metadata):
    records = np.load(dataset / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    ids = records["group_id"].copy()
    if len(ids) != metadata["splits"][name]["written_groups"] or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}: inconsistent v4 record count or duplicate group ID")
    members = pq.read_table(dataset / "group_members.parquet", filters=[("split", "=", name)],
                            columns=["group_id", "centroid_row"]).to_pydict()
    member_rows = defaultdict(list)
    for gid, row in zip(members["group_id"], members["centroid_row"]):
        member_rows[gid].append(row)
    if set(member_rows) != {gid.decode("ascii") for gid in ids}:
        raise ValueError(f"{name}: v4 records and member table cover different groups")

    source = v4.read_split(raw_dir, name)
    offsets, observed_mz, intensities = v4.read_peaks(source)
    labels = np.empty(len(ids), dtype=LABEL_DTYPE)
    for index, record in enumerate(records):
        seq_a, seq_b = (seq.decode("ascii") for seq in record["sequence"])
        site_a, site_b = map(int, record["link_sites"])
        key = v4.GroupKey(seq_a, site_a, seq_b, site_b, record["crosslinker"].decode("ascii"),
                          int(record["precursor_charge"]))
        gid = record["group_id"].decode("ascii")
        if v4.group_id(key) != gid:
            raise ValueError(f"{name}/{gid}: v4 sequence, site, linker or charge disagrees with group ID")
        physical_spectra = source.groups[key]
        # v4 uses the first centroid row of each distinct dataset/RAW/scan within the group.
        rows = [indices[0] for indices in physical_spectra.values()]
        all_rows = sorted(row for indices in physical_spectra.values() for row in indices)
        if len(rows) != int(record["n_spectra"]) or all_rows != sorted(member_rows[gid]):
            raise ValueError(f"{name}/{gid}: v4 members disagree with the centroid source")
        grid, valid = theoretical_grid(key)
        labels[index] = record["group_id"], grid, project(rows, offsets, observed_mz, intensities, grid, valid), valid

    path = staging / f"{name}.npy"
    np.save(path, labels, allow_pickle=False)
    stored = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(stored, labels)
    np.testing.assert_array_equal(stored["group_id"], ids)
    target, valid = stored["target"], stored["valid_mask"]
    if not (np.isfinite(target).all() and ((0 <= target) & (target <= 1)).all()
            and (target[~valid] == 0).all()):
        raise ValueError(f"{name}: invalid saved intensities")
    detected = (target > 0.03) & valid
    summary = {
        "groups": len(stored), "valid_cells": int(valid.sum()),
        "groups_with_target_gt_0.03": int(detected.any(axis=(1, 2, 3)).sum()),
        "cells_target_gt_0.03": int(detected.sum()),
        "group_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    print(f"{name}: {len(stored):,} aligned groups, {detected.sum():,} cells above 0.03", flush=True)
    return summary


def build(dataset, output, raw_dir=None):
    dataset, output = dataset.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    base_meta = (dataset / "meta.json").read_bytes()
    metadata = json.loads(base_meta)
    if not (metadata["schema_version"] == 4 and metadata["spectrum_type"] == "cms2"
            and metadata["average_across_nce"] and metadata["target_normalization"] == "sqrt_clip"
            and metadata["clip_intensity"] == v4.CLIP_INTENSITY and metadata["min_valid_mz"] == v4.MIN_VALID_MZ
            and metadata["mz_tolerances"] == dict.fromkeys(("filter", "average", "match"), v4.PEAK_TOLERANCE)
            and metadata["stub_masses_da"] == {k: list(v) for k, v in v4.STUB_MASSES.items()}):
        raise ValueError("Expected the supported v4 cMS2 labeling conventions")
    raw_dir = Path(raw_dir or metadata["input_dir"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".building-peptide-stubs-") as tmp:
        staging = Path(tmp) / "labels"
        staging.mkdir()
        summary = {name: build_split(name, dataset, raw_dir, staging, metadata) for name in v4.SPLITS}
        manifest = {
            "label_type": "intact_peptide_stub", "label_schema_version": 1,
            "source_v4": str(dataset), "source_meta_sha256": hashlib.sha256(base_meta).hexdigest(),
            "centroid_source": str(raw_dir), "row_order": "identical to v4 split, keyed by group_id",
            "fields": list(LABEL_DTYPE.names), "shape": list(SHAPE),
            "axis_order": ["chain", "stub", "fragment_charge"],
            "axis_values": [["alpha (v4 sequence[0])", "beta (v4 sequence[1])"], ["short", "long"], [1, 2, 3]],
            "stub_masses_da": metadata["stub_masses_da"], "fragment_mz_min": v4.MIN_VALID_MZ,
            "valid_mask": "fragment charge <= precursor charge and theoretical m/z >= fragment_mz_min",
            "mz_formula": "(modified neutral chain mass including terminal water + stub mass + z*proton) / z",
            "proton_mass_da": PROTON, "water_mass_da": H2O, "mz_tolerances": metadata["mz_tolerances"],
            "matching": "Filter raw centroid peaks against the valid intact-stub grid. Apply the v4 cross-NCE "
                        "average over all unique member spectra, including members with zero matching peaks. "
                        "Take the maximum averaged intensity within the tolerance for each candidate. "
                        "Overlapping windows can share a peak.",
            "target_normalization": "sqrt_clip", "clip_intensity": v4.CLIP_INTENSITY, "splits": summary,
        }
        (staging / "meta.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--centroid-dir", type=Path, help="override the v4 metadata's input_dir")
    args = parser.parse_args()
    build(args.v4_dir, args.output_dir, args.centroid_dir)
