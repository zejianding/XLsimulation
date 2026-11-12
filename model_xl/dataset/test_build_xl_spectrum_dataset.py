"""Regression tests for six-channel XL storage and legacy schema validation.

Run with: python -B -m unittest discover -s model_xl/dataset -p test_build_xl_spectrum_dataset.py
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import build_xl_spectrum_dataset as builder


# Independent masses keep the expected peaks separate from the builder's grid.
A = 71.037113805
F = 147.068413945
K = 128.094963050
W = 186.079312980
WATER = 18.0105646837
PROTON = 1.007276466812
LINKER = 138.06808
ISOTOPE_DIFF = 1.0033548378
CHANNEL_NAMES = ["B1+", "B2+", "B3+", "Y1+", "Y2+", "Y3+"]
LOW_B = A + PROTON
FA_B = F + A + PROTON
LINKED_Y = K + WATER + (F + A + K + WATER) + LINKER + PROTON


def encoded(sequence):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    values = [22 if aa == "K" else alphabet.index(aa) + 1 for aa in sequence]
    return values + [0] * (30 - len(values))


def write_source(directory, charge=2, chain_b="FAK", spectrum_peaks=None):
    """Two replicate spectra plus a separate all-empty group in each split."""
    directory.mkdir()
    if spectrum_peaks is None:
        spectrum_peaks = [(LOW_B, 1000.0), (FA_B, 1000.0), (LINKED_Y, 200000.0)]
    masses = {"A": A, "F": F, "K": K, "W": W}
    precursor = ((A + K + WATER) + (sum(masses[aa] for aa in chain_b) + WATER)
                 + LINKER) / charge + PROTON
    for split, annotation_name in (("train", "train"), ("val", "val"), ("test", "holdout")):
        raw_file = f"{split}_example"
        first = [
            {"sequence_integer_1": encoded("AK"), "sequence_integer_2": encoded(chain_b),
             "alpha_position_crosslinker": 2, "beta_position_crosslinker": len(chain_b)},
            {"sequence_integer_1": encoded(chain_b), "sequence_integer_2": encoded("AK"),
             "alpha_position_crosslinker": len(chain_b), "beta_position_crosslinker": 2},
            {"sequence_integer_1": encoded("GK"), "sequence_integer_2": encoded("HK"),
             "alpha_position_crosslinker": 2, "beta_position_crosslinker": 2},
        ]
        for scan, row in enumerate(first, start=1):
            row.update(raw_file=raw_file.encode(), scan_number=scan,
                       crosslinker=b"BS3", precursor_charge_onehot=[int(z == charge) for z in range(1, 7)])
        mirrors = [dict(row,
                        sequence_integer_1=row["sequence_integer_2"],
                        sequence_integer_2=row["sequence_integer_1"],
                        alpha_position_crosslinker=row["beta_position_crosslinker"],
                        beta_position_crosslinker=row["alpha_position_crosslinker"])
                   for row in first]
        pq.write_table(pa.Table.from_pylist(first + mirrors),
                       directory / f"nms2_{annotation_name}.parquet")
        centroid = [
            {"dataset": "synthetic", "raw_file": raw_file, "scan_number": scan,
             "precursor_charge": charge, "precursor_mz": precursor,
             "retention_time_seconds": float(10 * (2 * scan - 1)),
             "mz": [peak[0] for peak in spectrum_peaks] if scan == 1 else [],
             "intensity": [peak[1] for peak in spectrum_peaks] if scan == 1 else []}
            for scan in range(1, 4)
        ]
        pq.write_table(pa.Table.from_pylist(centroid), directory / f"{split}_centroid.parquet")


class ExpandedXLStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="xl-storage-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.source = cls.work / "source"
        cls.output = cls.work / "expanded"
        write_source(cls.source)
        with contextlib.redirect_stdout(io.StringIO()):
            builder.build_dataset(cls.source, cls.output, limit=None)
        cls.meta = json.loads((cls.output / "meta.json").read_text(encoding="utf-8"))

    def copy_output(self, destination):
        for name in ("train.npy", "val.npy", "test.npy", "meta.json", "group_members.parquet"):
            shutil.copyfile(self.output / name, destination / name)

    def verify(self, directory):
        with contextlib.redirect_stdout(io.StringIO()):
            builder.verify_dataset(directory)

    def test_expanded_raw_storage_preserves_low_mass_offsets_and_padding(self):
        self.assertEqual(self.meta["schema_version"], 2)
        self.assertEqual(self.meta["input_representation"], "expanded_raw_mz")
        self.assertEqual(self.meta["input_shape"], [2, 39, 46, 6])
        self.assertEqual(self.meta["grid_shape"], [2, 39, 46, 6])
        self.assertEqual(self.meta["input_dtype"], "float64")
        self.assertEqual(self.meta["main_token_id"], 24)
        self.assertEqual(self.meta["channel_names"], CHANNEL_NAMES)
        counts = self.meta["train_statistics"]["mz_channels"]["count"]
        self.assertEqual(len(counts), 6)
        self.assertEqual([counts[2], counts[5]], [0, 0])
        rows = np.load(self.output / "train.npy", allow_pickle=False)
        self.assertEqual(rows.shape, (1,))
        row = rows[0]
        grid = row["input_theo_mz"]
        self.assertEqual(grid.shape, (2, 39, 46, 6))
        self.assertEqual(grid.dtype, np.dtype("float64"))
        self.assertEqual(rows.dtype.itemsize, self.meta["record_bytes"])
        self.assertEqual(row["sequence"].tolist(), [b"AK", b"FAK"])
        np.testing.assert_array_equal(row["lengths"], [2, 3])
        np.testing.assert_array_equal(row["link_sites"], [2, 3])
        self.assertAlmostEqual(grid[0, 0, 24, 0], LOW_B, places=10)
        self.assertAlmostEqual(grid[0, 0, 0, 0], LOW_B - round(W, 6), places=10)
        self.assertLess(grid[0, 0, 0, 0], 0)
        self.assertLess(grid[0, 0, 24, 0], 100)
        # Token 28 is +A: a low base does not invalidate all of its offsets.
        self.assertAlmostEqual(grid[0, 0, 28, 0], LOW_B + round(A, 6), places=10)
        self.assertGreater(grid[0, 0, 28, 0], 100)
        self.assertAlmostEqual(grid[1, 1, 24, 0], FA_B, places=10)
        self.assertAlmostEqual(grid[1, 1, 24, 1], (F + A) / 2 + PROTON, places=10)
        self.assertAlmostEqual(grid[0, 0, 24, 3], LINKED_Y, places=10)
        self.assertFalse(np.any(grid[0, 1:]))
        self.assertFalse(np.any(grid[1, 2:]))
        self.assertFalse(np.any(grid[..., [2, 5]]))
        self.verify(self.output)

    def test_average_labels_full_support_and_group_accounting_are_unchanged(self):
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                row = np.load(self.output / f"{split}.npy", allow_pickle=False)[0]
                target = row["target"]
                self.assertEqual(target.shape, (2, 39, 46, 6))
                self.assertEqual(target.dtype, np.dtype("float32"))
                # The empty replicate still divides 1000 by two: sqrt(500/50000).
                self.assertAlmostEqual(float(target[1, 1, 24, 0]), 0.1, places=7)
                # The same peak has a sequence-supported +A representation at b1.
                self.assertAlmostEqual(float(target[1, 0, 28, 0]), 0.1, places=7)
                self.assertEqual(float(target[0, 0, 24, 3]), 1.0)
                # Low-m/z labels are retained on disk; the Dataset masks them later.
                self.assertAlmostEqual(float(target[0, 0, 24, 0]), 0.1, places=7)
                self.assertFalse(np.any(target[0, 1:]))
                self.assertFalse(np.any(target[1, 2:]))
                self.assertFalse(np.any(target[..., [2, 5]]))
                self.assertEqual(int(row["n_spectra"]), 2)
                self.assertAlmostEqual(float(row["rt_seconds"]), 20.0)
                report = self.meta["splits"][split]
                self.assertEqual(report["input_csms"], 3)
                self.assertEqual(report["source_groups"], 2)
                self.assertEqual(report["written_groups"], 1)
                self.assertEqual(report["empty_groups"], 1)
                self.assertEqual(report["empty_spectra_after_filter"], 2)
                self.assertEqual(report["member_csm_rows"], 2)
        members = pq.read_table(self.output / "group_members.parquet").to_pylist()
        self.assertEqual(len(members), 6)
        self.assertEqual({r["centroid_row"] for r in members}, {0, 1})
        self.assertTrue(all(r["nms2_mirror_row"] == r["centroid_row"] + 3 for r in members))

    def test_high_charge_precursors_use_six_fragment_channels(self):
        b_neutral = F + W + A
        y_neutral = K + WATER + (A + K + WATER) + LINKER
        peaks = [(b_neutral + PROTON, 1000.0)]
        for charge in (4, 5, 6):
            with self.subTest(charge=charge), tempfile.TemporaryDirectory(dir=self.work) as tmp:
                directory = Path(tmp)
                source, output = directory / "source", directory / "output"
                write_source(source, charge=charge, chain_b="FWAK", spectrum_peaks=peaks)
                with contextlib.redirect_stdout(io.StringIO()):
                    builder.build_dataset(source, output, limit=None)
                row = np.load(output / "train.npy", allow_pickle=False)[0]
                grid, target = row["input_theo_mz"], row["target"]
                self.assertEqual(int(row["precursor_charge"]), charge)
                self.assertEqual(grid.shape, (2, 39, 46, 6))
                self.assertEqual(target.shape, grid.shape)
                for q in (1, 2, 3):
                    self.assertAlmostEqual(grid[1, 2, 24, q - 1], b_neutral / q + PROTON, places=10)
                    self.assertAlmostEqual(grid[1, 2, 24, q + 2], y_neutral / q + PROTON, places=10)
                self.assertAlmostEqual(float(target[1, 2, 24, 0]), 0.1, places=7)
                self.assertFalse(np.any(grid[0, 1:]))
                self.assertFalse(np.any(grid[1, 3:]))
                self.assertFalse(np.any(target[0, 1:]))
                self.assertFalse(np.any(target[1, 3:]))
                self.verify(output)

    def test_fourth_charge_only_peaks_are_excluded_before_averaging(self):
        # These two 4+ peaks have no six-channel candidate within the matching window.
        b4_isotope = (F + W + A + ISOTOPE_DIFF) / 4 + PROTON
        y4 = (K + WATER + (A + K + WATER) + LINKER) / 4 + PROTON
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            directory = Path(tmp)
            source, output = directory / "source", directory / "output"
            write_source(source, charge=6, chain_b="FWAK", spectrum_peaks=[(b4_isotope, 9000.0), (y4, 4000.0)])
            with contextlib.redirect_stdout(io.StringIO()):
                builder.build_dataset(source, output, limit=None)
            meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            for split, report in meta["splits"].items():
                self.assertEqual(report["empty_spectra_after_filter"], 3)
                self.assertEqual(report["empty_groups"], 2)
                self.assertEqual(report["written_groups"], 0)
                self.assertEqual(np.load(output / f"{split}.npy", allow_pickle=False).size, 0)

    def make_schema_fixture(self, directory, schema_version):
        """Create old layouts independently of the builder's version switch."""
        self.copy_output(directory)
        grid_shape = (2, 39, 46, 8 if schema_version == 3 else 6)
        input_shape = (2, 39, 6) if schema_version == 1 else grid_shape
        for split in ("train", "val", "test"):
            path = directory / f"{split}.npy"
            source = np.load(path, allow_pickle=False)
            legacy_dtype = np.dtype([
                (name, "<f8", input_shape) if name == "input_theo_mz"
                else (name, "<f4", grid_shape) if name == "target"
                else (name, source.dtype.fields[name][0]) for name in source.dtype.names
            ], align=True)
            legacy = np.zeros(source.shape, dtype=legacy_dtype)
            for name in source.dtype.names:
                if schema_version == 1 and name == "input_theo_mz":
                    legacy[name] = source[name][:, :, :, 24, :]
                elif schema_version == 3 and name in ("input_theo_mz", "target"):
                    # The source is Z2, so the two extra channels are invalid.
                    legacy[name][..., :6] = source[name]
                else:
                    legacy[name] = source[name]
            np.save(path, legacy, allow_pickle=False)
        meta = json.loads(json.dumps(self.meta))
        meta.update(record_bytes=legacy_dtype.itemsize, grid_shape=list(grid_shape))
        if schema_version == 1:
            for name in ("schema_version", "input_representation", "input_shape", "input_dtype", "main_token_id",
                         "channel_names"):
                del meta[name]
        else:
            meta.update(schema_version=3, input_shape=list(input_shape), channel_names=[*CHANNEL_NAMES, "B4+", "Y4+"])
            stats = meta["train_statistics"]["mz_channels"]
            for name, fill in (("count", 0), ("mean", 0.0), ("std", 1.0)):
                stats[name].extend([fill, fill])
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    def test_unversioned_and_explicit_v1_remain_verifiable(self):
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            directory = Path(tmp)
            meta = self.make_schema_fixture(directory, schema_version=1)
            self.verify(directory)
            meta["schema_version"] = 1
            (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self.verify(directory)
            meta.update(input_representation="base_raw_mz", input_shape=[2, 39, 6],
                        input_dtype="float64", main_token_id=24)
            (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self.verify(directory)

    def test_v2_without_channel_names_remains_verifiable(self):
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            directory = Path(tmp)
            self.copy_output(directory)
            meta = {name: value for name, value in self.meta.items() if name != "channel_names"}
            (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self.verify(directory)

    def test_eight_channel_v3_remains_verifiable(self):
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            directory = Path(tmp)
            meta = self.make_schema_fixture(directory, schema_version=3)
            self.verify(directory)
            meta["channel_names"][-2:] = ["Y4+", "B4+"]
            (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.verify(directory)

    def test_version_and_metadata_conflicts_are_rejected(self):
        changes = [
            {"schema_version": 4},
            {"schema_version": 1},
            {"schema_version": 3},
            {"input_shape": [2, 39, 6]},
            {"input_shape": [2, 39, 46, 8]},
            {"grid_shape": [2, 39, 46, 8]},
            {"input_representation": "base_raw_mz"},
            {"input_dtype": "float32"},
            {"main_token_id": 0},
            {"channel_names": ["B1+", "B2+", "B3+", "Y1+", "Y3+", "Y2+"]},
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=self.work) as tmp:
                directory = Path(tmp)
                self.copy_output(directory)
                (directory / "meta.json").write_text(json.dumps(dict(self.meta, **change)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.verify(directory)
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            directory = Path(tmp)
            self.make_schema_fixture(directory, schema_version=1)
            # Valid v2 metadata must not allow a physical v1 record layout.
            (directory / "meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.verify(directory)

    def test_verifier_rejects_truncated_or_corrupted_expanded_grid(self):
        corruptions = [
            ((0, 0, 24, 0), 0.0),       # Incorrectly dropped the low-m/z base.
            ((0, 0, 0, 0), 0.0),        # Incorrectly dropped the negative raw offset.
            ((0, 1, 28, 0), 71.037114),  # Expanded a padded base instead of storing zero.
            ((1, 1, 24, 0), FA_B + A),  # A second offset was accidentally applied.
            ((0, 0, 24, 2), 123.0),     # A 2+ precursor cannot have a 3+ channel.
        ]
        for index, value in corruptions:
            with self.subTest(index=index), tempfile.TemporaryDirectory(dir=self.work) as tmp:
                directory = Path(tmp)
                self.copy_output(directory)
                rows = np.load(directory / "train.npy", allow_pickle=False)
                rows["input_theo_mz"][(0,) + index] = value
                np.save(directory / "train.npy", rows, allow_pickle=False)
                with self.assertRaises(ValueError):
                    self.verify(directory)


if __name__ == "__main__":
    unittest.main()
