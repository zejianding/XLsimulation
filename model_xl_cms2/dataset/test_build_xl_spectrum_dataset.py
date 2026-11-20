"""Regression tests for the cMS2 builder. All fixtures are local to this directory."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import build_xl_spectrum_dataset as builder


PROTON = 1.007276466812
WATER = 18.0105646837
F = 147.068413945
K = 128.094963050
STUBS = {"DSSO": (54.010565, 85.982635, 22), "DSBU": (85.052764, 111.032028, 23)}


def write_fixture(source):
    for split, public in builder.SPLITS.items():
        annotation = source / f"cms2_{public}.parquet"
        anns, cents, indices = [], [], []
        for linker, (short, long, code) in STUBS.items():
            # Two spectra at different NCEs, plus an out-of-order mirror of the first.
            for scan, swap, nce in [(10, True, 0.25), (20, False, 0.34), (10, False, 0.25)]:
                raw = f"sample_{linker}"
                if swap or scan == 20:
                    centroid_row = len(cents)
                    if scan == 10:
                        first = centroid_row
                    y1 = K + WATER + PROTON
                    cents.append(dict(dataset="fixture", raw_file=raw, scan_number=scan, precursor_charge=7,
                                      precursor_mz=400.0, retention_time_seconds=float(scan),
                                      mz=[F + PROTON, y1 + short, y1 + long] if scan == 10 else [],
                                      intensity=[1000.0, 4000.0, 9000.0] if scan == 10 else []))
                else:
                    centroid_row = first
                a, b = [5, 6, code, 0], [code, 19, 5, 0]  # FGK and KWF
                anns.append(dict(sequence_integer_1=b if swap else a, sequence_integer_2=a if swap else b,
                                 alpha_position_crosslinker=[1 if swap else 3],
                                 beta_position_crosslinker=[3 if swap else 1],
                                 crosslinker=[linker], raw_file=raw + ".pkl", scan_number=[scan],
                                 precursor_charge_onehot=[0.0] * 6, collision_energy_aligned_normed=nce))
                indices.append(centroid_row)
        pq.write_table(pa.Table.from_pylist(anns), annotation)
        mapping = np.array(indices, dtype="<u4").tobytes()
        meta = {b"source_split": public.encode(), b"source_annotation_file": annotation.name.encode(),
                b"source_annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest().encode(),
                b"source_row_to_output_row_b64": base64.b64encode(zlib.compress(mapping))}
        pq.write_table(pa.Table.from_pylist(cents).replace_schema_metadata(meta),
                       source / f"cms2_{split}_centroid.parquet")


class CleavableDatasetTests(unittest.TestCase):
    def test_mass_and_state_for_both_chains_and_charges(self):
        for linker, (short, long, _) in STUBS.items():
            for z in (2, 3, 7):
                with self.subTest(linker=linker, z=z):
                    key = builder.GroupKey("FGK", 3, "KWF", 1, linker, z)
                    base, states, lengths = builder.build_theoretical_mz(key)
                    grid, mask = builder.expand_grid(base, states)
                    self.assertEqual(grid.shape, (2, 39, 46, 12))
                    self.assertEqual(lengths.tolist(), [3, 3])
                    self.assertFalse(mask[:, 2:].any())
                    self.assertFalse(mask[0, :2, :, 6:9].any())
                    self.assertFalse(mask[1, :2, :, 9:12].any())
                    self.assertFalse(grid[~mask].any())
                    for q in range(1, min(z, 3) + 1):
                        self.assertAlmostEqual(base[0, 0, q - 1], F / q + PROTON)
                        self.assertAlmostEqual(base[0, 1, q + 2], (K + WATER + short) / q + PROTON)
                        self.assertAlmostEqual(base[0, 1, q + 8], (K + WATER + long) / q + PROTON)
                        self.assertAlmostEqual(base[1, 0, q - 1], (K + short) / q + PROTON)
                        self.assertAlmostEqual(base[1, 0, q + 5], (K + long) / q + PROTON)
                    if z == 2:
                        self.assertFalse(mask[..., [2, 5, 8, 11]].any())

    def test_residue_support_cannot_cross_linker_state(self):
        grid = np.zeros(builder.GRID_SHAPE)
        base = np.zeros(builder.BASE_SHAPE)
        states = np.full(base.shape, -1, dtype=np.int8)
        states[0, :2, 0] = [0, 1]
        token = next(t for t in builder.RESIDUE_TOKENS if np.isclose(builder.CMS2_OFFSETS[t, 0], 71.037114))
        base[0, :2, 0] = [200, 271.037114]
        cell = (0, 0, token, 0)
        grid[cell] = 271.037114
        matched = grid > 0
        self.assertFalse(builder.full_support(base, states, grid, matched)[cell])
        states[0, 1, 0] = 0
        self.assertTrue(builder.full_support(base, states, grid, matched)[cell])

    def test_build_merge_provenance_and_reconstruction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(tmp) / "source"
            source.mkdir()
            write_fixture(source)
            output = Path(tmp) / "output"
            output.mkdir()  # A pre-existing empty directory is a supported destination.
            builder.build_dataset(source, output, limit=None)
            builder.verify_dataset(output)
            with self.assertRaises(FileExistsError):
                builder.build_dataset(source, output, limit=None)
            meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["average_across_nce"])
            self.assertEqual(meta["group_by"], ["sequence_a", "site_a", "sequence_b", "site_b", "crosslinker", "charge"])
            self.assertEqual(meta["train_statistics"]["aux_order"], ["precursor_charge", "precursor_mz", "rt_seconds"])
            for split in builder.SPLITS:
                recs = np.load(output / f"{split}.npy", allow_pickle=False)
                self.assertEqual(len(recs), 2)
                self.assertEqual(meta["splits"][split]["mixed_nce_groups"], 2)
                self.assertFalse({"nce", "source_nce", "collision_energy_aligned_normed"}.intersection(recs.dtype.names))
                for row in recs:
                    self.assertEqual(row["n_spectra"], 2)
                    self.assertEqual(row["precursor_charge"], 7)
                    self.assertEqual(row["rt_seconds"], 15)
                    self.assertEqual(row["sequence"].tolist(), [b"FGK", b"KWF"])
                    main = row["target"][:, :, builder.MAIN_TOKEN_ID, :]
                    np.testing.assert_allclose([main[0, 0, 0], main[0, 1, 3], main[0, 1, 9]], [0.1, 0.2, 0.3])
            members = pq.read_table(output / "group_members.parquet").to_pylist()
            for split in builder.SPLITS:
                m = [r for r in members if r["split"] == split]
                self.assertEqual(len(m), 4)
                self.assertEqual(sorted(i for r in m for i in r["cms2_source_rows"]), list(range(6)))
                self.assertEqual({r["source_nce"] for r in m}, {0.25, 0.34})
            # A legal, positive intensity with the wrong magnitude must fail a full verify.
            recs = np.load(output / "test.npy", mmap_mode="r+")
            recs["target"][0, 0, 0, builder.MAIN_TOKEN_ID, 0] = 0.5
            recs.flush()
            del recs
            with self.assertRaisesRegex(ValueError, "source reconstruction"):
                builder.verify_dataset(output)


if __name__ == "__main__":
    unittest.main()

