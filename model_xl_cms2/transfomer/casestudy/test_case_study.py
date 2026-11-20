"""Boundary and serialization tests; optional real checkpoint integration check."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_study as case


class CaseStudyTests(unittest.TestCase):
    def test_sequence_validation(self):
        for sequence in ("PEPTIDE", "AC+57.021M(+15.995)K", "[+42.011]PEPTIDE", "A" * 40):
            self.assertGreaterEqual(case.validate_peptide(sequence, 2), 2)
        for sequence in ("A", "A" * 41, "PEPTXDE", "peptide", "M[UNIMOD:35]K", "PEP TIDE", "M(+15.99K"):
            with self.assertRaises(ValueError):
                case.validate_peptide(sequence, 2)
        with self.assertRaises(ValueError):
            case.validate_peptide("PEPTIDE", 0)

    def test_mask_and_modified_mass(self):
        mz = case.build_input_mz_matrix("AC+57.021MK", 100)
        mask, count = case.build_valid_mask("AC+57.021MK", 1, 100)
        self.assertEqual(count, 3)
        self.assertEqual(mz.shape, (39, 46, 6))
        self.assertFalse(mask[3:].any())
        self.assertFalse(mask[:, :, [1, 2, 4, 5]].any())
        self.assertTrue(np.all(mz[mask > 0] >= 100))
        self.assertAlmostEqual(case.calculate_pepmass("AC+57.021MK", 2) - case.calculate_pepmass("ACMK", 2), 57.021 / 2)

    def fixture(self):
        mz, prediction, mask = (np.zeros(case.SHAPE, dtype=np.float32) for _ in range(3))
        mz.flat[:4] = [200, 200.03, 200.06, 300]
        prediction.flat[:4] = [0.2, 0.8, 0.4, 0.01]
        mask.flat[:4] = 1
        return mz, prediction, mask

    def test_merge_uses_anchor_not_chain_and_threshold_is_strict(self):
        peaks = case.export_peaks(*self.fixture(), 0.01, 0.05, "model")
        np.testing.assert_allclose(peaks, [[200.03, 0.8], [200.06, 0.4]], rtol=1e-6)
        relative = case.export_peaks(*self.fixture(), 0.01, 0.05, "relative100")
        np.testing.assert_allclose(relative[:, 1], [100, 25])

    def test_invalid_and_empty_grid(self):
        mz, pred, mask = self.fixture()
        self.assertEqual(len(case.export_peaks(mz, pred, mask, 1, 0.05, "model")), 0)
        pred.flat[0] = np.nan
        with self.assertRaises(ValueError):
            case.export_peaks(mz, pred, mask, 0, 0.05, "model")
        with self.assertRaises(ValueError):
            case.export_peaks(*self.fixture(), 0, 0, "model")

    def test_saved_grid_export_roundtrip_and_overwrite(self):
        meta = dict(schema_version=1, peptide="PEPTIDE", peptide_length=7, charge=2, pepmass=400.2, rt_seconds=1800)
        args = argparse.Namespace(min_intensity=0.01, mz_merge_tol=0.05, intensity_space="model")
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            case.write_outputs(first, *self.fixture(), meta, args, save_grid=True)
            case.main(["export", "--grid", str(first / "grid.npz"), "--out-dir", str(second)])
            self.assertEqual((first / "prediction.mgf").read_bytes(), (second / "prediction.mgf").read_bytes())
            self.assertEqual(len((first / "grid.csv").read_text().splitlines()), 39 * 46 * 6 + 1)
            self.assertEqual(json.loads((second / "metadata.json").read_text())["export"]["n_peaks"], 2)
            (first / "notes.txt").write_text("keep this note")
            grid_before = (first / "grid.npz").read_bytes()
            # Re-export in place: replace peaks and export metadata, keep the source grid.
            case.main(["export", "--grid", str(first / "grid.npz"), "--out-dir", str(first), "--min-intensity", "0.5"])
            self.assertEqual((first / "grid.npz").read_bytes(), grid_before)
            self.assertEqual(json.loads((first / "metadata.json").read_text())["export"]["n_peaks"], 1)
            self.assertEqual(len((first / "peaks.tsv").read_text().splitlines()), 2)
            # A new prediction replaces the grid files as well as the peak files.
            mz, pred, mask = self.fixture()
            pred *= 0.5
            case.write_outputs(first, mz, pred, mask, meta, args, save_grid=True)
            np.testing.assert_array_equal(np.load(first / "prediction_grid.npy"), pred)
            with np.load(first / "grid.npz") as saved:
                np.testing.assert_array_equal(saved["prediction"], pred)
            self.assertEqual(json.loads((first / "metadata.json").read_text())["export"]["n_peaks"], 2)
            self.assertEqual((first / "notes.txt").read_text(), "keep this note")


if __name__ == "__main__":
    unittest.main()
