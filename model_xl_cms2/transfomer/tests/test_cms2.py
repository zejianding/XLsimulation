"""Regression tests for the cMS2 plain/short boundary and fragment recall."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spectrum_transformer.config import CMS2_CHANNEL_NAMES, MAIN_TOKEN_IDS
from spectrum_transformer.data.xl_dataset import AUX_FEATURE_NAMES, XLSpectrumDataset, _record_dtype
from spectrum_transformer.training.xl_engine import FRAGMENT_RECALL_NAMES, evaluate_xl
from spectrum_transformer.training.xl_losses import _local_peaks


class EchoPrediction(torch.nn.Module):
    def forward(self, x, aux_features, **kwargs):
        return {"prediction": x, "main_anchor_prob": torch.zeros_like(x)}


class CMS2Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        grid = (2, 39, 46, 12)
        widths = {"sequence_bytes": 4, "crosslinker_bytes": 4}
        dtype = _record_dtype(widths, grid, grid)
        records = np.zeros(2, dtype=dtype)
        records["sequence"] = [b"AKAA", b"AAAK"]
        records["lengths"] = [4, 4]
        records["link_sites"] = [2, 4]
        records["crosslinker"] = [b"DSSO", b"DSBU"]
        records["precursor_charge"] = [4, 7]
        self.main = MAIN_TOKEN_IDS[0]
        # Chain A cuts 1 and 2 straddle K2. Chain B links at its terminal K4.
        for chain, cut, channel, target in (
            (0, 1, 0, .4), (0, 2, 0, .8), (0, 2, 6, .7),
            (0, 2, 4, .4), (1, 2, 5, .4), (0, 1, 10, .03), (1, 2, 0, 0.),
        ):
            cell = (0, chain, cut - 1, self.main, channel)
            records["input_theo_mz"][cell] = 100.
            records["target"][cell] = target
        records[1]["input_theo_mz"][0, 0, self.main, 0] = 120.
        records[1]["target"][0, 0, self.main, 0] = .5
        np.save(path / "train.npy", records)
        metadata = {
            "schema_version": 4, "spectrum_type": "cms2", "average_across_nce": True,
            "input_representation": "expanded_raw_mz", "input_dtype": "float64",
            "grid_shape": grid, "input_shape": grid, "main_token_id": self.main,
            "string_widths": widths, "fields": dtype.names, "record_bytes": dtype.itemsize,
            "channel_names": CMS2_CHANNEL_NAMES, "min_valid_mz": 100, "sequence_constrained": "full_same_state",
            "clip_intensity": 50000, "target_normalization": "sqrt_clip", "splits": {"train": {"written_groups": 2}},
            "train_statistics": {"aux_order": AUX_FEATURE_NAMES, "aux_features": {"mean": [0]*3, "std": [1]*3},
                                 "mz_channels": {"mean": list(range(100, 112)), "std": [2]*12}},
        }
        (path / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.dataset = XLSpectrumDataset(path, "train")
        self.addCleanup(lambda: setattr(self, "dataset", None))  # close the mmap before Windows removes the file

    def test_adjacent_plain_short_peaks(self):
        row = self.dataset[0].copy()
        batch = self.dataset.collate_fn([row])
        valid = batch["valid_mask"]
        self.assertTrue(valid[0, 1, 1, self.main, 0])  # valid despite zero target and zero normalized m/z
        self.assertFalse(valid[0, 0, 0, self.main, 6])  # long B1 before K2
        self.assertEqual(batch["input"][0, 0, 1, self.main, 6], -3.)
        for grid in (batch["main_anchor_target"], _local_peaks(batch["target"], valid)):
            self.assertEqual(grid[0, 0, :2, self.main, 0].tolist(), [0, 1])
        # A stronger neighboring token suppresses the amplitude peak only.
        row["input_theo_mz"][0, 1, self.main + 1, 0] = 101.
        row["target"][0, 1, self.main + 1, 0] = .9
        batch = self.dataset.collate_fn([row])
        self.assertTrue(batch["main_anchor_target"][0, 0, 1, self.main, 0])
        self.assertFalse(_local_peaks(batch["target"], batch["valid_mask"])[0, 0, 1, self.main, 0])

    def test_group_recall_pools_valid_cells_across_batches(self):
        batches = [self.dataset.collate_fn([self.dataset[i]]) for i in range(2)]
        self.assertEqual([b["linker_id"].item() for b in batches], [0, 1])
        for batch in batches:
            batch["input"].zero_()
        pred = batches[0]["input"]
        for chain, cut, channel in ((0, 1, 0), (0, 2, 6), (1, 2, 5)):
            pred[0, chain, cut - 1, self.main, channel] = .1
        pred[0, 0, 1, self.main, 4] = .05  # threshold equality is a miss
        report = evaluate_xl(EchoPrediction(), batches, torch.device("cpu"))
        expected = (1/3, 1/2, 1., 1/2, 0., 1.)
        self.assertEqual(tuple(report[name] for name in FRAGMENT_RECALL_NAMES), expected)
        self.assertEqual(report["by_charge"]["4"]["plain_recall"], .5)
        self.assertIsNone(report["by_charge"]["7"]["long_recall"])
