"""Regression tests for cMS2 peak selection, ranking and fragment recall."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spectrum_transformer.config import (
    CMS2_CHANNEL_NAMES, CMS2_INTACT_CHANNELS, CMS2_INTACT_ROW, MAIN_TOKEN_IDS, SATELLITE_TOKEN_IDS,
)
from spectrum_transformer.data.xl_dataset import AUX_FEATURE_NAMES, XLSpectrumDataset, _record_dtype
from spectrum_transformer.models.xl_transformer import XLSpectrumTransformer
from spectrum_transformer.training.xl_engine import FRAGMENT_RECALL_NAMES, evaluate_xl, predict_xl_batch
from spectrum_transformer.training.xl_losses import (
    XL_LOSS_WEIGHTS, _local_peaks, _main_token_ranking_loss, _satellite_empty_loss, xl_compound_loss,
)


class EchoPrediction(torch.nn.Module):
    def forward(self, x, aux_features, **kwargs):
        return {"prediction": x, "main_anchor_prob": torch.zeros_like(x)}


class MainTokenRankingTests(unittest.TestCase):
    def test_all_pairs_are_averaged_by_group_then_spectrum(self):
        main = MAIN_TOKEN_IDS[0]
        target = torch.zeros(3, 2, 8, 46, 12)
        scores = torch.zeros_like(target)
        valid = torch.zeros_like(target, dtype=torch.bool)

        def group(sample, chain, channel, labels, predictions, token=main):
            cells = (sample, chain, slice(0, len(labels)), token, channel)
            target[cells] = torch.tensor(labels, dtype=target.dtype)
            scores[cells] = torch.tensor(predictions)
            valid[cells] = True

        # Two positive cuts x two empty cuts: penalties 0, 0, .04, .06.
        group(0, 0, 0, [.6, .1, 0, 0, .03, .01], [.15, .08, .10, .12, .99, .99])
        scores[0, 0, 6, main, 0] = .99  # invalid zero-target cut
        group(0, 1, 0, [.1, 0], [.03, .07])  # group mean .06
        group(1, 0, 9, [.1, 0], [.10, .10])  # group mean .02
        # Neither different channels, chains, spectra nor neighboring tokens may supply a pair.
        group(0, 0, 1, [.8], [.01])
        group(0, 0, 2, [0], [.99])
        group(0, 1, 1, [0], [.99])
        group(2, 0, 1, [0], [.99])
        group(2, 0, 1, [.8, 0], [.01, .99], token=main + 1)

        scores.requires_grad_()
        loss = _main_token_ranking_loss(scores, target, valid)
        self.assertAlmostEqual(loss.item(), ((.025 + .06) / 2 + .02) / 2)
        loss.backward()
        expected_grad = torch.zeros_like(scores)
        expected_grad[0, 0, 1:4, main, 0] = torch.tensor([-.125, .0625, .0625])
        expected_grad[0, 1, :2, main, 0] = torch.tensor([-.25, .25])
        expected_grad[1, 0, :2, main, 9] = torch.tensor([-.5, .5])
        torch.testing.assert_close(scores.grad, expected_grad)

    def test_batch_without_eligible_pairs_has_zero_loss_and_gradient(self):
        target = torch.ones(2, 2, 3, 46, 12)
        target[1] = 0
        scores = torch.zeros_like(target, requires_grad=True)
        loss = _main_token_ranking_loss(scores, target, torch.ones_like(target, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


class SatelliteEmptyTests(unittest.TestCase):
    def test_top_four_per_group_and_equal_spectrum_weight(self):
        first, second = SATELLITE_TOKEN_IDS[:2]
        target = torch.zeros(3, 2, 8, 46, 12)
        scores = torch.zeros_like(target)
        valid = torch.zeros_like(target, dtype=torch.bool)

        def set_scores(sample, chain, token, channel, values):
            cells = (sample, chain, slice(0, len(values)), token, channel)
            scores[cells] = torch.tensor(values)
            valid[cells] = True

        set_scores(0, 0, first, 0, [.07, .05, .08, .10, .14, .19, .99, .98])
        target[0, 0, 6, first, 0] = .001  # a weak observation is still occupied
        valid[0, 0, 7, first, 0] = False
        set_scores(0, 0, first, 1, [.05, .25])
        set_scores(0, 1, first, 0, [.11])
        set_scores(0, 0, second, 0, [.04])  # eligible group with zero penalty
        set_scores(1, 0, first, 0, [.35])
        set_scores(2, 0, MAIN_TOKEN_IDS[0], 0, [.99])
        set_scores(2, 0, 0, 0, [.98])  # residue-event token

        scores.requires_grad_()
        loss = _satellite_empty_loss(scores, target, valid)
        self.assertAlmostEqual(loss.item(), ((.0775 + .10 + .06 + 0) / 4 + .30) / 2)
        loss.backward()
        expected = torch.zeros_like(scores)
        expected[0, 0, 2:6, first, 0] = 1 / 32
        expected[0, 0, 1, first, 1] = 1 / 16
        expected[0, 1, 0, first, 0] = 1 / 8
        expected[1, 0, 0, first, 0] = 1 / 2
        torch.testing.assert_close(scores.grad, expected)

    def test_no_valid_empty_satellite_is_differentiable_zero(self):
        scores = torch.ones(2, 2, 2, 46, 12, requires_grad=True)
        valid = torch.ones_like(scores, dtype=torch.bool)
        valid[0] = False
        target = valid.float()  # invalid zeros in spectrum 0; observed peaks in spectrum 1
        loss = _satellite_empty_loss(scores, target, valid)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


class IntactLossTests(unittest.TestCase):
    def test_only_existing_reconstruction_terms_see_the_intact_row(self):
        row, token, channel = CMS2_INTACT_ROW, MAIN_TOKEN_IDS[0], CMS2_INTACT_CHANNELS[0]
        target = torch.zeros(1, 2, row + 1, 46, 12)
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[0, 0, row-2:, token, channel] = True
        # The intense intact target would suppress the last backbone peak if it entered local pooling.
        target[0, 0, row-1:, token, channel] = torch.tensor([.2, .9])
        scores = torch.full_like(target, .1)

        def evaluate(p, y, v):
            return xl_compound_loss(p, y, v, charges=torch.tensor([4]), lengths=torch.tensor([[40, 4]]),
                                    main_anchor_prob=torch.full_like(p, .45))

        _, backbone = evaluate(scores[:, :, :row], target[:, :, :row], valid[:, :, :row])
        p = scores.clone().requires_grad_()
        loss, terms = evaluate(p, target, valid)
        loss.backward()
        self.assertNotEqual(p.grad[0, 0, row, token, channel].item(), 0.)
        self.assertFalse(p.grad[~valid].any())
        scores[0, 0, row, token, channel] = .8
        _, changed = evaluate(scores, target, valid)
        recon = {"v20_backbone_l1", "v20_backbone_cosine_loss", "v20_backbone_global_loss",
                 "v20_family_weighted_loss", "loss"}
        for name in recon:
            self.assertNotAlmostEqual(terms[name], changed[name])
        for name in terms.keys() - recon:
            self.assertAlmostEqual(terms[name], backbone[name])
            self.assertAlmostEqual(terms[name], changed[name])
        expected = sum(weight * terms[name] for name, weight in XL_LOSS_WEIGHTS.items())
        np.testing.assert_allclose(terms["loss"], expected, rtol=1e-6)


class CMS2Tests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        grid = (2, 39, 46, 12)
        widths = {"sequence_bytes": 4, "crosslinker_bytes": 4}
        dtype = _record_dtype(widths, grid, grid)
        records = np.zeros(2, dtype=dtype)
        records["group_id"] = [b"one", b"two"]
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

    def intact_labels(self):
        directory = self.dataset.dataset_dir / "intact"
        directory.mkdir()
        dtype = [("group_id", "S64"), ("mz", "<f8", (2, 2, 3)),
                 ("target", "<f4", (2, 2, 3)), ("valid_mask", "?", (2, 2, 3))]
        labels = np.zeros(2, dtype=dtype)
        labels["group_id"] = [b"one", b"two"]
        labels["mz"] = np.arange(24).reshape(2, 2, 2, 3) + 200
        labels["target"] = np.arange(24).reshape(2, 2, 2, 3) / 25
        labels["valid_mask"] = True
        labels["mz"][0, 0, 0, 0] = 0
        labels["valid_mask"][0, 0, 0, 0] = False
        np.save(directory / "train.npy", labels)
        manifest = {"label_type": "intact_peptide_stub", "label_schema_version": 1,
                    "axis_values": [["alpha (v4 sequence[0])", "beta (v4 sequence[1])"],
                                    ["short", "long"], [1, 2, 3]], "target_normalization": "sqrt_clip",
                    "clip_intensity": 50000}
        (directory / "meta.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory, labels

    def test_intact_collation_and_metrics(self):
        directory, labels = self.intact_labels()
        ds = XLSpectrumDataset(self.dataset.dataset_dir, "train", directory)
        batch = ds.collate_fn([ds[1], ds[0]])
        row = CMS2_INTACT_ROW
        channels = list(CMS2_INTACT_CHANNELS)
        cells = (slice(None), slice(None), row, self.main, channels)
        for field, key in (("target", "target"), ("mz", "input_theo_mz"), ("valid_mask", "valid_mask")):
            expected = labels[field][[1, 0]].reshape(2, 2, 6).copy()
            np.testing.assert_array_equal(batch[key][cells].numpy(), expected)
        self.assertEqual(batch["valid_mask"][:, :, row].sum().item(), 23)
        self.assertFalse(batch["main_anchor_mask"][:, :, row].any())
        expected_input = (batch["input_theo_mz"][cells] - torch.tensor(channels) - 100) / 2
        torch.testing.assert_close(batch["input"][cells], expected_input.masked_fill(~batch["valid_mask"][cells], 0).float())
        old_batch = self.dataset.collate_fn([ds[1], ds[0]])
        for key in ("input", "target", "valid_mask", "main_anchor_target"):
            torch.testing.assert_close(batch[key][:, :, :row], old_batch[key])
        # Calling every intact candidate changes no plain/short/long backbone counts.
        batch["input"].zero_()
        batch["input"][cells] = batch["valid_mask"][cells].float()
        report = evaluate_xl(EchoPrediction(), [batch], torch.device("cpu"))
        self.assertEqual(report["intact_recall"], 1.)
        self.assertEqual(report["backbone_cosine"], 0.)
        self.assertGreater(report["cosine"], 0.)
        self.assertEqual(report["plain_recall"], 0.)
        self.assertEqual(report["long_recall"], 0.)
        np.save(directory / "train.npy", labels[::-1])
        with self.assertRaisesRegex(ValueError, "group_id"):
            XLSpectrumDataset(self.dataset.dataset_dir, "train", directory)

    def test_intact_uses_main_head_and_preserves_backbone_when_masked(self):
        directory, _ = self.intact_labels()
        ds = XLSpectrumDataset(self.dataset.dataset_dir, "train", directory)
        batch = ds.collate_fn([ds[0], ds[1]])
        base_batch = self.dataset.collate_fn([ds[0], ds[1]])
        config = dict(d_model=16, n_heads=2, n_layers=2, ff_dim=32, dropout=0)
        torch.manual_seed(5)
        base = XLSpectrumTransformer(**config).eval()
        torch.manual_seed(5)
        model = XLSpectrumTransformer(**config, intact_mode=True).eval()
        pred = predict_xl_batch(model, batch)["prediction"]
        pred[:, :, CMS2_INTACT_ROW].sum().backward()
        self.assertGreater(model.backbone_head.main_head[-1].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.intact_position.grad.abs().sum().item(), 0.)
        for head in (model.backbone_head.satellite_head, model.backbone_head.residue_head):
            self.assertFalse(head[-1].weight.grad.any())
        batch["valid_mask"][:, :, CMS2_INTACT_ROW] = False
        with torch.inference_mode():
            pred = predict_xl_batch(model, batch)["prediction"]
            torch.testing.assert_close(pred[:, :, :CMS2_INTACT_ROW], predict_xl_batch(base, base_batch)["prediction"])
            self.assertFalse(pred[:, :, CMS2_INTACT_ROW].any())

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
