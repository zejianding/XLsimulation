"""Score batches of linked peptide hypotheses with the Model XL test metric."""

from pathlib import Path

import numpy as np
import torch

from spectrum_transformer.data.dataset import SEQUENCE_CONTEXT_RESIDUE_TO_ID, normalize_features
from spectrum_transformer.data.xl_dataset import AUX_FEATURE_NAMES, XLInputCollator
from spectrum_transformer.models.xl_transformer import XLSpectrumTransformer
from spectrum_transformer.training.engine import _cosine_per_sample
from spectrum_transformer.training.xl_engine import predict_xl_batch

from xl_rescore_projection import XLProjection


class XLRescoreInference:
    """Keep the XL checkpoint and grid projection on one inference device."""

    def __init__(self, checkpoint_path, device):
        self.device = torch.device(device)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "xl_spectrum_transformer":
            raise ValueError("The checkpoint is not a Model XL checkpoint")
        metadata = checkpoint["data_metadata"]
        collator = XLInputCollator(metadata, checkpoint["data_schema_version"])
        if collator.schema_version != 2 or (
            metadata["sequence_constrained"], metadata["target_normalization"], metadata["clip_intensity"]
        ) != ("full", "sqrt_clip", 50000):
            raise ValueError("Expected the six-channel Model XL full-constrained sqrt_clip(50000) checkpoint")

        model = XLSpectrumTransformer(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(self.device).eval()
        self.projection = XLProjection(self.device)
        stats = metadata["train_statistics"]
        self._mz_mean = torch.tensor(stats["mz_channels"]["mean"], dtype=torch.float64, device=self.device)
        self._mz_std = torch.tensor(stats["mz_channels"]["std"], dtype=torch.float64, device=self.device)
        self._aux_mean = np.asarray(stats["aux_features"]["mean"], dtype=np.float64)
        self._aux_std = np.asarray(stats["aux_features"]["std"], dtype=np.float64)
        self._aux_order = list(AUX_FEATURE_NAMES)
        aa_ids = torch.zeros(256, dtype=torch.long)
        for residue, index in SEQUENCE_CONTEXT_RESIDUE_TO_ID.items():
            aa_ids[ord(residue)] = index
        self._aa_ids = aa_ids.to(self.device)
        self._cam_scale = float(np.tanh(57.021464 / 50.0))
        self._ox_scale = float(np.tanh(15.994915 / 50.0))

    def prepare_spectrum(self, mz, intensity):
        return self.projection.prepare_spectrum(mz, intensity)

    def _encode_cleavages(self, aa, ox, lengths, sites):
        cleavage = torch.arange(1, aa.shape[-1], device=self.device)[None, None, :]
        length = lengths[..., None]
        site = sites[..., None]
        scalars_valid = cleavage < length
        residue_ids = self._aa_ids[aa.long()]
        paired_ids = torch.stack((residue_ids[..., :-1], residue_ids[..., 1:]), dim=-1)
        ca = aa == ord("C")
        mod = ca | ox
        scaled = ca * self._cam_scale + ox * self._ox_scale
        scalars = torch.stack((
            (cleavage.double() / length).float(),
            ((length - cleavage).double() / length).float(),
            mod[..., :-1],
            mod[..., 1:],
            scaled[..., :-1],
            scaled[..., 1:],
            torch.zeros_like(scaled[..., :-1]),
            ((cleavage.double() + 0.5 - site) / length).float(),
            site <= cleavage,
            site == cleavage,
            site == cleavage + 1,
        ), dim=-1)
        valid = scalars_valid[..., None]
        return paired_ids.masked_fill(~valid, 0), scalars.masked_fill(~valid, 0)

    @torch.inference_mode()
    def score(self, aa, ox, lengths, sites, charge, precursor_mz, rt_seconds, peak_state):
        count = len(lengths)
        if count == 0:
            return np.empty(0, dtype=np.float32)

        # Dropped trailing columns are outside every row's cleavage mask.
        width = int(np.max(lengths))
        aa = torch.as_tensor(np.ascontiguousarray(aa[..., :width]), dtype=torch.uint8, device=self.device)
        ox = torch.as_tensor(np.ascontiguousarray(ox[..., :width]), dtype=torch.bool, device=self.device)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        sites = torch.as_tensor(sites, dtype=torch.long, device=self.device)
        grid, valid, target = self.projection.project(aa, ox, lengths, sites, charge, peak_state)
        normalized = ((grid - self._mz_mean) / self._mz_std).masked_fill(~valid, 0).float()
        aux_values = dict(precursor_charge=charge, precursor_mz=precursor_mz, rt_seconds=rt_seconds)
        aux = np.asarray([[aux_values[name] for name in self._aux_order]], dtype=np.float64)
        aux = torch.as_tensor(normalize_features(aux, self._aux_mean, self._aux_std), device=self.device)
        aa_ids, scalars = self._encode_cleavages(aa, ox, lengths, sites)
        batch = {
            "input": normalized,
            "valid_mask": valid,
            "aux_features": aux.expand(count, -1),
            "precursor_charge": torch.full((count,), charge, dtype=torch.long, device=self.device),
            "lengths": lengths,
            "cleavage_aa_ids": aa_ids,
            "cleavage_scalar_features": scalars,
        }

        prediction = predict_xl_batch(self.model, batch)["prediction"]
        return _cosine_per_sample(prediction, target, valid).cpu().numpy()
