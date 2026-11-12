"""Export final V20 or paired XL transformer predictions to MGF.

XL accepts a dataset directory, a structured split NPY, or a numeric/string NPZ.
An XL NPZ can contain the structured array as ``records`` or store its fields
as separate arrays. The required fields are ``input_theo_mz``, ``sequence``,
``lengths``, ``link_sites`` (1-based), ``precursor_charge``, ``precursor_mz``,
and ``rt_seconds``. Optional MGF metadata comes from ``group_id``,
``crosslinker``, and ``linker_mass``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from cluster_postprocess import (
    NUM_TOKENS,
    SCENE_TEMPLATES,
    get_length_bin,
    load_n_true_mean_from_stats,
    load_token_profile_prior,
    resolve_scene_params,
    run_cluster_postprocess,
    summarize_diagnostics,
)
from predict_mgf import prediction_to_mgf_peaks
from spectrum_transformer.config import CHANNEL_NAMES, MAX_CLEAVAGE_LEN
from spectrum_transformer.data.xl_dataset import XLInputCollator
from spectrum_transformer.models.xl_transformer import XLSpectrumTransformer
from spectrum_transformer.training.xl_engine import predict_xl_batch
from spectrum_transformer.warnings import configure_v20_warning_filters

configure_v20_warning_filters()


AA_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.094963050,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}

_SINGLE_MASS_RE = re.compile(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)")
_MOD_TOKEN_RE = re.compile(
    rf"(?:{_SINGLE_MASS_RE.pattern})(?:[+-](?:\d+\.\d+|\d+|\.\d+))*"
)


def _parse_mod_token(raw: str) -> float:
    parts = _SINGLE_MASS_RE.findall(str(raw))
    return sum(float(part) for part in parts) if parts else 0.0


def _try_read_mod_token(seq: str, i: int) -> tuple[float | None, int]:
    n = len(seq)
    if i >= n:
        return None, i
    if seq[i] in "([":
        close = ")" if seq[i] == "(" else "]"
        j = seq.find(close, i + 1)
        if j != -1:
            inner = seq[i + 1:j].strip()
            if inner and any(ch.isdigit() for ch in inner) and _MOD_TOKEN_RE.fullmatch(inner):
                return _parse_mod_token(inner), j + 1
        return None, i
    m = _MOD_TOKEN_RE.match(seq, i)
    if m and any(ch.isdigit() for ch in m.group(0)):
        return _parse_mod_token(m.group(0)), m.end()
    return None, i


def _is_amino_acid_char(ch: str) -> bool:
    return "A" <= ch <= "Z" and ch in AA_MASS


def parse_sequence_with_mods(seq: str) -> tuple[list[tuple[str, float]], float]:
    residues: list[tuple[str, float]] = []
    nterm_shift = 0.0
    s = str(seq).strip()
    n = len(s)
    i = 0

    while i < n:
        ch = s[i]
        if _is_amino_acid_char(ch):
            break
        delta, new_i = _try_read_mod_token(s, i)
        if delta is not None:
            nterm_shift += float(delta)
            i = new_i
            continue
        if ch in ".-_[]() ":
            i += 1
            continue
        break

    while i < n:
        ch = s[i]
        if not _is_amino_acid_char(ch):
            delta, new_i = _try_read_mod_token(s, i)
            if delta is not None and residues:
                aa, old_mass = residues[-1]
                residues[-1] = (aa, old_mass + float(delta))
                i = new_i
                continue
            i += 1
            continue

        aa = ch
        mass = AA_MASS[aa]
        i += 1
        while i < n:
            delta, new_i = _try_read_mod_token(s, i)
            if delta is None:
                break
            mass += float(delta)
            i = new_i
        residues.append((aa, mass))

    return residues, nterm_shift


def count_peptide_residues(seq: str) -> int:
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    return sum(1 for ch in str(seq or "") if ch in aa)


def _sanitize_group_filename(name: str, max_len: int = 80) -> str:
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    if not name:
        name = "EMPTY_SEQ"
    return name[:max_len]


def _make_group_filename(seq: str, charge: int) -> str:
    prefix = _sanitize_group_filename(seq, max_len=80)
    seq_hash = hashlib.md5(str(seq).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_charge{charge}_{seq_hash}.mgf"


def parse_cluster_export_tokens(value) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = str(value).replace(",", " ").split()
    tokens: list[int] = []
    seen = set()
    for item in raw_items:
        tok = int(item)
        if tok < 0 or tok >= NUM_TOKENS:
            raise ValueError(
                f"cluster export token out of range [0,{NUM_TOKENS - 1}]: {tok}"
            )
        if tok not in seen:
            tokens.append(tok)
            seen.add(tok)
    return tokens


if os.name == "nt":
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass

try:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
except Exception:
    pass


def load_model(checkpoint_path: str | Path, device: torch.device):
    transfomer_root = os.path.abspath(os.path.dirname(__file__))
    if transfomer_root not in sys.path:
        sys.path.insert(0, transfomer_root)

    from spectrum_transformer.models.factory import build_model, prepare_checkpoint_for_inference

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_type") == "xl_spectrum_transformer":
        model_config = ckpt["model_config"]
        model = XLSpectrumTransformer(**model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device).eval(), ckpt, model_config

    model_config, state_dict = prepare_checkpoint_for_inference(ckpt)
    if "base_width" in model_config or "aux_map_channels" in model_config:
        raise ValueError(
            "predict_npz_to_mgf.py is now V20 transformer-only. "
            "Legacy ResNet checkpoints are no longer supported here."
        )

    model = build_model(model_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, ckpt, model_config


class NPZPredictionDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        channel_means,
        channel_stds,
        aux_means,
        aux_stds,
        in_channels: int = 6,
        min_mz: float = 100.0,
    ) -> None:
        raw = dict(np.load(str(npz_path), allow_pickle=True))

        self.inputs = raw["inputs"] if "inputs" in raw else raw["input_mz_matrices"]
        self.position_masks = raw["position_masks"]
        self.channel_masks = raw["channel_masks"]
        self.charges = raw["charges"]
        self.pepmasses = raw["pepmasses"]
        self.rtinseconds = raw["rtinseconds"]
        self.seqs = raw.get("seqs", None)
        if self.inputs is None:
            raise KeyError(f"Cannot find input data. NPZ keys: {list(raw.keys())}")

        ch_means = np.asarray(channel_means, dtype=np.float32).reshape(1, 1, 1, -1)
        ch_stds = np.asarray(channel_stds, dtype=np.float32).reshape(1, 1, 1, -1)
        aux_m = np.asarray(aux_means, dtype=np.float32).reshape(1, -1)
        aux_s = np.asarray(aux_stds, dtype=np.float32).reshape(1, -1)

        vm = self.position_masks[:, :, None, None] * self.channel_masks[:, None, None, :]
        self.valid_masks = np.broadcast_to(vm.astype(np.float32), self.inputs.shape)
        mz_valid = (self.inputs >= float(min_mz)).astype(np.float32)
        self.valid_masks = self.valid_masks * mz_valid
        self.inputs = np.where(mz_valid, self.inputs, 0.0).astype(np.float32)

        self.norm_inputs = (self.inputs - ch_means) / np.maximum(ch_stds, 1e-6)
        self.norm_inputs = np.where(self.valid_masks, self.norm_inputs, 0.0).astype(np.float32)

        if int(in_channels) > 6:
            self.norm_inputs = np.concatenate(
                [self.norm_inputs, self.valid_masks.astype(np.float32)], axis=-1
            )

        aux_raw = np.stack(
            [
                self.charges.astype(np.float32),
                self.pepmasses.astype(np.float32),
                self.rtinseconds.astype(np.float32),
            ],
            axis=1,
        )
        self.norm_aux = ((aux_raw - aux_m) / np.maximum(aux_s, 1e-6)).astype(np.float32)
        self.n_samples = int(self.inputs.shape[0])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seq = "" if self.seqs is None else str(self.seqs[idx])
        return {
            "input": self.norm_inputs[idx],
            "valid_mask": self.valid_masks[idx],
            "aux_features": self.norm_aux[idx],
            "charge": int(self.charges[idx]),
            "pepmass": float(self.pepmasses[idx]),
            "rtinseconds": float(self.rtinseconds[idx]),
            "seq": seq,
            "source_idx": int(idx),
            "raw_input_mz": self.inputs[idx],
        }


def collate_prediction_batch(batch: list[dict]) -> dict:
    return {
        "input": torch.from_numpy(np.stack([b["input"] for b in batch])),
        "valid_mask": torch.from_numpy(np.stack([b["valid_mask"] for b in batch])),
        "aux_features": torch.from_numpy(np.stack([b["aux_features"] for b in batch])),
        "charge": torch.tensor([b["charge"] for b in batch], dtype=torch.long),
        "pepmass": torch.tensor([b["pepmass"] for b in batch], dtype=torch.float32),
        "rtinseconds": torch.tensor([b["rtinseconds"] for b in batch], dtype=torch.float32),
        "seq": [b["seq"] for b in batch],
        "source_idx": [b["source_idx"] for b in batch],
        "raw_input_mz": torch.from_numpy(np.stack([b["raw_input_mz"] for b in batch])),
    }


class XLPredictionDataset(Dataset):
    """Read paired inference fields while keeping structured NPY inputs memory-mapped."""

    _FIELD_SHAPES = {
        "sequence": (2,),
        "lengths": (2,),
        "link_sites": (2,),
        "precursor_charge": (),
        "precursor_mz": (),
        "rt_seconds": (),
    }
    _OPTIONAL_FIELDS = ("group_id", "crosslinker", "linker_mass")

    def __init__(self, path, split, metadata):
        self.path = Path(path)
        if self.path.is_dir():
            self.path = self.path / f"{split}.npy"
        self._columns = self._read_columns()
        required = (*self._FIELD_SHAPES, "input_theo_mz")
        missing = [name for name in required if name not in self._columns]
        if missing:
            raise ValueError(f"XL input {self.path} is missing fields: {', '.join(missing)}")

        grid = self._columns["input_theo_mz"]
        base_shape = (2, MAX_CLEAVAGE_LEN, len(CHANNEL_NAMES))
        expanded_shape = (2, MAX_CLEAVAGE_LEN, NUM_TOKENS, len(CHANNEL_NAMES))
        schema_by_shape = {base_shape: 1, expanded_shape: 2}
        schema_version = schema_by_shape.get(grid.shape[1:])
        if schema_version is None or grid.dtype.kind != "f":
            raise ValueError(f"XL input_theo_mz must have floating-point grids shaped "
                             f"{expanded_shape} or {base_shape}, got {grid.shape} with dtype {grid.dtype}")

        self.count = len(grid)
        optional = [name for name in self._OPTIONAL_FIELDS if name in self._columns]
        field_shapes = {**self._FIELD_SHAPES, **dict.fromkeys(optional, ())}
        for name, shape in field_shapes.items():
            if self._columns[name].shape != (self.count, *shape):
                raise ValueError(f"XL input field {name!r} must have shape {(self.count, *shape)}")
        for name in ("lengths", "link_sites", "precursor_charge"):
            if self._columns[name].dtype.kind not in "iu":
                raise ValueError(f"XL input field {name!r} must contain integers")
        if self._columns["sequence"].dtype.kind not in "SU":
            raise ValueError("XL input sequence must contain byte or Unicode strings")

        self.fields = (*required, *optional)
        self.input_collator = XLInputCollator(metadata, schema_version)

    def _read_columns(self):
        if self.path.suffix.lower() == ".npz":
            with np.load(self.path, allow_pickle=False) as archive:
                if "records" not in archive:
                    fields = (*self._FIELD_SHAPES, "input_theo_mz", *self._OPTIONAL_FIELDS)
                    return {name: archive[name] for name in fields if name in archive}
                records = archive["records"]
        elif self.path.suffix.lower() == ".npy":
            records = np.load(self.path, mmap_mode="r", allow_pickle=False)
        else:
            raise ValueError(f"Expected an XL dataset directory, NPY or NPZ file: {self.path}")
        if records.ndim != 1 or records.dtype.names is None:
            raise ValueError(f"Expected a one-dimensional XL structured array in {self.path}")
        return {name: records[name] for name in records.dtype.names}

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.path.suffix.lower() == ".npy":
            state["_columns"] = None
        return state

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self._columns is None:
            self._columns = self._read_columns()
        return {name: self._columns[name][index] for name in self.fields}

    def collate_fn(self, rows):
        batch = self.input_collator(rows)
        metadata = [{name: row[name] for name in self.fields if name != "input_theo_mz"} for row in rows]
        return batch, metadata


def merge_overlapping_peaks(candidates: list[dict], merge_tol: float = 0.05, mz_mode: str = "max") -> list[dict]:
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: x["mz"])
    merged = []
    i = 0
    n = len(candidates)

    while i < n:
        anchor_mz = candidates[i]["mz"]
        cluster = [candidates[i]]
        j = i + 1
        while j < n and candidates[j]["mz"] <= anchor_mz + merge_tol:
            cluster.append(candidates[j])
            j += 1

        intensities = np.array([x["intensity"] for x in cluster], dtype=np.float64)
        export_scores = np.array(
            [x.get("export_score", x["intensity"]) for x in cluster], dtype=np.float64
        )
        mzs = np.array([x["mz"] for x in cluster], dtype=np.float64)
        avg_intensity = float(intensities.mean())
        best_idx = int(np.argmax(intensities))
        best_score_idx = int(np.argmax(export_scores))

        if mz_mode == "max":
            final_mz = float(mzs[best_idx])
            final_intensity = float(intensities[best_idx])
        elif mz_mode == "anchor":
            final_mz = float(anchor_mz)
            final_intensity = avg_intensity
        elif mz_mode == "weighted":
            final_mz = float((mzs * intensities).sum() / (intensities.sum() + 1e-12))
            final_intensity = avg_intensity
        else:
            final_mz = float(mzs.mean())
            final_intensity = avg_intensity

        merged.append(
            {
                "mz": final_mz,
                "intensity": final_intensity,
                "export_score": float(export_scores[best_score_idx]),
                "num_slots": len(cluster),
            }
        )
        i = j

    return merged


def apply_postmerge_expression_calibration(
    peaks: list[dict],
    charge: int,
    peptide_length: int,
    scene: str,
    scene_params: dict | None = None,
) -> list[dict]:
    if scene != "v20_pp4" or not peaks:
        return peaks

    subgroup = None
    if charge == 2 and peptide_length <= 16:
        subgroup = "z2_short"
    elif charge == 2:
        subgroup = "z2_long"
    elif charge == 3 and peptide_length <= 16:
        subgroup = "z3_short"
    elif charge == 3:
        subgroup = "z3_long"

    params = dict(scene_params or {})
    subgroup_overrides = params.get("pp4_subgroup_overrides")
    if isinstance(subgroup_overrides, dict) and subgroup in subgroup_overrides and isinstance(subgroup_overrides[subgroup], dict):
        params.update(subgroup_overrides[subgroup])

    score_alpha = float(params.get("pp4_score_alpha", 0.20))
    rank_alpha = float(params.get("pp4_rank_alpha", 0.08))
    slot_alpha = float(params.get("pp4_slot_alpha", 0.06))
    tail_alpha = float(params.get("pp4_tail_alpha", 0.16))
    tail_gamma = float(params.get("pp4_tail_gamma", 1.20))
    floor = float(params.get("pp4_floor", 0.82))
    cap = float(params.get("pp4_cap", 1.16))
    energy_preserve = bool(params.get("pp4_energy_preserve", True))

    ordered = sorted(peaks, key=lambda p: float(p.get("export_score", p.get("intensity", 0.0))), reverse=True)
    scores = np.array([float(p.get("export_score", p.get("intensity", 0.0))) for p in ordered], dtype=np.float64)
    intensities = np.array([float(p.get("intensity", 0.0)) for p in ordered], dtype=np.float64)
    slot_counts = np.array([float(p.get("num_slots", 1.0)) for p in ordered], dtype=np.float64)

    if len(ordered) == 1:
        ranks = np.array([0.0], dtype=np.float64)
    else:
        ranks = np.linspace(0.0, 1.0, len(ordered), dtype=np.float64)
    score_ref = float(np.quantile(scores, 0.80)) if len(scores) > 0 else 0.0
    if score_ref <= 1e-8:
        score_ref = float(scores.max()) if len(scores) > 0 else 1.0
    score_term = (scores / max(score_ref, 1e-8)) - 1.0
    rank_term = 0.5 - ranks
    slot_term = np.log1p(np.maximum(slot_counts - 1.0, 0.0))
    tail_term = np.power(np.clip(ranks, 0.0, 1.0), tail_gamma)

    gain = 1.0 + score_alpha * score_term + rank_alpha * rank_term + slot_alpha * slot_term - tail_alpha * tail_term
    gain = np.clip(gain, floor, cap)
    calibrated = intensities * gain
    if energy_preserve:
        src_sum = float(intensities.sum())
        dst_sum = float(calibrated.sum())
        if src_sum > 1e-8 and dst_sum > 1e-8:
            calibrated *= src_sum / dst_sum

    result = []
    for peak, new_intensity, g in zip(ordered, calibrated, gain):
        p = dict(peak)
        p["intensity"] = float(new_intensity)
        p["pp4_gain"] = float(g)
        p["pp4_subgroup"] = subgroup or "other"
        result.append(p)
    result.sort(key=lambda p: float(p["mz"]))
    return result


def write_mgf_block(
    fout,
    seq: str,
    charge: int,
    pepmass: float,
    rtinseconds: float,
    peaks: list[dict],
    title: str,
    source_index: int,
    extra_comment: str | None = None,
    extra_headers: dict | None = None,
) -> None:
    fout.write("BEGIN IONS\n")
    fout.write("SOURCE=MODEL_BATCH_PREDICTION\n")
    fout.write(f"TITLE={title}\n")
    fout.write(f"PEPMASS={float(pepmass):.6f}\n")
    fout.write(f"RTINSECONDS={float(rtinseconds):.6f}\n")
    fout.write(f"CHARGE={int(charge)}+\n")
    fout.write(f"SCANS={source_index}\n")
    fout.write(f"SEQ={seq}\n")
    for key, value in (extra_headers or {}).items():
        fout.write(f"{key}={value}\n")

    comment = f"PREDICTED_SPECTRUM source_index={source_index} peaks={len(peaks)}"
    if extra_comment is not None:
        comment += " " + extra_comment
    fout.write(f"COMMENT={comment}\n")

    for peak in sorted(peaks, key=lambda x: x["mz"]):
        fout.write(f"{peak['mz']:.6f} {peak['intensity']:.6f}\n")

    fout.write("END IONS\n\n")


def _xl_spectrum_peaks(prediction, mz, valid, sequences, lengths, link_sites, charge, *,
                       postprocess_mode, raw_min_intensity, merge_tol, scene, params, n_true_mean):
    if postprocess_mode == "none":
        selected = valid & (prediction > raw_min_intensity)
        cells = [{"mz": mass, "intensity": intensity}
                 for mass, intensity in zip(mz[selected], prediction[selected])]
        return merge_overlapping_peaks(cells, merge_tol), len(cells), []

    residues, nterm_shifts = zip(*(parse_sequence_with_mods(seq) for seq in sequences))
    masked_prediction = np.where(valid, prediction, 0.0)
    chain_peaks = []
    diagnostics = []
    candidate_slots = 0
    for chain, chain_name in enumerate(("A", "B")):
        cells, diagnostic = run_cluster_postprocess(
            pred_matrix=masked_prediction[chain],
            input_mz=mz[chain],
            peptide_length=lengths[chain],
            precursor_charge=charge,
            scene=scene,
            params=params,
            n_true_mean=n_true_mean,
            residues=residues[chain],
            nterm_shift=nterm_shifts[chain],
            link_site=link_sites[chain],
            partner_residues=residues[1 - chain],
        )
        peaks = merge_overlapping_peaks(cells, merge_tol)
        chain_peaks.extend(apply_postmerge_expression_calibration(peaks, charge, lengths[chain], scene, params))
        candidate_slots += len(cells)
        diagnostics.append({**diagnostic, "chain": chain_name, "seq": sequences[chain],
                            "link_site": link_sites[chain], "chain_peaks": len(peaks)})
    return merge_overlapping_peaks(chain_peaks, merge_tol), candidate_slots, diagnostics


def _xl_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@contextmanager
def _staged_text_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        staging = Path(stream.name)
        try:
            yield stream
            stream.close()
            staging.replace(path)
        finally:
            stream.close()
            staging.unlink(missing_ok=True)


def _export_xl(model, checkpoint, checkpoint_metadata, npz_path, output_mgf, *, split, output_summary_txt,
               batch_size, num_workers, device, max_examples, min_mz, merge_tol, raw_min_intensity,
               postprocess_mode, scene, params, n_true_mean, cluster_diag_dir):
    if max_examples is not None and max_examples < 0:
        raise ValueError("--max_examples must be nonnegative")
    dataset = XLPredictionDataset(npz_path, split, checkpoint_metadata)
    count = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)), batch_size=batch_size, num_workers=num_workers,
        collate_fn=dataset.collate_fn, pin_memory=device.type == "cuda",
    )
    print(f"XL dataset: {dataset.path} ({count} of {len(dataset)} peptide pairs)")

    output_mgf = Path(output_mgf)
    output_summary_txt = Path(output_summary_txt) if output_summary_txt is not None else None
    diag_file = None
    if cluster_diag_dir and postprocess_mode == "cluster":
        diag_file = Path(cluster_diag_dir) / "cluster_diagnostics.json"
    outputs = [path for path in (output_mgf, output_summary_txt, diag_file) if path is not None]
    resolved_outputs = {path.resolve() for path in outputs}
    inputs = {dataset.path.resolve(), Path(checkpoint).resolve()}
    if len(resolved_outputs) != len(outputs) or resolved_outputs & inputs:
        raise ValueError("XL output paths must be distinct from each other, the input file and the checkpoint")

    all_diagnostics = []
    source_idx = total_peaks = 0
    with ExitStack() as files:
        fout = files.enter_context(_staged_text_output(output_mgf))
        summary = None
        if output_summary_txt is not None:
            summary = files.enter_context(_staged_text_output(output_summary_txt))
            summary.write(f"# postprocess_mode={postprocess_mode} cluster_scene={scene} "
                          f"merge_tol={merge_tol} intensity_space=sqrt_clip\n")
            summary.write("source_index\tgroup_id\tsequence_a\tsequence_b\tlink_site_a\tlink_site_b\t"
                          "charge\tpepmass\trtinseconds\tcandidate_slots\tfinal_peaks\n")

        for batch, rows in tqdm(loader, desc="Export XL spectra", leave=False):
            model_batch = {name: tensor.to(device, non_blocking=True)
                           for name, tensor in batch.items() if name != "input_theo_mz"}
            predictions = predict_xl_batch(model, model_batch)["prediction"].cpu().numpy()
            raw_mz = batch["input_theo_mz"].numpy()
            valid = batch["valid_mask"].numpy() & (raw_mz >= min_mz)
            for prediction, mz, mask, row in zip(predictions, raw_mz, valid, rows):
                sequences = tuple(_xl_text(seq) for seq in row["sequence"])
                lengths = tuple(map(int, row["lengths"]))
                link_sites = tuple(map(int, row["link_sites"]))
                charge = int(row["precursor_charge"])
                pepmass = float(row["precursor_mz"])
                rt = float(row["rt_seconds"])
                group_id = _xl_text(row.get("group_id", ""))
                peaks, slots, diagnostics = _xl_spectrum_peaks(
                    prediction, mz, mask, sequences, lengths, link_sites, charge,
                    postprocess_mode=postprocess_mode, raw_min_intensity=raw_min_intensity,
                    merge_tol=merge_tol, scene=scene, params=params, n_true_mean=n_true_mean,
                )
                headers = {"LINK_SITES": f"{link_sites[0]},{link_sites[1]}"}
                if group_id:
                    headers["XL_GROUP_ID"] = group_id
                if "crosslinker" in row:
                    headers["CROSSLINKER"] = _xl_text(row["crosslinker"])
                if "linker_mass" in row:
                    headers["LINKER_MASS"] = f"{float(row['linker_mass']):.6f}"
                write_mgf_block(
                    fout, "--".join(sequences), charge, pepmass, rt, peaks,
                    f"XL_MODEL_PREDICTED_{group_id or source_idx}", source_idx,
                    extra_comment=f"intensity_space=sqrt_clip candidate_slots={slots}", extra_headers=headers,
                )
                if summary is not None:
                    summary.write(f"{source_idx}\t{group_id}\t{sequences[0]}\t{sequences[1]}\t"
                                  f"{link_sites[0]}\t{link_sites[1]}\t{charge}\t{pepmass:.6f}\t"
                                  f"{rt:.6f}\t{slots}\t{len(peaks)}\n")
                if diag_file is not None:
                    all_diagnostics.extend({**diag, "source_idx": source_idx, "group_id": group_id,
                                            "charge": charge} for diag in diagnostics)
                source_idx += 1
                total_peaks += len(peaks)

        if diag_file is not None:
            diagnostics_out = files.enter_context(_staged_text_output(diag_file))
            json.dump(all_diagnostics, diagnostics_out, indent=2, default=str)
    print(f"Exported {source_idx} XL spectra and {total_peaks} peaks to {output_mgf}")


@torch.no_grad()
def predict_npz_to_mgf(
    npz_path,
    checkpoint,
    output_mgf,
    output_summary_txt=None,
    batch_size: int | None = None,
    num_workers: int = 0,
    min_mz: float = 100.0,
    merge_tol: float = 0.05,
    raw_min_intensity: float = 0.01,
    mz_mode: str = "max",
    max_examples: int | None = None,
    device: str | None = None,
    postprocess_mode: str = "cluster",
    cluster_scene: str = "v20_pp4",
    cluster_t_main: float | None = None,
    cluster_t_sat: float | None = None,
    cluster_R_target: float | None = None,
    cluster_channel_alpha: float | None = None,
    cluster_isolated_sat_mode: str | None = None,
    cluster_sat_main_ratio: float | None = None,
    cluster_peak_cap_ratio: float | None = None,
    cluster_max_sat_per_cluster: int | None = None,
    cluster_max_tokens_per_cluster: int | None = None,
    cluster_dense_guard: bool | None = None,
    cluster_state_guard: bool | None = None,
    cluster_token_profile_path: str | None = None,
    cluster_export_tokens=None,
    cluster_n_true_mean_path: str | None = None,
    cluster_diag_dir: str | None = None,
    family_aware_postprocess_prior: bool = True,
    split: str = "test",
) -> None:
    postprocess_mode = str(postprocess_mode or "cluster").lower()
    if postprocess_mode not in ("cluster", "none"):
        raise ValueError("postprocess_mode must be cluster|none")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("Device:", device)

    cluster_scene = str(cluster_scene or "search")
    if cluster_scene not in SCENE_TEMPLATES:
        raise ValueError(f"--cluster_scene must be one of: {list(SCENE_TEMPLATES.keys())}")

    cluster_params_overrides = {}
    if cluster_t_main is not None:
        cluster_params_overrides["t_main"] = float(cluster_t_main)
    if cluster_t_sat is not None:
        cluster_params_overrides["t_sat"] = float(cluster_t_sat)
    if cluster_R_target is not None:
        cluster_params_overrides["R_target"] = float(cluster_R_target)
    if cluster_channel_alpha is not None:
        cluster_params_overrides["channel_alpha"] = float(cluster_channel_alpha)
    if cluster_isolated_sat_mode is not None:
        cluster_params_overrides["isolated_sat_mode"] = str(cluster_isolated_sat_mode)
    if cluster_sat_main_ratio is not None:
        cluster_params_overrides["sat_main_ratio"] = float(cluster_sat_main_ratio)
    if cluster_peak_cap_ratio is not None:
        cluster_params_overrides["peak_cap_ratio"] = float(cluster_peak_cap_ratio)
    if cluster_max_sat_per_cluster is not None:
        cluster_params_overrides["max_sat_per_cluster"] = int(cluster_max_sat_per_cluster)
    if cluster_max_tokens_per_cluster is not None:
        cluster_params_overrides["max_tokens_per_cluster"] = int(cluster_max_tokens_per_cluster)
    if cluster_dense_guard is not None:
        cluster_params_overrides["dense_guard"] = bool(cluster_dense_guard)
    if cluster_state_guard is not None:
        cluster_params_overrides["state_guard"] = bool(cluster_state_guard)
    cluster_params_overrides["export_score_mode"] = (
        "family_rerank" if bool(family_aware_postprocess_prior) else "baseline"
    )

    parsed_cluster_export_tokens = parse_cluster_export_tokens(cluster_export_tokens)
    if parsed_cluster_export_tokens is not None:
        cluster_params_overrides["export_tokens"] = parsed_cluster_export_tokens
    if cluster_token_profile_path:
        cluster_params_overrides["token_profile_prior"] = load_token_profile_prior(
            cluster_token_profile_path
        )
        cluster_params_overrides["token_profile_path"] = str(cluster_token_profile_path)

    cluster_params = resolve_scene_params(cluster_scene, cluster_params_overrides)
    n_true_mean = load_n_true_mean_from_stats(cluster_n_true_mean_path)
    postprocess_score_source = "final_prediction"

    if postprocess_mode == "none":
        print(f"Postprocess mode: none (raw grid -> merged peaks, intensity floor={raw_min_intensity})")
        print(f"Postprocess path: final prediction -> grid cells > {raw_min_intensity} -> merged MGF peaks")
        print(f"  merge_tol: {merge_tol}   mz_mode: {mz_mode}   min_mz: {min_mz}")
    else:
        print("Postprocess mode: cluster")
        print("Postprocess path: final prediction -> cluster postprocess -> merged MGF peaks")
        print(f"  scene: {cluster_scene} ({SCENE_TEMPLATES[cluster_scene]['desc']})")
        print(f"  t_main: {cluster_params['t_main']:.4f}")
        print(f"  t_sat: {cluster_params['t_sat']:.4f}")
        print(f"  R_target: {cluster_params['R_target']:.2f}")
        print(f"  channel_alpha: {cluster_params['channel_alpha']:.2f}")
        print(f"  isolated_sat_mode: {cluster_params['isolated_sat_mode']}")
        print(f"  sat_main_ratio: {cluster_params.get('sat_main_ratio', 0.15):.3f}")
        print(f"  peak_cap_ratio: {cluster_params.get('peak_cap_ratio', -1):.2f}")
        if cluster_params.get("max_sat_per_cluster") is not None:
            print(f"  max_sat_per_cluster: {int(cluster_params['max_sat_per_cluster'])}")
        if cluster_params.get("max_tokens_per_cluster") is not None:
            print(f"  max_tokens_per_cluster: {int(cluster_params['max_tokens_per_cluster'])}")
        print(f"  dense_guard: {'on' if bool(cluster_params.get('dense_guard', False)) else 'off'}")
        print(f"  state_guard: {'on' if bool(cluster_params.get('state_guard', False)) else 'off'}")
        print(f"  export_score_mode: {cluster_params.get('export_score_mode', 'family_rerank')}")
        if cluster_params.get("token_profile_path") is not None:
            print(f"  token_profile: {cluster_params['token_profile_path']}")
        if cluster_params.get("export_tokens") is not None:
            tokens_text = ",".join(str(int(t)) for t in cluster_params["export_tokens"])
            print(f"  export_tokens: {tokens_text}")
        print(f"  postprocess_score_source: {postprocess_score_source}")

    model, ckpt, model_config = load_model(checkpoint, device=device)
    if isinstance(model, XLSpectrumTransformer):
        if mz_mode != "max":
            raise ValueError("XL export requires --mz_mode max")
        _export_xl(
            model, checkpoint, ckpt["data_metadata"], npz_path, output_mgf, split=split,
            output_summary_txt=output_summary_txt, batch_size=4 if batch_size is None else batch_size,
            num_workers=num_workers, device=device, max_examples=max_examples, min_mz=min_mz,
            merge_tol=merge_tol, raw_min_intensity=raw_min_intensity, postprocess_mode=postprocess_mode,
            scene=cluster_scene, params=cluster_params, n_true_mean=n_true_mean, cluster_diag_dir=cluster_diag_dir,
        )
        return

    channel_means = ckpt["channel_means"]
    channel_stds = ckpt["channel_stds"]
    aux_means = ckpt["aux_feature_means"]
    aux_stds = ckpt["aux_feature_stds"]
    in_channels = int(model_config.get("in_channels", 6))
    use_sequence_context = bool(model_config.get("use_sequence_context", False))
    print(f"Model in_channels: {in_channels}")
    print(f"cleavage_context_residual: {'on' if use_sequence_context else 'off'}")

    dataset = NPZPredictionDataset(
        npz_path=npz_path,
        channel_means=channel_means,
        channel_stds=channel_stds,
        aux_means=aux_means,
        aux_stds=aux_stds,
        in_channels=in_channels,
        min_mz=min_mz,
    )
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=64 if batch_size is None else batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_prediction_batch,
        drop_last=False,
    )

    encode_cleavage_context = None
    if use_sequence_context:
        try:
            from spectrum_transformer.data.dataset import encode_cleavage_context as _enc

            encode_cleavage_context = _enc
            print("Sequence context encoding: enabled")
        except Exception as exc:
            print(f"WARNING: failed to import sequence-context encoder: {exc}")
            print("  Sequence context will be skipped.")

    output_mgf = Path(output_mgf)
    output_mgf.parent.mkdir(parents=True, exist_ok=True)
    if output_summary_txt is not None:
        output_summary_txt = Path(output_summary_txt)
        output_summary_txt.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_candidate_slots = 0
    total_merged_peaks = 0
    per_spectrum_peaks: list[int] = []
    theory_ratios: list[float] = []
    by_charge_peaks: dict[int, list[int]] = {}
    by_length_peaks: dict[int, list[int]] = {}
    split_counts = {
        "main": 0,
        "loss": 0,
        "isotope": 0,
        "combined": 0,
        "rej_orphan": 0,
        "rej_ratio": 0,
        "rej_main_thresh": 0,
        "capped": 0,
        "capped_main": 0,
        "capped_isotope": 0,
        "capped_loss": 0,
        "capped_combined": 0,
        "rescued": 0,
        "rescued_sat": 0,
    }
    cluster_all_diags = [] if cluster_diag_dir else None

    with open(output_mgf, "w", encoding="utf-8") as fout:
        if output_summary_txt is not None:
            summary_f = open(output_summary_txt, "w", encoding="utf-8")
            summary_f.write("Batch prediction summary\n")
            summary_f.write("=" * 100 + "\n")
            summary_f.write(f"NPZ: {npz_path}\n")
            summary_f.write(f"Checkpoint: {checkpoint}\n")
            summary_f.write(f"Output MGF: {output_mgf}\n")
            summary_f.write(
                f"cleavage_context_residual: {'on' if use_sequence_context else 'off'}\n"
            )
            summary_f.write(f"model_version: {model_config.get('model_version', 'unknown')}\n")
            summary_f.write(f"postprocess_mode: {postprocess_mode}\n")
            summary_f.write(f"postprocess_score_source: {postprocess_score_source}\n")
            summary_f.write(f"min_mz: {min_mz}\n")
            summary_f.write(f"merge_tol: {merge_tol}\n")
            summary_f.write(f"raw_min_intensity: {raw_min_intensity}\n")
            summary_f.write(f"mz_mode: {mz_mode}\n")
            summary_f.write("\n")
            summary_f.write(
                "source_index\tseq\tcharge\tpepmass\trtinseconds\t"
                "candidate_slots\tmerged_peaks\tfinal_peaks\tadaptive_min\t"
                "adaptive_max\tresidues\tadaptive_max_per_residue\t"
                "sparse_seed_peaks\tsparse_guard_max\tsweet_min\tsweet_max\t"
                "sweet_seed_state\tadaptive_action\texpand_ratio\tstrict_count\t"
                "scan_curve\n"
            )
        else:
            summary_f = None

        batch_iter = tqdm(
            loader,
            desc="predict_npz_to_mgf",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_id, batch in enumerate(batch_iter, start=1):
            x = batch["input"].to(device, non_blocking=True)
            aux = batch["aux_features"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)

            forward_kwargs = {
                "aux_features": aux,
                "valid_mask": valid_mask,
                "charge": batch["charge"].to(device, non_blocking=True),
            }

            if use_sequence_context and encode_cleavage_context is not None:
                cleavage_aa_list = []
                cleavage_scalar_list = []
                for seq in batch["seq"]:
                    residue_count = max(count_peptide_residues(seq), 1)
                    aa_ids, scalar = encode_cleavage_context(
                        seq,
                        residue_count,
                        max_cleavage_len=39,
                    )
                    cleavage_aa_list.append(aa_ids.astype(np.int64, copy=False))
                    cleavage_scalar_list.append(scalar.astype(np.float32, copy=False))
                forward_kwargs["cleavage_aa_ids"] = (
                    torch.from_numpy(np.stack(cleavage_aa_list)).to(device)
                )
                forward_kwargs["cleavage_scalar_features"] = (
                    torch.from_numpy(np.stack(cleavage_scalar_list)).to(device)
                )

            out = model(x, **forward_kwargs)
            if "prediction" not in out:
                raise KeyError("Model output is missing 'prediction'; V20 final path requires it.")
            pred_np = out["prediction"].detach().cpu().numpy()

            for local_i, source_idx in enumerate(batch["source_idx"]):
                if max_examples is not None and total_examples >= max_examples:
                    break

                source_idx = int(source_idx)
                seq = batch["seq"][local_i]
                charge_value = int(batch["charge"][local_i])
                pepmass_value = float(batch["pepmass"][local_i])
                rt_value = float(batch["rtinseconds"][local_i])
                raw_input_mz = batch["raw_input_mz"][local_i].numpy()
                peptide_length = max(count_peptide_residues(seq), 1)

                residues = None
                nterm_shift = 0.0
                if seq:
                    residues, nterm_shift = parse_sequence_with_mods(seq)

                if postprocess_mode == "none":
                    _arr = prediction_to_mgf_peaks(
                        raw_input_mz, pred_np[local_i],
                        batch["valid_mask"][local_i].numpy(),
                        raw_min_intensity, min_mz, merge_tol,
                    )
                    cluster_cells = [{"mz": float(a), "intensity": float(b)} for a, b in _arr]
                    cluster_diag = {}
                    merged_peaks = merge_overlapping_peaks(
                        candidates=cluster_cells, merge_tol=merge_tol, mz_mode=mz_mode)
                    base_candidate_slots = len(cluster_cells)
                    base_merged_peaks = len(merged_peaks)
                    adaptive_min_bound = adaptive_max_bound = 0
                    effective_max_per_residue = 0.0
                    sparse_seed_peaks = sparse_guard_max = -1
                    sweet_min_bound = sweet_max_bound = -1
                    sweet_seed_state = "na"
                    expand_ratio = 1.0
                    strict_count = 0
                    adaptive_action = "none:raw-grid"
                    theory_ratios.append(0.0)
                else:
                    cluster_cells, cluster_diag = run_cluster_postprocess(
                    pred_matrix=pred_np[local_i].copy(),
                    input_mz=raw_input_mz,
                    peptide_length=peptide_length,
                    precursor_charge=charge_value,
                    scene=cluster_scene,
                    params=cluster_params,
                    n_true_mean=n_true_mean,
                    return_diagnostics=True,
                    residues=residues,
                    nterm_shift=nterm_shift,
                )
                merged_peaks = merge_overlapping_peaks(
                    candidates=cluster_cells,
                    merge_tol=merge_tol,
                    mz_mode=mz_mode,
                )
                if postprocess_mode != "none":   # raw mode: no expression calibration
                    merged_peaks = apply_postmerge_expression_calibration(
                        peaks=merged_peaks,
                        charge=charge_value,
                        peptide_length=peptide_length,
                        scene=cluster_scene,
                        scene_params=cluster_params,
                    )
                base_candidate_slots = len(cluster_cells)
                base_merged_peaks = len(merged_peaks)
                adaptive_min_bound = int(cluster_diag.get("K_target", 0))
                adaptive_max_bound = int(cluster_diag.get("K_target", 0))
                effective_max_per_residue = 0.0
                sparse_seed_peaks = -1
                sparse_guard_max = -1
                sweet_min_bound = -1
                sweet_max_bound = -1
                sweet_seed_state = "na"
                expand_ratio = 1.0
                strict_count = int(cluster_diag.get("n_clusters_selected", 0))
                if postprocess_mode != "none":   # raw 分支保留 "none:raw-grid" 标签
                    adaptive_action = (
                        f"cluster:{cluster_scene}/"
                        f"K={cluster_diag.get('K_target', '?')}/"
                        f"clusters={cluster_diag.get('n_clusters_selected', '?')}/"
                        f"cells={cluster_diag.get('n_cells_exported', '?')}/"
                        f"sat_removed={cluster_diag.get('n_isolated_sat_removed', '?')}"
                    )

                split_counts["main"] += int(cluster_diag.get("n_main_cells", 0))
                split_counts["loss"] += int(cluster_diag.get("n_loss_cells", 0))
                split_counts["isotope"] += int(cluster_diag.get("n_isotope_cells", 0))
                split_counts["combined"] += int(cluster_diag.get("n_combined_cells", 0))
                split_counts["rej_orphan"] += int(cluster_diag.get("n_rejected_sat_orphan", 0))
                split_counts["rej_ratio"] += int(cluster_diag.get("n_rejected_sat_ratio", 0))
                split_counts["rej_main_thresh"] += int(cluster_diag.get("n_rejected_main_thresh", 0))
                split_counts["capped"] += int(cluster_diag.get("n_cells_capped", 0))
                split_counts["capped_main"] += int(cluster_diag.get("n_capped_main", 0))
                split_counts["capped_isotope"] += int(cluster_diag.get("n_capped_isotope", 0))
                split_counts["capped_loss"] += int(cluster_diag.get("n_capped_loss", 0))
                split_counts["capped_combined"] += int(cluster_diag.get("n_capped_combined", 0))
                split_counts["rescued"] += int(cluster_diag.get("n_main_rescued", 0))
                split_counts["rescued_sat"] += int(cluster_diag.get("n_sat_rescued", 0))
                theory_ratios.append(float(cluster_diag.get("theory_ratio", 0.0)))

                if cluster_all_diags is not None:
                    cluster_diag["source_idx"] = source_idx
                    cluster_diag["seq"] = seq
                    cluster_diag["charge"] = charge_value
                    cluster_diag["group_filename"] = _make_group_filename(seq, charge_value)
                    cluster_diag["score_source"] = postprocess_score_source
                    cluster_all_diags.append(cluster_diag)

                title = f"MODEL_PREDICTED_INDEX_{source_idx}"
                write_mgf_block(
                    fout=fout,
                    seq=seq,
                    charge=charge_value,
                    pepmass=pepmass_value,
                    rtinseconds=rt_value,
                    peaks=merged_peaks,
                    title=title,
                    source_index=source_idx,
                    extra_comment=(
                        f"candidate_slots={base_candidate_slots} "
                        f"adaptive_action={adaptive_action}"
                    ),
                )

                if summary_f is not None:
                    summary_f.write(
                        f"{source_idx}\t"
                        f"{seq}\t"
                        f"{charge_value}\t"
                        f"{pepmass_value:.6f}\t"
                        f"{rt_value:.6f}\t"
                        f"{base_candidate_slots}\t"
                        f"{base_merged_peaks}\t"
                        f"{len(merged_peaks)}\t"
                        f"{adaptive_min_bound}\t"
                        f"{adaptive_max_bound}\t"
                        f"{peptide_length}\t"
                        f"{effective_max_per_residue:.2f}\t"
                        f"{sparse_seed_peaks}\t"
                        f"{sparse_guard_max}\t"
                        f"{sweet_min_bound}\t"
                        f"{sweet_max_bound}\t"
                        f"{sweet_seed_state}\t"
                        f"{adaptive_action}\t"
                        f"{expand_ratio:.2f}\t"
                        f"{strict_count}\t\n"
                    )

                total_examples += 1
                total_candidate_slots += base_candidate_slots
                total_merged_peaks += len(merged_peaks)
                per_spectrum_peaks.append(len(merged_peaks))
                by_charge_peaks.setdefault(charge_value, []).append(len(merged_peaks))
                by_length_peaks.setdefault(get_length_bin(peptide_length), []).append(len(merged_peaks))

            if max_examples is not None and total_examples >= max_examples:
                break

            batch_iter.set_postfix(exported=total_examples, merged=total_merged_peaks)
            if batch_id % 50 == 0:
                avg_pk = total_merged_peaks / max(total_examples, 1)
                print(
                    f"Processed batches={batch_id}, "
                    f"examples={total_examples}, "
                    f"merged_peaks={total_merged_peaks}, "
                    f"avg_peaks={avg_pk:.0f}"
                )
        batch_iter.close()

        if summary_f is not None:
            summary_f.write("\n")
            summary_f.write("=" * 100 + "\n")
            summary_f.write("Overall\n")
            summary_f.write(f"total_examples\t{total_examples}\n")
            summary_f.write(f"total_candidate_slots\t{total_candidate_slots}\n")
            summary_f.write(f"total_merged_peaks\t{total_merged_peaks}\n")
            if total_examples > 0:
                summary_f.write(
                    f"avg_candidate_slots\t{total_candidate_slots / total_examples:.4f}\n"
                )
                summary_f.write(
                    f"avg_merged_peaks\t{total_merged_peaks / total_examples:.4f}\n"
                )
            summary_f.close()

    if cluster_diag_dir and cluster_all_diags:
        diag_path = Path(cluster_diag_dir)
        diag_path.mkdir(parents=True, exist_ok=True)
        diag_file = diag_path / "cluster_diagnostics.json"
        with open(diag_file, "w", encoding="utf-8") as df:
            json.dump(cluster_all_diags, df, indent=2, default=str)
        print(f"\nCluster diagnostics saved to: {diag_file}")

        diag_summary = summarize_diagnostics(cluster_all_diags)
        if diag_summary:
            print(f"\nCluster Postprocessing Diagnostics ({len(cluster_all_diags)} spectra):")
            print(
                "  Avg nonzero before pipeline:     "
                f"{diag_summary.get('n_nonzero_before_position_mask_mean', 0):.1f}"
            )
            print(
                "  Avg nonzero after position mask: "
                f"{diag_summary.get('n_nonzero_after_position_mask_mean', 0):.1f}"
            )
            print(
                "  Avg nonzero after charge mask:   "
                f"{diag_summary.get('n_nonzero_after_charge_mask_mean', 0):.1f}"
            )
            print(
                "  Avg nonzero after dual thresh:   "
                f"{diag_summary.get('n_nonzero_after_dual_threshold_mean', 0):.1f}"
            )
            print(
                "  Avg charge-invalid removed:      "
                f"{diag_summary.get('n_charge_invalid_removed_mean', 0):.1f}"
            )
            print(
                "  Avg clusters active:             "
                f"{diag_summary.get('n_clusters_active_mean', 0):.1f}"
            )
            print(
                "  Avg weak main peaks:             "
                f"{diag_summary.get('n_weak_main_mean', 0):.1f}"
            )
            print(
                "  Avg isolated sat removed:        "
                f"{diag_summary.get('n_isolated_sat_removed_mean', 0):.1f}"
            )
            print(
                "  Avg K target:                    "
                f"{diag_summary.get('K_target_mean', 0):.1f}"
            )
            print(
                "  Avg clusters selected:           "
                f"{diag_summary.get('n_clusters_selected_mean', 0):.1f}"
            )
            print(
                "  Avg cells exported:              "
                f"{diag_summary.get('n_cells_exported_mean', 0):.1f}"
            )

    print("\nExport Summary")
    print(f"  Spectra written:        {total_examples}")
    print(f"  Total candidate slots:  {total_candidate_slots}")
    print(f"  Total merged peaks:     {total_merged_peaks}")
    if total_examples > 0:
        print(f"  Avg candidate slots:    {total_candidate_slots / total_examples:.2f}")
        print(f"  Avg merged peaks:       {total_merged_peaks / total_examples:.2f}")
    if per_spectrum_peaks:
        arr = np.asarray(per_spectrum_peaks, dtype=np.float64)
        print(
            "  Peak count percentiles: "
            f"p10={np.percentile(arr, 10):.0f} "
            f"p50={np.percentile(arr, 50):.0f} "
            f"p90={np.percentile(arr, 90):.0f}"
        )
    if theory_ratios:
        arr = np.asarray(theory_ratios, dtype=np.float64)
        print(
            "  Theory ratio:          "
            f"mean={arr.mean():.3f} median={np.median(arr):.3f}"
        )
    print(
        "  Split counts:          "
        f"main={split_counts['main']} "
        f"loss={split_counts['loss']} "
        f"isotope={split_counts['isotope']} "
        f"combined={split_counts['combined']}"
    )

    if by_charge_peaks:
        print("  Peaks by charge:")
        for charge_value in sorted(by_charge_peaks):
            arr = np.asarray(by_charge_peaks[charge_value], dtype=np.float64)
            print(
                f"    z={charge_value}: n={len(arr)} "
                f"avg={arr.mean():.0f} p50={np.median(arr):.0f} "
                f"p95={np.percentile(arr, 95):.0f}"
            )
    if by_length_peaks:
        labels = {0: "L<=8", 1: "9-15", 2: "16-25", 3: "L>25"}
        print("  Peaks by length:")
        for length_bin in sorted(by_length_peaks):
            arr = np.asarray(by_length_peaks[length_bin], dtype=np.float64)
            label = labels.get(length_bin, f"bin{length_bin}")
            print(
                f"    {label}: n={len(arr)} "
                f"avg={arr.mean():.0f} p50={np.median(arr):.0f} "
                f"p95={np.percentile(arr, 95):.0f}"
            )

    if output_summary_txt is not None:
        print(f"Summary TXT: {output_summary_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export XL or V20 transformer predictions to MGF."
    )
    parser.add_argument("--npz", "--input", required=True,
                        help="Input NPZ, XL split NPY, or XL dataset directory")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Split to read when the input is an XL dataset directory (default: test)")
    parser.add_argument("--checkpoint", required=True, help="Path to an XL or V20 transformer checkpoint")
    parser.add_argument("--output_mgf", required=True, help="Output MGF file path")
    parser.add_argument("--output_summary_txt", default=None, help="Optional summary text file")
    parser.add_argument("--batch_size", type=int, help="Batch size (default: 4 for XL, 64 for V20)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_mz", type=float, default=100.0, help="Minimum m/z for exported peaks")
    parser.add_argument("--raw_min_intensity", type=float, default=0.01,
                        help="grid->peak intensity floor for --postprocess_mode none")
    parser.add_argument("--merge_tol", type=float, default=0.05, help="m/z merge tolerance (Da)")
    parser.add_argument(
        "--postprocess_mode",
        choices=["cluster", "none"],
        default="cluster",
        help="cluster applies PP4. none exports merged raw model predictions.",
    )
    parser.add_argument(
        "--mz_mode",
        choices=["max", "mean", "anchor", "weighted"],
        default="max",
        help="Peak merge mode (XL uses max)",
    )
    parser.add_argument("--max_examples", type=int, default=None, help="Optional example limit")
    parser.add_argument("--device", default=None, help="Device to use, e.g. cuda:0 or cpu")

    parser.add_argument(
        "--cluster_scene",
        choices=list(SCENE_TEMPLATES.keys()),
        default="v20_pp4",
        help="Cluster scene template.",
    )
    parser.add_argument("--cluster_t_main", type=float, default=None)
    parser.add_argument("--cluster_t_sat", type=float, default=None)
    parser.add_argument("--cluster_R_target", type=float, default=None)
    parser.add_argument("--cluster_channel_alpha", type=float, default=None)
    parser.add_argument(
        "--cluster_isolated_sat_mode",
        choices=["loose", "default", "strict"],
        default=None,
    )
    parser.add_argument("--cluster_sat_main_ratio", type=float, default=None)
    parser.add_argument("--cluster_peak_cap_ratio", type=float, default=None)
    parser.add_argument("--cluster_max_sat_per_cluster", type=int, default=None)
    parser.add_argument("--cluster_max_tokens_per_cluster", type=int, default=None)
    parser.add_argument("--cluster_dense_guard", type=int, choices=[0, 1], default=None)
    parser.add_argument("--cluster_state_guard", type=int, choices=[0, 1], default=None)
    parser.add_argument("--cluster_token_profile_path", default=None)
    parser.add_argument("--cluster_export_tokens", default=None)
    parser.add_argument("--cluster_n_true_mean_path", default=None)
    parser.add_argument("--cluster_diag_dir", default=None)
    parser.add_argument(
        "--family-aware-postprocess-prior",
        type=int,
        default=1,
        help="Enable family-aware soft rerank prior in cluster postprocess (1=on, 0=off).",
    )

    args = parser.parse_args()
    predict_npz_to_mgf(
        npz_path=args.npz,
        checkpoint=args.checkpoint,
        output_mgf=args.output_mgf,
        output_summary_txt=args.output_summary_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_mz=args.min_mz,
        merge_tol=args.merge_tol,
        postprocess_mode=args.postprocess_mode,
        raw_min_intensity=args.raw_min_intensity,
        mz_mode=args.mz_mode,
        max_examples=args.max_examples,
        device=args.device,
        cluster_scene=args.cluster_scene,
        cluster_t_main=args.cluster_t_main,
        cluster_t_sat=args.cluster_t_sat,
        cluster_R_target=args.cluster_R_target,
        cluster_channel_alpha=args.cluster_channel_alpha,
        cluster_isolated_sat_mode=args.cluster_isolated_sat_mode,
        cluster_sat_main_ratio=args.cluster_sat_main_ratio,
        cluster_peak_cap_ratio=args.cluster_peak_cap_ratio,
        cluster_max_sat_per_cluster=args.cluster_max_sat_per_cluster,
        cluster_max_tokens_per_cluster=args.cluster_max_tokens_per_cluster,
        cluster_dense_guard=None if args.cluster_dense_guard is None else bool(args.cluster_dense_guard),
        cluster_state_guard=None if args.cluster_state_guard is None else bool(args.cluster_state_guard),
        cluster_token_profile_path=args.cluster_token_profile_path,
        cluster_export_tokens=args.cluster_export_tokens,
        cluster_n_true_mean_path=args.cluster_n_true_mean_path,
        cluster_diag_dir=args.cluster_diag_dir,
        family_aware_postprocess_prior=bool(args.family_aware_postprocess_prior),
        split=args.split,
    )


if __name__ == "__main__":
    main()
