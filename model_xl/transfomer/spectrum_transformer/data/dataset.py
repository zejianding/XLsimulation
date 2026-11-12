"""V20 minimal dataset utilities.

This module keeps only the data-loading and normalization pieces required by
the current V20 clean-baseline trainer, plus a compact cleavage-context path
for the upgraded native model.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


SEQUENCE_CONTEXT_PAD_ID = 0
SEQUENCE_CONTEXT_RESIDUES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SEQUENCE_CONTEXT_RESIDUE_TO_ID = {
    residue: index + 1 for index, residue in enumerate(SEQUENCE_CONTEXT_RESIDUES)
}
SEQUENCE_CONTEXT_SCALAR_DIM = 7

_SINGLE_MASS_RE = re.compile(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)")
_MOD_TOKEN_RE = re.compile(rf"(?:{_SINGLE_MASS_RE.pattern})(?:[+-](?:\d+\.\d+|\d+|\.\d+))*")


def _parse_mod_token(raw: str) -> float:
    parts = _SINGLE_MASS_RE.findall(raw)
    return sum(float(p) for p in parts) if parts else 0.0


def _try_read_mod_token(seq: str, i: int):
    n = len(seq)
    if i >= n:
        return None, i
    if seq[i] in "([":
        close = ")" if seq[i] == "(" else "]"
        j = seq.find(close, i + 1)
        if j != -1:
            inner = seq[i + 1:j].strip()
            if inner and _MOD_TOKEN_RE.fullmatch(inner):
                return _parse_mod_token(inner), j + 1
        return None, i
    m = _MOD_TOKEN_RE.match(seq, i)
    if m:
        token = m.group()
        if any(ch.isdigit() for ch in token):
            return _parse_mod_token(token), m.end()
    return None, i


def parse_modified_sequence(seq):
    seq = "" if seq is None else str(seq)
    residues = []
    residue_mod_deltas = []
    nterm_mod_delta = 0.0
    i = 0
    n = len(seq)

    while i < n:
        ch = seq[i]
        if "A" <= ch <= "Z":
            break
        delta, new_i = _try_read_mod_token(seq, i)
        if delta is not None:
            nterm_mod_delta += delta
            i = new_i
            continue
        i += 1

    while i < n:
        ch = seq[i]
        if "A" <= ch <= "Z":
            residues.append(ch)
            residue_mod_deltas.append(0.0)
            i += 1
            while i < n:
                delta, new_i = _try_read_mod_token(seq, i)
                if delta is not None:
                    residue_mod_deltas[-1] += delta
                    i = new_i
                    continue
                break
            continue
        delta, new_i = _try_read_mod_token(seq, i)
        if delta is not None and residues:
            residue_mod_deltas[-1] += delta
            i = new_i
            continue
        i += 1
    return residues, residue_mod_deltas, nterm_mod_delta


def encode_cleavage_context(seq, length, max_cleavage_len=39):
    aa_ids = np.zeros((max_cleavage_len, 2), dtype=np.int16)
    scalar = np.zeros((max_cleavage_len, SEQUENCE_CONTEXT_SCALAR_DIM), dtype=np.float16)

    residues, residue_mod_deltas, nterm_mod_delta = parse_modified_sequence(seq)
    residue_count = len(residues)
    expected_len = int(length) if length is not None else residue_count
    usable_len = max(min(residue_count, expected_len), 0)
    cleavage_count = min(max(usable_len - 1, 0), max_cleavage_len)
    if cleavage_count <= 0:
        return aa_ids, scalar

    denom = float(max(usable_len, 1))
    nterm_scaled = float(math.tanh(nterm_mod_delta / 50.0))
    for cleavage_idx in range(cleavage_count):
        left_residue = residues[cleavage_idx]
        right_residue = residues[cleavage_idx + 1]
        aa_ids[cleavage_idx, 0] = SEQUENCE_CONTEXT_RESIDUE_TO_ID.get(left_residue, SEQUENCE_CONTEXT_PAD_ID)
        aa_ids[cleavage_idx, 1] = SEQUENCE_CONTEXT_RESIDUE_TO_ID.get(right_residue, SEQUENCE_CONTEXT_PAD_ID)

        left_delta = float(residue_mod_deltas[cleavage_idx])
        right_delta = float(residue_mod_deltas[cleavage_idx + 1])
        prefix_fraction = (cleavage_idx + 1) / denom
        suffix_fraction = (usable_len - cleavage_idx - 1) / denom
        scalar[cleavage_idx] = np.asarray(
            [
                prefix_fraction,
                suffix_fraction,
                1.0 if abs(left_delta) > 1e-6 else 0.0,
                1.0 if abs(right_delta) > 1e-6 else 0.0,
                math.tanh(left_delta / 50.0),
                math.tanh(right_delta / 50.0),
                nterm_scaled,
            ],
            dtype=np.float16,
        )
    return aa_ids, scalar


def build_cleavage_context_arrays(seqs, lengths, max_cleavage_len=39):
    if seqs is None:
        return None, None
    n_samples = len(seqs)
    aa_ids = np.zeros((n_samples, max_cleavage_len, 2), dtype=np.int16)
    scalar = np.zeros((n_samples, max_cleavage_len, SEQUENCE_CONTEXT_SCALAR_DIM), dtype=np.float16)
    for index, (seq, length) in enumerate(zip(seqs, lengths)):
        aa_ids[index], scalar[index] = encode_cleavage_context(seq, length, max_cleavage_len=max_cleavage_len)
    return aa_ids, scalar


def build_valid_mask(position_masks, channel_masks):
    return position_masks[:, :, None, None] * channel_masks[:, None, None, :]


def load_npz_data(npz_path):
    npz_path = str(npz_path)
    if os.path.isdir(npz_path):
        if os.path.exists(os.path.join(npz_path, "meta.json")):
            out = _normalize_npz_keys(_load_memmap_data(npz_path))
        elif os.path.exists(os.path.join(npz_path, "index.json")):
            out = _normalize_npz_keys(load_sharded_npz(npz_path))
        else:
            raise FileNotFoundError(f"Unsupported dataset directory format: {npz_path}")
    else:
        raw = np.load(npz_path, allow_pickle=True)
        out = {k: raw[k] for k in raw.files}

    # Normalize naming differences across older dataset exports.
    if "inputs" not in out and "input_mz_matrices" in out:
        out["inputs"] = out["input_mz_matrices"]
    if "targets" not in out and "target_intensity_matrices" in out:
        out["targets"] = out["target_intensity_matrices"]
    if "inputs" not in out and "input" in out:
        out["inputs"] = out["input"]
    if "targets" not in out and "target" in out:
        out["targets"] = out["target"]
    if "valid_masks" not in out and "valid_mask" in out:
        out["valid_masks"] = out["valid_mask"]
    required = ["inputs", "targets"]
    for key in required:
        if key not in out:
            raise KeyError(f"Missing required NPZ field: {key}")
    if "valid_masks" not in out and not ("position_masks" in out and "channel_masks" in out):
        raise KeyError("Missing valid_masks and also missing position_masks/channel_masks.")

    if "aux_features" not in out:
        charges = out.get("charges", np.ones((out["inputs"].shape[0],), dtype=np.float32))
        pepmasses = out.get("pepmasses", np.zeros((out["inputs"].shape[0],), dtype=np.float32))
        rt = out.get("rtinseconds", np.zeros((out["inputs"].shape[0],), dtype=np.float32))
        out["aux_features"] = np.stack(
            [charges.astype(np.float32), pepmasses.astype(np.float32), rt.astype(np.float32)],
            axis=1,
        )

    if "charges" not in out:
        out["charges"] = np.ones((out["inputs"].shape[0],), dtype=np.int64)
    if "lengths" not in out:
        out["lengths"] = np.full((out["inputs"].shape[0],), 40, dtype=np.int64)
    if "seqs" not in out and "sequences" in out:
        out["seqs"] = out["sequences"]

    if "seqs" in out:
        aa_ids, scalar = build_cleavage_context_arrays(out["seqs"], out["lengths"])
        out["cleavage_aa_ids"] = aa_ids
        out["cleavage_scalar_features"] = scalar

    if "valid_masks" not in out and "position_masks" in out and "channel_masks" in out:
        # Keep low-memory raw masks for training; downstream eager paths can
        # still materialize dense masks lazily when needed.
        pass

    return out


def _normalize_npz_keys(out):
    out = dict(out)
    if "inputs" not in out and "input_mz_matrices" in out:
        out["inputs"] = out.pop("input_mz_matrices")
    if "targets" not in out and "target_intensity_matrices" in out:
        out["targets"] = out.pop("target_intensity_matrices")
    if "seqs" not in out and "sequences" in out:
        out["seqs"] = out["sequences"]
    if "aux_features" not in out and all(k in out for k in ("charges", "pepmasses", "rtinseconds")):
        out["aux_features"] = np.stack(
            [
                out["charges"].astype(np.float32),
                out["pepmasses"].astype(np.float32),
                out["rtinseconds"].astype(np.float32),
            ],
            axis=1,
        )
    return out


def _load_memmap_data(merged_dir):
    with open(os.path.join(merged_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    result = {}
    for key in meta.get("merge_keys", []):
        path = os.path.join(merged_dir, f"{key}.npy")
        result[key] = np.load(path, mmap_mode="r")
    for key in meta.get("obj_merge_keys", []):
        path = os.path.join(merged_dir, f"{key}.npy")
        result[key] = np.load(path, allow_pickle=True)
    for key in meta.get("scalar_keys", []):
        path = os.path.join(merged_dir, f"_scalar_{key}.npy")
        if os.path.exists(path):
            result[key] = np.load(path, allow_pickle=True)
    return result


# Row-count threshold that separates per-example arrays from constants in a
# shard.  Mirrors shards_to_npz.py / merge_shards_to_memmap.py.
_SHARD_CONST_ROW_THRESHOLD = 100


def load_sharded_npz(shard_dir):
    index_path = os.path.join(shard_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found in shard dir: {shard_dir}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    shard_files = index.get("shard_files", [])
    if not shard_files:
        shard_files = sorted(
            name for name in os.listdir(shard_dir)
            if name.startswith("shard_") and name.endswith(".npz")
        )
    shard0 = dict(np.load(os.path.join(shard_dir, shard_files[0]), allow_pickle=True))
    shard0 = _normalize_npz_keys(shard0)
    total = int(index.get("total_examples", 0))

    # Per-example arrays grow to ``total``; constants (channel_names, the token
    # tables, 0-d scalars) are kept as-is.  Prefer the authoritative key list
    # that NPZBuilder writes into index.json; fall back to the row-count
    # heuristic for older shard dirs (which mis-classifies shards of <= 100 rows).
    declared = index.get("per_example_keys")
    if declared is not None:
        # shard0 went through _normalize_npz_keys(), which renames two of these.
        _aliases = {
            "input_mz_matrices": "inputs",
            "target_intensity_matrices": "targets",
        }
        declared = set(declared) | {_aliases[k] for k in declared if k in _aliases}
    result = {}
    per_example_keys = []
    for key, val in shard0.items():
        if declared is not None:
            is_per_example = key in declared
        else:
            is_per_example = (
                getattr(val, "ndim", 0) >= 1
                and val.shape[0] > _SHARD_CONST_ROW_THRESHOLD
            )
        if not is_per_example:
            result[key] = val
        else:
            shape = list(val.shape)
            shape[0] = total
            result[key] = np.empty(tuple(shape), dtype=val.dtype)
            per_example_keys.append(key)

    if not per_example_keys:
        return result

    offset = 0
    for shard_name in shard_files:
        with np.load(os.path.join(shard_dir, shard_name), allow_pickle=True) as d:
            d = _normalize_npz_keys(dict(d))
            n = min(d[per_example_keys[0]].shape[0], max(total - offset, 0))
            if n <= 0:
                break
            for key in per_example_keys:
                result[key][offset:offset + n] = d[key][:n]
            offset += n
    return result


def compute_channel_stats(inputs, valid_mask):
    means = np.zeros((inputs.shape[-1],), dtype=np.float32)
    stds = np.ones((inputs.shape[-1],), dtype=np.float32)
    for channel in range(inputs.shape[-1]):
        mask = valid_mask[..., channel].astype(bool)
        values = inputs[..., channel][mask]
        if values.size > 0:
            means[channel] = float(values.mean())
            std = float(values.std())
            stds[channel] = std if std > 1e-6 else 1.0
    return means, stds


def compute_channel_stats_from_position_channel_masks(
    inputs,
    position_masks,
    channel_masks,
    min_mz=100.0,
    chunk_size=2048,
):
    n_samples = int(inputs.shape[0])
    n_channels = int(inputs.shape[-1])
    sum_vals = np.zeros((n_channels,), dtype=np.float64)
    sum_sq_vals = np.zeros((n_channels,), dtype=np.float64)
    count_vals = np.zeros((n_channels,), dtype=np.float64)
    n_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)
    for start in tqdm(
        range(0, n_samples, chunk_size),
        total=n_chunks,
        desc="channel stats",
        leave=False,
        unit="chunk",
        dynamic_ncols=True,
    ):
        end = min(start + chunk_size, n_samples)
        input_chunk = inputs[start:end]
        pos_chunk = position_masks[start:end]
        ch_chunk = channel_masks[start:end]
        vm = build_valid_mask(pos_chunk, ch_chunk).astype(bool)
        if min_mz is not None and float(min_mz) > 0:
            vm = vm & (input_chunk >= float(min_mz))
        for channel in range(n_channels):
            values = input_chunk[..., channel][vm[..., channel]]
            if values.size == 0:
                continue
            sum_vals[channel] += float(values.sum(dtype=np.float64))
            sum_sq_vals[channel] += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
            count_vals[channel] += float(values.size)

    means = np.zeros((n_channels,), dtype=np.float32)
    stds = np.ones((n_channels,), dtype=np.float32)
    valid = count_vals > 0
    means[valid] = (sum_vals[valid] / count_vals[valid]).astype(np.float32)
    var = np.zeros_like(sum_vals, dtype=np.float64)
    var[valid] = np.maximum(sum_sq_vals[valid] / count_vals[valid] - np.square(means[valid], dtype=np.float64), 1e-12)
    stds[valid] = np.sqrt(var[valid]).astype(np.float32)
    stds = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)
    return means, stds


def compute_feature_stats(aux_features):
    means = aux_features.mean(axis=0).astype(np.float32)
    stds = aux_features.std(axis=0).astype(np.float32)
    stds = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)
    return means, stds


def normalize_inputs(inputs, valid_mask, means, stds):
    normalized = (inputs - means.reshape((1, 1, 1, -1))) / stds.reshape((1, 1, 1, -1))
    normalized = np.where(valid_mask, normalized, 0.0).astype(np.float32)
    return normalized


def normalize_features(aux_features, means, stds):
    return ((aux_features - means) / stds).astype(np.float32)


class SpectrumDataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        valid_masks,
        aux_features,
        charges=None,
        lengths=None,
        dataset_indices=None,
        cleavage_aa_ids=None,
        cleavage_scalar_features=None,
    ):
        self.inputs = inputs.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.valid_masks = valid_masks.astype(np.float32, copy=False)
        self.aux_features = aux_features.astype(np.float32, copy=False)
        self.charges = charges if charges is not None else np.ones((len(inputs),), dtype=np.int64)
        self.lengths = lengths if lengths is not None else np.full((len(inputs),), 40, dtype=np.int64)
        self.dataset_indices = (
            np.asarray(dataset_indices, dtype=np.int64)
            if dataset_indices is not None
            else np.arange(len(inputs), dtype=np.int64)
        )
        self.cleavage_aa_ids = cleavage_aa_ids
        self.cleavage_scalar_features = cleavage_scalar_features

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        item = {
            "input": torch.from_numpy(self.inputs[index]),
            "target": torch.from_numpy(self.targets[index]),
            "valid_mask": torch.from_numpy(self.valid_masks[index]),
            "aux_features": torch.from_numpy(self.aux_features[index]),
            "charge": torch.tensor(int(self.charges[index]), dtype=torch.long),
            "length": torch.tensor(int(self.lengths[index]), dtype=torch.long),
            "dataset_index": torch.tensor(int(self.dataset_indices[index]), dtype=torch.long),
        }
        if self.cleavage_aa_ids is not None:
            item["cleavage_aa_ids"] = torch.from_numpy(self.cleavage_aa_ids[index]).long()
        if self.cleavage_scalar_features is not None:
            item["cleavage_scalar_features"] = torch.from_numpy(
                self.cleavage_scalar_features[index].astype(np.float32, copy=False)
            )
        return item


class ShardedSpectrumDataset(Dataset):
    def __init__(
        self,
        shard_dir,
        channel_means,
        channel_stds,
        aux_means,
        aux_stds,
        cache_size=2,
        min_mz=None,
    ):
        self.shard_dir = str(shard_dir)
        with open(os.path.join(self.shard_dir, "index.json"), "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.shard_files = self.index["shard_files"]
        self.total = int(self.index["total_examples"])
        self.cache_size = int(cache_size)
        self.channel_means = np.asarray(channel_means, dtype=np.float32).reshape(1, 1, -1)
        self.channel_stds = np.asarray(channel_stds, dtype=np.float32).reshape(1, 1, -1)
        self.aux_means = np.asarray(aux_means, dtype=np.float32).reshape(1, -1)
        self.aux_stds = np.asarray(aux_stds, dtype=np.float32).reshape(1, -1)
        self.min_mz = None if min_mz is None else float(min_mz)
        self._cache = {}
        self._access_order = []
        self.shard_boundaries = [0]
        for shard_name in self.shard_files:
            with np.load(os.path.join(self.shard_dir, shard_name), allow_pickle=True) as d:
                key = "input_mz_matrices" if "input_mz_matrices" in d else "inputs"
                self.shard_boundaries.append(self.shard_boundaries[-1] + int(d[key].shape[0]))

    def __len__(self):
        return self.total

    def _get_shard(self, shard_idx):
        if shard_idx not in self._cache:
            if len(self._cache) >= self.cache_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            data = dict(np.load(os.path.join(self.shard_dir, self.shard_files[shard_idx]), allow_pickle=True))
            self._cache[shard_idx] = _normalize_npz_keys(data)
            self._access_order.append(shard_idx)
        return self._cache[shard_idx]

    def __getitem__(self, index):
        shard_idx = bisect.bisect_right(self.shard_boundaries, index) - 1
        local_idx = index - self.shard_boundaries[shard_idx]
        d = self._get_shard(shard_idx)

        input_sample = d["inputs"][local_idx].astype(np.float32, copy=False)
        target_sample = d["targets"][local_idx].astype(np.float32, copy=False)
        if "valid_masks" in d:
            valid_mask = d["valid_masks"][local_idx].astype(np.float32, copy=False)
            if self.min_mz is not None and self.min_mz > 0:
                valid_mask = valid_mask * (input_sample >= self.min_mz).astype(np.float32)
        else:
            row_valid = d["position_masks"][local_idx][:, None, None] > 0
            channel_valid = d["channel_masks"][local_idx][None, None, :] > 0
            valid_mask = np.broadcast_to(row_valid & channel_valid, input_sample.shape).astype(np.float32, copy=False)
            if self.min_mz is not None and self.min_mz > 0:
                valid_mask = valid_mask * (input_sample >= self.min_mz).astype(np.float32)
        norm_input = (input_sample - self.channel_means) / np.maximum(self.channel_stds, 1e-6)
        norm_input = np.where(valid_mask, norm_input, 0.0).astype(np.float32, copy=False)
        target = np.where(valid_mask, target_sample, 0.0).astype(np.float32, copy=False)
        aux_raw = np.array(
            [float(d["charges"][local_idx]), float(d["pepmasses"][local_idx]), float(d["rtinseconds"][local_idx])],
            dtype=np.float32,
        )
        aux = ((aux_raw - self.aux_means.reshape(-1)) / np.maximum(self.aux_stds.reshape(-1), 1e-6)).astype(np.float32)
        item = {
            "input": torch.from_numpy(norm_input),
            "target": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid_mask.astype(np.float32, copy=False)),
            "aux_features": torch.from_numpy(aux),
            "charge": torch.tensor(int(d["charges"][local_idx]), dtype=torch.long),
            "length": torch.tensor(int(d["lengths"][local_idx]), dtype=torch.long),
            "dataset_index": torch.tensor(int(index), dtype=torch.long),
        }
        seqs = d.get("seqs")
        if seqs is not None:
            aa_ids, scalar = encode_cleavage_context(seqs[local_idx], int(d["lengths"][local_idx]), max_cleavage_len=39)
            item["cleavage_aa_ids"] = torch.from_numpy(aa_ids).long()
            item["cleavage_scalar_features"] = torch.from_numpy(scalar.astype(np.float32, copy=False))
        return item


class LazySpectrumDataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        aux_features,
        charges,
        lengths,
        dataset_indices,
        channel_means,
        channel_stds,
        valid_masks=None,
        position_masks=None,
        channel_masks=None,
        min_mz=100.0,
        cleavage_aa_ids=None,
        cleavage_scalar_features=None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.valid_masks = valid_masks
        self.position_masks = position_masks
        self.channel_masks = channel_masks
        self.aux_features = aux_features
        self.charges = charges
        self.lengths = lengths
        self.dataset_indices = np.asarray(dataset_indices)
        self.channel_means = np.asarray(channel_means, dtype=np.float32).reshape(1, 1, -1)
        self.channel_stds = np.asarray(channel_stds, dtype=np.float32).reshape(1, 1, -1)
        self.min_mz = None if min_mz is None or float(min_mz) <= 0.0 else float(min_mz)
        self.cleavage_aa_ids = cleavage_aa_ids
        self.cleavage_scalar_features = cleavage_scalar_features

    def __len__(self):
        return self.inputs.shape[0]

    def _valid_mask_for(self, index, input_sample):
        if self.valid_masks is not None:
            valid_mask = self.valid_masks[index].astype(np.float32, copy=False)
        else:
            row_valid = self.position_masks[index][:, None, None] > 0
            channel_valid = self.channel_masks[index][None, None, :] > 0
            valid_mask = np.broadcast_to(row_valid & channel_valid, input_sample.shape).astype(np.float32, copy=False)
        if self.min_mz is not None:
            valid_mask = valid_mask * (input_sample >= self.min_mz).astype(np.float32)
        return valid_mask

    def __getitem__(self, index):
        input_sample = self.inputs[index].astype(np.float32, copy=False)
        valid_mask = self._valid_mask_for(index, input_sample)
        target = self.targets[index].astype(np.float32, copy=False)
        norm_input = (input_sample - self.channel_means) / np.maximum(self.channel_stds, 1e-6)
        norm_input = np.where(valid_mask, norm_input, 0.0).astype(np.float32, copy=False)
        target = np.where(valid_mask, target, 0.0).astype(np.float32, copy=False)
        item = {
            "input": torch.from_numpy(norm_input),
            "target": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid_mask),
            "aux_features": torch.from_numpy(self.aux_features[index].astype(np.float32, copy=False)),
            "charge": torch.tensor(int(self.charges[index]), dtype=torch.long),
            "length": torch.tensor(int(self.lengths[index]), dtype=torch.long),
            "dataset_index": torch.tensor(int(self.dataset_indices[index]), dtype=torch.long),
        }
        if self.cleavage_aa_ids is not None:
            item["cleavage_aa_ids"] = torch.from_numpy(self.cleavage_aa_ids[index]).long()
        if self.cleavage_scalar_features is not None:
            item["cleavage_scalar_features"] = torch.from_numpy(
                self.cleavage_scalar_features[index].astype(np.float32, copy=False)
            )
        return item
