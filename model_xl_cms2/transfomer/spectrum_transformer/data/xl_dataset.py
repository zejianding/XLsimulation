"""Load structured XL peptide-pair datasets and assemble training batches."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import (
    CHANNEL_NAMES, CMS2_CHANNEL_NAMES, CMS2_FRAGMENT_CHARGES, CMS2_INTACT_CHANNELS, CMS2_INTACT_OFFSET_TOKENS,
    CMS2_INTACT_ROW,
    CMS2_LINKER_IDS, CMS2_NUM_CHANNELS,
    CMS2_SCHEMA_VERSION, DEFAULT_MIN_VALID_MZ, MAIN_TOKEN_IDS, MAX_CLEAVAGE_LEN, MAX_PEPTIDE_LEN, NUM_TOKENS,
)
from ..training.target_processing import build_main_peak_mask
from .dataset import (
    SEQUENCE_CONTEXT_RESIDUE_TO_ID,
    normalize_features,
    normalize_inputs,
    parse_modified_sequence,
)
from token_table import CHARGE_TOKENS


CLEAVAGE_SCALAR_FEATURE_NAMES = (
    "prefix_fraction",
    "suffix_fraction",
    "left_has_mod",
    "right_has_mod",
    "left_mod_tanh",
    "right_mod_tanh",
    "nterm_mod_tanh",
    "relative_link_site",
    "link_site_in_prefix",
    "link_site_is_left",
    "link_site_is_right",
)
AUX_FEATURE_NAMES = ("precursor_charge", "precursor_mz", "rt_seconds")
MAIN_ANCHOR_THRESHOLD = 0.03

_MAIN_TOKEN = MAIN_TOKEN_IDS[0]
_BASE_SHAPE = (2, MAX_CLEAVAGE_LEN, len(CHANNEL_NAMES))
_GRID_SHAPE = (2, MAX_CLEAVAGE_LEN, NUM_TOKENS, len(CHANNEL_NAMES))
_CMS2_GRID_SHAPE = (2, MAX_CLEAVAGE_LEN, NUM_TOKENS, CMS2_NUM_CHANNELS)
_CHANNEL_CHARGES = np.array(CMS2_FRAGMENT_CHARGES[:len(CHANNEL_NAMES)])
_TOKEN_OFFSETS = np.column_stack([CHARGE_TOKENS[charge] for charge in _CHANNEL_CHARGES])
_CLEAVAGES = np.arange(1, MAX_CLEAVAGE_LEN + 1)


def _record_dtype(widths, input_shape, target_shape):
    return np.dtype(
        [
            ("group_id", "S64"),
            ("n_spectra", "<u4"),
            ("sequence", f"S{widths['sequence_bytes']}", (2,)),
            ("lengths", "u1", (2,)),
            ("link_sites", "u1", (2,)),
            ("crosslinker", f"S{widths['crosslinker_bytes']}"),
            ("linker_mass", "<f8"),
            ("precursor_charge", "u1"),
            ("precursor_mz", "<f8"),
            ("rt_seconds", "<f8"),
            ("input_theo_mz", "<f8", input_shape),
            ("target", "<f4", target_shape),
        ],
        align=True,
    )


def _append_intact_row(grid, values):
    # Version 1 has no offset axis. Both versions flatten only (stub, charge) into channels.
    tokens = MAIN_TOKEN_IDS if values.ndim == 4 else tuple(CMS2_INTACT_OFFSET_TOKENS.values())
    by_channel = values.reshape(*grid.shape[:2], len(CMS2_INTACT_CHANNELS), len(tokens))
    row = np.zeros((*grid.shape[:2], 1, *grid.shape[3:]), dtype=grid.dtype)
    for offset, token in enumerate(tokens):
        row[:, :, 0, token, :][..., CMS2_INTACT_CHANNELS] = by_channel[..., offset]
    return np.concatenate((grid, row), axis=2)


def _load_intact_labels(directory, split, records):
    directory = Path(directory)
    manifest = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    version = manifest.get("label_schema_version")
    axes = [["alpha (v4 sequence[0])", "beta (v4 sequence[1])"], ["short", "long"], [1, 2, 3]]
    shape = (2, 2, 3)
    if version == 2:
        axes.append(list(CMS2_INTACT_OFFSET_TOKENS))
        shape += (len(CMS2_INTACT_OFFSET_TOKENS),)
        if (manifest.get("axis_order") != ["chain", "stub", "fragment_charge", "mass_offset"]
                or manifest.get("offset_token_ids") != list(CMS2_INTACT_OFFSET_TOKENS.values())):
            raise ValueError("Unexpected five-offset intact label axes or token IDs")
    if (manifest.get("label_type") != "intact_peptide_stub" or version not in (1, 2)
            or manifest.get("axis_values") != axes or manifest.get("target_normalization") != "sqrt_clip"
            or manifest.get("clip_intensity") != 50000):
        raise ValueError("Incompatible intact label schema, axis order or intensity scale")
    labels = np.load(directory / f"{split}.npy", allow_pickle=False)
    dtype = np.dtype([("group_id", "S64"), ("mz", "<f8", shape),
                      ("target", "<f4", shape), ("valid_mask", "?", shape)])
    if labels.dtype != dtype or labels.shape != records.shape:
        raise ValueError(f"{split}: intact label dtype or row count differs from the expected layout")
    ids = labels["group_id"]
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, records["group_id"]):
        raise ValueError(f"{split}: intact group_id values must be unique and in v4 row order")
    mz, target, valid = labels["mz"], labels["target"], labels["valid_mask"]
    allowed_charge = np.arange(1, 4) <= records["precursor_charge"][:, None, None, None]
    if version == 2:
        allowed_charge = allowed_charge[..., None]
    if (not np.isfinite(mz).all() or not np.array_equal(valid, (mz >= DEFAULT_MIN_VALID_MZ) & allowed_charge)
            or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any()
            or mz[~valid].any() or target[~valid].any()):
        raise ValueError(f"{split}: invalid intact m/z, target or mask")
    return labels, tuple(manifest["offset_token_ids"]) if version == 2 else MAIN_TOKEN_IDS


def _normalization_stats(metadata, key, width):
    stats = metadata["train_statistics"][key]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if mean.shape != (width,) or std.shape != (width,):
        raise ValueError(f"Expected {width} entries in train_statistics.{key} mean and std")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError(f"Invalid train_statistics.{key} mean or std")
    return mean, std


def _encode_chain(sequence, length, link_site, aa_ids, scalar_features):
    residues, mod_deltas, nterm_delta = parse_modified_sequence(sequence)
    if len(residues) != length or not 2 <= length <= MAX_PEPTIDE_LEN or not 1 <= link_site <= length:
        raise ValueError(f"Invalid XL chain length or link site: {sequence!r}, {length=}, {link_site=}")

    residue_ids = [SEQUENCE_CONTEXT_RESIDUE_TO_ID[residue] for residue in residues]
    aa_ids[:length - 1, 0] = residue_ids[:-1]
    aa_ids[:length - 1, 1] = residue_ids[1:]
    cleavage = _CLEAVAGES[:length - 1]
    deltas = np.asarray(mod_deltas, dtype=np.float64)
    scaled = np.tanh(deltas / 50.0)
    scalar_features[:length - 1] = np.column_stack(
        (
            cleavage / length,
            (length - cleavage) / length,
            np.abs(deltas[:-1]) > 1e-6,
            np.abs(deltas[1:]) > 1e-6,
            scaled[:-1],
            scaled[1:],
            np.full(length - 1, math.tanh(nterm_delta / 50.0)),
            (cleavage + 0.5 - link_site) / length,
            link_site <= cleavage,
            link_site == cleavage,
            link_site == cleavage + 1,
        )
    )


class XLInputCollator:
    """Prepare model inputs and retain the original XL m/z grid for export."""

    def __init__(self, metadata, schema_version):
        if type(schema_version) is not int or schema_version not in (1, 2, CMS2_SCHEMA_VERSION):
            raise ValueError(f"Unsupported XL input schema version {schema_version!r}; expected 1, 2 or 4")
        if metadata["train_statistics"].get("aux_order") != list(AUX_FEATURE_NAMES):
            raise ValueError("Unexpected XL auxiliary feature order in train_statistics")
        self.schema_version = schema_version
        self.is_cms2 = schema_version == CMS2_SCHEMA_VERSION
        if self.is_cms2 and (
            metadata.get("channel_names") != list(CMS2_CHANNEL_NAMES)
            or metadata.get("sequence_constrained") != "full_same_state"
        ):
            raise ValueError("Expected cMS2 v4 channel order and full_same_state targets")
        channels = CMS2_NUM_CHANNELS if self.is_cms2 else len(CHANNEL_NAMES)
        self._mz_mean, self._mz_std = _normalization_stats(metadata, "mz_channels", channels)
        self._aux_mean, self._aux_std = _normalization_stats(metadata, "aux_features", len(AUX_FEATURE_NAMES))

    def __call__(self, records, intact_labels=None):
        count = len(records)
        lengths = np.stack([row["lengths"] for row in records]).astype(np.int64)
        link_sites = np.stack([row["link_sites"] for row in records]).astype(np.int64)
        aux_raw = np.asarray([[row[name] for name in AUX_FEATURE_NAMES] for row in records], dtype=np.float64)
        charges = aux_raw[:, 0].astype(np.int64)
        mz = np.stack([row["input_theo_mz"] for row in records])
        if self.is_cms2:
            if mz.shape[1:] != _CMS2_GRID_SHAPE:
                raise ValueError(f"Expected expanded cMS2 m/z grids of shape {_CMS2_GRID_SHAPE}")
            valid = np.isfinite(mz) & (mz >= DEFAULT_MIN_VALID_MZ)
            if intact_labels is not None:
                mz = _append_intact_row(mz, intact_labels["mz"])
                valid = _append_intact_row(valid, intact_labels["valid_mask"])
        else:
            if self.schema_version == 1:
                mz = mz[:, :, :, None, :] + _TOKEN_OFFSETS
            candidate = (
                (_CLEAVAGES[None, None, :, None, None] < lengths[:, :, None, None, None])
                & (_CHANNEL_CHARGES <= charges[:, None, None, None, None])
                & np.isfinite(mz)
            )
            if self.schema_version == 1:
                mz = np.where(candidate, mz, 0.0)
            valid = candidate & (mz >= DEFAULT_MIN_VALID_MZ)

        aa_ids = np.zeros((count, 2, MAX_CLEAVAGE_LEN, 2), dtype=np.int64)
        scalar_features = np.zeros(
            (count, 2, MAX_CLEAVAGE_LEN, len(CLEAVAGE_SCALAR_FEATURE_NAMES)), dtype=np.float32
        )
        for index, row in enumerate(records):
            for chain, seq in enumerate(row["sequence"]):
                _encode_chain(
                    seq.decode("utf-8") if isinstance(seq, bytes) else str(seq),
                    int(lengths[index, chain]),
                    int(link_sites[index, chain]),
                    aa_ids[index, chain],
                    scalar_features[index, chain],
                )

        batch = {
            "input": torch.from_numpy(normalize_inputs(mz, valid, self._mz_mean, self._mz_std)),
            "valid_mask": torch.from_numpy(valid),
            "input_theo_mz": torch.from_numpy(mz),
            "aux_features": torch.from_numpy(normalize_features(aux_raw, self._aux_mean, self._aux_std)),
            "precursor_charge": torch.from_numpy(charges),
            "lengths": torch.from_numpy(lengths),
            "link_sites": torch.from_numpy(link_sites),
            "cleavage_aa_ids": torch.from_numpy(aa_ids),
            "cleavage_scalar_features": torch.from_numpy(scalar_features),
        }
        if self.is_cms2:
            linkers = [row["crosslinker"] for row in records]
            try:
                batch["linker_id"] = torch.tensor([
                    CMS2_LINKER_IDS[name.decode("ascii") if isinstance(name, bytes) else name]
                    for name in linkers
                ], dtype=torch.long)
            except KeyError as error:
                raise ValueError(f"Unknown cMS2 crosslinker {error.args[0]!r}") from error
        return batch


class XLSpectrumDataset(Dataset):
    """Read one XL average-spectrum group per record using a read-only mmap."""

    def __init__(self, dataset_dir, split, intact_labels_dir=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown XL split {split!r}; expected train, val, or test")
        self.dataset_dir = Path(dataset_dir).resolve()
        self.split = split
        self._records_path = self.dataset_dir / f"{split}.npy"
        self.metadata = json.loads((self.dataset_dir / "meta.json").read_text(encoding="utf-8"))
        self.schema_version = self.metadata.get("schema_version", 1)
        self._input_collator = XLInputCollator(self.metadata, self.schema_version)
        is_cms2 = self._input_collator.is_cms2
        grid_shape = _CMS2_GRID_SHAPE if is_cms2 else _GRID_SHAPE
        input_shape = _BASE_SHAPE if self.schema_version == 1 else grid_shape
        dtype = _record_dtype(self.metadata["string_widths"], input_shape, grid_shape)
        expected = {
            "input_representation": "base_raw_mz" if self.schema_version == 1 else "expanded_raw_mz",
            "input_shape": list(input_shape),
            "input_dtype": "float64",
            "main_token_id": _MAIN_TOKEN,
        }
        if is_cms2:
            expected.update(spectrum_type="cms2", average_across_nce=True)
        for field, value in expected.items():
            default = value if self.schema_version == 1 else None
            if self.metadata.get(field, default) != value:
                raise ValueError(f"Unexpected XL schema v{self.schema_version} metadata field {field!r}")
        if (
            self.metadata.get("grid_shape") != list(grid_shape)
            or self.metadata.get("fields") != list(dtype.names)
            or self.metadata.get("record_bytes") != dtype.itemsize
            or (not is_cms2 and self.metadata.get("channel_names", list(CHANNEL_NAMES)) != list(CHANNEL_NAMES))
        ):
            raise ValueError(f"XL dataset metadata does not match the v{self.schema_version} record layout")
        if (
            self.metadata.get("min_valid_mz") != DEFAULT_MIN_VALID_MZ
            or self.metadata.get("target_normalization") != "sqrt_clip"
            or self.metadata.get("clip_intensity") != 50_000
            or (not is_cms2 and self.metadata.get("sequence_constrained") != "full")
        ):
            raise ValueError("XL dataset must use minimum m/z 100 and full-constrained sqrt_clip(50000) targets")

        records = np.load(self._records_path, mmap_mode="r", allow_pickle=False)
        count = self.metadata["splits"][split]["written_groups"]
        if records.dtype != dtype or records.shape != (count,):
            raise ValueError(f"{self._records_path} has a record dtype or shape inconsistent with meta.json")
        self._count = len(records)
        self._records = records
        self.intact_mode = intact_labels_dir is not None
        self.intact_token_ids = ()
        if self.intact_mode:
            if not is_cms2:
                raise ValueError("Intact labels require a cMS2 v4 dataset")
            self._intact_labels, self.intact_token_ids = _load_intact_labels(intact_labels_dir, split, records)
            self._intact_indices = {gid: i for i, gid in enumerate(self._intact_labels["group_id"])}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_records"] = None
        return state

    def _get_records(self):
        if self._records is None:
            self._records = np.load(self._records_path, mmap_mode="r", allow_pickle=False)
        return self._records

    @property
    def precursor_charges(self):
        """Precursor charge per record, in the order of this split."""
        return self._get_records()["precursor_charge"].astype(np.int64)

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        return self._get_records()[index]

    def collate_fn(self, records):
        labels = None
        values = np.stack([row["target"] for row in records])
        if self.intact_mode:
            labels = self._intact_labels[[self._intact_indices[row["group_id"]] for row in records]]
            values = _append_intact_row(values, labels["target"])
        batch = self._input_collator(records, labels)
        valid_mask = batch["valid_mask"]
        target = torch.from_numpy(np.where(valid_mask.numpy(), values, np.float32(0)))
        main_token = slice(_MAIN_TOKEN, _MAIN_TOKEN + 1)
        backbone = (slice(None), slice(None), slice(CMS2_INTACT_ROW), main_token, slice(None))
        main_anchor_mask = torch.zeros_like(valid_mask)
        main_anchor_mask[backbone] = valid_mask[backbone]
        count = len(records)
        anchor_shape = (count * 2, MAX_CLEAVAGE_LEN, 1, target.shape[-1])
        main_values = target[backbone].reshape(anchor_shape)
        main_valid = main_anchor_mask[backbone].reshape(anchor_shape)
        local_max = build_main_peak_mask(main_values, main_valid)
        positive = local_max * (main_values >= MAIN_ANCHOR_THRESHOLD)
        main_anchor_target = torch.zeros_like(target)
        main_anchor_target[backbone] = positive.reshape(count, 2, *anchor_shape[1:])

        return {
            **batch,
            "target": target,
            "main_anchor_target": main_anchor_target,
            "main_anchor_mask": main_anchor_mask,
            "group_id": [row["group_id"].decode("ascii") for row in records],
        }
