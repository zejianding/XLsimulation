# build_spectrum_prediction_examples_len40.py

import os
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

# Shared token vocabulary — the single source of truth lives in
# <project_root>/token_table.py so that the filter table, the grid table and
# the inference table cannot drift apart.
_project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from token_table import (  # noqa: E402
    AA_MASS,
    CHANNEL_NAMES,
    CHARGE_TOKENS,
    H2O,
    MAX_CLEAVAGE_LEN,
    MAX_PEPTIDE_LEN,
    NUM_CHANNELS,
    NUM_TOKENS,
    PROTON,
)


# -----------------------------
# MassiveKB 数字质量修饰解析
# 支持格式: M+15.995, C+57.021, (+42.011)PEPTIDE, +43.006-17.027 等
# -----------------------------
SINGLE_MASS_PATTERN = r"[+-]?(?:\d+\.\d+|\d+|\.\d+)"

MOD_TOKEN_RE = re.compile(
    rf"{SINGLE_MASS_PATTERN}(?:[+-](?:\d+\.\d+|\d+|\.\d+))*"
)
MOD_COMPONENT_RE = re.compile(SINGLE_MASS_PATTERN)


def mod_token_to_mass(mod_token: str) -> float:
    token = str(mod_token).strip()
    if len(token) >= 2 and ((token[0] == "(" and token[-1] == ")") or (token[0] == "[" and token[-1] == "]")):
        token = token[1:-1].strip()
    components = MOD_COMPONENT_RE.findall(token)
    if not components:
        raise ValueError(f"修饰 token 中没有可解析质量: {mod_token}")
    return sum(float(x) for x in components)


def read_mod_token_at(seq: str, i: int):
    n = len(seq)
    if i >= n:
        return None, i

    if seq[i] in "([":
        close_char = ")" if seq[i] == "(" else "]"
        j = seq.find(close_char, i + 1)
        if j == -1:
            raise ValueError(f"修饰括号未闭合: {seq[i:]} in {seq}")
        inner = seq[i + 1:j].strip()
        if inner and any(ch.isdigit() for ch in inner) and MOD_TOKEN_RE.fullmatch(inner):
            return inner, j + 1
        return None, i

    m = MOD_TOKEN_RE.match(seq, i)
    if m and any(ch.isdigit() for ch in m.group(0)):
        return m.group(0), m.end()

    return None, i


# -----------------------------
# Token table: imported from token_table.py (46 columns, 6-decimal offsets)
# -----------------------------


# -----------------------------
# Sequence parsing
# -----------------------------
def parse_sequence_with_mods(seq: str):
    """
    解析 MassiveKB 数字质量格式的 SEQ。

    支持:
        M+15.995 / M(+15.995) / M[+15.995]
        C+57.021 / C(+57.021)
        (+42.011)PEPTIDE / +42.011PEPTIDE
        A(+42.011)CDE / A+42.011CDE
        +43.006-17.027 (组合修饰，净质量=25.979)

    Return:
        residues: [(AA, residue_mass_with_mods), ...]
        nterm_shift: N端修饰质量
    """
    residues = []
    nterm_shift = 0.0

    i = 0
    n = len(seq)

    # 1) N端修饰：第一个氨基酸之前的所有数字质量 token
    while i < n:
        ch = seq[i]
        if ch.isalpha() and ch.isupper():
            break

        raw_mod, new_i = read_mod_token_at(seq, i)
        if raw_mod is not None:
            nterm_shift += mod_token_to_mass(raw_mod)
            i = new_i
            continue

        if ch in ".-_ ":
            i += 1
            continue

        raise ValueError(f"无法解析 N端 序列位置 {i}: {seq[i:]} in {seq}")

    # 2) 氨基酸及残基修饰
    while i < n:
        ch = seq[i]

        if not (ch.isalpha() and ch.isupper()):
            raw_mod, new_i = read_mod_token_at(seq, i)
            if raw_mod is not None and residues:
                aa, old_mass = residues[-1]
                residues[-1] = (aa, old_mass + mod_token_to_mass(raw_mod))
                i = new_i
                continue
            # 非大写字母出现在残基之间，尝试作为前一个残基的修饰
            if residues:
                raw_mod, new_i = read_mod_token_at(seq, i)
                if raw_mod is not None:
                    aa, old_mass = residues[-1]
                    residues[-1] = (aa, old_mass + mod_token_to_mass(raw_mod))
                    i = new_i
                    continue

            raise ValueError(f"无法解析序列位置 {i}: {seq[i:]} in {seq}")

        if ch not in AA_MASS:
            raise ValueError(f"未知氨基酸: {ch} in {seq}")

        aa = ch
        mass = AA_MASS[aa]
        i += 1

        # 残基后可跟多个修饰
        while i < n:
            raw_mod, new_i = read_mod_token_at(seq, i)
            if raw_mod is None:
                break
            mass += mod_token_to_mass(raw_mod)
            i = new_i

        residues.append((aa, mass))

    return residues, nterm_shift


# -----------------------------
# Theoretical B/Y ion generation
# -----------------------------
def generate_by_ions_with_mods(seq: str, max_charge: int = 3):
    residues, nterm_shift = parse_sequence_with_mods(seq)
    masses = [m for _, m in residues]
    n = len(masses)

    if n < 2:
        return []

    prefix_sums = []
    s = 0.0

    for m in masses:
        s += m
        prefix_sums.append(s)

    total_mass = prefix_sums[-1] + nterm_shift

    ions = []

    for i in range(1, n):
        b_neutral = prefix_sums[i - 1] + nterm_shift
        y_neutral = (total_mass - b_neutral) + H2O

        for z in range(1, max_charge + 1):
            b_mz = (b_neutral + z * PROTON) / z
            y_mz = (y_neutral + z * PROTON) / z

            # cleavage_index aligns B_i and Y_(n-i) to the same row.
            ions.append({
                "ion_type": "B",
                "ion_number": i,
                "cleavage_index": i,
                "charge": z,
                "mz": b_mz,
            })

            ions.append({
                "ion_type": "Y",
                "ion_number": n - i,
                "cleavage_index": i,
                "charge": z,
                "mz": y_mz,
            })

    return ions


# -----------------------------
# MGF parsing
# -----------------------------
def parse_charge(charge_str):
    m = re.search(r"(\d+)", str(charge_str))
    return int(m.group(1)) if m else None


def iter_mgf_blocks(mgf_path):
    block = []
    in_block = False

    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if s == "BEGIN IONS":
                in_block = True
                block = [line]

            elif in_block:
                block.append(line)

                if s == "END IONS":
                    yield block
                    block = []
                    in_block = False


def parse_ions_block(block_lines):
    seq = None
    precursor_charge = None
    pepmass = 0.0
    rtinseconds = 0.0
    peaks = []

    for line in block_lines:
        s = line.strip()

        if not s:
            continue

        if s.upper().startswith("SEQ="):
            seq = s.split("=", 1)[1].strip()
            continue

        if s.upper().startswith("CHARGE="):
            precursor_charge = parse_charge(s.split("=", 1)[1])
            continue

        if s.upper().startswith("PEPMASS="):
            try:
                pepmass = float(s.split("=", 1)[1].strip().split()[0])
            except ValueError:
                pepmass = 0.0
            continue

        if s.upper().startswith("RTINSECONDS="):
            try:
                rtinseconds = float(s.split("=", 1)[1].strip())
            except ValueError:
                rtinseconds = 0.0
            continue

        parts = s.split()

        if len(parts) == 2:
            try:
                mz = float(parts[0])
                intensity = float(parts[1])
                peaks.append((mz, intensity))
            except ValueError:
                pass

    return seq, precursor_charge, pepmass, rtinseconds, peaks


# -----------------------------
# Peak matching
# -----------------------------
def find_matching_peaks(sorted_peaks, target_mz, tol=0.05):
    mzs = [p[0] for p in sorted_peaks]

    left = bisect_left(mzs, target_mz - tol)
    right = bisect_right(mzs, target_mz + tol)

    return sorted_peaks[left:right]


# -----------------------------
# Input matrix: theoretical m/z
# -----------------------------
def build_input_mz_matrix(seq):
    """
    Build theoretical input matrix.

    Shape:
        (39, 46, 6)

    Channel order:
        0: B1+
        1: B2+
        2: B3+
        3: Y1+
        4: Y2+
        5: Y3+

    Each value:
        theoretical BY m/z + token

    Example:
        B1, charge 1, token 0   -> B1 m/z
        B1, charge 1, token -17 -> B1 m/z - 17
    """
    residues, _ = parse_sequence_with_mods(seq)
    peptide_len = len(residues)

    if peptide_len < 2:
        raise ValueError(f"Peptide too short: {seq}")

    if peptide_len > MAX_PEPTIDE_LEN:
        raise ValueError(f"Peptide length {peptide_len} > {MAX_PEPTIDE_LEN}: {seq}")

    input_mz = np.zeros(
        (MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS),
        dtype=np.float32
    )

    ions = generate_by_ions_with_mods(seq, max_charge=3)

    for ion in ions:
        row_idx = ion["cleavage_index"] - 1
        z = ion["charge"]

        if row_idx >= MAX_CLEAVAGE_LEN:
            continue

        if ion["ion_type"] == "B":
            channel_idx = z - 1
        else:
            channel_idx = z + 2

        tokens = CHARGE_TOKENS[z]

        for token_idx, token in enumerate(tokens):
            input_mz[row_idx, token_idx, channel_idx] = ion["mz"] + token

    return input_mz


# -----------------------------
# Target matrix: real intensity
# -----------------------------
def build_target_intensity_matrix(seq, precursor_charge, peaks, tol=0.05,
                                  sequence_constrained=None):
    """
    Build target intensity matrix.

    Shape:
        (39, 46, 6)

    Values:
        matched experimental intensity.

    It follows the original token-matching rule:
        target_mz = theoretical BY m/z + token
        if experimental peak exists within ±tol, fill intensity.

    If multiple peaks match one cell, use max intensity.

    sequence_constrained:
        None（默认）保持原行为：纯 m/z 窗口匹配。
        "qk"   只对 Q/K 四列做序列约束（消除与 |Q-K|=0.036 < tol 有关的唯一歧义）。
        "full" 对所有残基偏移列都要求 m/z 与同通道内某个真实 b/y 离子重合。
        详见 sequence_token_map.py；基础列（main + 7 卫星）始终不受影响。
    """
    residues, _ = parse_sequence_with_mods(seq)
    peptide_len = len(residues)

    if peptide_len < 2:
        raise ValueError(f"Peptide too short: {seq}")

    if peptide_len > MAX_PEPTIDE_LEN:
        raise ValueError(f"Peptide length {peptide_len} > {MAX_PEPTIDE_LEN}: {seq}")

    target = np.zeros(
        (MAX_CLEAVAGE_LEN, NUM_TOKENS, NUM_CHANNELS),
        dtype=np.float32
    )

    max_frag_charge = min(3, precursor_charge)
    ions = generate_by_ions_with_mods(seq, max_charge=max_frag_charge)
    sorted_peaks = sorted(peaks, key=lambda x: x[0])

    for ion in ions:
        row_idx = ion["cleavage_index"] - 1
        z = ion["charge"]

        if row_idx >= MAX_CLEAVAGE_LEN:
            continue

        if ion["ion_type"] == "B":
            channel_idx = z - 1
        else:
            channel_idx = z + 2

        tokens = CHARGE_TOKENS[z]

        for token_idx, token in enumerate(tokens):
            target_mz = ion["mz"] + token
            matched = find_matching_peaks(sorted_peaks, target_mz, tol=tol)

            if matched:
                target[row_idx, token_idx, channel_idx] = max(
                    intensity for _, intensity in matched
                )

    if sequence_constrained:
        from sequence_token_map import apply_mask
        target = apply_mask(target, seq, mode=sequence_constrained)

    return target


# -----------------------------
# Normalization for target intensity
# -----------------------------
def normalize_target_intensity(matrix, method="sqrt_clip", clip_value=50000.0, eps=1e-8):
    """
    Normalize target intensity matrix.

    method:
        "none":
            no normalization

        "max":
            x / max(x)

        "sqrt":
            sqrt(x) / max(sqrt(x))

        "log":
            log1p(x) / max(log1p(x))

        "sqrt_clip":
            x_clipped = min(x, clip_value)
            normalized = sqrt(x_clipped) / sqrt(clip_value)

    推荐使用:
        method="sqrt_clip", clip_value=50000.0
    """

    matrix = matrix.astype(np.float32)

    if method == "none":
        return matrix, np.float32(1.0)

    if method == "max":
        scale_value = matrix.max()

        if scale_value <= 0:
            return matrix.astype(np.float32), np.float32(0.0)

        normalized = matrix / (scale_value + eps)
        return normalized.astype(np.float32), np.float32(scale_value)

    if method == "sqrt":
        transformed = np.sqrt(matrix)
        scale_value = transformed.max()

        if scale_value <= 0:
            return transformed.astype(np.float32), np.float32(0.0)

        normalized = transformed / (scale_value + eps)
        return normalized.astype(np.float32), np.float32(scale_value)

    if method == "log":
        transformed = np.log1p(matrix)
        scale_value = transformed.max()

        if scale_value <= 0:
            return transformed.astype(np.float32), np.float32(0.0)

        normalized = transformed / (scale_value + eps)
        return normalized.astype(np.float32), np.float32(scale_value)

    if method == "sqrt_clip":
        clipped = np.minimum(matrix, clip_value)
        transformed = np.sqrt(clipped)

        scale_value = np.sqrt(np.float32(clip_value))
        normalized = transformed / (scale_value + eps)

        return normalized.astype(np.float32), np.float32(scale_value)

    raise ValueError(f"Unknown normalization method: {method}")


# -----------------------------
# Masks
# -----------------------------
def build_position_mask(seq):
    residues, _ = parse_sequence_with_mods(seq)
    length = len(residues) - 1

    mask = np.zeros((MAX_CLEAVAGE_LEN,), dtype=np.float32)
    mask[:length] = 1.0

    return mask, length


def build_channel_mask(precursor_charge):
    """
    channel order:
        0: B1+
        1: B2+
        2: B3+
        3: Y1+
        4: Y2+
        5: Y3+
    """
    mask = np.zeros((NUM_CHANNELS,), dtype=np.float32)

    max_frag_charge = min(3, precursor_charge)

    for z in range(1, max_frag_charge + 1):
        mask[z - 1] = 1.0
        mask[z + 2] = 1.0

    return mask


# -----------------------------
# Main conversion
# -----------------------------
def build_training_examples_from_average_mgf(
    input_mgf,
    output_npz,
    tol=0.05,
    target_normalization="sqrt_clip",
    clip_intensity=50000.0,
    max_spectra=None,
):
    input_mz_matrices = []
    target_intensity_matrices = []
    position_masks = []
    channel_masks = []
    lengths = []
    charges = []
    pepmasses = []
    rtinseconds_list = []
    seqs = []
    target_scale_values = []

    total = 0
    used = 0
    skipped_invalid = 0
    skipped_long = 0
    skipped_error = 0

    for block in iter_mgf_blocks(input_mgf):
        total += 1

        if max_spectra is not None and used >= max_spectra:
            break

        seq, charge, pepmass, rtinseconds, peaks = parse_ions_block(block)

        if seq is None or charge is None or len(peaks) == 0:
            skipped_invalid += 1
            continue

        try:
            residues, _ = parse_sequence_with_mods(seq)
            peptide_len = len(residues)

            if peptide_len > MAX_PEPTIDE_LEN:
                skipped_long += 1
                continue

            input_mz = build_input_mz_matrix(seq)

            target = build_target_intensity_matrix(
                seq=seq,
                precursor_charge=charge,
                peaks=peaks,
                tol=tol,
            )

            target, scale_value = normalize_target_intensity(
                target,
                method=target_normalization,
                clip_value=clip_intensity,
            )

            position_mask, length = build_position_mask(seq)
            channel_mask = build_channel_mask(charge)

        except Exception as e:
            skipped_error += 1
            print(f"[Skip] spectrum {total} failed: {e}")
            continue

        input_mz_matrices.append(input_mz)
        target_intensity_matrices.append(target)
        position_masks.append(position_mask)
        channel_masks.append(channel_mask)
        lengths.append(length)
        charges.append(charge)
        pepmasses.append(pepmass)
        rtinseconds_list.append(rtinseconds)
        seqs.append(seq)
        target_scale_values.append(scale_value)

        used += 1

        if used % 1000 == 0:
            print(f"Converted examples: {used}")

    if used == 0:
        raise RuntimeError("No valid training examples were built.")

    input_mz_matrices = np.stack(input_mz_matrices, axis=0).astype(np.float32)
    target_intensity_matrices = np.stack(target_intensity_matrices, axis=0).astype(np.float32)
    position_masks = np.stack(position_masks, axis=0).astype(np.float32)
    channel_masks = np.stack(channel_masks, axis=0).astype(np.float32)

    lengths = np.array(lengths, dtype=np.int32)
    charges = np.array(charges, dtype=np.int32)
    pepmasses = np.array(pepmasses, dtype=np.float32)
    rtinseconds = np.array(rtinseconds_list, dtype=np.float32)
    seqs = np.array(seqs, dtype=object)
    target_scale_values = np.array(target_scale_values, dtype=np.float32)

    np.savez_compressed(
        output_npz,
        input_mz_matrices=input_mz_matrices,
        target_intensity_matrices=target_intensity_matrices,
        position_masks=position_masks,
        channel_masks=channel_masks,
        lengths=lengths,
        charges=charges,
        pepmasses=pepmasses,
        rtinseconds=rtinseconds,
        seqs=seqs,
        target_scale_values=target_scale_values,
        channel_names=np.array(CHANNEL_NAMES, dtype=object),
        tokens_charge1=np.array(CHARGE_TOKENS[1], dtype=np.float32),
        tokens_charge2=np.array(CHARGE_TOKENS[2], dtype=np.float32),
        tokens_charge3=np.array(CHARGE_TOKENS[3], dtype=np.float32),
        max_peptide_len=np.array(MAX_PEPTIDE_LEN, dtype=np.int32),
        max_cleavage_len=np.array(MAX_CLEAVAGE_LEN, dtype=np.int32),
        target_normalization=np.array(target_normalization, dtype=object),
        clip_intensity=np.array(clip_intensity, dtype=np.float32),
        mz_unit=np.array("Da", dtype=object),
    )

    print("\nDone.")
    print(f"Input MGF: {input_mgf}")
    print(f"Output NPZ: {output_npz}")
    print(f"Total spectra: {total}")
    print(f"Used examples: {used}")
    print(f"Skipped invalid: {skipped_invalid}")
    print(f"Skipped long peptides: {skipped_long}")
    print(f"Skipped errors: {skipped_error}")
    print(f"input_mz_matrices shape: {input_mz_matrices.shape}")
    print(f"target_intensity_matrices shape: {target_intensity_matrices.shape}")
    print(f"position_masks shape: {position_masks.shape}")
    print(f"channel_masks shape: {channel_masks.shape}")


if __name__ == "__main__":
    input_mgf = r"D:\2026MS_SpectrumGeneration\CasanovoV2\Filtered_spectrum\all_species_average_training_examples.mgf"
    output_npz = r"D:\2026MS_SpectrumGeneration\CasanovoV2\Filtered_spectrum\all_species_average_training_examples_len40_sqrtclip50000.npz"

    build_training_examples_from_average_mgf(
        input_mgf=input_mgf,
        output_npz=output_npz,
        tol=0.05,
        target_normalization="sqrt_clip",
        clip_intensity=50000.0,
        max_spectra=None,
    )
