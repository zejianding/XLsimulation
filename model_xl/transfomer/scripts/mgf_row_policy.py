from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
import re

from predict_mgf import calculate_pepmass


@dataclass(frozen=True)
class MgfRowPolicy:
    drop_seq_with_o: bool = False
    precursor_mismatch_mode: str = "off"
    precursor_mismatch_ppm: float = 50.0


@dataclass
class MgfPolicyStats:
    total_spectra: int = 0
    kept_spectra: int = 0
    dropped_seq_with_o: int = 0
    dropped_precursor_mismatch: int = 0
    fixed_precursor_mismatch: int = 0


@dataclass(frozen=True)
class MgfRowCatalogEntry:
    ordinal: int
    mgf_path: Path
    byte_offset: int
    scan_raw: str
    scan_key: str
    title: str
    seq: str
    keep: bool
    row_drop_reason: str | None
    precursor_mismatch_corrected: bool


def scan_key(text: str) -> str:
    nums = re.findall(r"\d+", str(text))
    return nums[-1] if nums else str(text)


def parse_charge_from_meta(meta: dict[str, str]) -> int:
    raw = str(meta.get("CHARGE", "2")).strip()
    raw = raw.rstrip("+") if raw.endswith("+") else raw
    try:
        return int(raw)
    except ValueError:
        match = re.search(r"(\d+)", raw)
        return int(match.group(1)) if match else 2


def parse_pepmass_from_meta(meta: dict[str, str]) -> float:
    raw = str(meta.get("PEPMASS", "")).strip()
    if not raw:
        return 0.0
    token = raw.replace(",", " ").split()[0]
    try:
        return float(token)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
        return float(match.group(0)) if match else 0.0


def replace_pepmass_mz(raw: str | None, new_mz: float) -> str:
    tail = ""
    if raw:
        parts = str(raw).strip().replace(",", " ").split()
        if len(parts) > 1:
            tail = " " + " ".join(parts[1:])
    return f"{float(new_mz):.6f}{tail}"


def compute_ground_truth_precursor_ppm(meta: dict[str, str]) -> tuple[float | None, float | None]:
    seq = str(meta.get("SEQ", "")).strip()
    if not seq:
        return None, None

    charge = parse_charge_from_meta(meta)
    exp_pepmass = parse_pepmass_from_meta(meta)
    if charge <= 0 or exp_pepmass <= 0:
        return None, None

    try:
        calc_pepmass = float(calculate_pepmass(seq, charge))
    except Exception:
        return None, None

    ppm = abs(calc_pepmass - exp_pepmass) / exp_pepmass * 1e6
    return ppm, calc_pepmass


def apply_mgf_row_policy(
    meta: dict[str, str],
    policy: MgfRowPolicy | None,
) -> tuple[bool, dict[str, str], dict[str, float | str | bool | None]]:
    if policy is None:
        policy = MgfRowPolicy()

    seq = str(meta.get("SEQ", "")).strip()
    diag: dict[str, float | str | bool | None] = {
        "row_drop_reason": None,
        "contains_seq_o": bool(seq and "O" in seq),
        "original_pepmass": parse_pepmass_from_meta(meta),
        "precursor_ppm_error": None,
        "corrected_pepmass": None,
        "precursor_mismatch_corrected": False,
    }

    if policy.drop_seq_with_o and diag["contains_seq_o"]:
        diag["row_drop_reason"] = "seq_contains_O"
        return False, meta, diag

    ppm_error, calc_pepmass = compute_ground_truth_precursor_ppm(meta)
    diag["precursor_ppm_error"] = ppm_error

    mismatch_mode = str(policy.precursor_mismatch_mode or "off").lower()
    ppm_threshold = float(policy.precursor_mismatch_ppm)
    if (
        mismatch_mode in {"fix", "drop"}
        and ppm_error is not None
        and calc_pepmass is not None
        and ppm_error > ppm_threshold
    ):
        if mismatch_mode == "drop":
            diag["row_drop_reason"] = "precursor_mismatch"
            return False, meta, diag

        fixed_meta = dict(meta)
        fixed_meta["PEPMASS"] = replace_pepmass_mz(meta.get("PEPMASS"), calc_pepmass)
        diag["corrected_pepmass"] = calc_pepmass
        diag["precursor_mismatch_corrected"] = True
        return True, fixed_meta, diag

    return True, meta, diag


def build_mgf_row_catalog(
    mgf_dir: Path,
    policy: MgfRowPolicy | None,
) -> tuple[list[MgfRowCatalogEntry], MgfPolicyStats]:
    mgf_files = sorted(mgf_dir.glob("*.mgf"))
    if not mgf_files:
        raise FileNotFoundError(f"No .mgf files found in {mgf_dir}")

    entries: list[MgfRowCatalogEntry] = []
    stats = MgfPolicyStats()
    ordinal = 0

    for mgf_path in mgf_files:
        with open(mgf_path, "r", encoding="utf-8", errors="ignore") as handle:
            in_block = False
            block_offset = -1
            block: list[str] = []

            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped == "BEGIN IONS":
                    in_block = True
                    block_offset = pos
                    block = [line]
                    continue
                if not in_block:
                    continue
                block.append(line)
                if stripped != "END IONS":
                    continue

                meta: dict[str, str] = {}
                for block_line in block:
                    token = block_line.strip()
                    if "=" in token and not token.upper().startswith(("BEGIN IONS", "END IONS")):
                        key, value = token.split("=", 1)
                        meta[key.upper()] = value.strip()

                keep, fixed_meta, diag = apply_mgf_row_policy(meta, policy)
                stats.total_spectra += 1
                if keep:
                    stats.kept_spectra += 1
                    if diag["precursor_mismatch_corrected"]:
                        stats.fixed_precursor_mismatch += 1
                else:
                    if diag["row_drop_reason"] == "seq_contains_O":
                        stats.dropped_seq_with_o += 1
                    elif diag["row_drop_reason"] == "precursor_mismatch":
                        stats.dropped_precursor_mismatch += 1

                entries.append(
                    MgfRowCatalogEntry(
                        ordinal=ordinal,
                        mgf_path=mgf_path,
                        byte_offset=block_offset,
                        scan_raw=str(meta.get("SCANS", "")),
                        scan_key=scan_key(meta.get("SCANS", meta.get("TITLE", ""))),
                        title=str(meta.get("TITLE", "")),
                        seq=str(fixed_meta.get("SEQ", "")),
                        keep=bool(keep),
                        row_drop_reason=diag["row_drop_reason"],
                        precursor_mismatch_corrected=bool(diag["precursor_mismatch_corrected"]),
                    )
                )
                ordinal += 1
                block = []
                in_block = False

    return entries, stats


def build_occurrence_aligned_pairs(
    row_scan_values: list[str],
    catalog: list[MgfRowCatalogEntry],
) -> tuple[list[tuple[int, MgfRowCatalogEntry]], dict[str, int]]:
    by_scan: dict[str, list[MgfRowCatalogEntry]] = defaultdict(list)
    for entry in catalog:
        by_scan[entry.scan_key].append(entry)

    used = Counter()
    pairs: list[tuple[int, MgfRowCatalogEntry]] = []
    missing_labels = 0
    overflow_reuse = 0
    skipped_dropped = 0

    for row_idx, value in enumerate(row_scan_values):
        key = scan_key(value)
        entries = by_scan.get(key)
        if not entries:
            missing_labels += 1
            continue
        occ_idx = used[key]
        if occ_idx >= len(entries):
            overflow_reuse += 1
            occ_idx = len(entries) - 1
        used[key] += 1
        entry = entries[occ_idx]
        if not entry.keep:
            skipped_dropped += 1
            continue
        pairs.append((row_idx, entry))

    return pairs, {
        "compared_rows": len(pairs),
        "missing_labels": missing_labels,
        "overflow_reuse": overflow_reuse,
        "skipped_dropped_rows": skipped_dropped,
    }


def collect_filtered_row_order_sequences(
    mgf_dir: Path,
    policy: MgfRowPolicy | None,
) -> tuple[list[int], list[str], MgfPolicyStats]:
    catalog, stats = build_mgf_row_catalog(mgf_dir, policy)
    keep_ordinals = [entry.ordinal for entry in catalog if entry.keep]
    seqs = [entry.seq for entry in catalog if entry.keep]
    return keep_ordinals, seqs, stats
