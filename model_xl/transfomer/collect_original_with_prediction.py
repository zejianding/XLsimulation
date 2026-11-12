"""
Collect original spectra from 9-species directory that match predicted MGF entries.

For each unique (sequence, charge) pair in the predicted MGF, this script:
  1. Scans all species subdirectories for matching original spectra (peaks.db.mgf)
  2. Writes each group to a separate MGF file containing:
     - All matched original spectra (SOURCE=ORIGINAL, with species in COMMENT)
     - The predicted spectrum at the end (SOURCE=MODEL_PREDICTION)

Usage:
    python collect_original_with_prediction.py \
        --predicted-mgf predicted_test.mgf \
        --root-dir /path/to/9species/ \
        --output-dir ./Original_vs_Predicted/ \
        --max-groups 1000
"""

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path
from collections import defaultdict


SKIP_DIRS = {
    ".idea",
    "SameSequence",
    "Filtered_spectrum",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_charge(charge_str):
    if charge_str is None:
        return None
    m = re.search(r"(\d+)", str(charge_str))
    return int(m.group(1)) if m else None


def sanitize_filename(name, max_len=80):
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    if not name:
        name = "EMPTY_SEQ"
    return name[:max_len]


def make_group_filename(seq, charge):
    prefix = sanitize_filename(seq, max_len=80)
    seq_hash = hashlib.md5(seq.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_charge{charge}_{seq_hash}.mgf"


def iter_mgf_blocks(mgf_path):
    """Yield MGF blocks (list of lines) from a file."""
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


def parse_mgf_block_basic(block_lines):
    """Extract basic metadata from an MGF block."""
    seq = None
    charge = None
    title = None
    scans = None
    pepmass = None
    rtinseconds = None
    peak_count = 0

    for line in block_lines:
        s = line.strip()
        if not s:
            continue
        upper = s.upper()
        if upper.startswith("SEQ="):
            seq = s.split("=", 1)[1].strip()
        elif upper.startswith("CHARGE="):
            charge = parse_charge(s.split("=", 1)[1].strip())
        elif upper.startswith("TITLE="):
            title = s.split("=", 1)[1].strip()
        elif upper.startswith("SCANS="):
            scans = s.split("=", 1)[1].strip()
        elif upper.startswith("PEPMASS="):
            pepmass = s.split("=", 1)[1].strip()
        elif upper.startswith("RTINSECONDS="):
            rtinseconds = s.split("=", 1)[1].strip()
        else:
            parts = s.split()
            if len(parts) == 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    peak_count += 1
                except ValueError:
                    pass

    return {
        "seq": seq,
        "charge": charge,
        "title": title,
        "scans": scans,
        "pepmass": pepmass,
        "rtinseconds": rtinseconds,
        "peak_count": peak_count,
    }


def replace_or_insert_source(block_lines, source_tag):
    """Replace or insert SOURCE= tag in an MGF block."""
    out = []
    inserted = False

    for line in block_lines:
        s = line.strip()
        if s.upper().startswith("SOURCE="):
            if not inserted:
                out.append(f"SOURCE={source_tag}\n")
                inserted = True
            continue
        out.append(line if line.endswith("\n") else line + "\n")
        if s == "BEGIN IONS" and not inserted:
            out.append(f"SOURCE={source_tag}\n")
            inserted = True

    if not inserted:
        out.insert(0, "BEGIN IONS\n")
        out.insert(1, f"SOURCE={source_tag}\n")

    return out


def write_block_append(path, block_lines):
    """Append an MGF block to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in block_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# Step 1: Read predicted MGF
# ---------------------------------------------------------------------------

def load_predicted_mgf(predicted_mgf, max_groups=None):
    """Read predicted MGF and group by (seq, charge) key."""
    predicted_blocks = {}
    predicted_info = {}

    total = 0
    skipped = 0
    duplicated = 0

    print(f"Reading predicted MGF: {predicted_mgf}")

    for block in iter_mgf_blocks(predicted_mgf):
        total += 1
        info = parse_mgf_block_basic(block)
        seq = info["seq"]
        charge = info["charge"]

        if seq is None or charge is None:
            skipped += 1
            continue

        key = (seq, charge)

        if key in predicted_blocks:
            duplicated += 1
            print(f"[Warning] duplicated predicted key: {key}")

        predicted_blocks[key] = block
        predicted_info[key] = info

        if max_groups is not None and len(predicted_blocks) >= max_groups:
            print(f"Reached max_groups={max_groups}, stop reading predicted MGF.")
            break

    print("Predicted MGF loaded.")
    print(f"  total predicted blocks read: {total}")
    print(f"  valid predicted groups: {len(predicted_blocks)}")
    print(f"  skipped (no seq/charge): {skipped}")
    print(f"  duplicated keys: {duplicated}")

    return predicted_blocks, predicted_info


# ---------------------------------------------------------------------------
# Step 2: Find original MGF files
# ---------------------------------------------------------------------------

def find_original_mgf_files(root_dir, skip_dirs):
    """Find original MGF files from one of three supported layouts.

    Supported root_dir forms:
      1. A single MGF file path
      2. A flat directory containing *.mgf files
      3. A parent directory whose child directories each contain *.mgf files
    """
    root = Path(root_dir)
    mgf_files = []  # list of (group_name, mgf_path)

    if root.is_file():
        if root.suffix.lower() != ".mgf":
            raise ValueError(f"--root-dir points to a file, but it is not .mgf: {root}")
        mgf_files.append((root.stem, str(root)))
    elif root.is_dir():
        flat_mgfs = sorted(root.glob("*.mgf"))
        if flat_mgfs:
            for mgf_path in flat_mgfs:
                mgf_files.append((mgf_path.stem, str(mgf_path)))
        else:
            for item in sorted(root.iterdir()):
                if not item.is_dir():
                    continue
                if item.name in skip_dirs:
                    continue

                species_mgfs = sorted(item.glob("*.mgf"))
                if not species_mgfs:
                    print(f"[Skip] no .mgf files in: {item}")
                    continue

                for mgf_path in species_mgfs:
                    mgf_files.append((item.name, str(mgf_path)))
    else:
        raise FileNotFoundError(f"--root-dir not found: {root_dir}")

    print(f"\nOriginal MGF files found: {len(mgf_files)}")
    from collections import Counter
    species_counts = Counter(name for name, _ in mgf_files)
    for species, count in sorted(species_counts.items()):
        print(f"  {species}: {count} mgf file(s)")
    if len(mgf_files) > 20:
        print(f"  ... ({len(mgf_files)} total files)")

    return mgf_files


# ---------------------------------------------------------------------------
# Step 3 & 4: Match & write groups
# ---------------------------------------------------------------------------

def collect_original_and_prediction(
    root_dir,
    predicted_mgf,
    output_dir,
    skip_dirs=None,
    max_groups=None,
):
    if skip_dirs is None:
        skip_dirs = SKIP_DIRS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predicted spectra
    predicted_blocks, predicted_info = load_predicted_mgf(
        predicted_mgf, max_groups=max_groups
    )
    target_keys = set(predicted_blocks.keys())

    if len(target_keys) == 0:
        raise RuntimeError("No valid predicted spectra found.")

    # Find original MGFs
    mgf_files = find_original_mgf_files(root_dir, skip_dirs)

    # Build output path mapping and clear old files
    key_to_filename = {}
    key_to_output_path = {}
    key_to_original_count = defaultdict(int)
    key_to_species_count = defaultdict(lambda: defaultdict(int))

    for seq, charge in target_keys:
        filename = make_group_filename(seq, charge)
        out_path = output_dir / filename
        key_to_filename[(seq, charge)] = filename
        key_to_output_path[(seq, charge)] = out_path
        if out_path.exists():
            out_path.unlink()

    total_original_blocks = 0
    matched_original_blocks = 0

    # Scan each species MGF
    for species_idx, (species_name, mgf_path) in enumerate(mgf_files, start=1):
        print(f"\n[{species_idx}/{len(mgf_files)}] Scanning: {species_name}")
        print(f"  path = {mgf_path}")

        species_total = 0
        species_matched = 0

        for block in iter_mgf_blocks(mgf_path):
            total_original_blocks += 1
            species_total += 1

            info = parse_mgf_block_basic(block)
            seq = info["seq"]
            charge = info["charge"]

            if seq is None or charge is None:
                continue

            key = (seq, charge)
            if key not in target_keys:
                continue

            out_path = key_to_output_path[key]

            # Tag as ORIGINAL; species info stored in COMMENT for traceability
            original_block = replace_or_insert_source(block, "ORIGINAL")

            # Insert species info as COMMENT after SOURCE line
            has_species_comment = any(
                line.strip().upper().startswith("COMMENT=") and "species:" in line.lower()
                for line in original_block
            )
            if not has_species_comment:
                out_lines = []
                for line in original_block:
                    out_lines.append(line)
                    if line.strip().upper().startswith("SOURCE="):
                        out_lines.append(f"COMMENT=species:{species_name}\n")
                original_block = out_lines

            write_block_append(out_path, original_block)

            key_to_original_count[key] += 1
            key_to_species_count[key][species_name] += 1

            matched_original_blocks += 1
            species_matched += 1

            if species_matched % 1000 == 0:
                print(f"  matched in this species: {species_matched}")

        print(f"  species total blocks: {species_total}, matched: {species_matched}")

    # ------------------------------------------------------------------
    # Append predicted spectrum to each group file
    # ------------------------------------------------------------------
    print("\nAppending predicted spectra as MODEL_PREDICTION...")

    for key, pred_block in predicted_blocks.items():
        out_path = key_to_output_path[key]
        pred_block = replace_or_insert_source(pred_block, "MODEL_PREDICTION")
        write_block_append(out_path, pred_block)

    # ------------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------------
    map_csv = output_dir / "group_map.csv"
    missing_txt = output_dir / "missing_original_matches.txt"

    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "sequence", "charge",
            "original_count", "predicted_peak_count",
            "species_counts", "output_path",
        ])
        for key in sorted(target_keys):
            seq, charge = key
            pred_info = predicted_info[key]
            species_counts = ";".join(
                f"{sp}:{cnt}"
                for sp, cnt in sorted(key_to_species_count[key].items())
            )
            writer.writerow([
                key_to_filename[key],
                seq,
                charge,
                key_to_original_count[key],
                pred_info["peak_count"],
                species_counts,
                str(key_to_output_path[key]),
            ])

    with open(missing_txt, "w", encoding="utf-8") as f:
        f.write("Predicted spectra with no matched original spectra\n")
        f.write("=" * 100 + "\n\n")
        f.write("sequence\tcharge\tpredicted_peak_count\toutput_file\n")
        missing = 0
        for key in sorted(target_keys):
            if key_to_original_count[key] == 0:
                missing += 1
                seq, charge = key
                pred_info = predicted_info[key]
                f.write(
                    f"{seq}\t{charge}\t"
                    f"{pred_info['peak_count']}\t"
                    f"{key_to_output_path[key]}\n"
                )
        f.write(f"\nTotal missing original groups: {missing}\n")

    print("\nDone.")
    print(f"Output directory: {output_dir}")
    print(f"Target predicted groups: {len(target_keys)}")
    print(f"Total original spectra scanned: {total_original_blocks}")
    print(f"Matched original spectra: {matched_original_blocks}")
    print(f"Group map CSV: {map_csv}")
    print(f"Missing original report: {missing_txt}")

    no_original = sum(1 for k in target_keys if key_to_original_count[k] == 0)
    print(f"Groups with no matched original spectra: {no_original}")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect original spectra matching predicted MGF entries, "
                    "and create per-group MGF files for PiPrime embedding evaluation."
    )
    parser.add_argument("--predicted-mgf", required=True,
                        help="Path to the predicted MGF file (output of predict_npz_to_mgf.py)")
    parser.add_argument("--root-dir", required=True,
                        help="Original spectra source. Supports: "
                             "(1) a single .mgf file, "
                             "(2) a flat directory of .mgf files, or "
                             "(3) a directory of species subdirectories with .mgf files.")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for per-group MGF files")
    parser.add_argument("--max-groups", type=int, default=None,
                        help="Limit to first N groups (for testing)")
    args = parser.parse_args()

    collect_original_and_prediction(
        root_dir=args.root_dir,
        predicted_mgf=args.predicted_mgf,
        output_dir=args.output_dir,
        max_groups=args.max_groups,
    )


if __name__ == "__main__":
    main()
