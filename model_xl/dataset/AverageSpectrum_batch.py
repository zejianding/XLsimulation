# merge_filtered_and_average_all_species.py

import os
import re
from collections import defaultdict


# ============================================================
# Paths
# ============================================================

INPUT_DIR = r"D:\2026MS_SpectrumGeneration\CasanovoV2\Filtered_spectrum"

MERGED_MGF = os.path.join(
    INPUT_DIR,
    "all_species_filtered_by_tokens.mgf"
)

AVERAGED_MGF = os.path.join(
    INPUT_DIR,
    "all_species_average_training_examples.mgf"
)

TOL = 0.1
GROUP_BY_CHARGE = True


# ============================================================
# MGF merge
# ============================================================

def find_filtered_mgf_files(input_dir):
    """
    找到当前目录下所有 filtered_by_tokens_*.mgf 文件。
    """
    mgf_files = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".mgf") and filename.startswith("filtered_by_tokens_"):
            mgf_files.append(os.path.join(input_dir, filename))

    mgf_files = sorted(mgf_files)
    return mgf_files


def merge_mgf_files(mgf_files, output_mgf):
    """
    streaming 合并多个 MGF 文件。
    不一次性读入内存。
    """
    total_lines = 0
    total_files = 0

    with open(output_mgf, "w", encoding="utf-8") as fout:
        for mgf_path in mgf_files:
            total_files += 1
            print(f"[Merge {total_files}/{len(mgf_files)}] {mgf_path}")

            with open(mgf_path, "r", encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    fout.write(line)
                    total_lines += 1

                # 保证不同文件之间有空行分隔
                fout.write("\n")

    print("\nMerge finished.")
    print(f"Input MGF files: {len(mgf_files)}")
    print(f"Output merged MGF: {output_mgf}")
    print(f"Total lines written: {total_lines}")


# ============================================================
# AverageSpectrum logic
# copied from your AverageSpectrum.py, with minor path wrapping
# ============================================================

def parse_charge(charge_str):
    if charge_str is None:
        return None
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


def parse_mgf_block(block_lines):
    seq = None
    charge = None
    pepmass = None
    rtinseconds = None
    title = None
    scans = None

    peaks = []

    for line in block_lines:
        s = line.strip()

        if not s:
            continue

        if s == "BEGIN IONS" or s == "END IONS":
            continue

        if s.upper().startswith("SEQ="):
            seq = s.split("=", 1)[1].strip()
            continue

        if s.upper().startswith("CHARGE="):
            charge = parse_charge(s.split("=", 1)[1].strip())
            continue

        if s.upper().startswith("PEPMASS="):
            try:
                pepmass = float(s.split("=", 1)[1].strip().split()[0])
            except ValueError:
                pepmass = None
            continue

        if s.upper().startswith("RTINSECONDS="):
            try:
                rtinseconds = float(s.split("=", 1)[1].strip())
            except ValueError:
                rtinseconds = None
            continue

        if s.upper().startswith("TITLE="):
            title = s.split("=", 1)[1].strip()
            continue

        if s.upper().startswith("SCANS="):
            scans = s.split("=", 1)[1].strip()
            continue

        parts = s.split()

        if len(parts) == 2:
            try:
                mz = float(parts[0])
                intensity = float(parts[1])
                peaks.append((mz, intensity))
                continue
            except ValueError:
                pass

    return {
        "seq": seq,
        "charge": charge,
        "pepmass": pepmass,
        "rtinseconds": rtinseconds,
        "title": title,
        "scans": scans,
        "peaks": peaks,
        "block_lines": block_lines,
    }


def get_group_key(item, group_by_charge=True):
    if group_by_charge:
        return (item["seq"], item["charge"])
    return (item["seq"],)


def average_peaks_by_tolerance(spectra_peaks, tol=0.05):
    """
    多条谱按 mz tolerance 计算平均谱。

    规则：
        A: 187.00 1000
        B: 187.05 500
        tol=0.05
        输出: 187.00 750

        A: 133.00 1000
        B: missing
        输出: 133.00 500
    """
    n_spectra = len(spectra_peaks)

    if n_spectra == 0:
        return []

    all_peaks = []

    for spec_idx, peaks in enumerate(spectra_peaks):
        for mz, intensity in peaks:
            all_peaks.append((mz, intensity, spec_idx))

    if not all_peaks:
        return []

    all_peaks.sort(key=lambda x: x[0])

    averaged_peaks = []
    i = 0
    n = len(all_peaks)

    while i < n:
        anchor_mz = all_peaks[i][0]

        per_spectrum_intensity = {}

        j = i
        while j < n and all_peaks[j][0] <= anchor_mz + tol:
            mz, intensity, spec_idx = all_peaks[j]

            if spec_idx not in per_spectrum_intensity:
                per_spectrum_intensity[spec_idx] = intensity
            else:
                if intensity > per_spectrum_intensity[spec_idx]:
                    per_spectrum_intensity[spec_idx] = intensity

            j += 1

        avg_intensity = sum(per_spectrum_intensity.values()) / n_spectra
        averaged_peaks.append((anchor_mz, avg_intensity))

        i = j

    return averaged_peaks


def average_metadata(items):
    seq = items[0]["seq"]
    charge = items[0]["charge"]

    pepmasses = [x["pepmass"] for x in items if x["pepmass"] is not None]
    rts = [x["rtinseconds"] for x in items if x["rtinseconds"] is not None]

    avg_pepmass = sum(pepmasses) / len(pepmasses) if pepmasses else None
    avg_rt = sum(rts) / len(rts) if rts else None

    return {
        "seq": seq,
        "charge": charge,
        "pepmass": avg_pepmass,
        "rtinseconds": avg_rt,
        "num_spectra": len(items),
    }


def rebuild_average_block(meta, peaks):
    out = []

    out.append("BEGIN IONS\n")
    out.append("SOURCE=AVERAGED_ALL_SPECIES\n")
    out.append(f"TITLE=AVERAGED_SEQ_CHARGE_COUNT_{meta['num_spectra']}\n")

    if meta["pepmass"] is not None:
        out.append(f"PEPMASS={meta['pepmass']:.6f}\n")

    if meta["rtinseconds"] is not None:
        out.append(f"RTINSECONDS={meta['rtinseconds']:.6f}\n")

    if meta["charge"] is not None:
        out.append(f"CHARGE={meta['charge']}+\n")

    out.append(f"SEQ={meta['seq']}\n")
    out.append(f"COMMENT=AVERAGE_SPECTRUM_ALL_SPECIES num_spectra={meta['num_spectra']} peaks={len(peaks)}\n")

    for mz, intensity in peaks:
        out.append(f"{mz:.6f} {intensity:.6f}\n")

    out.append("END IONS\n")
    out.append("\n")

    return out


def rebuild_single_block(item):
    """
    只有一条谱的 SEQ+CHARGE group，直接转录。
    为了避免重复 SOURCE，这里会替换已有 SOURCE。
    """
    out = []
    inserted_source = False

    for line in item["block_lines"]:
        s = line.strip()

        if s.upper().startswith("SOURCE="):
            if not inserted_source:
                out.append("SOURCE=SINGLE_COPY_ALL_SPECIES\n")
                inserted_source = True
            continue

        out.append(line if line.endswith("\n") else line + "\n")

        if s == "BEGIN IONS" and not inserted_source:
            out.append("SOURCE=SINGLE_COPY_ALL_SPECIES\n")
            inserted_source = True

    out.append("\n")
    return out


def build_average_mgf(
    input_mgf,
    output_mgf,
    tol=0.05,
    group_by_charge=True,
):
    groups = defaultdict(list)

    total_blocks = 0
    skipped_no_seq = 0
    skipped_no_charge = 0

    print(f"\nReading merged MGF: {input_mgf}")

    for block in iter_mgf_blocks(input_mgf):
        total_blocks += 1

        item = parse_mgf_block(block)

        if item["seq"] is None:
            skipped_no_seq += 1
            continue

        if group_by_charge and item["charge"] is None:
            skipped_no_charge += 1
            continue

        key = get_group_key(item, group_by_charge=group_by_charge)
        groups[key].append(item)

        if total_blocks % 10000 == 0:
            print(f"Read {total_blocks} spectra... groups={len(groups)}")

    print("\nGrouping finished.")
    print(f"Total spectra blocks: {total_blocks}")
    print(f"Skipped no SEQ: {skipped_no_seq}")
    print(f"Skipped no CHARGE: {skipped_no_charge}")
    print(f"Total SEQ+CHARGE groups: {len(groups)}")

    single_groups = 0
    averaged_groups = 0
    output_spectra = 0

    with open(output_mgf, "w", encoding="utf-8") as fout:
        for idx, key in enumerate(sorted(groups.keys()), start=1):
            items = groups[key]

            if len(items) == 1:
                fout.writelines(rebuild_single_block(items[0]))
                single_groups += 1
                output_spectra += 1

            else:
                spectra_peaks = [x["peaks"] for x in items]

                avg_peaks = average_peaks_by_tolerance(
                    spectra_peaks=spectra_peaks,
                    tol=tol,
                )

                meta = average_metadata(items)
                fout.writelines(rebuild_average_block(meta, avg_peaks))

                averaged_groups += 1
                output_spectra += 1

            if idx % 1000 == 0:
                print(f"Wrote {idx}/{len(groups)} groups...")

    print("\nAverage finished.")
    print(f"Output averaged MGF: {output_mgf}")
    print(f"Input spectra: {total_blocks}")
    print(f"Output spectra/examples: {output_spectra}")
    print(f"Single-spectrum groups: {single_groups}")
    print(f"Averaged groups: {averaged_groups}")


# ============================================================
# Main
# ============================================================

def main():
    mgf_files = find_filtered_mgf_files(INPUT_DIR)

    print("Filtered MGF files found:")
    for path in mgf_files:
        print("  ", path)

    if len(mgf_files) == 0:
        raise RuntimeError(f"No filtered_by_tokens_*.mgf files found in {INPUT_DIR}")

    merge_mgf_files(
        mgf_files=mgf_files,
        output_mgf=MERGED_MGF,
    )

    build_average_mgf(
        input_mgf=MERGED_MGF,
        output_mgf=AVERAGED_MGF,
        tol=TOL,
        group_by_charge=GROUP_BY_CHARGE,
    )


if __name__ == "__main__":
    main()