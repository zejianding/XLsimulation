import lzma
import os
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

# -----------------------------
# 基本常数 / token 表：统一从 <project_root>/token_table.py 导入
# -----------------------------
_project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "token_table.py").exists()
)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from token_table import (  # noqa: E402
    AA_MASS,
    CHARGE_TOKENS,
    H2O,
    PROTON,
)

# -----------------------------
# 修饰质量：MassiveKB 数字质量格式
# -----------------------------
# 你统计到的 Top raw modification tokens:
#   +57.021
#   +15.995
#   +0.984
#   +43.006
#   +42.011
#   -17.027
#   +43.006-17.027
#
# 注意：
# 1) 这里不再使用旧格式，例如 M(+15.99), C(+57.02), Q(+.98)。
# 2) 解析时不依赖固定 key，而是直接从 SEQ 中读取数字质量。
# 3) +43.006-17.027 这种连续 signed mass 会被当成一个组合修饰，
#    净质量 = 43.006 - 17.027 = 25.979。
COMMON_MASSIVEKB_MODS = {
    "+57.021": 57.021,
    "+15.995": 15.995,
    "+0.984": 0.984,
    "+43.006": 43.006,
    "+42.011": 42.011,
    "-17.027": -17.027,
    "+43.006-17.027": 43.006 - 17.027,
}

# 一个单独的质量数字，可以有正负号。
# 例子：+57.021, +15.995, +0.984, -17.027, 42.011, .984
SINGLE_MASS_PATTERN = r"[+-]?(?:\d+\.\d+|\d+|\.\d+)"

# 一个修饰 token：
# - 普通修饰：+57.021
# - 组合修饰：+43.006-17.027
# 后续的组合部分必须带 + 或 -，这样 +43.006-17.027 会整体读成一个 token。
MOD_TOKEN_RE = re.compile(
    rf"{SINGLE_MASS_PATTERN}(?:[+-](?:\d+\.\d+|\d+|\.\d+))*"
)
MOD_COMPONENT_RE = re.compile(SINGLE_MASS_PATTERN)


def mod_token_to_mass(mod_token: str) -> float:
    """
    将一个 MassiveKB 数字修饰 token 转成净质量。

    Examples:
        +57.021 -> 57.021
        -17.027 -> -17.027
        +43.006-17.027 -> 25.979
        (+42.011) -> 42.011
        [+42.011] -> 42.011
    """
    token = str(mod_token).strip()

    # 去掉一层常见括号。
    if len(token) >= 2 and ((token[0] == "(" and token[-1] == ")") or (token[0] == "[" and token[-1] == "]")):
        token = token[1:-1].strip()

    components = MOD_COMPONENT_RE.findall(token)
    if not components:
        raise ValueError(f"修饰 token 中没有可解析质量: {mod_token}")

    return sum(float(x) for x in components)


def read_mod_token_at(seq: str, i: int):
    """
    尝试从 seq[i:] 读取一个修饰 token。

    支持：
        (+42.011)
        [+42.011]
        (+43.006-17.027)
        M+15.995
        C+57.021
        +42.011PEPTIDE   # N端裸写法
        -17.027PEPTIDE   # N端裸写法

    Return:
        (raw_token, new_i) 或 (None, i)
    """
    n = len(seq)
    if i >= n:
        return None, i

    # 1) 括号/方括号包裹的数字质量修饰
    if seq[i] in "([":
        close_char = ")" if seq[i] == "(" else "]"
        j = seq.find(close_char, i + 1)
        if j == -1:
            raise ValueError(f"修饰括号未闭合: {seq[i:]} in {seq}")

        inner = seq[i + 1:j].strip()
        if inner and any(ch.isdigit() for ch in inner) and MOD_TOKEN_RE.fullmatch(inner):
            return inner, j + 1

        # 如果括号里不是数字修饰，就不在这里消费。
        return None, i

    # 2) 裸写数字质量修饰，例如 +15.995, -17.027, +43.006-17.027
    m = MOD_TOKEN_RE.match(seq, i)
    if m and any(ch.isdigit() for ch in m.group(0)):
        return m.group(0), m.end()

    return None, i

# -----------------------------
# Token 表由 token_table.py 统一提供（46 列，1+ 精确质量差定义，
# 内部按 fragment charge 自动缩放：z=2 除以 2，z=3 除以 3）。
# 见文件顶部的 CHARGE_TOKENS / BASE_MASS_TOKENS 导入。
# -----------------------------


def parse_charge(charge_str: str):
    if charge_str is None:
        return None
    m = re.search(r"(\d+)", str(charge_str))
    return int(m.group(1)) if m else None


def parse_sequence_with_mods(seq: str):
    """
    解析 MassiveKB 风格的 SEQ，返回：
        residues: [(AA, residue_mass_with_internal_mods), ...]
        nterm_shift: N端修饰质量

    支持的修饰写法包括：
        M+15.995
        M(+15.995)
        C+57.021
        C[+57.021]
        N+0.984
        Q+0.984
        (+42.011)PEPTIDE
        +42.011PEPTIDE
        A(+42.011)CDE
        A+42.011CDE
        (+43.006-17.027)PEPTIDE
        A+43.006-17.027CDE

    关键点：
        只要 SEQ 中出现数字质量 token，就按修饰读取；
        +43.006-17.027 这种连续质量被当成一个组合修饰，净质量加入。
    """
    residues = []
    nterm_shift = 0.0

    i = 0
    n = len(seq)

    # -----------------------------
    # 1) 读取 N 端修饰
    # -----------------------------
    # 在第一个氨基酸之前出现的数字修饰，全部作为 nterm_shift。
    while i < n:
        ch = seq[i]

        # 第一个大写氨基酸开始后，N端修饰结束。
        if ch.isalpha() and ch.isupper():
            break

        raw_mod, new_i = read_mod_token_at(seq, i)
        if raw_mod is not None:
            nterm_shift += mod_token_to_mass(raw_mod)
            i = new_i
            continue

        # 忽略一些分隔符，避免格式中有点号/下划线导致直接失败。
        if ch in ".-_ ":
            i += 1
            continue

        raise ValueError(f"无法解析 N端 序列位置 {i}: {seq[i:]} in {seq}")

    # -----------------------------
    # 2) 读取氨基酸和残基后修饰
    # -----------------------------
    while i < n:
        ch = seq[i]

        if not (ch.isalpha() and ch.isupper()):
            raw_mod, new_i = read_mod_token_at(seq, i)
            if raw_mod is not None and residues:
                aa, old_mass = residues[-1]
                residues[-1] = (aa, old_mass + mod_token_to_mass(raw_mod))
                i = new_i
                continue
            # 如果出现在两个残基之间但不是大写字母，尝试当成“前一个残基的修饰”。
            # 这主要处理一些非标准但仍然数字可解析的写法。
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

        # 一个残基后面可能连续跟多个修饰，例如 M+15.995+0.984，
        # 或者 A(+43.006-17.027)。
        while i < n:
            raw_mod, new_i = read_mod_token_at(seq, i)
            if raw_mod is None:
                break

            mass += mod_token_to_mass(raw_mod)
            i = new_i

        residues.append((aa, mass))

    return residues, nterm_shift



def generate_by_ions_with_mods(seq: str, max_charge: int):
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
        y_neutral = (total_mass - (prefix_sums[i - 1] + nterm_shift)) + H2O

        for z in range(1, max_charge + 1):
            b_mz = (b_neutral + z * PROTON) / z
            y_mz = (y_neutral + z * PROTON) / z

            ions.append({
                "ion_name": f"B{i}",
                "ion_type": "B",
                "ion_number": i,
                "charge": z,
                "mz": b_mz,
            })
            ions.append({
                "ion_name": f"Y{n - i}",
                "ion_type": "Y",
                "ion_number": n - i,
                "charge": z,
                "mz": y_mz,
            })

    return ions


def split_into_blocks(lines):
    blocks = []
    current_block = []
    in_block = False

    for line in lines:
        s = line.strip()
        if s == "BEGIN IONS":
            in_block = True
            current_block = []
        elif s == "END IONS":
            in_block = False
            blocks.append(current_block)
        else:
            if in_block:
                current_block.append(line)

    return blocks


def parse_ions_block(block_lines):
    seq = None
    precursor_charge = None
    headers = []
    peaks = []

    for line in block_lines:
        s = line.strip()

        if not s:
            continue

        # 关键修正：
        # streaming block 已经包含 BEGIN IONS / END IONS，
        # 这里必须跳过，否则 rebuild 时会产生嵌套 MGF block。
        if s == "BEGIN IONS" or s == "END IONS":
            continue

        if s.startswith("SEQ="):
            seq = s.split("=", 1)[1].strip()
            headers.append(line.rstrip("\n"))
            continue

        if s.upper().startswith("CHARGE="):
            precursor_charge = parse_charge(s.split("=", 1)[1].strip())
            headers.append(line.rstrip("\n"))
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

        headers.append(line.rstrip("\n"))

    return seq, precursor_charge, headers, peaks


def find_matching_peaks(sorted_peaks, target_mz, tol=0.05):
    mzs = [p[0] for p in sorted_peaks]
    left = bisect_left(mzs, target_mz - tol)
    right = bisect_right(mzs, target_mz + tol)
    return sorted_peaks[left:right]


def process_one_spectrum(seq, precursor_charge, peaks, tol=0.05):
    """和原始逻辑完全一致，仅做性能优化：m/z列表提前提取、二分查找内联、本地引用缓存"""
    if seq is None or precursor_charge is None or len(peaks) == 0:
        return []

    max_frag_charge = min(3, precursor_charge)
    theoretical_ions = generate_by_ions_with_mods(seq, max_charge=max_frag_charge)
    sorted_peaks = sorted(peaks, key=lambda x: x[0])
    peak_mzs = [p[0] for p in sorted_peaks]
    n_peaks = len(peak_mzs)

    charge_tokens = CHARGE_TOKENS
    _bl = bisect_left
    _br = bisect_right
    kept_peaks = set()

    for ion in theoretical_ions:
        z = ion["charge"]
        if z not in charge_tokens:
            continue

        tokens = charge_tokens[z]
        base_mz = ion["mz"]

        for token in tokens:
            target_mz = base_mz + token
            lo = _bl(peak_mzs, target_mz - tol, 0, n_peaks)
            hi = _br(peak_mzs, target_mz + tol, lo, n_peaks)
            for idx in range(lo, hi):
                kept_peaks.add(sorted_peaks[idx])

    return sorted(kept_peaks, key=lambda x: x[0])


def raw_peak_line_map(block_lines):
    """{(mz, intensity): 原始峰行文本}（同一数值首次出现的行优先）。

    用于过滤时原样保留输入行的格式：只删峰、不改写文本。这样输出文件大小
    与峰数成正比，不再被 %.6f 重写撑大（MassiveKB 上能省三成体积）。
    """
    out = {}
    for line in block_lines:
        s = line.strip()
        parts = s.split()
        if len(parts) == 2:
            try:
                key = (float(parts[0]), float(parts[1]))
            except ValueError:
                continue
            out.setdefault(key, s)
    return out


def rebuild_filtered_block(headers, kept_peaks, raw_lines=None):
    out = ["BEGIN IONS\n"]
    for h in headers:
        out.append(h + "\n")
    out.append(f"COMMENT=TOKEN_FILTER kept_peaks={len(kept_peaks)}\n")
    for mz, intensity in kept_peaks:
        raw = raw_lines.get((mz, intensity)) if raw_lines is not None else None
        out.append((raw + "\n") if raw is not None else f"{mz:.6f} {intensity:.6f}\n")
    out.append("END IONS\n")
    return out


def iter_mgf_blocks(mgf_path):
    """
    逐行读取 MGF 文件，每次 yield 一个完整的 spectrum block。
    不会一次性把整个 MGF 文件读入内存。

    支持 .mgf 与 .mgf.xz —— .xz 走流式解压，无需先解出中间 .mgf 文件。
    """
    block = []
    in_block = False

    if str(mgf_path).lower().endswith(".xz"):
        fh = lzma.open(mgf_path, "rt", encoding="utf-8", errors="ignore")
    else:
        fh = open(mgf_path, "r", encoding="utf-8", errors="ignore")

    with fh as f:
        for line in f:
            stripped = line.strip()

            if stripped == "BEGIN IONS":
                in_block = True
                block = [line]

            elif in_block:
                block.append(line)

                if stripped == "END IONS":
                    yield block
                    block = []
                    in_block = False

    # 如果文件最后有不完整 block，一般直接丢弃
    if in_block and block:
        print(f"[警告] 文件末尾存在不完整 spectrum block，已跳过: {mgf_path}")


def process_one_mgf_streaming(input_mgf, output_mgf, tol=0.05):
    print(f"\n正在处理: {input_mgf}")

    total_blocks = 0
    kept_blocks = 0
    skipped_no_seq_or_charge = 0
    skipped_error = 0

    with open(output_mgf, "w", encoding="utf-8") as fout:
        for idx, block in enumerate(iter_mgf_blocks(input_mgf), start=1):
            total_blocks += 1

            seq, precursor_charge, headers, peaks = parse_ions_block(block)

            if seq is None or precursor_charge is None:
                skipped_no_seq_or_charge += 1
                continue

            try:
                kept_peaks = process_one_spectrum(
                    seq=seq,
                    precursor_charge=precursor_charge,
                    peaks=peaks,
                    tol=tol,
                )
            except Exception as e:
                skipped_error += 1
                print(f"[跳过] spectrum {idx} 处理失败: {e}")
                continue

            fout.writelines(rebuild_filtered_block(
                headers, kept_peaks, raw_peak_line_map(block)))
            fout.write("\n")

            kept_blocks += 1

            if idx % 1000 == 0:
                print(f"已处理 {idx} 个 spectra")

    print(f"已输出过滤后的 MGF: {output_mgf}")
    print(f"总 spectrum blocks: {total_blocks}")
    print(f"成功保留 blocks: {kept_blocks}")
    print(f"跳过无 SEQ 或 CHARGE 的 blocks: {skipped_no_seq_or_charge}")
    print(f"处理失败 blocks: {skipped_error}")


def main():
    root_dir = r"D:\MassiveKBV1"
    tol = 0.05

    skip_dirs = {
        ".idea",
        "SameSequence",
        "Apis-mellifera",
        "Mus_mouse"
    }

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        if folder_name in skip_dirs:
            print(f"\n跳过文件夹: {folder_name}")
            continue

        input_mgf = os.path.join(folder_path, "peaks.db.mgf")

        if not os.path.exists(input_mgf):
            print(f"\n未找到 peaks.db.mgf，跳过: {folder_path}")
            continue

        output_mgf = os.path.join(
            folder_path,
            f"filtered_by_tokens_{folder_name}.mgf"
        )

        process_one_mgf_streaming(
            input_mgf=input_mgf,
            output_mgf=output_mgf,
            tol=tol,
        )


if __name__ == "__main__":
    main()
