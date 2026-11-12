#!/usr/bin/env python3
"""Step 1 — discover the modifications actually present in the MGF SEQ= lines.

For each species MGF, parse every ``SEQ=`` ground-truth annotation and collect
the distinct ``(residue, shift)`` modification tokens.  Compare against the
known set (fixed C:+57.021, variable M:+15.995).  Any token NOT in the known
set is reported as ``new_mods_found`` and, if present, merged into the variable
mods so downstream enumeration covers it.

Writes:
    <config-dir>/mods_config.json          final fixed/variable/max_variable_mods
    <out-dir>/records/record_step1_<sp>.json   process record

Usage:
    python step1_scan_mgf_mods.py --species mouse --root /home/wangyuhang/seq2mass/v1-search
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import ds_common as dc


KNOWN_FIXED = {"C": [57.021]}
KNOWN_VARIABLE = {"M": [15.995]}
MAX_VARIABLE_MODS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT,
                   help="v1-search data root (contains <species>/ dirs)")
    p.add_argument("--out-dir", required=True, help="output/<species> dir")
    p.add_argument("--config-dir", required=True, help="dir for mods_config.json")
    p.add_argument("--max-scans", type=int, default=None,
                   help="limit spectra scanned (for fast iteration)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    fasta_path, mgf_path = dc.resolve_species_files(args.root, args.species)

    res_counter: Counter = Counter()
    nterm_counter: Counter = Counter()
    n_spectra = 0
    n_with_seq = 0
    n_seq_with_mod = 0
    example_new = []

    for meta, _peaks in dc.iter_mgf_spectra(mgf_path, max_scans=args.max_scans):
        n_spectra += 1
        seq = meta.get("SEQ", "")
        if not seq:
            continue
        n_with_seq += 1
        res_tokens, nterm_tokens = dc.extract_mod_tokens(seq)
        has_mod = False
        for res, shift in res_tokens:
            key = f"{res}{shift:+.3f}"
            res_counter[key] += 1
            has_mod = True
        for shift in nterm_tokens:
            key = f"NTERM{shift:+.3f}"
            nterm_counter[key] += 1
            has_mod = True
        if has_mod:
            n_seq_with_mod += 1

    known_res = set()
    for res, shifts in {**KNOWN_FIXED, **KNOWN_VARIABLE}.items():
        for s in shifts:
            known_res.add(f"{res}{s:+.3f}")

    new_mods_found = sorted(
        [k for k in res_counter if k not in known_res]
        + [k for k in nterm_counter if k not in known_res]
    )

    variable = {res: list(shifts) for res, shifts in KNOWN_VARIABLE.items()}
    merged_new = False
    for token in new_mods_found:
        if token.startswith("NTERM"):
            key = "NTERM"
            shift = float(token[5:])
        else:
            key = token[0]
            shift = float(token[1:])
        variable.setdefault(key, [])
        if shift not in variable[key]:
            variable[key].append(shift)
            merged_new = True

    config = {
        "fixed": {r: list(v) for r, v in KNOWN_FIXED.items()},
        "variable": variable,
        "max_variable_mods": MAX_VARIABLE_MODS,
        "new_mods_found": new_mods_found,
        "mod_token_counts": dict(res_counter),
        "nterm_mod_counts": dict(nterm_counter),
        "n_spectra": n_spectra,
        "n_spectra_with_seq": n_with_seq,
        "n_spectra_with_mod": n_seq_with_mod,
        "source_mgf": mgf_path.name,
        "known_set": {
            "fixed": KNOWN_FIXED,
            "variable": KNOWN_VARIABLE,
        },
        "merged_new_mods": merged_new,
    }

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "mods_config.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)

    if new_mods_found:
        print(f"[step1] WARNING: found mods beyond the known set: {new_mods_found}")
    else:
        print(f"[step1] no mods beyond {{C+57.021, M+15.995}}; "
              f"mod_token_counts={dict(res_counter)}")

    record = {
        "step": 1,
        "species": args.species,
        "n_spectra": n_spectra,
        "mod_token_counts": dict(res_counter),
        "nterm_mod_counts": dict(nterm_counter),
        "new_mods_found": new_mods_found,
        "merged_new_mods": merged_new,
        "config_path": str(config_path),
        "source_mgf": str(mgf_path),
        "input_fasta": str(fasta_path),
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_step1_{args.species}.json", record)
    print(f"[step1] done: {n_spectra} spectra, {n_seq_with_mod} with mods, "
          f"new_mods={new_mods_found}, config -> {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
