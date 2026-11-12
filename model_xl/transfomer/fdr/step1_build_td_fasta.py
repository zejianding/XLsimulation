#!/usr/bin/env python3
"""Step 1 — build the target + decoy FASTA for the FDR experiment.

Input  : the species FASTA (plain base peptides, recommended default) or a
         step2-style variant FASTA (``MASS=... MOD=<annoseq>`` headers).
Targets: FULL modification variants of each base peptide, exactly as the DS
         step2 library enumerates them: fixed C:+57.021 on every C + variable
         M:+15.995 on up to ``--max-variable-mods`` M residues.  This is the
         same target space the DS database search actually uses, so M-oxidized
         spectra's true peptides ARE present (they were missing with plain-C57
         only).  ``--no-mox`` restores the old plain+C57 fast mode; a prebuilt
         ``--variants-fasta`` is used verbatim.
Decoys : the reversed target (same neutral mass, guaranteed not to exist).

Each record is written in two-line FASTA form with a self-describing header:

    >TD_TGT_<i> MASS=<neutral:.6f> LABEL=T SEQ=<annoseq>
    >TD_DEC_<i> MASS=<neutral:.6f> LABEL=D SEQ=<annoseq>

Decoys that collide with an existing target (palindrome / reverse-in-DB) are
dropped so no decoy can ever be a valid target.

Writes:
    <out-fasta>  <species>_td.fasta
    <out-dir>/records/record_step1_<species>.json

Usage:
    python step1_build_td_fasta.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --out-fasta <fdr>/data/mouse/mouse_td.fasta \
        --out-dir <fdr>/output/mouse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fdr_common as fc
import ds_common as dc  # noqa: F401  (path bootstrap + helpers)
from step2_build_candidate_fasta import enumerate_variants  # noqa: E402

MIN_LEN, MAX_LEN = 7, 32

# Modifications observed across the 9-species MGFs (C carbamidomethylation fixed,
# M oxidation variable) — identical to the DS pipeline's mods_config for these
# datasets.  Enumerating them keeps the FDR target library aligned with the real
# search space.
FIXED_MODS: dict[str, list[str]] = {"C": ["57.021"]}
VARIABLE_MODS: dict[str, list[str]] = {"M": ["15.995"]}


def enumerate_all_variants(base_peptides, max_var_mods: int,
                           fixed_mods=None, variable_mods=None):
    """Yield every annotated variant of each base peptide.

    Mods default to fixed C+57.021 / variable M+15.995 (the 9-species data);
    pass ``fixed_mods``/``variable_mods`` from a mods_config for other datasets.
    """
    fixed = fixed_mods if fixed_mods is not None else FIXED_MODS
    variable = variable_mods if variable_mods is not None else VARIABLE_MODS
    for pep in base_peptides:
        for anno in enumerate_variants(pep, fixed, variable, max_var_mods):
            yield anno


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=fc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--out-fasta", required=True, help="output target+decoy FASTA path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants-fasta", default=None,
                   help="use a prebuilt step2 variants FASTA as the target set verbatim "
                        "(headers must carry MASS= and MOD=<annoseq>); skips enumeration")
    p.add_argument("--mods-config", default=None,
                   help="mods_config JSON {\"fixed\": {res:[shift]}, "
                        "\"variable\": {res:[shift]}, \"max_variable_mods\": n} for "
                        "datasets with other modifications (default: fixed C+57.021 / "
                        "variable M+15.995)")
    p.add_argument("--no-mox", action="store_true",
                   help="plain C+57.021 only (no variable mods) — fast iteration mode; "
                        "the DEFAULT enumerates full variable-mod variants so modified "
                        "spectra are recoverable")
    p.add_argument("--max-variable-mods", type=int, default=None,
                   help="max number of variable-mod sites per peptide when enumerating "
                        "(default 3, or the mods_config's value)")
    p.add_argument("--max-peptides", type=int, default=None,
                   help="only use the first N base peptides (fast iteration)")
    return p.parse_args()


def count_records(fasta_path: Path) -> int:
    """Count FASTA records (header lines) — used as the tqdm total."""
    with open(fasta_path, encoding="utf-8", errors="ignore") as fh:
        return sum(1 for line in fh if line.startswith(">"))


def iter_fasta_records(fasta_path: Path):
    """Yield (header, sequence) per FASTA record, joining multi-line sequences."""
    with open(fasta_path, encoding="utf-8", errors="ignore") as fh:
        header = None
        seq_parts: list[str] = []
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            elif header is not None:
                seq_parts.append(line.strip())
        if header is not None:
            yield header, "".join(seq_parts)


def read_plain_peptides(fasta_path: Path, max_peptides: int | None) -> list[str]:
    """Read every sequence record of a plain peptide FASTA (skip U/O, len 7-32)."""
    total = count_records(fasta_path)
    seqs: list[str] = []
    for _hdr, seq in dc.progress(iter_fasta_records(fasta_path), total=total,
                                 desc="step1 read peptides", unit="rec",
                                 mininterval=1.0):
        if max_peptides is not None and len(seqs) >= max_peptides:
            break
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue
        if any(aa in ("U", "O") for aa in seq):
            continue
        seqs.append(seq)
    return seqs


def read_variant_peptides(fasta_path: Path, max_peptides: int | None) -> list[str]:
    """Read annotated sequences from a step2 variants FASTA (MASS=... MOD=...)."""
    total = count_records(fasta_path)
    annoseqs: list[str] = []
    for header, seq in dc.progress(iter_fasta_records(fasta_path), total=total,
                                   desc="step1 read variants", unit="rec",
                                   mininterval=1.0):
        parts = header.split()
        anno = parts[2][4:] if len(parts) > 2 and parts[2].startswith("MOD=") else seq
        if max_peptides is not None and len(annoseqs) >= max_peptides:
            break
        annoseqs.append(anno)
    return annoseqs


def main() -> int:
    args = parse_args()
    out_fasta = Path(args.out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    # resolve modification set: --mods-config JSON overrides the C/M defaults
    fixed_mods, variable_mods, max_var = FIXED_MODS, VARIABLE_MODS, 3
    if args.mods_config:
        cfg = json.load(open(args.mods_config, encoding="utf-8"))
        fixed_mods = cfg.get("fixed", FIXED_MODS)
        variable_mods = cfg.get("variable", VARIABLE_MODS)
        max_var = int(cfg.get("max_variable_mods", max_var))
    if args.max_variable_mods is not None:
        max_var = args.max_variable_mods

    if args.variants_fasta:
        target_inputs = read_variant_peptides(Path(args.variants_fasta), args.max_peptides)
        source = str(args.variants_fasta)
        src_kind = "variants"
        iterator = target_inputs
        desc = "step1 dedupe targets"
    else:
        fasta_path, _mgf = dc.resolve_species_files(args.root, args.species)
        base = read_plain_peptides(fasta_path, args.max_peptides)
        source = str(fasta_path)
        src_kind = "species"
        target_inputs = base
        if args.no_mox:
            iterator = (fc.annotate_plain(s) for s in base)
            desc = "step1 annotate fixed mods"
        else:
            iterator = enumerate_all_variants(base, max_var, fixed_mods, variable_mods)
            desc = "step1 enumerate variants"

    # Targets: unique annotated inputs (C+57.021; M-ox variants when enabled).
    # The enumeration generator's item count is unknown ahead of time, so its
    # progress bar shows count+rate without a percentage.
    targets: list[str] = []
    seen: set[str] = set()
    for a in dc.progress(iterator, total=None if not args.variants_fasta and not args.no_mox
                         else len(target_inputs), desc=desc, unit="pep",
                         mininterval=1.0):
        if a not in seen:
            seen.add(a)
            targets.append(a)

    # Decoys: reversed targets, O(1) collision checks via sets (never a valid target).
    target_set = set(targets)
    decoys: list[str] = []
    decoy_set: set[str] = set()
    n_collision = 0
    for a in dc.progress(targets, total=len(targets), desc="step1 build decoys",
                         unit="pep", mininterval=1.0):
        r = fc.reverse_annotated(a)
        if r is None:
            continue
        if r in target_set or r in decoy_set:
            n_collision += 1
            continue
        decoys.append(r)
        decoy_set.add(r)

    annoseqs = targets + decoys
    labels = [fc.LABEL_TARGET] * len(targets) + [fc.LABEL_DECOY] * len(decoys)

    with open(out_fasta, "w", encoding="utf-8") as fout:
        for i, (a, lab) in enumerate(
            dc.progress(zip(annoseqs, labels), total=len(annoseqs),
                        desc="step1 write TD fasta", unit="rec", mininterval=1.0)
        ):
            tag = "TGT" if lab == fc.LABEL_TARGET else "DEC"
            mass = fc.neutral_mass(a)
            fout.write(f">TD_{tag}_{i} MASS={mass:.6f} LABEL={lab} SEQ={a}\n{a}\n")

    # self-check: header mass == recomputed mass for a few records.
    self_check = {"ok": True, "detail": ""}
    try:
        mass_arr, seq_list, label_arr = fc.load_td_fasta(out_fasta)
        for i in (0, len(seq_list) - 1):
            if abs(float(mass_arr[i]) - fc.neutral_mass(seq_list[i])) > 1e-3:
                self_check = {"ok": False, "detail": seq_list[i]}
                break
    except Exception as exc:  # pragma: no cover
        self_check = {"ok": False, "detail": str(exc)}

    record = {
        "step": 1,
        "species": args.species,
        "source_kind": src_kind,
        "source": source,
        "mod_variants": "enumerated M+15.995 (max %d)" % args.max_variable_mods
        if not args.variants_fasta and not args.no_mox else
        ("plain C+57.021 only" if args.no_mox else "prebuilt variants FASTA"),
        "max_variable_mods": args.max_variable_mods if not args.variants_fasta and not args.no_mox else None,
        "n_input_peptides": len(target_inputs),
        "n_target": len(targets),
        "n_decoy": len(decoys),
        "n_collision_dropped": n_collision,
        "out_fasta": str(out_fasta),
        "out_size_bytes": out_fasta.stat().st_size if out_fasta.exists() else 0,
        "self_check": self_check,
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_step1_{args.species}.json", record)
    print(f"[step1] {args.species}: {len(targets)} targets + {len(decoys)} decoys "
          f"(mode={record['mod_variants']}, collisions_dropped={n_collision}, "
          f"self_check={self_check['ok']})")
    return 0 if self_check["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
