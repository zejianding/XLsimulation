#!/usr/bin/env python3
"""Step 2 — enumerate modified peptide variants into a new per-species FASTA.

For every peptide in the species FASTA:
  - fixed C:+57.021 is always applied on every C (not a variable-mod slot);
  - variable M:+15.995 is applied on 0..min(#M, 3) M residues (all C(#M,k)
    position subsets), i.e. the unmodified variant plus up to 3 oxidations;
  - peptides containing U/O are skipped (V20 AA_MASS has neither; the MGF
    ground truth never carries U/O) unless --keep-u is given.

Each variant is written as a two-line FASTA record (original format) with the
precise neutral mass (V20 AA_MASS + H2O + parsed mod shifts) in the header:

    >{original_header} MASS={neutral:.6f} MOD={annotated_seq}
    {annotated_seq}

The annotated sequence uses the exact MGF SEQ= annotation format
(3-decimal `C+57.021` / `M+15.995`).

Writes:
    <out-fasta>   variants_<sp>.fasta
    <out-dir>/records/record_step2_<sp>.json   process record

Usage:
    python step2_build_candidate_fasta.py --species mouse \
        --root /home/wangyuhang/seq2mass/v1-search \
        --mods-config <ds>/configs/mods_config.json \
        --out-fasta <ds>/data/mouse/variants_mouse.fasta \
        --out-dir <ds>/output/mouse
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import ds_common as dc
from predict_mgf import AA_MASS

MIN_LEN, MAX_LEN = 7, 32
U_MASS = 150.953630  # v1 U mass, only used with --keep-u


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", required=True)
    p.add_argument("--root", default=dc.SERVER_V1_SEARCH_ROOT)
    p.add_argument("--mods-config", required=True, help="mods_config.json from step1")
    p.add_argument("--out-fasta", required=True, help="output variants FASTA path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-peptides", type=int, default=None,
                   help="only process the first N base peptides (fast iteration)")
    p.add_argument("--keep-u", action="store_true",
                   help="keep U-containing peptides using U=150.953630 (v1 mass)")
    return p.parse_args()


def load_mods_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def enumerate_variants(peptide: str, fixed_mods: dict, variable_mods: dict,
                       max_var_mods: int) -> list[str]:
    """Return annotated variant sequences for one plain peptide.

    General enumeration driven by the mods_config:
      - fixed: every site of a fixed-mod residue always receives the summed
        shift (never consumes a variable-mod slot); fixed NTERM is always added.
      - variable (residue): each site of a variable-mod residue is either
        unmodified or takes one of its shifts; across ALL variable residue
        types, at most ``max_var_mods`` sites are modified.
      - variable NTERM: the N-term is either unmodified or takes one shift,
        consuming one variable-mod slot.

    Annotated-sequence format matches the MGF SEQ= convention
    (3-decimal ``+57.021`` / ``+15.995`` / ``+42.011``; n-term shift prefixed).
    """
    fixed_shifts: dict[int, float] = {}
    fixed_nterm = 0.0
    for res, shifts in fixed_mods.items():
        if res == "NTERM":
            fixed_nterm += sum(float(s) for s in shifts)
            continue
        shift_sum = sum(float(s) for s in shifts)
        if shift_sum != 0.0:
            for i, aa in enumerate(peptide):
                if aa == res:
                    fixed_shifts[i] = fixed_shifts.get(i, 0.0) + shift_sum

    var_sites: list[tuple[int, list[float]]] = []
    for res, shifts in variable_mods.items():
        if res == "NTERM":
            continue
        opts = [float(s) for s in shifts]
        for i, aa in enumerate(peptide):
            if aa == res:
                var_sites.append((i, opts))
    nterm_options = [None] + [float(s) for s in variable_mods.get("NTERM", [])]

    variants: list[str] = []
    n_sites = len(var_sites)
    for nt_choice in nterm_options:
        remaining = max_var_mods - (1 if nt_choice is not None else 0)
        if remaining < 0:
            continue
        n_max = min(n_sites, remaining)
        for k in range(0, n_max + 1):
            for site_combo in itertools.combinations(range(n_sites), k):
                chosen = [var_sites[i] for i in site_combo]
                for choice in itertools.product(*[opts for _, opts in chosen]):
                    shift_map = dict(fixed_shifts)
                    for (pos, _), sh in zip(chosen, choice):
                        shift_map[pos] = shift_map.get(pos, 0.0) + sh
                    nterm_shift = fixed_nterm + (nt_choice if nt_choice is not None else 0.0)
                    chars: list[str] = []
                    if nterm_shift != 0.0:
                        chars.append(f"{nterm_shift:+.3f}")
                    for i, aa in enumerate(peptide):
                        chars.append(aa)
                        s = shift_map.get(i, 0.0)
                        if s != 0.0:
                            chars.append(f"{s:+.3f}")
                    variants.append("".join(chars))
    return variants


def neutral_mass_with_table(annoseq: str, mass_table: dict) -> float:
    residues, deltas, nterm = dc.parse_modified_sequence(annoseq)
    return sum(mass_table.get(aa, 0.0) + d for aa, d in zip(residues, deltas)) + nterm + dc.H2O


def main() -> int:
    args = parse_args()
    out_fasta = Path(args.out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = dc.StepTimer()

    fasta_path, _mgf = dc.resolve_species_files(args.root, args.species)
    mods = load_mods_config(Path(args.mods_config))
    fixed_mods = mods["fixed"]
    variable_mods = mods["variable"]
    max_var_mods = int(mods["max_variable_mods"])

    # Enumeration is fully driven by mods_config: fixed always applied,
    # variable (all residue types + optional NTERM) up to max_variable_mods.

    mass_table = dict(AA_MASS)
    if args.keep_u:
        mass_table["U"] = U_MASS
    forbidden = {"U", "O"}
    if args.keep_u:
        forbidden.discard("U")

    n_base = 0
    n_variants = 0
    n_skipped_uo = 0
    n_skipped_len = 0
    last_peptide_ok = False

    with open(fasta_path, encoding="utf-8", errors="ignore") as _cf:
        n_records = sum(1 for _l in _cf if _l.startswith(">"))

    with open(fasta_path, encoding="utf-8", errors="ignore") as fin, \
         open(out_fasta, "w", encoding="utf-8") as fout:
        header = None
        for raw in dc.progress(fin, total=n_records, desc="step2 enumerate",
                               unit="pep", mininterval=1.0):
            line = raw.rstrip("\n")
            if line.startswith(">"):
                header = line[1:]
                continue
            if header is None:
                continue
            seq = line.strip()
            header = header
            if not seq:
                header = None
                continue
            if args.max_peptides is not None and n_base >= args.max_peptides:
                break

            n_base += 1
            if not (MIN_LEN <= len(seq) <= MAX_LEN):
                n_skipped_len += 1
                header = None
                continue
            if any(aa in forbidden for aa in seq):
                n_skipped_uo += 1
                header = None
                continue

            variants = enumerate_variants(
                seq, fixed_mods, variable_mods, max_var_mods
            )
            for anno in variants:
                mass = neutral_mass_with_table(anno, mass_table)
                fout.write(f">{header} MASS={mass:.6f} MOD={anno}\n{anno}\n")
                n_variants += 1
            last_peptide_ok = True
            header = None

    # self-check: re-read first and last records and verify the header mass
    self_check = {"ok": True, "detail": ""}
    try:
        with open(out_fasta, encoding="utf-8") as fh:
            lines = fh.readlines()
        if lines:
            for probe in (0, len(lines) - 2):
                hdr = lines[probe].strip()
                anno = lines[probe + 1].strip()
                oh, mass_tok, mod_tok = hdr.split()
                recompute = neutral_mass_with_table(mod_tok[4:], mass_table)
                if abs(float(mass_tok[5:]) - recompute) > 1e-4:
                    self_check = {"ok": False, "detail": hdr}
                    break
    except Exception as exc:  # pragma: no cover
        self_check = {"ok": False, "detail": str(exc)}

    multiplier = n_variants / n_base if n_base else 0.0
    record = {
        "step": 2,
        "species": args.species,
        "n_base_peptides": n_base,
        "n_variants": n_variants,
        "n_skipped_uo": n_skipped_uo,
        "n_skipped_len": n_skipped_len,
        "multiplier": round(multiplier, 4),
        "fixed": mods["fixed"],
        "variable": mods["variable"],
        "max_variable_mods": max_var_mods,
        "header_format": ">HEADER MASS=<neutral:.6f> MOD=<annotated_seq>",
        "self_check": self_check,
        "out_fasta": str(out_fasta),
        "input_fasta": str(fasta_path),
        "out_size_bytes": out_fasta.stat().st_size if out_fasta.exists() else 0,
        **timer.summary(),
    }
    dc.write_record(out_dir / "records" / f"record_step2_{args.species}.json", record)
    print(f"[step2] {args.species}: {n_base} base peptides -> {n_variants} variants "
          f"(x{multiplier:.3f}, skip_uo={n_skipped_uo}, self_check={self_check['ok']})")
    return 0 if self_check["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
