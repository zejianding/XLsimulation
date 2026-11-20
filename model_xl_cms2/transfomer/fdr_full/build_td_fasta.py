#!/usr/bin/env python3
"""Build a plain target+decoy FASTA for engine searches (1% FDR, reusable).

For every plain peptide in a base FASTA (each entry = one peptide sequence),
write it as a TARGET and its REVERSAL as a DECOY.  Sequences stay PLAIN (no
inline modifications) because search engines (MS-GF+ / MSFragger) apply
modifications themselves from a mod file/config at search time — feeding them
annotated sequences breaks parsing and inflates the library.

The mods_config JSON is used only to derive the engine modification file
(MS-GF+ ``-mod`` format) when ``--mods-out`` is given, so the engine's search
space (fixed + variable mods applied to the plain peptides) matches the target
space of our V20 pipeline.

Reversal preserves the exact neutral mass -> a ±ppm precursor window retrieves
comparable numbers of targets and decoys; a decoy can never be the true
peptide.  Decoys that collide with an existing target (a palindrome, or a
target whose sequence is the reverse of another target in the same library) or
with an already-written decoy are DROPPED (same rule as fdr/step1), so a decoy
is never a valid target.  Decoy headers carry the ``rev_`` prefix (MSFragger
decoy_prefix=rev_ and the generic FDR counters use it).  No ground truth is
needed.

Headers:
    ><name>_TGT_<i> MASS=<neutral:.6f> SEQ=<plain>   target (i = 1..n_target)
    >rev_<name>_TGT_<j> MASS=<neutral:.6f> SEQ=<rev>  decoy (j = 1..n_decoy)

Usage:
    python build_td_fasta.py \
        --fasta for_mouse_test_inscope_peptide.fasta \
        --mods-config mods_config.example.json \
        --out-fasta mouse_td.fasta --mods-out mouse_msgfplus_mods.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_FDR_FULL_DIR = Path(__file__).resolve().parent
_TRANSFORMER_DIR = _FDR_FULL_DIR.parent
_FDR_DIR = _TRANSFORMER_DIR / "fdr"
_DS_DIR = _TRANSFORMER_DIR / "ds"
_SCRIPTS_DIR = _TRANSFORMER_DIR / "scripts"
for _p in (_TRANSFORMER_DIR, _SCRIPTS_DIR, _DS_DIR, _FDR_DIR, _FDR_FULL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fdr_common as fc  # noqa: E402  (neutral mass on plain seqs)

MIN_LEN, MAX_LEN = 7, 32
FORBIDDEN = {"U", "O"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fasta", required=True, help="base peptide FASTA (plain sequences)")
    p.add_argument("--mods-config", required=True,
                   help="mods_config JSON {fixed, variable, max_variable_mods}")
    p.add_argument("--out-fasta", required=True, help="output target+decoy FASTA")
    p.add_argument("--mods-out", default=None,
                   help="also write the engine (MS-GF+ -mod) mod file from the JSON")
    p.add_argument("--name", default=None,
                   help="id prefix for headers (default: input FASTA stem)")
    p.add_argument("--max-peptides", type=int, default=None,
                   help="only process first N base peptides (fast iteration)")
    return p.parse_args()


def write_msgfplus_mods(path: Path, mods: dict):
    """Convert a mods_config JSON to the MS-GF+ -mod file format."""
    lines = [f"NumMods={int(mods.get('max_variable_mods', 3))}"]
    for res, shifts in mods.get("fixed", {}).items():
        for s in shifts:
            pos = "Prot-N-term" if res == "NTERM" else "any"
            lines.append(f"{s},{'*' if res == 'NTERM' else res},fix,{pos},Fixed{res}")
    for res, shifts in mods.get("variable", {}).items():
        for s in shifts:
            pos = "Prot-N-term" if res == "NTERM" else "any"
            lines.append(f"{s},{'*' if res == 'NTERM' else res},opt,{pos},Variable{res}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    mods = json.load(open(args.mods_config, encoding="utf-8"))
    name = args.name or Path(args.fasta).stem
    out_fasta = Path(args.out_fasta)
    if args.mods_out:
        write_msgfplus_mods(Path(args.mods_out), mods)
        print(f"mods file -> {args.mods_out}")

    n_base = n_skip = 0
    peps: list[str] = []
    seen: set[str] = set()
    with open(Path(args.fasta), encoding="utf-8", errors="ignore") as fin:
        header = None
        for raw in fin:
            line = raw.strip()
            if line.startswith(">"):
                header = line
                continue
            if header is None or not line:
                continue
            if args.max_peptides is not None and n_base >= args.max_peptides:
                break
            n_base += 1
            pep = line
            if not (MIN_LEN <= len(pep) <= MAX_LEN) or any(a in FORBIDDEN for a in pep):
                n_skip += 1
                continue
            if pep not in seen:
                seen.add(pep)
                peps.append(pep)

    # Decoys: reversals, dropped when they collide with any target (palindrome
    # / reverse-in-library) or with an already-accepted decoy — mirror of fdr/step1.
    target_set = seen
    n_target = len(peps)
    decoys: list[str] = []
    decoy_set: set[str] = set()
    n_collision = 0
    for pep in peps:
        decoy = pep[::-1]
        if decoy in target_set or decoy in decoy_set:
            n_collision += 1
            continue
        decoys.append(decoy)
        decoy_set.add(decoy)
    n_decoy = len(decoys)

    with open(out_fasta, "w", encoding="utf-8") as fout:
        for i, pep in enumerate(peps, start=1):
            mass = fc.neutral_plain(pep)
            fout.write(f">{name}_TGT_{i} MASS={mass:.6f} SEQ={pep}\n{pep}\n")
        for j, decoy in enumerate(decoys, start=1):
            mass = fc.neutral_plain(decoy)
            fout.write(f">rev_{name}_TGT_{j} MASS={mass:.6f} SEQ={decoy}\n{decoy}\n")
    print(f"done: base={n_base} target={n_target} decoy={n_decoy} "
          f"collision_dropped={n_collision} dup_skipped={n_base - len(peps)} "
          f"skip={n_skip} -> {out_fasta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
