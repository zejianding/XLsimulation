#!/usr/bin/env python3
"""Probe the Koina Prosit backbone-length limit (multifrag25 'intseq' = [-1,30]).

For each requested peptide length, predict a single unmodified peptide with the
SAME Koina interface as step4_prosit_score.py (koinapy.Koina + a pandas
DataFrame with peptide_sequences / precursor_charges / fragmentation_types).
Prints success (rows/shape) or the exact exception, so we can confirm where the
deterministic >30aa failure starts.

Usage (server, where koinapy is installed):
    python prosit_len_probe.py                          # 28..33 aa, charge 2,3
    python prosit_len_probe.py --lengths 30,31 --charge 2
"""

import argparse
import sys
import time

AA = "ACDEFGHIKLMNPQRSTVWY"  # standard residues only (Prosit vocab)


def probe(model, pep: str, charge: int) -> tuple[bool, str]:
    import pandas as pd
    df = pd.DataFrame({
        "peptide_sequences": [pep],
        "precursor_charges": [charge],
        "fragmentation_types": ["HCD"],
    })
    t0 = time.time()
    try:
        pred = model.predict(df)
        dt = time.time() - t0
        shape = getattr(pred, "shape", None)
        cols = list(getattr(pred, "columns", []))
        return True, f"OK rows={len(pred)} shape={shape} cols={cols} ({dt:.1f}s)"
    except Exception as exc:
        dt = time.time() - t0
        return False, f"FAILED {type(exc).__name__}: {exc} ({dt:.1f}s)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Prosit_2025_intensity_MultiFrag")
    p.add_argument("--host", default="koina.wilhelmlab.org:443")
    p.add_argument("--lengths", default="28,29,30,31,32,33",
                   help="comma-separated backbone lengths to probe")
    p.add_argument("--charges", default="2,3",
                   help="comma-separated precursor charges to probe")
    args = p.parse_args()

    from koinapy import Koina  # server-side import
    model = Koina(args.model, args.host)
    print(f"[probe] model={args.model} host={args.host}")
    for attr in ("inputs", "outputs", "model_config", "input_schema", "output_schema"):
        obj = getattr(model, attr, None)
        if obj is not None:
            print(f"[probe] {attr} = {obj}")

    lengths = [int(x) for x in args.lengths.split(",")]
    charges = [int(x) for x in args.charges.split(",")]
    n_fail = 0
    for n in lengths:
        pep = (AA * 3)[:n]  # deterministic mixed-sequence of length n
        for charge in charges:
            ok, msg = probe(model, pep, charge)
            print(f"len={n:2d} ({pep}) charge={charge}: {msg}", flush=True)
            n_fail += 0 if ok else 1
    print(f"\n[probe] done: {sum(len(charges) for _ in lengths) - n_fail} OK, "
          f"{n_fail} failed across lengths {lengths}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
