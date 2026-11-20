#!/usr/bin/env python3
"""Sample complete peptide+charge groups, preferring N>=3, with N>=2 fallback.

Both passes scan the same full source. --max-spectra is explicitly a biased debug
prefix. The sample keeps original headers/peaks and adds a stable SPECTRUM_ID
(source block ordinal). Existing completed samples are never overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import time
from mgf_io import iter_blocks, open_binary_stream, parse_block_key, make_group_filename, finish_stream

SCHEMA_VERSION = 2


def select_groups(counts, n_groups, seed, min_spectra=2, prefer_spectra=3):
    if n_groups <= 0 or min_spectra < 2 or prefer_spectra < min_spectra:
        raise ValueError("Require n_groups>0 and prefer_spectra>=min_spectra>=2")
    preferred = sorted(k for k, n in counts.items() if n >= prefer_spectra)
    fallback = sorted(k for k, n in counts.items() if min_spectra <= n < prefer_spectra)
    if len(preferred) + len(fallback) < n_groups:
        raise ValueError("Not enough eligible groups; no partial sample will be committed")
    rng = random.Random(seed)
    picked = rng.sample(preferred, min(n_groups, len(preferred)))
    picked += rng.sample(fallback, n_groups - len(picked))
    return sorted(picked), len(preferred), len(fallback)


def signature(path):
    p = Path(path).resolve(); st = p.stat()
    return dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mgf-xz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-groups", type=int, default=30000)
    p.add_argument("--min-spectra", type=int, default=2)
    p.add_argument("--prefer-spectra", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--max-spectra", type=int)
    p.add_argument("--progress-every", type=int, default=1000000)
    p.add_argument("--dump-all-groups", action="store_true")
    p.add_argument("--unique-prosit-input", action="store_true", help="Select at most one modified group per unmodified sequence+charge, length 2..30")
    p.add_argument("--verify-prosit", action="store_true", help="Require successful exported Prosit spectrum before selecting each group; implies unique input")
    a = p.parse_args()
    if a.verify_prosit:
        a.unique_prosit_input = True
        if a.min_spectra != a.prefer_spectra:
            p.error("--verify-prosit requires --min-spectra equal to --prefer-spectra")
    if a.max_spectra is not None and a.max_spectra <= 0:
        p.error("--max-spectra must be positive")
    select_groups({("validation", 2): max(a.prefer_spectra, 2)}, 1, a.seed,
                  a.min_spectra, a.prefer_spectra)
    if a.n_groups <= 0:
        p.error("--n-groups must be positive")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    names = ["massTest.mgf", "massTest_groups.tsv", "massTest_meta.json"]
    if a.verify_prosit:
        names.insert(0, "prosit_predicted.mgf")
    if any((out / n).exists() for n in names):
        raise FileExistsError("Use a new output directory; existing sample is preserved")
    before = signature(a.mgf_xz)
    counts = Counter(); total = missing = 0; t0 = time.time()
    stream, proc, size = open_binary_stream(a.mgf_xz, a.threads)
    with stream:
        for block in iter_blocks(stream, a.max_spectra, a.progress_every, "count", size):
            total += 1
            key = parse_block_key(block)
            if key is None:
                missing += 1
            else:
                counts[key] += 1
    finish_stream(proc, capped=a.max_spectra is not None)
    pass1 = time.time() - t0
    prosit_report = None
    if a.unique_prosit_input:
        from prosit_sampling import select_unique
        picked, n_preferred, n_fallback, prosit_report = select_unique(
            counts, a.n_groups, a.seed, a.min_spectra, a.prefer_spectra, out, a.verify_prosit)
        print(json.dumps(prosit_report, indent=2), flush=True)
    else:
        picked, n_preferred, n_fallback = select_groups(counts, a.n_groups, a.seed, a.min_spectra, a.prefer_spectra)
    selected = set(picked); seen = Counter(); t1 = time.time()
    stream, proc, size = open_binary_stream(a.mgf_xz, a.threads)
    with stream, (out / "massTest.mgf.partial").open("w", encoding="utf-8", newline="\n") as fh:
        for ordinal, block in enumerate(iter_blocks(stream, a.max_spectra, a.progress_every, "extract", size), 1):
            key = parse_block_key(block)
            if key not in selected:
                continue
            if "\nSPECTRUM_ID=" in block:
                raise ValueError("Source already has SPECTRUM_ID; refusing to replace identities")
            block = block.lstrip("\n").replace("BEGIN IONS\n", f"BEGIN IONS\nSPECTRUM_ID=massivekb:{ordinal}\n", 1)
            fh.write(block + "\nEND IONS\n")
            seen[key] += 1
    finish_stream(proc, capped=a.max_spectra is not None)
    if signature(a.mgf_xz) != before or seen != Counter({k: counts[k] for k in picked}):
        raise RuntimeError("Source changed or extracted group counts differ; sample not committed")
    with (out / "massTest_groups.tsv.partial").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["group_id", "group_file", "sequence", "charge", "n_spectra", "emitted", "sampling_stratum"])
        for gid, (seq, ch) in enumerate(picked, 1):
            n = counts[(seq, ch)]
            w.writerow([gid, make_group_filename(seq, ch), seq, ch, n, seen[(seq, ch)],
                        "preferred" if n >= a.prefer_spectra else "fallback"])
    sizes = [counts[k] for k in picked]
    meta = dict(schema_version=SCHEMA_VERSION, source=before, source_mgf_xz=str(a.mgf_xz),
                seed=a.seed, n_groups_requested=a.n_groups, min_spectra=a.min_spectra,
                prefer_spectra=a.prefer_spectra, max_spectra_debug=a.max_spectra,
                n_spectra_total=total, n_groups_total=len(counts), n_blocks_without_key=missing,
                candidate_pool_size=n_preferred+n_fallback, preferred_pool_size=n_preferred,
                fallback_pool_size=n_fallback, n_preferred_selected=sum(n>=a.prefer_spectra for n in sizes),
                n_fallback_selected=sum(n<a.prefer_spectra for n in sizes),
                n_groups_selected=len(picked), n_spectra_emitted=sum(sizes),
                selected_group_size_histogram=dict(sorted(Counter(sizes).items())),
                selected_group_size_min=min(sizes), selected_group_size_max=max(sizes),
                selected_group_size_mean=sum(sizes)/len(sizes),
                seconds_pass1=pass1, seconds_pass2=time.time()-t1)
    if prosit_report is not None:
        meta["prosit_sampling"] = prosit_report
    if a.dump_all_groups:
        with (out / "groups_all.csv.partial").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh); w.writerow(["sequence", "charge", "n_spectra"])
            w.writerows((seq, ch, n) for (seq, ch), n in sorted(counts.items()))
        (out / "groups_all.csv.partial").replace(out / "groups_all.csv")
    (out / "massTest_meta.json.partial").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for name in names:  # metadata committed last
        (out / (name + ".partial")).replace(out / name)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
