"""Select one original group per Prosit input and optionally verify predictions."""
from __future__ import annotations
from collections import Counter
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

TF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TF))
from spectrum_transformer.data.dataset import parse_modified_sequence
from mgf_io import iter_blocks, parse_block_key


def backbone(sequence):
    return ''.join(parse_modified_sequence(sequence)[0])


def candidates(counts, seed, min_spectra=3, prefer_spectra=3):
    """Uniform keys then uniform representative within the best available stratum."""
    rng = random.Random(seed)
    pools = {}
    excluded = Counter()
    for key in sorted(counts):
        seq, charge = key
        if counts[key] < min_spectra:
            excluded['too_few_spectra'] += 1
            continue
        bare = backbone(seq)
        if not 2 <= len(bare) <= 30:
            excluded['backbone_length_outside_2_30'] += 1
            continue
        if set(bare) - set('ACDEFGHIKLMNPQRSTVWY') or charge < 1:
            excluded['nonstandard_residue_or_invalid_charge'] += 1
            continue
        target = (bare, charge)
        preferred = counts[key] >= prefer_spectra
        if target not in pools or preferred > pools[target][0]:
            pools[target] = (preferred, [key])
        elif preferred == pools[target][0]:
            pools[target][1].append(key)
    high, low = [], []
    for target in sorted(pools):
        pref, options = pools[target]
        (high if pref else low).append(rng.choice(options))
    rng.shuffle(high)
    rng.shuffle(low)
    return high, low, dict(excluded)


def select_unique(counts, n_groups, seed, min_spectra, prefer_spectra,
                  out, verify=False, batch_size=1500):
    import numpy as np
    high, low, excluded = candidates(counts, seed, min_spectra, prefer_spectra)
    initial_high, initial_low = len(high), len(low)
    if initial_high + initial_low < n_groups:
        raise ValueError('Not enough unique eligible Prosit inputs')
    out = Path(out)
    picked = []
    rejected = []
    successes = {}
    attempted = 0
    batch_number = 0
    if verify and min_spectra != prefer_spectra:
        raise ValueError('Verified sampling requires min_spectra == prefer_spectra; no unverified fallback')
    for pool in (high, low):
        offset = 0
        while len(picked) < n_groups and offset < len(pool):
            n = min(batch_size, n_groups - len(picked))
            batch = pool[offset:offset+n]
            offset += len(batch)
            if not verify:
                picked.extend(batch)
                continue
            batch_number += 1
            folder = out / 'prediction_batches' / f'{batch_number:05d}'
            folder.mkdir(parents=True, exist_ok=False)
            np.savez(folder/'inputs.npz', seqs=np.array([s for s,c in batch]),
                     charges=np.array([c for s,c in batch]))
            cmd = [sys.executable, '-u', str(TF/'eval_10k_gt/prosit_predict_mgf.py'),
                   '--npz', str(folder/'inputs.npz'), '--out', str(folder/'predicted.mgf'),
                   '--cache', str(folder/'koina_cache.tsv.gz'),
                   '--skipped-out', str(folder/'skipped.csv'), '--batch-size', str(batch_size)]
            print('VERIFY_PROSIT', batch_number, 'accepted', len(picked), 'requesting', len(batch), flush=True)
            subprocess.run(cmd, check=True)
            got = {}
            with (folder/'predicted.mgf').open('rb') as fh:
                for block in iter_blocks(fh):
                    key = parse_block_key(block)
                    if key in got or key not in batch:
                        raise ValueError('Unexpected/duplicate verified prediction')
                    got[key] = block
            with (folder/'skipped.csv').open(encoding='utf8', newline='') as fh:
                skipped = list(csv.DictReader(fh))
            absent = {(r['sequence'], int(r['charge'])) for r in skipped}
            if len(absent) != len(skipped) or absent != set(batch)-set(got):
                raise ValueError('Prediction success/failure partition mismatch')
            attempted += len(batch)
            rejected.extend(skipped)
            picked.extend(key for key in batch if key in got)
            successes.update(got)
        if len(picked) == n_groups:
            break
    if len(picked) != n_groups:
        raise ValueError('Not enough successful unique Prosit predictions; sample not committed')
    picked.sort()
    if verify:
        with (out/'prosit_predicted.mgf.partial').open('w', encoding='utf8', newline='\n') as fh:
            for key in picked:
                fh.write(successes[key].lstrip('\n') + '\nEND IONS\n')
        with (out/'sampling_prosit_rejected.csv').open('w', encoding='utf8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['sequence','charge','reason','detail'])
            writer.writeheader()
            writer.writerows(rejected)
    report = dict(policy='unique_unmodified_sequence_and_charge',
                  representative_policy='uniform_random_original_group_in_best_stratum',
                  backbone_length_min=2, backbone_length_max=30,
                  eligible_unique_preferred_pool=initial_high,
                  eligible_unique_fallback_pool=initial_low,
                  excluded_original_groups=excluded,
                  prediction_verified=verify, prediction_attempted_groups=attempted,
                  prediction_rejected_groups=len(rejected),
                  prediction_successful_groups=len(successes),
                  n_unique_prosit_inputs=len({(backbone(s),c) for s,c in picked}))
    if report['n_unique_prosit_inputs'] != n_groups:
        raise ValueError('Prosit input collision')
    return picked, initial_high, initial_low, report
