"""Validate and package a Prosit-verified sample for transfer to the server."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import zipfile
from run_pipeline import validate_sample, scan
from prosit_sampling import backbone


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for data in iter(lambda: fh.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', required=True)
    parser.add_argument('--n-groups', type=int, default=30000)
    args = parser.parse_args()
    folder = Path(args.sample_dir).resolve()
    archive = folder.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    print('Validating all real spectra, identities and predictions...', flush=True)
    counts, identities = validate_sample(folder, args.n_groups, 3)
    meta = json.loads((folder/'massTest_meta.json').read_text(encoding='utf8'))
    assert meta['max_spectra_debug'] is None
    assert meta['prosit_sampling']['prediction_verified']
    assert min(counts.values()) >= 3
    assert meta['n_fallback_selected'] == 0
    inputs = {(backbone(s),c) for s,c in counts}
    predictions,_ = scan(folder/'prosit_predicted.mgf')
    assert predictions == {key:1 for key in counts}
    assert len(inputs) == args.n_groups
    ordinals = [int(s.removeprefix('massivekb:')) for s in identities]
    assert all(a<b for a,b in zip(ordinals,ordinals[1:]))
    with (folder/'prosit_input_map.tsv').open('w',encoding='utf8',newline='') as fh:
        w=csv.writer(fh,delimiter='\t',lineterminator='\n')
        w.writerow(['sequence','backbone','charge','n_real_spectra'])
        for (seq,charge),n in sorted(counts.items()):
            w.writerow([seq,backbone(seq),charge,n])
    files=['massTest.mgf','massTest_groups.tsv','massTest_meta.json',
           'prosit_predicted.mgf','sampling_prosit_rejected.csv','prosit_input_map.tsv']
    report=dict(status='passed',verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                n_original_groups=len(counts),n_unique_prosit_inputs=len(inputs),
                n_unique_backbones=len({s for s,c in inputs}),
                n_real_spectra=len(identities),n_unique_spectrum_ids=len(identities),
                n_prosit_spectra=sum(predictions.values()),
                group_size_min=min(counts.values()),group_size_max=max(counts.values()),
                group_size_mean=len(identities)/len(counts),
                source_filename=Path(meta['source']['path']).name,
                runtime_versions={name:importlib.metadata.version(name) for name in
                                  ('numpy','pandas','torch')},
                sampling_code_sha256={name:sha(Path(__file__).parent/name) for name in
                                      ('sample_massivekb_test.py','prosit_sampling.py','mgf_io.py')},
                selection=meta['prosit_sampling'],
                checks=['no duplicate backbone+charge','all original group members retained',
                        'one successful Prosit spectrum per selected original group',
                        'unique source-ordered real spectrum IDs','no debug prefix'],
                sha256={name:sha(folder/name) for name in files})
    (folder/'validation_report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf8'))
    readme = f'''# massiveTest v3: unique Prosit inputs, verified predictions

Validated original groups: {len(counts):,}
Unique unmodified sequence+charge inputs: {len(inputs):,}
Real spectra: {len(identities):,}
Successful Prosit spectra: {sum(predictions.values()):,}
Group sizes: {min(counts.values())}..{max(counts.values())}; mean {len(identities)/len(counts):.6f}.
Seed: {meta['seed']}; min_spectra=3; prefer_spectra=3.

Only one original modified sequence+charge group is selected for each unmodified
sequence+charge key. Different charges of the same backbone are allowed. Original
modifications and all original group members are retained; different modified
groups are NOT merged. Backbones have 2..30 standard amino acids. Selection also
requires a nonempty exported Prosit prediction; rejected candidates are recorded
in sampling_prosit_rejected.csv. This is a Prosit-success-conditioned cohort and
differs from v2; do not compare v3 and v2 scores as if they used the same sample.

The included predictions use Prosit_2025_intensity_MultiFrag / HCD with reanchor,
as implemented by the project. They were generated online before final selection.
Unique requests and successful exported spectra were independently counted.
PI encoding and comparisons have not yet been performed for this cohort.

## Verify on Linux

```bash
cd /data/wangyuhang2/datasets/kb30000/{folder.name}
sha256sum -c SHA256SUMS.txt
```

## Continue on the server

Sync the updated source code, including prosit_sampling.py, to the server first.
Use a NEW output directory and import the included predictions:

```bash
REPO=/home/wangyuhang/seq2mass
SAMPLE=/data/wangyuhang2/datasets/kb30000/{folder.name}
OUT=/data/massiveTest_v3_unique30k_analysis
MODEL="$REPO/piprime_embedding/piprime_embedding/model.ckpt"
mkdir -p "$OUT"
set -o pipefail
python -u "$REPO/v20_0918/transfomer/massivekb_pi_cosine/run_pipeline.py" \\
  --sample-dir "$SAMPLE" --out-dir "$OUT" --model "$MODEL" \\
  --prosit-mgf "$SAMPLE/prosit_predicted.mgf" \\
  --n-groups 30000 --min-spectra 3 --prefer-spectra 3 --seed {meta['seed']} \\
  --domains none,sqrt_clip --workers 8 --gpu 0 --stop-after full \\
  2>&1 | tee "$OUT/full.log"
```

Replace paths as needed. --prosit-mgf prevents an unnecessary new online request.
Do not use the v2 min_spectra=2 argument here. A successful prediction does not
guarantee a valid PI embedding; review downstream validity statistics separately.
No source XZ, local prediction cache or prediction_batches folder is needed on
the server. Keep the six data/audit files and this README/report/checksum together.
'''
    (folder/'README_server.md').write_bytes(readme.encode('utf8'))
    files += ['validation_report.json','README_server.md']
    checks=''.join(f'{sha(folder/name)}  {name}\n' for name in files)
    (folder/'SHA256SUMS.txt').write_bytes(checks.encode('ascii'))
    files += ['SHA256SUMS.txt']
    partial=Path(str(archive)+'.partial')
    print('Building upload archive...',flush=True)
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name in files:
            z.write(folder/name,arcname=folder.name+'/'+name)
    print('Checking archive CRC...',flush=True)
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
    partial.replace(archive)
    digest=sha(archive)
    Path(str(archive)+'.sha256').write_bytes(f'{digest}  {archive.name}\n'.encode('ascii'))
    print(json.dumps(dict(report=report,archive=str(archive),archive_bytes=archive.stat().st_size,
                          archive_sha256=digest),indent=2),flush=True)


if __name__ == '__main__':
    main()
