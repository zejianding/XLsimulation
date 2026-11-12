"""Filter MGF peaks: m/z cutoff + intensity threshold + top-N by intensity."""
import sys
import numpy as np
from pyteomics import mgf

input_mgf = sys.argv[1]
output_mgf = sys.argv[2]
min_mz = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
min_intensity = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
top_n = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # 0 = keep all

with mgf.MGF(input_mgf) as reader:
    with open(output_mgf, 'w') as out:
        for spec in reader:
            params = spec.get('params', {})
            mz = spec.get('m/z array', np.array([]))
            intensity = spec.get('intensity array', np.array([]))

            # Filter by m/z
            keep = mz >= min_mz
            mz, intensity = mz[keep], intensity[keep]

            # Filter by intensity
            keep = intensity >= min_intensity
            mz, intensity = mz[keep], intensity[keep]

            # Keep top-N by intensity
            if top_n > 0 and len(mz) > top_n:
                idx = np.argsort(intensity)[-top_n:]
                idx = np.sort(idx)
                mz, intensity = mz[idx], intensity[idx]

            # Write
            title = params.get('title', params.get('TITLE', ''))
            pepmass = params.get('pepmass', params.get('PEPMASS', ''))
            charge = params.get('charge', params.get('CHARGE', ''))
            rt = params.get('rtinseconds', params.get('RTINSECONDS', ''))
            seq = params.get('seq', params.get('SEQ', ''))
            source = params.get('source', params.get('SOURCE', ''))

            if isinstance(pepmass, (list, tuple)):
                pepmass = float(pepmass[0]) if len(pepmass) > 0 else 0.0
            if isinstance(charge, (list, tuple)):
                charge = int(charge[0]) if len(charge) > 0 else 1
            elif isinstance(charge, str):
                import re
                m = re.search(r'(\d+)', charge)
                charge = int(m.group(1)) if m else 1

            out.write('BEGIN IONS\n')
            out.write(f'TITLE={title}\n')
            out.write(f'PEPMASS={pepmass}\n')
            if rt:
                out.write(f'RTINSECONDS={rt}\n')
            out.write(f'CHARGE={charge}+\n')
            if seq:
                out.write(f'SEQ={seq}\n')
            if source:
                out.write(f'SOURCE={source}\n')
            for mzv, intv in zip(mz, intensity):
                out.write(f'{mzv} {intv}\n')
            out.write('END IONS\n\n')

print(f'Filtered: min_mz>={min_mz}, min_intensity>={min_intensity}, top_n={top_n or "all"} -> {len(mz) if len(mz) > 0 else 0} peaks')

