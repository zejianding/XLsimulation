"""
Remove the last spectrum (baseline prediction) from the original MGF,
append the ResNet-predicted spectrum as the new last entry.
"""
import sys
from pyteomics import mgf

orig_mgf = sys.argv[1]      # QLEEAEEEAQR_28.mgf
resnet_mgf = sys.argv[2]    # ResNet single-spectrum MGF
output_mgf = sys.argv[3]

# Read all spectra from original MGF except last
spectra = []
with mgf.MGF(orig_mgf) as reader:
    for spec in reader:
        spectra.append(spec)

print(f"Original: {len(spectra)} spectra")
real_spectra = spectra[:-1]  # drop last (baseline prediction)
print(f"Real (all but last): {len(real_spectra)} spectra")
print(f"Dropped last (baseline prediction)")

# Read ResNet spectrum
with mgf.MGF(resnet_mgf) as reader:
    for spec in reader:
        resnet_spec = spec
        break

# Write: real spectra + ResNet prediction
with open(output_mgf, 'w') as out:
    for spec in real_spectra:
        params = spec.get('params', {})
        title = params.get('title', params.get('TITLE', ''))
        pepmass = params.get('pepmass', params.get('PEPMASS', ''))
        charge = params.get('charge', params.get('CHARGE', ''))
        rt = params.get('rtinseconds', params.get('RTINSECONDS', ''))
        seq = params.get('seq', params.get('SEQ', ''))
        scans = params.get('scans', params.get('SCANS', ''))
        source = params.get('source', params.get('SOURCE', ''))

        # Normalize PEPMASS
        if isinstance(pepmass, (list, tuple)):
            pepmass = float(pepmass[0]) if len(pepmass) > 0 else 0.0
        # Normalize CHARGE
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
        if scans:
            out.write(f'SCANS={scans}\n')
        if seq:
            out.write(f'SEQ={seq}\n')
        if source:
            out.write(f'SOURCE={source}\n')
        mz_arr = spec.get('m/z array', [])
        int_arr = spec.get('intensity array', [])
        for mz_val, inte in zip(mz_arr, int_arr):
            out.write(f'{mz_val} {inte}\n')
        out.write('END IONS\n\n')

    # Append ResNet spectrum
    params = resnet_spec.get('params', {})
    title = params.get('title', params.get('TITLE', 'RESNET_PRED'))
    pepmass = params.get('pepmass', params.get('PEPMASS', 0.0))
    charge = params.get('charge', params.get('CHARGE', 2))
    if isinstance(pepmass, (list, tuple)):
        pepmass = float(pepmass[0]) if len(pepmass) > 0 else 0.0
    if isinstance(charge, (list, tuple)):
        charge = int(charge[0]) if len(charge) > 0 else 2
    elif isinstance(charge, str):
        import re
        m = re.search(r'(\d+)', charge)
        charge = int(m.group(1)) if m else 2

    out.write('BEGIN IONS\n')
    out.write(f'TITLE={title}\n')
    out.write(f'PEPMASS={pepmass}\n')
    out.write(f'CHARGE={charge}+\n')
    out.write('SOURCE=RESNET_PRED\n')
    mz_arr = resnet_spec.get('m/z array', [])
    int_arr = resnet_spec.get('intensity array', [])
    for mz_val, inte in zip(mz_arr, int_arr):
        out.write(f'{mz_val} {inte}\n')
    out.write('END IONS\n\n')

print(f"Wrote: {output_mgf}")
print(f"  {len(real_spectra)} real + 1 ResNet = {len(real_spectra) + 1} total")
