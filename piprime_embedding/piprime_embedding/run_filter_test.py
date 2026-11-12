"""Run full pipeline for each filtered V7 prediction variant.
To be run via: conda run -n respre python run_filter_test.py
"""
import sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from build_resnet_mgf import main as build_main  # noqa: E402

variants = [
    ('v7_filt_50_top200.mgf', 'v7_filt_50_top200'),
    ('v7_filt_50_top100.mgf', 'v7_filt_50_top100'),
    ('v7_filt_50_top80.mgf',  'v7_filt_50_top80'),
    ('v7_filt_100_top200.mgf','v7_filt_100_top200'),
]

for mgf_file, tag in variants:
    combined = f'QLEEAEEEAQR_{tag}.mgf'
    npz_path = f'QLEEAEEEAQR_{tag}.npz'
    xlsx_path = f'QLEEAEEEAQR_{tag}.xlsx'

    # Build combined MGF
    sys.argv = ['build_resnet_mgf.py', 'QLEEAEEEAQR_28.mgf', mgf_file, combined]
    # Simple approach: append filtered MGF to 28 real spectra from original
    from pyteomics import mgf
    spectra = list(mgf.MGF('QLEEAEEEAQR_28.mgf'))
    with open(combined, 'w') as out:
        for spec in spectra[:-1]:
            params = spec.get('params', {})
            title = params.get('title', params.get('TITLE', ''))
            pepmass = params.get('pepmass', params.get('PEPMASS', ''))
            charge = params.get('charge', params.get('CHARGE', ''))
            rt = params.get('rtinseconds', params.get('RTINSECONDS', ''))
            seq = params.get('seq', params.get('SEQ', ''))
            scans = params.get('scans', params.get('SCANS', ''))
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
            if rt: out.write(f'RTINSECONDS={rt}\n')
            out.write(f'CHARGE={charge}+\n')
            if scans: out.write(f'SCANS={scans}\n')
            if seq: out.write(f'SEQ={seq}\n')
            if source: out.write(f'SOURCE={source}\n')
            for mzv, intv in zip(spec.get('m/z array', []), spec.get('intensity array', [])):
                out.write(f'{mzv} {intv}\n')
            out.write('END IONS\n\n')
        with open(mgf_file) as f:
            out.write(f.read())

    # Extract embeddings
    sys.path.insert(0, '.')
    import run_iterative_correction as ric
    model, config, device = ric.load_model('model.ckpt')

    embeddings = []
    from mgf_to_embedding import _parse_precursor_mz, _parse_charge
    with mgf.MGF(combined) as reader:
        for i, spec in enumerate(reader):
            params = spec.get('params', {})
            precursor_mz = _parse_precursor_mz(params)
            precursor_charge = _parse_charge(params)
            emb = ric.extract_embedding(
                spec.get('m/z array', np.array([], dtype=np.float32)),
                spec.get('intensity array', np.array([], dtype=np.float32)),
                precursor_mz, precursor_charge, model, config, device,
            )
            embeddings.append(emb)

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    titles_arr = np.array([
        str(spec.get('params', {}).get('title', spec.get('params', {}).get('TITLE', f'index_{i}')))
        for i, spec in enumerate(mgf.MGF(combined))], dtype=np.str_)
    p_mz_arr = np.array([_parse_precursor_mz(spec.get('params', {}))
        for spec in mgf.MGF(combined)], dtype=np.float32)
    ch_arr = np.array([_parse_charge(spec.get('params', {}))
        for spec in mgf.MGF(combined)], dtype=np.int32)

    np.savez_compressed(npz_path, embeddings=emb_arr, titles=titles_arr,
                        precursor_mz=p_mz_arr, charges=ch_arr)

    # Cosine similarity
    from CosineDistance import analyze
    analyze(npz_path, xlsx_path)

    # Summary
    data = np.load(npz_path, allow_pickle=True)
    emb = data['embeddings']
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    sim = (emb / norms) @ (emb / norms).T
    n = sim.shape[0]
    last_mean = (np.sum(sim[-1]) - sim[-1, -1]) / (n - 1)
    real_means = [(np.sum(sim[i]) - sim[i, i]) / (n - 1) for i in range(n - 1)]
    print(f'[{tag}] pred_vs_real: {last_mean:.4f}  |  real_avg: {np.mean(real_means):.4f}')
