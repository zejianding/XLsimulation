"""Cross-peptide cosine similarity analysis with V7 + mz>=50 filter."""
import os, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from pyteomics import mgf
import run_iterative_correction as ric
from mgf_to_embedding import _parse_precursor_mz, _parse_charge


def _write_spec(out, spec):
    params = spec.get('params', {})
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
    if source:
        out.write(f'SOURCE={source}\n')
    for mzv, intv in zip(spec.get('m/z array', []), spec.get('intensity array', [])):
        out.write(f'{mzv} {intv}\n')
    out.write('END IONS\n\n')


def extract_and_analyze(mgf_path, tag):
    """Extract embeddings from MGF + compute cosine stats. Returns stats dict."""
    npz_path = tag + '.npz'

    # Load model once
    model, config, device = ric.load_model('model.ckpt')

    embeddings = []
    with mgf.MGF(mgf_path) as reader:
        for spec in reader:
            params = spec.get('params', {})
            emb = ric.extract_embedding(
                spec.get('m/z array', np.array([], dtype=np.float32)),
                spec.get('intensity array', np.array([], dtype=np.float32)),
                _parse_precursor_mz(params), _parse_charge(params),
                model, config, device)
            embeddings.append(emb)

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    titles_arr = np.array([str(i) for i in range(len(embeddings))], dtype=np.str_)
    np.savez_compressed(npz_path, embeddings=emb_arr, titles=titles_arr,
                        precursor_mz=np.zeros(len(embeddings), dtype=np.float32),
                        charges=np.ones(len(embeddings), dtype=np.int32))

    # Compute stats
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    sim = (emb_arr / norms) @ (emb_arr / norms).T
    n = sim.shape[0]
    last_mean = (np.sum(sim[-1]) - sim[-1, -1]) / (n - 1)
    sims_no_self = sim[-1, :-1]
    real_means = [(np.sum(sim[i]) - sim[i, i]) / (n - 1) for i in range(n - 1)]

    return {
        'n_real': n - 1,
        'pred_vs_real_mean': float(last_mean),
        'real_avg': float(np.mean(real_means)),
        'min': float(sims_no_self.min()),
        'max': float(sims_no_self.max()),
    }


# ── 1: ATR pred vs ATR real ──
print('=== 1: ATR_pred vs ATR_real ===')
real_atr = list(mgf.MGF('QLEEAEEEATR_32.mgf'))
pred_atr = list(mgf.MGF('ATR_v7_mz50_top75.mgf'))[0]
with open('ATR_combined.mgf', 'w') as f:
    for s in real_atr:
        _write_spec(f, s)
    _write_spec(f, pred_atr)
r1 = extract_and_analyze('ATR_combined.mgf', 'ATR_combined')
print(f"  {r1}")

# ── 2: ATR pred vs AQR real ──
print('\n=== 2: ATR_pred vs AQR_real ===')
real_aqr = list(mgf.MGF('QLEEAEEEAQR_28.mgf'))[:-1]  # drop MODEL_PREDICTION
pred_atr = list(mgf.MGF('ATR_v7_mz50_top75.mgf'))[0]
with open('ATRpred_AQRreal.mgf', 'w') as f:
    for s in real_aqr:
        _write_spec(f, s)
    _write_spec(f, pred_atr)
r2 = extract_and_analyze('ATRpred_AQRreal.mgf', 'ATRpred_AQRreal')
print(f"  {r2}")

# ── 3: AQR pred vs ATR real ──
print('\n=== 3: AQR_pred vs ATR_real ===')
real_atr = list(mgf.MGF('QLEEAEEEATR_32.mgf'))
pred_aqr = list(mgf.MGF('v7_mz50_i005.mgf'))[0]
with open('AQRpred_ATRreal.mgf', 'w') as f:
    for s in real_atr:
        _write_spec(f, s)
    _write_spec(f, pred_aqr)
r3 = extract_and_analyze('AQRpred_ATRreal.mgf', 'AQRpred_ATRreal')
print(f"  {r3}")

# ── Summary ──
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f"{'Metric':<40} {'Value':>8}")
print('-' * 48)
print(f"{'ATR_real self-similarity':<40} {r1['real_avg']:>8.4f}")
print(f"{'ATR_pred vs ATR_real':<40} {r1['pred_vs_real_mean']:>8.4f}")
print(f"{'AQR_real self-similarity':<40} {r2['real_avg']:>8.4f}")
print(f"{'ATR_pred vs AQR_real (cross)':<40} {r2['pred_vs_real_mean']:>8.4f}")
print(f"{'AQR_pred vs ATR_real (cross)':<40} {r3['pred_vs_real_mean']:>8.4f}")
