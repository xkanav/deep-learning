import numpy as np


def metrics(pred, target, mask):
    m = mask.astype(bool)
    if not m.any():
        return None
    p, t = pred[:, m].astype(float), target[:, m].astype(float)
    mse = float(np.mean((p - t) ** 2))
    norm = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)
    nz = norm > 1e-12
    sam = np.degrees(np.arccos(np.clip((p * t).sum(axis=0)[nz] / norm[nz], -1, 1)))
    return {'mae': float(np.abs(p - t).mean()), 'mse': mse,
            'psnr_db': float(-10 * np.log10(mse)) if mse > 0 else None,
            'perfect_match': mse == 0, 'sam_deg': float(sam.mean()) if nz.any() else None,
            'sam_pixels': int(nz.sum()), 'pixels': int(m.sum())}


def evaluate(gate, paths):
    from .data import load
    rows = []
    for path in paths:
        d = load(path)
        out, _, _ = gate.forward(d)
        candidates = {'cloudy': d['cloudy'], 'mean_raw': d['predictions'].mean(axis=0), 'gated': out}
        for name, p in zip(gate.models, d['predictions']):
            candidates[name] = p
        mask = d['cloud_mask'][None]
        candidates['mean_preserved'] = mask * candidates['mean_raw'] + (1-mask) * d['cloudy']
        for name, p in candidates.items():
            for region, m in [('all', d['valid']), ('cloud', d['valid'] * (d['cloud_mask'] >= .5)),
                              ('clear', d['valid'] * (d['cloud_mask'] < .5))]:
                rows.append({'sample': path.name, 'scene': str(d['scene'].item()), 'method': name,
                             'region': region, 'metrics': metrics(p, d['target'], m)})
    summary = []
    for name in candidates:
        for region in ['all', 'cloud', 'clear']:
            vals = [r['metrics'] for r in rows if r['method'] == name and r['region'] == region and r['metrics']]
            summary.append({'method': name, 'region': region, 'samples': len(vals),
                            **{k: float(np.mean([v[k] for v in vals if v[k] is not None]))
                               if any(v[k] is not None for v in vals) else None
                               for k in ['mae', 'mse', 'psnr_db', 'sam_deg']}})
    return {'aggregation': 'macro mean across nonempty sample regions; PSNR null for exact matches',
            'summary': summary, 'per_sample': rows}
