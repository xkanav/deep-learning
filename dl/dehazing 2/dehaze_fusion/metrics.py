import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .model import dark_channel

C1, C2 = 0.01 ** 2, 0.03 ** 2  # data range 1


def _gauss(size=11, sigma=1.5):
    x = np.arange(size) - size // 2
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _blur(img, k):
    r = len(k) // 2
    p = np.pad(img, r, mode='symmetric')
    p = sliding_window_view(p, len(k), axis=0) @ k
    return sliding_window_view(p, len(k), axis=1) @ k


def ssim_map(pred, target):
    """Per-pixel SSIM (Gaussian 11x11, sigma 1.5), averaged over RGB. Inputs 3 x H x W in [0,1]."""
    k = _gauss()
    maps = []
    for x, y in zip(pred.astype(float), target.astype(float)):
        mx, my = _blur(x, k), _blur(y, k)
        sxx = _blur(x * x, k) - mx * mx
        syy = _blur(y * y, k) - my * my
        sxy = _blur(x * y, k) - mx * my
        maps.append(((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2)))
    return np.mean(maps, axis=0)


def metrics(pred, target, mask, ssim=None):
    m = mask.astype(bool)
    if not m.any():
        return None
    p, t = pred[:, m].astype(float), target[:, m].astype(float)
    mse = float(np.mean((p - t) ** 2))
    return {'mae': float(np.abs(p - t).mean()), 'mse': mse,
            'psnr_db': float(-10 * np.log10(mse)) if mse > 0 else None,
            'perfect_match': mse == 0,
            'ssim': float(ssim[m].mean()) if ssim is not None else None,
            'pixels': int(m.sum())}


def regions(d):
    """all / dense / thin split by the hazy input's dark channel, at that sample's median over valid pixels."""
    valid = d['valid'].astype(bool)
    density = dark_channel(d['hazy'].astype(float))
    med = np.median(density[valid])
    return [('all', valid), ('dense', valid & (density >= med)), ('thin', valid & (density < med))]


KEYS = ['mae', 'mse', 'psnr_db', 'ssim']


def evaluate(gate, paths):
    from .data import load
    rows = []
    names = None
    for path in paths:
        d = load(path)
        out, _, _ = gate.forward(d)
        candidates = {'hazy_input': d['hazy'], 'mean_raw': d['predictions'].mean(axis=0), 'gated': out}
        for name, p in zip(gate.models, d['predictions']):
            candidates[name] = p
        names = list(candidates)
        regs = regions(d)
        for name, p in candidates.items():
            smap = ssim_map(p, d['target'])
            for region, m in regs:
                rows.append({'sample': path.name, 'scene': str(d['scene'].item()), 'method': name,
                             'region': region, 'metrics': metrics(p, d['target'], m, smap)})
    summary = []
    for name in names:
        for region in ['all', 'dense', 'thin']:
            vals = [r['metrics'] for r in rows if r['method'] == name and r['region'] == region and r['metrics']]
            summary.append({'method': name, 'region': region, 'samples': len(vals),
                            **{k: float(np.mean([v[k] for v in vals if v[k] is not None]))
                               if any(v[k] is not None for v in vals) else None for k in KEYS}})
    return {'aggregation': 'macro mean across nonempty sample regions; PSNR null for exact matches; '
                           'SSIM map computed on the whole image then averaged inside each region',
            'summary': summary, 'per_sample': rows}


def table(summary):
    """Markdown table of the summary, for pasting into RESULTS.md."""
    lines = ['| Method | Region | PSNR (dB) | SSIM | MAE | N |', '|---|---|---|---|---|---|']
    fmt = lambda v, f: '—' if v is None else format(v, f)
    for r in summary:
        lines.append(f"| {r['method']} | {r['region']} | {fmt(r['psnr_db'], '.2f')} | "
                     f"{fmt(r['ssim'], '.4f')} | {fmt(r['mae'], '.4f')} | {r['samples']} |")
    return '\n'.join(lines)
