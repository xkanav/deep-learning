import argparse
import json
from pathlib import Path
import numpy as np
from .data import load, files, check_splits
from .model import Gate
from .metrics import evaluate, table

SYNTHETIC_NAMES = ['synthetic_thin_expert', 'synthetic_dense_expert', 'synthetic_noisy_expert']


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def smooth_field(rng, shape, cells=4):
    """Low-frequency random field in [0,1] (bilinear upsampling of a coarse grid)."""
    h, w = shape
    g = rng.random((cells + 1, cells + 1))
    y, x = np.linspace(0, cells, h), np.linspace(0, cells, w)
    y0, x0 = np.minimum(y.astype(int), cells - 1), np.minimum(x.astype(int), cells - 1)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    a, b = g[y0][:, x0], g[y0][:, x0 + 1]
    c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx


def synthetic_sample(rng, k, size=32):
    """Atmospheric scattering model I = J*t + A*(1-t) with non-uniform transmission.
    The 'experts' are hand-made corruptions of J with complementary errors - NOT real model outputs."""
    clear = np.stack([.15 + .6 * smooth_field(rng, (size, size), 6) for _ in range(3)])
    clear += rng.normal(0, .02, clear.shape)
    clear = np.clip(clear, 0, 1)
    t = .25 + .7 * smooth_field(rng, (size, size), 3)
    air = rng.uniform(.8, .95)
    hazy = clear * t[None] + air * (1 - t[None])
    leftover = np.clip((.6 - t[None]) / .35, 0, 1) * (1 - t[None])  # only where haze is dense (t < .6)
    preds = [clear + .8 * leftover * (air - clear) + rng.normal(0, .005, clear.shape),  # clean in thin haze, residual veil in dense haze
             1.06 * clear - .02 + rng.normal(0, .04, clear.shape),                       # handles dense haze, but bias + noise everywhere
             clear + rng.normal(0, .08, clear.shape)][:k]                                  # very noisy
    return (hazy.astype('float32'), clear.astype('float32'),
            np.clip(np.stack(preds), 0, 1).astype('float32'))


def main():
    p = argparse.ArgumentParser(prog='python -m dehaze_fusion')
    s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('synthetic', help='generate a labelled synthetic smoke-test dataset')
    q.add_argument('--out', default='demo_data')
    q.add_argument('--models', type=int, choices=[2, 3], default=2)
    q = s.add_parser('check')
    q.add_argument('--data', required=True)
    q = s.add_parser('train')
    q.add_argument('--data', required=True)
    q.add_argument('--out', default='runs/gate')
    q.add_argument('--epochs', type=int, default=30)
    q.add_argument('--lr', type=float, default=.05)
    q.add_argument('--edge', type=float, default=.1)
    q.add_argument('--seed', type=int, default=42)
    q.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    q = s.add_parser('evaluate')
    q.add_argument('--data', required=True)
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--out', default='runs/evaluation.json')
    q.add_argument('--table', help='also write a markdown results table here')
    q = s.add_parser('infer')
    q.add_argument('--sample', required=True)
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--out', default='runs/prediction.npz')
    q.add_argument('--png', help='also save the fused RGB image as PNG (needs Pillow)')
    a = p.parse_args()

    if a.command == 'synthetic':
        rng = np.random.default_rng(42)
        for split, n in [('train', 12), ('val', 4), ('test', 4)]:
            dest = Path(a.out) / split
            dest.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                hazy, clear, preds = synthetic_sample(rng, a.models)
                np.savez_compressed(dest / f'{i:03}.npz', hazy=hazy, target=clear, predictions=preds,
                                    valid=np.ones(hazy.shape[1:], 'float32'), scene=np.array(f'{split}_{i}'),
                                    models=np.array(SYNTHETIC_NAMES[:a.models]))
        print(json.dumps(check_splits(a.out)))
    elif a.command == 'check':
        print(json.dumps(check_splits(a.data), indent=2))
    elif a.command == 'train':
        if a.epochs < 1 or a.lr <= 0 or a.edge < 0:
            raise ValueError('epochs >= 1, lr > 0, edge >= 0 required')
        info = check_splits(a.data)
        gate = Gate(info['models'])
        dest = Path(a.out)
        dest.mkdir(parents=True, exist_ok=True)
        train = files(Path(a.data) / 'train')
        val = files(Path(a.data) / 'val')
        best, history = float('inf'), []
        rng = np.random.default_rng(a.seed)
        for epoch in range(a.epochs):
            losses = [gate.step(load(train[i]), a.lr, a.edge, a.optimizer) for i in rng.permutation(len(train))]
            report = evaluate(gate, val)
            score = next(r['mse'] for r in report['summary'] if r['method'] == 'gated' and r['region'] == 'all')
            if score < best:
                best = score
                gate.save(dest / 'best.npz')
            history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'val_mse': score})
        write(dest / 'history.json', {'settings': vars(a), 'data': info, 'history': history})
        print(json.dumps(history[-1]))
    elif a.command == 'evaluate':
        report = evaluate(Gate.restore(a.checkpoint), files(a.data))
        write(a.out, report)
        if a.table:
            Path(a.table).parent.mkdir(parents=True, exist_ok=True)
            Path(a.table).write_text(table(report['summary']) + '\n')
        print(table(report['summary']))
    elif a.command == 'infer':
        out, weights, _ = Gate.restore(a.checkpoint).forward(load(a.sample, target=False))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out, prediction=out.astype('float32'), weights=weights.astype('float32'))
        if a.png:
            from PIL import Image
            Image.fromarray((np.clip(out, 0, 1).transpose(1, 2, 0) * 255 + .5).astype('uint8')).save(a.png)
        print(a.out)


if __name__ == '__main__':
    main()
