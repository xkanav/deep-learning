import argparse
import json
from pathlib import Path
import numpy as np
from .data import load, files, check_splits
from .model import Gate
from .metrics import evaluate


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('synthetic')
    q.add_argument('--out', default='demo_data')
    q.add_argument('--models', type=int, choices=[2, 3], default=3)
    q = s.add_parser('check')
    q.add_argument('--data', required=True)
    q = s.add_parser('train')
    q.add_argument('--data', required=True)
    q.add_argument('--out', default='runs/gate')
    q.add_argument('--epochs', type=int, default=30)
    q.add_argument('--lr', type=float, default=.05)
    q.add_argument('--edge', type=float, default=.1)
    q = s.add_parser('evaluate')
    q.add_argument('--data', required=True)
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--out', default='runs/evaluation.json')
    q = s.add_parser('infer')
    q.add_argument('--sample', required=True)
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--out', default='runs/prediction.npz')
    a = p.parse_args()
    if a.command == 'synthetic':
        rng = np.random.default_rng(42)
        for split, n in [('train', 8), ('val', 3), ('test', 3)]:
            dest = Path(a.out) / split
            dest.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                target = rng.uniform(.05, .7, (13, 16, 16)).astype('float32')
                mask = (rng.random((16, 16)) > .4).astype('float32')
                cloudy = target * (1-mask) + (.25 * target + .7) * mask
                preds = np.stack([np.clip(target + rng.normal(0, sd, target.shape), 0, 1)
                                  for sd in [.02, .12, .20][:a.models]]).astype('float32')
                np.savez_compressed(dest / f'{i:03}.npz', cloudy=cloudy, target=target, predictions=preds,
                                    cloud_mask=mask, valid=np.ones_like(mask), scene=np.array(f'{split}_{i}'),
                                    models=np.array(['synthetic_good', 'synthetic_noisy', 'synthetic_noisier'][:a.models]))
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
        best = float('inf')
        history = []
        rng = np.random.default_rng(42)
        for epoch in range(a.epochs):
            losses = [gate.step(load(train[i]), a.lr, a.edge) for i in rng.permutation(len(train))]
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
        print(json.dumps(report['summary'], indent=2))
    elif a.command == 'infer':
        out, weights, _ = Gate.restore(a.checkpoint).forward(load(a.sample, target=False))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out, prediction=out.astype('float32'), weights=weights.astype('float32'))
        print(a.out)


if __name__ == '__main__':
    main()
