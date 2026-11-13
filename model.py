import numpy as np


def features(d):
    p = d['predictions'].astype(np.float64)
    mean = p.mean(axis=0)
    parts = [np.ones_like(d['cloud_mask']), d['cloud_mask']]
    for pred in p:
        parts += [np.abs(pred - mean).mean(axis=0),
                  np.abs(pred - d['cloudy']).mean(axis=0), pred.mean(axis=0)]
    return np.stack(parts, axis=-1)


class Gate:
    def __init__(self, models):
        self.models = list(models)
        self.w = np.zeros((2 + 3 * len(models), len(models)), dtype=np.float64)

    def forward(self, d):
        if d['models'].tolist() != self.models:
            raise ValueError('Model names/order differ from trained gate')
        x = features(d)
        logits = x @ self.w
        logits -= logits.max(axis=-1, keepdims=True)
        a = np.exp(logits)
        a /= a.sum(axis=-1, keepdims=True)
        fused = np.einsum('hwk,kchw->chw', a, d['predictions'])
        mask = d['cloud_mask'][None]
        out = mask * fused + (1 - mask) * d['cloudy']
        return out, a, x

    def step(self, d, lr=0.01, edge=0.1):
        out, a, x = self.forward(d)
        loss, grad = objective(out, d['target'], d['valid'], edge)
        g = np.einsum('chw,kchw->hwk', grad * d['cloud_mask'][None], d['predictions'])
        dz = a * (g - (a * g).sum(axis=-1, keepdims=True))
        dw = np.einsum('hwf,hwk->fk', x, dz)
        self.w -= lr * dw
        return loss

    def save(self, path):
        np.savez(path, weights=self.w, models=np.array(self.models), version=np.array(1))

    @classmethod
    def restore(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if z['version'].item() != 1:
                raise ValueError('Unsupported checkpoint version')
            obj = cls(z['models'].tolist())
            obj.w = z['weights'].copy()
        if obj.w.shape != (2 + 3 * len(obj.models), len(obj.models)) or not np.isfinite(obj.w).all():
            raise ValueError('Invalid checkpoint')
        return obj


def objective(out, target, valid, edge):
    err = out - target
    denom = valid.sum() * out.shape[0]
    loss = (err ** 2 * valid[None]).sum() / denom
    grad = 2 * err * valid[None] / denom
    for axis in [1, 2]:
        diff = np.diff(err, axis=axis)
        pairs = valid[1:, :] * valid[:-1, :] if axis == 1 else valid[:, 1:] * valid[:, :-1]
        n = max(float(pairs.sum() * out.shape[0]), 1)
        loss += edge * (diff ** 2 * pairs[None]).sum() / n
        g = 2 * edge * diff * pairs[None] / n
        if axis == 1:
            grad[:, 1:, :] += g
            grad[:, :-1, :] -= g
        else:
            grad[:, :, 1:] += g
            grad[:, :, :-1] -= g
    return float(loss), grad
