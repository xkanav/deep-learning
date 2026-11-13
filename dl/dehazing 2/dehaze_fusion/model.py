import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PATCH_RADIUS = 3  # 7x7 dark-channel window


def min_filter(x, r=PATCH_RADIUS):
    padded = np.pad(x, r, mode='edge')
    return sliding_window_view(padded, (2 * r + 1, 2 * r + 1)).min(axis=(-1, -2))


def dark_channel(img, r=PATCH_RADIUS):
    """He et al. dark channel: min over RGB, then min over a local patch. img: 3 x H x W."""
    return min_filter(img.min(axis=0), r)


def n_features(k):
    return 2 + 4 * k


def features(d):
    p = d['predictions'].astype(np.float64)
    hazy = d['hazy'].astype(np.float64)
    mean = p.mean(axis=0)
    density = dark_channel(hazy)  # haze-density proxy computed from the input only
    parts = [np.ones_like(density), density]
    for pred in p:
        parts += [np.abs(pred - mean).mean(axis=0),   # disagreement with other experts
                  np.abs(pred - hazy).mean(axis=0),   # how much this expert changed the input
                  pred.mean(axis=0),                  # brightness
                  pred.min(axis=0)]                   # residual haze / veil in the output
    return np.stack(parts, axis=-1)


class Gate:
    def __init__(self, models):
        self.models = list(models)
        self.w = np.zeros((n_features(len(models)), len(models)), dtype=np.float64)
        self._m = np.zeros_like(self.w)  # Adam state (not saved; training restarts from zero moments)
        self._v = np.zeros_like(self.w)
        self._t = 0

    def forward(self, d):
        if d['models'].tolist() != self.models:
            raise ValueError('Model names/order differ from trained gate')
        x = features(d)
        logits = x @ self.w
        logits -= logits.max(axis=-1, keepdims=True)
        a = np.exp(logits)
        a /= a.sum(axis=-1, keepdims=True)
        out = np.einsum('hwk,kchw->chw', a, d['predictions'])
        return out, a, x

    def step(self, d, lr=0.05, edge=0.1, optimizer='adam'):
        out, a, x = self.forward(d)
        loss, grad = objective(out, d['target'], d['valid'], edge)
        g = np.einsum('chw,kchw->hwk', grad, d['predictions'])
        dz = a * (g - (a * g).sum(axis=-1, keepdims=True))
        dw = np.einsum('hwf,hwk->fk', x, dz)
        if optimizer == 'sgd':
            self.w -= lr * dw
        else:  # Adam, beta1=.9, beta2=.999
            self._t += 1
            self._m = .9 * self._m + .1 * dw
            self._v = .999 * self._v + .001 * dw ** 2
            mh, vh = self._m / (1 - .9 ** self._t), self._v / (1 - .999 ** self._t)
            self.w -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss

    def save(self, path):
        np.savez(path, weights=self.w, models=np.array(self.models),
                 kind=np.array('dehaze_gate'), version=np.array(1))

    @classmethod
    def restore(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if z['version'].item() != 1 or str(z['kind'].item()) != 'dehaze_gate':
                raise ValueError('Unsupported checkpoint')
            obj = cls(z['models'].tolist())
            obj.w = z['weights'].copy()
            obj._m, obj._v = np.zeros_like(obj.w), np.zeros_like(obj.w)
        if obj.w.shape != (n_features(len(obj.models)), len(obj.models)) or not np.isfinite(obj.w).all():
            raise ValueError('Invalid checkpoint')
        return obj


def objective(out, target, valid, edge):
    """Valid-pixel MSE + edge * gradient MSE in each spatial direction. Returns (loss, dL/dout)."""
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
