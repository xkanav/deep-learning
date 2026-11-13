"""Pack one hazy image, its clear reference and the dehazed outputs of 2-3 existing models into an NPZ.
Accepts .npy (3,H,W float in [0,1]) or 8/16-bit RGB images (.png/.jpg/.tif). Does NOT run any model."""
import argparse
from pathlib import Path
import numpy as np


def read_rgb(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False).astype('float32')
        if a.ndim == 3 and a.shape[-1] == 3 and a.shape[0] != 3:
            a = a.transpose(2, 0, 1)
        return a
    from PIL import Image
    img = Image.open(path)
    a = np.asarray(img)
    if a.ndim == 2:
        raise ValueError(f'{path}: expected an RGB image, got single channel')
    a = a[..., :3]
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return (a.astype('float32') / scale).transpose(2, 0, 1)


def pack(hazy, predictions, models, scene, out, target=None, valid=None):
    from dehaze_fusion.data import load
    h = read_rgb(hazy)
    d = dict(hazy=h, predictions=np.stack([read_rgb(f) for f in predictions]),
             models=np.array(models), scene=np.array(scene),
             valid=np.load(valid, allow_pickle=False).astype('float32') if valid else np.ones(h.shape[1:], 'float32'))
    if target:
        d['target'] = read_rgb(target)
    shapes = {k: d[k].shape[-2:] for k in ['hazy', 'predictions'] + (['target'] if target else [])}
    if len(set(shapes.values())) != 1:
        raise ValueError(f'Spatial sizes differ (crop/resize outputs to the input grid first): {shapes}')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    try:
        load(out, target=bool(target))
    except Exception:
        out.unlink()
        raise
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hazy', required=True)
    p.add_argument('--target', help='clear reference; required for train/val/test, omit for inference')
    p.add_argument('--predictions', nargs='+', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--valid', help='optional H x W binary .npy; default all pixels valid')
    p.add_argument('--scene', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if len(a.predictions) != len(a.models):
        p.error('--predictions and --models must have the same length')
    print(pack(a.hazy, a.predictions, a.models, a.scene, a.out, a.target, a.valid))
