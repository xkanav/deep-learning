from pathlib import Path
import numpy as np


def load(path, target=True):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    required = ['cloudy', 'predictions', 'cloud_mask', 'valid', 'models', 'scene']
    if target:
        required += ['target']
    for k in required:
        if k not in d:
            raise ValueError(f'{path}: missing {k}')
    c = d['cloudy']
    p = d['predictions']
    if c.ndim != 3 or c.shape[0] != 13 or min(c.shape[1:]) < 2:
        raise ValueError('cloudy must be 13 x H x W, H,W >= 2')
    if p.ndim != 4 or p.shape[0] not in (2, 3) or p.shape[1:] != c.shape:
        raise ValueError('predictions must be K x 13 x H x W; K=2 or 3')
    if d['models'].shape != (p.shape[0],) or len(set(d['models'].tolist())) != p.shape[0]:
        raise ValueError('models must have K unique names in prediction order')
    if d['scene'].ndim != 0 or not str(d['scene'].item()):
        raise ValueError('scene must be a nonempty scalar identifier')
    for k in ['cloud_mask', 'valid']:
        if d[k].shape != c.shape[1:]:
            raise ValueError(f'{k} must be H x W')
    for k in ['cloudy', 'predictions', 'cloud_mask', 'valid'] + (['target'] if target else []):
        a = d[k]
        if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError(f'{k} must be finite and in [0,1], including invalid pixels')
    if 'target' in d and d['target'].shape != c.shape:
        raise ValueError('target shape differs from cloudy')
    if not np.isin(d['valid'], [0, 1]).all() or not d['valid'].any():
        raise ValueError('valid must be binary with at least one valid pixel')
    return d


def files(root):
    paths = sorted(Path(root).glob('*.npz'))
    if not paths:
        raise ValueError(f'No NPZ samples in {root}')
    return paths


def check_splits(root):
    seen = {}
    models = None
    counts = {}
    for split in ['train', 'val', 'test']:
        paths = files(Path(root) / split)
        for path in paths:
            d = load(path)
            scene = str(d['scene'].item())
            if scene in seen and seen[scene] != split:
                raise ValueError(f'Scene leakage: {scene} in {seen[scene]} and {split}')
            seen[scene] = split
            names = d['models'].tolist()
            if models is not None and names != models:
                raise ValueError('Model order differs across samples')
            models = names
        counts[split] = len(paths)
    return {'samples': counts, 'models': models, 'unique_scenes': len(seen)}
