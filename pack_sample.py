import argparse
from pathlib import Path
import numpy as np
from fusion.data import load

p = argparse.ArgumentParser(description='Pack co-registered, normalized CHW NPY outputs; does not run upstream models')
p.add_argument('--cloudy', required=True)
p.add_argument('--target')
p.add_argument('--predictions', nargs='+', required=True)
p.add_argument('--models', nargs='+', required=True)
p.add_argument('--mask', required=True)
p.add_argument('--valid', required=True)
p.add_argument('--scene', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
d = dict(cloudy=np.load(a.cloudy, allow_pickle=False),
         predictions=np.stack([np.load(f, allow_pickle=False) for f in a.predictions]),
         models=np.array(a.models), cloud_mask=np.load(a.mask, allow_pickle=False),
         valid=np.load(a.valid, allow_pickle=False), scene=np.array(a.scene))
if a.target:
    d['target'] = np.load(a.target, allow_pickle=False)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, **d)
try:
    load(a.out, target=bool(a.target))
except Exception:
    Path(a.out).unlink()
    raise
print(a.out)
