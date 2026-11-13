"""Pack a whole folder of images at once.

Expects files with the same stem in every folder, e.g.
  hazy/0001.png  clear/0001.png  rsdehazenet_out/0001.png  fctf_out/0001.png

Scene IDs come from --scene-regex applied to the file stem (default: whole stem). If several
patches come from one large image or location, make the regex capture that shared ID so the
packer keeps them in one split.

Example:
  python pack_folder.py --hazy SateHaze1k/thin/test/input --clear SateHaze1k/thin/test/target \
      --expert RSDehazeNet=outputs/rsdehazenet/thin_test --expert FCTF-Net=outputs/fctf/thin_test \
      --out data/test
"""
import argparse
import re
from pathlib import Path
from pack_sample import pack

EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy'}


def index(folder):
    out = {}
    for f in Path(folder).iterdir():
        if f.suffix.lower() in EXT:
            out[f.stem] = f
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hazy', required=True)
    p.add_argument('--clear', help='omit for inference-only packing')
    p.add_argument('--expert', action='append', required=True, metavar='NAME=DIR')
    p.add_argument('--scene-regex', default=r'(.+)', help='regex on stem; group 1 is the scene ID')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    experts = [e.split('=', 1) for e in a.expert]
    if any(len(e) != 2 for e in experts) or len(experts) not in (2, 3):
        p.error('give 2 or 3 --expert NAME=DIR arguments')
    hazy = index(a.hazy)
    clear = index(a.clear) if a.clear else None
    outs = {name: index(d) for name, d in experts}
    rx = re.compile(a.scene_regex)
    done, skipped = 0, []
    for stem in sorted(hazy):
        missing = [n for n, o in outs.items() if stem not in o] + (['clear'] if clear is not None and stem not in clear else [])
        if missing:
            skipped.append((stem, missing))
            continue
        m = rx.fullmatch(stem)
        if not m:
            p.error(f'--scene-regex does not match stem {stem!r}')
        pack(hazy[stem], [outs[n][stem] for n, _ in experts], [n for n, _ in experts],
             m.group(1), Path(a.out) / f'{stem}.npz', clear[stem] if clear is not None else None)
        done += 1
    print(f'packed {done} samples into {a.out}')
    for stem, missing in skipped:
        print(f'skipped {stem}: missing {", ".join(missing)}')
