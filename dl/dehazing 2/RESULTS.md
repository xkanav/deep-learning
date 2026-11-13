# Results

> **Status: not yet run on real data.** Everything below is a template. Fill each table only by pasting the `--table` output of `python -m dehaze_fusion evaluate` on real RSDehazeNet / FCTF-Net outputs. Don't copy numbers from `VERIFICATION.txt`; those come from synthetic fake experts.

## Setup

- Dataset / subset: _TBD_
- Train / val / test images: _TBD_
- RSDehazeNet: checkpoint / training details: _TBD_
- FCTF-Net: checkpoint / training details: _TBD_
- Gate: epochs _TBD_, lr _TBD_, edge _TBD_, optimizer _TBD_, seed _TBD_

## Test set (all pixels)

| Method | PSNR (dB) | SSIM | MAE |
|---|---|---|---|
| Hazy input | — | — | — |
| RSDehazeNet | — | — | — |
| FCTF-Net | — | — | — |
| Average of both | — | — | — |
| **Gated fusion (ours)** | — | — | — |

## Dense vs thin haze (test set)

| Method | Dense PSNR | Dense SSIM | Thin PSNR | Thin SSIM |
|---|---|---|---|---|
| RSDehazeNet | — | — | — | — |
| FCTF-Net | — | — | — | — |
| Gated fusion | — | — | — | — |

## Ablations

| Variant | PSNR | SSIM |
|---|---|---|
| Full gate | — | — |
| edge = 0 | — | — |
| SGD instead of Adam | — | — |
| No density feature | — | — |

## Qualitative examples

_Add `infer --png` outputs and weight maps here._
