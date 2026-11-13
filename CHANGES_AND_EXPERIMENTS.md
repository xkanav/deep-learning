# Proposed research contribution and estimated changes

We add mask-aware, spectrally shared adaptive prediction fusion on top of existing trained cloud-removal models. GLF-CR + HPN-CR is the first experiment; add HS2P as a third expert only if it adds complementary information. The document motivated these candidates; this delivery does not reproduce or alter their original models.

## What is implemented

Two- and three-expert fusion, learned pixel-level weights, clear-pixel preservation, valid-pixel and gradient supervision, train/validation/test scene-ID checks, checkpoint selection, prediction exports, metrics, packing tool, synthetic data generator, tests and GitHub CI. No trained satellite expert or real satellite samples were provided. The package is fully runnable for fusion once normalized expert outputs are available. Upstream inference remains a separate prerequisite.

## Why it might help

Different fusion architectures may make different errors on edges, dense clouds and spectral intensities. A learned mixture can prefer an expert conditionally rather than averaging everything equally. A mask limits unnecessary modifications to clear areas. The gradient term encourages agreement with reference boundaries. A mixture cannot invent a correct value outside the experts' convex range at a pixel/band and cannot solve common errors shared by every expert. The small gate has no spatial receptive field beyond statistics at the pixel, so it is intentionally a conservative initial experiment.

## Estimates, not results

There is no defensible measured PSNR/SSIM gain without real training and held-out testing. For project planning only, use a target of **+0.2 to +0.8 dB PSNR over the strongest single expert**, with no increase in SAM or clear-region MAE. This is a speculative success criterion, not an evidence-backed expected range. Zero gain or degradation is possible. A +0.2 dB gain corresponds to about 4.5% lower MSE and +0.8 dB to about 16.8% lower MSE at fixed data range. Do not put these estimates in a results table.

At mask=0 the clear-preserving gate changes the cloudy observation by exactly zero. This is a guaranteed code property, not proof that clear-reference MAE will be zero: clear observations may differ between acquisition dates.

## Ablations to run on the same held-out scenes

| Run | Experts | Fusion | Purpose |
|---|---|---|---|
| A | GLF only | Original output | Single-model baseline |
| B | HPN only | Original output | Single-model baseline |
| C | HS2P only | Original output | Optional third-model baseline |
| D | 2 or 3 | Raw equal average | Is averaging enough? |
| E | Same as D | Average + clear preservation | Is the mask responsible for gains? |
| F | Same as D | Learned gate, edge=0 | Is adaptive selection useful? |
| G | Same as D | Learned gate, edge=0.1 | Does structure supervision help? |

Train F and G with separate output directories. Repeat two-expert and three-expert versions with separately packed data. Only add the third model if it improves validation metrics enough to justify its inference cost. All selected experts must have avoided the evaluation scenes during their own training; a pretrained expert trained on your test geography invalidates a claimed unseen-scene benchmark.

Record whole-image/cloud/clear PSNR, MAE, SAM, cloud fraction, geography, runtime, peak memory and checkpoint provenance. Evaluation reports the implemented metrics; timing, SSIM and confidence intervals require additional measurement. For publication, aggregate uncertainty by geographic scene rather than treating neighboring pixels as independent, and compare across several geographic partitions. Inspect common-scale RGB crops and spectral curves because temporal reference differences can distort pixel metrics.

## Presentation wording

“We propose a lightweight adaptive ensemble of existing Sentinel-2 cloud-removal models. It learns pixel-level model weights while preserving observations marked clear. Our current implementation passes synthetic code checks. Whether it improves real restoration quality remains to be tested on geographically held-out satellite data.”
