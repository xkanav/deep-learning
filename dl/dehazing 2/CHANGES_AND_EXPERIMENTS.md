# Changes and experiments

## Why output-level fusion

RSDehazeNet (channel-refinement residual blocks) and FCTF-Net (coarse encoder-decoder stage followed by a fine refinement stage, with channel attention) are built differently and are likely to fail differently. Two-stage coarse-to-fine networks tend to handle dense, non-uniform haze better, while residual refinement networks can preserve texture and colour better in thin haze. Neither claim is verified for these two models; the experiments below test it. Fusing outputs instead of merging architectures means:

- each model keeps its own training recipe and framework;
- the fusion stage trains in seconds on a CPU;
- the comparison against each single model is direct and fair, on identical images.

## Differences from the cloud-removal fusion

| | Cloud removal | Dehazing |
|---|---|---|
| Input bands | 13 Sentinel-2 bands | RGB |
| Where to fuse | Only inside the external cloud/shadow mask; clear pixels copied | Everywhere (haze is global) |
| Location feature | Cloud mask | Dark channel of the hazy input (haze-density proxy, needs no ground truth) |
| Extra per-expert feature | — | min-RGB of the output (residual veil indicator) |
| Optimizer | SGD | Adam (default) or SGD |
| Metrics | PSNR, MAE, MSE, SAM | PSNR, SSIM, MAE, MSE |
| Regions | all / cloud / clear | all / dense haze / thin haze |

## Experiments to run (fill RESULTS.md from these)

1. **Main comparison.** Train the gate on the train split, select on val, and report on test: hazy input, RSDehazeNet, FCTF-Net, raw average, gated.
2. **Per haze level.** If you use SateHaze1k, repeat for the thin, moderate and thick subsets separately, and also train one gate on all three.
3. **Region analysis.** Compare dense versus thin rows in the evaluate output. This shows whether the gate is routing (each expert winning where it is stronger) or just averaging.
4. **Ablations.** `--edge 0` versus `0.1`; SGD versus Adam; remove the density feature (edit `features()`); two seeds.
5. **Qualitative.** Use `infer --png` on a few test images and show them next to both experts. `weights` in the output NPZ can be shown as a heatmap of which model was chosen where.
6. **Failure cases.** Report images where the gate is worse than the best single expert.

## Known limitations

- The gate only re-weights existing outputs. It cannot fix an error both models make.
- The dark-channel prior is unreliable on bright surfaces (snow, clouds, white roofs), so density there is overestimated.
- Outputs must be pixel-aligned. Models that change resolution or crop differently must be brought onto the input grid first.
- Synthetic smoke-test numbers in `VERIFICATION.txt` describe hand-made fake experts and are not results.
