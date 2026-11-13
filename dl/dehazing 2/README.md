# Remote-sensing dehazing: adaptive prediction fusion

A lightweight ensemble trainer, evaluator and inference tool that combines the **outputs** of two (optionally three) existing dehazing models. Recommended pair: **RSDehazeNet + FCTF-Net**. This is new fusion code, not a reimplementation of either network. It does not bundle their weights, run their inference scripts or download data. Run the original models separately, export their dehazed images, then fuse them here. No GPU or PyTorch is needed for the fusion stage.

It follows the same design as the cloud-removal fusion in this repository, adapted for haze: haze covers the whole image rather than masked regions, so there is no clear-region copy, and the cloud mask is replaced by a haze-density feature computed from the hazy input.

## Quick smoke test

Python 3.10+, from this directory:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m dehaze_fusion synthetic --out demo_data --models 2
python -m dehaze_fusion check --data demo_data
python -m dehaze_fusion train --data demo_data --epochs 30
python -m dehaze_fusion evaluate --data demo_data/test --checkpoint runs/gate/best.npz --out runs/test_metrics.json
python -m dehaze_fusion infer --sample demo_data/test/000.npz --checkpoint runs/gate/best.npz --out runs/prediction.npz --png runs/prediction.png
```

The synthetic data is hazed with the atmospheric scattering model `I = J·t + A·(1−t)` using non-uniform transmission. The "experts" are hand-made corruptions of the clear image with deliberately complementary errors: one leaves a residual veil only in dense haze, the other has a uniform bias plus noise. **They are not outputs of RSDehazeNet or FCTF-Net, and the numbers they produce must not be reported as model results.** Passing this test checks the code plumbing and that the gate can learn routing. It says nothing about real dehazing quality. See `VERIFICATION.txt` for the actual output of this run.

## What we changed

| Component | Existing single-model workflow | New code |
|---|---|---|
| Prediction | One network's dehazed image | Convex per-pixel mixture of 2 or 3 existing dehazed images |
| Selection | One model everywhere | Softmax weights from haze density (dark channel of input), expert disagreement, change from input, brightness and residual-veil (min-RGB) per expert |
| Colour treatment | Depends on upstream model | One weight per model per pixel, shared across R, G, B, so the gate cannot invent new hues |
| Optimization | Original network losses | Fit only a tiny gate: valid-pixel MSE + horizontal/vertical gradient MSE, Adam |
| Baselines | Separate published scores | Hazy input, each expert, raw average and trained gate, all on identical samples |
| Evaluation | Whole-image PSNR/SSIM | PSNR, SSIM, MAE over all pixels **and** split into dense-haze / thin-haze halves |
| Data safety | Repository-specific loading | Shape/range checks, scene-ID split-leakage checks |

No upstream architecture or checkpoint is modified. This is output-level ensembling, not a cascade or joint end-to-end training. It can only help if the two models make different mistakes in different places (for example, one better in thin haze and the other in dense haze), and the gate generalizes. If one model is better everywhere, expect the gate to roughly match it, not beat it.

## Architecture and training

For K experts with outputs `P_k` (3×H×W), hazy input `I`, and pixel x:

`density = darkchannel(I)` (min over RGB, then a 7×7 min filter)

`features = [1, density, |P_k − mean(P)|, |P_k − I|, mean_rgb(P_k), min_rgb(P_k) for each k]` (per-pixel, channel-averaged where applicable)

`weights = softmax(features @ W)`

`output = Σ_k weights_k · P_k`

The gate has `(2+4K)·K` parameters: 20 for two experts and 42 for three. Training visits samples in seeded random order and uses Adam (`--optimizer sgd` is available). The objective is valid-pixel MSE plus `edge` × gradient MSE in each spatial direction. Defaults (lr 0.05, 30 epochs, edge 0.1) are starting points, not tuned values; select on validation data. `best.npz` is chosen by validation MSE, and `history.json` records settings and per-epoch losses. Test targets never touch training. Checkpoints are small NPZ files loaded without pickle.

## Real-data workflow

1. **Data.** Use paired hazy/clear remote-sensing images, for example SateHaze1k (thin, moderate and thick subsets) or RICE-I. Keep the dataset's official train/val/test split if it has one.
2. **Run each model.** Produce dehazed images for every hazy image in train, val and test with each model, using its own preprocessing and checkpoint. See `UPSTREAM_NOTES.md`: RSDehazeNet's repository contains only building blocks, and FCTF-Net has no released weights, so both need training first.
3. **Same scale, same grid.** Every output must be RGB, the same size as the hazy input, and in [0,1]. FCTF-Net's `test.py` already clamps to [0,1] and saves uint8 PNGs. Undo any `Normalize(0.5, 0.5)` before saving other models' outputs. Don't resize one model's output to match another's crop; crop all inputs the same way first.
4. **Scene IDs.** Patches cut from the same large image or location must share a scene ID so they stay in one split. `check` refuses leakage across splits.
5. **Pack.** Either one image at a time:

```bash
python pack_sample.py --hazy hazy/0001.png --target clear/0001.png \
    --predictions rsdehazenet/0001.png fctf/0001.png --models RSDehazeNet FCTF-Net \
    --scene 0001 --out data/train/0001.npz
```

or a whole folder (same file stem in every folder):

```bash
python pack_folder.py --hazy SateHaze1k/thin/train/input --clear SateHaze1k/thin/train/target \
    --expert RSDehazeNet=outputs/rsdehazenet/thin_train --expert FCTF-Net=outputs/fctf/thin_train \
    --out data/train
# repeat for val and test
python -m dehaze_fusion check --data data
python -m dehaze_fusion train --data data --out runs/real_gate
python -m dehaze_fusion evaluate --data data/test --checkpoint runs/real_gate/best.npz \
    --out runs/real_test.json --table runs/real_test.md
```

Then paste `runs/real_test.md` into `RESULTS.md`.

NPZ fields: `hazy` (3,H,W), `predictions` (K,3,H,W), `target` (3,H,W), `valid` (H,W, binary), `models` (K strings), `scene` (scalar string). `target` is optional at inference. Model names and order must be identical across all samples. Inputs may be `.png/.jpg/.tif` (8- or 16-bit) or `.npy`.

## Evaluation

`evaluate` reports PSNR (data range 1), SSIM (Gaussian 11×11, σ=1.5, averaged over RGB), MAE and MSE for each method, over all valid pixels and over dense/thin-haze halves. The halves are split at each image's median dark-channel value of the hazy input, so they need no ground truth. Values are macro-averaged across images. SSIM here is computed in RGB on [0,1] floats. Papers often compute it on grayscale or the Y channel, so don't compare these numbers directly with published tables unless you match their protocol.

## Files

- `dehaze_fusion/data.py`: sample validation and split-leakage checks.
- `dehaze_fusion/model.py`: dark channel, features, gate with analytical gradients, Adam/SGD, checkpoint I/O.
- `dehaze_fusion/metrics.py`: PSNR/SSIM/MAE, haze-density regions, baselines, markdown table.
- `dehaze_fusion/__main__.py`: `synthetic`, `check`, `train`, `evaluate`, `infer` commands.
- `pack_sample.py` / `pack_folder.py`: image/NPY → NPZ packing.
- `tests/test_dehaze_fusion.py`: gradient check, convexity, dark channel, checkpoint, validation, metrics, PNG packing.
- `CHANGES_AND_EXPERIMENTS.md`: design rationale and the experiments to run.
- `RESULTS.md`: results template, to fill in from real runs only.
- `UPSTREAM_NOTES.md`: state of the two upstream repositories and how to get outputs from them.
- `VERIFICATION.txt`: actual test and smoke-test output from this delivery.
- `dehaze-smoke.yml`: GitHub Actions workflow; move it to `.github/workflows/` at the repository root.

No data, weights or credentials are included, and no upstream source has been copied. Check upstream licences and cite both papers when you use their outputs. FCTF-Net: Y. Li and X. Chen, "A Coarse-to-Fine Two-Stage Attentive Network for Haze Removal of Remote Sensing Images," IEEE GRSL, 2020, doi:10.1109/LGRS.2020.3006533.
