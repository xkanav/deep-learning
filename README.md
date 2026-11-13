# Sentinel-2 adaptive prediction fusion

A complete lightweight ensemble trainer, evaluator and inference tool for combining **outputs** from two or three existing cloud-removal models. Recommended experiment: GLF-CR + HPN-CR, optionally HS2P. This is new fusion code, not a reimplementation of their networks. It does not bundle their weights, execute their inference scripts, or download satellite data. Run the original models separately in their supported environments, then export aligned predictions. No GPU or PyTorch is needed for the fusion stage.

## Quick smoke test

Python 3.10+; commands work in a terminal from this directory:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m fusion synthetic --out demo_data --models 3
python -m fusion check --data demo_data
python -m fusion train --data demo_data --epochs 30 --lr 0.05 --edge 0.1
python -m fusion evaluate --data demo_data/test --checkpoint runs/gate/best.npz --out runs/test_metrics.json
python -m fusion infer --sample demo_data/test/000.npz --checkpoint runs/gate/best.npz --out runs/prediction.npz
```

For a two-model smoke test, generate a separate directory with `--models 2` and use it in train/check/evaluate. Synthetic predictions are deliberately noisy arrays with one stronger expert. They are not outputs from GLF-CR, HPN-CR or HS2P. Passing this test validates code plumbing, not satellite restoration quality.

## What we changed

| Component | Existing single-model workflow | New code |
|---|---|---|
| Prediction | One network's estimate | Convex mixture of 2 or 3 existing estimates |
| Selection | One model everywhere | Per-pixel softmax weights using mask, disagreement, distance from cloudy observation, and mean intensity |
| Spectral treatment | Depends on upstream model | One weight per model per pixel, shared across all 13 bands |
| Clear regions | May be modified | Exact input copy where cloud mask is zero |
| Optimization | Original network losses | Fit only a tiny gate with masked MSE + horizontal/vertical gradient MSE |
| Baselines | Often separate published scores | Each expert, raw average, clear-preserving average and trained gate on identical samples |
| Data safety | Repository-specific loading | Explicit shape/range checks and scene-ID split checks |

No upstream architecture or checkpoint has been modified. This is output-level ensemble learning, not a sequential cascade or joint end-to-end training. It can improve quality only if the experts have complementary errors and the gate generalizes. Shared spectral weights avoid independently mixing different bands, but do not guarantee spectral correctness.

## Architecture and training

For K experts and pixel x:

`features = [1, mask, abs(P_k - mean(P)).mean(bands), abs(P_k - cloudy).mean(bands), P_k.mean(bands) for each k]`

`weights = softmax(features @ W)`

`fused = sum_k weights_k * P_k`

`output = mask * fused + (1-mask) * cloudy`

The cloud/shadow mask is provided externally and must be available at deployment. It must not be derived from the clear reference. Binary masks preserve clear regions exactly; soft masks blend proportionally. Incorrect masks can preserve cloud or overwrite real ground changes.

The gate has `(2+3K)*K` parameters: 48 for two experts and 99 for three. It uses deterministic seeded sample order and plain stochastic gradient descent. The objective is valid-pixel MSE plus `edge` times the sum of valid-neighbor gradient MSE in each spatial direction. Default learning rate 0.05, 30 epochs, edge weight 0.1 are starting choices, not published hyperparameters or proven optimal settings. Select using validation data. `best.npz` is selected by validation whole-image MSE; `history.json` records settings and training/validation history. Test targets never train the gate. Checkpoint is a small numeric NPZ, loaded without pickle.

## Real-data workflow

1. Obtain paired/registered cloudy Sentinel-2, Sentinel-1 and clear reference patches, e.g. SEN12MS-CR. Run each original expert with its own expected sensor preprocessing, band order and checkpoint. The fusion layer receives optical predictions, so SAR is used upstream rather than as a fusion input.
2. Reverse each model's output normalization into the SAME physical optical scale, then represent it as float reflectance in [0,1] under a declared clipping/scaling protocol. Do not simply divide every repository's outputs by the same number. Check all 13 bands and order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12. Never substitute RGB renders.
3. Resample/register everything onto the same grid before packing. Equal shapes do not establish geospatial alignment. Use common valid pixels across every input and prediction. Fill invalid pixels with finite zeros and exclude them via `valid`. This prototype expects all bands valid together.
4. Export each tensor as `.npy`, channel-first (13,H,W), masks as (H,W). Both mask and valid must be [0,1]; valid must be binary. Give each acquisition/geographic group a stable scene ID. Assign all overlapping patches and related acquisitions of a location to one split; code only checks the IDs supplied, not geographic overlap itself.
5. Pack samples as below, under `data/train`, `data/val`, `data/test`. Clear targets are required for training/evaluation but optional at inference. A scene can have many patches in the same split. Keep model names/order identical throughout.

```bash
python pack_sample.py --cloudy cloudy.npy --target clear.npy --predictions glf.npy hpn.npy hs2p.npy --models GLF-CR HPN-CR HS2P --mask cloud_shadow.npy --valid valid.npy --scene location_001 --out data/train/patch_001.npz
python -m fusion check --data data
python -m fusion train --data data --out runs/real_gate --epochs 30
python -m fusion evaluate --data data/test --checkpoint runs/real_gate/best.npz --out runs/real_test.json
```

Drop `hs2p.npy` and `HS2P` for two experts. Repeat the pack command for each patch/split. The check command reads every sample; it is a data smoke test before fitting. Expected NPZ fields: cloudy (13,H,W), predictions (K,13,H,W), target (13,H,W), cloud_mask (H,W), valid (H,W), models (K strings), scene (scalar string). This explicit export boundary permits combining different ML frameworks without fragile imports. No direct upstream adapters are claimed.

For inference pack without `--target`, retaining scene, mask, valid and model names. Output NPZ contains `prediction` (13,H,W) and `weights` (H,W,K). It is an array artifact without geospatial metadata; use the source raster transform/CRS when exporting GeoTIFF outside this package. Invalid output pixels remain unspecified and must be masked downstream.

## Evaluation and expected improvements

Read `CHANGES_AND_EXPERIMENTS.md`. JSON reports PSNR (data range 1), MAE, MSE and spectral angle in degrees, separately over all valid, cloudy (mask >= .5) and clear pixels. Regions use macro averages across patches, not pooled pixel statistics. Zero-length spectra are excluded from SAM and their valid count is recorded. Exact-match PSNR is null with `perfect_match=true`, since infinity is not standard JSON. SSIM is not implemented; do not compare these numbers directly with unrelated paper tables.

Training can be done on CPU. The code loads one patch's expert predictions at a time; feature/logit computations use float64. Upstream inference dominates compute and disk use: 3 experts' 13-band 256x256 float32 predictions consume about 9.75 MiB per patch before compression, excluding cloudy/target. Full-scene tiling, nodata conversion, GeoTIFF I/O, model inference and uncertainty calibration are not included.

## Files

- `fusion/data.py`: sample validation and split checks.
- `fusion/model.py`: adaptive gate, analytical gradients, checkpoint I/O.
- `fusion/metrics.py`: baselines and region-wise metrics.
- `fusion/__main__.py`: synthetic/check/train/evaluate/infer commands.
- `pack_sample.py`: NPY-to-NPZ data preparation.
- `tests/test_fusion.py`: numerical gradient, two/three experts, identity, checkpoint, validation and metrics checks.
- `.github/workflows/smoke.yml`: repeatable GitHub smoke workflow.
- `VERIFICATION.txt`: actual test output from this delivery.

Upload the contents of this directory to the root of your GitHub repository. No data, pretrained weights, or credentials are included. Confirm upstream model/data licenses and cite their papers when using their outputs. Original fusion source is provided for your project; no upstream source has been copied.

