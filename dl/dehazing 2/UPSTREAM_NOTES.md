# Upstream repositories: what is there and how to get outputs

Checked against the public repositories in November 2026.

## RSDehazeNet — github.com/tianqiong123/RSDehazeNet

- Contents: `model/RSDehazeNet.py` (76 lines), an empty README and a licence file.
- The file defines four building blocks only: `CRB` (channel attention), `RCRB` (residual channel refinement block, 32 channels), `TRCRB` (three stacked RCRBs) and `FFB` (feature fusion block, 384→96→32 channels). **There is no complete network class, no training or test script, no data loader and no pretrained weights.**
- Bug: `FFB.forward` calls `torch.concat([x1, x2, x3, x4])` without `dim=1`, so it concatenates along the batch dimension and the following 384-channel conv will fail. It should be `torch.cat([x1, x2, x3, x4], dim=1)`.
- To get outputs you must assemble the full network from these blocks following the paper, then train it. A reasonable route is to reuse FCTF-Net's training loop and data loaders with this network swapped in.

## FCTF-Net — github.com/Neooolee/FCTF-Net

- Contents: `code/` (net, losses, perceptual loss, residual dense block, `train_first.py` for stage one, `train.py` for the full model, `test.py`) and a tiny `data/` sample (3 training pairs, 1 test pair). The README lists the dataset as "TO DO". Code is derived from GridDehazeNet.
- Requirements in the README: Python 3.7, PyTorch 1.4, CUDA 9. Recent PyTorch generally works after replacing `Variable(...)` with plain tensors.
- `test.py` loads `./best_psnr.pth`, which is **not included**, so the model must be trained first.
- Inputs are normalized with mean/std 0.5. The model returns `(coarse, final)`; `test.py` clamps `final` to [0,1] and saves uint8 PNGs to `./results/` with the input file name. Those PNGs can be passed straight to `pack_folder.py`.

## Export checklist for both models

- Save one RGB PNG per hazy image, using the same file stem as the hazy input.
- Same height and width as the hazy input: no padding left in, no resizing.
- Values in [0,1] before converting to uint8 (undo the 0.5/0.5 normalization).
- Produce outputs for **train, val and test**, because the gate needs expert outputs on training images too. Expert outputs on their own training images can look better than on unseen images. If possible, train the experts and the gate on different subsets, or at least note this in the report.
