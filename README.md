# NTIRE2026 Image Super-Resolution x4

This repository contains the `team17_RandomSeed42` pipeline for NTIRE 2026 Image Super-Resolution x4.

The current inference pipeline has two stages:

1. `MambaIRv2` performs 4x upsampling from the low-quality input.
2. `HYPIR` based on Stable Diffusion 2.1 performs 1x refinement on the stage-1 result.

The main entry point is [test.py](./test.py). The default model is `model_id=17`.

## Environment Requirements

- Python `3.10`
- Conda
- NVIDIA GPU with CUDA support
- Tested runtime: `PyTorch 2.0.1`

Required model files under [model_zoo/team17_RandomSeed42](./model_zoo/team17_RandomSeed42):

- `mambair_v2.pth`
- `hypir_sd21.pth`
- `stable-diffusion-2-1-base/`

## Installation Instructions

Create the environment:

```bash
conda create -n py310_ntire python=3.10 -y
conda activate py310_ntire
```

Install PyTorch first:

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

Then install the Mamba dependencies in this order:

```bash
python -m pip install causal-conv1d==1.0.0
python -m pip install mamba-ssm==1.0.1
```

Finally install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` intentionally does not include `causal-conv1d` or `mamba-ssm`, since these two packages must be installed beforehand in the order above.

If you specifically need the official CUDA 11.8 PyTorch wheels, replace the PyTorch install command above with the official `cu118` command from PyTorch before installing `requirements.txt`.

## Running Commands

### Validation

Run validation on a folder of low-quality x4 inputs:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --valid_dir /path/to/validation/LQ \
  --save_dir results
```

With the default `model_id=17`, outputs are written to:

- `results/17_RandomSeed42/valid/result`
- `results/17_RandomSeed42/valid/prompt`

### Test

Run test inference on a folder of low-quality x4 inputs:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --test_dir /path/to/test/LQ \
  --save_dir results
```

Outputs are written to:

- `results/17_RandomSeed42/test/result`
- `results/17_RandomSeed42/test/prompt`

### Evaluation

For validation data with available ground truth, run:

```bash
python eval.py \
  --output_folder results/17_RandomSeed42/valid/result \
  --target_folder /path/to/DIV2K_valid_HR \
  --metrics_save_path ./IQA_results \
  --gpu_ids 0
```

The evaluation script reports:

- `PSNR`
- `SSIM`
- `LPIPS`
- `DISTS`
- `CLIP-IQA`
- `MUSIQ`
- `MANIQA`
- `NIQE`
- `Total Score`

The weighted `Total Score` is computed as:

```text
(1 - LPIPS) + (1 - DISTS) + CLIP-IQA + MANIQA + MUSIQ / 100 + max(0, (10 - NIQE) / 10)
```

## Notes

- `test.py` defaults to `model_id=17`; you only need to specify `--model_id` when switching models.
- The current Mamba checkpoint is resolved from `model_zoo/team17_RandomSeed42/mambair_v2.pth`.
- The HYPIR weights are loaded from `model_zoo/team17_RandomSeed42/hypir_sd21.pth`.
