# BCGFormer: Band-Contextual Gating Transformer for Hyperspectral Image Classification

## Overview

BCGFormer is a CNN-Transformer hybrid for hyperspectral image (HSI) classification. It introduces three novel components:

1. **Band-Contextual Gating (BCG)** — Conv1d spectral neighbourhood context with learnable temperature sharpening for adaptive band selection
2. **Spectral Summary Token** — BCG-gated spectral representation injected as a learned token that attends jointly with spatial tokens, forming a spectral-spatial bridge
3. **Linear Attention with RoPE** — ELU-kernel linear attention (O(N) complexity) with single-pass Rotary Positional Encoding

## Repository Structure

```
├── models/
│   └── model.py          # BCGFormer architecture
├── dataset/              # HSI benchmark datasets (.mat files) (Add this folder and download the datasets)
├── main.py               # Training, evaluation, and classification map generation
├── evaluation.py         # Metrics and GFLOPs utilities
└── requirements.txt
```

## Datasets

The following benchmark datasets are supported:

| Dataset | Abbreviation |
|---|---|
| Pavia University | `pavia` |
| Houston 2013 | `houston` |
| Houston 2018 | `houston18` |
| Salinas | `salinas` |
| Indian Pines | `indiana` |
| WHU-Hi-HongHu | `honghu` |
| WHU-Hi-HanChuan | `hanchuan` |
| WHU-Hi-LongKou | `longkou` |

Place dataset `.mat` files under `./dataset/`. WHU datasets go under `./dataset/WHU/`.

Links:
https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
https://ieee-dataport.org/documents/hyperspectral-dataset-0
https://zenodo.org/records/15771735
https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/
https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/


Houston 2013 uses the official GRSS 2013 fixed split when `Houston13_7gt_test.mat` is present, otherwise falls back to a spatial-safe random split.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Basic training:**
```bash
python main.py --dataset pavia --epochs 100 --num_runs 5
```

**With classification map:**
```bash
python main.py --dataset pavia --epochs 100 --num_runs 5 --save_maps
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `houston` | Dataset to use |
| `--train_samples` | `200` | Training samples per class |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--num_runs` | `1` | Number of independent runs |
| `--save_path` | `./results_avg` | Output directory |
| `--save_maps` | `False` | Save classification map after last run |

## Training Protocol

- Spatial-safe split: N samples per class with spatial exclusion within class
- AdamW optimiser with cosine LR schedule and 10% linear warmup
- Label smoothing 0.1
- Best model checkpoint by validation accuracy
- Results reported as mean ± std over multiple runs
