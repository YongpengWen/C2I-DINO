# C2I-DINO: Category-to-Instance Discrimination for Remote Sensing Visual Grounding

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-3.3.0-green.svg)](https://github.com/open-mmlab/mmdetection)

## Overview

Remote sensing visual grounding (RSVG) aims to localize objects described by natural-language expressions in large-scale aerial images. Compared with conventional visual grounding, RSVG contains many repeated objects and long-range spatial relations, requiring a model to preserve the requested category cue while distinguishing the referred instance from similar objects.

To address this challenge, we propose **C2I-DINO**, a category-to-instance framework built on Grounding DINO. It introduces **Category-Focused Prompting (CFP)**, which retrieves learnable category-specific prompts and appends them to the projected text memory before cross-modal fusion, and **Spatial Instance Learning (SIL)**, which uses ranking and suppression losses to separate the matched query from high-response queries at incorrect locations during training. Both modules retain the original image-expression input and inference interface. SIL is confined to training and leaves inference unchanged.

Extensive experiments on RSVG, OPT-RSVG, and DIOR-RSVG demonstrate that C2I-DINO achieves state-of-the-art performance on most reported metrics.

## Method Overview

The figure below summarizes C2I-DINO. Category-Focused Prompting (CFP) strengthens category cues with class-specific learnable suffixes, while Spatial Instance Learning (SIL) suppresses confusing responses from incorrect instances during training.

![C2I-DINO framework](assets/framework.png)

## Key Features

- Grounding DINO with a Swin-T visual backbone.
- Class-specific learnable text suffixes for stronger category representations.
- Spatial Instance Learning (SIL) for suppressing confusing nearby proposals.
- Training and evaluation configurations for RSVG, DIOR-RSVG, and OPT-RSVG.
- Data conversion, evaluation, visualization, and experiment scripts.

## Repository Structure

```text
.
├── configs/                         # Dataset and experiment configurations
│   ├── rsvg/
│   ├── dior_rsvg/
│   └── opt_rsvg/
├── mmdet/                           # MMDetection framework and customized modules
├── tools/                           # Training, evaluation, conversion, and visualization tools
├── scripts/                         # Final training and testing scripts for all three datasets
├── pretrained/                      # Place downloaded pretrained models here
└── assets/                          # Method overview and result figures
```

## Datasets

The experiments use RSVG, DIOR-RSVG, and OPT-RSVG. Please download each dataset from its official source and comply with its license. Datasets and generated annotations are not included in this repository.

The expected layout is:

```text
datasets/
├── RSVG/rsvg/
│   ├── images/
│   ├── mdetr_annotations/
│   └── odvg_ann/
├── DIOR-RSVG/
│   ├── JPEGImages/
│   ├── Annotations/
│   ├── mdetr_annotations_official_split/
│   └── odvg_ann_official_split/
└── opt-rsvg/
    ├── Image/
    ├── Annotations/
    ├── split/
    ├── mdetr_annotations_official_split/
    └── odvg_ann_official_split/
```

The repository expects the converted MDETR/RefCOCO-style annotations and ODVG JSONL files under each dataset directory. If you start from the original XML annotations and split files, run:

```bash
# Run from the repository root.
python tools/dior_rsvg/dior_rsvg_xml_to_grounding.py \
  --data-root datasets/DIOR-RSVG

python tools/opt_rsvg/opt_rsvg_xml_to_mdetr.py \
  --data-root datasets/opt-rsvg

python tools/opt_rsvg/opt_rsvg_xml_to_odvg.py \
  --data-root datasets/opt-rsvg
```

RSVG uses the provided `datasets/RSVG/rsvg/mdetr_annotations/` and `odvg_ann/` files. Set `RSVG_DATA_ROOT`, `DIOR_DATA_ROOT`, or `OPT_DATA_ROOT` to override the repository-relative defaults.

## Results

![Results](assets/results.png)

## Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/YongpengWen/C2I-DINO.git
   cd C2I-DINO
   ```

2. Create the environment. The experiments were run with Python 3.8, PyTorch 2.0.1, CUDA 11.7, MMCV 2.0.0, MMEngine 0.10.4, MMDetection 3.3.0, and Transformers 4.46.3.

   ```bash
   conda create -n mmdet python=3.8 -y
   conda activate mmdet

   # Install a PyTorch and torchvision build compatible with your CUDA version.
   pip install -r requirements/runtime.txt
   pip install -v -e .
   ```

## Download Pretrained Models

The experiments use a Grounding DINO Swin-T checkpoint and the `bert-base-uncased` tokenizer/model. Download the bundled files from Baidu Netdisk:

| Resource | Download | Destination |
| --- | --- | --- |
| Grounding DINO Swin-T checkpoint and BERT files | [Baidu Netdisk](https://pan.baidu.com/s/1oEEg-FusTiF92BDbxJEWVA?pwd=1124) (extraction code: `1124`) | `pretrained/` |

After downloading and extracting, the repository should contain:

```text
pretrained/
├── grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
└── bert-base-uncased/
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

The configurations refer to these repository-relative paths by default. If you store the files elsewhere, update `load_from` and `lang_model_name` or provide an equivalent local path.

## Training

Run commands from the repository root:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# RSVG
python tools/train.py \
  configs/rsvg/c2i_dino_rsvg.py \
  --work-dir work_dirs/rsvg

# DIOR-RSVG
python tools/train.py \
  configs/dior_rsvg/c2i_dino_dior_rsvg.py \
  --work-dir work_dirs/dior_rsvg

# OPT-RSVG
python tools/train.py \
  configs/opt_rsvg/c2i_dino_opt_rsvg.py \
  --work-dir work_dirs/opt_rsvg
```

## Evaluation

```bash
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --work-dir <OUTPUT_DIRECTORY>
```

Replace the placeholders with the relevant configuration, trained checkpoint, and output directory.

## Acknowledgments

This project is built on [MMDetection](https://github.com/open-mmlab/mmdetection) and Grounding DINO. We thank the original authors and the dataset providers for their open-source contributions.
