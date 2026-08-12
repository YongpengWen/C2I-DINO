#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmdet
cd /root/autodl-tmp/mmdetection-dino

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/root/autodl-tmp/mmdetection-dino:${PYTHONPATH:-}
python tools/train.py configs/rsvg/grounding_dino_swin-t_spatial_contrast_finetune_rsvg_640_12e.py
