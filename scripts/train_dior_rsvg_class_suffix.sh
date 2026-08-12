#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmdet
cd /root/autodl-tmp/mmdetection-dino

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=/root/autodl-tmp/mmdetection-dino:${PYTHONPATH:-}
python tools/train.py configs/dior_rsvg/grounding_dino_swin-t_class_suffix_finetune_dior_rsvg_official_split_resize640.py

