#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmdet
cd /root/autodl-tmp/mmdetection-dino

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=/root/autodl-tmp/mmdetection-dino:${PYTHONPATH:-}
python tools/train.py configs/rsvg/grounding_dino_swin-t_target_phrase_finetune_rsvg_1024_12e.py
