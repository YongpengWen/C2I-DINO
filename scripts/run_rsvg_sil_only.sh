#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/mmdetection-dino
py_bin=/root/miniconda3/envs/mmdet/bin/python
cfg=configs/rsvg/grounding_dino_swin-t_target_phrase_spatial_instance_ranking_finetune_rsvg_1024_12e.py
work_dir=/root/autodl-tmp/work_dirs/rsvg_swin_t_spatial_instance_learning_1024_bs8_12e

"${py_bin}" tools/train.py "${cfg}" > "${work_dir}.train.log" 2>&1
checkpoint=$(find "${work_dir}" -maxdepth 1 -name 'best_refexp_rsvg_val_Pr@0.5_epoch_*.pth' -print -quit)
"${py_bin}" tools/test.py "${cfg}" "${checkpoint}" > "${work_dir}.test.log" 2>&1
