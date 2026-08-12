#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/mmdetection-dino
python_bin=/root/miniconda3/envs/mmdet/bin/python

run_training() {
    local config=$1
    local work_dir=$2
    mkdir -p "$work_dir"
    cd "$repo_dir"
    "$python_bin" tools/train.py "$config" --work-dir "$work_dir"
}

run_training \
    configs/rsvg/grounding_dino_swin-t_target_phrase_sil_phrase_max_finetune_rsvg_1024_12e.py \
    /root/autodl-tmp/work_dirs/rsvg_sil_k0_phrase_max_seed42
run_training \
    configs/rsvg/grounding_dino_swin-t_class_suffix_k8_spatial_contrast_phrase_max_finetune_rsvg_1024_12e.py \
    /root/autodl-tmp/work_dirs/rsvg_cfp_sil_k8_phrase_max_seed42
run_training \
    configs/rsvg/grounding_dino_swin-t_class_suffix_k16_spatial_contrast_phrase_max_finetune_rsvg_1024_12e.py \
    /root/autodl-tmp/work_dirs/rsvg_cfp_sil_k16_phrase_max_seed42
