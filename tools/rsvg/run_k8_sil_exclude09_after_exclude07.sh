#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/mmdetection-dino
source_pattern='tools/train.py configs/rsvg/grounding_dino_swin-t_class_suffix_k8_spatial_contrast_phrase_max_finetune_rsvg_1024_12e.py'
source_dir=/root/autodl-tmp/work_dirs/rsvg_cfp_sil_k8_phrase_max_seed42
target_dir=/root/autodl-tmp/work_dirs/rsvg_cfp_sil_k8_exclude09_seed42

while pgrep -f "$source_pattern" >/dev/null; do
    sleep 60
done

latest_log=$(find "$source_dir" -path '*/20*.log' -type f | sort | tail -n 1)
if [[ -z "$latest_log" ]] || ! grep -q 'Epoch(val) \[12\]\[151/151\]' "$latest_log"; then
    echo 'The exclude-0.7 run did not finish all 12 epochs; not starting exclude-0.9.' >&2
    exit 1
fi

mkdir -p "$target_dir"
cd "$repo_dir"
exec /root/miniconda3/envs/mmdet/bin/python tools/train.py \
    configs/rsvg/grounding_dino_swin-t_class_suffix_k8_sil_exclude09_finetune_rsvg_1024_12e.py \
    --work-dir "$target_dir"
