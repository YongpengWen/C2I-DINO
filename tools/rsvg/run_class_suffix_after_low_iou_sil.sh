#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/mmdetection-dino
source_dir=/root/autodl-tmp/work_dirs/rsvg_cfp_sil_phrase_max_seed42
target_dir=/root/autodl-tmp/work_dirs/rsvg_cfp_class_only_seed42
source_pattern='tools/train.py .*phrase_score_max_finetune'

mkdir -p "$target_dir"

while true; do
    latest_log=$(find "$source_dir" -path '*/20*.log' -type f | sort | tail -n 1)
    if [[ -n "$latest_log" ]] && \
            grep -q 'Epoch(val) \[12\]\[151/151\]' "$latest_log"; then
        break
    fi
    if ! pgrep -f "$source_pattern" >/dev/null; then
        echo 'The low-IoU SIL run exited before completing epoch 12.' >&2
        exit 1
    fi
    sleep 60
done

cd "$repo_dir"
exec /root/miniconda3/envs/mmdet/bin/python tools/train.py \
    configs/rsvg/grounding_dino_swin-t_class_suffix_only_finetune_rsvg_1024_12e.py \
    --work-dir "$target_dir"
