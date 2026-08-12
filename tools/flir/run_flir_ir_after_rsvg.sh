#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/mmdetection-dino
rsvg_dir=/root/autodl-tmp/work_dirs/rsvg_cfp_class_only_seed42
flir_dir=/root/autodl-tmp/work_dirs/flir_ir_gdino_swin_t_640_12e

while true; do
    latest_log=$(find "$rsvg_dir" -path '*/20*.log' -type f | sort | tail -n 1)
    if [[ -n "$latest_log" ]] && grep -q 'Epoch(val) \[12\]\[151/151\]' "$latest_log"; then
        break
    fi
    if ! pgrep -f 'grounding_dino_swin-t_class_suffix_only_finetune_rsvg_1024_12e' >/dev/null; then
        echo 'RSVG run exited before completing epoch 12.' >&2
        exit 1
    fi
    sleep 60
done

cd "$repo_dir"
exec /root/miniconda3/envs/mmdet/bin/python tools/train.py \
    configs/flir/grounding_dino_swin-t_finetune_flir_ir_640_12e.py \
    --work-dir "$flir_dir"
