#!/usr/bin/env bash
set -euo pipefail

while screen -ls | grep -q '\.flir_rgb_after_ir'; do
  sleep 60
done

ir_checkpoint=$(find /root/autodl-tmp/work_dirs/flir_ir_gdino_swin_t_224_12e -maxdepth 1 -name 'best_refexp*.pth' -print -quit)
rgb_checkpoint=$(find /root/autodl-tmp/work_dirs/flir_rgb_gdino_swin_t_224_12e -maxdepth 1 -name 'best_refexp*.pth' -print -quit)

test -n "$ir_checkpoint"
test -n "$rgb_checkpoint"

tools/flir/evaluate_flir_splits.sh \
  configs/flir/grounding_dino_swin-t_finetune_flir_ir_224_eval.py \
  "$ir_checkpoint" ir
tools/flir/evaluate_flir_splits.sh \
  configs/flir/grounding_dino_swin-t_finetune_flir_rgb_224_eval.py \
  "$rgb_checkpoint" rgb
