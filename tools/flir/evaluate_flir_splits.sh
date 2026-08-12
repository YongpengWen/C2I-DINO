#!/usr/bin/env bash
set -euo pipefail

config_path=$1
checkpoint_path=$2
modality=$3
data_root=/root/autodl-tmp/datasets/flir

for split in val test testA testB testC; do
  dataset_name="flir_${modality}_${split}"
  PYTHONPATH=/root/autodl-tmp/mmdetection-dino${PYTHONPATH:+:$PYTHONPATH} \
    /root/miniconda3/envs/mmdet/bin/python tools/test.py "$config_path" "$checkpoint_path" \
    --cfg-options \
      "test_dataloader.dataset.ann_file=mdetr_annotations/finetune_flir_${split}.json" \
      "test_evaluator.ann_file=${data_root}/mdetr_annotations/finetune_flir_${split}.json" \
      "test_evaluator.dataset_name=${dataset_name}"
done
