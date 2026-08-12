#!/usr/bin/env bash
# Run after the active seed-42 CFP+SIL baseline releases the only GPU.
set -euo pipefail

project_root="/root/autodl-tmp/mmdetection-dino"
python_bin="/root/miniconda3/envs/mmdet/bin/python"
config="configs/rsvg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py"
data_root="/root/autodl-tmp/datasets/RSVG/rsvg"
baseline_dir="/root/autodl-tmp/work_dirs/rsvg_cfp_sil_seed42_fixed"
work_root="/root/autodl-tmp/work_dirs/rsvg_sil_topk_ablation_seed42"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

wait_for_gpu() {
  while true; do
    gpu_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [[ "${gpu_memory}" -lt 1000 ]]; then
      return
    fi
    sleep 60
  done
}

evaluate() {
  local checkpoint="$1"
  local result_dir="$2"
  "${python_bin}" tools/test.py "${config}" "${checkpoint}" \
    --work-dir "${result_dir}/test" \
    --cfg-options test_evaluator.iou_thrs="[0.25,0.5]"
  "${python_bin}" tools/test.py "${config}" "${checkpoint}" \
    --work-dir "${result_dir}/val" \
    --cfg-options \
      test_dataloader.dataset.ann_file="mdetr_annotations_target_np/finetune_rsvg_val.json" \
      test_evaluator.ann_file="${data_root}/mdetr_annotations/finetune_rsvg_val.json" \
      test_evaluator.dataset_name="rsvg_val" \
      test_evaluator.iou_thrs="[0.25,0.5]"
}

run_topk() {
  local topk="$1"
  local run_dir="${work_root}/topk${topk}"
  local checkpoint
  mkdir -p "${run_dir}"
  checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'best_refexp_rsvg_val_Pr@0.5_epoch_*.pth' -print -quit)"
  if [[ -z "${checkpoint}" ]]; then
    "${python_bin}" tools/train.py "${config}" --work-dir "${run_dir}" --cfg-options \
      randomness.seed=42 randomness.deterministic=False \
      model.bbox_head.spatial_contrast_cfg.topk="${topk}"
    checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name 'best_refexp_rsvg_val_Pr@0.5_epoch_*.pth' -print -quit)"
  fi
  [[ -n "${checkpoint}" ]] || { echo "No best checkpoint for topk=${topk}" >&2; exit 1; }
  evaluate "${checkpoint}" "${run_dir}/evaluation"
}

mkdir -p "${work_root}"
cd "${project_root}"
wait_for_gpu

baseline_checkpoint="$(find "${baseline_dir}" -maxdepth 1 -type f -name 'best_refexp_rsvg_val_Pr@0.5_epoch_*.pth' -print -quit)"
[[ -n "${baseline_checkpoint}" ]] || { echo "No best checkpoint for the seed-42 baseline" >&2; exit 1; }
evaluate "${baseline_checkpoint}" "${baseline_dir}/evaluation_025"
run_topk 5
run_topk 20
