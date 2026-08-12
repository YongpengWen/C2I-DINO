#!/usr/bin/env bash
# Sequential runner for the missing RSVG ablations.
set -euo pipefail

project_root="/root/autodl-tmp/mmdetection-dino"
python_bin="/root/miniconda3/envs/mmdet/bin/python"
work_root="/root/autodl-tmp/work_dirs/rsvg_ablation_sil_terms"
data_root="/root/autodl-tmp/datasets/RSVG/rsvg"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
mkdir -p "${work_root}"
cd "${project_root}"

evaluate() {
  local config="$1" checkpoint="$2" result_root="$3"
  "${python_bin}" tools/test.py "${config}" "${checkpoint}" --cfg-options test_evaluator.iou_thrs="[0.25,0.5]" work_dir="${result_root}/test"
  "${python_bin}" tools/test.py "${config}" "${checkpoint}" --cfg-options test_dataloader.dataset.ann_file="mdetr_annotations_target_np/finetune_rsvg_val.json" test_evaluator.ann_file="${data_root}/mdetr_annotations/finetune_rsvg_val.json" test_evaluator.dataset_name="rsvg_val" test_evaluator.iou_thrs="[0.25,0.5]" work_dir="${result_root}/val"
}

run_one() {
  local name="$1" config="$2" run_dir="${work_root}/$1" checkpoint
  mkdir -p "${run_dir}"
  checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name "best_refexp_rsvg_val_Pr@0.5*.pth" -print -quit)"
  if [[ -z "${checkpoint}" ]]; then
    echo "[$(date "+%F %T")] training ${name}"
    "${python_bin}" tools/train.py "${config}" --work-dir "${run_dir}"
    checkpoint="$(find "${run_dir}" -maxdepth 1 -type f -name "best_refexp_rsvg_val_Pr@0.5*.pth" -print -quit)"
  else
    echo "[$(date "+%F %T")] reusing ${checkpoint}"
  fi
  [[ -n "${checkpoint}" ]] || { echo "Missing best checkpoint for ${name}" >&2; exit 1; }
  echo "[$(date "+%F %T")] evaluating ${name}"
  evaluate "${config}" "${checkpoint}" "${run_dir}/evaluation"
}

# The SIL-only and CFP-only rows are not part of the final ablation. Both
# remaining loss-design rows retain CFP and isolate the two terms inside SIL.
run_one "cfp_sil_ranking_only" "configs/rsvg/grounding_dino_swin-t_class_suffix_scl_rank_only_finetune_rsvg_1024_12e.py"
run_one "cfp_sil_suppression_only" "configs/rsvg/grounding_dino_swin-t_class_suffix_scl_suppress_only_finetune_rsvg_1024_12e.py"
echo "[$(date "+%F %T")] all RSVG ablations finished"
