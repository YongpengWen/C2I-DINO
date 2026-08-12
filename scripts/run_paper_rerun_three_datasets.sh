#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/mmdetection-dino
python_bin=/root/miniconda3/envs/mmdet/bin/python
log_root=/root/autodl-tmp/paper_rerun_logs

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${log_root}"
cd "${repo_root}"

run_dataset() {
    local dataset_name=$1
    local config_path=$2
    local work_dir=$3

    echo "[$(date -u +%FT%TZ)] START train ${dataset_name}"
    "${python_bin}" tools/train.py "${config_path}" \
        2>&1 | tee "${log_root}/${dataset_name}_train.log"

    local best_checkpoint
    best_checkpoint=$(find "${work_dir}" -maxdepth 1 -type f \
        -name 'best_*.pth' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -z "${best_checkpoint}" ]]; then
        echo "No best checkpoint found for ${dataset_name} in ${work_dir}" >&2
        return 1
    fi

    echo "[$(date -u +%FT%TZ)] START test ${dataset_name}: ${best_checkpoint}"
    "${python_bin}" tools/test.py "${config_path}" "${best_checkpoint}" \
        2>&1 | tee "${log_root}/${dataset_name}_test.log"
    echo "[$(date -u +%FT%TZ)] DONE ${dataset_name}"
}

run_dataset \
    rsvg \
    configs/rsvg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py \
    /root/autodl-tmp/work_dirs/paper_rerun_bert_trainable_rsvg_1024_bs8_12e

run_dataset \
    dior_rsvg \
    configs/dior_rsvg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_dior_rsvg_official_split_resize640_12e.py \
    /root/autodl-tmp/work_dirs/paper_rerun_bert_trainable_dior_rsvg_640_bs8_12e

run_dataset \
    opt_rsvg \
    configs/opt_vg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_opt_vg_official_split_12e.py \
    /root/autodl-tmp/work_dirs/paper_rerun_bert_trainable_opt_rsvg_ms8_11_bs8_12e

echo "[$(date -u +%FT%TZ)] ALL DATASETS COMPLETE"
