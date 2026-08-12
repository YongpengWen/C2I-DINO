#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
config="${repo_root}/configs/opt_rsvg/c2i_dino_opt_rsvg.py"
work_dir="${WORK_DIR:-${repo_root}/work_dirs/opt_rsvg}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
"${python_bin}" tools/train.py "${config}" --work-dir "${work_dir}"
