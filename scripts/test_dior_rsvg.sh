#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
config="${repo_root}/configs/dior_rsvg/c2i_dino_dior_rsvg.py"
checkpoint="${1:?Usage: $0 CHECKPOINT [WORK_DIR]}"
work_dir="${2:-${repo_root}/work_dirs/dior_rsvg_test}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" tools/test.py "${config}" "${checkpoint}" --work-dir "${work_dir}"
