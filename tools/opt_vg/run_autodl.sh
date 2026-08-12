#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
PYTHON="${PYTHON:-/root/miniconda3/envs/mmdet/bin/python}"
OPT_DATA_ROOT="${OPT_DATA_ROOT:-/root/autodl-tmp/opt-rsvg}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/work_dirs/opt_vg}"
CONFIG="${CONFIG:-configs/opt_vg/grounding_dino_swin-t_finetune_opt_vg.py}"
WEIGHTS="${WEIGHTS:-}"
IMAGE="${IMAGE:-${OPT_DATA_ROOT}/Image/000000.jpg}"
TEXT="${TEXT:-airplane . ship . vehicle . bridge . storage tank .}"

export OPT_DATA_ROOT

check_data() {
  test -d "${OPT_DATA_ROOT}/Image" || {
    echo "Missing image directory: ${OPT_DATA_ROOT}/Image" >&2
    exit 1
  }
  test -d "${OPT_DATA_ROOT}/odvg_ann" || {
    echo "Missing ODVG annotations: ${OPT_DATA_ROOT}/odvg_ann" >&2
    echo "Run: bash tools/opt_vg/run_autodl.sh convert" >&2
    exit 1
  }
}

case "${MODE}" in
  convert)
    "${PYTHON}" tools/opt_vg/opt_vg_to_odvg.py --data-root "${OPT_DATA_ROOT}"
    ;;
  train)
    check_data
    "${PYTHON}" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" "${@:2}"
    ;;
  train-small)
    check_data
    "${PYTHON}" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" \
      --cfg-options train_dataloader.batch_size=1 "${@:2}"
    ;;
  test)
    check_data
    test -n "${WEIGHTS}" || {
      echo "Set WEIGHTS=/path/to/checkpoint.pth" >&2
      exit 1
    }
    "${PYTHON}" tools/test.py "${CONFIG}" "${WEIGHTS}" "${@:2}"
    ;;
  infer)
    check_data
    test -n "${WEIGHTS}" || {
      echo "Set WEIGHTS=/path/to/checkpoint.pth" >&2
      exit 1
    }
    "${PYTHON}" tools/opt_vg/infer_single.py \
      --weights "${WEIGHTS}" \
      --image "${IMAGE}" \
      --text "${TEXT}" \
      --out-dir outputs/opt_vg_demo "${@:2}"
    ;;
  *)
    echo "Usage: bash tools/opt_vg/run_autodl.sh {convert|train|train-small|test|infer}" >&2
    exit 1
    ;;
esac
