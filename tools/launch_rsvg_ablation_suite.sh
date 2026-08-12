#!/usr/bin/env bash
set -euo pipefail

while true; do
  gpu_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d " ")"
  if [[ "${gpu_memory}" -lt 1000 ]]; then
    break
  fi
  echo "[$(date "+%F %T")] waiting for GPU; ${gpu_memory} MiB is in use"
  sleep 60
done

exec bash /root/autodl-tmp/mmdetection-dino/tools/run_rsvg_ablation_suite.sh
