#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

for GPU_ID in 0 1 2 3 4 5 6 7; do
  IFS=, read -r GPU_MEMORY_USED GPU_MEMORY_FREE < <(
    nvidia-smi -i "${GPU_ID}" \
      --query-gpu=memory.used,memory.free \
      --format=csv,noheader,nounits | tr -d ' '
  )
  GPU_PROCESSES="$(
    nvidia-smi -i "${GPU_ID}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits
  )"
  if [[ ! "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]] || \
    [[ ! "${GPU_MEMORY_FREE}" =~ ^[0-9]+$ ]]; then
    echo "Could not read GPU ${GPU_ID} memory usage" >&2
    exit 3
  fi
  printf '%s gpu=%s used=%sMiB free=%sMiB processes=%s\n' \
    "$(date -Is)" "${GPU_ID}" "${GPU_MEMORY_USED}" "${GPU_MEMORY_FREE}" \
    "$([[ -n "${GPU_PROCESSES}" ]] && printf observed || printf none)"
done
echo "$(date -Is) all eight GPUs were observed; launching B0 without a configured memory limit"
exec /bin/bash "${SCRIPT_DIR}/run_areal_official_b0.sh"
