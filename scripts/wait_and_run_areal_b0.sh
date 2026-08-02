#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

POLL_SECONDS="${JPH_B0_POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${JPH_B0_MAX_WAIT_SECONDS:-86400}"
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MAX_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MAX_WAIT_SECONDS must be a positive integer" >&2
  exit 2
fi

DEADLINE=$((SECONDS + MAX_WAIT_SECONDS))
while ((SECONDS < DEADLINE)); do
  BUSY_GPUS=()
  for GPU_ID in 0 1 2 3 4 5 6 7; do
    GPU_MEMORY_USED="$(
      nvidia-smi -i "${GPU_ID}" \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' '
    )"
    if [[ ! "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]]; then
      echo "Could not read GPU ${GPU_ID} memory usage" >&2
      exit 3
    fi
    if ((GPU_MEMORY_USED >= 500)); then
      BUSY_GPUS+=("${GPU_ID}:${GPU_MEMORY_USED}MiB")
    fi
  done

  if ((${#BUSY_GPUS[@]} == 0)); then
    echo "$(date -Is) all eight GPUs passed the <500MiB gate; launching B0"
    exec "${SCRIPT_DIR}/run_areal_official_b0.sh"
  fi

  echo "$(date -Is) waiting for GPUs: ${BUSY_GPUS[*]}"
  sleep "${POLL_SECONDS}"
done

echo "Timed out after ${MAX_WAIT_SECONDS}s without eight free GPUs" >&2
exit 4
