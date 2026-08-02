#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

POLL_SECONDS="${JPH_B0_POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${JPH_B0_MAX_WAIT_SECONDS:-86400}"
MIN_FREE_MEMORY_MIB="${JPH_B0_MIN_FREE_MEMORY_MIB:-71680}"
MAX_USED_MEMORY_MIB="${JPH_B0_MAX_USED_MEMORY_MIB:-10240}"
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MAX_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MAX_WAIT_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MIN_FREE_MEMORY_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MIN_FREE_MEMORY_MIB must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MAX_USED_MEMORY_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MAX_USED_MEMORY_MIB must be a positive integer" >&2
  exit 2
fi

DEADLINE=$((SECONDS + MAX_WAIT_SECONDS))
while ((SECONDS < DEADLINE)); do
  BLOCKED_GPUS=()
  for GPU_ID in 0 1 2 3 4 5 6 7; do
    IFS=, read -r GPU_MEMORY_USED GPU_MEMORY_FREE < <(
      nvidia-smi -i "${GPU_ID}" \
        --query-gpu=memory.used,memory.free \
        --format=csv,noheader,nounits | tr -d ' '
    )
    if [[ ! "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]] || \
      [[ ! "${GPU_MEMORY_FREE}" =~ ^[0-9]+$ ]]; then
      echo "Could not read GPU ${GPU_ID} memory usage" >&2
      exit 3
    fi
    if ((GPU_MEMORY_FREE < MIN_FREE_MEMORY_MIB || GPU_MEMORY_USED > MAX_USED_MEMORY_MIB)); then
      BLOCKED_GPUS+=("${GPU_ID}:used=${GPU_MEMORY_USED}MiB,free=${GPU_MEMORY_FREE}MiB")
    fi
  done

  if ((${#BLOCKED_GPUS[@]} == 0)); then
    echo "$(date -Is) all eight GPUs passed the memory headroom gate (used<=${MAX_USED_MEMORY_MIB}MiB, free>=${MIN_FREE_MEMORY_MIB}MiB); launching B0"
    exec /bin/bash "${SCRIPT_DIR}/run_areal_official_b0.sh"
  fi

  echo "$(date -Is) waiting for GPU headroom: ${BLOCKED_GPUS[*]}"
  sleep "${POLL_SECONDS}"
done

echo "Timed out after ${MAX_WAIT_SECONDS}s without enough memory headroom on all eight GPUs" >&2
exit 4
