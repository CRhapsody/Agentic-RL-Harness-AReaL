#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_sglang_logprob_screen.sh GPU_ID c0|c1" >&2
  exit 2
fi

GPU_ID="$1"
CELL="$2"
case "${CELL}" in
  c0)
    LOGPROB_MODE="standard-log-of-softmax-v1"
    ;;
  c1)
    LOGPROB_MODE="original-log-softmax-v1"
    ;;
  *)
    echo "Cell must be c0 or c1: ${CELL}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! "${JPH_SCREEN_PAIR_ID:-}" =~ ^[A-Za-z0-9._-]{16,160}$ ]]; then
  echo "JPH_SCREEN_PAIR_ID must be supplied by the serial pair runner" >&2
  exit 2
fi
if [[ -z "${JPH_SCREEN_CELL_POINTER:-}" ]]; then
  echo "JPH_SCREEN_CELL_POINTER must be supplied by the serial pair runner" >&2
  exit 2
fi
if [[ "$(basename "${JPH_SCREEN_CELL_POINTER}")" != "${CELL}-run-root.txt" ]]; then
  echo "Screen cell pointer does not match cell ${CELL}" >&2
  exit 2
fi

exec /usr/bin/env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  JPH_BRIDGE_RUN_KIND=logprob-mechanism-screen-v1 \
  JPH_AREAL_JOINT_BRIDGE_TASKS=4 \
  JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET=32 \
  JPH_SGLANG_LOGPROB_MODE="${LOGPROB_MODE}" \
  JPH_SCREEN_PAIR_ID="${JPH_SCREEN_PAIR_ID}" \
  JPH_SCREEN_CELL_POINTER="${JPH_SCREEN_CELL_POINTER}" \
  JPH_CLEAN_ENVIRONMENT_POLICY=env-i-v1 \
  /bin/bash "${SCRIPT_DIR}/run_areal_joint_bridge.sh" "${GPU_ID}"
