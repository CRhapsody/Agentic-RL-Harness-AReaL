#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_remote_smoke.sh GPU_ID [MODEL] [TASK]" >&2
  exit 2
fi

GPU_ID="$1"
MODEL_NAME="${2:-Qwen/Qwen2.5-1.5B-Instruct}"
TASK_ID="${3:-add-17-25}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer: ${GPU_ID}" >&2
  exit 2
fi
if [[ ! "${TASK_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "TASK must contain only letters, digits, dot, underscore, or dash: ${TASK_ID}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

case "${PROJECT_DIR}" in
  "${JPH_ROOT}"/*) ;;
  *)
    echo "Refusing smoke: project is outside ${JPH_ROOT}: ${PROJECT_DIR}" >&2
    exit 2
    ;;
esac

VENV_DIR="${JPH_ROOT}/venvs/jphrl-smoke"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}; run bash scripts/bootstrap_remote.sh first" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SAFE_MODEL="${MODEL_NAME//\//--}"
TRACE_PATH="${JPH_ROOT}/artifacts/smoke/${RUN_STAMP}-${SAFE_MODEL}-${TASK_ID}.json"
LOG_PATH="${JPH_ROOT}/logs/${RUN_STAMP}-${SAFE_MODEL}-${TASK_ID}.log"
AUDIT_PATH="${JPH_ROOT}/artifacts/smoke/${RUN_STAMP}-${SAFE_MODEL}-${TASK_ID}.audit.json"
PREFETCH_REPORT="${JPH_ROOT}/artifacts/smoke/${RUN_STAMP}-${SAFE_MODEL}.snapshot.json"

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/scripts/prefetch_hf_model.py" \
  "${MODEL_NAME}" --output "${PREFETCH_REPORT}" \
  2>&1 | tee "${LOG_PATH}"
MODEL_REVISION="$("${VENV_DIR}/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["resolved_commit"])' "${PREFETCH_REPORT}")"

mkdir -p "${JPH_ROOT}/runtime/locks"
exec 9> "${JPH_ROOT}/runtime/locks/gpu-${GPU_ID}.lock"
if ! flock -n 9; then
  echo "GPU ${GPU_ID} is already reserved by another JPH process" >&2
  exit 3
fi

GPU_MEMORY_USED="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]]; then
  echo "Cannot read memory usage for GPU ${GPU_ID}: ${GPU_MEMORY_USED}" >&2
  exit 2
fi
if (( GPU_MEMORY_USED >= 500 )); then
  echo "GPU ${GPU_ID} is not free: ${GPU_MEMORY_USED} MiB is in use" >&2
  exit 3
fi

echo "GPU=${GPU_ID} model=${MODEL_NAME} revision=${MODEL_REVISION} task=${TASK_ID}"
echo "trace=${TRACE_PATH} log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
  "${VENV_DIR}/bin/python" -m jphrl.cli \
  --backend hf \
  --model "${MODEL_NAME}" \
  --revision "${MODEL_REVISION}" \
  --device cuda \
  --task "${TASK_ID}" \
  --output "${TRACE_PATH}" \
  2>&1 | tee -a "${LOG_PATH}"

"${VENV_DIR}/bin/python" "${PROJECT_DIR}/scripts/verify_real_smoke_trace.py" "${TRACE_PATH}" \
  | tee -a "${LOG_PATH}" "${AUDIT_PATH}"
