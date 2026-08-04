#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_m0_live_pipeline.sh GPU_ID" >&2
  exit 2
fi
GPU_ID="$1"
if [[ ! "${GPU_ID}" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be an integer from 0 through 7: ${GPU_ID}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"
source "${SCRIPT_DIR}/m0_gpu_lock.sh"
umask 077

AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
PROJECT_COMMIT="$(git -C "${JPH_PROJECT_DIR}" rev-parse HEAD)"
if [[ -n "$(git -C "${JPH_PROJECT_DIR}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Project worktree must be clean before the M0 pipeline" >&2
  exit 2
fi
if ! jph_acquire_m0_gpu_lock "${GPU_ID}"; then
  exit 3
fi

PIPELINE_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_NONCE="$(
  "${AREAL_VENV}/bin/python" -c 'import secrets; print(secrets.token_hex(16))'
)"
PIPELINE_ID="${PIPELINE_STAMP}-m0-pipeline-${PIPELINE_NONCE}"
PIPELINE_ROOT="${JPH_ROOT}/artifacts/m0-live-pipeline/${PIPELINE_ID}"
TEMPLATE_PATH="${PIPELINE_ROOT}/frozen-dual-credit-template.json"
ROLLOUT_LOG="${PIPELINE_ROOT}/rollout.log"
JOINT_LOG="${PIPELINE_ROOT}/joint-update.log"
mkdir -m 700 -p "${PIPELINE_ROOT}"
touch "${ROLLOUT_LOG}" "${JOINT_LOG}"
chmod 600 "${ROLLOUT_LOG}" "${JOINT_LOG}"

"${AREAL_VENV}/bin/python" \
  "${SCRIPT_DIR}/write_m0_rlvr_estimator_template.py" \
  --output "${TEMPLATE_PATH}" \
  --allowed-root "${JPH_ROOT}" \
  > "${PIPELINE_ROOT}/template-record.log"
chmod 600 "${PIPELINE_ROOT}/template-record.log"

echo "pipeline=${PIPELINE_ID} project=${PROJECT_COMMIT} physical_gpu=${GPU_ID}" \
  | tee -a "${ROLLOUT_LOG}"
JPH_PROJECT_COMMIT="${PROJECT_COMMIT}" \
JPH_BRIDGE_RUN_KIND=m0-torch-joint-v1 \
JPH_AREAL_JOINT_BRIDGE_TASKS=4 \
JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET=0 \
JPH_SGLANG_LOGPROB_MODE=standard-log-of-softmax-v1 \
JPH_CLEAN_ENVIRONMENT_POLICY=filtered-inherited-v1 \
JPH_SGLANG_DISABLE_CUDA_GRAPH=false \
JPH_EXPERIMENTAL_AXIS=none-v1 \
JPH_HARNESS_CONTROLLER_KIND=torch \
JPH_HARNESS_HIDDEN_SIZE=32 \
JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH="${TEMPLATE_PATH}" \
  bash "${SCRIPT_DIR}/run_areal_joint_bridge.sh" "${GPU_ID}" \
  2>&1 | tee -a "${ROLLOUT_LOG}"

ROLLOUT_RUN_ROOT="$(
  sed -n 's/^run_root=\([^ ]*\) .*/\1/p' "${ROLLOUT_LOG}" | tail -n 1
)"
if [[ -z "${ROLLOUT_RUN_ROOT}" ]] || [[ ! -d "${ROLLOUT_RUN_ROOT}" ]]; then
  echo "Cannot resolve the completed M0 rollout root" >&2
  exit 4
fi

JPH_PROJECT_COMMIT="${PROJECT_COMMIT}" \
  bash "${SCRIPT_DIR}/run_m0_live_joint.sh" \
    "${GPU_ID}" "${ROLLOUT_RUN_ROOT}" \
    2>&1 | tee -a "${JOINT_LOG}"

echo "M0 live pipeline complete: ${PIPELINE_ROOT}"
