#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_areal_joint_bridge.sh GPU_ID" >&2
  exit 2
fi
GPU_ID="$1"
if [[ ! "${GPU_ID}" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be an integer from 0 through 7: ${GPU_ID}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"
umask 077

AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
EXPECTED_AREAL_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/qwen2.5-1.5b-snapshot.json"
DATASET_REPORT="${JPH_ROOT}/artifacts/bootstrap/gsm8k-snapshot.json"
CUDA_TOOLKIT_ROOT="${JPH_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.6}"
MAX_USED_MEMORY_MIB="${JPH_BRIDGE_MAX_USED_MEMORY_MIB:-10240}"
MIN_FREE_MEMORY_MIB="${JPH_BRIDGE_MIN_FREE_MEMORY_MIB:-65536}"

for path in "${AREAL_REPO}/.git" "${AREAL_VENV}/bin/python" "${MODEL_REPORT}" "${DATASET_REPORT}" "${CUDA_TOOLKIT_ROOT}/bin/nvcc"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done
export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${AREAL_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
"${CUDACXX}" -std=c++20 -x cu -c /dev/null \
  -o "${JPH_ROOT}/tmp/nvcc-joint-bridge-preflight.o"
ACTUAL_AREAL_COMMIT="$(git -C "${AREAL_REPO}" rev-parse HEAD)"
if [[ "${ACTUAL_AREAL_COMMIT}" != "${EXPECTED_AREAL_COMMIT}" ]]; then
  echo "AReaL commit mismatch: ${ACTUAL_AREAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${AREAL_REPO}" status --porcelain)" ]]; then
  echo "Pinned AReaL worktree must be clean before a bridge run" >&2
  exit 2
fi
PROJECT_COMMIT="$(git -C "${JPH_PROJECT_DIR}" rev-parse HEAD)"
if [[ -n "${JPH_PROJECT_COMMIT:-}" ]] && [[ "${JPH_PROJECT_COMMIT}" != "${PROJECT_COMMIT}" ]]; then
  echo "Project commit mismatch: expected=${JPH_PROJECT_COMMIT} actual=${PROJECT_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${JPH_PROJECT_DIR}" status --porcelain)" ]]; then
  echo "Project worktree must be clean before a bridge run" >&2
  exit 2
fi

readarray -t SNAPSHOT_VALUES < <(
  "${AREAL_VENV}/bin/python" -c '
import json, sys
model = json.load(open(sys.argv[1]))
data = json.load(open(sys.argv[2]))
print(model["snapshot_path"])
print(model["resolved_commit"])
print(data["snapshot_path"])
print(data["resolved_commit"])
' "${MODEL_REPORT}" "${DATASET_REPORT}"
)
MODEL_SNAPSHOT="${SNAPSHOT_VALUES[0]}"
MODEL_REVISION="${SNAPSHOT_VALUES[1]}"
DATASET_SNAPSHOT="${SNAPSHOT_VALUES[2]}"
DATASET_REVISION="${SNAPSHOT_VALUES[3]}"
for snapshot in "${MODEL_SNAPSHOT}" "${DATASET_SNAPSHOT}"; do
  if [[ ! -d "${snapshot}" ]]; then
    echo "Missing pinned snapshot: ${snapshot}" >&2
    exit 2
  fi
  case "${snapshot}" in
    "${JPH_ROOT}"/*) ;;
    *)
      echo "Snapshot escapes ${JPH_ROOT}: ${snapshot}" >&2
      exit 2
      ;;
  esac
done

mkdir -p -m 700 "${JPH_ROOT}/runtime/locks"
exec 9> "${JPH_ROOT}/runtime/locks/gpu-${GPU_ID}.lock"
if ! flock -n 9; then
  echo "GPU ${GPU_ID} is reserved by another JPH process" >&2
  exit 3
fi
IFS=, read -r GPU_MEMORY_USED GPU_MEMORY_FREE < <(
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=memory.used,memory.free \
    --format=csv,noheader,nounits | tr -d ' '
)
if [[ ! "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]] || [[ ! "${GPU_MEMORY_FREE}" =~ ^[0-9]+$ ]]; then
  echo "Cannot read GPU ${GPU_ID} memory state" >&2
  exit 3
fi
if ((GPU_MEMORY_USED > MAX_USED_MEMORY_MIB || GPU_MEMORY_FREE < MIN_FREE_MEMORY_MIB)); then
  echo "GPU ${GPU_ID} lacks headroom: used=${GPU_MEMORY_USED}MiB free=${GPU_MEMORY_FREE}MiB" >&2
  exit 3
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${JPH_ROOT}/artifacts/areal-joint-bridge/${RUN_STAMP}"
BRIDGE_DIR="${RUN_ROOT}/bridge-records"
SAME_BACKEND_SCORE_DIR="${RUN_ROOT}/same-backend-scores"
AUDIT_PATH="${RUN_ROOT}/audit.json"
NAME_RESOLVE_ROOT="${JPH_ROOT}/runtime/name_resolve/joint-bridge-${RUN_STAMP}"
LOG_PATH="${RUN_ROOT}/run.log"
RUN_ADMIN_API_KEY="$(
  "${AREAL_VENV}/bin/python" -c \
    'import secrets; print("jph-bridge-" + secrets.token_urlsafe(32))'
)"
mkdir -p -m 700 \
  "${BRIDGE_DIR}" \
  "${SAME_BACKEND_SCORE_DIR}" \
  "${NAME_RESOLVE_ROOT}"
touch "${LOG_PATH}"
chmod 600 "${LOG_PATH}"

GPU_MONITOR_PID=""
SECRET_REDACTOR_PID=""
cleanup() {
  if [[ -n "${SECRET_REDACTOR_PID}" ]]; then
    kill "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
    wait "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  fi
  JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
      "${RUN_ROOT}" >/dev/null 2>&1 || true
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
nvidia-smi dmon -i "${GPU_ID}" -s pucm -d 5 -o TD > "${RUN_ROOT}/gpu-dmon.log" 2>&1 &
GPU_MONITOR_PID="$!"
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
    --watch-seconds 1200 "${RUN_ROOT}" >/dev/null 2>&1 &
SECRET_REDACTOR_PID="$!"

echo "project=${PROJECT_COMMIT} AReaL=${ACTUAL_AREAL_COMMIT} physical_gpu=${GPU_ID} model_revision=${MODEL_REVISION} dataset_revision=${DATASET_REVISION}" | tee -a "${LOG_PATH}"
echo "run_root=${RUN_ROOT} bridge_dir=${BRIDGE_DIR} same_backend_score_dir=${SAME_BACKEND_SCORE_DIR}" | tee -a "${LOG_PATH}"

cd "${AREAL_REPO}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
JPH_AREAL_JOINT_BRIDGE_TASKS=4 \
JPH_AREAL_JOINT_BRIDGE_DIR="${BRIDGE_DIR}" \
JPH_AREAL_SAME_BACKEND_SCORE_DIR="${SAME_BACKEND_SCORE_DIR}" \
JPH_AREAL_COMMIT="${ACTUAL_AREAL_COMMIT}" \
JPH_PROJECT_COMMIT="${PROJECT_COMMIT}" \
JPH_BEHAVIOR_SNAPSHOT="${MODEL_SNAPSHOT}" \
JPH_BEHAVIOR_REVISION="${MODEL_REVISION}" \
JPH_DATASET_REVISION="${DATASET_REVISION}" \
JPH_EXPECTED_POLICY_VERSION=0 \
  "${AREAL_VENV}/bin/python" "${JPH_PROJECT_DIR}/scripts/run_areal_joint_bridge_eval.py" \
  --config "${AREAL_REPO}/examples/math/gsm8k_grpo.yaml" \
  scheduler.type=local \
  experiment_name=jph-areal-joint-bridge \
  trial_name="${RUN_STAMP}" \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=1 \
  cluster.fileroot="${RUN_ROOT}" \
  cluster.name_resolve.nfs_record_root="${NAME_RESOLVE_ROOT}" \
  actor.path="${MODEL_SNAPSHOT}" \
  tokenizer_path="${MODEL_SNAPSHOT}" \
  valid_dataset.path="${DATASET_SNAPSHOT}" \
  valid_dataset.batch_size=1 \
  valid_dataset.num_workers=0 \
  gconfig.n_samples=1 \
  gconfig.temperature=1.0 \
  gconfig.max_new_tokens=64 \
  gconfig.max_tokens=512 \
  rollout.backend=sglang:d1p1t1 \
  rollout.max_concurrent_rollouts=1 \
  rollout.dump_to_file=false \
  '+rollout.agent.admin_api_key=${oc.env:JPH_AREAL_ADMIN_API_KEY}' \
  sglang.mem_fraction_static=0.35 \
  sglang.context_length=1024 \
  sglang.max_running_requests=1 \
  2>&1 | tee -a "${LOG_PATH}"

if [[ -n "${SECRET_REDACTOR_PID}" ]]; then
  kill "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  wait "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  SECRET_REDACTOR_PID=""
fi
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
    "${RUN_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  "${AREAL_VENV}/bin/python" "${JPH_PROJECT_DIR}/scripts/verify_areal_joint_bridge.py" \
  "${BRIDGE_DIR}" \
  --same-backend-score-dir "${SAME_BACKEND_SCORE_DIR}" \
  --model-report "${MODEL_REPORT}" \
  --dataset-report "${DATASET_REPORT}" \
  --expected-areal-commit "${EXPECTED_AREAL_COMMIT}" \
  --expected-project-commit "${PROJECT_COMMIT}" \
  --expected-policy-version 0 \
  --expected-count 4 \
  --device cuda:0 \
  --max-tokens-per-trace 64 \
  --max-abs-error 0.25 \
  --max-mean-abs-error 0.05 \
  --output "${AUDIT_PATH}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "AReaL joint bridge audit passed: ${AUDIT_PATH}" | tee -a "${LOG_PATH}"
