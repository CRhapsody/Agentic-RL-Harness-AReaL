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

while IFS='=' read -r variable_name _; do
  case "${variable_name}" in
    SGLANG_CACHE_DIR) ;;
    SGLANG_*)
      echo "Unexpected inherited SGLang environment variable: ${variable_name}" >&2
      exit 2
      ;;
  esac
done < <(env)

AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
EXPECTED_AREAL_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
EXPECTED_SGLANG_VERSION="0.5.10.post1"
MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/qwen2.5-1.5b-snapshot.json"
DATASET_REPORT="${JPH_ROOT}/artifacts/bootstrap/gsm8k-snapshot.json"
CUDA_TOOLKIT_ROOT="${JPH_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.6}"
MAX_USED_MEMORY_MIB="${JPH_BRIDGE_MAX_USED_MEMORY_MIB:-10240}"
MIN_FREE_MEMORY_MIB="${JPH_BRIDGE_MIN_FREE_MEMORY_MIB:-65536}"
BRIDGE_RUN_KIND="${JPH_BRIDGE_RUN_KIND:-formal-v1}"
TASK_COUNT="${JPH_AREAL_JOINT_BRIDGE_TASKS:-4}"
TASK_OFFSET="${JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET:-0}"
SGLANG_LOGPROB_MODE="${JPH_SGLANG_LOGPROB_MODE:-standard-log-of-softmax-v1}"
CLEAN_ENVIRONMENT_POLICY="${JPH_CLEAN_ENVIRONMENT_POLICY:-filtered-inherited-v1}"

if [[ ! "${TASK_COUNT}" =~ ^[1-8]$ ]]; then
  echo "JPH_AREAL_JOINT_BRIDGE_TASKS must be an integer from 1 through 8" >&2
  exit 2
fi
if [[ ! "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET must be from 0 through 1024" >&2
  exit 2
fi
TASK_OFFSET="$((10#${TASK_OFFSET}))"
if ((TASK_OFFSET > 1024)); then
  echo "JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET must be from 0 through 1024" >&2
  exit 2
fi
DATASET_SELECTION="sequential-valid-offset${TASK_OFFSET}-count${TASK_COUNT}-v1"
case "${SGLANG_LOGPROB_MODE}" in
  standard-log-of-softmax-v1)
    SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE=0
    ;;
  original-log-softmax-v1)
    SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE=1
    ;;
  *)
    echo "Unknown JPH_SGLANG_LOGPROB_MODE: ${SGLANG_LOGPROB_MODE}" >&2
    exit 2
    ;;
esac
case "${BRIDGE_RUN_KIND}" in
  formal-v1)
    if [[ "${TASK_COUNT}" != 4 || "${TASK_OFFSET}" != 0 ]] \
      || [[ "${SGLANG_LOGPROB_MODE}" != standard-log-of-softmax-v1 ]]; then
      echo "formal-v1 requires count=4 offset=0 and the standard log-prob mode" >&2
      exit 2
    fi
    ARTIFACT_GROUP="areal-joint-bridge"
    ;;
  logprob-mechanism-screen-v1)
    if [[ "${TASK_COUNT}" != 4 || "${TASK_OFFSET}" != 32 ]]; then
      echo "logprob mechanism screen requires count=4 and offset=32" >&2
      exit 2
    fi
    if [[ "${CLEAN_ENVIRONMENT_POLICY}" != env-i-v1 ]]; then
      echo "logprob mechanism screen requires the env-i-v1 launch policy" >&2
      exit 2
    fi
    if [[ ! "${JPH_SCREEN_PAIR_ID:-}" =~ ^[A-Za-z0-9._-]{16,160}$ ]]; then
      echo "logprob mechanism screen requires a safe JPH_SCREEN_PAIR_ID" >&2
      exit 2
    fi
    ARTIFACT_GROUP="sglang-logprob-screen/${SGLANG_LOGPROB_MODE}"
    ;;
  *)
    echo "Unknown JPH_BRIDGE_RUN_KIND: ${BRIDGE_RUN_KIND}" >&2
    exit 2
    ;;
esac

for path in "${AREAL_REPO}/.git" "${AREAL_VENV}/bin/python" "${MODEL_REPORT}" "${DATASET_REPORT}" "${CUDA_TOOLKIT_ROOT}/bin/nvcc"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done
SGLANG_VERSION="$(
  "${AREAL_VENV}/bin/python" -c \
    'from importlib.metadata import version; print(version("sglang"))'
)"
if [[ "${SGLANG_VERSION}" != "${EXPECTED_SGLANG_VERSION}" ]]; then
  echo "SGLang version mismatch: ${SGLANG_VERSION}" >&2
  exit 2
fi
SGLANG_ENV_MODE_AUDIT="$(
  SGLANG_RETURN_ORIGINAL_LOGPROB="${SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE}" \
  PYTHONDONTWRITEBYTECODE=1 \
    "${AREAL_VENV}/bin/python" -c \
      'from sglang.srt.environ import envs; print(int(envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()))'
)"
if [[ "${SGLANG_ENV_MODE_AUDIT}" != "${SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE}" ]]; then
  echo "SGLang log-prob environment mode preflight failed" >&2
  exit 2
fi
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
GPU_UUID="$(
  nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader,nounits \
    | tr -d '\r' | head -n 1
)"
GPU_NAME="$(
  nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader,nounits \
    | tr -d '\r' | head -n 1
)"
GPU_DRIVER_VERSION="$(
  nvidia-smi -i "${GPU_ID}" --query-gpu=driver_version \
    --format=csv,noheader,nounits | tr -d '\r' | head -n 1
)"
if [[ -z "${GPU_UUID}" || -z "${GPU_NAME}" || -z "${GPU_DRIVER_VERSION}" ]]; then
  echo "Cannot bind GPU identity for GPU ${GPU_ID}" >&2
  exit 3
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NONCE="$("${AREAL_VENV}/bin/python" -c 'import secrets; print(secrets.token_hex(16))')"
RUN_ID="${RUN_STAMP}-${SGLANG_LOGPROB_MODE}-${RUN_NONCE}"
RUN_ROOT="${JPH_ROOT}/artifacts/${ARTIFACT_GROUP}/${RUN_ID}"
BRIDGE_DIR="${RUN_ROOT}/bridge-records"
SAME_BACKEND_SCORE_DIR="${RUN_ROOT}/same-backend-scores"
AUDIT_PATH="${RUN_ROOT}/audit.json"
NAME_RESOLVE_ROOT="${JPH_ROOT}/runtime/name_resolve/joint-bridge-${RUN_ID}"
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
RUN_TREE_FINALIZED=0
cleanup() {
  if [[ -n "${SECRET_REDACTOR_PID}" ]]; then
    kill "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
    wait "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${RUN_TREE_FINALIZED}" != 1 ]]; then
    JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
      "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
        "${RUN_ROOT}" >/dev/null 2>&1 || true
  fi
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
echo "run_id=${RUN_ID} run_kind=${BRIDGE_RUN_KIND} dataset_selection=${DATASET_SELECTION} sglang_version=${SGLANG_VERSION} generation_logprob_mode=${SGLANG_LOGPROB_MODE}" | tee -a "${LOG_PATH}"
echo "gpu_uuid=${GPU_UUID} gpu_name=${GPU_NAME} driver=${GPU_DRIVER_VERSION} clean_environment_policy=${CLEAN_ENVIRONMENT_POLICY}" | tee -a "${LOG_PATH}"
echo "run_root=${RUN_ROOT} bridge_dir=${BRIDGE_DIR} same_backend_score_dir=${SAME_BACKEND_SCORE_DIR}" | tee -a "${LOG_PATH}"

cd "${AREAL_REPO}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
SGLANG_RETURN_ORIGINAL_LOGPROB="${SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE}" \
JPH_AREAL_JOINT_BRIDGE_TASKS="${TASK_COUNT}" \
JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET="${TASK_OFFSET}" \
JPH_AREAL_JOINT_BRIDGE_DIR="${BRIDGE_DIR}" \
JPH_AREAL_SAME_BACKEND_SCORE_DIR="${SAME_BACKEND_SCORE_DIR}" \
JPH_AREAL_COMMIT="${ACTUAL_AREAL_COMMIT}" \
JPH_PROJECT_COMMIT="${PROJECT_COMMIT}" \
JPH_BEHAVIOR_SNAPSHOT="${MODEL_SNAPSHOT}" \
JPH_BEHAVIOR_REVISION="${MODEL_REVISION}" \
JPH_DATASET_REVISION="${DATASET_REVISION}" \
JPH_DATASET_SELECTION="${DATASET_SELECTION}" \
JPH_SGLANG_LOGPROB_MODE="${SGLANG_LOGPROB_MODE}" \
JPH_SGLANG_VERSION="${SGLANG_VERSION}" \
JPH_RUN_ID="${RUN_ID}" \
JPH_SCREEN_PAIR_ID="${JPH_SCREEN_PAIR_ID:-}" \
JPH_CLEAN_ENVIRONMENT_POLICY="${CLEAN_ENVIRONMENT_POLICY}" \
JPH_PHYSICAL_GPU_ID="${GPU_ID}" \
JPH_GPU_UUID="${GPU_UUID}" \
JPH_GPU_NAME="${GPU_NAME}" \
JPH_GPU_DRIVER_VERSION="${GPU_DRIVER_VERSION}" \
JPH_EXPECTED_POLICY_VERSION=0 \
  "${AREAL_VENV}/bin/python" "${JPH_PROJECT_DIR}/scripts/run_areal_joint_bridge_eval.py" \
  --config "${AREAL_REPO}/examples/math/gsm8k_grpo.yaml" \
  scheduler.type=local \
  experiment_name=jph-areal-joint-bridge \
  trial_name="${RUN_ID}" \
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
if [[ -n "${GPU_MONITOR_PID}" ]]; then
  kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  GPU_MONITOR_PID=""
fi
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
    "${RUN_ROOT}"

if [[ "${BRIDGE_RUN_KIND}" == logprob-mechanism-screen-v1 ]]; then
  echo "SGLang log-prob screen cell finished; beginning immutable final audit" \
    >> "${LOG_PATH}"
  JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    "${AREAL_VENV}/bin/python" \
      "${SCRIPT_DIR}/audit_sglang_logprob_screen_cell.py" "${RUN_ROOT}"
  RUN_TREE_FINALIZED=1
  POINTER_PATH="${JPH_SCREEN_CELL_POINTER:-}"
  if [[ "${SGLANG_LOGPROB_MODE}" == standard-log-of-softmax-v1 ]]; then
    SCREEN_CELL=c0
  else
    SCREEN_CELL=c1
  fi
  "${AREAL_VENV}/bin/python" \
    "${SCRIPT_DIR}/write_sglang_logprob_screen_pointer.py" \
    --pair-id "${JPH_SCREEN_PAIR_ID}" \
    --cell "${SCREEN_CELL}" \
    --pointer "${POINTER_PATH}" \
    --run-root "${RUN_ROOT}"
  echo "SGLang log-prob screen cell complete: ${RUN_ROOT}"
  exit 0
fi

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
  --expected-dataset-selection "${DATASET_SELECTION}" \
  --expected-generation-logprob-mode "${SGLANG_LOGPROB_MODE}" \
  --expected-sglang-version "${SGLANG_VERSION}" \
  --expected-count "${TASK_COUNT}" \
  --device cuda:0 \
  --max-tokens-per-trace 64 \
  --max-abs-error 0.25 \
  --max-mean-abs-error 0.05 \
  --output "${AUDIT_PATH}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "AReaL joint bridge audit passed: ${AUDIT_PATH}" | tee -a "${LOG_PATH}"
