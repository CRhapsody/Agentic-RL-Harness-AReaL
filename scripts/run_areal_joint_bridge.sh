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
source "${SCRIPT_DIR}/m0_gpu_lock.sh"
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
SGLANG_DISABLE_CUDA_GRAPH="${JPH_SGLANG_DISABLE_CUDA_GRAPH:-false}"
EXPERIMENTAL_AXIS="${JPH_EXPERIMENTAL_AXIS:-none-v1}"
HARNESS_CONTROLLER_KIND="${JPH_HARNESS_CONTROLLER_KIND:-tabular}"
HARNESS_HIDDEN_SIZE="${JPH_HARNESS_HIDDEN_SIZE:-32}"
RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH="${JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH:-}"
RLVR_RUNNER_ADMISSION_MODE=""
SGLANG_MEM_FRACTION_STATIC="0.35"
MAX_NEW_GPU_MEMORY_MIB=30720
REQUIRE_EMPTY_COMPUTE_PROCESSES=false

if [[ ! "${TASK_COUNT}" =~ ^[1-8]$ ]]; then
  echo "JPH_AREAL_JOINT_BRIDGE_TASKS must be an integer from 1 through 8" >&2
  exit 2
fi
if [[ ! "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET must be from 0 through 1024" >&2
  exit 2
fi
if [[ ! "${HARNESS_HIDDEN_SIZE}" =~ ^[1-9][0-9]{0,3}$ ]] \
  || ((10#${HARNESS_HIDDEN_SIZE} > 1024)); then
  echo "JPH_HARNESS_HIDDEN_SIZE must be an integer from 1 through 1024" >&2
  exit 2
fi
if [[ "${HARNESS_CONTROLLER_KIND}" != tabular ]] \
  && [[ "${HARNESS_CONTROLLER_KIND}" != torch ]]; then
  echo "JPH_HARNESS_CONTROLLER_KIND must be tabular or torch" >&2
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
case "${SGLANG_DISABLE_CUDA_GRAPH}" in
  true|false) ;;
  *)
    echo "JPH_SGLANG_DISABLE_CUDA_GRAPH must be true or false" >&2
    exit 2
    ;;
esac
case "${BRIDGE_RUN_KIND}" in
  formal-v1)
    if [[ "${TASK_COUNT}" != 4 || "${TASK_OFFSET}" != 0 ]] \
      || [[ "${SGLANG_LOGPROB_MODE}" != standard-log-of-softmax-v1 ]] \
      || [[ "${SGLANG_DISABLE_CUDA_GRAPH}" != false ]] \
      || [[ "${EXPERIMENTAL_AXIS}" != none-v1 ]] \
      || [[ "${HARNESS_CONTROLLER_KIND}" != tabular ]]; then
      echo "formal-v1 requires count=4 offset=0, standard log-prob, CUDA Graph enabled, and no experimental axis" >&2
      exit 2
    fi
    ARTIFACT_GROUP="areal-joint-bridge"
    ;;
  m0-torch-joint-v1)
    if [[ "${TASK_COUNT}" != 4 || "${TASK_OFFSET}" != 0 ]] \
      || [[ "${SGLANG_LOGPROB_MODE}" != standard-log-of-softmax-v1 ]] \
      || [[ "${SGLANG_DISABLE_CUDA_GRAPH}" != false ]] \
      || [[ "${EXPERIMENTAL_AXIS}" != none-v1 ]] \
      || [[ "${HARNESS_CONTROLLER_KIND}" != torch ]]; then
      echo "m0-torch-joint-v1 requires count=4 offset=0, Torch Harness, standard log-prob, CUDA Graph enabled, and no experimental axis" >&2
      exit 2
    fi
    if [[ -z "${RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH}" ]]; then
      echo "m0-torch-joint-v1 requires JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH" >&2
      exit 2
    fi
    case "${RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH}" in
      "${JPH_ROOT}"/*) ;;
      *)
        echo "RLVR frozen estimator template escapes ${JPH_ROOT}" >&2
        exit 2
        ;;
    esac
    ARTIFACT_GROUP="m0-torch-joint-rollout"
    RLVR_RUNNER_ADMISSION_MODE="m0-torch-joint-v1"
    # The first shared-GPU M0 task must keep the SGLang static allocation in
    # the audited 24--26 GiB envelope on an 80 GiB A100.  Do not inherit the
    # older 0.35 bridge setting merely because the surrounding launch path is
    # shared.
    SGLANG_MEM_FRACTION_STATIC="0.29"
    MAX_NEW_GPU_MEMORY_MIB=26624
    REQUIRE_EMPTY_COMPUTE_PROCESSES=true
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
    if [[ "${HARNESS_CONTROLLER_KIND}" != tabular ]]; then
      echo "logprob mechanism screen requires the tabular control Harness" >&2
      exit 2
    fi
    if [[ ! "${JPH_SCREEN_PAIR_ID:-}" =~ ^[A-Za-z0-9._-]{16,160}$ ]]; then
      echo "logprob mechanism screen requires a safe JPH_SCREEN_PAIR_ID" >&2
      exit 2
    fi
    if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" != false ]] \
      || [[ "${EXPERIMENTAL_AXIS}" != generation-logprob-formula-v1 ]]; then
      echo "logprob mechanism screen requires CUDA Graph enabled and the formula axis" >&2
      exit 2
    fi
    ARTIFACT_GROUP="sglang-logprob-screen/${SGLANG_LOGPROB_MODE}"
    ;;
  cuda-graph-mechanism-screen-v1)
    if [[ "${TASK_COUNT}" != 4 || "${TASK_OFFSET}" != 64 ]] \
      || [[ "${SGLANG_LOGPROB_MODE}" != standard-log-of-softmax-v1 ]]; then
      echo "CUDA Graph screen requires count=4 offset=64 and standard log-prob" >&2
      exit 2
    fi
    if [[ "${CLEAN_ENVIRONMENT_POLICY}" != env-i-v1 ]]; then
      echo "CUDA Graph screen requires the env-i-v1 launch policy" >&2
      exit 2
    fi
    if [[ "${HARNESS_CONTROLLER_KIND}" != tabular ]]; then
      echo "CUDA Graph screen requires the tabular control Harness" >&2
      exit 2
    fi
    if [[ ! "${JPH_SCREEN_PAIR_ID:-}" =~ ^[A-Za-z0-9._-]{16,160}$ ]]; then
      echo "CUDA Graph screen requires a safe JPH_SCREEN_PAIR_ID" >&2
      exit 2
    fi
    if [[ "${EXPERIMENTAL_AXIS}" != cuda-graph-v1 ]]; then
      echo "CUDA Graph screen requires the cuda-graph-v1 axis" >&2
      exit 2
    fi
    if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == false ]]; then
      CUDA_GRAPH_CELL=c2a
      CUDA_GRAPH_VARIANT=cuda-graph-enabled-v1
    else
      CUDA_GRAPH_CELL=c2b
      CUDA_GRAPH_VARIANT=cuda-graph-disabled-v1
    fi
    ARTIFACT_GROUP="sglang-cuda-graph-screen/${CUDA_GRAPH_VARIANT}"
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
NVCC_PREFLIGHT_OBJECT="$(
  mktemp "${JPH_ROOT}/tmp/nvcc-joint-bridge-preflight.XXXXXXXX.o"
)"
if ! "${CUDACXX}" -std=c++20 -x cu -c /dev/null \
  -o "${NVCC_PREFLIGHT_OBJECT}"; then
  rm -f -- "${NVCC_PREFLIGHT_OBJECT}"
  exit 2
fi
rm -f -- "${NVCC_PREFLIGHT_OBJECT}"
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

if [[ -n "${RLVR_RUNNER_ADMISSION_MODE}" ]]; then
  PYTHONPATH="${JPH_PROJECT_DIR}:${AREAL_REPO}" \
    "${AREAL_VENV}/bin/python" -c \
    'import sys; from jphrl.trajectory.rlvr_workflow_admission import load_frozen_dual_credit_estimator_template_file; load_frozen_dual_credit_estimator_template_file(sys.argv[1], allowed_root=sys.argv[2])' \
    "${RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH}" "${JPH_ROOT}"
fi

if ! jph_acquire_m0_gpu_lock "${GPU_ID}"; then
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
GPU_MEMORY_USED_AT_LAUNCH="${GPU_MEMORY_USED}"
GPU_MEMORY_FREE_AT_LAUNCH="${GPU_MEMORY_FREE}"
GPU_PROCESS_SNAPSHOT="$({
  nvidia-smi -i "${GPU_ID}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
} 2>&1)"
if [[ "${REQUIRE_EMPTY_COMPUTE_PROCESSES}" == true ]] \
  && [[ -n "${GPU_PROCESS_SNAPSHOT}" ]]; then
  echo "GPU ${GPU_ID} has an existing compute process; M0 will wait for an idle card" >&2
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
if [[ "${BRIDGE_RUN_KIND}" == cuda-graph-mechanism-screen-v1 ]]; then
  RUN_ID="${RUN_STAMP}-${CUDA_GRAPH_VARIANT}-${RUN_NONCE}"
else
  RUN_ID="${RUN_STAMP}-${SGLANG_LOGPROB_MODE}-${RUN_NONCE}"
fi
RUN_ROOT="${JPH_ROOT}/artifacts/${ARTIFACT_GROUP}/${RUN_ID}"
BRIDGE_DIR="${RUN_ROOT}/bridge-records"
RLVR_RUNNER_ADMISSION_DIR=""
if [[ -n "${RLVR_RUNNER_ADMISSION_MODE}" ]]; then
  RLVR_RUNNER_ADMISSION_DIR="${RUN_ROOT}/rlvr-runner-admissions"
fi
SAME_BACKEND_SCORE_DIR="${RUN_ROOT}/same-backend-scores"
AUDIT_PATH="${RUN_ROOT}/audit.json"
NAME_RESOLVE_ROOT="${JPH_ROOT}/runtime/name_resolve/joint-bridge-${RUN_ID}"
LOG_PATH="${RUN_ROOT}/run.log"
MEMORY_SAMPLES="${RUN_ROOT}/gpu-memory.csv"
MEMORY_BREACH="${RUN_ROOT}/gpu-memory-breach.txt"
WATCHDOG_TARGET="${RUN_ROOT}/watchdog-target-pgid.txt"
RUN_ADMIN_API_KEY="$(
  "${AREAL_VENV}/bin/python" -c \
    'import secrets; print("jph-bridge-" + secrets.token_urlsafe(32))'
)"
mkdir -p -m 700 \
  "${BRIDGE_DIR}" \
  "${SAME_BACKEND_SCORE_DIR}" \
  "${NAME_RESOLVE_ROOT}"
if [[ -n "${RLVR_RUNNER_ADMISSION_MODE}" ]]; then
  mkdir -m 700 "${RLVR_RUNNER_ADMISSION_DIR}"
fi
touch "${LOG_PATH}"
chmod 600 "${LOG_PATH}"

GPU_MONITOR_PID=""
GPU_MEMORY_MONITOR_PID=""
SECRET_REDACTOR_PID=""
RUN_TREE_FINALIZED=0
RUN_COMMAND_SESSION_ID=""
stop_gpu_monitors() {
  if [[ -n "${GPU_MEMORY_MONITOR_PID}" ]]; then
    kill "${GPU_MEMORY_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MEMORY_MONITOR_PID}" >/dev/null 2>&1 || true
    GPU_MEMORY_MONITOR_PID=""
  fi
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    GPU_MONITOR_PID=""
  fi
}
audit_gpu_memory() {
  if [[ ! -s "${MEMORY_SAMPLES}" ]] \
    || [[ -e "${RUN_ROOT}/gpu-memory-audit.json" ]]; then
    return 0
  fi
  "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/audit_gpu_memory_envelope.py" \
    --samples "${MEMORY_SAMPLES}" \
    --output "${RUN_ROOT}/gpu-memory-audit.json" \
    --physical-gpu-id "${GPU_ID}" \
    --baseline-used-mib "${GPU_MEMORY_USED_AT_LAUNCH}" \
    --max-new-memory-mib "${MAX_NEW_GPU_MEMORY_MIB}" \
    --run-kind "${BRIDGE_RUN_KIND}" \
    --project-commit "${PROJECT_COMMIT}" \
    >> "${LOG_PATH}"
}
stop_run_session() {
  local run_pids
  run_pids=""
  if [[ -n "${RUN_COMMAND_SESSION_ID}" ]]; then
    run_pids="$(
      ps -eo pid=,sid= \
        | awk -v sid="${RUN_COMMAND_SESSION_ID}" '$2 == sid {print $1}' \
        | tr '\n' ' '
    )"
  fi
  if [[ -n "${run_pids// /}" ]]; then
    kill -TERM ${run_pids} >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
      run_pids="$(
        ps -eo pid=,sid= \
          | awk -v sid="${RUN_COMMAND_SESSION_ID}" '$2 == sid {print $1}' \
          | tr '\n' ' '
      )"
      if [[ -z "${run_pids// /}" ]]; then
        break
      fi
      sleep 1
    done
    if [[ -n "${run_pids// /}" ]]; then
      kill -KILL ${run_pids} >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${WATCHDOG_TARGET}"
}
redact_and_check_secret() {
  if ! JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
      --verify-absent "${RUN_ROOT}" >/dev/null; then
    echo "Runtime admin API key redaction or absence verification failed" >&2
    return 1
  fi
  return 0
}
assert_no_run_gpu_processes() {
  local processes pid process_sid
  if ! processes="$({
    nvidia-smi -i "${GPU_ID}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits
  } 2>/dev/null)"; then
    echo "Cannot verify bridge GPU process cleanup" >&2
    return 1
  fi
  while IFS=, read -r pid _; do
    pid="${pid// /}"
    if [[ -z "${pid}" ]] || [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
      continue
    fi
    process_sid="$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${RUN_COMMAND_SESSION_ID}" ]] \
      && [[ "${process_sid}" == "${RUN_COMMAND_SESSION_ID}" ]]; then
      echo "Bridge cleanup left a project GPU process: pid=${pid}" >&2
      return 1
    fi
  done <<< "${processes}"
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_run_session
  if [[ -n "${SECRET_REDACTOR_PID}" ]]; then
    kill "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
    wait "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
    SECRET_REDACTOR_PID=""
  fi
  stop_gpu_monitors
  if [[ "${RUN_TREE_FINALIZED}" != 1 ]]; then
    audit_gpu_memory || status=4
    redact_and_check_secret || status=4
    assert_no_run_gpu_processes || status=4
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM
nvidia-smi dmon -i "${GPU_ID}" -s pucm -d 5 -o TD > "${RUN_ROOT}/gpu-dmon.log" 2>&1 &
GPU_MONITOR_PID="$!"
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
    --watch-seconds 1200 "${RUN_ROOT}" >/dev/null 2>&1 &
SECRET_REDACTOR_PID="$!"

echo "project=${PROJECT_COMMIT} AReaL=${ACTUAL_AREAL_COMMIT} physical_gpu=${GPU_ID} model_revision=${MODEL_REVISION} dataset_revision=${DATASET_REVISION}" | tee -a "${LOG_PATH}"
echo "run_id=${RUN_ID} run_kind=${BRIDGE_RUN_KIND} dataset_selection=${DATASET_SELECTION} sglang_version=${SGLANG_VERSION} generation_logprob_mode=${SGLANG_LOGPROB_MODE} disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH} experimental_axis=${EXPERIMENTAL_AXIS} harness_controller=${HARNESS_CONTROLLER_KIND} harness_hidden_size=${HARNESS_HIDDEN_SIZE} sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}" | tee -a "${LOG_PATH}"
echo "gpu_uuid=${GPU_UUID} gpu_name=${GPU_NAME} driver=${GPU_DRIVER_VERSION} clean_environment_policy=${CLEAN_ENVIRONMENT_POLICY}" | tee -a "${LOG_PATH}"
if [[ -n "${GPU_PROCESS_SNAPSHOT}" ]]; then
  echo "gpu_compute_processes_at_preflight:" | tee -a "${LOG_PATH}"
  printf '%s\n' "${GPU_PROCESS_SNAPSHOT}" | tee -a "${LOG_PATH}"
else
  echo "gpu_compute_processes_at_preflight=none" | tee -a "${LOG_PATH}"
fi
echo "run_root=${RUN_ROOT} bridge_dir=${BRIDGE_DIR} same_backend_score_dir=${SAME_BACKEND_SCORE_DIR}" | tee -a "${LOG_PATH}"

IFS=, read -r GPU_MEMORY_USED_AT_LAUNCH GPU_MEMORY_FREE_AT_LAUNCH < <(
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=memory.used,memory.free \
    --format=csv,noheader,nounits | tr -d ' '
)
GPU_PROCESS_SNAPSHOT_AT_LAUNCH="$({
  nvidia-smi -i "${GPU_ID}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
} 2>&1)"
if [[ "${REQUIRE_EMPTY_COMPUTE_PROCESSES}" == true ]] \
  && [[ -n "${GPU_PROCESS_SNAPSHOT_AT_LAUNCH}" ]]; then
  echo "GPU ${GPU_ID} gained a compute process before launch; M0 remains fail closed" >&2
  exit 3
fi
if [[ ! "${GPU_MEMORY_USED_AT_LAUNCH}" =~ ^[0-9]+$ ]] \
  || [[ ! "${GPU_MEMORY_FREE_AT_LAUNCH}" =~ ^[0-9]+$ ]]; then
  echo "Cannot reread GPU ${GPU_ID} memory state immediately before launch" >&2
  exit 3
fi
if ((GPU_MEMORY_USED_AT_LAUNCH > MAX_USED_MEMORY_MIB \
  || GPU_MEMORY_FREE_AT_LAUNCH < MIN_FREE_MEMORY_MIB)); then
  echo "GPU ${GPU_ID} headroom changed before launch: used=${GPU_MEMORY_USED_AT_LAUNCH}MiB free=${GPU_MEMORY_FREE_AT_LAUNCH}MiB" >&2
  exit 3
fi
echo "gpu_memory_at_launch=used:${GPU_MEMORY_USED_AT_LAUNCH}MiB,free:${GPU_MEMORY_FREE_AT_LAUNCH}MiB" | tee -a "${LOG_PATH}"
if [[ -n "${GPU_PROCESS_SNAPSHOT_AT_LAUNCH}" ]]; then
  echo "gpu_compute_processes_at_launch:" | tee -a "${LOG_PATH}"
  printf '%s\n' "${GPU_PROCESS_SNAPSHOT_AT_LAUNCH}" | tee -a "${LOG_PATH}"
else
  echo "gpu_compute_processes_at_launch=none" | tee -a "${LOG_PATH}"
fi
cd "${AREAL_REPO}"
rm -f "${MEMORY_BREACH}" "${WATCHDOG_TARGET}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
SGLANG_RETURN_ORIGINAL_LOGPROB="${SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE}" \
JPH_AREAL_JOINT_BRIDGE_TASKS="${TASK_COUNT}" \
JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET="${TASK_OFFSET}" \
JPH_AREAL_JOINT_BRIDGE_DIR="${BRIDGE_DIR}" \
JPH_RLVR_RUNNER_ADMISSION_MODE="${RLVR_RUNNER_ADMISSION_MODE}" \
JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH="${RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH}" \
JPH_RLVR_RUNNER_ADMISSION_DIR="${RLVR_RUNNER_ADMISSION_DIR}" \
JPH_AREAL_SAME_BACKEND_SCORE_DIR="${SAME_BACKEND_SCORE_DIR}" \
JPH_AREAL_COMMIT="${ACTUAL_AREAL_COMMIT}" \
JPH_PROJECT_COMMIT="${PROJECT_COMMIT}" \
JPH_BEHAVIOR_SNAPSHOT="${MODEL_SNAPSHOT}" \
JPH_BEHAVIOR_REVISION="${MODEL_REVISION}" \
JPH_DATASET_REVISION="${DATASET_REVISION}" \
JPH_DATASET_SELECTION="${DATASET_SELECTION}" \
JPH_SGLANG_LOGPROB_MODE="${SGLANG_LOGPROB_MODE}" \
JPH_SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH}" \
JPH_EXPERIMENTAL_AXIS="${EXPERIMENTAL_AXIS}" \
JPH_HARNESS_CONTROLLER_KIND="${HARNESS_CONTROLLER_KIND}" \
JPH_HARNESS_HIDDEN_SIZE="${HARNESS_HIDDEN_SIZE}" \
JPH_SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC}" \
JPH_SGLANG_VERSION="${SGLANG_VERSION}" \
JPH_RUN_ID="${RUN_ID}" \
JPH_SCREEN_PAIR_ID="${JPH_SCREEN_PAIR_ID:-}" \
JPH_CLEAN_ENVIRONMENT_POLICY="${CLEAN_ENVIRONMENT_POLICY}" \
JPH_PHYSICAL_GPU_ID="${GPU_ID}" \
JPH_GPU_UUID="${GPU_UUID}" \
JPH_GPU_NAME="${GPU_NAME}" \
JPH_GPU_DRIVER_VERSION="${GPU_DRIVER_VERSION}" \
JPH_EXPECTED_POLICY_VERSION=0 \
  setsid "${AREAL_VENV}/bin/python" "${JPH_PROJECT_DIR}/scripts/run_areal_joint_bridge_eval.py" \
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
  sglang.mem_fraction_static="${SGLANG_MEM_FRACTION_STATIC}" \
  +sglang.disable_cuda_graph="${SGLANG_DISABLE_CUDA_GRAPH}" \
  sglang.context_length=1024 \
  sglang.max_running_requests=1 \
  > >(tee -a "${LOG_PATH}") 2>&1 &
RUN_COMMAND_SESSION_ID="$!"
printf '%s\n' "${RUN_COMMAND_SESSION_ID}" > "${WATCHDOG_TARGET}"
chmod 600 "${WATCHDOG_TARGET}"
(
  while true; do
    if IFS=, read -r sampled_used_mib sampled_free_mib < <(
      nvidia-smi -i "${GPU_ID}" \
        --query-gpu=memory.used,memory.free \
        --format=csv,noheader,nounits | tr -d ' '
    ); then
      printf '%s,%s,%s\n' \
        "$(date -u +%s)" "${sampled_used_mib}" "${sampled_free_mib}" \
        >> "${MEMORY_SAMPLES}"
      if [[ "${sampled_used_mib}" =~ ^[0-9]+$ ]] \
        && ((sampled_used_mib - GPU_MEMORY_USED_AT_LAUNCH > MAX_NEW_GPU_MEMORY_MIB)); then
        printf 'used=%s baseline=%s delta=%s limit=%s\n' \
          "${sampled_used_mib}" \
          "${GPU_MEMORY_USED_AT_LAUNCH}" \
          "$((sampled_used_mib - GPU_MEMORY_USED_AT_LAUNCH))" \
          "${MAX_NEW_GPU_MEMORY_MIB}" > "${MEMORY_BREACH}"
        chmod 600 "${MEMORY_BREACH}"
        target_session="$(tr -d ' ' < "${WATCHDOG_TARGET}" 2>/dev/null || true)"
        if [[ "${target_session}" =~ ^[0-9]+$ ]]; then
          target_pids="$(
            ps -eo pid=,sid= \
              | awk -v sid="${target_session}" '$2 == sid {print $1}' \
              | tr '\n' ' '
          )"
          if [[ -n "${target_pids// /}" ]]; then
            kill -TERM ${target_pids} >/dev/null 2>&1 || true
          fi
        fi
        exit 4
      fi
    else
      printf '%s,ERROR,ERROR\n' "$(date -u +%s)" >> "${MEMORY_SAMPLES}"
    fi
    sleep 1
  done
) &
GPU_MEMORY_MONITOR_PID="$!"
if ! wait "${RUN_COMMAND_SESSION_ID}"; then
  if [[ -f "${MEMORY_BREACH}" ]]; then
    echo "Bridge run was stopped by the GPU memory watchdog" >&2
  else
    echo "AReaL bridge rollout process failed" >&2
  fi
  exit 4
fi
rm -f "${WATCHDOG_TARGET}"
assert_no_run_gpu_processes

if [[ -n "${SECRET_REDACTOR_PID}" ]]; then
  kill "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  wait "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  SECRET_REDACTOR_PID=""
fi
redact_and_check_secret

if [[ "${BRIDGE_RUN_KIND}" == logprob-mechanism-screen-v1 ]] \
  || [[ "${BRIDGE_RUN_KIND}" == cuda-graph-mechanism-screen-v1 ]]; then
  stop_gpu_monitors
  audit_gpu_memory
  echo "SGLang log-prob screen cell finished; beginning immutable final audit" \
    >> "${LOG_PATH}"
  JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    "${AREAL_VENV}/bin/python" \
      "${SCRIPT_DIR}/audit_sglang_logprob_screen_cell.py" "${RUN_ROOT}"
  RUN_TREE_FINALIZED=1
  POINTER_PATH="${JPH_SCREEN_CELL_POINTER:-}"
  if [[ "${BRIDGE_RUN_KIND}" == cuda-graph-mechanism-screen-v1 ]]; then
    SCREEN_CELL="${CUDA_GRAPH_CELL}"
    POINTER_GROUP=sglang-cuda-graph-screen
  elif [[ "${SGLANG_LOGPROB_MODE}" == standard-log-of-softmax-v1 ]]; then
    SCREEN_CELL=c0
    POINTER_GROUP=sglang-logprob-screen
  else
    SCREEN_CELL=c1
    POINTER_GROUP=sglang-logprob-screen
  fi
  "${AREAL_VENV}/bin/python" \
    "${SCRIPT_DIR}/write_sglang_logprob_screen_pointer.py" \
    --pair-id "${JPH_SCREEN_PAIR_ID}" \
    --pair-artifact-group "${POINTER_GROUP}" \
    --cell "${SCREEN_CELL}" \
    --pointer "${POINTER_PATH}" \
    --run-root "${RUN_ROOT}"
  echo "SGLang log-prob screen cell complete: ${RUN_ROOT}"
  exit 0
fi

IFS=, read -r GPU_MEMORY_USED_BEFORE_VERIFY GPU_MEMORY_FREE_BEFORE_VERIFY < <(
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=memory.used,memory.free \
    --format=csv,noheader,nounits | tr -d ' '
)
GPU_PROCESS_SNAPSHOT_BEFORE_VERIFY="$({
  nvidia-smi -i "${GPU_ID}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
} 2>&1)"
if [[ ! "${GPU_MEMORY_USED_BEFORE_VERIFY}" =~ ^[0-9]+$ ]] \
  || [[ ! "${GPU_MEMORY_FREE_BEFORE_VERIFY}" =~ ^[0-9]+$ ]]; then
  echo "Cannot reread GPU ${GPU_ID} immediately before bridge verification" >&2
  exit 3
fi
if ((GPU_MEMORY_USED_BEFORE_VERIFY > MAX_USED_MEMORY_MIB \
  || GPU_MEMORY_FREE_BEFORE_VERIFY < MIN_FREE_MEMORY_MIB)); then
  echo "GPU ${GPU_ID} lacks headroom before bridge verification: used=${GPU_MEMORY_USED_BEFORE_VERIFY}MiB free=${GPU_MEMORY_FREE_BEFORE_VERIFY}MiB" >&2
  exit 3
fi
if [[ "${REQUIRE_EMPTY_COMPUTE_PROCESSES}" == true ]] \
  && [[ -n "${GPU_PROCESS_SNAPSHOT_BEFORE_VERIFY}" ]]; then
  echo "GPU ${GPU_ID} gained a compute process before M0 verification" >&2
  exit 3
fi
echo "gpu_memory_before_verify=used:${GPU_MEMORY_USED_BEFORE_VERIFY}MiB,free:${GPU_MEMORY_FREE_BEFORE_VERIFY}MiB" \
  | tee -a "${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  setsid "${AREAL_VENV}/bin/python" "${JPH_PROJECT_DIR}/scripts/verify_areal_joint_bridge.py" \
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
  > >(tee -a "${LOG_PATH}") 2>&1 &
RUN_COMMAND_SESSION_ID="$!"
printf '%s\n' "${RUN_COMMAND_SESSION_ID}" > "${WATCHDOG_TARGET}"
chmod 600 "${WATCHDOG_TARGET}"
if ! wait "${RUN_COMMAND_SESSION_ID}"; then
  if [[ -f "${MEMORY_BREACH}" ]]; then
    echo "Bridge verification was stopped by the GPU memory watchdog" >&2
  else
    echo "AReaL bridge verification process failed" >&2
  fi
  exit 4
fi
rm -f "${WATCHDOG_TARGET}"
assert_no_run_gpu_processes

stop_gpu_monitors
audit_gpu_memory
redact_and_check_secret
RUN_TREE_FINALIZED=1
echo "AReaL joint bridge audit passed: ${AUDIT_PATH}" | tee -a "${LOG_PATH}"
