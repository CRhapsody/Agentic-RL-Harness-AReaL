#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_m0_live_joint.sh GPU_ID ROLLOUT_RUN_ROOT" >&2
  exit 2
fi
GPU_ID="$1"
ROLLOUT_RUN_ROOT="$2"
if [[ ! "${GPU_ID}" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be an integer from 0 through 7: ${GPU_ID}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"
source "${SCRIPT_DIR}/m0_gpu_lock.sh"
umask 077

AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
EXPECTED_AREAL_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/qwen2.5-1.5b-snapshot.json"
MAX_USED_MEMORY_MIB=10240
MIN_FREE_MEMORY_MIB=65536
MAX_NEW_GPU_MEMORY_MIB=26624
EXPECTED_MASTER_PORT="$((61000 + GPU_ID))"
M0_MASTER_PORT="${JPH_M0_MASTER_PORT:-${EXPECTED_MASTER_PORT}}"
if [[ "${M0_MASTER_PORT}" != "${EXPECTED_MASTER_PORT}" ]]; then
  echo "M0 MASTER_PORT must be ${EXPECTED_MASTER_PORT} for GPU ${GPU_ID}" >&2
  exit 2
fi

for path in \
  "${AREAL_REPO}/.git" \
  "${AREAL_VENV}/bin/python" \
  "${MODEL_REPORT}" \
  "${ROLLOUT_RUN_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done
ROLLOUT_RUN_ROOT="$(realpath "${ROLLOUT_RUN_ROOT}")"
case "${ROLLOUT_RUN_ROOT}" in
  "${JPH_ROOT}/artifacts/m0-torch-joint-rollout/"*) ;;
  *)
    echo "Rollout source is outside the M0 rollout artifact group" >&2
    exit 2
    ;;
esac
if [[ -L "${ROLLOUT_RUN_ROOT}" ]] || [[ ! -d "${ROLLOUT_RUN_ROOT}" ]]; then
  echo "Rollout source root is unsafe" >&2
  exit 2
fi
RUNNER_ADMISSION_DIR="${ROLLOUT_RUN_ROOT}/rlvr-runner-admissions"
if [[ ! -d "${RUNNER_ADMISSION_DIR}" ]] || [[ -L "${RUNNER_ADMISSION_DIR}" ]]; then
  echo "Rollout runner-admission directory is missing or unsafe" >&2
  exit 2
fi

ACTUAL_AREAL_COMMIT="$(git -C "${AREAL_REPO}" rev-parse HEAD)"
PROJECT_COMMIT="$(git -C "${JPH_PROJECT_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_AREAL_COMMIT}" != "${EXPECTED_AREAL_COMMIT}" ]]; then
  echo "AReaL commit mismatch: ${ACTUAL_AREAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${AREAL_REPO}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Pinned AReaL worktree must be clean before M0" >&2
  exit 2
fi
if [[ -n "$(git -C "${JPH_PROJECT_DIR}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Project worktree must be clean before M0" >&2
  exit 2
fi
if [[ -n "${JPH_PROJECT_COMMIT:-}" ]] \
  && [[ "${JPH_PROJECT_COMMIT}" != "${PROJECT_COMMIT}" ]]; then
  echo "Project commit mismatch: expected=${JPH_PROJECT_COMMIT} actual=${PROJECT_COMMIT}" >&2
  exit 2
fi

readarray -t MODEL_VALUES < <(
  "${AREAL_VENV}/bin/python" -c '
import json, sys
record = json.load(open(sys.argv[1]))
print(record["snapshot_path"])
print(record["resolved_commit"])
' "${MODEL_REPORT}"
)
MODEL_SNAPSHOT="${MODEL_VALUES[0]}"
MODEL_REVISION="${MODEL_VALUES[1]}"
if [[ ! -d "${MODEL_SNAPSHOT}" ]] || [[ -L "${MODEL_SNAPSHOT}" ]]; then
  echo "Pinned model snapshot is missing or unsafe" >&2
  exit 2
fi
case "${MODEL_SNAPSHOT}" in
  "${JPH_ROOT}/"*) ;;
  *)
    echo "Pinned model snapshot escapes ${JPH_ROOT}" >&2
    exit 2
    ;;
esac

if ! jph_acquire_m0_gpu_lock "${GPU_ID}"; then
  exit 3
fi

read_gpu_memory() {
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=memory.used,memory.free \
    --format=csv,noheader,nounits | tr -d ' '
}
read_gpu_processes() {
  nvidia-smi -i "${GPU_ID}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
}
require_idle_gpu() {
  local stage="$1"
  local memory_used memory_free processes
  IFS=, read -r memory_used memory_free < <(read_gpu_memory)
  processes="$(read_gpu_processes)"
  if [[ ! "${memory_used}" =~ ^[0-9]+$ ]] \
    || [[ ! "${memory_free}" =~ ^[0-9]+$ ]]; then
    echo "Cannot read GPU ${GPU_ID} memory at ${stage}" >&2
    exit 3
  fi
  if ((memory_used > MAX_USED_MEMORY_MIB || memory_free < MIN_FREE_MEMORY_MIB)); then
    echo "GPU ${GPU_ID} lacks M0 headroom at ${stage}: used=${memory_used}MiB free=${memory_free}MiB" >&2
    exit 3
  fi
  if [[ -n "${processes}" ]]; then
    echo "GPU ${GPU_ID} has an existing compute process at ${stage}; M0 remains fail closed" >&2
    exit 3
  fi
  GPU_MEMORY_USED_SNAPSHOT="${memory_used}"
  GPU_MEMORY_FREE_SNAPSHOT="${memory_free}"
}

require_idle_gpu "preflight"
GPU_MEMORY_USED_PREFLIGHT="${GPU_MEMORY_USED_SNAPSHOT}"
GPU_MEMORY_FREE_PREFLIGHT="${GPU_MEMORY_FREE_SNAPSHOT}"
GPU_MEMORY_USED_AT_LAUNCH="${GPU_MEMORY_USED_PREFLIGHT}"
GPU_MEMORY_FREE_AT_LAUNCH="${GPU_MEMORY_FREE_PREFLIGHT}"
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

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NONCE="$("${AREAL_VENV}/bin/python" -c 'import secrets; print(secrets.token_hex(16))')"
RUN_ID="${RUN_STAMP}-m0-live-joint-${RUN_NONCE}"
RUN_ROOT="${JPH_ROOT}/artifacts/m0-live-joint/${RUN_ID}"
SELECTION_ROOT="${RUN_ROOT}/selection"
ARTIFACT_ROOT="${RUN_ROOT}/joint-update"
LOG_PATH="${RUN_ROOT}/run.log"
MEMORY_SAMPLES="${RUN_ROOT}/gpu-memory.csv"
MEMORY_BREACH="${RUN_ROOT}/gpu-memory-breach.txt"
WATCHDOG_TARGET="${RUN_ROOT}/watchdog-target-pgid.txt"
mkdir -m 700 -p "${RUN_ROOT}"
touch "${LOG_PATH}" "${MEMORY_SAMPLES}"
chmod 600 "${LOG_PATH}" "${MEMORY_SAMPLES}"

GPU_MEMORY_MONITOR_PID=""
GPU_DMON_PID=""
SECRET_REDACTOR_PID=""
RUN_COMMAND_PGID=""
RUN_FINALIZED=0
stop_monitors() {
  if [[ -n "${GPU_MEMORY_MONITOR_PID}" ]]; then
    kill "${GPU_MEMORY_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MEMORY_MONITOR_PID}" >/dev/null 2>&1 || true
    GPU_MEMORY_MONITOR_PID=""
  fi
  if [[ -n "${GPU_DMON_PID}" ]]; then
    kill "${GPU_DMON_PID}" >/dev/null 2>&1 || true
    wait "${GPU_DMON_PID}" >/dev/null 2>&1 || true
    GPU_DMON_PID=""
  fi
}
stop_run_process_group() {
  local run_pids
  run_pids=""
  if [[ -n "${RUN_COMMAND_PGID}" ]]; then
    run_pids="$(
      ps -eo pid=,sid= \
        | awk -v sid="${RUN_COMMAND_PGID}" '$2 == sid {print $1}' \
        | tr '\n' ' '
    )"
  fi
  if [[ -n "${run_pids// /}" ]]; then
    kill -TERM ${run_pids} >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
      run_pids="$(
        ps -eo pid=,sid= \
          | awk -v sid="${RUN_COMMAND_PGID}" '$2 == sid {print $1}' \
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
  if [[ ! -d "${RUN_ROOT}" ]]; then
    return 0
  fi
  if ! JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
      --verify-absent "${RUN_ROOT}" >/dev/null; then
    echo "Runtime admin API key redaction or absence verification failed" >&2
    return 1
  fi
  return 0
}
audit_memory_if_possible() {
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
    --run-kind m0-live-joint-v1 \
    --project-commit "${PROJECT_COMMIT}" \
    >> "${LOG_PATH}"
}
assert_no_run_processes() {
  local processes pid process_sid
  if ! processes="$(read_gpu_processes)"; then
    echo "Cannot verify M0 GPU process cleanup" >&2
    return 1
  fi
  while IFS=, read -r pid _; do
    pid="${pid// /}"
    if [[ -z "${pid}" ]] || [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
      continue
    fi
    process_sid="$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${RUN_COMMAND_PGID}" ]] \
      && [[ "${process_sid}" == "${RUN_COMMAND_PGID}" ]]; then
      echo "M0 cleanup left a project GPU compute process: pid=${pid}" >&2
      return 1
    fi
  done <<< "${processes}"
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_run_process_group
  if [[ -n "${SECRET_REDACTOR_PID}" ]]; then
    kill "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
    wait "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
    SECRET_REDACTOR_PID=""
  fi
  stop_monitors
  if [[ "${RUN_FINALIZED}" != 1 ]]; then
    audit_memory_if_possible || status=4
    redact_and_check_secret || status=4
    assert_no_run_processes || status=4
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

RUN_ADMIN_API_KEY="$(
  "${AREAL_VENV}/bin/python" -c \
    'import secrets; print("jph-m0-" + secrets.token_urlsafe(48))'
)"
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
    --watch-seconds 7200 "${RUN_ROOT}" >/dev/null 2>&1 &
SECRET_REDACTOR_PID="$!"
nvidia-smi dmon -i "${GPU_ID}" -s pucm -d 1 -o TD \
  > "${RUN_ROOT}/gpu-dmon.log" 2>&1 &
GPU_DMON_PID="$!"

echo "project=${PROJECT_COMMIT} AReaL=${ACTUAL_AREAL_COMMIT} physical_gpu=${GPU_ID} model_revision=${MODEL_REVISION}" | tee -a "${LOG_PATH}"
echo "gpu_uuid=${GPU_UUID} gpu_name=${GPU_NAME} driver=${GPU_DRIVER_VERSION} preflight_used=${GPU_MEMORY_USED_PREFLIGHT}MiB preflight_free=${GPU_MEMORY_FREE_PREFLIGHT}MiB" | tee -a "${LOG_PATH}"
echo "source_rollout=${ROLLOUT_RUN_ROOT} run_root=${RUN_ROOT}" | tee -a "${LOG_PATH}"

require_idle_gpu "immediately-before-python"
GPU_MEMORY_USED_AT_LAUNCH="${GPU_MEMORY_USED_SNAPSHOT}"
GPU_MEMORY_FREE_AT_LAUNCH="${GPU_MEMORY_FREE_SNAPSHOT}"
if ! "${AREAL_VENV}/bin/python" - "${M0_MASTER_PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", port))
PY
then
  echo "M0 MASTER_PORT ${M0_MASTER_PORT} is unavailable" >&2
  exit 3
fi
printf '%s,%s,%s\n' \
  "$(date +%s)" \
  "${GPU_MEMORY_USED_AT_LAUNCH}" \
  "${GPU_MEMORY_FREE_AT_LAUNCH}" >> "${MEMORY_SAMPLES}"

rm -f "${MEMORY_BREACH}" "${WATCHDOG_TARGET}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
WORLD_SIZE=1 \
RANK=0 \
LOCAL_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT="${M0_MASTER_PORT}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_PHYSICAL_GPU_ID="${GPU_ID}" \
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  setsid "${AREAL_VENV}/bin/python" -m jphrl.experiments.m0_live_joint \
    --runner-admission-dir "${RUNNER_ADMISSION_DIR}" \
    --selection-root "${SELECTION_ROOT}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --model-path "${MODEL_SNAPSHOT}" \
    --areal-root "${AREAL_REPO}" \
    --project-commit "${PROJECT_COMMIT}" \
    --transaction-id "${RUN_ID}" \
    --macro-step 0 \
    --physical-gpu-id "${GPU_ID}" \
    --experiment-name jph-m0-live-joint \
    --trial-name "${RUN_ID}" \
    > >(tee -a "${LOG_PATH}") 2>&1 &
RUN_COMMAND_PGID="$!"
printf '%s\n' "${RUN_COMMAND_PGID}" > "${WATCHDOG_TARGET}"
chmod 600 "${WATCHDOG_TARGET}"
(
  while true; do
    IFS=, read -r sample_used sample_free < <(read_gpu_memory)
    if [[ "${sample_used}" =~ ^[0-9]+$ ]] \
      && [[ "${sample_free}" =~ ^[0-9]+$ ]]; then
      printf '%s,%s,%s\n' "$(date +%s)" "${sample_used}" "${sample_free}" \
        >> "${MEMORY_SAMPLES}"
      if ((sample_used - GPU_MEMORY_USED_AT_LAUNCH > MAX_NEW_GPU_MEMORY_MIB)); then
        printf 'used=%s baseline=%s delta=%s limit=%s\n' \
          "${sample_used}" \
          "${GPU_MEMORY_USED_AT_LAUNCH}" \
          "$((sample_used - GPU_MEMORY_USED_AT_LAUNCH))" \
          "${MAX_NEW_GPU_MEMORY_MIB}" > "${MEMORY_BREACH}"
        chmod 600 "${MEMORY_BREACH}"
        target_pgid="$(tr -d ' ' < "${WATCHDOG_TARGET}" 2>/dev/null || true)"
        if [[ "${target_pgid}" =~ ^[0-9]+$ ]]; then
          kill -TERM -- "-${target_pgid}" >/dev/null 2>&1 || true
        fi
        exit 4
      fi
    fi
    sleep 1
  done
) &
GPU_MEMORY_MONITOR_PID="$!"
if ! wait "${RUN_COMMAND_PGID}"; then
  if [[ -f "${MEMORY_BREACH}" ]]; then
    echo "M0 was stopped by the 26 GiB GPU memory watchdog" >&2
  else
    echo "M0 live joint Python process failed" >&2
  fi
  exit 4
fi
rm -f "${WATCHDOG_TARGET}"

stop_monitors
audit_memory_if_possible

if [[ -n "${SECRET_REDACTOR_PID}" ]]; then
  kill "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  wait "${SECRET_REDACTOR_PID}" >/dev/null 2>&1 || true
  SECRET_REDACTOR_PID=""
fi
redact_and_check_secret

"${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/verify_m0_live_joint.py" \
  --run-root "${RUN_ROOT}" \
  --expected-project-commit "${PROJECT_COMMIT}" \
  --expected-areal-commit "${EXPECTED_AREAL_COMMIT}" \
  --output "${RUN_ROOT}/verification.json" \
  >> "${LOG_PATH}"

assert_no_run_processes
for path in \
  "${ARTIFACT_ROOT}/m0-summary.json" \
  "${ARTIFACT_ROOT}/production-worker-cleanup.json" \
  "${RUN_ROOT}/gpu-memory-audit.json" \
  "${RUN_ROOT}/verification.json"; do
  if [[ ! -f "${path}" ]] || [[ -L "${path}" ]]; then
    echo "M0 output is missing or unsafe: ${path}" >&2
    exit 4
  fi
done
readarray -t PRODUCTION_ATTESTATIONS < <(
  find "${ARTIFACT_ROOT}/release-store/activation-attestations" \
    -maxdepth 1 -type f -name 'activation-*.json' -print
)
if [[ "${#PRODUCTION_ATTESTATIONS[@]}" -ne 1 ]]; then
  echo "M0 must produce exactly one production activation attestation" >&2
  exit 4
fi

RUN_FINALIZED=1
echo "M0 live joint experiment complete: ${RUN_ROOT}" | tee -a "${LOG_PATH}"
