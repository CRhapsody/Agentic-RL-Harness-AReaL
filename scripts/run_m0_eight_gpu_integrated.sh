#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "formal eight-GPU integrated M0 must run inside tmux" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"
umask 077

readonly AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
readonly AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
readonly CUDA_TOOLKIT="/usr/local/cuda-12.6"
readonly MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/qwen2.5-1.5b-snapshot.json"
readonly DATASET_REPORT="${JPH_ROOT}/artifacts/bootstrap/gsm8k-snapshot.json"
readonly PRE_BATCH_PATCH="${JPH_PROJECT_DIR}/patches/areal-v2.0.0-data-proxy-pre-batch-hook.patch"
readonly EXPECTED_AREAL_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
readonly EXPECTED_PROJECT_COMMIT="${JPH_PROJECT_COMMIT:-}"

# AReaL opens one RPC/HTTP connection per worker and inference request.  Raise
# the soft limit to the account's hard limit before LocalScheduler exists, and
# fail closed when the host cannot provide the audited minimum.
NOFILE_HARD_LIMIT="$(ulimit -Hn)"
if ! ulimit -Sn "${NOFILE_HARD_LIMIT}"; then
  echo "cannot raise nofile soft limit to hard limit" >&2
  exit 2
fi
NOFILE_SOFT_LIMIT="$(ulimit -Sn)"
if [[ "${NOFILE_SOFT_LIMIT}" != "unlimited" ]] \
  && { [[ ! "${NOFILE_SOFT_LIMIT}" =~ ^[0-9]+$ ]] \
    || ((NOFILE_SOFT_LIMIT < 65536)); }; then
  echo "formal integrated M0 requires ulimit -n >= 65536" >&2
  exit 2
fi

if [[ ! "${EXPECTED_PROJECT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "JPH_PROJECT_COMMIT must be the exact formal project commit" >&2
  exit 2
fi
for required_path in \
  "${JPH_PROJECT_DIR}/.git" \
  "${AREAL_REPO}/.git" \
  "${AREAL_VENV}/bin/python" \
  "${AREAL_VENV}/bin/python3" \
  "${CUDA_TOOLKIT}/bin/nvcc" \
  "${MODEL_REPORT}" \
  "${DATASET_REPORT}" \
  "${PRE_BATCH_PATCH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "missing formal integrated dependency: ${required_path}" >&2
    exit 2
  fi
done

# Pinned AReaL's SGLang command builder deliberately emits ``python3``.
# LocalScheduler and its guards inherit PATH, so keep that literal command on
# the same Python 3.12 virtual environment as the controller instead of the
# host's Python 3.10 interpreter.
export CUDA_HOME="${CUDA_TOOLKIT}"
export CUDA_PATH="${CUDA_TOOLKIT}"
export PATH="${AREAL_VENV}/bin:${CUDA_TOOLKIT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLKIT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ "$(command -v python3)" != "${AREAL_VENV}/bin/python3" ]]; then
  echo "formal integrated child python3 does not resolve to the AReaL venv" >&2
  exit 2
fi
if [[ "$(command -v nvcc)" != "${CUDA_TOOLKIT}/bin/nvcc" ]] \
  || ! nvcc --version | grep -Fq "release 12.6"; then
  echo "formal integrated SGLang JIT requires the pinned CUDA 12.6 compiler" >&2
  exit 2
fi

PROJECT_COMMIT="$(git -C "${JPH_PROJECT_DIR}" rev-parse HEAD)"
AREAL_COMMIT="$(git -C "${AREAL_REPO}" rev-parse HEAD)"
if [[ "${PROJECT_COMMIT}" != "${EXPECTED_PROJECT_COMMIT}" ]]; then
  echo "formal project commit mismatch" >&2
  exit 2
fi
if [[ "${AREAL_COMMIT}" != "${EXPECTED_AREAL_COMMIT}" ]]; then
  echo "pinned AReaL commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${JPH_PROJECT_DIR}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Project worktree must be clean before formal integrated M0" >&2
  exit 2
fi
if [[ -n "$(git -C "${AREAL_REPO}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Pinned AReaL worktree must be clean before formal integrated M0" >&2
  exit 2
fi

mkdir -p -m 700 "${JPH_ROOT}/runtime/locks"
exec 9> "${JPH_ROOT}/runtime/locks/m0-eight-gpu-integrated.lock"
if ! flock -n 9; then
  echo "another formal integrated M0 owns the orchestration lock" >&2
  exit 3
fi
declare -A GPU_LOCK_FDS=()
for gpu_id in 0 1 2 3 4 5 6 7; do
  gpu_lock_fd=""
  exec {gpu_lock_fd}> "${JPH_ROOT}/runtime/locks/gpu-${gpu_id}.lock"
  if ! flock -n "${gpu_lock_fd}"; then
    echo "GPU ${gpu_id} is reserved; formal integrated M0 remains fail closed" >&2
    exit 3
  fi
  GPU_LOCK_FDS[${gpu_id}]="${gpu_lock_fd}"
done

read_gpu_memory() {
  local gpu_id="$1"
  nvidia-smi -i "${gpu_id}" \
    --query-gpu=memory.used,memory.free \
    --format=csv,noheader,nounits | tr -d ' '
}

read_gpu_processes() {
  local gpu_id="$1"
  nvidia-smi -i "${gpu_id}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
}

snapshot_all_eight_gpus() {
  local stage="$1"
  local gpu_id used_mib free_mib process_line process_pid process_name process_memory
  local process_sid process_uid process_user
  for gpu_id in 0 1 2 3 4 5 6 7; do
    IFS=, read -r used_mib free_mib < <(read_gpu_memory "${gpu_id}")
    if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] \
      || [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
      echo "cannot read GPU ${gpu_id} memory at ${stage}" >&2
      exit 3
    fi
    echo "${stage},$(date +%s),${gpu_id},${used_mib},${free_mib}" >> "${MEMORY_SAMPLES}"
    while IFS= read -r process_line; do
      [[ -z "${process_line}" ]] && continue
      IFS=, read -r process_pid process_name process_memory <<< "${process_line}"
      process_pid="${process_pid// /}"
      process_memory="${process_memory// /}"
      if [[ ! "${process_pid}" =~ ^[0-9]+$ ]] || ((process_pid <= 1)); then
        echo "cannot parse GPU ${gpu_id} process at ${stage}" >&2
        exit 3
      fi
      read -r process_sid process_uid process_user < <(
        ps -o sid=,uid=,user= -p "${process_pid}" 2>/dev/null || true
      )
      echo "${stage},$(date +%s),${gpu_id},${process_pid},${process_sid:-unknown},${process_uid:-unknown},${process_user:-unknown},${process_name//,/ },${process_memory:-unknown}" \
        >> "${PROCESS_SAMPLES}"
    done < <(read_gpu_processes "${gpu_id}")
  done
}

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NONCE="$("${AREAL_VENV}/bin/python" -c 'import secrets; print(secrets.token_hex(16))')"
RUN_ID="${RUN_STAMP}-m0-eight-gpu-integrated-${RUN_NONCE}"
RUN_ROOT="${JPH_ROOT}/artifacts/m0-eight-gpu-integrated/${RUN_ID}"
LOG_PATH="${JPH_ROOT}/logs/m0-eight-gpu-integrated-${RUN_ID}.log"
MONITOR_ROOT="${JPH_ROOT}/tmp/${RUN_ID}-monitor"
AREAL_CHILD_OVERLAY="${JPH_ROOT}/runtime/areal-overlays/${RUN_ID}"
MEMORY_SAMPLES="${MONITOR_ROOT}/gpu-memory.csv"
PROCESS_SAMPLES="${MONITOR_ROOT}/gpu-processes.csv"
WATCHDOG_ERROR="${MONITOR_ROOT}/watchdog-observation-error.txt"
RUN_ADMIN_API_KEY="$("${AREAL_VENV}/bin/python" -c 'import secrets; print(secrets.token_urlsafe(48))')"
export JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "formal integrated run root must be new" >&2
  exit 2
fi
mkdir -p -m 700 "${MONITOR_ROOT}" "$(dirname "${LOG_PATH}")"
touch "${LOG_PATH}" "${MEMORY_SAMPLES}" "${PROCESS_SAMPLES}"
chmod 600 "${LOG_PATH}" "${MEMORY_SAMPLES}" "${PROCESS_SAMPLES}"

# Keep the pinned checkout pristine while deploying the reviewed two-file
# DataProxy hook to run-scoped child processes.  The controller imports from
# the clean checkout before the adapter prepends this overlay for new Guards.
AREAL_OVERLAY_PARENT="$(dirname "${AREAL_CHILD_OVERLAY}")"
mkdir -p -m 700 "${AREAL_OVERLAY_PARENT}"
chmod 700 "${AREAL_OVERLAY_PARENT}"
if [[ ! -d "${AREAL_OVERLAY_PARENT}" || -L "${AREAL_OVERLAY_PARENT}" ]] ||
   [[ "$(stat -c '%u:%a' "${AREAL_OVERLAY_PARENT}")" != "$(id -u):700" ]]; then
  echo "formal AReaL overlay parent must be an owned private directory" >&2
  exit 2
fi
if ! mkdir -m 700 "${AREAL_CHILD_OVERLAY}"; then
  echo "formal AReaL child overlay must be atomically new" >&2
  exit 2
fi
git -C "${AREAL_REPO}" archive "${EXPECTED_AREAL_COMMIT}" areal |
  tar -x -C "${AREAL_CHILD_OVERLAY}"
patch --batch --forward --fuzz=0 -s -d "${AREAL_CHILD_OVERLAY}" -p1 \
  < "${PRE_BATCH_PATCH}"
JPH_AREAL_PRE_BATCH_PATCH_SHA256="$(sha256sum "${PRE_BATCH_PATCH}" | awk '{print $1}')"
JPH_AREAL_PATCHED_APP_SHA256="$(sha256sum \
  "${AREAL_CHILD_OVERLAY}/areal/v2/inference_service/data_proxy/app.py" | awk '{print $1}')"
JPH_AREAL_PATCHED_MAIN_SHA256="$(sha256sum \
  "${AREAL_CHILD_OVERLAY}/areal/v2/inference_service/data_proxy/__main__.py" | awk '{print $1}')"
export JPH_AREAL_CHILD_OVERLAY="${AREAL_CHILD_OVERLAY}"
export JPH_AREAL_PRE_BATCH_PATCH_SHA256
"${AREAL_VENV}/bin/python" - "${AREAL_CHILD_OVERLAY}/jph-overlay-manifest.json" \
  "${PROJECT_COMMIT}" "${JPH_AREAL_PRE_BATCH_PATCH_SHA256}" \
  "${JPH_AREAL_PATCHED_APP_SHA256}" "${JPH_AREAL_PATCHED_MAIN_SHA256}" <<'PY'
import json
import os
import sys

path, project_commit, patch_sha256, patched_app_sha256, patched_main_sha256 = sys.argv[1:]
record = {
    "schema_version": "jph.areal-child-overlay.v1",
    "areal_base_commit": "fee938eada49208a5aabdbc1095730a13076a349",
    "hook_import_path": (
        "jphrl.trajectory.rlvr_online_binding."
        "pre_batch_finalize_rlvr_v2_agent_admission"
    ),
    "patch_sha256": patch_sha256,
    "patched_files": {
        "areal/v2/inference_service/data_proxy/__main__.py": patched_main_sha256,
        "areal/v2/inference_service/data_proxy/app.py": patched_app_sha256,
    },
    "project_commit": project_commit,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(record, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY

snapshot_all_eight_gpus "preflight"

JOB_PID=""
JOB_SID=""
JOB_START_TIME=""
WATCHDOG_PID=""
FINALIZED=0

process_start_time() {
  local pid="$1"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ((pid <= 1)) || [[ ! -r "/proc/${pid}/stat" ]]; then
    return 1
  fi
  awk '{print $22}' "/proc/${pid}/stat"
}

session_pids() {
  local exact_sid="$1"
  if [[ ! "${exact_sid}" =~ ^[0-9]+$ ]] || ((exact_sid <= 1)); then
    return 1
  fi
  ps -eo pid=,sid= | awk -v sid="${exact_sid}" '$2 == sid && $1 > 1 {print $1}'
}

stop_exact_job_session() {
  local observed_start pid
  local -a pids=()
  if [[ -z "${JOB_PID}" || -z "${JOB_SID}" || -z "${JOB_START_TIME}" ]]; then
    return 0
  fi
  observed_start="$(process_start_time "${JOB_PID}" 2>/dev/null || true)"
  if [[ -n "${observed_start}" && "${observed_start}" != "${JOB_START_TIME}" ]]; then
    echo "refusing cleanup: formal job PID start time changed" >&2
    return 1
  fi
  mapfile -t pids < <(session_pids "${JOB_SID}")
  if ((${#pids[@]} > 0)); then
    kill -TERM -- "${pids[@]}" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
      sleep 1
      mapfile -t pids < <(session_pids "${JOB_SID}")
      ((${#pids[@]} == 0)) && break
    done
    if ((${#pids[@]} > 0)); then
      kill -KILL -- "${pids[@]}" >/dev/null 2>&1 || true
    fi
  fi
}

stop_watchdog() {
  if [[ -n "${WATCHDOG_PID}" ]]; then
    kill "${WATCHDOG_PID}" >/dev/null 2>&1 || true
    wait "${WATCHDOG_PID}" >/dev/null 2>&1 || true
    WATCHDOG_PID=""
  fi
}

redact_and_verify_runtime_secret() {
  local -a targets=("${LOG_PATH}")
  [[ -d "${RUN_ROOT}" ]] && targets+=("${RUN_ROOT}")
  JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
      --verify-absent "${targets[@]}" >/dev/null
}

copy_monitor_artifacts() {
  if [[ -d "${RUN_ROOT}" ]]; then
    mkdir -p -m 700 "${RUN_ROOT}/gpu-memory"
    if [[ -f "${MEMORY_SAMPLES}" && ! -e "${RUN_ROOT}/gpu-memory/gpu-memory.csv" ]]; then
      cp -p "${MEMORY_SAMPLES}" "${RUN_ROOT}/gpu-memory/gpu-memory.csv"
      chmod 600 "${RUN_ROOT}/gpu-memory/gpu-memory.csv"
    fi
    if [[ -f "${PROCESS_SAMPLES}" && ! -e "${RUN_ROOT}/gpu-memory/gpu-processes.csv" ]]; then
      cp -p "${PROCESS_SAMPLES}" "${RUN_ROOT}/gpu-memory/gpu-processes.csv"
      chmod 600 "${RUN_ROOT}/gpu-memory/gpu-processes.csv"
    fi
    if [[ -f "${WATCHDOG_ERROR}" && ! -e "${RUN_ROOT}/gpu-memory/watchdog-observation-error.txt" ]]; then
      cp -p "${WATCHDOG_ERROR}" "${RUN_ROOT}/gpu-memory/watchdog-observation-error.txt"
      chmod 600 "${RUN_ROOT}/gpu-memory/watchdog-observation-error.txt"
    fi
  fi
}

finalize() {
  local status=$?
  if ((FINALIZED != 0)); then
    return "${status}"
  fi
  FINALIZED=1
  stop_watchdog || status=1
  stop_exact_job_session || status=1
  copy_monitor_artifacts || status=1
  redact_and_verify_runtime_secret || status=1
  unset JPH_AREAL_ADMIN_API_KEY RUN_ADMIN_API_KEY
  return "${status}"
}
trap finalize EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

watch_gpu_memory_every_second() {
  local exact_sid="$1"
  local gpu_id timestamp used_mib free_mib process_line process_pid process_name
  local process_memory process_sid process_uid process_user
  while session_pids "${exact_sid}" | head -n 1 | grep -q .; do
    timestamp="$(date +%s)"
    for gpu_id in 0 1 2 3 4 5 6 7; do
      IFS=, read -r used_mib free_mib < <(read_gpu_memory "${gpu_id}")
      if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
        echo "unreadable GPU ${gpu_id} sample" > "${WATCHDOG_ERROR}"
        stop_exact_job_session
        return 1
      fi
      echo "runtime,${timestamp},${gpu_id},${used_mib},${free_mib}" >> "${MEMORY_SAMPLES}"
      while IFS= read -r process_line; do
        [[ -z "${process_line}" ]] && continue
        IFS=, read -r process_pid process_name process_memory <<< "${process_line}"
        process_pid="${process_pid// /}"
        if [[ ! "${process_pid}" =~ ^[0-9]+$ ]] || ((process_pid <= 1)); then
          echo "GPU ${gpu_id} returned an invalid process PID" > "${WATCHDOG_ERROR}"
          stop_exact_job_session
          return 1
        fi
        read -r process_sid process_uid process_user < <(
          ps -o sid=,uid=,user= -p "${process_pid}" 2>/dev/null || true
        )
        echo "runtime,${timestamp},${gpu_id},${process_pid},${process_sid:-unknown},${process_uid:-unknown},${process_user:-unknown},${process_name//,/ },${process_memory// /}" \
          >> "${PROCESS_SAMPLES}"
      done < <(read_gpu_processes "${gpu_id}")
    done
    sleep 1
  done
}

echo "formal-integrated project=${PROJECT_COMMIT} AReaL=${AREAL_COMMIT} GPUs=0,1,2,3,4,5,6,7 run_root=${RUN_ROOT}" \
  >> "${LOG_PATH}"

snapshot_all_eight_gpus "immediately-before-scheduler"

setsid "${AREAL_VENV}/bin/python" \
  "${SCRIPT_DIR}/run_m0_eight_gpu_integrated.py" \
  --entry-mode execute \
  --run-root "${RUN_ROOT}" \
  --expected-project-commit "${PROJECT_COMMIT}" \
  --model-report "${MODEL_REPORT}" \
  --dataset-report "${DATASET_REPORT}" \
  >> "${LOG_PATH}" 2>&1 &
JOB_PID=$!
JOB_SID="$(ps -o sid= -p "${JOB_PID}" | tr -d ' ')"
JOB_START_TIME="$(process_start_time "${JOB_PID}")"
if [[ "${JOB_SID}" != "${JOB_PID}" ]] || [[ -z "${JOB_START_TIME}" ]]; then
  echo "setsid did not create the exact formal job SID" >&2
  exit 3
fi

watch_gpu_memory_every_second "${JOB_SID}" &
WATCHDOG_PID=$!

set +e
wait "${JOB_PID}"
JOB_STATUS=$?
set -e
stop_watchdog
copy_monitor_artifacts
if [[ -s "${WATCHDOG_ERROR}" ]]; then
  echo "formal integrated GPU observation failed" >&2
  exit 3
fi
if ((JOB_STATUS != 0)); then
  echo "formal integrated entry failed closed; inspect ${LOG_PATH}" >&2
  exit "${JOB_STATUS}"
fi

echo "formal integrated M0 completed: ${RUN_ROOT}"
