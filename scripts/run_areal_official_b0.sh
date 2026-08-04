#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/run_areal_official_b0.sh" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "Official B0 must run inside tmux" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"
umask 077

readonly -a GPU_IDS=(0 1 2 3 4 5 6 7)
readonly AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
readonly AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
readonly EXPECTED_AREAL_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
readonly MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/qwen2.5-1.5b-snapshot.json"
readonly DATASET_REPORT="${JPH_ROOT}/artifacts/bootstrap/gsm8k-snapshot.json"
readonly CUDA_TOOLKIT_ROOT="${JPH_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.6}"
readonly MIN_FREE_MEMORY_MIB="${JPH_B0_MIN_FREE_MEMORY_MIB:-71680}"
readonly MAX_USED_MEMORY_MIB="${JPH_B0_MAX_USED_MEMORY_MIB:-10240}"
readonly SOFT_MAX_NEW_GPU_MEMORY_MIB=26624
readonly HARD_MAX_NEW_GPU_MEMORY_MIB=30720

if [[ ! "${MIN_FREE_MEMORY_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MIN_FREE_MEMORY_MIB must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MAX_USED_MEMORY_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MAX_USED_MEMORY_MIB must be a positive integer" >&2
  exit 2
fi
if ((MIN_FREE_MEMORY_MIB < 71680)); then
  echo "JPH_B0_MIN_FREE_MEMORY_MIB cannot relax the 71680 MiB floor" >&2
  exit 2
fi
if ((MAX_USED_MEMORY_MIB > 10240)); then
  echo "JPH_B0_MAX_USED_MEMORY_MIB cannot relax the 10240 MiB ceiling" >&2
  exit 2
fi
if ((SOFT_MAX_NEW_GPU_MEMORY_MIB > HARD_MAX_NEW_GPU_MEMORY_MIB)); then
  echo "B0 soft GPU-memory cap exceeds the immutable hard cap" >&2
  exit 2
fi

for path in \
  "${JPH_PROJECT_DIR}/.git" \
  "${AREAL_REPO}/.git" \
  "${AREAL_VENV}/bin/python" \
  "${CUDA_TOOLKIT_ROOT}/bin/nvcc" \
  "${MODEL_REPORT}" \
  "${DATASET_REPORT}"; do
  if [[ ! -e "${path}" ]] || [[ -L "${path}" ]]; then
    echo "Missing or unsafe official B0 dependency: ${path}" >&2
    exit 2
  fi
done

PROJECT_COMMIT="$(git -C "${JPH_PROJECT_DIR}" rev-parse HEAD)"
ACTUAL_AREAL_COMMIT="$(git -C "${AREAL_REPO}" rev-parse HEAD)"
if [[ -n "${JPH_PROJECT_COMMIT:-}" ]] \
  && [[ "${JPH_PROJECT_COMMIT}" != "${PROJECT_COMMIT}" ]]; then
  echo "Project commit mismatch: expected=${JPH_PROJECT_COMMIT} actual=${PROJECT_COMMIT}" >&2
  exit 2
fi
if [[ "${ACTUAL_AREAL_COMMIT}" != "${EXPECTED_AREAL_COMMIT}" ]]; then
  echo "AReaL commit mismatch: expected=${EXPECTED_AREAL_COMMIT} actual=${ACTUAL_AREAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${JPH_PROJECT_DIR}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Project worktree must be clean before official B0" >&2
  exit 2
fi
if [[ -n "$(git -C "${AREAL_REPO}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Pinned AReaL worktree must be clean before official B0" >&2
  exit 2
fi
if ! grep -q "total_train_steps" "${AREAL_REPO}/areal/api/cli_args.py"; then
  echo "Pinned AReaL source does not expose total_train_steps; refusing an unbounded B0" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${AREAL_VENV}/bin:${CUDA_HOME}/bin:${PATH}"
NVCC_PREFLIGHT_OBJECT="$(mktemp --suffix=.o "${JPH_ROOT}/tmp/nvcc-b0-preflight.XXXXXXXX")"
if ! "${CUDACXX}" -std=c++20 -x cu -c /dev/null -o "${NVCC_PREFLIGHT_OBJECT}"; then
  rm -f -- "${NVCC_PREFLIGHT_OBJECT}"
  exit 2
fi
rm -f -- "${NVCC_PREFLIGHT_OBJECT}"

MODEL_SNAPSHOT="$(
  "${AREAL_VENV}/bin/python" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["snapshot_path"])' \
    "${MODEL_REPORT}"
)"
DATASET_SNAPSHOT="$(
  "${AREAL_VENV}/bin/python" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["snapshot_path"])' \
    "${DATASET_REPORT}"
)"
for snapshot in "${MODEL_SNAPSHOT}" "${DATASET_SNAPSHOT}"; do
  if [[ ! -d "${snapshot}" ]] || [[ -L "${snapshot}" ]]; then
    echo "Pinned snapshot directory is missing or unsafe: ${snapshot}" >&2
    exit 2
  fi
  snapshot="$(realpath "${snapshot}")"
  case "${snapshot}" in
    "${JPH_ROOT}"/*) ;;
    *)
      echo "Pinned snapshot escapes ${JPH_ROOT}: ${snapshot}" >&2
      exit 2
      ;;
  esac
done
MODEL_SNAPSHOT="$(realpath "${MODEL_SNAPSHOT}")"
DATASET_SNAPSHOT="$(realpath "${DATASET_SNAPSHOT}")"

GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" != 8 ]]; then
  echo "Official B0 requires exactly 8 visible GPUs; found ${GPU_COUNT}" >&2
  exit 3
fi

# One coordinator owns all eight leases.  No per-GPU replica is launched.
mkdir -p -m 700 "${JPH_ROOT}/runtime/locks"
exec 9> "${JPH_ROOT}/runtime/locks/areal-official-b0-8gpu.lock"
if ! flock -n 9; then
  echo "Another official eight-GPU B0 already holds the orchestration lock" >&2
  exit 3
fi
declare -A GPU_LOCK_FDS=()
for gpu_id in "${GPU_IDS[@]}"; do
  exec {gpu_lock_fd}> "${JPH_ROOT}/runtime/locks/gpu-${gpu_id}.lock"
  if ! flock -n "${gpu_lock_fd}"; then
    echo "GPU ${gpu_id} is reserved; official B0 was not launched" >&2
    exit 3
  fi
  GPU_LOCK_FDS[${gpu_id}]="${gpu_lock_fd}"
done

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NONCE="$("${AREAL_VENV}/bin/python" -c 'import secrets; print(secrets.token_hex(16))')"
RUN_ID="${RUN_STAMP}-official-b0-${RUN_NONCE}"
RUN_ROOT="${JPH_ROOT}/artifacts/areal-b0/${RUN_ID}"
LOG_PATH="${JPH_ROOT}/logs/areal-b0-${RUN_ID}.log"
NAME_RESOLVE_ROOT="${RUN_ROOT}/name-resolve"
AUDIT_PATH="${RUN_ROOT}/gpu-memory-audit.json"
MEMORY_BREACH="${RUN_ROOT}/gpu-memory-breach.txt"
RUN_STATUS_PATH="${RUN_ROOT}/run.status"
LOG_FIFO="${RUN_ROOT}/coordinator-output.fifo"
RUN_ADMIN_API_KEY="$(
  "${AREAL_VENV}/bin/python" -c \
    'import secrets; print("jph-b0-" + secrets.token_urlsafe(48))'
)"
mkdir -p -m 700 "${RUN_ROOT}" "${NAME_RESOLVE_ROOT}"
touch "${LOG_PATH}"
chmod 600 "${LOG_PATH}"

declare -A GPU_BASELINE_USED=()
declare -A GPU_BASELINE_FREE=()
JOB_PID=""
JOB_START_TIME=""
JOB_SESSION_ID=""
JOB_LEADER_REAPED=0
MEMORY_MONITOR_PID=""
GPU_DMON_PID=""
SECRET_REDACTOR_PID=""
LOG_TEE_PID=""
AUDIT_READY=0
AUDIT_WRITTEN=0
RUN_FAILURE_REASON=""
B0_ORCHESTRATOR_PID="$$"

process_start_time() {
  local pid="$1"
  local stat_line stat_tail
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] \
    || ! IFS= read -r stat_line < "/proc/${pid}/stat"; then
    return 1
  fi
  stat_tail="${stat_line##*) }"
  set -- ${stat_tail}
  if [[ $# -lt 20 ]] || [[ ! "${20}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf '%s\n' "${20}"
}

bind_job_identity() {
  local attempt actual_start actual_sid
  for attempt in $(seq 1 100); do
    actual_start="$(process_start_time "${JOB_PID}" 2>/dev/null || true)"
    actual_sid="$(ps -o sid= -p "${JOB_PID}" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${actual_start}" ]] && [[ "${actual_sid}" == "${JOB_PID}" ]]; then
      JOB_START_TIME="${actual_start}"
      JOB_SESSION_ID="${actual_sid}"
      return 0
    fi
    if ! kill -0 "${JOB_PID}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 0.05
  done
  return 1
}

stop_unbound_direct_child() {
  if [[ -n "${JOB_PID}" ]]; then
    # JOB_PID is an unreaped direct child of this shell, so the kernel cannot
    # reuse it for an external process before this wait completes.
    kill -TERM "${JOB_PID}" >/dev/null 2>&1 || true
    wait "${JOB_PID}" >/dev/null 2>&1 || true
  fi
  JOB_PID=""
  JOB_START_TIME=""
  JOB_SESSION_ID=""
  JOB_LEADER_REAPED=0
}

is_owned_job_leader() {
  local actual_start actual_sid
  if [[ -z "${JOB_PID}" || -z "${JOB_START_TIME}" || -z "${JOB_SESSION_ID}" ]]; then
    return 1
  fi
  actual_start="$(process_start_time "${JOB_PID}" 2>/dev/null || true)"
  actual_sid="$(ps -o sid= -p "${JOB_PID}" 2>/dev/null | tr -d ' ' || true)"
  [[ "${actual_start}" == "${JOB_START_TIME}" ]] \
    && [[ "${actual_sid}" == "${JOB_SESSION_ID}" ]] \
    && [[ "${JOB_SESSION_ID}" == "${JOB_PID}" ]]
}

stop_owned_job_session() {
  local attempt can_signal=0
  if is_owned_job_leader; then
    can_signal=1
  elif ((JOB_LEADER_REAPED != 0)) \
    && [[ -n "${JOB_SESSION_ID}" ]] \
    && pgrep -s "${JOB_SESSION_ID}" >/dev/null 2>&1; then
    # The leader identity was bound before launch and has just been reaped by
    # this shell.  Any remaining member of that still-live SID is a straggler
    # from the same owned session; an external process cannot join it.
    can_signal=1
  fi
  if ((can_signal == 0)); then
    return 0
  fi
  pkill -TERM -s "${JOB_SESSION_ID}" >/dev/null 2>&1 || true
  for attempt in $(seq 1 150); do
    if ! pgrep -s "${JOB_SESSION_ID}" >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
  if pgrep -s "${JOB_SESSION_ID}" >/dev/null 2>&1; then
    pkill -KILL -s "${JOB_SESSION_ID}" >/dev/null 2>&1 || true
  fi
  if ((JOB_LEADER_REAPED == 0)); then
    wait "${JOB_PID}" >/dev/null 2>&1 || true
  fi
  JOB_PID=""
  JOB_START_TIME=""
  JOB_SESSION_ID=""
  JOB_LEADER_REAPED=0
}

stop_background_process() {
  local variable_name="$1"
  local pid="${!variable_name:-}"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
    printf -v "${variable_name}" '%s' ""
  fi
}

close_inherited_b0_locks() {
  local gpu_id fd_to_close
  for gpu_id in "${GPU_IDS[@]}"; do
    fd_to_close="${GPU_LOCK_FDS[${gpu_id}]}"
    exec {fd_to_close}>&-
  done
  exec 9>&-
}

finish_log_tee() {
  local attempt
  if [[ -z "${LOG_TEE_PID}" ]]; then
    return 0
  fi
  for attempt in $(seq 1 50); do
    if ! kill -0 "${LOG_TEE_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "${LOG_TEE_PID}" >/dev/null 2>&1; then
    kill -TERM "${LOG_TEE_PID}" >/dev/null 2>&1 || true
  fi
  wait "${LOG_TEE_PID}" >/dev/null 2>&1 || true
  LOG_TEE_PID=""
}

read_gpu_state() {
  local gpu_id="$1"
  local stage="$2"
  local snapshot_path="$3"
  local memory_output memory_used memory_free processes
  if ! memory_output="$(
    nvidia-smi -i "${gpu_id}" \
      --query-gpu=memory.used,memory.free \
      --format=csv,noheader,nounits
  )"; then
    echo "Cannot read GPU ${gpu_id} memory at ${stage}" >&2
    return 1
  fi
  memory_output="${memory_output// /}"
  IFS=, read -r memory_used memory_free <<< "${memory_output}"
  if [[ ! "${memory_used}" =~ ^[0-9]+$ ]] \
    || [[ ! "${memory_free}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ${gpu_id} memory state at ${stage}" >&2
    return 1
  fi
  if ! processes="$(
    nvidia-smi -i "${gpu_id}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits
  )"; then
    echo "Cannot read GPU ${gpu_id} compute processes at ${stage}" >&2
    return 1
  fi
  printf 'stage=%s\ngpu_id=%s\nmemory_used_mib=%s\nmemory_free_mib=%s\ncompute_processes=%s\n' \
    "${stage}" "${gpu_id}" "${memory_used}" "${memory_free}" \
    "${processes:-none}" > "${snapshot_path}"
  chmod 600 "${snapshot_path}"
  if ((memory_used > MAX_USED_MEMORY_MIB || memory_free < MIN_FREE_MEMORY_MIB)); then
    echo "GPU ${gpu_id} lacks B0 headroom at ${stage}: used=${memory_used}MiB free=${memory_free}MiB" >&2
    return 1
  fi
  if [[ -n "${processes}" ]]; then
    echo "GPU ${gpu_id} has an existing compute process at ${stage}" >&2
    return 1
  fi
  if [[ "${stage}" == immediately-before-launch ]]; then
    GPU_BASELINE_USED[${gpu_id}]="${memory_used}"
    GPU_BASELINE_FREE[${gpu_id}]="${memory_free}"
  fi
}

run_all_gpu_gate() {
  local stage="$1"
  local gpu_id failed=0
  for gpu_id in "${GPU_IDS[@]}"; do
    if ! read_gpu_state \
      "${gpu_id}" "${stage}" "${RUN_ROOT}/gpu-${gpu_id}-${stage}.env"; then
      failed=1
    fi
  done
  if ((failed != 0)); then
    echo "At least one GPU failed the ${stage} gate; official B0 was not launched" >&2
    return 1
  fi
}

append_gpu_sample() {
  local gpu_id="$1"
  local output memory_used memory_free
  if ! output="$(
    nvidia-smi -i "${gpu_id}" \
      --query-gpu=memory.used,memory.free \
      --format=csv,noheader,nounits
  )"; then
    return 1
  fi
  output="${output// /}"
  IFS=, read -r memory_used memory_free <<< "${output}"
  if [[ ! "${memory_used}" =~ ^[0-9]+$ ]] \
    || [[ ! "${memory_free}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf '%s,%s,%s\n' "$(date -u +%s)" "${memory_used}" "${memory_free}" \
    >> "${RUN_ROOT}/gpu-${gpu_id}-memory.csv"
}

record_breach_once() {
  local message="$1"
  if [[ ! -e "${MEMORY_BREACH}" ]]; then
    printf '%s\n' "${message}" > "${MEMORY_BREACH}"
    chmod 600 "${MEMORY_BREACH}"
  fi
}

append_final_samples() {
  local gpu_id failed=0
  if ((AUDIT_READY == 0)); then
    return 0
  fi
  for gpu_id in "${GPU_IDS[@]}"; do
    if ! append_gpu_sample "${gpu_id}"; then
      record_breach_once "final-sample-failed gpu=${gpu_id}"
      failed=1
    fi
  done
  return "${failed}"
}

write_gpu_memory_audit() {
  local audit_status=0
  if ((AUDIT_READY == 0 || AUDIT_WRITTEN != 0)); then
    return 0
  fi
  if JPH_B0_RUN_FAILURE_REASON="${RUN_FAILURE_REASON}" \
    "${AREAL_VENV}/bin/python" - \
      "${RUN_ROOT}" "${AUDIT_PATH}" "${PROJECT_COMMIT}" \
      "${ACTUAL_AREAL_COMMIT}" "${SOFT_MAX_NEW_GPU_MEMORY_MIB}" \
      "${HARD_MAX_NEW_GPU_MEMORY_MIB}" \
      "${GPU_BASELINE_USED[0]}" "${GPU_BASELINE_USED[1]}" \
      "${GPU_BASELINE_USED[2]}" "${GPU_BASELINE_USED[3]}" \
      "${GPU_BASELINE_USED[4]}" "${GPU_BASELINE_USED[5]}" \
      "${GPU_BASELINE_USED[6]}" "${GPU_BASELINE_USED[7]}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    run_root_raw,
    output_raw,
    project_commit,
    areal_commit,
    soft_raw,
    hard_raw,
    *baseline_raw,
) = sys.argv[1:]
run_root = Path(run_root_raw).resolve(strict=True)
output = Path(output_raw)
soft_cap = int(soft_raw)
hard_cap = int(hard_raw)
baselines = [int(value) for value in baseline_raw]
failure_reason = os.environ.get("JPH_B0_RUN_FAILURE_REASON", "")
breach_path = run_root / "gpu-memory-breach.txt"
gpus = []
for gpu_id, baseline in enumerate(baselines):
    sample_path = run_root / f"gpu-{gpu_id}-memory.csv"
    samples = []
    for line_number, line in enumerate(
        sample_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split(",")
        if len(fields) != 3:
            raise ValueError(f"invalid GPU {gpu_id} sample at line {line_number}")
        timestamp, used_mib, free_mib = (int(field) for field in fields)
        if timestamp <= 0 or used_mib < 0 or free_mib < 0:
            raise ValueError(f"negative GPU {gpu_id} sample at line {line_number}")
        samples.append((timestamp, used_mib, free_mib))
    if not samples:
        raise ValueError(f"GPU {gpu_id} has no measured memory sample")
    _, peak_used, peak_free = max(samples, key=lambda item: item[1])
    delta = max(0, peak_used - baseline)
    gpu_passed = delta <= soft_cap and delta <= hard_cap
    gpus.append(
        {
            "physical_gpu_id": gpu_id,
            "baseline_used_mib": baseline,
            "peak_used_mib": peak_used,
            "peak_free_mib": peak_free,
            "peak_delta_mib": delta,
            "soft_cap_mib": soft_cap,
            "hard_cap_mib": hard_cap,
            "sample_count": len(samples),
            "passed": gpu_passed,
        }
    )
passed = (
    soft_cap <= hard_cap
    and not failure_reason
    and not breach_path.exists()
    and all(gpu["passed"] for gpu in gpus)
)
record = {
    "schema_version": "jph.areal-official-b0-gpu-memory-audit.v1",
    "run_kind": "areal-official-b0-v1",
    "project_commit": project_commit,
    "areal_commit": areal_commit,
    "soft_cap_mib": soft_cap,
    "hard_cap_mib": hard_cap,
    "passed": passed,
    "gpus": gpus,
}
canonical = json.dumps(
    record,
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
record["record_sha256"] = hashlib.sha256(canonical).hexdigest()
payload = json.dumps(
    record,
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
descriptor, temporary_raw = tempfile.mkstemp(
    prefix=f".{output.name}.", dir=output.parent
)
temporary = Path(temporary_raw)
try:
    os.chmod(temporary, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, output)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)
if not passed:
    raise SystemExit(4)
PY
  then
    audit_status=0
  else
    audit_status=$?
  fi
  AUDIT_WRITTEN=1
  return "${audit_status}"
}

redact_and_verify_secret() {
  JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
      --verify-absent "${RUN_ROOT}" "${LOG_PATH}" >/dev/null
}

cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  if ((status == 0)) && [[ -n "${RUN_FAILURE_REASON}" ]]; then
    status=4
  fi
  if ((status != 0)) && [[ -z "${RUN_FAILURE_REASON}" ]]; then
    RUN_FAILURE_REASON="launcher-exit-${status}"
  fi
  stop_owned_job_session
  stop_background_process MEMORY_MONITOR_PID
  stop_background_process GPU_DMON_PID
  stop_background_process SECRET_REDACTOR_PID
  finish_log_tee
  rm -f -- "${LOG_FIFO}" >/dev/null 2>&1 || true
  if ! append_final_samples; then
    status=4
    RUN_FAILURE_REASON="${RUN_FAILURE_REASON:-final-gpu-sample-failed}"
  fi
  if ! redact_and_verify_secret; then
    status=4
    RUN_FAILURE_REASON="${RUN_FAILURE_REASON:-secret-redaction-or-verification-failed}"
  fi
  if ! write_gpu_memory_audit; then
    status=4
  fi
  if ((status == 0)); then
    if ! "${AREAL_VENV}/bin/python" \
      "${SCRIPT_DIR}/verify_areal_official_b0.py" \
        --run-root "${RUN_ROOT}" \
        --run-log "${LOG_PATH}" \
        --expected-project-commit "${PROJECT_COMMIT}" \
        --expected-areal-commit "${ACTUAL_AREAL_COMMIT}" \
        --output "${RUN_ROOT}/verification.json" >/dev/null; then
      status=4
      RUN_FAILURE_REASON="official-b0-verification-failed"
    fi
  fi
  if ((status == 0)); then
    printf 'state=passed\nexit_code=0\nproject_commit=%s\nareal_commit=%s\nrun_root=%s\naudit_path=%s\n' \
      "${PROJECT_COMMIT}" "${ACTUAL_AREAL_COMMIT}" "${RUN_ROOT}" "${AUDIT_PATH}" \
      > "${RUN_STATUS_PATH}" || status=4
  else
    printf 'state=failed\nexit_code=%s\nreason=%s\nproject_commit=%s\nareal_commit=%s\nrun_root=%s\naudit_path=%s\n' \
      "${status}" "${RUN_FAILURE_REASON}" "${PROJECT_COMMIT}" \
      "${ACTUAL_AREAL_COMMIT}" "${RUN_ROOT}" "${AUDIT_PATH}" \
      > "${RUN_STATUS_PATH}" || true
  fi
  chmod 600 "${RUN_STATUS_PATH}" >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

run_all_gpu_gate preflight

AREAL_ARGS=(
  --config "${AREAL_REPO}/examples/math/gsm8k_grpo.yaml"
  scheduler.type=local
  experiment_name=jph-b0
  trial_name="${RUN_ID}"
  cluster.n_nodes=1
  cluster.n_gpus_per_node=8
  cluster.fileroot="${RUN_ROOT}"
  cluster.name_resolve.nfs_record_root="${NAME_RESOLVE_ROOT}"
  +total_train_steps=1
  actor.backend=fsdp:d4p1t1
  actor.path="${MODEL_SNAPSHOT}"
  actor.optimizer.lr=1.70e-5
  actor.optimizer.lr_scheduler_type=constant
  actor.optimizer.warmup_steps_proportion=0.0
  actor.kl_ctl=0.0
  tokenizer_path="${MODEL_SNAPSHOT}"
  ref.backend=fsdp:d4p1t1
  ref.path="${MODEL_SNAPSHOT}"
  ref.scheduling_strategy.type=colocation
  ref.scheduling_strategy.target=actor
  rollout.backend=sglang:d4p1t1
  rollout.max_concurrent_rollouts=8
  rollout.max_head_offpolicyness=0
  rollout.dump_to_file=true
  '+rollout.agent.admin_api_key=${oc.env:JPH_AREAL_ADMIN_API_KEY}'
  sglang.mem_fraction_static=0.29
  sglang.context_length=1024
  sglang.max_running_requests=2
  train_dataset.path="${DATASET_SNAPSHOT}"
  train_dataset.batch_size=8
  train_dataset.num_workers=2
  valid_dataset.path="${DATASET_SNAPSHOT}"
  valid_dataset.batch_size=8
  valid_dataset.num_workers=2
  gconfig.n_samples=2
  gconfig.max_new_tokens=256
  gconfig.max_tokens=512
  saver.fileroot="${RUN_ROOT}"
  saver.freq_steps=1
  saver.freq_epochs=null
  saver.freq_secs=null
  +saver.mode=sync
  recover.mode=disabled
)

(
  close_inherited_b0_locks
  JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
    exec "${AREAL_VENV}/bin/python" "${SCRIPT_DIR}/redact_runtime_admin_key.py" \
      --watch-seconds 86400 "${RUN_ROOT}" "${LOG_PATH}"
) >/dev/null 2>&1 &
SECRET_REDACTOR_PID="$!"

# Compose and validate all Hydra overrides without loading a tokenizer, model,
# CUDA context, or dataset.  load_expr_config also persists the resolved config
# under the external RUN_ROOT; the redactor watches that tree continuously.
cd "${AREAL_REPO}"
if ! CUDA_VISIBLE_DEVICES="" \
  JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  JPH_B0_PREFLIGHT_PATH="${RUN_ROOT}/hydra-preflight.json" \
  "${AREAL_VENV}/bin/python" - "${AREAL_ARGS[@]}" \
    > "${RUN_ROOT}/hydra-preflight.log" 2>&1 <<'PY'
import json
import math
import os
import sys
from pathlib import Path

from areal.api.cli_args import GRPOConfig, load_expr_config

config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
checks = {
    "cluster_n_gpus_per_node": config.cluster.n_gpus_per_node,
    "actor_backend": config.actor.backend,
    "rollout_backend": config.rollout.backend,
    "ref_backend": config.ref.backend,
    "ref_scheduling_type": config.ref.scheduling_strategy.type,
    "ref_scheduling_target": config.ref.scheduling_strategy.target,
    "sglang_mem_fraction_static": config.sglang.mem_fraction_static,
    "sglang_context_length": config.sglang.context_length,
    "sglang_max_running_requests": config.sglang.max_running_requests,
    "saver_freq_steps": config.saver.freq_steps,
    "saver_freq_epochs": config.saver.freq_epochs,
    "saver_freq_secs": config.saver.freq_secs,
    "saver_mode": config.saver.mode,
    "actor_kl_ctl": config.actor.kl_ctl,
    "actor_lr": config.actor.optimizer.lr,
    "actor_lr_scheduler_type": config.actor.optimizer.lr_scheduler_type,
    "warmup_steps_proportion": config.actor.optimizer.warmup_steps_proportion,
    "recover_mode": config.recover.mode,
    "total_train_steps": config.total_train_steps,
    "train_batch_size": config.train_dataset.batch_size,
    "valid_batch_size": config.valid_dataset.batch_size,
    "n_samples": config.gconfig.n_samples,
    "max_new_tokens": config.gconfig.max_new_tokens,
    "max_tokens": config.gconfig.max_tokens,
    "rollout_max_concurrent": config.rollout.max_concurrent_rollouts,
    "rollout_max_head_offpolicyness": config.rollout.max_head_offpolicyness,
}
assert checks["cluster_n_gpus_per_node"] == 8
assert checks["actor_backend"] == "fsdp:d4p1t1"
assert checks["rollout_backend"] == "sglang:d4p1t1"
assert checks["ref_backend"] == "fsdp:d4p1t1"
assert str(checks["ref_scheduling_type"]) == "colocation"
assert checks["ref_scheduling_target"] == "actor"
assert math.isclose(checks["sglang_mem_fraction_static"], 0.29)
assert checks["sglang_context_length"] == 1024
assert checks["sglang_max_running_requests"] == 2
assert checks["saver_freq_steps"] == 1
assert checks["saver_freq_epochs"] is None
assert checks["saver_freq_secs"] is None
assert checks["saver_mode"] == "sync"
assert math.isclose(checks["actor_kl_ctl"], 0.0)
assert math.isclose(checks["actor_lr"], 1.70e-5)
assert checks["actor_lr_scheduler_type"] == "constant"
assert math.isclose(checks["warmup_steps_proportion"], 0.0)
assert checks["recover_mode"] == "disabled"
assert checks["total_train_steps"] == 1
assert checks["train_batch_size"] == 8
assert checks["valid_batch_size"] == 8
assert checks["n_samples"] == 2
assert checks["max_new_tokens"] == 256
assert checks["max_tokens"] == 512
assert checks["rollout_max_concurrent"] == 8
assert checks["rollout_max_head_offpolicyness"] == 0
path = Path(os.environ["JPH_B0_PREFLIGHT_PATH"])
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(checks, stream, allow_nan=False, sort_keys=True)
    stream.flush()
    os.fsync(stream.fileno())
PY
then
  RUN_FAILURE_REASON="hydra-compose-preflight-failed"
  exit 2
fi
if ! redact_and_verify_secret; then
  RUN_FAILURE_REASON="hydra-preflight-secret-verification-failed"
  exit 4
fi

(
  close_inherited_b0_locks
  exec nvidia-smi dmon -s pucm -d 5 -o TD
) > "${RUN_ROOT}/gpu-dmon.log" 2>&1 &
GPU_DMON_PID="$!"
mkfifo -m 600 "${LOG_FIFO}"
(
  close_inherited_b0_locks
  # Coordinator output must not be duplicated into tmux's uncontrolled outer
  # capture.  Operators follow LOG_PATH, which is inside the redaction boundary.
  exec tee -a "${LOG_PATH}" >/dev/null
) < "${LOG_FIFO}" &
LOG_TEE_PID="$!"

# This is the second all-card gate and the sole memory baseline.  Nothing that
# creates a CUDA context runs between this gate and the one coordinator launch.
run_all_gpu_gate immediately-before-launch
for gpu_id in "${GPU_IDS[@]}"; do
  printf '%s,%s,%s\n' "$(date -u +%s)" \
    "${GPU_BASELINE_USED[${gpu_id}]}" "${GPU_BASELINE_FREE[${gpu_id}]}" \
    > "${RUN_ROOT}/gpu-${gpu_id}-memory.csv"
  chmod 600 "${RUN_ROOT}/gpu-${gpu_id}-memory.csv"
done
AUDIT_READY=1

printf 'project=%s AReaL=%s GPUs=0,1,2,3,4,5,6,7 run_root=%s\n' \
  "${PROJECT_COMMIT}" "${ACTUAL_AREAL_COMMIT}" "${RUN_ROOT}" \
  | tee -a "${LOG_PATH}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}" \
  "${AREAL_VENV}/bin/python" - \
    "${AREAL_VENV}/bin/python" examples/math/gsm8k_rl.py \
    "${AREAL_ARGS[@]}" > "${LOG_FIFO}" 2>&1 <<'PY' &
import os
import sys

# The supervisor never forks: its PID becomes both the new SID and the real
# AReaL coordinator PID, so the launcher can bind a stable process identity.
os.setsid()
os.execv(sys.argv[1], sys.argv[1:])
PY
JOB_PID="$!"
if ! bind_job_identity; then
  RUN_FAILURE_REASON="coordinator-setsid-identity-unbound"
  stop_unbound_direct_child
  exit 4
fi

(
  close_inherited_b0_locks
  while true; do
    for gpu_id in "${GPU_IDS[@]}"; do
      if ! sample_output="$(
        nvidia-smi -i "${gpu_id}" \
          --query-gpu=memory.used,memory.free \
          --format=csv,noheader,nounits
      )"; then
        record_breach_once "runtime-sample-failed gpu=${gpu_id}"
        kill -TERM "${B0_ORCHESTRATOR_PID}" >/dev/null 2>&1 || true
        exit 4
      fi
      sample_output="${sample_output// /}"
      IFS=, read -r sample_used sample_free <<< "${sample_output}"
      if [[ ! "${sample_used}" =~ ^[0-9]+$ ]] \
        || [[ ! "${sample_free}" =~ ^[0-9]+$ ]]; then
        record_breach_once "runtime-sample-invalid gpu=${gpu_id}"
        kill -TERM "${B0_ORCHESTRATOR_PID}" >/dev/null 2>&1 || true
        exit 4
      fi
      printf '%s,%s,%s\n' "$(date -u +%s)" "${sample_used}" "${sample_free}" \
        >> "${RUN_ROOT}/gpu-${gpu_id}-memory.csv"
      delta=$((sample_used - GPU_BASELINE_USED[${gpu_id}]))
      if ((delta > SOFT_MAX_NEW_GPU_MEMORY_MIB)); then
        record_breach_once \
          "soft-cap-exceeded gpu=${gpu_id} used=${sample_used} baseline=${GPU_BASELINE_USED[${gpu_id}]} delta=${delta} soft_cap=${SOFT_MAX_NEW_GPU_MEMORY_MIB} hard_cap=${HARD_MAX_NEW_GPU_MEMORY_MIB}"
        kill -TERM "${B0_ORCHESTRATOR_PID}" >/dev/null 2>&1 || true
        exit 4
      fi
    done
    sleep 1
  done
) &
MEMORY_MONITOR_PID="$!"

if wait "${JOB_PID}"; then
  job_exit_code=0
else
  job_exit_code=$?
fi
JOB_LEADER_REAPED=1
if pgrep -s "${JOB_SESSION_ID}" >/dev/null 2>&1; then
  RUN_FAILURE_REASON="areal-session-straggler-after-coordinator-exit"
  exit 4
fi
JOB_PID=""
JOB_START_TIME=""
JOB_SESSION_ID=""
JOB_LEADER_REAPED=0
if ((job_exit_code != 0)); then
  if [[ -e "${MEMORY_BREACH}" ]]; then
    RUN_FAILURE_REASON="gpu-memory-watchdog-stopped-job"
  else
    RUN_FAILURE_REASON="areal-coordinator-exit-${job_exit_code}"
  fi
  exit 4
fi
exit 0
