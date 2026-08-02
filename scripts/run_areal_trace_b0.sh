#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_areal_trace_b0.sh GPU_ID" >&2
  exit 2
fi
GPU_ID="$1"
if [[ ! "${GPU_ID}" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be an integer from 0 through 7: ${GPU_ID}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
EXPECTED_AREAL_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/qwen2.5-1.5b-snapshot.json"
DATASET_REPORT="${JPH_ROOT}/artifacts/bootstrap/gsm8k-snapshot.json"
CUDA_TOOLKIT_ROOT="${JPH_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.6}"
MAX_USED_MEMORY_MIB="${JPH_TRACE_MAX_USED_MEMORY_MIB:-10240}"
MIN_FREE_MEMORY_MIB="${JPH_TRACE_MIN_FREE_MEMORY_MIB:-65536}"

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
  -o "${JPH_ROOT}/tmp/nvcc-trace-preflight.o"
ACTUAL_AREAL_COMMIT="$(git -C "${AREAL_REPO}" rev-parse HEAD)"
if [[ "${ACTUAL_AREAL_COMMIT}" != "${EXPECTED_AREAL_COMMIT}" ]]; then
  echo "AReaL commit mismatch: ${ACTUAL_AREAL_COMMIT}" >&2
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
' "${MODEL_REPORT}" "${DATASET_REPORT}"
)
MODEL_SNAPSHOT="${SNAPSHOT_VALUES[0]}"
MODEL_REVISION="${SNAPSHOT_VALUES[1]}"
DATASET_SNAPSHOT="${SNAPSHOT_VALUES[2]}"
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

mkdir -p "${JPH_ROOT}/runtime/locks"
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
RUN_ROOT="${JPH_ROOT}/artifacts/areal-trace-b0/${RUN_STAMP}"
TRACE_DIR="${RUN_ROOT}/traces"
AUDIT_PATH="${RUN_ROOT}/audit.json"
NAME_RESOLVE_ROOT="${JPH_ROOT}/runtime/name_resolve/trace-${RUN_STAMP}"
LOG_PATH="${JPH_ROOT}/logs/areal-trace-b0-${RUN_STAMP}.log"
mkdir -p "${TRACE_DIR}" "${NAME_RESOLVE_ROOT}"
umask 077

GPU_MONITOR_PID=""
cleanup() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
nvidia-smi dmon -i "${GPU_ID}" -s pucm -d 5 -o TD > "${RUN_ROOT}/gpu-dmon.log" 2>&1 &
GPU_MONITOR_PID="$!"

echo "AReaL=${ACTUAL_AREAL_COMMIT} physical_gpu=${GPU_ID} model_revision=${MODEL_REVISION}"
echo "run_root=${RUN_ROOT} trace_dir=${TRACE_DIR}"

cd "${AREAL_REPO}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
JPH_AREAL_TRACE_TASKS=1 \
JPH_AREAL_TRACE_DIR="${TRACE_DIR}" \
JPH_AREAL_COMMIT="${ACTUAL_AREAL_COMMIT}" \
JPH_BEHAVIOR_SNAPSHOT="${MODEL_SNAPSHOT}" \
JPH_BEHAVIOR_REVISION="${MODEL_REVISION}" \
  "${AREAL_VENV}/bin/python" "${JPH_PROJECT_DIR}/scripts/run_areal_trace_eval.py" \
  --config "${AREAL_REPO}/examples/math/gsm8k_grpo.yaml" \
  scheduler.type=local \
  experiment_name=jph-areal-trace-b0 \
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
  sglang.mem_fraction_static=0.35 \
  sglang.context_length=1024 \
  sglang.max_running_requests=1 \
  2>&1 | tee "${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
  "${AREAL_VENV}/bin/python" "${JPH_PROJECT_DIR}/scripts/verify_areal_trace.py" \
  "${TRACE_DIR}" \
  --model-report "${MODEL_REPORT}" \
  --expected-areal-commit "${EXPECTED_AREAL_COMMIT}" \
  --expected-policy-version 0 \
  --device cuda:0 \
  --max-tokens-per-trace 64 \
  --max-traces 1 \
  --max-abs-error 0.25 \
  --max-mean-abs-error 0.05 \
  --output "${AUDIT_PATH}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "AReaL trace audit passed: ${AUDIT_PATH}"
