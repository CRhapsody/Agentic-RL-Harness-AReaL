#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
EXPECTED_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/qwen2.5-1.5b-snapshot.json"
DATASET_REPORT="${JPH_ROOT}/artifacts/bootstrap/gsm8k-snapshot.json"
MIN_FREE_MEMORY_MIB="${JPH_B0_MIN_FREE_MEMORY_MIB:-73728}"
MAX_USED_MEMORY_MIB="${JPH_B0_MAX_USED_MEMORY_MIB:-8192}"

if [[ ! "${MIN_FREE_MEMORY_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MIN_FREE_MEMORY_MIB must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MAX_USED_MEMORY_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JPH_B0_MAX_USED_MEMORY_MIB must be a positive integer" >&2
  exit 2
fi

if [[ ! -d "${AREAL_REPO}/.git" ]]; then
  echo "Missing ${AREAL_REPO}; bootstrap the pinned AReaL source first" >&2
  exit 2
fi
if [[ ! -x "${AREAL_VENV}/bin/python" ]]; then
  echo "Missing ${AREAL_VENV}; bootstrap the pinned AReaL environment first" >&2
  exit 2
fi
for REPORT in "${MODEL_REPORT}" "${DATASET_REPORT}"; do
  if [[ ! -f "${REPORT}" ]]; then
    echo "Missing pinned Hugging Face snapshot report: ${REPORT}" >&2
    exit 2
  fi
done

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
for SNAPSHOT in "${MODEL_SNAPSHOT}" "${DATASET_SNAPSHOT}"; do
  if [[ ! -d "${SNAPSHOT}" ]]; then
    echo "Pinned snapshot directory is missing: ${SNAPSHOT}" >&2
    exit 2
  fi
  case "${SNAPSHOT}" in
    "${JPH_ROOT}"/*) ;;
    *)
      echo "Pinned snapshot escapes ${JPH_ROOT}: ${SNAPSHOT}" >&2
      exit 2
      ;;
  esac
done

ACTUAL_COMMIT="$(git -C "${AREAL_REPO}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "AReaL commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 2
fi
if ! grep -q "total_train_steps" "${AREAL_REPO}/areal/api/cli_args.py"; then
  echo "Pinned AReaL source does not expose total_train_steps; refusing an unbounded B0" >&2
  exit 2
fi

GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" != "8" ]]; then
  echo "Official B0 requires exactly 8 visible GPUs; found ${GPU_COUNT}" >&2
  exit 3
fi
for GPU_ID in 0 1 2 3 4 5 6 7; do
  IFS=, read -r GPU_MEMORY_USED GPU_MEMORY_FREE < <(
    nvidia-smi -i "${GPU_ID}" \
      --query-gpu=memory.used,memory.free \
      --format=csv,noheader,nounits | tr -d ' '
  )
  if [[ ! "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]] || \
    [[ ! "${GPU_MEMORY_FREE}" =~ ^[0-9]+$ ]]; then
    echo "Could not read GPU ${GPU_ID} memory usage" >&2
    exit 3
  fi
  if ((GPU_MEMORY_FREE < MIN_FREE_MEMORY_MIB || GPU_MEMORY_USED > MAX_USED_MEMORY_MIB)); then
    echo "GPU ${GPU_ID} failed the memory headroom gate: used=${GPU_MEMORY_USED}MiB free=${GPU_MEMORY_FREE}MiB; require used<=${MAX_USED_MEMORY_MIB}MiB and free>=${MIN_FREE_MEMORY_MIB}MiB" >&2
    exit 3
  fi
done

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${JPH_ROOT}/artifacts/areal-b0/${RUN_STAMP}"
LOG_PATH="${JPH_ROOT}/logs/areal-b0-${RUN_STAMP}.log"
NAME_RESOLVE_ROOT="${JPH_ROOT}/runtime/name_resolve/${RUN_STAMP}"
mkdir -p "${RUN_ROOT}" "${NAME_RESOLVE_ROOT}"

GPU_MONITOR_PID=""
cleanup() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
nvidia-smi dmon -s pucm -d 5 -o TD > "${RUN_ROOT}/gpu-dmon.log" 2>&1 &
GPU_MONITOR_PID="$!"

cd "${AREAL_REPO}"
echo "AReaL=${ACTUAL_COMMIT} GPUs=0,1,2,3,4,5,6,7 run_root=${RUN_ROOT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
  "${AREAL_VENV}/bin/python" examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  experiment_name=jph-b0 \
  trial_name="${RUN_STAMP}" \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=8 \
  total_train_steps=1 \
  actor.path="${MODEL_SNAPSHOT}" \
  tokenizer_path="${MODEL_SNAPSHOT}" \
  train_dataset.path="${DATASET_SNAPSHOT}" \
  valid_dataset.path="${DATASET_SNAPSHOT}" \
  train_dataset.batch_size=8 \
  train_dataset.num_workers=2 \
  valid_dataset.batch_size=8 \
  valid_dataset.num_workers=2 \
  gconfig.n_samples=2 \
  gconfig.max_new_tokens=256 \
  gconfig.max_tokens=512 \
  rollout.max_concurrent_rollouts=8 \
  rollout.max_head_offpolicyness=0 \
  rollout.dump_to_file=true \
  cluster.fileroot="${RUN_ROOT}" \
  cluster.name_resolve.nfs_record_root="${NAME_RESOLVE_ROOT}" \
  2>&1 | tee "${LOG_PATH}"
