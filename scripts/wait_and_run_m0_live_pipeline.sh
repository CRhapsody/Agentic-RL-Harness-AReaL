#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/wait_and_run_m0_live_pipeline.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"
umask 077

POLL_SECONDS="${JPH_M0_POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${JPH_M0_MAX_WAIT_SECONDS:-604800}"

if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "${MAX_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "M0 poll and maximum-wait values must be positive integers" >&2
  exit 2
fi
if ((POLL_SECONDS > 60)); then
  echo "JPH_M0_POLL_SECONDS cannot exceed 60" >&2
  exit 2
fi

PROJECT_COMMIT="$(git -C "${JPH_PROJECT_DIR}" rev-parse HEAD)"
if [[ -n "$(
  git -C "${JPH_PROJECT_DIR}" status --porcelain=v1 --untracked-files=all
)" ]]; then
  echo "Project worktree must be clean before waiting for M0" >&2
  exit 2
fi

deadline="$(( $(date +%s) + MAX_WAIT_SECONDS ))"
echo "Selecting one observed M0 GPU without a configured memory limit: project=${PROJECT_COMMIT} deadline=${deadline}"
while (( $(date +%s) < deadline )); do
  for gpu_id in 0 1 2 3 4 5 6 7; do
    if ! snapshot="$(
      nvidia-smi -i "${gpu_id}" \
        --query-gpu=memory.used,memory.free \
        --format=csv,noheader,nounits | tr -d ' '
    )"; then
      echo "$(date -u +%FT%TZ) gpu=${gpu_id} state=unreadable"
      continue
    fi
    IFS=, read -r memory_used memory_free <<< "${snapshot}"
    if ! processes="$(
      nvidia-smi -i "${gpu_id}" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits
    )"; then
      echo "$(date -u +%FT%TZ) gpu=${gpu_id} processes=unreadable"
      continue
    fi
    if [[ ! "${memory_used}" =~ ^[0-9]+$ ]] \
      || [[ ! "${memory_free}" =~ ^[0-9]+$ ]]; then
      echo "$(date -u +%FT%TZ) gpu=${gpu_id} memory=invalid"
      continue
    fi
    process_count=0
    if [[ -n "${processes}" ]]; then
      process_count="$(printf '%s\n' "${processes}" | wc -l | tr -d ' ')"
    fi
    echo "$(date -u +%FT%TZ) gpu=${gpu_id} used=${memory_used}MiB free=${memory_free}MiB processes=${process_count}"
    if [[ "${memory_used}" =~ ^[0-9]+$ ]] \
      && [[ "${memory_free}" =~ ^[0-9]+$ ]]; then
      echo "GPU ${gpu_id} was observed; handing it to the launcher's fresh checks"
      set +e
      JPH_PROJECT_COMMIT="${PROJECT_COMMIT}" \
        bash "${SCRIPT_DIR}/run_m0_live_pipeline.sh" "${gpu_id}"
      pipeline_status=$?
      set -e
      if [[ "${pipeline_status}" -eq 0 ]]; then
        exit 0
      fi
      if [[ "${pipeline_status}" -eq 3 ]]; then
        echo "GPU ${gpu_id} changed or was locked before launch; resuming read-only wait"
        break
      fi
      echo "M0 pipeline failed with status ${pipeline_status}; refusing an automatic experimental retry" >&2
      exit "${pipeline_status}"
    fi
  done
  sleep "${POLL_SECONDS}"
done

echo "No GPU could be observed before the wait deadline" >&2
exit 3
