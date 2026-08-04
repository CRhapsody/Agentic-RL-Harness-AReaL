#!/usr/bin/env bash

# This file is sourced by the M0 launchers.  The outer pipeline keeps one
# per-GPU flock for its complete rollout -> joint-update lifetime and passes
# the already-locked descriptor to both inner stages.  Direct inner-stage
# invocations still acquire the same lock themselves.

jph_acquire_m0_gpu_lock() {
  if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-7]$ ]]; then
    echo "jph_acquire_m0_gpu_lock requires a GPU ID from 0 through 7" >&2
    return 1
  fi

  local gpu_id="$1"
  local lock_dir="${JPH_ROOT}/runtime/locks"
  local lock_path="${lock_dir}/gpu-${gpu_id}.lock"
  local inherited_fd="${JPH_M0_GPU_LOCK_FD:-}"
  local actual_path expected_path

  mkdir -p -m 700 "${lock_dir}"
  if [[ -n "${inherited_fd}" ]]; then
    if [[ ! "${inherited_fd}" =~ ^[0-9]+$ ]] \
      || [[ ! -e "/proc/$$/fd/${inherited_fd}" ]]; then
      echo "Inherited M0 GPU lock descriptor is invalid" >&2
      return 1
    fi
    actual_path="$(readlink -f "/proc/$$/fd/${inherited_fd}")"
    expected_path="$(readlink -f "${lock_path}")"
    if [[ "${actual_path}" != "${expected_path}" ]]; then
      echo "Inherited M0 GPU lock does not match GPU ${gpu_id}" >&2
      return 1
    fi
    if ! flock -n "${inherited_fd}"; then
      echo "Inherited M0 GPU lock is no longer held" >&2
      return 1
    fi
    return 0
  fi

  exec 9> "${lock_path}"
  if ! flock -n 9; then
    echo "GPU ${gpu_id} is reserved by another JPH process" >&2
    return 1
  fi
  export JPH_M0_GPU_LOCK_FD=9
}
