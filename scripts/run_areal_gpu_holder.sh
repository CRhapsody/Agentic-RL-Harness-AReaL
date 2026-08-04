#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/run_areal_gpu_holder.sh" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "AReaL GPU holder must run inside tmux" >&2
  exit 2
fi
if [[ ! "${JPH_HOLDER_RUN_ID:-}" =~ ^[0-9]{8}T[0-9]{6}Z-gpu-holder-[0-9a-f]{16}$ ]]; then
  echo "Set JPH_HOLDER_RUN_ID to a canonical, unique holder run ID" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JPH_B0_RUN_MODE=holder
exec /bin/bash "${SCRIPT_DIR}/run_areal_official_b0.sh"
