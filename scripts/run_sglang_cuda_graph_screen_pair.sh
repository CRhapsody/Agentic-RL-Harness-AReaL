#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_sglang_cuda_graph_screen_pair.sh GPU_ID" >&2
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

PAIR_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PAIR_NONCE="$(
  "${JPH_ROOT}/venvs/areal-v2.0.0/bin/python" -c \
    'import secrets; print(secrets.token_hex(16))'
)"
PAIR_ID="${PAIR_STAMP}-gpu${GPU_ID}-${PAIR_NONCE}"
PAIR_ROOT="${JPH_ROOT}/artifacts/sglang-cuda-graph-screen/pairs/${PAIR_ID}"
C2A_POINTER="${PAIR_ROOT}/c2a-run-root.txt"
C2B_POINTER="${PAIR_ROOT}/c2b-run-root.txt"
REPORT_PATH="${PAIR_ROOT}/comparison.json"
CONFIG_PREFLIGHT_PATH="${PAIR_ROOT}/config-preflight.json"
mkdir -p -m 700 "${PAIR_ROOT}"

CUDA_VISIBLE_DEVICES="" \
PYTHONDONTWRITEBYTECODE=1 \
  "${JPH_ROOT}/venvs/areal-v2.0.0/bin/python" \
  "${SCRIPT_DIR}/validate_sglang_cuda_graph_config.py" \
  --config "${JPH_ROOT}/src/AReaL-v2.0.0/examples/math/gsm8k_grpo.yaml" \
  --output "${CONFIG_PREFLIGHT_PATH}"

export JPH_SCREEN_PAIR_ID="${PAIR_ID}"
export JPH_SCREEN_CELL_POINTER="${C2A_POINTER}"
/bin/bash "${SCRIPT_DIR}/run_sglang_cuda_graph_screen.sh" "${GPU_ID}" c2a

export JPH_SCREEN_CELL_POINTER="${C2B_POINTER}"
/bin/bash "${SCRIPT_DIR}/run_sglang_cuda_graph_screen.sh" "${GPU_ID}" c2b

C2A_ROOT="$(<"${C2A_POINTER}")"
C2B_ROOT="$(<"${C2B_POINTER}")"
"${JPH_ROOT}/venvs/areal-v2.0.0/bin/python" \
  "${SCRIPT_DIR}/compare_sglang_cuda_graph_screen.py" \
  "${C2A_ROOT}" "${C2B_ROOT}" --output "${REPORT_PATH}"

echo "SGLang CUDA Graph screen pair complete: ${REPORT_PATH}"
