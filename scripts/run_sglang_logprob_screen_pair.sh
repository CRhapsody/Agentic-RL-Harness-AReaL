#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_sglang_logprob_screen_pair.sh GPU_ID" >&2
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
PAIR_ROOT="${JPH_ROOT}/artifacts/sglang-logprob-screen/pairs/${PAIR_ID}"
C0_POINTER="${PAIR_ROOT}/c0-run-root.txt"
C1_POINTER="${PAIR_ROOT}/c1-run-root.txt"
REPORT_PATH="${PAIR_ROOT}/comparison.json"
mkdir -p -m 700 "${PAIR_ROOT}"

export JPH_SCREEN_PAIR_ID="${PAIR_ID}"
export JPH_SCREEN_CELL_POINTER="${C0_POINTER}"
/bin/bash "${SCRIPT_DIR}/run_sglang_logprob_screen.sh" "${GPU_ID}" c0

export JPH_SCREEN_CELL_POINTER="${C1_POINTER}"
/bin/bash "${SCRIPT_DIR}/run_sglang_logprob_screen.sh" "${GPU_ID}" c1

C0_ROOT="$(<"${C0_POINTER}")"
C1_ROOT="$(<"${C1_POINTER}")"
"${JPH_ROOT}/venvs/areal-v2.0.0/bin/python" \
  "${SCRIPT_DIR}/compare_sglang_logprob_screen.py" \
  "${C0_ROOT}" "${C1_ROOT}" --output "${REPORT_PATH}"

echo "SGLang log-prob screen pair complete: ${REPORT_PATH}"
