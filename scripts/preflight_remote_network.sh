#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

OUTPUT="${JPH_ROOT}/artifacts/bootstrap/network-preflight.txt"
UV_VERSION="0.11.26"
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the root-local bootstrap" >&2
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required for the pinned AReaL checkout" >&2
  exit 2
fi

check_url() {
  local label="$1"
  local url="$2"
  echo "checking=${label} url=${url}"
  curl --proto '=https' --tlsv1.2 --connect-timeout 10 --max-time 30 \
    --retry 2 --retry-delay 1 -fsSIL "${url}" >/dev/null
  echo "ok=${label}"
}

{
  check_url "hf-mirror" "https://hf-mirror.com/"
  check_url "tsinghua-pypi" "https://pypi.tuna.tsinghua.edu.cn/simple/"
  check_url "uv-installer" "https://astral.sh/uv/${UV_VERSION}/install.sh"
  check_url "flash-attn-wheel" "${FLASH_ATTN_WHEEL}"
  check_url "pythonhosted-artifacts" "https://files.pythonhosted.org/"
  check_url "pytorch-cu129-index" "https://download.pytorch.org/whl/cu129/"
  echo "checking=areal-tag"
  git ls-remote --exit-code --tags https://github.com/areal-project/AReaL.git \
    refs/tags/v2.0.0 refs/tags/v2.0.0^{}
  echo "ok=areal-tag"
} 2>&1 | tee "${OUTPUT}"
