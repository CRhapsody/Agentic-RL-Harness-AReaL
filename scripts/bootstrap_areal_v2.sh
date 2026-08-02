#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

AREAL_REPO="${JPH_ROOT}/src/AReaL-v2.0.0"
AREAL_VENV="${JPH_ROOT}/venvs/areal-v2.0.0"
EXPECTED_COMMIT="fee938eada49208a5aabdbc1095730a13076a349"
UV_VERSION="0.11.26"
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
BOOTSTRAP_ARTIFACTS="${JPH_ROOT}/artifacts/bootstrap/areal-v2.0.0"

case "${PROJECT_DIR}" in
  "${JPH_ROOT}"/*) ;;
  *)
    echo "Refusing AReaL bootstrap: project is outside ${JPH_ROOT}: ${PROJECT_DIR}" >&2
    exit 2
    ;;
esac

mkdir -p "${JPH_ROOT}/src" "${BOOTSTRAP_ARTIFACTS}"

bash "${SCRIPT_DIR}/preflight_remote_network.sh"

if [[ ! -d "${AREAL_REPO}/.git" ]]; then
  if [[ -e "${AREAL_REPO}" ]]; then
    echo "Refusing clone over non-git path: ${AREAL_REPO}" >&2
    exit 2
  fi
  git clone \
    --branch v2.0.0 \
    --depth 1 \
    https://github.com/areal-project/AReaL.git \
    "${AREAL_REPO}"
fi

ACTUAL_COMMIT="$(git -C "${AREAL_REPO}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "AReaL commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${AREAL_REPO}" status --porcelain --untracked-files=no)" ]]; then
  echo "Pinned AReaL checkout has tracked modifications; refusing bootstrap" >&2
  exit 2
fi

UV_BIN="${JPH_ROOT}/bin/uv"
ACTUAL_UV_VERSION=""
if [[ -x "${UV_BIN}" ]]; then
  ACTUAL_UV_VERSION="$("${UV_BIN}" --version | awk '{print $2}')"
fi
if [[ "${ACTUAL_UV_VERSION}" != "${UV_VERSION}" ]]; then
  curl --proto '=https' --tlsv1.2 -LsSf \
    "https://astral.sh/uv/${UV_VERSION}/install.sh" \
    | env UV_UNMANAGED_INSTALL="${JPH_ROOT}/bin" sh
fi
ACTUAL_UV_VERSION="$("${UV_BIN}" --version | awk '{print $2}')"
if [[ "${ACTUAL_UV_VERSION}" != "${UV_VERSION}" ]]; then
  echo "uv version mismatch after standalone install: expected ${UV_VERSION}, got ${ACTUAL_UV_VERSION}" >&2
  exit 2
fi
"${UV_BIN}" --version | tee "${BOOTSTRAP_ARTIFACTS}/uv-version.txt"

"${UV_BIN}" python install 3.12
if [[ ! -x "${AREAL_VENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.12 "${AREAL_VENV}"
fi

PYTHON_VERSION="$("${AREAL_VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
  echo "AReaL environment must use Python 3.12, got ${PYTHON_VERSION}" >&2
  exit 2
fi

cd "${AREAL_REPO}"
VIRTUAL_ENV="${AREAL_VENV}" \
UV_PROJECT_ENVIRONMENT="${AREAL_VENV}" \
  "${UV_BIN}" sync \
  --python "${AREAL_VENV}/bin/python" \
  --locked \
  --extra cuda \
  --default-index "${PIP_INDEX_URL}"

# flash-attn is intentionally outside v2.0.0's exact uv lock. Install it after
# sync with --no-deps; a later exact sync would otherwise remove it.
VIRTUAL_ENV="${AREAL_VENV}" \
UV_PROJECT_ENVIRONMENT="${AREAL_VENV}" \
  "${UV_BIN}" pip install \
  --python "${AREAL_VENV}/bin/python" \
  --no-deps \
  "${FLASH_ATTN_WHEEL}"

VALIDATION_GPU_ID=""
GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ ! "${GPU_COUNT}" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 1 )); then
  echo "No NVIDIA GPU is visible for AReaL validation" >&2
  exit 3
fi
for ((GPU_ID = 0; GPU_ID < GPU_COUNT; GPU_ID++)); do
  GPU_MEMORY_USED="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${GPU_MEMORY_USED}" =~ ^[0-9]+$ ]] && (( GPU_MEMORY_USED < 500 )); then
    VALIDATION_GPU_ID="${GPU_ID}"
    break
  fi
done
if [[ -z "${VALIDATION_GPU_ID}" ]]; then
  echo "No GPU below 500 MiB is available for the Flash Attention ABI smoke" >&2
  exit 3
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader \
  > "${BOOTSTRAP_ARTIFACTS}/gpu-inventory.csv"
CUDA_VISIBLE_DEVICES="${VALIDATION_GPU_ID}" \
  "${AREAL_VENV}/bin/python" areal/tools/validate_installation.py \
  2>&1 | tee "${BOOTSTRAP_ARTIFACTS}/validate-installation.log"
"${UV_BIN}" pip freeze --python "${AREAL_VENV}/bin/python" \
  > "${BOOTSTRAP_ARTIFACTS}/pip-freeze.txt"
CUDA_VISIBLE_DEVICES="${VALIDATION_GPU_ID}" \
JPH_PHYSICAL_GPU_ID="${VALIDATION_GPU_ID}" \
  "${AREAL_VENV}/bin/python" "${PROJECT_DIR}/scripts/validate_areal_runtime.py" \
  | tee "${BOOTSTRAP_ARTIFACTS}/runtime-versions.json"

printf '%s\n' "${ACTUAL_COMMIT}" > "${BOOTSTRAP_ARTIFACTS}/areal-commit.txt"
"${AREAL_VENV}/bin/python" "${PROJECT_DIR}/scripts/audit_remote_paths.py"
