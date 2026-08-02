#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

case "${PROJECT_DIR}" in
  "${JPH_ROOT}"/*) ;;
  *)
    echo "Refusing installation: project is outside ${JPH_ROOT}: ${PROJECT_DIR}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_DIR}"

python3 -m jphrl.cli \
  --backend mock \
  --task add-17-25 \
  --output "${JPH_ROOT}/artifacts/bootstrap/mock-preinstall.json"

VENV_DIR="${JPH_ROOT}/venvs/jphrl-smoke"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install \
  --index-url "${PIP_INDEX_URL}" \
  --upgrade pip setuptools wheel

"${VENV_DIR}/bin/python" -m pip install \
  --index-url "${PIP_INDEX_URL}" \
  --requirement "${PROJECT_DIR}/requirements-smoke.txt"

"${VENV_DIR}/bin/python" -m pip freeze > "${JPH_ROOT}/artifacts/bootstrap/pip-freeze.txt"
"${VENV_DIR}/bin/python" -m unittest discover -s "${PROJECT_DIR}/tests" -v
"${VENV_DIR}/bin/python" "${PROJECT_DIR}/scripts/audit_remote_paths.py"
