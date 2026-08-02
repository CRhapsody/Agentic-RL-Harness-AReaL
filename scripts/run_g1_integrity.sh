#!/usr/bin/env bash
set -euo pipefail

umask 077

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/remote_env.sh"

python_bin="${JPH_PYTHON:-/mnt/sdb/ljw/chizm/venvs/areal-v2.0.0/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "G1 Python interpreter is unavailable: ${python_bin}" >&2
  exit 2
fi

"${python_bin}" -c 'from jphrl.paths import assert_remote_environment; assert_remote_environment()'

run_stamp="${JPH_G1_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${JPH_G1_RUN_ROOT:-/mnt/sdb/ljw/chizm/artifacts/g1/${run_stamp}}"
mkdir -p -m 700 "${run_root}"
chmod 700 "${run_root}"

log_path="${run_root}/run.log"
touch "${log_path}"
chmod 600 "${log_path}"
exec > >(tee -a "${log_path}") 2>&1

project_commit="$(git -C "${project_root}" rev-parse HEAD)"
echo "run_stamp=${run_stamp}"
echo "project_commit=${project_commit}"
echo "run_root=${run_root}"

"${python_bin}" -m jphrl.experiments.g1_integrity \
  --version-fixtures 1000 \
  --work-dir "${run_root}/work" \
  --output "${run_root}/result.json" \
  --project-commit "${project_commit}"

"${python_bin}" "${project_root}/scripts/verify_g1_integrity.py" \
  --result "${run_root}/result.json" \
  --audit "${run_root}/audit.json"

echo "G1 integrity experiment passed: ${run_root}"
