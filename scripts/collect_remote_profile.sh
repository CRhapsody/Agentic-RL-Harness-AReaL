#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

OUTPUT="${JPH_ROOT}/artifacts/bootstrap/remote-profile.txt"
{
  echo "__HOST__"
  hostname
  id
  echo "__GPU__"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version --format=csv,noheader
  echo "__TOPOLOGY__"
  nvidia-smi topo -m
  echo "__FILESYSTEMS__"
  df -h "${JPH_ROOT}" /tmp /
  echo "__CPU__"
  lscpu
  echo "__RAM__"
  free -h
  echo "__PYTHON__"
  command -v python3
  python3 --version
  for PYTHON_BIN in python3.11 python3.12; do
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      command -v "${PYTHON_BIN}"
      "${PYTHON_BIN}" --version
    fi
  done
  echo "__CUDA_TOOLKIT__"
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    nvcc --version
  else
    echo "nvcc=missing"
  fi
  echo "__TOOLS__"
  for TOOL in git curl uv docker; do
    if command -v "${TOOL}" >/dev/null 2>&1; then
      echo "${TOOL}=$(command -v "${TOOL}")"
    else
      echo "${TOOL}=missing"
    fi
  done
  echo "__DOCKER_ROOT__"
  if command -v docker >/dev/null 2>&1; then
    docker info --format '{{.DockerRootDir}}' 2>&1 || true
  else
    echo "docker=missing"
  fi
} | tee "${OUTPUT}"
