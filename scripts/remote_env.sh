#!/usr/bin/env bash

set -euo pipefail

export JPH_ROOT="/mnt/sdb/ljw/chizm"
export JPH_PROJECT_DIR="${JPH_ROOT}/src/Agentic-RL-Harness-AReaL"
export PATH="${JPH_ROOT}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${JPH_PROJECT_DIR}"

unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV PYTHONHOME PYENV_ROOT PYENV_VERSION
unset CUDA_VISIBLE_DEVICES NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX GLOO_SOCKET_IFNAME

export HOME="${JPH_ROOT}/runtime_home"
export XDG_CACHE_HOME="${JPH_ROOT}/cache/xdg"
export XDG_CONFIG_HOME="${JPH_ROOT}/config/xdg"
export XDG_DATA_HOME="${JPH_ROOT}/data/xdg"
export HF_HOME="${JPH_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${JPH_ROOT}/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${JPH_ROOT}/cache/torchinductor"
export PIP_CACHE_DIR="${JPH_ROOT}/cache/pip"
export PYTHONUSERBASE="${JPH_ROOT}/python_userbase"
export UV_CACHE_DIR="${JPH_ROOT}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${JPH_ROOT}/runtime/python"
export UV_PYTHON_BIN_DIR="${JPH_ROOT}/bin"
export UV_PYTHON_PREFERENCE="only-managed"
export UV_MANAGED_PYTHON="1"
export UV_LINK_MODE="copy"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export TMPDIR="${JPH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TRITON_CACHE_DIR="${JPH_ROOT}/cache/triton"
export CUDA_CACHE_PATH="${JPH_ROOT}/cache/cuda"
export SGLANG_CACHE_DIR="${JPH_ROOT}/cache/sglang"
export VLLM_CACHE_ROOT="${JPH_ROOT}/cache/vllm"
export FLASHINFER_WORKSPACE_BASE="${JPH_ROOT}/cache/flashinfer"
export NUMBA_CACHE_DIR="${JPH_ROOT}/cache/numba"
export MPLCONFIGDIR="${JPH_ROOT}/config/matplotlib"
export RAY_TMPDIR="${JPH_ROOT}/tmp/ray"
export WANDB_DIR="${JPH_ROOT}/logs/wandb"
export WANDB_CACHE_DIR="${JPH_ROOT}/cache/wandb"
export WANDB_CONFIG_DIR="${JPH_ROOT}/config/wandb"
export WANDB_DATA_DIR="${JPH_ROOT}/data/wandb"

export HF_ENDPOINT="https://hf-mirror.com"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_CONFIG_FILE="/dev/null"
export PYTHONNOUSERSITE="1"
export HF_HUB_DISABLE_TELEMETRY="1"
export TOKENIZERS_PARALLELISM="false"
export WANDB_MODE="offline"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export MAKEFLAGS="${MAKEFLAGS:--j8}"
export JPH_DATALOADER_WORKERS="${JPH_DATALOADER_WORKERS:-2}"

mkdir -p \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}" \
  "${UV_PYTHON_BIN_DIR}" \
  "${TMPDIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${SGLANG_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${NUMBA_CACHE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${RAY_TMPDIR}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${JPH_ROOT}/artifacts" \
  "${JPH_ROOT}/logs" \
  "${JPH_ROOT}/models" \
  "${JPH_ROOT}/venvs" \
  "${JPH_ROOT}/artifacts/bootstrap" \
  "${JPH_ROOT}/artifacts/smoke"
