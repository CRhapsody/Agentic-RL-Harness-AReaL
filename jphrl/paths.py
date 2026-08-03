from __future__ import annotations

import os
from pathlib import Path

REMOTE_ROOT = Path("/mnt/sdb/ljw/chizm")


def configured_root() -> Path | None:
    value = os.environ.get("JPH_ROOT")
    return Path(value).expanduser().resolve() if value else None


def require_within_configured_root(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    root = configured_root()
    if root is None:
        return destination
    if destination != root and root not in destination.parents:
        raise ValueError(f"path {destination} is outside configured JPH_ROOT {root}")
    return destination


def repository_root() -> Path:
    """Resolve the real source checkout instead of trusting a caller path."""

    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"cannot locate the Git checkout containing {source}")


def require_outside_repository(path: str | Path) -> Path:
    """Require a runtime artifact path to be outside this actual checkout."""

    destination = require_within_configured_root(path)
    project = repository_root()
    if destination == project or project in destination.parents:
        raise ValueError(
            f"runtime artifact {destination} must be outside Git checkout {project}"
        )
    return destination


def assert_remote_environment() -> None:
    root = configured_root()
    if root is None:
        raise RuntimeError("JPH_ROOT is not set")
    if root != REMOTE_ROOT:
        raise RuntimeError(f"remote JPH_ROOT must be {REMOTE_ROOT}, got {root}")
    forbidden_home = Path("/home/ljw").resolve()
    home = Path.home().resolve()
    if home == forbidden_home or forbidden_home in home.parents:
        raise RuntimeError(
            "HOME still points to /home/ljw; source scripts/remote_env.sh"
        )
    for variable in (
        "HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TORCH_HOME",
        "PIP_CACHE_DIR",
        "TMPDIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        value = os.environ.get(variable)
        if not value:
            raise RuntimeError(f"required path variable {variable} is unset")
        require_within_configured_root(value)
