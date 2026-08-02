from __future__ import annotations

import json
import os
from pathlib import Path
import site
import sys

from jphrl.paths import REMOTE_ROOT, assert_remote_environment, require_within_configured_root


def main() -> None:
    assert_remote_environment()
    checked: dict[str, str] = {}
    for key in (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "PIP_CACHE_DIR",
        "PYTHONUSERBASE",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "UV_PYTHON_BIN_DIR",
        "TMPDIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "SGLANG_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "FLASHINFER_WORKSPACE_BASE",
        "NUMBA_CACHE_DIR",
        "MPLCONFIGDIR",
        "RAY_TMPDIR",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
        "WANDB_DATA_DIR",
    ):
        value = os.environ[key]
        checked[key] = str(require_within_configured_root(value))
    for package_path in site.getsitepackages():
        require_within_configured_root(package_path)
    forbidden_user_root = Path("/home/ljw").resolve()
    for executable_path in os.environ.get("PATH", "").split(os.pathsep):
        if not executable_path:
            continue
        resolved_executable_path = Path(executable_path).expanduser().resolve()
        if resolved_executable_path == forbidden_user_root or forbidden_user_root in resolved_executable_path.parents:
            raise RuntimeError(f"PATH leaks into remote home: {resolved_executable_path}")
    for import_path in sys.path:
        if not import_path:
            continue
        resolved_import_path = Path(import_path).expanduser().resolve()
        if resolved_import_path == forbidden_user_root or forbidden_user_root in resolved_import_path.parents:
            raise RuntimeError(f"sys.path leaks into remote home: {resolved_import_path}")
    if site.ENABLE_USER_SITE:
        raise RuntimeError("Python user site must be disabled")
    if Path(sys.prefix).resolve() == Path("/usr").resolve():
        raise RuntimeError("smoke must run from the project venv")
    report = {
        "ok": True,
        "remote_root": str(REMOTE_ROOT),
        "python": sys.executable,
        "prefix": sys.prefix,
        "site_packages": site.getsitepackages(),
        "worker_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "MAX_JOBS": os.environ.get("MAX_JOBS"),
            "JPH_DATALOADER_WORKERS": os.environ.get("JPH_DATALOADER_WORKERS"),
        },
        "paths": checked,
    }
    destination = REMOTE_ROOT / "artifacts" / "bootstrap" / "path-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
