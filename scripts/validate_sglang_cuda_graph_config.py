from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config
import torch


def _require_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"config escapes JPH_ROOT: {resolved}")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"invalid AReaL config: {resolved}")
    return resolved


def _compose(config_path: Path, expected: bool) -> dict[str, object]:
    value = "true" if expected else "false"
    config, _ = load_expr_config(
        [
            "--config",
            str(config_path),
            f"+sglang.disable_cuda_graph={value}",
        ],
        GRPOConfig,
    )
    composed = config.sglang.disable_cuda_graph
    if type(composed) is not bool or composed is not expected:
        raise ValueError(
            "Hydra did not compose sglang.disable_cuda_graph as the expected bool"
        )
    server_args = SGLangConfig.build_args(
        config.sglang,
        tp_size=1,
        base_gpu_id=0,
    )
    built = server_args.get("disable_cuda_graph")
    if type(built) is not bool or built is not expected:
        raise ValueError(
            "SGLangConfig.build_args did not preserve disable_cuda_graph"
        )
    return {
        "cli_override": f"+sglang.disable_cuda_graph={value}",
        "composed_type": type(composed).__name__,
        "composed_value": composed,
        "server_arg_type": type(built).__name__,
        "server_arg_value": built,
    }


def _write_report(report: dict[str, object], output: Path, root: Path) -> None:
    resolved = output.resolve()
    if not resolved.is_relative_to(root) or resolved.parent.is_symlink():
        raise ValueError(f"preflight output escapes JPH_ROOT: {resolved}")
    if not resolved.parent.is_dir():
        raise ValueError(f"preflight output parent is missing: {resolved.parent}")
    unsigned = {key: value for key, value in report.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    report["record_sha256"] = hashlib.sha256(payload).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(resolved, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate both C2 CUDA Graph overrides without starting CUDA"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(os.environ["JPH_ROOT"]).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"invalid JPH_ROOT: {root}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the config-only preflight")
    config_path = _require_within(args.config, root)
    cells = {
        "c2a": _compose(config_path, False),
        "c2b": _compose(config_path, True),
    }
    if torch.cuda.is_initialized():
        raise RuntimeError("config-only preflight unexpectedly initialized CUDA")
    report = {
        "schema_version": "jph.sglang-cuda-graph-config-preflight.v1",
        "config": str(config_path),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cells": cells,
        "passed": True,
    }
    _write_report(report, args.output, root)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
