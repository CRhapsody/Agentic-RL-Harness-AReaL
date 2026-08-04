#!/usr/bin/env python3

from __future__ import annotations

"""Fail-closed, CPU-only verification for an official one-step AReaL B0 run."""

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


PINNED_AREAL_COMMIT = "fee938eada49208a5aabdbc1095730a13076a349"
MEMORY_AUDIT_SCHEMA = "jph.areal-official-b0-gpu-memory-observation.v2"
MEMORY_RUN_KIND = "areal-official-b0-v1"
VERIFICATION_SCHEMA = "jph.areal-official-b0-verification.v1"
EXPECTED_GPU_IDS = tuple(range(8))

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_YAML_MAPPING = re.compile(r"^( *)([A-Za-z_][A-Za-z0-9_-]*):(?:[ ](.*))?$")
_NUMBER = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z"
)
_WORKER_STARTED = re.compile(
    r"Worker (actor|rollout)/(\d+) started \(PID: \d+, GPUs: \[(\d+)\], ports:"
)
_TRAIN_STEP_DONE = re.compile(
    r"StatsLogger INFO: Epoch \d+/\d+ Step \d+/\d+ "
    r"Train step (\d+)/(\d+) done\."
)
_FSDP_ENGINE = re.compile(
    r"Engine 'actor/(\d+)' \(class: "
    r"areal\.engine\.fsdp_engine\.FSDPPPOActor\) instantiated successfully"
)
_FSDP_MESH = re.compile(
    r"\[FSDPEngine Rank (\d+)\] INFO: Initializing device mesh with parallel "
    r"dims \(dp=4, sp=1, tp=1, ep=1, etp=1, world_size=4\)\."
)
_FSDP_MICROBATCH = re.compile(
    r"\[FSDPEngine Rank (\d+)\] INFO: Microbatch #tokens \(rank \1\):"
)
_FSDP_OPTIMIZER = re.compile(
    r"\[FSDPEngine Rank (\d+)\] INFO: Create optimizer time: "
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)[\"']?(admin_api_key|session_api_key)[\"']?[ ]*[:=][ ]*([^\s,}]+)"
)
_RUNTIME_SECRET = re.compile(r"(?<![A-Za-z0-9_])jph-b0-[A-Za-z0-9_-]+")

_REQUIRED_MEMORY_KEYS = {
    "schema_version",
    "run_kind",
    "project_commit",
    "areal_commit",
    "memory_limit_enforced",
    "max_new_memory_mib",
    "gpus",
    "passed",
    "record_sha256",
}
_REQUIRED_GPU_MEMORY_KEYS = {
    "physical_gpu_id",
    "baseline_used_mib",
    "peak_used_mib",
    "peak_delta_mib",
    "sample_count",
    "memory_limit_enforced",
    "max_new_memory_mib",
    "passed",
}
_OPTIONAL_GPU_MEMORY_KEYS = {"peak_free_mib"}
_TEXT_SUFFIXES = {"", ".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}
_ALLOWED_SECRET_VALUES = {
    "",
    "null",
    "none",
    "''",
    '""',
    "<redacted-runtime-admin-key>",
    "<redacted-default-admin-key>",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _record_sha256(record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical_json(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_int(value: object) -> bool:
    return type(value) is int


def _require_git_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git SHA")
    return value


def _require_regular_file(path: Path, *, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe")
    return path


def _read_text(path: Path, *, label: str) -> str:
    _require_regular_file(path, label=label)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} is not UTF-8 text") from exc


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text).replace("\r", "")


def _load_strict_json(path: Path, *, label: str) -> dict[str, object]:
    text = _read_text(path, label=label)
    try:
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _parse_yaml_paths(text: str) -> dict[tuple[str, ...], list[str]]:
    """Parse the scalar mapping paths needed from AReaL's resolved YAML.

    This deliberately is not a general YAML parser.  It accepts the indentation-based
    mapping form emitted by OmegaConf and records scalar values by their full path.
    Required values must occur exactly once later, so aliases, duplicate keys, tabs,
    malformed indentation, and ambiguous structures fail closed.
    """

    values: dict[tuple[str, ...], list[str]] = {}
    stack: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if "\t" in raw_line:
            raise ValueError(f"config.yaml contains a tab at line {line_number}")
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        match = _YAML_MAPPING.fullmatch(raw_line)
        if match is None:
            # OmegaConf emits list items for scheduling specs.  They are irrelevant to
            # the scalar contract and remain nested below the last mapping key.
            if raw_line.lstrip().startswith("- "):
                continue
            continue
        indent = len(match.group(1))
        if indent % 2:
            raise ValueError(f"config.yaml has odd indentation at line {line_number}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        key = match.group(2)
        raw_value = match.group(3)
        path = tuple(item[1] for item in stack) + (key,)
        if raw_value is None or not raw_value.strip():
            stack.append((indent, key))
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values.setdefault(path, []).append(value)
    return values


def _one_yaml_value(
    values: Mapping[tuple[str, ...], list[str]], path: tuple[str, ...]
) -> str:
    found = values.get(path, [])
    if len(found) != 1:
        raise ValueError(f"config.yaml must contain exactly one {'.'.join(path)}")
    return found[0]


def _require_yaml_value(
    values: Mapping[tuple[str, ...], list[str]],
    path: tuple[str, ...],
    expected: str,
) -> None:
    actual = _one_yaml_value(values, path)
    if actual != expected:
        raise ValueError(
            f"config.yaml {'.'.join(path)} differs: expected {expected!r}, got {actual!r}"
        )


def _yaml_float(
    values: Mapping[tuple[str, ...], list[str]], path: tuple[str, ...]
) -> float:
    raw = _one_yaml_value(values, path)
    try:
        result = float(raw)
    except ValueError as exc:
        raise ValueError(f"config.yaml {'.'.join(path)} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"config.yaml {'.'.join(path)} is not finite")
    return result


def _discover_areal_log_dir(root: Path) -> tuple[Path, Path, Path]:
    configs = tuple(
        path
        for path in root.rglob("config.yaml")
        if path.is_file() and not path.is_symlink()
    )
    if len(configs) != 1:
        raise ValueError("RUN_ROOT must contain exactly one safe AReaL config.yaml")
    config_path = configs[0]
    log_dir = config_path.parent
    main_logs = tuple(
        path
        for path in root.rglob("main.log")
        if path.is_file() and not path.is_symlink()
    )
    actor_logs = tuple(
        path
        for path in root.rglob("actor.log")
        if path.is_file() and not path.is_symlink()
    )
    if main_logs != (log_dir / "main.log",) or actor_logs != (log_dir / "actor.log",):
        raise ValueError("RUN_ROOT must contain exactly one AReaL coordinator/actor log pair")
    try:
        relative = log_dir.relative_to(root)
    except ValueError as exc:  # pragma: no cover - guarded by rglob
        raise ValueError("AReaL log directory escapes RUN_ROOT") from exc
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != "logs"
        or not parts[1]
        or parts[2] != "jph-b0"
        or parts[3] != root.name
    ):
        raise ValueError("AReaL log directory does not have the pinned LocalScheduler layout")
    main_log = _require_regular_file(log_dir / "main.log", label="AReaL main.log")
    actor_log = _require_regular_file(log_dir / "actor.log", label="AReaL actor.log")
    return config_path, main_log, actor_log


def _verify_config(config_path: Path, root: Path) -> dict[str, object]:
    text = _read_text(config_path, label="AReaL config.yaml")
    values = _parse_yaml_paths(text)
    expected_values = {
        ("experiment_name",): "jph-b0",
        ("trial_name",): root.name,
        ("cluster", "n_nodes"): "1",
        ("cluster", "n_gpus_per_node"): "8",
        ("scheduler", "type"): "local",
        ("total_train_steps",): "1",
        ("train_dataset", "batch_size"): "8",
        ("valid_dataset", "batch_size"): "8",
        ("gconfig", "n_samples"): "2",
        ("gconfig", "max_new_tokens"): "256",
        ("gconfig", "max_tokens"): "512",
        ("sglang", "mem_fraction_static"): "0.29",
        ("sglang", "context_length"): "1024",
        ("sglang", "max_running_requests"): "2",
        ("saver", "mode"): "sync",
        ("saver", "freq_epochs"): "null",
        ("saver", "freq_steps"): "1",
        ("saver", "freq_secs"): "null",
        ("actor", "backend"): "fsdp:d4p1t1",
        ("actor", "kl_ctl"): "0.0",
        ("rollout", "backend"): "sglang:d4p1t1",
        ("rollout", "max_concurrent_rollouts"): "8",
        ("rollout", "max_head_offpolicyness"): "0",
        ("ref", "backend"): "fsdp:d4p1t1",
        ("ref", "scheduling_strategy", "type"): "colocation",
        ("ref", "scheduling_strategy", "target"): "actor",
    }
    for path, expected in expected_values.items():
        _require_yaml_value(values, path, expected)

    fileroot = Path(_one_yaml_value(values, ("cluster", "fileroot"))).resolve()
    if fileroot != root:
        raise ValueError("config.yaml cluster.fileroot differs from RUN_ROOT")
    warmup = _yaml_float(values, ("actor", "optimizer", "warmup_steps_proportion"))
    if warmup != 0.0:
        raise ValueError("actor optimizer warmup_steps_proportion must be exactly 0.0")

    # ref is a configuration contract only.  With kl_ctl=0, pinned AReaL does not
    # instantiate a reference worker; this verifier intentionally makes no such claim.
    return {
        "experiment_name": "jph-b0",
        "trial_name": root.name,
        "n_nodes": 1,
        "n_gpus_per_node": 8,
        "total_train_steps": 1,
        "actor_backend": "fsdp:d4p1t1",
        "rollout_backend": "sglang:d4p1t1",
        "ref_backend": "fsdp:d4p1t1",
        "ref_colocation_target": "actor",
        "actor_warmup_steps_proportion": warmup,
        "actor_kl_ctl": 0.0,
        "train_batch_size": 8,
        "valid_batch_size": 8,
        "n_samples": 2,
        "max_new_tokens": 256,
        "max_tokens": 512,
        "rollout_max_concurrent": 8,
        "rollout_max_head_offpolicyness": 0,
        "sglang_mem_fraction_static": 0.29,
        "sglang_context_length": 1024,
        "sglang_max_running_requests": 2,
        "saver_mode": "sync",
        "saver_freq_steps": 1,
    }


def _extract_stats(main_text: str) -> dict[str, float]:
    metrics: dict[str, list[float]] = {}
    for raw_line in main_text.splitlines():
        if "│" not in raw_line:
            continue
        fields = raw_line.split("│")
        for index in range(1, len(fields) - 1, 2):
            if index + 1 >= len(fields):
                break
            name = fields[index].strip()
            raw_value = fields[index + 1].strip()
            if not name or _NUMBER.fullmatch(raw_value) is None:
                continue
            value = float(raw_value)
            metrics.setdefault(name, []).append(value)

    required = {
        "ppo_actor/update/update_successful",
        "ppo_actor/update/grad_norm",
        "ppo_actor/update/lr",
        "ppo_actor/update/n_valid_tokens",
        "ppo_actor/update/actor_loss/avg",
        "ppo_actor/advantages/max",
        "ppo_actor/advantages/min",
        "timeperf/train_step",
        "timeperf/update_weights",
    }
    result: dict[str, float] = {}
    for name in sorted(required):
        found = metrics.get(name, [])
        if len(found) != 1:
            raise ValueError(f"AReaL main.log must contain exactly one metric {name}")
        value = found[0]
        if not math.isfinite(value):
            raise ValueError(f"AReaL metric {name} is not finite")
        result[name] = value

    if result["ppo_actor/update/update_successful"] != 1.0:
        raise ValueError("AReaL FSDP optimizer did not report update_successful=1")
    if result["ppo_actor/update/grad_norm"] <= 0.0:
        raise ValueError("AReaL FSDP optimizer grad_norm must be positive")
    if result["ppo_actor/update/lr"] <= 0.0:
        raise ValueError("AReaL FSDP optimizer learning rate must be positive")
    if result["ppo_actor/update/n_valid_tokens"] <= 0.0:
        raise ValueError("AReaL FSDP update must contain valid training tokens")
    if result["ppo_actor/advantages/max"] <= 0.0:
        raise ValueError("AReaL advantages/max must be positive")
    if result["ppo_actor/advantages/min"] >= 0.0:
        raise ValueError("AReaL advantages/min must be negative")
    if result["timeperf/train_step"] <= 0.0:
        raise ValueError("AReaL train_step timing must be positive")
    if result["timeperf/update_weights"] <= 0.0:
        raise ValueError("AReaL update_weights timing must be positive")
    return result


def _verify_main_log(main_log: Path) -> tuple[dict[str, object], dict[str, float]]:
    text = _strip_ansi(_read_text(main_log, label="AReaL main.log"))
    if re.search(r"(?m)\bERROR\b|Traceback \(most recent call last\):", text):
        raise ValueError("AReaL main.log contains a fatal error marker")

    scheduler_marker = (
        "LocalScheduler initialized with GPU devices: [0, 1, 2, 3, 4, 5, 6, 7]"
    )
    if text.count(scheduler_marker) != 1:
        raise ValueError("AReaL main.log must prove exactly one 8-GPU LocalScheduler")
    if text.count("TrainController initialization complete") != 1:
        raise ValueError("AReaL main.log must prove exactly one TrainController")
    for role in ("actor", "rollout"):
        marker = f"Creating 4 workers for role '{role}'"
        if text.count(marker) != 1:
            raise ValueError(f"AReaL main.log must create exactly four {role} workers")

    assignments: dict[str, dict[int, int]] = {"actor": {}, "rollout": {}}
    for role, raw_rank, raw_gpu in _WORKER_STARTED.findall(text):
        rank = int(raw_rank)
        gpu = int(raw_gpu)
        if rank in assignments[role]:
            raise ValueError(f"AReaL main.log repeats {role}/{rank} GPU assignment")
        assignments[role][rank] = gpu
    for role in ("actor", "rollout"):
        if set(assignments[role]) != set(range(4)):
            raise ValueError(f"AReaL main.log {role} ranks differ from 0..3")
        if len(set(assignments[role].values())) != 4:
            raise ValueError(f"AReaL main.log {role} GPU assignments are not unique")
    actor_gpus = set(assignments["actor"].values())
    rollout_gpus = set(assignments["rollout"].values())
    if actor_gpus & rollout_gpus or actor_gpus | rollout_gpus != set(EXPECTED_GPU_IDS):
        raise ValueError("AReaL actor/rollout workers do not occupy eight disjoint GPUs")

    train_steps = _TRAIN_STEP_DONE.findall(text)
    if len(train_steps) != 1 or train_steps[0][0] != "1":
        raise ValueError("AReaL main.log must contain exactly one completed train step")
    if text.count("StatsLogger INFO: Stats (1/1):") != 1:
        raise ValueError("AReaL main.log must contain exactly one committed stats table")
    if text.count("StatsLogger INFO: Training completes!") != 1:
        raise ValueError("AReaL main.log does not prove clean training completion")
    if text.count("TrainController destroyed") != 1:
        raise ValueError("AReaL main.log does not prove TrainController cleanup")

    step_position = text.find("Train step 1/")
    completion_position = text.find("StatsLogger INFO: Training completes!")
    destroy_position = text.find("TrainController destroyed")
    if not 0 <= step_position < completion_position < destroy_position:
        raise ValueError("AReaL completion markers are out of order")

    stats = _extract_stats(text)
    topology: dict[str, object] = {
        "single_coordinator": True,
        "scheduler_gpu_ids": list(EXPECTED_GPU_IDS),
        "actor_rank_to_gpu": {
            str(rank): assignments["actor"][rank] for rank in range(4)
        },
        "rollout_rank_to_gpu": {
            str(rank): assignments["rollout"][rank] for rank in range(4)
        },
    }
    return topology, stats


def _verify_actor_log(actor_log: Path) -> dict[str, object]:
    text = _strip_ansi(_read_text(actor_log, label="AReaL actor.log"))
    if re.search(r"(?m)\bERROR\b|Traceback \(most recent call last\):", text):
        raise ValueError("AReaL actor.log contains a fatal error marker")

    engines = tuple(int(rank) for rank in _FSDP_ENGINE.findall(text))
    meshes = tuple(int(rank) for rank in _FSDP_MESH.findall(text))
    optimizers = tuple(
        (int(rank), float(duration))
        for rank, duration in _FSDP_OPTIMIZER.findall(text)
    )
    microbatch_ranks = {int(rank) for rank in _FSDP_MICROBATCH.findall(text)}
    if sorted(engines) != list(range(4)) or len(engines) != 4:
        raise ValueError("AReaL actor.log does not prove four real FSDPPPOActor engines")
    if sorted(meshes) != list(range(4)) or len(meshes) != 4:
        raise ValueError("AReaL actor.log does not prove the fsdp:d4 device mesh")
    if sorted(rank for rank, _ in optimizers) != list(range(4)) or len(optimizers) != 4:
        raise ValueError("AReaL actor.log does not prove four FSDP optimizers")
    if any(not math.isfinite(duration) or duration <= 0 for _, duration in optimizers):
        raise ValueError("AReaL actor optimizer initialization timing is invalid")
    if microbatch_ranks != set(range(4)):
        raise ValueError("AReaL actor.log lacks training microbatch activity on every rank")
    return {
        "engine_class": "areal.engine.fsdp_engine.FSDPPPOActor",
        "fsdp_world_size": 4,
        "active_ranks": list(range(4)),
    }


def _verify_memory_audit(
    path: Path, *, expected_project_commit: str, expected_areal_commit: str
) -> tuple[dict[str, object], tuple[dict[str, int], ...]]:
    if path.stat().st_mode & 0o077:
        raise ValueError("gpu-memory-audit.json must be private")
    record = _load_strict_json(path, label="gpu-memory-audit.json")
    if set(record) != _REQUIRED_MEMORY_KEYS:
        raise ValueError("gpu-memory-audit.json field set differs")
    if record.get("record_sha256") != _record_sha256(record):
        raise ValueError("gpu-memory-audit.json record SHA-256 differs")
    if (
        record.get("schema_version") != MEMORY_AUDIT_SCHEMA
        or record.get("run_kind") != MEMORY_RUN_KIND
        or record.get("project_commit") != expected_project_commit
        or record.get("areal_commit") != expected_areal_commit
        or record.get("memory_limit_enforced") is not False
        or record.get("max_new_memory_mib") is not None
        or record.get("passed") is not True
    ):
        raise ValueError("gpu-memory-audit.json run contract or result differs")
    raw_gpus = record.get("gpus")
    if not isinstance(raw_gpus, list) or len(raw_gpus) != 8:
        raise ValueError("gpu-memory-audit.json must contain exactly eight GPU records")

    summaries: list[dict[str, int]] = []
    seen: set[int] = set()
    for raw in raw_gpus:
        if not isinstance(raw, dict):
            raise TypeError("GPU memory audit entry must be an object")
        keys = set(raw)
        if (
            not _REQUIRED_GPU_MEMORY_KEYS <= keys
            or (keys - _REQUIRED_GPU_MEMORY_KEYS) - _OPTIONAL_GPU_MEMORY_KEYS
        ):
            raise ValueError("GPU memory audit entry field set differs")
        gpu_id = raw.get("physical_gpu_id")
        baseline = raw.get("baseline_used_mib")
        peak = raw.get("peak_used_mib")
        delta = raw.get("peak_delta_mib")
        sample_count = raw.get("sample_count")
        if not all(_is_int(value) for value in (gpu_id, baseline, peak, delta, sample_count)):
            raise TypeError("GPU memory audit numeric fields must be integers")
        assert isinstance(gpu_id, int)
        assert isinstance(baseline, int)
        assert isinstance(peak, int)
        assert isinstance(delta, int)
        assert isinstance(sample_count, int)
        if gpu_id in seen:
            raise ValueError("GPU memory audit repeats a physical GPU")
        seen.add(gpu_id)
        if baseline < 0 or peak < baseline or delta != peak - baseline:
            raise ValueError(f"GPU {gpu_id} baseline/peak/delta are inconsistent")
        if sample_count <= 0:
            raise ValueError(f"GPU {gpu_id} memory audit has no samples")
        if (
            raw.get("memory_limit_enforced") is not False
            or raw.get("max_new_memory_mib") is not None
            or raw.get("passed") is not True
        ):
            raise ValueError(f"GPU {gpu_id} memory observation failed")
        if "peak_free_mib" in raw:
            peak_free = raw["peak_free_mib"]
            if not _is_int(peak_free) or peak_free < 0:
                raise ValueError(f"GPU {gpu_id} peak_free_mib is invalid")
        summaries.append(
            {
                "physical_gpu_id": gpu_id,
                "baseline_used_mib": baseline,
                "peak_used_mib": peak,
                "peak_delta_mib": delta,
                "sample_count": sample_count,
            }
        )
    if seen != set(EXPECTED_GPU_IDS):
        raise ValueError("GPU memory audit IDs differ from 0..7")
    summaries.sort(key=lambda item: item["physical_gpu_id"])
    return record, tuple(summaries)


def _verify_checkpoint(root: Path) -> dict[str, object]:
    checkpoints = tuple(
        path
        for path in root.glob(
            "checkpoints/*/jph-b0/*/default/epoch0epochstep0globalstep0"
        )
        if path.is_dir() and not path.is_symlink()
    )
    if len(checkpoints) != 1:
        raise ValueError("RUN_ROOT must contain exactly one post-update step-0 HF checkpoint")
    checkpoint = checkpoints[0]
    relative = checkpoint.relative_to(root)
    if relative.parts[-3] != root.name:
        raise ValueError("post-update checkpoint trial does not match RUN_ROOT")

    config_path = _require_regular_file(checkpoint / "config.json", label="HF config")
    config = _load_strict_json(config_path, label="HF config")
    if not config:
        raise ValueError("HF checkpoint config.json is empty")
    tokenizer = _require_regular_file(
        checkpoint / "tokenizer.json", label="HF tokenizer.json"
    )
    tokenizer_config = _require_regular_file(
        checkpoint / "tokenizer_config.json", label="HF tokenizer_config.json"
    )
    if tokenizer.stat().st_size <= 0 or tokenizer_config.stat().st_size <= 0:
        raise ValueError("HF checkpoint tokenizer files are empty")

    single_model = checkpoint / "model.safetensors"
    index_path = checkpoint / "model.safetensors.index.json"
    model_files: list[Path] = []
    if single_model.is_file() and not single_model.is_symlink() and not index_path.exists():
        model_files = [single_model]
    elif index_path.is_file() and not index_path.is_symlink() and not single_model.exists():
        index = _load_strict_json(index_path, label="HF safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("HF safetensors index has no weight_map")
        shard_names = set(weight_map.values())
        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in shard_names
        ):
            raise ValueError("HF safetensors index contains an unsafe shard path")
        model_files = [
            _require_regular_file(checkpoint / str(name), label="HF model shard")
            for name in sorted(shard_names)
        ]
    else:
        raise ValueError("HF checkpoint must contain one safetensors layout")
    if any(path.stat().st_size <= 0 for path in model_files):
        raise ValueError("HF checkpoint contains an empty model file")
    return {
        "relative_path": str(relative),
        "format": "huggingface-safetensors",
        "model_file_count": len(model_files),
        "config_sha256": _file_sha256(config_path),
        "model_sha256": [_file_sha256(path) for path in model_files],
    }


def _verify_provenance_marker(
    run_log: Path,
    *,
    root: Path,
    expected_project_commit: str,
    expected_areal_commit: str,
) -> None:
    text = _strip_ansi(_read_text(run_log, label="B0 wrapper run log"))
    matches: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        project = re.search(r"(?:^|\s)(?:project|project_commit)=([0-9a-f]{40})(?:\s|$)", line)
        areal = re.search(r"(?:^|\s)(?:AReaL|areal_commit)=([0-9a-f]{40})(?:\s|$)", line)
        gpus = re.search(r"(?:^|\s)GPUs=([^\s]+)(?:\s|$)", line)
        run_root = re.search(r"(?:^|\s)run_root=([^\s]+)(?:\s|$)", line)
        if all(match is not None for match in (project, areal, gpus, run_root)):
            assert project is not None and areal is not None
            assert gpus is not None and run_root is not None
            matches.append(
                (project.group(1), areal.group(1), gpus.group(1), run_root.group(1))
            )
    if len(matches) != 1:
        raise ValueError("wrapper run log must contain exactly one full provenance marker")
    project, areal, gpus, marker_root = matches[0]
    if project != expected_project_commit or areal != expected_areal_commit:
        raise ValueError("wrapper run log commit provenance differs")
    if gpus != ",".join(str(gpu) for gpu in EXPECTED_GPU_IDS):
        raise ValueError("wrapper run log GPU provenance differs")
    if Path(marker_root).resolve() != root:
        raise ValueError("wrapper run log RUN_ROOT provenance differs")


def _normalized_secret_value(raw: str) -> str:
    value = raw.strip().rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.lower()


def _scan_for_secrets(root: Path, run_log: Path) -> None:
    paths = [run_log]
    paths.extend(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES
    )
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_symlink():
            raise ValueError(f"text evidence contains an unsafe symlink: {path}")
        text = _read_text(path, label=f"text evidence {path}")
        if "areal-admin-key" in text or _RUNTIME_SECRET.search(text):
            raise ValueError(f"unredacted runtime credential entered evidence: {path}")
        for match in _SENSITIVE_ASSIGNMENT.finditer(text):
            value = _normalized_secret_value(match.group(2))
            if value not in _ALLOWED_SECRET_VALUES:
                raise ValueError(
                    f"unredacted {match.group(1)} entered text evidence: {path}"
                )


def verify_areal_official_b0(
    *,
    run_root: str | Path,
    run_log: str | Path,
    expected_project_commit: str,
    expected_areal_commit: str = PINNED_AREAL_COMMIT,
) -> dict[str, object]:
    expected_project_commit = _require_git_sha(
        expected_project_commit, label="expected project commit"
    )
    expected_areal_commit = _require_git_sha(
        expected_areal_commit, label="expected AReaL commit"
    )
    if expected_areal_commit != PINNED_AREAL_COMMIT:
        raise ValueError("expected AReaL commit differs from the pinned B0 commit")

    raw_root = Path(run_root).expanduser()
    if raw_root.is_symlink():
        raise ValueError("RUN_ROOT is missing or unsafe")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError("RUN_ROOT is missing or unsafe")
    raw_wrapper_log = Path(run_log).expanduser()
    if raw_wrapper_log.is_symlink():
        raise ValueError("B0 wrapper run log is missing or unsafe")
    wrapper_log = raw_wrapper_log.resolve()
    _require_regular_file(wrapper_log, label="B0 wrapper run log")

    config_path, main_log, actor_log = _discover_areal_log_dir(root)
    config = _verify_config(config_path, root)
    topology, stats = _verify_main_log(main_log)
    actor = _verify_actor_log(actor_log)
    _verify_provenance_marker(
        wrapper_log,
        root=root,
        expected_project_commit=expected_project_commit,
        expected_areal_commit=expected_areal_commit,
    )
    memory, gpu_summaries = _verify_memory_audit(
        _require_regular_file(
            root / "gpu-memory-audit.json", label="gpu-memory-audit.json"
        ),
        expected_project_commit=expected_project_commit,
        expected_areal_commit=expected_areal_commit,
    )
    checkpoint = _verify_checkpoint(root)
    _scan_for_secrets(root, wrapper_log)

    training = {
        "completed_train_steps": 1,
        "policy_optimizer_evidence_source": (
            "pinned AReaL StatsLogger export of FSDPEngine.optimizer_step"
        ),
        "update_successful": stats["ppo_actor/update/update_successful"],
        "learning_rate": stats["ppo_actor/update/lr"],
        "grad_norm": stats["ppo_actor/update/grad_norm"],
        "n_valid_tokens": stats["ppo_actor/update/n_valid_tokens"],
        "actor_loss_avg": stats["ppo_actor/update/actor_loss/avg"],
        "advantages_max": stats["ppo_actor/advantages/max"],
        "advantages_min": stats["ppo_actor/advantages/min"],
        "train_step_seconds": stats["timeperf/train_step"],
        "update_weights_seconds": stats["timeperf/update_weights"],
        "post_update_checkpoint": checkpoint,
    }
    report: dict[str, object] = {
        "schema_version": VERIFICATION_SCHEMA,
        "run_root": str(root),
        "project_commit": expected_project_commit,
        "areal_commit": expected_areal_commit,
        "scope": "official-areal-policy-only-b0",
        "config": config,
        "topology": {**topology, **actor},
        "training": training,
        "memory": {
            "memory_limit_enforced": False,
            "max_new_memory_mib": None,
            "gpus": list(gpu_summaries),
        },
        "claims": {
            "policy_optimizer_update": True,
            "harness_optimizer_update": False,
            "harness_optimizer_evidence": False,
            "ref_worker_instantiation_evidence": False,
        },
        "evidence_sha256": {
            "wrapper_run_log": _file_sha256(wrapper_log),
            "config": _file_sha256(config_path),
            "main_log": _file_sha256(main_log),
            "actor_log": _file_sha256(actor_log),
            "gpu_memory_audit": memory["record_sha256"],
        },
        "passed": True,
    }
    report["record_sha256"] = _record_sha256(report)
    return report


def _write_new_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--expected-project-commit", required=True)
    parser.add_argument("--expected-areal-commit", default=PINNED_AREAL_COMMIT)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = verify_areal_official_b0(
        run_root=args.run_root,
        run_log=args.run_log,
        expected_project_commit=args.expected_project_commit,
        expected_areal_commit=args.expected_areal_commit,
    )
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        root = Path(args.run_root).expanduser().resolve()
        if output.parent != root:
            raise ValueError("verification output must remain directly under RUN_ROOT")
        _write_new_json(output, report)
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
