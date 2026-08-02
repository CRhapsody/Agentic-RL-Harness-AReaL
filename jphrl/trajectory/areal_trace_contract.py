from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "jph.areal-model-response-roundtrip.v1"
REQUIRED_TENSOR_FIELDS = (
    "input_ids",
    "loss_mask",
    "logprobs",
    "versions",
    "attention_mask",
    "rewards",
    "original_rewards",
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


class ArealTraceContractError(ValueError):
    """Raised when an AReaL trace does not satisfy the token contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealTraceContractError(message)


def _plain(value: Any) -> Any:
    """Convert a tensor-like value to JSON-compatible Python containers."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _canonical_payload(record: Mapping[str, Any]) -> bytes:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(record)).hexdigest()


def build_areal_trace_record(
    *,
    task_id: int,
    request_id: str,
    model_response: Any,
    interaction: Any,
    tensor_dict: Mapping[str, Any],
    areal_commit: str,
    behavior_snapshot_path: str,
    behavior_revision: str,
) -> dict[str, Any]:
    """Build a trace from AReaL's real ModelResponse and tensor roundtrip."""

    tensors = {key: _plain(value) for key, value in tensor_dict.items()}
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "origin": {
            "generation_path": "areal.api.ModelResponse",
            "roundtrip_path": (
                "areal.experimental.openai.types."
                "InteractionWithTokenLogpReward.to_tensor_dict"
            ),
            "areal_commit": areal_commit,
            "behavior_snapshot_path": behavior_snapshot_path,
            "behavior_revision": behavior_revision,
        },
        "task_id": task_id,
        "request_id": request_id,
        "model_response": {
            "input_tokens": list(model_response.input_tokens),
            "output_tokens": list(model_response.output_tokens),
            "output_logprobs": list(model_response.output_logprobs),
            "output_versions": list(model_response.output_versions),
            "stop_reason": model_response.stop_reason,
        },
        "interaction": {
            "reward": float(interaction.reward),
            "original_reward": float(interaction.original_reward),
            "chat_template_type": interaction.chat_template_type,
        },
        "tensor_dict": tensors,
    }
    record["record_sha256"] = record_sha256(record)
    validate_areal_trace_record(record)
    return record


def _single_batch_vector(
    tensor_dict: Mapping[str, Any], key: str, *, allow_scalar_batch: bool = False
) -> list[Any]:
    value = tensor_dict.get(key)
    _require(isinstance(value, list), f"tensor_dict.{key} must be a list")
    if allow_scalar_batch:
        return value
    _require(len(value) == 1, f"tensor_dict.{key} must have batch size 1")
    row = value[0]
    _require(isinstance(row, list), f"tensor_dict.{key}[0] must be a list")
    return row


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_areal_trace_record(
    record: Mapping[str, Any], *, expected_policy_version: int | None = None
) -> dict[str, Any]:
    """Fail closed unless the source response and tensor data roundtrip exactly."""

    _require(record.get("schema_version") == SCHEMA_VERSION, "unknown schema_version")
    _require(record.get("record_sha256") == record_sha256(record), "record hash mismatch")

    origin = record.get("origin")
    _require(isinstance(origin, dict), "origin must be an object")
    _require(origin.get("generation_path") == "areal.api.ModelResponse", "wrong generation path")
    _require(
        origin.get("roundtrip_path")
        == "areal.experimental.openai.types.InteractionWithTokenLogpReward.to_tensor_dict",
        "wrong roundtrip path",
    )
    for key in ("areal_commit", "behavior_snapshot_path", "behavior_revision"):
        _require(isinstance(origin.get(key), str) and origin[key], f"origin.{key} is missing")

    _require(_is_int(record.get("task_id")), "task_id must be an integer")
    request_id = record.get("request_id")
    _require(isinstance(request_id, str) and request_id, "request_id is missing")

    response = record.get("model_response")
    _require(isinstance(response, dict), "model_response must be an object")
    input_tokens = response.get("input_tokens")
    output_tokens = response.get("output_tokens")
    output_logprobs = response.get("output_logprobs")
    output_versions = response.get("output_versions")
    _require(isinstance(input_tokens, list) and input_tokens, "input_tokens must be non-empty")
    _require(isinstance(output_tokens, list) and output_tokens, "output_tokens must be non-empty")
    _require(isinstance(output_logprobs, list), "output_logprobs must be a list")
    _require(isinstance(output_versions, list), "output_versions must be a list")
    _require(
        len(output_tokens) == len(output_logprobs) == len(output_versions),
        "output token/logprob/version lengths differ",
    )
    _require(
        all(_is_int(token) and token >= 0 for token in input_tokens + output_tokens),
        "token ids must be non-negative integers",
    )
    _require(
        all(_is_finite_number(logprob) and float(logprob) <= 1e-5 for logprob in output_logprobs),
        "output logprobs must be finite and non-positive",
    )
    _require(
        all(_is_int(version) and version >= 0 for version in output_versions),
        "output versions must be non-negative integers",
    )

    tensor_dict = record.get("tensor_dict")
    _require(isinstance(tensor_dict, dict), "tensor_dict must be an object")
    missing = [key for key in REQUIRED_TENSOR_FIELDS if key not in tensor_dict]
    _require(not missing, f"tensor_dict is missing fields: {missing}")

    ids = _single_batch_vector(tensor_dict, "input_ids")
    loss_mask = _single_batch_vector(tensor_dict, "loss_mask")
    logprobs = _single_batch_vector(tensor_dict, "logprobs")
    versions = _single_batch_vector(tensor_dict, "versions")
    attention_mask = _single_batch_vector(tensor_dict, "attention_mask")
    rewards = _single_batch_vector(tensor_dict, "rewards", allow_scalar_batch=True)
    original_rewards = _single_batch_vector(
        tensor_dict, "original_rewards", allow_scalar_batch=True
    )

    expected_ids = input_tokens + output_tokens
    prompt_len = len(input_tokens)
    output_len = len(output_tokens)
    total_len = prompt_len + output_len
    for key, value in (
        ("input_ids", ids),
        ("loss_mask", loss_mask),
        ("logprobs", logprobs),
        ("versions", versions),
        ("attention_mask", attention_mask),
    ):
        _require(len(value) == total_len, f"tensor_dict.{key} has the wrong sequence length")

    _require(ids == expected_ids, "input_ids do not equal input_tokens + output_tokens")
    _require(
        loss_mask == [0] * prompt_len + [1] * output_len,
        "loss_mask does not mark prompt=0 and generation=1",
    )
    _require(
        all(_is_finite_number(value) for value in logprobs),
        "tensor logprobs contain non-finite values",
    )
    _require(
        all(float(value) == 0.0 for value in logprobs[:prompt_len]),
        "prompt logprobs must be zero",
    )
    roundtrip_errors = [
        abs(float(tensor_value) - float(response_value))
        for tensor_value, response_value in zip(
            logprobs[prompt_len:], output_logprobs
        )
    ]
    roundtrip_max_abs_error = max(roundtrip_errors)
    _require(
        all(
            math.isclose(
                float(tensor_value),
                float(response_value),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            for tensor_value, response_value in zip(
                logprobs[prompt_len:], output_logprobs
            )
        ),
        "generated tensor logprobs differ from ModelResponse beyond float32 tolerance",
    )
    _require(
        versions == [-1] * prompt_len + output_versions,
        "versions do not mark prompt=-1 and generation=output_versions",
    )
    _require(
        all(value is True or value == 1 for value in attention_mask),
        "attention_mask must be active for every unpadded token",
    )
    _require(
        len(rewards) == 1 and _is_finite_number(rewards[0]),
        "rewards must contain one finite value",
    )
    _require(
        len(original_rewards) == 1 and _is_finite_number(original_rewards[0]),
        "original_rewards must contain one finite value",
    )

    interaction = record.get("interaction")
    _require(isinstance(interaction, dict), "interaction must be an object")
    _require(
        float(rewards[0]) == float(interaction.get("reward")),
        "tensor reward differs from interaction reward",
    )
    _require(
        float(original_rewards[0]) == float(interaction.get("original_reward")),
        "tensor original_reward differs from interaction original_reward",
    )

    unique_versions = sorted(set(output_versions))
    _require(len(unique_versions) == 1, "single-turn trace contains mixed output versions")
    if expected_policy_version is not None:
        _require(
            unique_versions == [expected_policy_version],
            f"expected policy version {expected_policy_version}, got {unique_versions}",
        )

    return {
        "ok": True,
        "task_id": record["task_id"],
        "request_id": request_id,
        "prompt_tokens": prompt_len,
        "generated_tokens": output_len,
        "policy_versions": unique_versions,
        "reward": float(rewards[0]),
        "roundtrip_logprob_max_abs_error": roundtrip_max_abs_error,
        "roundtrip_logprob_tolerance": {"atol": 1e-6, "rtol": 1e-6},
        "record_sha256": record["record_sha256"],
    }


def _require_within(path: Path, root: Path) -> None:
    try:
        common = Path(os.path.commonpath((path, root)))
    except ValueError as exc:
        raise ArealTraceContractError(f"cannot compare {path} and {root}") from exc
    _require(common == root, f"path escapes configured root: {path}")


def write_areal_trace_record(
    record: Mapping[str, Any], *, trace_dir: str | Path, allowed_root: str | Path
) -> Path:
    """Write one unique, mode-0600 trace under the configured experiment root."""

    validate_areal_trace_record(record)
    root = Path(allowed_root).expanduser().resolve()
    directory = Path(trace_dir).expanduser().resolve()
    _require(root.is_dir(), f"configured root does not exist: {root}")
    _require_within(directory, root)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    task_component = str(record["task_id"])
    request_component = str(record["request_id"])
    _require(_SAFE_COMPONENT.fullmatch(task_component) is not None, "unsafe task id")
    _require(_SAFE_COMPONENT.fullmatch(request_component) is not None, "unsafe request id")
    version = validate_areal_trace_record(record)["policy_versions"][0]
    path = directory / f"trace-v{version}-task{task_component}-{request_component}.json"
    _require_within(path.resolve(), root)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    return path
