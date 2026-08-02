from __future__ import annotations

import math
from typing import Any


def _require_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def validate_token_metadata(
    *,
    input_token_ids: Any,
    output_token_ids: Any,
    output_token_logprobs: Any,
    completion_loss_mask: Any,
    policy_kind: Any,
    token_metadata_status: Any,
) -> None:
    input_ids = _require_list("input_token_ids", input_token_ids)
    output_ids = _require_list("output_token_ids", output_token_ids)
    logprobs = _require_list("output_token_logprobs", output_token_logprobs)
    loss_mask = _require_list("completion_loss_mask", completion_loss_mask)

    if token_metadata_status == "not_applicable":
        if policy_kind != "scripted":
            raise ValueError("only a scripted policy may mark token metadata not applicable")
        if input_ids or output_ids or logprobs or loss_mask:
            raise ValueError("scripted policy must not fabricate token metadata")
        return
    if token_metadata_status != "available":
        raise ValueError(f"unknown token metadata status: {token_metadata_status}")

    if not isinstance(policy_kind, str) or not policy_kind or policy_kind == "scripted":
        raise ValueError("available token metadata requires a non-scripted policy kind")
    if not input_ids:
        raise ValueError("available prompt token metadata cannot be empty")
    if not output_ids:
        raise ValueError("available completion token metadata cannot be empty")
    if not (len(output_ids) == len(logprobs) == len(loss_mask)):
        raise ValueError("token IDs, old log-probs, and loss mask have different lengths")

    for token_id in [*input_ids, *output_ids]:
        if type(token_id) is not int or token_id < 0:
            raise ValueError("token IDs must be non-negative integers")
    for logprob in logprobs:
        if type(logprob) not in (int, float) or not math.isfinite(float(logprob)):
            raise ValueError("old token log-probs must be finite numbers")
        if float(logprob) > 0.0:
            raise ValueError("old token log-probs cannot be positive")
    for mask_value in loss_mask:
        if type(mask_value) is not int or mask_value not in (0, 1):
            raise ValueError("completion loss mask must contain only integer 0 or 1")
