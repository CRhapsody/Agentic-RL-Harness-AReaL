from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Mapping

from jphrl.experiments.sglang_logprob_screen import (
    C0_MODE,
    MAX_IMPORTANCE_RATIO_ERROR,
    MAX_MEAN_IMPORTANCE_RATIO_ERROR,
    MAX_PAIRED_RESCORED_LOGPROB_ABS_DELTA,
    MIN_PAIRED_STORED_LOGPROB_ABS_DELTA,
    SGLANG_VERSION,
    _load_cell,
    _paired_score_deltas,
    _ratio_metrics,
    _record_sha256,
    _single_row,
    paired_generation_equal,
    write_screen_report,
)


CUDA_GRAPH_SCREEN_SCHEMA_VERSION = "jph.sglang-cuda-graph-screen.v1"
CUDA_GRAPH_DATASET_SELECTION = "sequential-valid-offset64-count4-v1"
CUDA_GRAPH_AXIS = "cuda-graph-v1"
C2A_TREATMENT = {
    "disable_cuda_graph": False,
    "experimental_axis": CUDA_GRAPH_AXIS,
    "generation_logprob_mode": C0_MODE,
    "sglang_return_original_logprob": False,
}
C2B_TREATMENT = {
    "disable_cuda_graph": True,
    "experimental_axis": CUDA_GRAPH_AXIS,
    "generation_logprob_mode": C0_MODE,
    "sglang_return_original_logprob": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _runtime_invariants(runtime: Mapping[str, object]) -> dict[str, object]:
    """Remove only the pre-registered CUDA Graph treatment from fixed fields."""

    fixed = copy.deepcopy(runtime["fixed"])
    _require(isinstance(fixed, dict), "runtime fixed fields are missing")
    server_args = fixed.get("server_args")
    _require(isinstance(server_args, dict), "runtime server args are missing")
    _require(
        type(server_args.get("disable_cuda_graph")) is bool,
        "runtime server args do not declare disable_cuda_graph",
    )
    del server_args["disable_cuda_graph"]
    return fixed


def _unaligned_score_diagnostic() -> dict[str, object]:
    return {
        "score_alignment_equal": False,
        "max_rescored_logprob_abs_delta": None,
        "rescored_logprobs_within_tolerance": False,
        "max_stored_logprob_abs_delta": None,
        "stored_logprobs_changed": False,
    }


def compare_cuda_graph_screen_runs(
    c2a_root: Path,
    c2b_root: Path,
) -> dict[str, object]:
    """Compare CUDA Graph enabled/disabled cells on an exact paired slice."""

    configured_root = Path(os.environ["JPH_ROOT"]).resolve()
    c2a_root = c2a_root.resolve()
    c2b_root = c2b_root.resolve()
    _require(c2a_root.is_relative_to(configured_root), "C2a root escapes JPH_ROOT")
    _require(c2b_root.is_relative_to(configured_root), "C2b root escapes JPH_ROOT")
    c2a_cell = _load_cell(
        c2a_root,
        expected_mode=C0_MODE,
        expected_dataset_selection=CUDA_GRAPH_DATASET_SELECTION,
        expected_treatment=C2A_TREATMENT,
    )
    c2b_cell = _load_cell(
        c2b_root,
        expected_mode=C0_MODE,
        expected_dataset_selection=CUDA_GRAPH_DATASET_SELECTION,
        expected_treatment=C2B_TREATMENT,
    )
    c2a = c2a_cell["by_prompt"]
    c2b = c2b_cell["by_prompt"]
    _require(isinstance(c2a, dict) and isinstance(c2b, dict), "invalid C2 cell")
    _require(set(c2a) == set(c2b), "C2a and C2b selected different base prompts")

    left_runtime = c2a_cell["runtime_contract"]
    right_runtime = c2b_cell["runtime_contract"]
    _require(
        isinstance(left_runtime, dict) and isinstance(right_runtime, dict),
        "C2 runtime contracts are missing",
    )
    pair_id = left_runtime["identity"]["screen_pair_id"]
    runtime_invariants_equal = bool(
        isinstance(pair_id, str)
        and pair_id
        and pair_id == right_runtime["identity"]["screen_pair_id"]
        and left_runtime["schema_version"] == "jph.sglang-inference-runtime.v2"
        and right_runtime["schema_version"] == "jph.sglang-inference-runtime.v2"
        and _runtime_invariants(left_runtime) == _runtime_invariants(right_runtime)
        and left_runtime["treatment"] == C2A_TREATMENT
        and right_runtime["treatment"] == C2B_TREATMENT
    )

    traces: list[dict[str, object]] = []
    all_generation_equal = True
    all_score_alignment_equal = True
    all_rescored_equal = True
    any_stored_changed = False
    all_c2b_non_worse = True
    any_c2b_strictly_better = False
    for prompt_key in sorted(c2a):
        left = c2a[prompt_key]
        right = c2b[prompt_key]
        generation_equal = paired_generation_equal(
            left["bridge"],
            right["bridge"],
        )
        all_generation_equal = all_generation_equal and generation_equal
        score_alignment_equal = all(
            left["score"].get(field) == right["score"].get(field)
            for field in ("input_ids", "loss_mask", "versions")
        )
        if score_alignment_equal:
            paired_score = _paired_score_deltas(left["score"], right["score"])
            paired_score["score_alignment_equal"] = True
            common_rescored_target = _single_row(
                left["score"]["rescored_logprobs"],
                "C2a common rescored target",
            )
            right_metrics: dict[str, object] | None = _ratio_metrics(
                right["score"],
                rescored_override=common_rescored_target,
            )
            left_metrics = left["metrics"]
            mean_non_worse = bool(
                right_metrics["mean_importance_ratio_error"]
                <= left_metrics["mean_importance_ratio_error"] + 1e-12
            )
            max_non_worse = bool(
                right_metrics["max_importance_ratio_error"]
                <= left_metrics["max_importance_ratio_error"] + 1e-12
            )
            strictly_better = bool(paired_score["stored_logprobs_changed"]) and (
                right_metrics["mean_importance_ratio_error"]
                < left_metrics["mean_importance_ratio_error"] - 1e-12
                or right_metrics["max_importance_ratio_error"]
                < left_metrics["max_importance_ratio_error"] - 1e-12
            )
        else:
            paired_score = _unaligned_score_diagnostic()
            right_metrics = None
            mean_non_worse = False
            max_non_worse = False
            strictly_better = False
        all_score_alignment_equal = (
            all_score_alignment_equal and score_alignment_equal
        )
        all_rescored_equal = all_rescored_equal and bool(
            paired_score["rescored_logprobs_within_tolerance"]
        )
        any_stored_changed = any_stored_changed or bool(
            paired_score["stored_logprobs_changed"]
        )
        all_c2b_non_worse = (
            all_c2b_non_worse and mean_non_worse and max_non_worse
        )
        any_c2b_strictly_better = any_c2b_strictly_better or strictly_better
        traces.append(
            {
                "base_messages_sha256": prompt_key,
                "generation_equal": generation_equal,
                "paired_score": paired_score,
                "c2a": left["metrics"],
                "c2b": right_metrics,
                "c2b_observed_rescored_target_diagnostic": right["metrics"],
            }
        )

    c2b_all_passed = all(
        isinstance(item["c2b"], dict)
        and bool(item["c2b"]["passed_pre_registered_gate"])
        for item in traces
    )
    mechanism_supported = bool(
        all_generation_equal
        and all_score_alignment_equal
        and runtime_invariants_equal
        and all_rescored_equal
        and any_stored_changed
        and all_c2b_non_worse
        and any_c2b_strictly_better
        and c2b_all_passed
    )
    invariant_fields = _runtime_invariants(left_runtime)
    report: dict[str, object] = {
        "schema_version": CUDA_GRAPH_SCREEN_SCHEMA_VERSION,
        "experiment": "sglang-cuda-graph-logprob-screen",
        "screen_pair_id": pair_id,
        "dataset_selection": CUDA_GRAPH_DATASET_SELECTION,
        "sglang_version": SGLANG_VERSION,
        "runtime_invariants_sha256": _record_sha256(
            {"invariants": invariant_fields}
        ),
        "cells": {
            "c2a": {
                "disable_cuda_graph": False,
                "run_root": str(c2a_root),
                "inference_runtime_contract_sha256": c2a_cell[
                    "runtime_contract_sha256"
                ],
            },
            "c2b": {
                "disable_cuda_graph": True,
                "run_root": str(c2b_root),
                "inference_runtime_contract_sha256": c2b_cell[
                    "runtime_contract_sha256"
                ],
            },
        },
        "pre_registered_gate": {
            "max_trace_mean_importance_ratio_error": (
                MAX_MEAN_IMPORTANCE_RATIO_ERROR
            ),
            "max_token_importance_ratio_error": MAX_IMPORTANCE_RATIO_ERROR,
            "paired_generation_must_be_identical": True,
            "paired_score_alignment_must_be_identical": True,
            "paired_runtime_invariants_must_be_identical": True,
            "only_treatment_difference": "disable_cuda_graph:false-to-true",
            "paired_metric_target": "c2a-rescored-logprobs-v1",
            "max_paired_rescored_logprob_abs_delta": (
                MAX_PAIRED_RESCORED_LOGPROB_ABS_DELTA
            ),
            "at_least_one_active_stored_logprob_must_change_by_more_than": (
                MIN_PAIRED_STORED_LOGPROB_ABS_DELTA
            ),
            "c2b_must_be_non_worse_on_every_trace": True,
            "c2b_must_be_strictly_better_on_at_least_one_trace": True,
            "c2b_all_four_traces_must_pass_original_gate": True,
        },
        "traces": traces,
        "summary": {
            "paired_generation_equal": all_generation_equal,
            "paired_score_alignment_equal": all_score_alignment_equal,
            "paired_runtime_invariants_equal": runtime_invariants_equal,
            "paired_rescored_logprobs_within_tolerance": all_rescored_equal,
            "at_least_one_active_stored_logprob_changed": any_stored_changed,
            "c2b_non_worse_on_every_trace": all_c2b_non_worse,
            "c2b_strictly_better_on_at_least_one_trace": (
                any_c2b_strictly_better
            ),
            "c2b_all_pre_registered_gates_passed": c2b_all_passed,
            "mechanism_supported": mechanism_supported,
        },
        "claim_boundary": {
            "screen_only": True,
            "may_change_threshold": False,
            "may_unlock_joint_optimizer": False,
            "requires_32_tune_and_32_sealed_confirmation": mechanism_supported,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    report["record_sha256"] = _record_sha256(report)
    return report


__all__ = [
    "C2A_TREATMENT",
    "C2B_TREATMENT",
    "CUDA_GRAPH_AXIS",
    "CUDA_GRAPH_DATASET_SELECTION",
    "compare_cuda_graph_screen_runs",
    "write_screen_report",
]
