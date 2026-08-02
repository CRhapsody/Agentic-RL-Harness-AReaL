from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from jphrl.trajectory.areal_joint_bridge import (
    inference_runtime_contract_sha256,
    validate_areal_joint_bridge_record,
)


SCREEN_SCHEMA_VERSION = "jph.sglang-logprob-mechanism-screen.v1"
SCORE_SCHEMA_VERSION = "jph.areal-same-backend-logprob.v6"
SGLANG_VERSION = "0.5.10.post1"
SCREEN_DATASET_SELECTION = "sequential-valid-offset32-count4-v1"
C0_MODE = "standard-log-of-softmax-v1"
C1_MODE = "original-log-softmax-v1"
MAX_MEAN_IMPORTANCE_RATIO_ERROR = 0.02
MAX_IMPORTANCE_RATIO_ERROR = 0.10
MAX_PAIRED_RESCORED_LOGPROB_ABS_DELTA = 1e-6
MIN_PAIRED_STORED_LOGPROB_ABS_DELTA = 1e-8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Mapping[str, object]) -> bytes:
    unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
    return json.dumps(
        unsigned,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _record_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def artifact_tree_sha256(run_root: Path) -> str:
    """Bind the final cell tree, excluding the audit record that stores this hash."""

    entries: list[dict[str, object]] = []
    for path in [run_root, *sorted(run_root.rglob("*"))]:
        relative = "." if path == run_root else path.relative_to(run_root).as_posix()
        if relative == "cell-audit.json":
            continue
        _require(not path.is_symlink(), f"screen artifact is a symlink: {path}")
        mode = path.stat().st_mode & 0o777
        if path.is_dir():
            entries.append({"kind": "directory", "mode": mode, "path": relative})
        elif path.is_file():
            entries.append(
                {
                    "kind": "file",
                    "mode": mode,
                    "path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                }
            )
    payload = json.dumps(
        entries,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trajectory_binding(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "input_ids",
            "loss_mask",
            "logprobs",
            "versions",
            "attention_mask",
            "rewards",
        )
    }


def _single_row(value: object, name: str) -> list[Any]:
    _require(
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list),
        f"{name} must have batch size one",
    )
    return value[0]


def _ratio_metrics(
    score: Mapping[str, object],
    *,
    rescored_override: list[Any] | None = None,
) -> dict[str, object]:
    mask = _single_row(score.get("loss_mask"), "loss_mask")
    stored = _single_row(score.get("stored_logprobs"), "stored_logprobs")
    rescored = (
        rescored_override
        if rescored_override is not None
        else _single_row(score.get("rescored_logprobs"), "rescored_logprobs")
    )
    _require(
        len(mask) == len(stored) == len(rescored),
        "score vectors have different lengths",
    )
    active = [index for index, value in enumerate(mask) if value == 1]
    _require(bool(active), "score has no active policy tokens")
    errors: list[float] = []
    for index in active:
        left = float(stored[index])
        right = float(rescored[index])
        _require(
            math.isfinite(left) and math.isfinite(right),
            "score contains a non-finite log-prob",
        )
        errors.append(abs(math.exp(right - left) - 1.0))
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    return {
        "checked_tokens": len(active),
        "mean_importance_ratio_error": mean_error,
        "max_importance_ratio_error": max_error,
        "passed_pre_registered_gate": (
            mean_error <= MAX_MEAN_IMPORTANCE_RATIO_ERROR
            and max_error <= MAX_IMPORTANCE_RATIO_ERROR
        ),
    }


def _load_cell(
    run_root: Path,
    *,
    expected_mode: str,
    expected_dataset_selection: str = SCREEN_DATASET_SELECTION,
    expected_treatment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    bridge_dir = run_root / "bridge-records"
    score_dir = run_root / "same-backend-scores"
    for directory in (run_root, bridge_dir, score_dir):
        _require(directory.is_dir(), f"missing screen directory: {directory}")
        _require(not directory.is_symlink(), f"screen directory is a symlink: {directory}")
        _require(
            directory.stat().st_mode & 0o777 == 0o700,
            f"screen directory is not private: {directory}",
        )
    for path in sorted(run_root.rglob("*")):
        _require(not path.is_symlink(), f"screen artifact is a symlink: {path}")
        mode = path.stat().st_mode & 0o777
        if path.is_dir():
            _require(mode == 0o700, f"screen directory is not private: {path}")
        elif path.is_file():
            _require(mode == 0o600, f"screen file is not private: {path}")

    manifest_path = run_root / "launch-manifest.json"
    _require(manifest_path.is_file(), f"missing launch manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == "jph.sglang-launch-manifest.v1",
        f"{manifest_path}: unexpected launch manifest schema",
    )
    _require(
        manifest.get("record_sha256") == _record_sha256(manifest),
        f"{manifest_path}: launch manifest hash mismatch",
    )
    runtime_contract = manifest.get("inference_runtime_contract")
    _require(
        isinstance(runtime_contract, dict),
        f"{manifest_path}: runtime contract missing",
    )
    runtime_contract_hash = inference_runtime_contract_sha256(runtime_contract)
    _require(
        manifest.get("inference_runtime_contract_sha256")
        == runtime_contract_hash,
        f"{manifest_path}: runtime contract hash mismatch",
    )
    _require(
        runtime_contract["identity"]["run_id"] == run_root.name,
        f"{manifest_path}: runtime run ID differs from directory",
    )
    _require(
        runtime_contract["fixed"]["clean_environment_policy"] == "env-i-v1",
        f"{manifest_path}: screen was not launched with a clean environment",
    )
    _require(
        runtime_contract["fixed"]["dataset_selection"]
        == expected_dataset_selection,
        f"{manifest_path}: runtime dataset selection mismatch",
    )
    _require(
        runtime_contract["fixed"]["sglang_version"] == SGLANG_VERSION,
        f"{manifest_path}: runtime SGLang version mismatch",
    )
    if expected_treatment is None:
        expected_treatment = {
            "generation_logprob_mode": expected_mode,
            "sglang_return_original_logprob": expected_mode == C1_MODE,
        }
        if (
            runtime_contract.get("schema_version")
            == "jph.sglang-inference-runtime.v2"
        ):
            expected_treatment = {
                "disable_cuda_graph": False,
                "experimental_axis": "generation-logprob-formula-v1",
                **expected_treatment,
            }
    _require(
        runtime_contract["treatment"] == expected_treatment,
        f"{manifest_path}: runtime treatment mismatch",
    )
    cell_audit_path = run_root / "cell-audit.json"
    _require(cell_audit_path.is_file(), f"missing cell audit: {cell_audit_path}")
    cell_audit = json.loads(cell_audit_path.read_text(encoding="utf-8"))
    _require(
        cell_audit.get("schema_version")
        == "jph.sglang-logprob-screen-cell-audit.v1"
        and cell_audit.get("record_sha256") == _record_sha256(cell_audit)
        and cell_audit.get("run_root") == str(run_root)
        and cell_audit.get("audited_tree_sha256")
        == artifact_tree_sha256(run_root)
        and cell_audit.get("passed") is True,
        f"{cell_audit_path}: invalid cell audit",
    )
    mode_audit = cell_audit.get("mode_audit_before_report")
    sensitive_audit = cell_audit.get("sensitive_artifact_audit")
    _require(
        isinstance(mode_audit, dict)
        and not mode_audit.get("mode_violations")
        and not mode_audit.get("symlinks")
        and isinstance(sensitive_audit, dict)
        and not sensitive_audit.get("unsafe_fields")
        and sensitive_audit.get("runtime_secret_matches") == 0
        and sensitive_audit.get("default_key_matches") == 0,
        f"{cell_audit_path}: cell safety audit did not pass",
    )
    bridges = sorted(bridge_dir.glob("bridge-*.json"))
    scores = sorted(score_dir.glob("same-backend-score-*.json"))
    _require(len(bridges) == 4, f"{run_root}: expected four bridge records")
    _require(len(scores) == 4, f"{run_root}: expected four score records")
    for path in [*bridges, *scores]:
        _require(not path.is_symlink(), f"screen record is a symlink: {path}")
        _require(
            path.resolve().is_relative_to(run_root),
            f"screen record escapes run root: {path}",
        )
        _require(
            path.stat().st_mode & 0o777 == 0o600,
            f"screen record is not private: {path}",
        )

    scores_by_request: dict[str, dict[str, object]] = {}
    for path in scores:
        score = json.loads(path.read_text(encoding="utf-8"))
        _require(
            score.get("schema_version") == SCORE_SCHEMA_VERSION,
            f"{path}: unexpected score schema",
        )
        _require(
            score.get("record_sha256") == _record_sha256(score),
            f"{path}: score hash mismatch",
        )
        request_id = score.get("request_id")
        _require(
            isinstance(request_id, str) and request_id not in scores_by_request,
            f"{path}: invalid or duplicate request ID",
        )
        origin = score.get("scoring_origin")
        _require(isinstance(origin, dict), f"{path}: score origin missing")
        _require(
            origin.get("generation_logprob_mode") == expected_mode,
            f"{path}: score mode mismatch",
        )
        _require(
            origin.get("dataset_selection") == expected_dataset_selection,
            f"{path}: score dataset selection mismatch",
        )
        _require(
            origin.get("sglang_version") == SGLANG_VERSION,
            f"{path}: score SGLang version mismatch",
        )
        _require(
            origin.get("inference_runtime_contract_sha256")
            == runtime_contract_hash,
            f"{path}: score runtime contract mismatch",
        )
        scores_by_request[request_id] = score

    by_prompt: dict[str, dict[str, object]] = {}
    for path in bridges:
        bridge = json.loads(path.read_text(encoding="utf-8"))
        audit = validate_areal_joint_bridge_record(
            bridge,
            expected_policy_version=0,
        )
        _require(
            audit["generation_logprob_mode"] == expected_mode,
            f"{path}: bridge mode mismatch",
        )
        _require(
            audit["dataset_selection"] == expected_dataset_selection,
            f"{path}: bridge dataset selection mismatch",
        )
        _require(
            audit["sglang_version"] == SGLANG_VERSION,
            f"{path}: bridge SGLang version mismatch",
        )
        _require(
            audit["inference_runtime_contract_sha256"]
            == runtime_contract_hash,
            f"{path}: bridge runtime contract hash mismatch",
        )
        _require(
            bridge["policy_binding"]["inference_runtime_contract"]
            == runtime_contract,
            f"{path}: bridge runtime contract differs from launch manifest",
        )
        request_id = str(bridge["request_id"])
        score = scores_by_request.get(request_id)
        _require(score is not None, f"{path}: matching score is missing")
        _require(
            score["bridge_record_sha256"] == bridge["record_sha256"],
            f"{path}: score is not bound to bridge",
        )
        tensors = bridge["areal_trace"]["tensor_dict"]
        _require(
            score.get("trajectory_binding_sha256")
            == _record_sha256(_trajectory_binding(tensors)),
            f"{path}: trajectory binding hash mismatch",
        )
        for score_field, tensor_field in (
            ("input_ids", "input_ids"),
            ("loss_mask", "loss_mask"),
            ("stored_logprobs", "logprobs"),
            ("versions", "versions"),
        ):
            _require(
                score.get(score_field) == tensors[tensor_field],
                f"{path}: {score_field} differs from bridge trajectory",
            )
        origin = score["scoring_origin"]
        expected_origin = {
            "api": "RolloutController.compute_logp",
            "controller_api_version": "v1",
            "lifecycle": "same-controller-after-wait-before-destroy",
            "score_parser": "jph-tail-before-conversion-v1",
            "score_token_id_validation": "exact-requested-tail-v1",
            "transport_localization": "RTensor.localize-before-score-and-write-v1",
            "backend": "sglang:d1p1t1",
            "engine_version_before_score": 0,
            "engine_version_after_score": 0,
            "policy_release_id": bridge["joint_version"]["policy"],
            "behavior_revision": bridge["areal_trace"]["origin"][
                "behavior_revision"
            ],
            "areal_commit": bridge["areal_trace"]["origin"]["areal_commit"],
            "project_commit": bridge["origin"]["project_commit"],
            "generation_logprob_mode": expected_mode,
            "dataset_selection": expected_dataset_selection,
            "sglang_version": SGLANG_VERSION,
            "inference_runtime_contract_sha256": runtime_contract_hash,
        }
        _require(origin == expected_origin, f"{path}: score origin mismatch")
        prompt_key = str(bridge["prompt_binding"]["base_messages_sha256"])
        _require(prompt_key not in by_prompt, f"{path}: duplicate base prompt hash")
        by_prompt[prompt_key] = {
            "bridge": bridge,
            "score": score,
            "metrics": _ratio_metrics(score),
        }
    _require(
        set(scores_by_request)
        == {str(item["bridge"]["request_id"]) for item in by_prompt.values()},
        f"{run_root}: not every score was consumed",
    )
    return {
        "by_prompt": by_prompt,
        "manifest": manifest,
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": runtime_contract_hash,
    }


def _paired_score_deltas(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, object]:
    for field in ("input_ids", "loss_mask", "versions"):
        _require(left.get(field) == right.get(field), f"paired score {field} changed")
    mask = _single_row(left.get("loss_mask"), "paired loss_mask")
    left_stored = _single_row(left.get("stored_logprobs"), "C0 stored_logprobs")
    right_stored = _single_row(right.get("stored_logprobs"), "C1 stored_logprobs")
    left_rescored = _single_row(
        left.get("rescored_logprobs"), "C0 rescored_logprobs"
    )
    right_rescored = _single_row(
        right.get("rescored_logprobs"), "C1 rescored_logprobs"
    )
    _require(
        len(mask)
        == len(left_stored)
        == len(right_stored)
        == len(left_rescored)
        == len(right_rescored),
        "paired score vectors have different lengths",
    )
    active = [index for index, value in enumerate(mask) if value == 1]
    _require(bool(active), "paired score has no active policy tokens")
    rescored_deltas = [
        abs(float(left_rescored[index]) - float(right_rescored[index]))
        for index in active
    ]
    stored_deltas = [
        abs(float(left_stored[index]) - float(right_stored[index]))
        for index in active
    ]
    _require(
        all(math.isfinite(value) for value in [*rescored_deltas, *stored_deltas]),
        "paired score contains a non-finite delta",
    )
    max_rescored_delta = max(rescored_deltas)
    max_stored_delta = max(stored_deltas)
    return {
        "max_rescored_logprob_abs_delta": max_rescored_delta,
        "rescored_logprobs_within_tolerance": (
            max_rescored_delta <= MAX_PAIRED_RESCORED_LOGPROB_ABS_DELTA
        ),
        "max_stored_logprob_abs_delta": max_stored_delta,
        "stored_logprobs_changed": (
            max_stored_delta > MIN_PAIRED_STORED_LOGPROB_ABS_DELTA
        ),
    }


def paired_generation_equal(
    left_bridge: Mapping[str, object],
    right_bridge: Mapping[str, object],
) -> bool:
    """Check every pre-registered non-logprob rollout field in a paired trace."""

    left_joint = {
        key: value
        for key, value in left_bridge["joint_version"].items()
        if key != "policy"
    }
    right_joint = {
        key: value
        for key, value in right_bridge["joint_version"].items()
        if key != "policy"
    }
    return bool(
        left_bridge["origin"] == right_bridge["origin"]
        and left_bridge["task_id"] == right_bridge["task_id"]
        and left_bridge["request_id"] == right_bridge["request_id"]
        and left_joint == right_joint
        and left_bridge["prompt_binding"] == right_bridge["prompt_binding"]
        and left_bridge["harness"]["state"]
        == right_bridge["harness"]["state"]
        and left_bridge["harness"]["controller_checkpoint_before_decision"]
        == right_bridge["harness"]["controller_checkpoint_before_decision"]
        and left_bridge["harness"]["applied_instruction"]
        == right_bridge["harness"]["applied_instruction"]
        and left_bridge["harness"]["decision"]
        == right_bridge["harness"]["decision"]
        and left_bridge["areal_trace"]["origin"]
        == right_bridge["areal_trace"]["origin"]
        and left_bridge["areal_trace"]["model_response"]["input_tokens"]
        == right_bridge["areal_trace"]["model_response"]["input_tokens"]
        and left_bridge["areal_trace"]["model_response"]["output_tokens"]
        == right_bridge["areal_trace"]["model_response"]["output_tokens"]
        and left_bridge["areal_trace"]["model_response"]["output_versions"]
        == right_bridge["areal_trace"]["model_response"]["output_versions"]
        and left_bridge["areal_trace"]["model_response"]["stop_reason"]
        == right_bridge["areal_trace"]["model_response"]["stop_reason"]
        and left_bridge["areal_trace"]["interaction"]["reward"]
        == right_bridge["areal_trace"]["interaction"]["reward"]
    )


def compare_screen_runs(c0_root: Path, c1_root: Path) -> dict[str, object]:
    configured_root = Path(os.environ["JPH_ROOT"]).resolve()
    c0_root = c0_root.resolve()
    c1_root = c1_root.resolve()
    _require(c0_root.is_relative_to(configured_root), "C0 root escapes JPH_ROOT")
    _require(c1_root.is_relative_to(configured_root), "C1 root escapes JPH_ROOT")
    c0_cell = _load_cell(c0_root, expected_mode=C0_MODE)
    c1_cell = _load_cell(c1_root, expected_mode=C1_MODE)
    c0 = c0_cell["by_prompt"]
    c1 = c1_cell["by_prompt"]
    _require(isinstance(c0, dict) and isinstance(c1, dict), "invalid screen cell")
    _require(set(c0) == set(c1), "C0 and C1 selected different base prompts")

    c0_runtime = c0_cell["runtime_contract"]
    c1_runtime = c1_cell["runtime_contract"]
    _require(
        isinstance(c0_runtime, dict) and isinstance(c1_runtime, dict),
        "screen runtime contracts are missing",
    )
    pair_id = c0_runtime["identity"]["screen_pair_id"]
    c0_treatment = {
        "generation_logprob_mode": C0_MODE,
        "sglang_return_original_logprob": False,
    }
    c1_treatment = {
        "generation_logprob_mode": C1_MODE,
        "sglang_return_original_logprob": True,
    }
    if c0_runtime.get("schema_version") == "jph.sglang-inference-runtime.v2":
        c0_treatment = {
            "disable_cuda_graph": False,
            "experimental_axis": "generation-logprob-formula-v1",
            **c0_treatment,
        }
        c1_treatment = {
            "disable_cuda_graph": False,
            "experimental_axis": "generation-logprob-formula-v1",
            **c1_treatment,
        }
    runtime_fixed_equal = (
        isinstance(pair_id, str)
        and bool(pair_id)
        and pair_id == c1_runtime["identity"]["screen_pair_id"]
        and c0_runtime["schema_version"] == c1_runtime["schema_version"]
        and c0_runtime["fixed"] == c1_runtime["fixed"]
        and c0_runtime["treatment"] == c0_treatment
        and c1_runtime["treatment"] == c1_treatment
    )

    traces: list[dict[str, object]] = []
    all_generation_equal = True
    all_rescored_equal = True
    any_stored_changed = False
    all_c1_non_worse = True
    any_c1_strictly_better = False
    for prompt_key in sorted(c0):
        left = c0[prompt_key]
        right = c1[prompt_key]
        left_bridge = left["bridge"]
        right_bridge = right["bridge"]
        generation_equal = paired_generation_equal(left_bridge, right_bridge)
        all_generation_equal = all_generation_equal and generation_equal
        paired_score = _paired_score_deltas(left["score"], right["score"])
        all_rescored_equal = (
            all_rescored_equal
            and bool(paired_score["rescored_logprobs_within_tolerance"])
        )
        any_stored_changed = (
            any_stored_changed or bool(paired_score["stored_logprobs_changed"])
        )
        left_metrics = left["metrics"]
        common_rescored_target = _single_row(
            left["score"]["rescored_logprobs"],
            "C0 common rescored target",
        )
        right_observed_metrics = right["metrics"]
        right_metrics = _ratio_metrics(
            right["score"],
            rescored_override=common_rescored_target,
        )
        mean_non_worse = (
            right_metrics["mean_importance_ratio_error"]
            <= left_metrics["mean_importance_ratio_error"] + 1e-12
        )
        max_non_worse = (
            right_metrics["max_importance_ratio_error"]
            <= left_metrics["max_importance_ratio_error"] + 1e-12
        )
        all_c1_non_worse = all_c1_non_worse and mean_non_worse and max_non_worse
        strictly_better = bool(paired_score["stored_logprobs_changed"]) and (
            right_metrics["mean_importance_ratio_error"]
            < left_metrics["mean_importance_ratio_error"] - 1e-12
            or right_metrics["max_importance_ratio_error"]
            < left_metrics["max_importance_ratio_error"] - 1e-12
        )
        any_c1_strictly_better = any_c1_strictly_better or strictly_better
        traces.append(
            {
                "base_messages_sha256": prompt_key,
                "generation_equal": generation_equal,
                "paired_score": paired_score,
                "c0": left_metrics,
                "c1": right_metrics,
                "c1_observed_rescored_target_diagnostic": (
                    right_observed_metrics
                ),
            }
        )

    c1_all_passed = all(
        bool(item["c1"]["passed_pre_registered_gate"]) for item in traces
    )
    mechanism_supported = (
        all_generation_equal
        and runtime_fixed_equal
        and all_rescored_equal
        and any_stored_changed
        and all_c1_non_worse
        and any_c1_strictly_better
        and c1_all_passed
    )
    report: dict[str, object] = {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "experiment": "sglang-generation-logprob-formula-screen",
        "screen_pair_id": pair_id,
        "dataset_selection": SCREEN_DATASET_SELECTION,
        "sglang_version": SGLANG_VERSION,
        "fixed_runtime_sha256": _record_sha256(
            {"fixed": c0_runtime["fixed"]}
        ),
        "cells": {
            "c0": {
                "generation_logprob_mode": C0_MODE,
                "run_root": str(c0_root),
                "inference_runtime_contract_sha256": c0_cell[
                    "runtime_contract_sha256"
                ],
            },
            "c1": {
                "generation_logprob_mode": C1_MODE,
                "run_root": str(c1_root),
                "inference_runtime_contract_sha256": c1_cell[
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
            "paired_runtime_fixed_fields_must_be_identical": True,
            "paired_metric_target": "c0-rescored-logprobs-v1",
            "max_paired_rescored_logprob_abs_delta": (
                MAX_PAIRED_RESCORED_LOGPROB_ABS_DELTA
            ),
            "at_least_one_active_stored_logprob_must_change_by_more_than": (
                MIN_PAIRED_STORED_LOGPROB_ABS_DELTA
            ),
            "c1_must_be_non_worse_on_every_trace": True,
            "c1_must_be_strictly_better_on_at_least_one_trace": True,
            "c1_all_four_traces_must_pass_original_gate": True,
        },
        "traces": traces,
        "summary": {
            "paired_generation_equal": all_generation_equal,
            "paired_runtime_fixed_fields_equal": runtime_fixed_equal,
            "paired_rescored_logprobs_within_tolerance": all_rescored_equal,
            "at_least_one_active_stored_logprob_changed": any_stored_changed,
            "c1_non_worse_on_every_trace": all_c1_non_worse,
            "c1_strictly_better_on_at_least_one_trace": any_c1_strictly_better,
            "c1_all_pre_registered_gates_passed": c1_all_passed,
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


def write_screen_report(report: Mapping[str, object], output: Path) -> None:
    root = Path(os.environ["JPH_ROOT"]).resolve()
    resolved = output.resolve()
    _require(resolved.is_relative_to(root), f"output escapes JPH_ROOT: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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
