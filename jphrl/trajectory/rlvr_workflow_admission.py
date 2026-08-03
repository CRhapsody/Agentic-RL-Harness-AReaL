from __future__ import annotations

"""Fail-closed RLVR-workflow pre-batch admission into Q/R/S.

This route starts from the v3 joint bridge and the *same live* AReaL
``InteractionWithTokenLogpReward`` still owned by the workflow.  It never
creates Agent Service or Hermes receipts and never invents session/trajectory
identity.  The persisted runner envelope is self-contained and stops before
either optimizer is called.
"""

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .areal_interaction_sidecar import (
    REQUIRED_TENSOR_FIELDS,
    archive_premerged_exported_interactions,
    validate_bound_training_sample_archive,
    validate_interaction_adapter_sidecar,
)
from .areal_joint_bridge import (
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
)
from .areal_joint_bridge import (
    ArealJointBridgeError,
    validate_areal_joint_bridge_record,
)
from .areal_policy_admission import (
    RLVR_WORKFLOW_POLICY_ADMISSION_SCHEMA_VERSION,
    _validate_and_materialize_samples,
    validate_policy_training_admission,
)
from .harness_action_admission import (
    admit_pre_batch_bound_harness_action_samples,
    validate_harness_action_admission_record,
)
from .joint_credit_alignment import (
    ESTIMATOR_VERSION,
    DualCreditEstimatorSpec,
    build_frozen_joint_credit_alignment,
    validate_frozen_joint_credit_alignment,
)
from .schema import EpisodeTrace, JointVersion, TraceEvent

ROUTE_KIND = "rlvr-workflow"
RLVR_PRE_BATCH_STAGE = "rlvr-workflow-output-before-training-batch"
RLVR_PRE_BATCH_SCHEMA_VERSION = "jph.rlvr-workflow-pre-batch-training-record.v1"
RLVR_RUNNER_ADMISSION_SCHEMA_VERSION = "jph.rlvr-workflow-runner-admission.v1"
RLVR_FROZEN_ESTIMATOR_TEMPLATE_SCHEMA_VERSION = (
    "jph.rlvr-workflow-frozen-dual-credit-template.v1"
)

_TRAINABLE_REWARDS = frozenset({0.0, 1.0})
_TRACE_EVENT_KINDS = (
    "episode_started",
    "harness_decision",
    "model_request",
    "model_response",
    "reward_assigned",
    "rlvr_terminal_outcome",
)
_PRE_BATCH_FIELDS = frozenset(
    {
        "schema_version",
        "route_kind",
        "identity",
        "joint_version",
        "source",
        "harness_checkpoint_binding",
        "bridge_record",
        "episode_trace",
        "training_archive",
        "evidence_scope",
        "record_sha256",
    }
)
_PRE_BATCH_IDENTITY_FIELDS = frozenset(
    {
        "route_kind",
        "episode_id",
        "task_id",
        "request_id",
        "model_call_id",
        "interaction_id",
        "ordinal",
        "parent_model_call_id",
        "session_id",
        "trajectory_id",
        "joint_version_id",
    }
)
_PRE_BATCH_SOURCE_FIELDS = frozenset(
    {
        "bridge_record_sha256",
        "areal_trace_record_sha256",
        "interaction_sidecar_sha256",
        "episode_trace_sha256",
        "pre_batch_stage",
        "interaction_type",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "decision_id",
        "controller_version",
        "checkpoint_schema_version",
        "checkpoint_sha256",
    }
)
_RUNNER_FIELDS = frozenset(
    {
        "schema_version",
        "route_kind",
        "bridge_record_sha256",
        "bridge_record",
        "rlvr_pre_batch_record",
        "episode_trace_sha256",
        "episode_trace",
        "s_joint_credit",
        "materialized_estimator",
        "mem_fraction_static_provenance",
        "evidence_scope",
        "record_sha256",
    }
)
_MEMORY_PROVENANCE_FIELDS = frozenset(
    {"value", "source_path", "runtime_contract_sha256"}
)
_ESTIMATOR_TEMPLATE_FIELDS = frozenset(
    {
        "schema_version",
        "estimator_version",
        "policy_source",
        "harness_source",
        "policy_baseline_snapshot_id",
        "harness_baseline_snapshot_id",
        "policy_baseline",
        "harness_baseline",
        "record_sha256",
    }
)
_PRE_BATCH_EVIDENCE_SCOPE = {
    "real_areal_rollout": True,
    "rlvr_workflow_pre_batch_binding": True,
    "agent_service_receipt": False,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}
_POLICY_EVIDENCE_SCOPE = {
    "pre_batch_interaction_binding": True,
    "policy_samples_admitted": True,
    "policy_advantages_attached": False,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}
_RUNNER_EVIDENCE_SCOPE = {
    "rlvr_workflow_pre_batch_binding": True,
    "policy_samples_admitted": True,
    "harness_action_samples_admitted": True,
    "policy_advantages_aligned": True,
    "harness_advantages_aligned": True,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}
_SECRET_FIELD_NAMES = {
    "admin_api_key",
    "api_key",
    "authorization",
    "cookie",
    "credentials",
    "github_token",
    "password",
    "refresh_token",
    "secret",
    "session_api_key",
    "token",
    "access_token",
}
_SECRET_FIELD_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_cookie",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)
_FORBIDDEN_ESTIMATOR_PROVENANCE_MARKERS = (
    "fixture",
    "placeholder",
    "raw-reward-only",
    "synthetic",
)


class RlvrWorkflowAdmissionError(ValueError):
    """Raised when an RLVR bridge cannot safely reach a frozen S record."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RlvrWorkflowAdmissionError(message)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RlvrWorkflowAdmissionError(
            "RLVR admission is not finite canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _record_sha256(record: Mapping[str, object]) -> str:
    return _sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _assert_no_secret_fields(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            _require(
                normalized not in _SECRET_FIELD_NAMES
                and not normalized.endswith(_SECRET_FIELD_SUFFIXES),
                f"credential field cannot enter RLVR admission: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


def _exact_mapping(
    value: object, expected_fields: frozenset[str] | set[str], label: str
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == set(expected_fields), f"{label} field set differs")
    return value


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _joint_version(raw: object) -> JointVersion:
    value = _exact_mapping(
        raw,
        set(JointVersion.__dataclass_fields__),
        "RLVR JointVersion",
    )
    _require(
        all(_is_non_empty_string(item) for item in value.values()),
        "RLVR JointVersion fields must be non-empty strings",
    )
    try:
        return JointVersion(**dict(value))
    except TypeError as exc:  # pragma: no cover - exact field check owns this
        raise RlvrWorkflowAdmissionError("invalid RLVR JointVersion") from exc


def dual_credit_estimator_spec_from_record(
    raw: Mapping[str, object],
) -> DualCreditEstimatorSpec:
    """Load an exact, non-secret frozen estimator config for the live workflow."""

    value = _exact_mapping(
        raw,
        set(DualCreditEstimatorSpec.__dataclass_fields__),
        "frozen dual-credit estimator",
    )
    _assert_no_secret_fields(value, "estimator")
    policy_baselines = value.get("policy_baselines")
    harness_baselines = value.get("harness_baselines")
    _require(
        isinstance(policy_baselines, Mapping)
        and isinstance(harness_baselines, Mapping),
        "frozen estimator baseline maps are missing",
    )
    try:
        return DualCreditEstimatorSpec(
            estimator_version=value["estimator_version"],
            parent_joint_version_id=value["parent_joint_version_id"],
            policy_source=value["policy_source"],
            harness_source=value["harness_source"],
            policy_baseline_snapshot_id=value["policy_baseline_snapshot_id"],
            harness_baseline_snapshot_id=value["harness_baseline_snapshot_id"],
            policy_baselines={
                str(key): float(item) for key, item in policy_baselines.items()
            },
            harness_baselines={
                str(key): float(item) for key, item in harness_baselines.items()
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RlvrWorkflowAdmissionError(
            "invalid frozen dual-credit estimator"
        ) from exc


@dataclass(frozen=True)
class FrozenDualCreditEstimatorTemplate:
    estimator_version: str
    policy_source: str
    harness_source: str
    policy_baseline_snapshot_id: str
    harness_baseline_snapshot_id: str
    policy_baseline: float
    harness_baseline: float
    record_sha256: str

    def validate(self) -> None:
        _require(
            self.estimator_version == ESTIMATOR_VERSION,
            "unknown frozen estimator template version",
        )
        for value, label in (
            (self.policy_source, "Policy credit source"),
            (self.harness_source, "Harness credit source"),
            (self.policy_baseline_snapshot_id, "Policy baseline snapshot ID"),
            (self.harness_baseline_snapshot_id, "Harness baseline snapshot ID"),
        ):
            _require(_is_non_empty_string(value), f"{label} must be non-empty")
        _require(
            self.policy_source != self.harness_source,
            "Policy and Harness credit sources must remain distinct",
        )
        for source in (self.policy_source, self.harness_source):
            normalized = source.lower()
            _require(
                not any(
                    marker in normalized
                    for marker in _FORBIDDEN_ESTIMATOR_PROVENANCE_MARKERS
                ),
                "synthetic or placeholder provenance cannot enter the frozen template",
            )
        _require(
            self.policy_baseline_snapshot_id != self.harness_baseline_snapshot_id,
            "Policy and Harness baseline snapshots must remain distinct",
        )
        _require(
            _is_finite_number(self.policy_baseline)
            and _is_finite_number(self.harness_baseline),
            "frozen template baselines must be finite scalars",
        )
        _require(_is_sha256(self.record_sha256), "template SHA-256 is invalid")


def build_frozen_dual_credit_estimator_template(
    *,
    policy_source: str,
    harness_source: str,
    policy_baseline_snapshot_id: str,
    harness_baseline_snapshot_id: str,
    policy_baseline: float,
    harness_baseline: float,
) -> dict[str, object]:
    """Build the strict per-run template shared by multiple rollout requests."""

    record: dict[str, object] = {
        "schema_version": RLVR_FROZEN_ESTIMATOR_TEMPLATE_SCHEMA_VERSION,
        "estimator_version": ESTIMATOR_VERSION,
        "policy_source": policy_source,
        "harness_source": harness_source,
        "policy_baseline_snapshot_id": policy_baseline_snapshot_id,
        "harness_baseline_snapshot_id": harness_baseline_snapshot_id,
        "policy_baseline": policy_baseline,
        "harness_baseline": harness_baseline,
    }
    _assert_no_secret_fields(record, "estimator_template")
    record["record_sha256"] = _record_sha256(record)
    frozen_dual_credit_estimator_template_from_record(record)
    return record


def frozen_dual_credit_estimator_template_from_record(
    raw: Mapping[str, object],
) -> FrozenDualCreditEstimatorTemplate:
    value = _exact_mapping(
        raw,
        _ESTIMATOR_TEMPLATE_FIELDS,
        "frozen estimator template",
    )
    _require(
        value.get("schema_version") == RLVR_FROZEN_ESTIMATOR_TEMPLATE_SCHEMA_VERSION,
        "unknown frozen estimator template schema",
    )
    _require(
        value.get("record_sha256") == _record_sha256(value),
        "frozen estimator template hash mismatch",
    )
    _assert_no_secret_fields(value, "estimator_template")
    try:
        template = FrozenDualCreditEstimatorTemplate(
            estimator_version=value["estimator_version"],
            policy_source=value["policy_source"],
            harness_source=value["harness_source"],
            policy_baseline_snapshot_id=value["policy_baseline_snapshot_id"],
            harness_baseline_snapshot_id=value["harness_baseline_snapshot_id"],
            policy_baseline=float(value["policy_baseline"]),
            harness_baseline=float(value["harness_baseline"]),
            record_sha256=value["record_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RlvrWorkflowAdmissionError("invalid frozen estimator template") from exc
    template.validate()
    return template


def materialize_dual_credit_estimator_from_template(
    template: FrozenDualCreditEstimatorTemplate,
    *,
    joint_version: JointVersion,
    model_call_id: str,
    harness_decision_id: str,
) -> DualCreditEstimatorSpec:
    """Bind frozen scalar baselines deterministically to live exact IDs."""

    _require(
        type(template) is FrozenDualCreditEstimatorTemplate,
        "estimator materialization requires the strict frozen template",
    )
    template.validate()
    _require(
        type(joint_version) is JointVersion, "materialization JointVersion is invalid"
    )
    _require(
        _is_non_empty_string(model_call_id), "materialization model call ID is missing"
    )
    _require(
        _is_non_empty_string(harness_decision_id),
        "materialization Harness decision ID is missing",
    )
    _require(
        model_call_id != harness_decision_id,
        "Policy and Harness materialized targets must remain distinct",
    )
    estimator = DualCreditEstimatorSpec(
        estimator_version=template.estimator_version,
        parent_joint_version_id=joint_version.version_id,
        policy_source=template.policy_source,
        harness_source=template.harness_source,
        policy_baseline_snapshot_id=template.policy_baseline_snapshot_id,
        harness_baseline_snapshot_id=template.harness_baseline_snapshot_id,
        policy_baselines={model_call_id: float(template.policy_baseline)},
        harness_baselines={harness_decision_id: float(template.harness_baseline)},
    )
    try:
        estimator.validate(
            joint_version=joint_version,
            policy_model_call_ids=(model_call_id,),
            harness_decision_ids=(harness_decision_id,),
        )
    except ValueError as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    return estimator


def _episode_trace_from_record(raw: object) -> EpisodeTrace:
    value = _exact_mapping(
        raw,
        set(EpisodeTrace.__dataclass_fields__),
        "RLVR EpisodeTrace",
    )
    events_raw = value.get("events")
    _require(isinstance(events_raw, list), "RLVR EpisodeTrace events must be a list")
    events: list[TraceEvent] = []
    for raw_event in events_raw:
        event = _exact_mapping(
            raw_event,
            set(TraceEvent.__dataclass_fields__),
            "RLVR TraceEvent",
        )
        _require(isinstance(event.get("payload"), Mapping), "trace payload is missing")
        try:
            events.append(
                TraceEvent(
                    index=event["index"],
                    event_id=event["event_id"],
                    parent_event_id=event["parent_event_id"],
                    kind=event["kind"],
                    producer=event["producer"],
                    joint_version_id=event["joint_version_id"],
                    payload=deepcopy(dict(event["payload"])),
                )
            )
        except (KeyError, TypeError) as exc:
            raise RlvrWorkflowAdmissionError("invalid RLVR TraceEvent") from exc
    try:
        trace = EpisodeTrace(
            episode_id=value["episode_id"],
            task_id=value["task_id"],
            seed=value["seed"],
            joint_version=_joint_version(value["joint_version"]),
            harness_spec_hash=value["harness_spec_hash"],
            events=events,
            reward=value["reward"],
            success=value["success"],
            validity_class=value["validity_class"],
            failure_category=value["failure_category"],
        )
        trace.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    return trace


def _source_binding(bridge: Mapping[str, object]) -> Mapping[str, object]:
    sidecar = bridge.get("interaction_adapter_sidecar")
    _require(isinstance(sidecar, Mapping), "bridge interaction sidecar is missing")
    try:
        sidecar_audit = validate_interaction_adapter_sidecar(sidecar)
    except ValueError as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    _require(
        sidecar_audit.get("route_kind") == ROUTE_KIND,
        "RLVR admission rejects an Agent Service or unknown route",
    )
    _require(
        sidecar_audit.get("session_id") is None
        and sidecar_audit.get("trajectory_id") is None,
        "RLVR route cannot carry session or trajectory identity",
    )
    bindings = sidecar.get("bindings")
    _require(
        isinstance(bindings, list) and len(bindings) == 1,
        "single-turn RLVR admission requires exactly one interaction binding",
    )
    binding = bindings[0]
    _require(isinstance(binding, Mapping), "RLVR interaction binding is invalid")
    _require(
        binding.get("ordinal") == 0 and binding.get("parent_interaction_id") is None,
        "single-turn RLVR admission requires one root interaction at ordinal zero",
    )
    return binding


def _expected_policy_version(bridge: Mapping[str, object]) -> int:
    policy = bridge.get("policy_binding")
    _require(isinstance(policy, Mapping), "bridge policy binding is missing")
    value = policy.get("expected_inference_engine_version")
    _require(
        type(value) is int and value >= 0,
        "bridge expected policy version is invalid",
    )
    return value


def _validate_bridge(
    bridge: Mapping[str, object], *, active_joint_version: JointVersion | None
) -> tuple[dict[str, object], JointVersion, Mapping[str, object]]:
    _require(
        bridge.get("schema_version") == BRIDGE_SCHEMA_VERSION,
        "RLVR admission requires the validated v3 bridge",
    )
    _assert_no_secret_fields(bridge, "bridge")
    try:
        audit = validate_areal_joint_bridge_record(
            bridge,
            expected_policy_version=_expected_policy_version(bridge),
        )
    except (ArealJointBridgeError, ValueError) as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    joint_version = _joint_version(bridge.get("joint_version"))
    if active_joint_version is not None:
        _require(
            joint_version == active_joint_version,
            "RLVR bridge JointVersion differs from lag-zero active version",
        )
    binding = _source_binding(bridge)
    _require(
        audit.get("model_call_id") == binding.get("model_call_id")
        and audit.get("interaction_id") == binding.get("interaction_id"),
        "RLVR bridge audit differs from its interaction binding",
    )
    return dict(audit), joint_version, binding


def _require_real_pre_batch_interaction(
    interaction: Any,
    *,
    bridge: Mapping[str, object],
    binding: Mapping[str, object],
    pre_batch_stage: str,
) -> None:
    _require(
        pre_batch_stage == RLVR_PRE_BATCH_STAGE,
        "post-batch RLVR data cannot be bound to interaction identity",
    )
    try:
        from areal.experimental.openai.types import (
            InteractionWithTokenLogpReward,
        )
    except ImportError as exc:
        raise RlvrWorkflowAdmissionError(
            "pinned AReaL InteractionWithTokenLogpReward is unavailable"
        ) from exc
    _require(
        type(interaction) is InteractionWithTokenLogpReward,
        "RLVR admission requires the real pre-batch "
        "InteractionWithTokenLogpReward object",
    )
    areal_trace = bridge.get("areal_trace")
    _require(isinstance(areal_trace, Mapping), "bridge AReaL trace is missing")
    recorded_interaction = areal_trace.get("interaction")
    response = areal_trace.get("model_response")
    tensors = areal_trace.get("tensor_dict")
    _require(
        isinstance(recorded_interaction, Mapping)
        and isinstance(response, Mapping)
        and isinstance(tensors, Mapping),
        "bridge AReaL interaction roundtrip is incomplete",
    )
    _require(
        interaction.interaction_id
        == binding.get("interaction_id")
        == recorded_interaction.get("interaction_id"),
        "live RLVR interaction identity differs from the bridge",
    )
    _require(
        interaction.parent is None,
        "single-turn RLVR pre-batch interaction cannot have a parent",
    )
    _require(
        interaction.chat_template_type
        == recorded_interaction.get("chat_template_type"),
        "live RLVR chat-template type differs from the bridge",
    )
    _require(
        _is_finite_number(interaction.reward)
        and float(interaction.reward) == float(recorded_interaction.get("reward")),
        "live RLVR reward differs from the bridge",
    )
    source_response = interaction.model_response
    _require(source_response is not None, "live RLVR interaction has no ModelResponse")
    _require(
        list(source_response.input_tokens) == response.get("input_tokens")
        and list(source_response.output_tokens) == response.get("output_tokens")
        and list(source_response.output_versions) == response.get("output_versions")
        and getattr(source_response, "stop_reason", None)
        == response.get("stop_reason"),
        "live RLVR ModelResponse differs from the bridge",
    )
    recorded_logprobs = response.get("output_logprobs")
    _require(
        isinstance(recorded_logprobs, list)
        and len(source_response.output_logprobs) == len(recorded_logprobs)
        and all(
            math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
            for actual, expected in zip(
                source_response.output_logprobs, recorded_logprobs
            )
        ),
        "live RLVR old logprobs differ from the bridge",
    )
    raw_tensors = interaction.to_tensor_dict()
    _require(
        isinstance(raw_tensors, Mapping)
        and set(raw_tensors) == set(REQUIRED_TENSOR_FIELDS),
        "live RLVR interaction does not expose the exact six-field tensor contract",
    )
    _require(
        _plain(raw_tensors) == tensors,
        "live RLVR pre-batch tensors differ from the bridge roundtrip",
    )


def _build_episode_trace(
    bridge: Mapping[str, object],
    *,
    joint_version: JointVersion,
    binding: Mapping[str, object],
) -> EpisodeTrace:
    areal_trace = bridge["areal_trace"]
    response = areal_trace["model_response"]
    tensors = areal_trace["tensor_dict"]
    harness = bridge["harness"]
    decision = harness["decision"]
    state = harness["state"]
    policy = bridge["policy_binding"]
    runtime = policy["inference_runtime_contract"]
    seed = runtime["fixed"].get("seed")
    _require(type(seed) is int and seed >= 0, "bridge runtime seed is invalid")
    reward = float(bridge["credit_binding"]["raw_terminal_reward"])
    _require(
        reward in _TRAINABLE_REWARDS,
        "single-turn GSM8K RLVR admission requires a binary terminal reward",
    )
    input_tokens = list(response["input_tokens"])
    output_tokens = list(response["output_tokens"])
    output_mask = list(tensors["loss_mask"][0][len(input_tokens) :])
    _require(
        output_mask == [1] * len(output_tokens),
        "RLVR completion mask differs from the real AReaL action span",
    )
    artifact = bridge["harness_artifact"]
    trace = EpisodeTrace(
        episode_id=str(binding["episode_id"]),
        task_id=str(bridge["task_id"]),
        seed=seed,
        joint_version=joint_version,
        harness_spec_hash=str(artifact["sha256"]),
    )
    trace.append(
        "episode_started",
        ROUTE_KIND,
        {
            "route_kind": ROUTE_KIND,
            "source_bridge_sha256": bridge["record_sha256"],
            "request_id": bridge["request_id"],
            "task_id": bridge["task_id"],
            "seed": seed,
            "session_id": None,
            "trajectory_id": None,
        },
    )
    harness_payload = deepcopy(dict(decision))
    harness_payload["state"] = deepcopy(dict(state))
    trace.append("harness_decision", "harness", harness_payload)
    trace.append(
        "model_request",
        ROUTE_KIND,
        {
            "route_kind": ROUTE_KIND,
            "model_call_id": binding["model_call_id"],
            "ordinal": 0,
            "parent_model_call_id": None,
            "interaction_id": binding["interaction_id"],
            "effective_messages_sha256": bridge["prompt_binding"][
                "effective_messages_sha256"
            ],
        },
    )
    trace.append(
        "model_response",
        "areal",
        {
            "model_call_id": binding["model_call_id"],
            "interaction_id": binding["interaction_id"],
            "input_token_ids": input_tokens,
            "output_token_ids": output_tokens,
            "output_token_logprobs": list(response["output_logprobs"]),
            "output_versions": list(response["output_versions"]),
            "completion_loss_mask": output_mask,
            "policy_release_id": joint_version.policy,
            "tokenizer_version": joint_version.tokenizer,
            "policy_kind": "causal_lm",
            "token_metadata_status": "available",
            "stop_reason": response.get("stop_reason"),
        },
    )
    trace.append(
        "reward_assigned",
        "areal",
        {
            "reward": reward,
            "target_model_call_ids": [binding["model_call_id"]],
            "target_harness_decision_ids": [decision["decision_id"]],
            "source_bridge_sha256": bridge["record_sha256"],
        },
    )
    trace.append(
        "rlvr_terminal_outcome",
        ROUTE_KIND,
        {
            "route_kind": ROUTE_KIND,
            "terminal": True,
            "raw_terminal_reward": reward,
            "evaluator_positive": reward > 0.0,
            "source_bridge_sha256": bridge["record_sha256"],
        },
    )
    trace.reward = reward
    if reward == 0.0:
        trace.success = False
        trace.validity_class = "policy_failure"
        trace.failure_category = "evaluator"
    else:
        # EpisodeTrace's success=True value is intentionally reserved for the
        # frozen 13-event smoke contract.  The exact terminal RLVR outcome is
        # carried by the route-specific terminal event and persisted reward.
        trace.success = None
        trace.validity_class = "valid"
        trace.failure_category = None
    try:
        trace.validate()
    except ValueError as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    return trace


def _validate_rlvr_trace(
    trace: EpisodeTrace,
    *,
    bridge: Mapping[str, object],
    binding: Mapping[str, object],
    joint_version: JointVersion,
) -> None:
    try:
        trace.validate()
    except ValueError as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    _require(
        trace.joint_version == joint_version
        and trace.episode_id == binding.get("episode_id")
        and trace.task_id == str(bridge.get("task_id")),
        "RLVR EpisodeTrace identity or JointVersion differs from the bridge",
    )
    _require(
        tuple(event.kind for event in trace.events) == _TRACE_EVENT_KINDS,
        "RLVR EpisodeTrace event contract differs from the exact workflow route",
    )
    _require(
        trace.events[0].producer == ROUTE_KIND
        and trace.events[2].producer == ROUTE_KIND
        and trace.events[-1].producer == ROUTE_KIND,
        "RLVR EpisodeTrace producer differs from route_kind",
    )
    _require(
        trace.events[0].payload.get("route_kind") == ROUTE_KIND
        and trace.events[0].payload.get("session_id") is None
        and trace.events[0].payload.get("trajectory_id") is None,
        "RLVR EpisodeTrace fabricated session or trajectory identity",
    )
    request = trace.events[2].payload
    response = trace.events[3].payload
    reward_event = trace.events[4].payload
    terminal = trace.events[5].payload
    areal_trace = bridge["areal_trace"]
    bridge_response = areal_trace["model_response"]
    bridge_reward = float(bridge["credit_binding"]["raw_terminal_reward"])
    _require(
        request.get("route_kind") == ROUTE_KIND
        and request.get("model_call_id") == binding.get("model_call_id")
        and request.get("interaction_id") == binding.get("interaction_id")
        and request.get("ordinal") == 0
        and request.get("parent_model_call_id") is None,
        "RLVR EpisodeTrace request binding differs from the bridge",
    )
    _require(
        response.get("model_call_id") == binding.get("model_call_id")
        and response.get("interaction_id") == binding.get("interaction_id")
        and response.get("input_token_ids") == bridge_response["input_tokens"]
        and response.get("output_token_ids") == bridge_response["output_tokens"]
        and response.get("output_versions") == bridge_response["output_versions"]
        and response.get("policy_release_id") == joint_version.policy,
        "RLVR EpisodeTrace response binding differs from the bridge",
    )
    _require(
        response.get("output_token_logprobs") == bridge_response["output_logprobs"],
        "RLVR EpisodeTrace old logprobs differ from the bridge",
    )
    _require(
        reward_event.get("reward") == bridge_reward
        and reward_event.get("target_model_call_ids") == [binding.get("model_call_id")]
        and terminal.get("route_kind") == ROUTE_KIND
        and terminal.get("terminal") is True
        and terminal.get("raw_terminal_reward") == bridge_reward
        and trace.reward == bridge_reward,
        "RLVR EpisodeTrace terminal outcome differs from the bridge",
    )
    if bridge_reward == 0.0:
        _require(
            trace.success is False
            and trace.validity_class == "policy_failure"
            and trace.failure_category == "evaluator",
            "zero-reward RLVR trace must remain a policy failure",
        )
    else:
        _require(
            bridge_reward == 1.0
            and trace.success is None
            and trace.validity_class == "valid"
            and trace.failure_category is None,
            "positive RLVR trace differs from the non-smoke terminal contract",
        )


def _checkpoint_binding(bridge: Mapping[str, object]) -> dict[str, object]:
    harness = bridge["harness"]
    decision = harness["decision"]
    checkpoint = harness["controller_checkpoint_before_decision"]
    schema_version = checkpoint.get("schema_version")
    _require(
        _is_non_empty_string(schema_version),
        "Harness checkpoint schema version is missing",
    )
    checkpoint_sha256 = checkpoint.get("record_sha256")
    if schema_version == "jph.torch-harness-rollout-checkpoint.v1":
        _require(
            _is_sha256(checkpoint_sha256),
            "Torch Harness rollout checkpoint record SHA-256 is missing",
        )
    else:
        checkpoint_sha256 = _sha256(checkpoint)
    return {
        "decision_id": decision["decision_id"],
        "controller_version": decision["controller_version"],
        "checkpoint_schema_version": schema_version,
        "checkpoint_sha256": checkpoint_sha256,
    }


def validate_rlvr_workflow_pre_batch_record(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
) -> dict[str, object]:
    """Revalidate the persistent RLVR pre-batch record without a live object."""

    _exact_mapping(record, _PRE_BATCH_FIELDS, "RLVR pre-batch record")
    _require(
        record.get("schema_version") == RLVR_PRE_BATCH_SCHEMA_VERSION,
        "unknown RLVR pre-batch schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "RLVR pre-batch record hash mismatch",
    )
    _assert_no_secret_fields(record)
    _require(record.get("route_kind") == ROUTE_KIND, "RLVR route kind is invalid")
    bridge = record.get("bridge_record")
    _require(isinstance(bridge, Mapping), "embedded RLVR bridge must be an object")
    bridge_audit, joint_version, binding = _validate_bridge(
        bridge, active_joint_version=active_joint_version
    )
    _require(
        _joint_version(record.get("joint_version")) == joint_version,
        "RLVR pre-batch JointVersion differs from the bridge",
    )
    identity = _exact_mapping(
        record.get("identity"), _PRE_BATCH_IDENTITY_FIELDS, "RLVR pre-batch identity"
    )
    expected_identity = {
        "route_kind": ROUTE_KIND,
        "episode_id": binding["episode_id"],
        "task_id": str(bridge["task_id"]),
        "request_id": bridge["request_id"],
        "model_call_id": binding["model_call_id"],
        "interaction_id": binding["interaction_id"],
        "ordinal": 0,
        "parent_model_call_id": None,
        "session_id": None,
        "trajectory_id": None,
        "joint_version_id": joint_version.version_id,
    }
    _require(dict(identity) == expected_identity, "RLVR pre-batch identity mismatch")
    source = _exact_mapping(
        record.get("source"), _PRE_BATCH_SOURCE_FIELDS, "RLVR pre-batch source"
    )
    _require(
        source.get("bridge_record_sha256") == bridge["record_sha256"]
        and source.get("areal_trace_record_sha256")
        == bridge["areal_trace"]["record_sha256"]
        and source.get("interaction_sidecar_sha256")
        == bridge["interaction_adapter_sidecar"]["sidecar_sha256"]
        and source.get("pre_batch_stage") == RLVR_PRE_BATCH_STAGE
        and source.get("interaction_type")
        == "areal.experimental.openai.types.InteractionWithTokenLogpReward",
        "RLVR pre-batch provenance differs from the bridge or hook stage",
    )
    trace = _episode_trace_from_record(record.get("episode_trace"))
    _validate_rlvr_trace(
        trace,
        bridge=bridge,
        binding=binding,
        joint_version=joint_version,
    )
    trace_sha256 = _sha256(trace.to_dict())
    _require(
        source.get("episode_trace_sha256") == trace_sha256,
        "RLVR EpisodeTrace hash mismatch",
    )
    checkpoint = _exact_mapping(
        record.get("harness_checkpoint_binding"),
        _CHECKPOINT_FIELDS,
        "Harness checkpoint binding",
    )
    _require(
        dict(checkpoint) == _checkpoint_binding(bridge),
        "Harness checkpoint binding differs from the replayed bridge checkpoint",
    )
    archive = record.get("training_archive")
    _require(isinstance(archive, Mapping), "RLVR training archive is missing")
    try:
        archive_audit = validate_bound_training_sample_archive(archive)
    except ValueError as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    _require(
        archive.get("interaction_sidecar") == bridge.get("interaction_adapter_sidecar"),
        "RLVR archive sidecar differs from the bridge",
    )
    samples = archive.get("samples")
    _require(
        isinstance(samples, list) and len(samples) == 1,
        "single-turn RLVR archive requires exactly one sample",
    )
    sample = samples[0]
    _require(isinstance(sample, Mapping), "RLVR archive sample is invalid")
    _require(
        sample.get("included_model_call_ids") == [binding["model_call_id"]]
        and sample.get("included_interaction_ids") == [binding["interaction_id"]]
        and sample.get("tensor_dict") == bridge["areal_trace"]["tensor_dict"],
        "RLVR archive token or interaction binding differs from the bridge",
    )
    _require(
        record.get("evidence_scope") == _PRE_BATCH_EVIDENCE_SCOPE,
        "RLVR pre-batch evidence scope differs from contract",
    )
    return {
        "ok": True,
        "route_kind": ROUTE_KIND,
        "episode_id": identity["episode_id"],
        "task_id": identity["task_id"],
        "request_id": identity["request_id"],
        "model_call_ids": [identity["model_call_id"]],
        "interaction_ids": [identity["interaction_id"]],
        "session_id": None,
        "trajectory_id": None,
        "joint_version_id": joint_version.version_id,
        "reward": float(trace.reward),
        "validity_class": trace.validity_class,
        "trace_sha256": trace_sha256,
        "archive_sha256": archive["record_sha256"],
        "archive_export_style": archive_audit["export_style"],
        "bridge_record_sha256": bridge_audit["record_sha256"],
        "record_sha256": record["record_sha256"],
    }


def _build_pre_batch_record(
    bridge: Mapping[str, object],
    *,
    interaction: Any,
    trace: EpisodeTrace,
    joint_version: JointVersion,
    binding: Mapping[str, object],
    export_style: str,
    turn_discount: float | None,
) -> dict[str, object]:
    archive = archive_premerged_exported_interactions(
        exported_interactions={str(binding["interaction_id"]): interaction},
        interaction_sidecar=bridge["interaction_adapter_sidecar"],
        export_style=export_style,
        turn_discount=turn_discount,
    )
    trace_payload = trace.to_dict()
    record: dict[str, object] = {
        "schema_version": RLVR_PRE_BATCH_SCHEMA_VERSION,
        "route_kind": ROUTE_KIND,
        "identity": {
            "route_kind": ROUTE_KIND,
            "episode_id": binding["episode_id"],
            "task_id": str(bridge["task_id"]),
            "request_id": bridge["request_id"],
            "model_call_id": binding["model_call_id"],
            "interaction_id": binding["interaction_id"],
            "ordinal": 0,
            "parent_model_call_id": None,
            "session_id": None,
            "trajectory_id": None,
            "joint_version_id": joint_version.version_id,
        },
        "joint_version": asdict(joint_version),
        "source": {
            "bridge_record_sha256": bridge["record_sha256"],
            "areal_trace_record_sha256": bridge["areal_trace"]["record_sha256"],
            "interaction_sidecar_sha256": bridge["interaction_adapter_sidecar"][
                "sidecar_sha256"
            ],
            "episode_trace_sha256": _sha256(trace_payload),
            "pre_batch_stage": RLVR_PRE_BATCH_STAGE,
            "interaction_type": (
                "areal.experimental.openai.types.InteractionWithTokenLogpReward"
            ),
        },
        "harness_checkpoint_binding": _checkpoint_binding(bridge),
        "bridge_record": deepcopy(dict(bridge)),
        "episode_trace": trace_payload,
        "training_archive": archive,
        "evidence_scope": dict(_PRE_BATCH_EVIDENCE_SCOPE),
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    validate_rlvr_workflow_pre_batch_record(record, active_joint_version=joint_version)
    return record


def _build_policy_admission(
    pre_batch_record: Mapping[str, object], *, active_joint_version: JointVersion
) -> dict[str, object]:
    p_audit = validate_rlvr_workflow_pre_batch_record(
        pre_batch_record, active_joint_version=active_joint_version
    )
    archive = pre_batch_record["training_archive"]
    model_call_ids = tuple(str(value) for value in p_audit["model_call_ids"])
    _, summary = _validate_and_materialize_samples(
        archive["samples"], expected_model_call_ids=model_call_ids
    )
    samples = json.loads(_canonical_json(archive["samples"]).decode("utf-8"))
    identity = pre_batch_record["identity"]
    record: dict[str, object] = {
        "schema_version": RLVR_WORKFLOW_POLICY_ADMISSION_SCHEMA_VERSION,
        "identity": {
            "route_kind": ROUTE_KIND,
            "episode_id": identity["episode_id"],
            "task_id": identity["task_id"],
            "request_id": identity["request_id"],
            "session_id": None,
            "trajectory_id": None,
            "joint_version_id": identity["joint_version_id"],
        },
        "joint_version": asdict(active_joint_version),
        "model_call_ids": list(model_call_ids),
        "source": {
            "rlvr_pre_batch_record_sha256": pre_batch_record["record_sha256"],
            "training_archive_sha256": archive["record_sha256"],
            "interaction_sidecar_sha256": archive["source_sidecar_sha256"],
            "bridge_record_sha256": pre_batch_record["source"]["bridge_record_sha256"],
        },
        "export_style": archive["export_style"],
        "samples": samples,
        "summary": summary,
        "evidence_scope": dict(_POLICY_EVIDENCE_SCOPE),
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    policy = validate_policy_training_admission(
        record, active_joint_version=active_joint_version
    )
    _require(policy.route_kind == ROUTE_KIND, "Policy admission route kind changed")
    return record


def _mem_fraction_static_provenance(
    bridge: Mapping[str, object],
) -> dict[str, object]:
    policy = bridge.get("policy_binding")
    runtime = (
        policy.get("inference_runtime_contract")
        if isinstance(policy, Mapping)
        else None
    )
    fixed = runtime.get("fixed") if isinstance(runtime, Mapping) else None
    server_args = fixed.get("server_args") if isinstance(fixed, Mapping) else None
    value = (
        server_args.get("mem_fraction_static")
        if isinstance(server_args, Mapping)
        else None
    )
    _require(
        _is_finite_number(value) and 0.0 < float(value) <= 1.0,
        "RLVR bridge does not bind a finite SGLang mem_fraction_static in (0, 1]",
    )
    runtime_hash = policy.get("inference_runtime_contract_sha256")
    _require(_is_sha256(runtime_hash), "inference runtime contract hash is invalid")
    return {
        "value": float(value),
        "source_path": (
            "policy_binding.inference_runtime_contract.fixed."
            "server_args.mem_fraction_static"
        ),
        "runtime_contract_sha256": runtime_hash,
    }


def build_rlvr_workflow_runner_admission(
    *,
    bridge_record: Mapping[str, object],
    rlvr_pre_batch_record: Mapping[str, object],
    episode_trace: EpisodeTrace,
    s_joint_credit: Mapping[str, object],
    active_joint_version: JointVersion,
) -> dict[str, object]:
    """Persist the exact RLVR inputs a T/U/V/W/X/Y runner may load."""

    validate_rlvr_workflow_pre_batch_record(
        rlvr_pre_batch_record, active_joint_version=active_joint_version
    )
    validate_frozen_joint_credit_alignment(
        s_joint_credit, active_joint_version=active_joint_version
    )
    trace_payload = episode_trace.to_dict()
    record: dict[str, object] = {
        "schema_version": RLVR_RUNNER_ADMISSION_SCHEMA_VERSION,
        "route_kind": ROUTE_KIND,
        "bridge_record_sha256": bridge_record["record_sha256"],
        "bridge_record": deepcopy(dict(bridge_record)),
        "rlvr_pre_batch_record": deepcopy(dict(rlvr_pre_batch_record)),
        "episode_trace_sha256": _sha256(trace_payload),
        "episode_trace": trace_payload,
        "s_joint_credit": deepcopy(dict(s_joint_credit)),
        "materialized_estimator": deepcopy(dict(s_joint_credit["estimator"])),
        "mem_fraction_static_provenance": _mem_fraction_static_provenance(
            bridge_record
        ),
        "evidence_scope": dict(_RUNNER_EVIDENCE_SCOPE),
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    validate_rlvr_workflow_runner_admission(
        record, active_joint_version=active_joint_version
    )
    return record


def validate_rlvr_workflow_runner_admission(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
) -> dict[str, object]:
    """Audit a self-contained RLVR runner envelope before any update."""

    _exact_mapping(record, _RUNNER_FIELDS, "RLVR runner admission")
    _require(
        record.get("schema_version") == RLVR_RUNNER_ADMISSION_SCHEMA_VERSION,
        "unknown RLVR runner admission schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "RLVR runner admission hash mismatch",
    )
    _assert_no_secret_fields(record)
    _require(record.get("route_kind") == ROUTE_KIND, "runner route kind is invalid")
    bridge = record.get("bridge_record")
    pre_batch = record.get("rlvr_pre_batch_record")
    s_record = record.get("s_joint_credit")
    _require(
        isinstance(bridge, Mapping)
        and isinstance(pre_batch, Mapping)
        and isinstance(s_record, Mapping),
        "RLVR runner source records are missing",
    )
    bridge_audit, joint_version, binding = _validate_bridge(
        bridge, active_joint_version=active_joint_version
    )
    p_audit = validate_rlvr_workflow_pre_batch_record(
        pre_batch, active_joint_version=joint_version
    )
    _require(
        record.get("bridge_record_sha256")
        == bridge.get("record_sha256")
        == p_audit.get("bridge_record_sha256")
        and pre_batch.get("bridge_record") == bridge,
        "runner bridge differs from its RLVR pre-batch record",
    )
    trace = _episode_trace_from_record(record.get("episode_trace"))
    _validate_rlvr_trace(
        trace,
        bridge=bridge,
        binding=binding,
        joint_version=joint_version,
    )
    _require(
        record.get("episode_trace_sha256")
        == _sha256(trace.to_dict())
        == pre_batch["source"]["episode_trace_sha256"]
        and record.get("episode_trace") == pre_batch.get("episode_trace"),
        "runner EpisodeTrace differs from its RLVR pre-batch record",
    )
    try:
        s_audit = validate_frozen_joint_credit_alignment(
            s_record, active_joint_version=joint_version
        )
    except ValueError as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    materialized_estimator = record.get("materialized_estimator")
    _require(
        isinstance(materialized_estimator, Mapping)
        and materialized_estimator == s_record.get("estimator"),
        "runner materialized estimator differs from the exact S estimator",
    )
    admissions = s_record.get("admissions")
    policy_record = (
        admissions.get("policy_admission_record")
        if isinstance(admissions, Mapping)
        else None
    )
    policy_identity = (
        policy_record.get("identity") if isinstance(policy_record, Mapping) else None
    )
    policy_source = (
        policy_record.get("source") if isinstance(policy_record, Mapping) else None
    )
    _require(
        isinstance(policy_identity, Mapping)
        and policy_identity.get("route_kind") == ROUTE_KIND
        and policy_identity.get("session_id") is None
        and policy_identity.get("trajectory_id") is None,
        "runner S does not contain the dedicated RLVR Policy admission",
    )
    _require(
        isinstance(policy_source, Mapping)
        and policy_source.get("rlvr_pre_batch_record_sha256")
        == pre_batch.get("record_sha256"),
        "runner S is not bound to the same RLVR pre-batch record",
    )
    provenance = _exact_mapping(
        record.get("mem_fraction_static_provenance"),
        _MEMORY_PROVENANCE_FIELDS,
        "mem_fraction_static provenance",
    )
    _require(
        dict(provenance) == _mem_fraction_static_provenance(bridge),
        "runner mem_fraction_static provenance differs from the bridge runtime",
    )
    _require(
        record.get("evidence_scope") == _RUNNER_EVIDENCE_SCOPE,
        "RLVR runner evidence scope differs from contract",
    )
    return {
        "ok": True,
        "route_kind": ROUTE_KIND,
        "episode_id": p_audit["episode_id"],
        "task_id": p_audit["task_id"],
        "model_call_ids": p_audit["model_call_ids"],
        "interaction_ids": p_audit["interaction_ids"],
        "session_id": None,
        "trajectory_id": None,
        "joint_version_id": joint_version.version_id,
        "reward": p_audit["reward"],
        "validity_class": p_audit["validity_class"],
        "bridge_record_sha256": bridge_audit["record_sha256"],
        "rlvr_pre_batch_record_sha256": pre_batch["record_sha256"],
        "s_record_sha256": s_record["record_sha256"],
        "s_policy_sample_count": s_audit["policy_sample_count"],
        "s_harness_action_count": s_audit["harness_action_count"],
        "rollout_sglang_mem_fraction_static": provenance["value"],
        "record_sha256": record["record_sha256"],
    }


@dataclass(frozen=True)
class LoadedRlvrWorkflowRunnerAdmission:
    route_kind: str
    joint_version: JointVersion
    bridge_record: Mapping[str, object]
    bridge_record_sha256: str
    rlvr_pre_batch_record: Mapping[str, object]
    episode_trace: EpisodeTrace
    s_joint_credit: Mapping[str, object]
    materialized_estimator: Mapping[str, object]
    rollout_sglang_mem_fraction_static: float
    mem_fraction_static_source_path: str
    record_sha256: str

    @property
    def summary(self) -> dict[str, object]:
        return {
            "route_kind": self.route_kind,
            "episode_id": self.episode_trace.episode_id,
            "joint_version_id": self.joint_version.version_id,
            "bridge_record_sha256": self.bridge_record_sha256,
            "rlvr_pre_batch_record_sha256": self.rlvr_pre_batch_record["record_sha256"],
            "s_record_sha256": self.s_joint_credit["record_sha256"],
            "materialized_estimator": deepcopy(dict(self.materialized_estimator)),
            "rollout_sglang_mem_fraction_static": (
                self.rollout_sglang_mem_fraction_static
            ),
            "record_sha256": self.record_sha256,
        }


def load_rlvr_workflow_runner_admission(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
) -> LoadedRlvrWorkflowRunnerAdmission:
    """Return typed, revalidated source records for the production runner."""

    _require(
        type(active_joint_version) is JointVersion,
        "runner loader requires the lag-zero active JointVersion",
    )
    validate_rlvr_workflow_runner_admission(
        record, active_joint_version=active_joint_version
    )
    provenance = record["mem_fraction_static_provenance"]
    return LoadedRlvrWorkflowRunnerAdmission(
        route_kind=ROUTE_KIND,
        joint_version=active_joint_version,
        bridge_record=deepcopy(dict(record["bridge_record"])),
        bridge_record_sha256=str(record["bridge_record_sha256"]),
        rlvr_pre_batch_record=deepcopy(dict(record["rlvr_pre_batch_record"])),
        episode_trace=_episode_trace_from_record(record["episode_trace"]),
        s_joint_credit=deepcopy(dict(record["s_joint_credit"])),
        materialized_estimator=deepcopy(dict(record["materialized_estimator"])),
        rollout_sglang_mem_fraction_static=float(provenance["value"]),
        mem_fraction_static_source_path=str(provenance["source_path"]),
        record_sha256=str(record["record_sha256"]),
    )


def _path_within_allowed_root(
    path: str | Path,
    *,
    allowed_root: str | Path,
    require_existing: bool,
) -> Path:
    try:
        root = Path(allowed_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RlvrWorkflowAdmissionError(
            "RLVR runner allowed root does not exist"
        ) from exc
    _require(root.is_dir(), "RLVR runner allowed root is not a directory")
    candidate = Path(path).expanduser()
    try:
        if require_existing:
            resolved = candidate.resolve(strict=True)
        else:
            parent = candidate.parent.resolve(strict=True)
            _require(
                parent.is_dir(), "RLVR runner destination parent is not a directory"
            )
            _require(bool(candidate.name), "RLVR runner destination filename is empty")
            resolved = parent / candidate.name
        common = Path(os.path.commonpath((resolved, root)))
    except (OSError, ValueError) as exc:
        raise RlvrWorkflowAdmissionError("invalid RLVR runner path") from exc
    _require(common == root, "RLVR runner path escapes the allowed root")
    return resolved


def _require_private_owned_directory(directory: Path, label: str) -> None:
    try:
        metadata = directory.stat()
    except OSError as exc:
        raise RlvrWorkflowAdmissionError(f"cannot stat {label}") from exc
    _require(stat.S_ISDIR(metadata.st_mode), f"{label} is not a directory")
    _require(
        stat.S_IMODE(metadata.st_mode) == 0o700,
        f"{label} permissions must be exactly 0700",
    )
    if hasattr(os, "geteuid"):
        _require(metadata.st_uid == os.geteuid(), f"{label} is owned by another user")


def rlvr_runner_admission_path_for_request(
    *,
    output_dir: str | Path,
    request_id: str,
    allowed_root: str | Path,
) -> Path:
    """Derive a safe deterministic filename without embedding a request ID."""

    _require(_is_non_empty_string(request_id), "RLVR runner request ID is missing")
    directory_input = Path(output_dir).expanduser()
    _require(
        not directory_input.is_symlink(),
        "RLVR runner admission directory cannot be a symlink",
    )
    directory = _path_within_allowed_root(
        directory_input,
        allowed_root=allowed_root,
        require_existing=True,
    )
    _require_private_owned_directory(directory, "RLVR runner admission directory")
    identity_sha256 = _sha256(
        {
            "schema_version": "jph.rlvr-workflow-runner-filename.v1",
            "request_id": request_id,
        }
    )
    return directory / f"rlvr-runner-admission-{identity_sha256}.json"


def load_frozen_dual_credit_estimator_template_file(
    input_path: str | Path,
    *,
    allowed_root: str | Path,
) -> FrozenDualCreditEstimatorTemplate:
    """Load one strict frozen scalar template for a multi-request M0 rollout."""

    source_input = Path(input_path).expanduser()
    _require(not source_input.is_symlink(), "frozen estimator cannot be a symlink")
    source = _path_within_allowed_root(
        source_input,
        allowed_root=allowed_root,
        require_existing=True,
    )
    try:
        metadata = source.stat()
    except OSError as exc:  # pragma: no cover - strict resolution owns normal absence
        raise RlvrWorkflowAdmissionError("cannot stat frozen estimator") from exc
    _require(stat.S_ISREG(metadata.st_mode), "frozen estimator is not a file")
    _require(
        stat.S_IMODE(metadata.st_mode) & 0o077 == 0,
        "frozen estimator template permissions are not private",
    )
    if hasattr(os, "geteuid"):
        _require(
            metadata.st_uid == os.geteuid(),
            "frozen estimator template is owned by another user",
        )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RlvrWorkflowAdmissionError("cannot decode frozen estimator") from exc
    _require(isinstance(raw, Mapping), "frozen estimator file is not an object")
    return frozen_dual_credit_estimator_template_from_record(raw)


def write_rlvr_workflow_runner_admission(
    record: Mapping[str, object],
    *,
    output_path: str | Path,
    allowed_root: str | Path,
    active_joint_version: JointVersion,
) -> Path:
    """O_EXCL-write a private runner envelope outside the Git repository."""

    validate_rlvr_workflow_runner_admission(
        record, active_joint_version=active_joint_version
    )
    destination = _path_within_allowed_root(
        output_path,
        allowed_root=allowed_root,
        require_existing=False,
    )
    _require_private_owned_directory(
        destination.parent, "RLVR runner admission directory"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise RlvrWorkflowAdmissionError(
            "RLVR runner admission destination already exists"
        ) from exc
    except OSError as exc:
        raise RlvrWorkflowAdmissionError(
            "cannot create RLVR runner admission destination"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        payload = (
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return destination


def load_rlvr_workflow_runner_admission_file(
    input_path: str | Path,
    *,
    allowed_root: str | Path,
    active_joint_version: JointVersion,
) -> LoadedRlvrWorkflowRunnerAdmission:
    """Read a private envelope from the allowed run root and revalidate it."""

    source_input = Path(input_path).expanduser()
    _require(not source_input.is_symlink(), "RLVR runner admission cannot be a symlink")
    source = _path_within_allowed_root(
        source_input,
        allowed_root=allowed_root,
        require_existing=True,
    )
    try:
        metadata = source.stat()
    except OSError as exc:  # pragma: no cover - strict resolution owns normal absence
        raise RlvrWorkflowAdmissionError("cannot stat RLVR runner admission") from exc
    _require(stat.S_ISREG(metadata.st_mode), "RLVR runner admission is not a file")
    _require(
        stat.S_IMODE(metadata.st_mode) & 0o077 == 0,
        "RLVR runner admission permissions are not private",
    )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RlvrWorkflowAdmissionError("cannot decode RLVR runner admission") from exc
    _require(isinstance(raw, Mapping), "RLVR runner admission file is not an object")
    return load_rlvr_workflow_runner_admission(
        raw, active_joint_version=active_joint_version
    )


@dataclass(frozen=True)
class RlvrWorkflowJointAdmission:
    route_kind: str
    joint_version: JointVersion
    episode_trace: EpisodeTrace
    rlvr_pre_batch_record: Mapping[str, object]
    q_policy_admission: Mapping[str, object]
    r_harness_admission: Mapping[str, object]
    s_joint_credit: Mapping[str, object]
    runner_admission: Mapping[str, object]


def prepare_rlvr_workflow_joint_admission(
    bridge_record: Mapping[str, object],
    *,
    pre_batch_interaction: Any,
    estimator: DualCreditEstimatorSpec,
    active_joint_version: JointVersion,
    export_style: str = "individual",
    turn_discount: float | None = 1.0,
    pre_batch_stage: str = RLVR_PRE_BATCH_STAGE,
) -> RlvrWorkflowJointAdmission:
    """Turn one real RLVR workflow output into P/Q/R/S and a runner envelope."""

    _require(
        type(active_joint_version) is JointVersion,
        "RLVR admission requires the lag-zero active JointVersion",
    )
    _require(
        type(estimator) is DualCreditEstimatorSpec,
        "RLVR admission requires a frozen DualCreditEstimatorSpec",
    )
    _, joint_version, binding = _validate_bridge(
        bridge_record, active_joint_version=active_joint_version
    )
    _require_real_pre_batch_interaction(
        pre_batch_interaction,
        bridge=bridge_record,
        binding=binding,
        pre_batch_stage=pre_batch_stage,
    )
    trace = _build_episode_trace(
        bridge_record,
        joint_version=joint_version,
        binding=binding,
    )
    pre_batch = _build_pre_batch_record(
        bridge_record,
        interaction=pre_batch_interaction,
        trace=trace,
        joint_version=joint_version,
        binding=binding,
        export_style=export_style,
        turn_discount=turn_discount,
    )
    q_record = _build_policy_admission(
        pre_batch, active_joint_version=active_joint_version
    )
    r_batch = admit_pre_batch_bound_harness_action_samples(
        trace=trace,
        active_joint_version=active_joint_version,
        source_training_record_sha256=str(pre_batch["record_sha256"]),
        trace_sha256=str(pre_batch["source"]["episode_trace_sha256"]),
        bound_model_call_ids=(str(binding["model_call_id"]),),
    )
    r_record = r_batch.to_record()
    validate_harness_action_admission_record(
        r_record, active_joint_version=active_joint_version
    )
    try:
        s_record = build_frozen_joint_credit_alignment(
            policy_admission=q_record,
            harness_admission=r_record,
            active_joint_version=active_joint_version,
            estimator=estimator,
        )
        validate_frozen_joint_credit_alignment(
            s_record, active_joint_version=active_joint_version
        )
    except ValueError as exc:
        raise RlvrWorkflowAdmissionError(str(exc)) from exc
    runner_record = build_rlvr_workflow_runner_admission(
        bridge_record=bridge_record,
        rlvr_pre_batch_record=pre_batch,
        episode_trace=trace,
        s_joint_credit=s_record,
        active_joint_version=active_joint_version,
    )
    return RlvrWorkflowJointAdmission(
        route_kind=ROUTE_KIND,
        joint_version=joint_version,
        episode_trace=trace,
        rlvr_pre_batch_record=pre_batch,
        q_policy_admission=q_record,
        r_harness_admission=r_record,
        s_joint_credit=s_record,
        runner_admission=runner_record,
    )
