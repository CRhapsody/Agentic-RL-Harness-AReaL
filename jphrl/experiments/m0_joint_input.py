from __future__ import annotations

"""Strict AA/M0 bridge-to-training-input conversion.

This module does not run either optimizer.  It converts one already validated,
zero-reward AReaL interaction bridge into the existing P/Q/R/S contracts while
retaining the original token tensors and Agent Service identities.
"""

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from jphrl.trajectory.areal_agent_service_adapter import (
    AgentServiceModelCallReceipt,
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    prepare_agent_service_training_record,
    session_receipt_from_start_response,
    trajectory_receipt_from_set_reward_response,
    validate_agent_service_training_record,
)
from jphrl.trajectory.areal_data_proxy_pre_batch import (
    PreBatchTrajectoryExport,
    validate_pre_batch_trajectory_export,
)
from jphrl.trajectory.areal_interaction_sidecar import REQUIRED_TENSOR_FIELDS
from jphrl.trajectory.areal_joint_bridge import (
    ArealJointBridgeError,
    validate_areal_joint_bridge_record,
)
from jphrl.trajectory.areal_policy_admission import (
    build_policy_training_admission,
    validate_policy_training_admission,
)
from jphrl.trajectory.harness_action_admission import (
    admit_real_harness_action_samples,
    validate_harness_action_admission_record,
)
from jphrl.trajectory.hermes_model_call_receipts import (
    HermesModelCallReceipt,
    HermesModelCallReceiptError,
    validate_hermes_model_call_receipts,
)
from jphrl.trajectory.joint_credit_alignment import (
    DualCreditEstimatorSpec,
    build_frozen_joint_credit_alignment,
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.schema import EpisodeTrace, JointVersion


class M0JointInputError(ValueError):
    """Raised when a bridge cannot safely enter the AA joint update."""


_EXPECTED_EVIDENCE_SCOPES: dict[str, dict[str, bool]] = {
    "P": {
        "pre_batch_interaction_binding": True,
        "policy_optimizer_update": False,
        "harness_optimizer_update": False,
    },
    "Q": {
        "pre_batch_interaction_binding": True,
        "policy_samples_admitted": True,
        "policy_advantages_attached": False,
        "policy_optimizer_update": False,
        "harness_optimizer_update": False,
    },
    "R": {
        "pre_batch_interaction_binding": True,
        "harness_action_samples_admitted": True,
        "harness_advantages_attached": False,
        "policy_optimizer_update": False,
        "harness_optimizer_update": False,
    },
    "S": {
        "policy_samples_admitted": True,
        "harness_action_samples_admitted": True,
        "policy_advantages_aligned": True,
        "harness_advantages_aligned": True,
        "policy_optimizer_update": False,
        "harness_optimizer_update": False,
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0JointInputError(message)


def _joint_version(raw: object) -> JointVersion:
    _require(isinstance(raw, Mapping), "bridge JointVersion must be an object")
    _require(
        set(raw) == set(JointVersion.__dataclass_fields__),
        "bridge JointVersion field set differs from schema",
    )
    _require(
        all(isinstance(value, str) and bool(value) for value in raw.values()),
        "bridge JointVersion fields must be non-empty strings",
    )
    try:
        return JointVersion(**dict(raw))
    except TypeError as exc:  # pragma: no cover - exact field check owns this branch
        raise M0JointInputError("invalid bridge JointVersion") from exc


@dataclass(frozen=True)
class M0ModelResponseSnapshot:
    """The token-bearing part of the real AReaL ``ModelResponse``."""

    input_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]
    output_logprobs: tuple[float, ...]
    output_versions: tuple[int, ...]
    stop_reason: str | None


@dataclass(frozen=True)
class M0VerifiedPreBatchInteraction:
    """A checked view of the real object exported by AReaL ``SessionData``."""

    interaction_id: str
    export_style: str
    source: Any
    response_snapshot: M0ModelResponseSnapshot


@dataclass(frozen=True)
class M0JointTrainingInput:
    """The complete, optimizer-free AA input produced from one bridge."""

    trace: EpisodeTrace
    session_receipt: AgentServiceSessionReceipt
    model_call_receipt: AgentServiceModelCallReceipt
    trajectory_receipt: AgentServiceTrajectoryReceipt
    exported_interaction: M0VerifiedPreBatchInteraction
    p_training_record: Mapping[str, object]
    q_policy_admission: Mapping[str, object]
    r_harness_admission: Mapping[str, object]
    s_joint_credit: Mapping[str, object]


def _require_torch_harness_checkpoint(bridge: Mapping[str, object]) -> None:
    harness = bridge.get("harness")
    _require(isinstance(harness, Mapping), "bridge Harness payload is missing")
    checkpoint = harness.get("controller_checkpoint_before_decision")
    _require(isinstance(checkpoint, Mapping), "bridge Harness checkpoint is missing")
    try:
        from jphrl.harness.torch_learning import (
            ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
        )
    except ImportError as exc:
        raise M0JointInputError(
            "Torch is required to verify the production Harness rollout checkpoint"
        ) from exc
    _require(
        checkpoint.get("schema_version") == ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
        "AA requires a real Torch Harness rollout checkpoint",
    )


def _source_binding(bridge: Mapping[str, object]) -> Mapping[str, object]:
    sidecar = bridge.get("interaction_adapter_sidecar")
    _require(isinstance(sidecar, Mapping), "bridge interaction sidecar is missing")
    bindings = sidecar.get("bindings")
    _require(
        isinstance(bindings, list) and len(bindings) == 1,
        "AA M0 requires exactly one source interaction binding",
    )
    binding = bindings[0]
    _require(isinstance(binding, Mapping), "source interaction binding is invalid")
    _require(
        binding.get("route_kind") == "agent-service-session",
        "AA M0 requires an Agent Service session bridge; RLVR IDs cannot mint receipts",
    )
    _require(
        binding.get("parent_interaction_id") is None
        and binding.get("ordinal") == 0,
        "AA M0 requires one root model call with ordinal zero",
    )
    return binding


def _build_trace(
    bridge: Mapping[str, object],
    *,
    joint_version: JointVersion,
    binding: Mapping[str, object],
) -> EpisodeTrace:
    areal_trace = bridge["areal_trace"]
    _require(isinstance(areal_trace, Mapping), "bridge AReaL trace is missing")
    response = areal_trace.get("model_response")
    tensors = areal_trace.get("tensor_dict")
    harness = bridge.get("harness")
    runtime = bridge.get("policy_binding")
    _require(
        isinstance(response, Mapping)
        and isinstance(tensors, Mapping)
        and isinstance(harness, Mapping)
        and isinstance(runtime, Mapping),
        "bridge response, tensors, Harness, or policy binding is missing",
    )
    decision = harness.get("decision")
    state = harness.get("state")
    _require(
        isinstance(decision, Mapping) and isinstance(state, Mapping),
        "bridge Harness decision or state is missing",
    )
    runtime_contract = runtime.get("inference_runtime_contract")
    fixed = runtime_contract.get("fixed") if isinstance(runtime_contract, Mapping) else None
    seed = fixed.get("seed") if isinstance(fixed, Mapping) else None
    _require(type(seed) is int and seed >= 0, "bridge runtime seed is invalid")

    input_tokens = response.get("input_tokens")
    output_tokens = response.get("output_tokens")
    output_logprobs = response.get("output_logprobs")
    output_versions = response.get("output_versions")
    loss_mask = tensors.get("loss_mask")
    _require(
        isinstance(input_tokens, list)
        and isinstance(output_tokens, list)
        and isinstance(output_logprobs, list)
        and isinstance(output_versions, list)
        and isinstance(loss_mask, list)
        and len(loss_mask) == 1
        and isinstance(loss_mask[0], list),
        "bridge token metadata is incomplete",
    )
    output_mask = loss_mask[0][len(input_tokens) :]
    _require(
        output_mask == [1] * len(output_tokens),
        "bridge completion mask differs from the real AReaL action span",
    )

    artifact = bridge.get("harness_artifact")
    _require(isinstance(artifact, Mapping), "bridge Harness artifact is missing")
    harness_spec_hash = artifact.get("sha256")
    _require(
        isinstance(harness_spec_hash, str) and len(harness_spec_hash) == 64,
        "bridge Harness artifact hash is invalid",
    )
    episode_id = binding.get("episode_id")
    model_call_id = binding.get("model_call_id")
    _require(
        isinstance(episode_id, str)
        and bool(episode_id)
        and isinstance(model_call_id, str)
        and bool(model_call_id),
        "bridge episode or model-call identity is invalid",
    )
    trace = EpisodeTrace(
        episode_id=episode_id,
        task_id=str(bridge["task_id"]),
        seed=seed,
        joint_version=joint_version,
        harness_spec_hash=harness_spec_hash,
    )
    trace.append(
        "episode_started",
        "m0-bridge-adapter",
        {
            "source_bridge_sha256": bridge["record_sha256"],
            "request_id": bridge["request_id"],
            "task_id": bridge["task_id"],
            "seed": seed,
        },
    )
    harness_payload = deepcopy(dict(decision))
    harness_payload["state"] = deepcopy(dict(state))
    trace.append("harness_decision", "harness", harness_payload)
    trace.append(
        "model_request",
        "agent-service",
        {
            "model_call_id": model_call_id,
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
            "model_call_id": model_call_id,
            "interaction_id": binding["interaction_id"],
            "input_token_ids": deepcopy(input_tokens),
            "output_token_ids": deepcopy(output_tokens),
            "output_token_logprobs": deepcopy(output_logprobs),
            "output_versions": deepcopy(output_versions),
            "completion_loss_mask": deepcopy(output_mask),
            "policy_version": joint_version.policy,
            "tokenizer_version": joint_version.tokenizer,
            "policy_kind": "causal_lm",
            "token_metadata_status": "available",
            "stop_reason": response.get("stop_reason"),
        },
    )
    decision_id = decision.get("decision_id")
    trace.append(
        "reward_assigned",
        "areal",
        {
            "reward": 0.0,
            "target_model_call_ids": [model_call_id],
            "target_harness_decision_ids": [decision_id],
            "source_bridge_sha256": bridge["record_sha256"],
        },
    )
    trace.append(
        "episode_ended",
        "m0-bridge-adapter",
        {
            "success": False,
            "reward": 0.0,
            "validity_class": "policy_failure",
            "failure_category": "evaluator",
            "termination_reason": "real AReaL terminal reward was zero",
        },
    )
    trace.reward = 0.0
    trace.success = False
    trace.validity_class = "policy_failure"
    trace.failure_category = "evaluator"
    try:
        trace.validate()
    except ValueError as exc:
        raise M0JointInputError(str(exc)) from exc
    return trace


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


def _verified_exported_interaction(
    bridge: Mapping[str, object],
    binding: Mapping[str, object],
    *,
    session: AgentServiceSessionReceipt,
    trajectory: AgentServiceTrajectoryReceipt,
    pre_batch_exported_interactions: Mapping[str, Any],
    export_style: str,
    turn_discount: float,
) -> M0VerifiedPreBatchInteraction:
    areal_trace = bridge["areal_trace"]
    response = areal_trace["model_response"]
    interaction = areal_trace["interaction"]
    tensors = areal_trace["tensor_dict"]
    _require(
        set(tensors) == set(REQUIRED_TENSOR_FIELDS),
        "AA requires the exact six-field pre-batch AReaL tensor contract",
    )
    _require(
        isinstance(pre_batch_exported_interactions, Mapping),
        "AA requires the real pre-batch interaction mapping",
    )
    try:
        validate_pre_batch_trajectory_export(
            PreBatchTrajectoryExport(
                session_id=session.session_id,
                trajectory_id=trajectory.trajectory_id,
                exported_interactions=pre_batch_exported_interactions,
                export_style=export_style,
                turn_discount=turn_discount,
            )
        )
    except ValueError as exc:
        raise M0JointInputError(str(exc)) from exc
    _require(
        list(pre_batch_exported_interactions) == [binding["interaction_id"]],
        "pre-batch export differs from the bridge interaction identity",
    )
    source = pre_batch_exported_interactions[str(binding["interaction_id"])]
    source_type = type(source)
    _require(
        source_type.__module__ == "areal.experimental.openai.types"
        and source_type.__name__ == "InteractionWithTokenLogpReward",
        "AA requires a real AReaL InteractionWithTokenLogpReward export",
    )
    source_response = getattr(source, "model_response", None)
    _require(source_response is not None, "pre-batch interaction has no ModelResponse")
    _require(
        list(source_response.input_tokens) == response["input_tokens"]
        and list(source_response.output_tokens) == response["output_tokens"]
        and list(source_response.output_versions) == response["output_versions"],
        "pre-batch ModelResponse token identity differs from the bridge",
    )
    _require(
        len(source_response.output_logprobs) == len(response["output_logprobs"])
        and all(
            abs(float(actual) - float(expected)) <= 1e-6
            for actual, expected in zip(
                source_response.output_logprobs,
                response["output_logprobs"],
            )
        ),
        "pre-batch ModelResponse logprobs differ from the bridge",
    )
    _require(
        getattr(source_response, "stop_reason", None) == response.get("stop_reason"),
        "pre-batch ModelResponse stop reason differs from the bridge",
    )
    _require(
        getattr(source, "chat_template_type", None)
        == interaction.get("chat_template_type"),
        "pre-batch interaction chat-template type differs from the bridge",
    )
    source_tensors = source.to_tensor_dict()
    _require(
        isinstance(source_tensors, Mapping)
        and set(source_tensors) == set(REQUIRED_TENSOR_FIELDS)
        and _plain(source_tensors) == tensors,
        "pre-batch six-field tensors differ from the bridge roundtrip",
    )
    snapshot = M0ModelResponseSnapshot(
        input_tokens=tuple(response["input_tokens"]),
        output_tokens=tuple(response["output_tokens"]),
        output_logprobs=tuple(float(value) for value in response["output_logprobs"]),
        output_versions=tuple(response["output_versions"]),
        stop_reason=response.get("stop_reason"),
    )
    return M0VerifiedPreBatchInteraction(
        interaction_id=str(binding["interaction_id"]),
        export_style=export_style,
        source=source,
        response_snapshot=snapshot,
    )


def prepare_m0_joint_training_input_from_receipts(
    bridge_record: Mapping[str, object],
    *,
    session_receipt: AgentServiceSessionReceipt,
    hermes_model_call_receipt: HermesModelCallReceipt,
    trajectory_receipt: AgentServiceTrajectoryReceipt,
    pre_batch_exported_interactions: Mapping[str, Any],
    estimator: DualCreditEstimatorSpec,
    export_style: str = "individual",
    turn_discount: float = 1.0,
) -> M0JointTrainingInput:
    """Build AA P/Q/R/S from sanitized receipts and one real pre-batch export.

    The source must be a current bridge with a replayable Torch Harness rollout
    checkpoint and an Agent Service sidecar.  Only a real zero terminal reward
    is accepted; positive reward is deliberately left for a future, separately
    specified successful GSM8K EpisodeTrace conversion.
    """

    _require(isinstance(bridge_record, Mapping), "bridge record must be an object")
    _require(
        type(turn_discount) in (int, float)
        and math.isfinite(float(turn_discount))
        and 0.0 <= float(turn_discount) <= 1.0,
        "AA M0 requires the real finite SessionData reward discount in [0, 1]",
    )
    bridge = deepcopy(dict(bridge_record))
    policy = bridge.get("policy_binding")
    _require(isinstance(policy, Mapping), "bridge policy binding is missing")
    expected_version = policy.get("expected_inference_engine_version")
    _require(
        type(expected_version) is int and expected_version >= 0,
        "bridge expected policy version is invalid",
    )
    try:
        audit = validate_areal_joint_bridge_record(
            bridge,
            expected_policy_version=expected_version,
        )
    except ArealJointBridgeError as exc:
        raise M0JointInputError(str(exc)) from exc
    _require_torch_harness_checkpoint(bridge)
    _require(
        float(audit["reward"]) == 0.0,
        "this AA M0 policy_failure debug contract accepts only a real zero-reward "
        "bridge; successful episodes require their separately specified trace path",
    )
    binding = _source_binding(bridge)
    joint_version = _joint_version(bridge["joint_version"])
    _require(
        type(session_receipt) is AgentServiceSessionReceipt,
        "AA M0 requires a sanitized Agent Service session receipt",
    )
    _require(
        type(trajectory_receipt) is AgentServiceTrajectoryReceipt,
        "AA M0 requires a sanitized Agent Service trajectory receipt",
    )
    _require(
        type(hermes_model_call_receipt) is HermesModelCallReceipt,
        "AA M0 requires one exact five-field Hermes receipt",
    )
    try:
        validate_hermes_model_call_receipts(
            [hermes_model_call_receipt],
            expected_session_id=session_receipt.session_id,
        )
    except HermesModelCallReceiptError as exc:
        raise M0JointInputError(str(exc)) from exc
    _require(
        hermes_model_call_receipt.ordinal == 0
        and hermes_model_call_receipt.parent_model_call_id is None,
        "AA M0 requires one root Hermes model call with ordinal zero",
    )
    session = session_receipt
    trajectory = trajectory_receipt
    _require(
        session.session_id == binding.get("session_id"),
        "start-session receipt differs from the bridge session",
    )
    _require(
        trajectory.session_id == binding.get("session_id")
        and trajectory.trajectory_id == binding.get("trajectory_id")
        and trajectory.interaction_count == 1,
        "reward trajectory receipt differs from the bridge trajectory",
    )
    _require(
        hermes_model_call_receipt.model_call_id == binding.get("model_call_id")
        and hermes_model_call_receipt.interaction_id
        == binding.get("interaction_id")
        and hermes_model_call_receipt.session_id == binding.get("session_id"),
        "Hermes receipt differs from the bridge interaction binding",
    )
    model_call = AgentServiceModelCallReceipt(
        model_call_id=hermes_model_call_receipt.model_call_id,
        interaction_id=hermes_model_call_receipt.interaction_id,
        ordinal=hermes_model_call_receipt.ordinal,
        parent_model_call_id=hermes_model_call_receipt.parent_model_call_id,
    )
    trace = _build_trace(
        bridge,
        joint_version=joint_version,
        binding=binding,
    )
    exported = _verified_exported_interaction(
        bridge,
        binding,
        session=session,
        trajectory=trajectory,
        pre_batch_exported_interactions=pre_batch_exported_interactions,
        export_style=export_style,
        turn_discount=turn_discount,
    )
    try:
        p_record = prepare_agent_service_training_record(
            trace=trace,
            session=session,
            model_calls=[model_call],
            trajectory=trajectory,
            exported_interactions=pre_batch_exported_interactions,
            export_style=export_style,
            turn_discount=turn_discount,
        )
        q_record = build_policy_training_admission(
            p_record,
            active_joint_version=joint_version,
        )
        r_batch = admit_real_harness_action_samples(
            trace=trace,
            active_joint_version=joint_version,
            pre_batch_training_record=p_record,
        )
        r_record = r_batch.to_record()
        s_record = build_frozen_joint_credit_alignment(
            policy_admission=q_record,
            harness_admission=r_batch,
            active_joint_version=joint_version,
            estimator=estimator,
        )
        validate_agent_service_training_record(p_record)
        validate_policy_training_admission(
            q_record,
            active_joint_version=joint_version,
        )
        validate_harness_action_admission_record(
            r_record,
            active_joint_version=joint_version,
        )
        validate_frozen_joint_credit_alignment(
            s_record,
            active_joint_version=joint_version,
        )
    except ValueError as exc:
        raise M0JointInputError(str(exc)) from exc

    for label, record in (
        ("P", p_record),
        ("Q", q_record),
        ("R", r_record),
        ("S", s_record),
    ):
        scope = record.get("evidence_scope")
        _require(
            isinstance(scope, Mapping)
            and dict(scope) == _EXPECTED_EVIDENCE_SCOPES[label],
            f"{label} evidence scope differs from the optimizer-free contract",
        )

    return M0JointTrainingInput(
        trace=trace,
        session_receipt=session,
        model_call_receipt=model_call,
        trajectory_receipt=trajectory,
        exported_interaction=exported,
        p_training_record=p_record,
        q_policy_admission=q_record,
        r_harness_admission=r_record,
        s_joint_credit=s_record,
    )


def prepare_m0_joint_training_input(
    bridge_record: Mapping[str, object],
    *,
    start_session_response: Mapping[str, object],
    set_reward_response: Mapping[str, object],
    pre_batch_exported_interactions: Mapping[str, Any],
    estimator: DualCreditEstimatorSpec,
    export_style: str = "individual",
    turn_discount: float = 1.0,
    session_index: int = 0,
    hermes_model_call_receipt: HermesModelCallReceipt | None = None,
) -> M0JointTrainingInput:
    """Raw HTTP compatibility entry; secrets are stripped before the core call.

    New Agent Service deployments should pass the exact Hermes receipt.  The
    optional fallback rehydrates it from an already validated agent-service
    bridge solely for compatibility with earlier bridge-first callers.
    """

    try:
        session = session_receipt_from_start_response(
            start_session_response,
            session_index=session_index,
        )
        trajectory = trajectory_receipt_from_set_reward_response(
            set_reward_response
        )
    except ValueError as exc:
        raise M0JointInputError(str(exc)) from exc
    if hermes_model_call_receipt is None:
        _require(
            isinstance(bridge_record, Mapping),
            "bridge record must be an object",
        )
        binding = _source_binding(bridge_record)
        hermes_model_call_receipt = HermesModelCallReceipt(
            model_call_id=str(binding["model_call_id"]),
            interaction_id=str(binding["interaction_id"]),
            ordinal=int(binding["ordinal"]),
            parent_model_call_id=None,
            session_id=str(binding["session_id"]),
        )
    return prepare_m0_joint_training_input_from_receipts(
        bridge_record,
        session_receipt=session,
        hermes_model_call_receipt=hermes_model_call_receipt,
        trajectory_receipt=trajectory,
        pre_batch_exported_interactions=pre_batch_exported_interactions,
        estimator=estimator,
        export_style=export_style,
        turn_discount=turn_discount,
    )
