from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jphrl.harness.controller import HarnessDecision, HarnessState
from jphrl.harness.learning import TabularHarnessController
from jphrl.harness.spec import HarnessAction

from .areal_interaction_sidecar import (
    validate_interaction_adapter_sidecar,
)
from .areal_trace_contract import (
    ArealTraceContractError,
    build_areal_trace_record,
    validate_areal_trace_record,
)
from .schema import EpisodeTrace, JointVersion

SCHEMA_VERSION = "jph.areal-joint-interaction-bridge.v3"
INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION = "jph.sglang-inference-runtime.v2"
DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION = (
    "jph.sglang-inference-runtime.v3"
)
LEGACY_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION = "jph.sglang-inference-runtime.v1"
CONTEXT_BUILDER_VERSION = "gsm8k-harness-prompt-v1"
HARNESS_ARTIFACT_VERSION = "gsm8k-bounded-prompt-actions-v1"
GENERATION_LOGPROB_MODES = frozenset(
    {
        "standard-log-of-softmax-v1",
        "original-log-softmax-v1",
    }
)
HARNESS_INSTRUCTIONS: dict[HarnessAction, str] = {
    HarnessAction.DIRECT: (
        "Solve the problem directly. Keep the reasoning focused and end with a clear "
        "numeric answer."
    ),
    HarnessAction.RETRIEVE_SKILL: (
        "Before solving, recall the relevant arithmetic or algebra pattern, then apply "
        "that pattern to the problem."
    ),
    HarnessAction.VERIFY: (
        "Solve the problem, then independently verify the arithmetic before giving the "
        "final numeric answer."
    ),
    HarnessAction.REPLAN: (
        "Write a short plan, check whether it covers every quantity in the question, "
        "then carry it out."
    ),
    HarnessAction.COMPRESS: (
        "First compress the problem into known quantities, the unknown quantity, and the "
        "required operations; then solve it."
    ),
}


class ArealJointBridgeError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealJointBridgeError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def inference_runtime_contract_sha256(
    contract: Mapping[str, object],
) -> str:
    """Return the content identity of a normalized, non-secret launch contract."""

    schema_version = contract.get("schema_version")
    _require(
        schema_version
        in {
            LEGACY_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
            INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
            DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
        },
        "unknown inference runtime contract schema",
    )
    identity = contract.get("identity")
    fixed = contract.get("fixed")
    treatment = contract.get("treatment")
    _require(
        isinstance(identity, Mapping)
        and isinstance(fixed, Mapping)
        and isinstance(treatment, Mapping),
        "inference runtime identity, fixed fields, and treatment must be objects",
    )
    _require(
        isinstance(identity.get("run_id"), str) and bool(identity["run_id"]),
        "inference runtime run ID cannot be empty",
    )
    _require(
        identity.get("screen_pair_id") is None
        or (
            isinstance(identity.get("screen_pair_id"), str)
            and bool(identity["screen_pair_id"])
        ),
        "screen pair ID must be null or a non-empty string",
    )
    required_fixed = {
        "areal_commit",
        "areal_version",
        "behavior_revision",
        "clean_environment_policy",
        "cuda_runtime_version",
        "cuda_visible_devices",
        "dataset_revision",
        "dataset_selection",
        "driver_version",
        "generation",
        "gpu_name",
        "gpu_uuid",
        "physical_gpu_id",
        "python_version",
        "project_commit",
        "rollout",
        "seed",
        "server_args",
        "sglang_environment",
        "sglang_version",
        "torch_version",
        "transformers_version",
    }
    if schema_version == DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION:
        required_fixed -= {
            "cuda_visible_devices",
            "gpu_name",
            "gpu_uuid",
            "physical_gpu_id",
        }
        required_fixed |= {
            "cuda_visible_devices_by_rank",
            "gpu_names",
            "gpu_uuids",
            "physical_gpu_ids",
        }
    _require(
        set(fixed) == required_fixed,
        "inference runtime fixed field set differs from contract",
    )
    if schema_version == LEGACY_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION:
        required_treatment = {
            "generation_logprob_mode",
            "sglang_return_original_logprob",
        }
    else:
        required_treatment = {
            "disable_cuda_graph",
            "experimental_axis",
            "generation_logprob_mode",
            "sglang_return_original_logprob",
        }
    _require(
        set(treatment) == required_treatment,
        "inference runtime treatment field set differs from contract",
    )
    if schema_version in {
        INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
        DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
    }:
        server_args = fixed.get("server_args")
        _require(
            isinstance(server_args, Mapping)
            and type(server_args.get("disable_cuda_graph")) is bool
            and server_args["disable_cuda_graph"] is treatment["disable_cuda_graph"],
            "CUDA Graph treatment differs from effective server args",
        )
        _require(
            treatment.get("experimental_axis")
            in {
                "none-v1",
                "generation-logprob-formula-v1",
                "cuda-graph-v1",
            },
            "unknown inference runtime experimental axis",
        )
    try:
        payload = _canonical_json(dict(contract))
    except (TypeError, ValueError) as exc:
        raise ArealJointBridgeError(
            "inference runtime contract is not finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def prompt_context_chars(messages: Sequence[Mapping[str, str]]) -> int:
    """Count the UTF-8 prompt representation used by the bridge state encoder."""

    return len(_canonical_json([dict(message) for message in messages]).decode("utf-8"))


def deterministic_bridge_request_id(
    *,
    task_id: int,
    dataset_selection: str,
    base_messages: Sequence[Mapping[str, str]],
) -> str:
    """Keep request identity identical across paired, fully restarted cells."""

    _require(type(task_id) is int and task_id >= 0, "task ID must be non-negative")
    _require(
        isinstance(dataset_selection, str) and bool(dataset_selection),
        "dataset selection cannot be empty",
    )
    payload = {
        "schema_version": "jph.areal-bridge-request-identity.v1",
        "task_id": task_id,
        "dataset_selection": dataset_selection,
        "base_messages": [dict(message) for message in base_messages],
    }
    return f"jph-areal-{_sha256(payload)[:32]}"


def harness_artifact_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "version": HARNESS_ARTIFACT_VERSION,
        "context_builder_version": CONTEXT_BUILDER_VERSION,
        "instructions": {
            action.value: HARNESS_INSTRUCTIONS[action] for action in HarnessAction
        },
    }
    payload["sha256"] = _sha256(payload)
    return payload


def inject_harness_instruction(
    messages: Sequence[Mapping[str, str]],
    action: HarnessAction,
) -> tuple[list[dict[str, str]], str]:
    """Return a copied prompt with an auditable Harness instruction prepended."""

    base: list[dict[str, str]] = []
    for message in messages:
        _require(isinstance(message, Mapping), "each prompt message must be an object")
        role = message.get("role")
        content = message.get("content")
        _require(
            isinstance(role, str) and role and isinstance(content, str),
            "prompt messages require non-empty role and string content",
        )
        base.append({"role": role, "content": content})
    _require(bool(base), "base prompt messages cannot be empty")

    instruction = HARNESS_INSTRUCTIONS[action]
    harness_message = {
        "role": "system",
        "content": f"JPH Harness action={action.value}. {instruction}",
    }
    return [harness_message, *base], instruction


def build_joint_version(
    *,
    policy_release_id: str,
    harness_controller_version: str,
    areal_commit: str,
    behavior_revision: str,
    dataset_revision: str,
    dataset_selection: str,
    sglang_version: str,
    generation_logprob_mode: str,
    inference_runtime_contract_sha256: str,
) -> JointVersion:
    for name, value in (
        ("policy_release_id", policy_release_id),
        ("harness_controller_version", harness_controller_version),
        ("areal_commit", areal_commit),
        ("behavior_revision", behavior_revision),
        ("dataset_revision", dataset_revision),
        ("dataset_selection", dataset_selection),
        ("sglang_version", sglang_version),
        ("generation_logprob_mode", generation_logprob_mode),
        (
            "inference_runtime_contract_sha256",
            inference_runtime_contract_sha256,
        ),
    ):
        _require(isinstance(value, str) and value, f"{name} cannot be empty")
    _require(
        generation_logprob_mode in GENERATION_LOGPROB_MODES,
        "unknown generation log-prob mode",
    )
    _require(
        len(inference_runtime_contract_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in inference_runtime_contract_sha256
        ),
        "inference runtime contract hash must be a lowercase SHA-256",
    )
    artifact = harness_artifact_payload()
    return JointVersion(
        policy=(
            f"{policy_release_id}:sglang={sglang_version}"
            f":generation-logprob={generation_logprob_mode}"
            f":runtime={inference_runtime_contract_sha256}"
        ),
        harness_controller=harness_controller_version,
        harness_artifact=f"{HARNESS_ARTIFACT_VERSION}@{artifact['sha256']}",
        tool_schema="no-tools-single-turn-v1",
        parser=f"areal-gsm8k-reward@{areal_commit}",
        environment=(f"gsm8k-test@{dataset_revision}:selection={dataset_selection}"),
        evaluator=f"areal-gsm8k-reward@{areal_commit}",
        tokenizer=f"hf-tokenizer@{behavior_revision}",
        context_builder=CONTEXT_BUILDER_VERSION,
    )


def _decision_payload(decision: HarnessDecision) -> dict[str, object]:
    payload = asdict(decision)
    payload["action"] = decision.action.value
    payload["action_ids"] = list(decision.action_ids)
    payload["action_mask"] = list(decision.action_mask)
    payload["pre_mask_logits"] = list(decision.pre_mask_logits)
    return payload


def _validate_harness_decision(
    decision: Mapping[str, object],
    state: Mapping[str, object],
    joint_version: JointVersion,
) -> None:
    _require(
        isinstance(decision.get("decision_id"), str) and bool(decision["decision_id"]),
        "Harness decision ID must be a non-empty string",
    )
    _require(
        isinstance(decision.get("controller_version"), str)
        and bool(decision["controller_version"]),
        "Harness controller version must be a non-empty string",
    )
    try:
        harness_state = HarnessState(**state)
        parsed = HarnessDecision(
            decision_id=decision["decision_id"],
            action=HarnessAction(str(decision["action"])),
            old_harness_logprob=float(decision["old_harness_logprob"]),
            controller_version=decision["controller_version"],
            action_ids=tuple(decision["action_ids"]),
            action_mask=tuple(decision["action_mask"]),
            pre_mask_logits=tuple(
                float(value) for value in decision["pre_mask_logits"]
            ),
            harness_loss_mask=decision["harness_loss_mask"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArealJointBridgeError(
            "invalid Harness state or decision payload"
        ) from exc
    trace = EpisodeTrace(
        episode_id="bridge-decision-validation",
        task_id="bridge",
        seed=0,
        joint_version=joint_version,
        harness_spec_hash=str(harness_artifact_payload()["sha256"]),
    )
    payload = _decision_payload(parsed)
    payload["state"] = asdict(harness_state)
    trace.append("harness_decision", "harness", payload)
    try:
        trace.validate()
    except ValueError as exc:
        raise ArealJointBridgeError(str(exc)) from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return _sha256(unsigned)


def build_areal_joint_bridge_record(
    *,
    task_id: int,
    request_id: str,
    joint_version: JointVersion,
    expected_policy_version: int,
    harness_state: HarnessState,
    harness_decision: HarnessDecision,
    harness_controller_checkpoint: Mapping[str, object],
    base_messages: Sequence[Mapping[str, str]],
    effective_messages: Sequence[Mapping[str, str]],
    base_input_tokens: Sequence[int],
    effective_input_tokens: Sequence[int],
    model_response: Any,
    interaction: Any,
    tensor_dict: Mapping[str, Any],
    project_commit: str,
    areal_commit: str,
    behavior_snapshot_path: str,
    behavior_revision: str,
    dataset_selection: str,
    sglang_version: str,
    generation_logprob_mode: str,
    inference_runtime_contract: Mapping[str, object],
    interaction_adapter_sidecar: Mapping[str, object],
) -> dict[str, object]:
    """Bind a real AReaL interaction and a prompt-effective Harness decision."""

    areal_trace = build_areal_trace_record(
        task_id=task_id,
        request_id=request_id,
        model_response=model_response,
        interaction=interaction,
        tensor_dict=tensor_dict,
        areal_commit=areal_commit,
        behavior_snapshot_path=behavior_snapshot_path,
        behavior_revision=behavior_revision,
    )
    instruction = HARNESS_INSTRUCTIONS[harness_decision.action]
    runtime_contract = dict(inference_runtime_contract)
    runtime_contract_sha256 = inference_runtime_contract_sha256(runtime_contract)
    sidecar = dict(interaction_adapter_sidecar)
    sidecar_audit = validate_interaction_adapter_sidecar(sidecar)
    _require(
        sidecar_audit["binding_count"] == 1,
        "single-interaction bridge requires exactly one adapter binding",
    )
    binding = sidecar["bindings"][0]
    _require(
        binding["joint_version_id"] == joint_version.version_id,
        "interaction sidecar JointVersion differs from bridge",
    )
    _require(
        binding["interaction_id"] == interaction.interaction_id,
        "interaction sidecar differs from the AReaL interaction",
    )
    _require(
        len(project_commit) == 40
        and all(character in "0123456789abcdef" for character in project_commit),
        "project commit must be a lowercase 40-character Git object ID",
    )
    _require(
        request_id
        == deterministic_bridge_request_id(
            task_id=task_id,
            dataset_selection=dataset_selection,
            base_messages=base_messages,
        ),
        "request ID differs from deterministic prompt binding",
    )
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "origin": {"project_commit": project_commit},
        "evidence_scope": {
            "real_areal_rollout": True,
            "harness_decision_changed_prompt": True,
            "joint_interaction_sidecar": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
            "joint_learning_claim": False,
        },
        "task_id": task_id,
        "request_id": request_id,
        "episode_id": binding["episode_id"],
        "joint_version": asdict(joint_version),
        "joint_version_id": joint_version.version_id,
        "interaction_adapter_sidecar": sidecar,
        "harness_artifact": harness_artifact_payload(),
        "harness": {
            "state": asdict(harness_state),
            "decision": _decision_payload(harness_decision),
            "controller_checkpoint_before_decision": dict(
                harness_controller_checkpoint
            ),
            "applied_instruction": instruction,
        },
        "prompt_binding": {
            "base_messages": [dict(message) for message in base_messages],
            "effective_messages": [dict(message) for message in effective_messages],
            "base_messages_sha256": _sha256(list(base_messages)),
            "effective_messages_sha256": _sha256(list(effective_messages)),
            "base_input_tokens": list(base_input_tokens),
            "effective_input_tokens": list(effective_input_tokens),
            "prompt_tokens_changed": list(base_input_tokens)
            != list(effective_input_tokens),
        },
        "policy_binding": {
            "policy_release_id": joint_version.policy,
            "expected_inference_engine_version": expected_policy_version,
            "dataset_selection": dataset_selection,
            "sglang_version": sglang_version,
            "generation_logprob_mode": generation_logprob_mode,
            "inference_runtime_contract": runtime_contract,
            "inference_runtime_contract_sha256": runtime_contract_sha256,
        },
        "credit_binding": {
            "status": "raw-terminal-outcome-only",
            "raw_terminal_reward": float(interaction.reward),
            "policy_target_model_call_id": binding["model_call_id"],
            "harness_target_decision_id": harness_decision.decision_id,
            "policy_advantage": None,
            "harness_advantage": None,
        },
        "areal_trace": areal_trace,
    }
    record["record_sha256"] = _record_sha256(record)
    validate_areal_joint_bridge_record(
        record, expected_policy_version=expected_policy_version
    )
    return record


def validate_areal_joint_bridge_record(
    record: Mapping[str, object],
    *,
    expected_policy_version: int | None = None,
) -> dict[str, object]:
    _require(record.get("schema_version") == SCHEMA_VERSION, "unknown schema version")
    _require(
        record.get("record_sha256") == _record_sha256(record), "record hash mismatch"
    )

    scope = record.get("evidence_scope")
    _require(isinstance(scope, Mapping), "evidence_scope must be an object")
    expected_scope = {
        "real_areal_rollout": True,
        "harness_decision_changed_prompt": True,
        "joint_interaction_sidecar": True,
        "policy_optimizer_update": False,
        "harness_optimizer_update": False,
        "joint_learning_claim": False,
    }
    _require(
        dict(scope) == expected_scope, "evidence scope differs from bridge contract"
    )
    origin = record.get("origin")
    _require(isinstance(origin, Mapping), "origin must be an object")
    project_commit = origin.get("project_commit")
    _require(
        isinstance(project_commit, str)
        and len(project_commit) == 40
        and all(character in "0123456789abcdef" for character in project_commit),
        "origin project commit is not a full Git object ID",
    )

    joint_payload = record.get("joint_version")
    _require(isinstance(joint_payload, Mapping), "joint_version must be an object")
    try:
        joint_version = JointVersion(**joint_payload)
    except TypeError as exc:
        raise ArealJointBridgeError("invalid JointVersion fields") from exc
    _require(
        record.get("joint_version_id") == joint_version.version_id,
        "JointVersion ID mismatch",
    )

    sidecar = record.get("interaction_adapter_sidecar")
    _require(
        isinstance(sidecar, Mapping),
        "interaction adapter sidecar must be an object",
    )
    try:
        sidecar_audit = validate_interaction_adapter_sidecar(sidecar)
    except ValueError as exc:
        raise ArealJointBridgeError(str(exc)) from exc
    _require(
        sidecar_audit["binding_count"] == 1,
        "single-interaction bridge requires exactly one adapter binding",
    )
    binding = sidecar["bindings"][0]
    _require(
        sidecar_audit["episode_id"] == record.get("episode_id")
        and sidecar_audit["joint_version_id"] == joint_version.version_id,
        "interaction sidecar episode or JointVersion differs from bridge",
    )

    artifact = record.get("harness_artifact")
    _require(artifact == harness_artifact_payload(), "Harness artifact mismatch")
    _require(
        joint_version.harness_artifact
        == f"{HARNESS_ARTIFACT_VERSION}@{artifact['sha256']}",
        "JointVersion does not bind the Harness artifact",
    )

    harness = record.get("harness")
    _require(isinstance(harness, Mapping), "harness must be an object")
    state = harness.get("state")
    decision = harness.get("decision")
    controller_checkpoint = harness.get("controller_checkpoint_before_decision")
    _require(
        isinstance(state, Mapping)
        and isinstance(decision, Mapping)
        and isinstance(controller_checkpoint, Mapping),
        "Harness state, decision, and controller checkpoint must be objects",
    )
    _validate_harness_decision(decision, state, joint_version)
    action = HarnessAction(str(decision["action"]))
    try:
        if (
            controller_checkpoint.get("schema_version")
            == TabularHarnessController.schema_version
        ):
            restored_controller = TabularHarnessController.from_checkpoint(
                controller_checkpoint
            )
        else:
            from jphrl.harness.torch_learning import (
                ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
                load_torch_harness_rollout_checkpoint,
            )

            _require(
                controller_checkpoint.get("schema_version")
                == ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
                "unknown Harness rollout checkpoint schema",
            )
            restored_controller = load_torch_harness_rollout_checkpoint(
                controller_checkpoint
            )
        replayed_decision = restored_controller.choose(HarnessState(**state))
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise ArealJointBridgeError(
            "Harness controller checkpoint cannot replay the decision"
        ) from exc
    _require(
        restored_controller.version == joint_version.harness_controller,
        "Harness checkpoint version differs from JointVersion",
    )
    for field in (
        "action",
        "old_harness_logprob",
        "controller_version",
        "action_ids",
        "action_mask",
        "pre_mask_logits",
        "harness_loss_mask",
    ):
        recorded_value = decision[field]
        replayed_value = getattr(replayed_decision, field)
        if field == "action":
            replayed_value = replayed_value.value
        elif isinstance(replayed_value, tuple):
            replayed_value = list(replayed_value)
        _require(
            recorded_value == replayed_value,
            f"Harness decision field {field} differs from checkpoint replay",
        )
    _require(
        harness.get("applied_instruction") == HARNESS_INSTRUCTIONS[action],
        "applied instruction differs from the chosen Harness action",
    )

    prompt = record.get("prompt_binding")
    _require(isinstance(prompt, Mapping), "prompt_binding must be an object")
    base_messages = prompt.get("base_messages")
    effective_messages = prompt.get("effective_messages")
    _require(
        isinstance(base_messages, list) and isinstance(effective_messages, list),
        "prompt messages must be lists",
    )
    expected_effective, instruction = inject_harness_instruction(base_messages, action)
    _require(
        effective_messages == expected_effective, "Harness prompt transform mismatch"
    )
    _require(
        prompt.get("base_messages_sha256") == _sha256(base_messages)
        and prompt.get("effective_messages_sha256") == _sha256(effective_messages),
        "prompt message hash mismatch",
    )
    base_tokens = prompt.get("base_input_tokens")
    effective_tokens = prompt.get("effective_input_tokens")
    _require(
        isinstance(base_tokens, list)
        and isinstance(effective_tokens, list)
        and base_tokens
        and effective_tokens,
        "base and effective input tokens must be non-empty lists",
    )
    _require(
        all(
            type(token) is int and token >= 0
            for token in base_tokens + effective_tokens
        ),
        "prompt token IDs must be non-negative integers",
    )
    _require(
        base_tokens != effective_tokens, "Harness action did not change prompt tokens"
    )
    _require(prompt.get("prompt_tokens_changed") is True, "prompt change flag is false")
    _require(
        instruction == harness["applied_instruction"], "instruction binding mismatch"
    )
    _require(
        type(state.get("turn")) is int
        and state["turn"] == 0
        and type(state.get("remaining_tool_calls")) is int
        and state["remaining_tool_calls"] == 0
        and type(state.get("remaining_model_retries")) is int
        and state["remaining_model_retries"] == 0
        and type(state.get("context_chars")) is int
        and state["context_chars"] == prompt_context_chars(base_messages)
        and state.get("last_error") is None
        and state.get("retrieval_hit") is False
        and state.get("verifier_status") == "not-run"
        and state.get("task_domain") == "gsm8k",
        "Harness state does not match the frozen prompt observation",
    )

    areal_trace = record.get("areal_trace")
    _require(isinstance(areal_trace, Mapping), "areal_trace must be an object")
    try:
        trace_audit = validate_areal_trace_record(
            areal_trace,
            expected_policy_version=expected_policy_version,
        )
    except ArealTraceContractError as exc:
        raise ArealJointBridgeError(str(exc)) from exc
    _require(
        areal_trace["request_id"] == record.get("request_id")
        and areal_trace["task_id"] == record.get("task_id"),
        "AReaL trace identity differs from bridge identity",
    )
    _require(
        areal_trace["interaction"]["interaction_id"] == binding["interaction_id"],
        "AReaL trace interaction differs from adapter sidecar",
    )
    _require(
        areal_trace["model_response"]["input_tokens"] == effective_tokens,
        "AReaL ModelResponse did not consume the Harness-effective prompt",
    )

    policy = record.get("policy_binding")
    _require(isinstance(policy, Mapping), "policy_binding must be an object")
    _require(
        policy.get("policy_release_id") == joint_version.policy,
        "policy release ID differs from JointVersion",
    )
    generation_logprob_mode = policy.get("generation_logprob_mode")
    _require(
        generation_logprob_mode in GENERATION_LOGPROB_MODES,
        "unknown generation log-prob mode",
    )
    _require(
        joint_version.policy.endswith(
            f":generation-logprob={generation_logprob_mode}"
            f":runtime={policy.get('inference_runtime_contract_sha256')}"
        ),
        "JointVersion policy does not bind generation mode and runtime contract",
    )
    sglang_version = policy.get("sglang_version")
    _require(
        isinstance(sglang_version, str) and bool(sglang_version),
        "SGLang version must be a non-empty string",
    )
    _require(
        f":sglang={sglang_version}:generation-logprob=" in joint_version.policy,
        "JointVersion policy does not bind the SGLang version",
    )
    runtime_contract = policy.get("inference_runtime_contract")
    _require(
        isinstance(runtime_contract, Mapping),
        "inference runtime contract must be an object",
    )
    runtime_contract_hash = inference_runtime_contract_sha256(runtime_contract)
    _require(
        policy.get("inference_runtime_contract_sha256") == runtime_contract_hash,
        "inference runtime contract hash mismatch",
    )
    runtime_fixed = runtime_contract["fixed"]
    runtime_treatment = runtime_contract["treatment"]
    runtime_server_args = runtime_fixed["server_args"]
    runtime_rollout = runtime_fixed["rollout"]
    _require(
        runtime_fixed["sglang_version"] == sglang_version
        and runtime_fixed["dataset_selection"] == policy.get("dataset_selection")
        and runtime_fixed["project_commit"] == project_commit
        and runtime_fixed["areal_commit"] == areal_trace["origin"]["areal_commit"]
        and runtime_fixed["behavior_revision"]
        == areal_trace["origin"]["behavior_revision"],
        "inference runtime fixed identity differs from bridge identity",
    )
    _require(
        joint_version.environment
        == (
            f"gsm8k-test@{runtime_fixed['dataset_revision']}:selection="
            f"{runtime_fixed['dataset_selection']}"
        ),
        "inference runtime dataset revision differs from JointVersion",
    )
    _require(
        isinstance(runtime_server_args, Mapping)
        and runtime_server_args.get("model_path")
        == areal_trace["origin"]["behavior_snapshot_path"]
        and runtime_server_args.get("tokenizer_path")
        == areal_trace["origin"]["behavior_snapshot_path"]
        and runtime_server_args.get("tp_size") == 1
        and runtime_server_args.get("base_gpu_id") == 0,
        "inference runtime server args differ from behavior snapshot or topology",
    )
    if (
        runtime_contract["schema_version"]
        == DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION
    ):
        _require(
            isinstance(runtime_rollout, Mapping)
            and runtime_rollout.get("backend") == "sglang:d4"
            and type(runtime_rollout.get("max_concurrent_rollouts")) is int
            and runtime_rollout["max_concurrent_rollouts"] >= 1,
            "distributed inference runtime rollout topology differs from bridge contract",
        )
        physical_gpu_ids = runtime_fixed["physical_gpu_ids"]
        cuda_visible_devices_by_rank = runtime_fixed[
            "cuda_visible_devices_by_rank"
        ]
        gpu_uuids = runtime_fixed["gpu_uuids"]
        gpu_names = runtime_fixed["gpu_names"]
        _require(
            isinstance(physical_gpu_ids, list)
            and len(physical_gpu_ids) == 4
            and all(type(gpu_id) is int and gpu_id >= 0 for gpu_id in physical_gpu_ids)
            and len(set(physical_gpu_ids)) == 4
            and isinstance(cuda_visible_devices_by_rank, list)
            and cuda_visible_devices_by_rank
            == [str(gpu_id) for gpu_id in physical_gpu_ids]
            and isinstance(gpu_uuids, list)
            and len(gpu_uuids) == 4
            and all(isinstance(value, str) and bool(value) for value in gpu_uuids)
            and len(set(gpu_uuids)) == 4
            and isinstance(gpu_names, list)
            and len(gpu_names) == 4
            and all(isinstance(value, str) and bool(value) for value in gpu_names),
            "distributed inference runtime GPU identities are inconsistent",
        )
    else:
        _require(
            isinstance(runtime_rollout, Mapping)
            and runtime_rollout.get("backend") == "sglang:d1p1t1"
            and runtime_rollout.get("max_concurrent_rollouts") == 1,
            "inference runtime rollout topology differs from bridge contract",
        )
        _require(
            type(runtime_fixed["physical_gpu_id"]) is int
            and runtime_fixed["physical_gpu_id"] >= 0
            and runtime_fixed["cuda_visible_devices"]
            == str(runtime_fixed["physical_gpu_id"])
            and isinstance(runtime_fixed["gpu_uuid"], str)
            and bool(runtime_fixed["gpu_uuid"]),
            "inference runtime GPU identity is inconsistent",
        )
    _require(
        runtime_treatment["generation_logprob_mode"] == generation_logprob_mode
        and runtime_treatment["sglang_return_original_logprob"]
        is (generation_logprob_mode == "original-log-softmax-v1"),
        "inference runtime treatment differs from generation mode",
    )
    if runtime_contract["schema_version"] in {
        INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
        DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
    }:
        _require(
            runtime_treatment["disable_cuda_graph"]
            is runtime_server_args.get("disable_cuda_graph"),
            "inference runtime CUDA Graph treatment differs from server args",
        )
    dataset_selection = policy.get("dataset_selection")
    _require(
        isinstance(dataset_selection, str) and bool(dataset_selection),
        "dataset selection must be a non-empty string",
    )
    _require(
        joint_version.environment.endswith(f":selection={dataset_selection}"),
        "JointVersion environment does not bind the dataset selection",
    )
    _require(
        record.get("request_id")
        == deterministic_bridge_request_id(
            task_id=record["task_id"],
            dataset_selection=dataset_selection,
            base_messages=base_messages,
        ),
        "request ID differs from deterministic prompt binding",
    )
    recorded_engine_version = policy.get("expected_inference_engine_version")
    _require(
        type(recorded_engine_version) is int and recorded_engine_version >= 0,
        "expected inference engine version must be a non-negative integer",
    )
    if expected_policy_version is not None:
        _require(
            recorded_engine_version == expected_policy_version,
            "policy binding engine version differs from expected version",
        )

    credit = record.get("credit_binding")
    _require(isinstance(credit, Mapping), "credit_binding must be an object")
    _require(
        credit.get("status") == "raw-terminal-outcome-only"
        and credit.get("policy_advantage") is None
        and credit.get("harness_advantage") is None,
        "bridge must not fabricate policy or Harness advantage",
    )
    _require(
        credit.get("policy_target_model_call_id") == binding["model_call_id"]
        and credit.get("harness_target_decision_id") == decision.get("decision_id"),
        "credit targets are not bound to their decision types",
    )
    reward = credit.get("raw_terminal_reward")
    _require(
        type(reward) in (int, float)
        and math.isfinite(float(reward))
        and float(reward) == float(areal_trace["interaction"]["reward"]),
        "raw terminal reward differs from AReaL interaction",
    )

    return {
        "ok": True,
        "task_id": record["task_id"],
        "request_id": record["request_id"],
        "episode_id": record["episode_id"],
        "model_call_id": binding["model_call_id"],
        "interaction_id": binding["interaction_id"],
        "interaction_sidecar_sha256": sidecar_audit["sidecar_sha256"],
        "joint_version_id": joint_version.version_id,
        "policy_release_id": joint_version.policy,
        "generation_logprob_mode": generation_logprob_mode,
        "sglang_version": sglang_version,
        "dataset_selection": dataset_selection,
        "inference_runtime_contract_sha256": runtime_contract_hash,
        "inference_engine_versions": trace_audit["policy_versions"],
        "harness_controller_version": joint_version.harness_controller,
        "harness_action": action.value,
        "harness_loss_mask": decision["harness_loss_mask"],
        "prompt_tokens_changed": True,
        "base_prompt_tokens": len(base_tokens),
        "effective_prompt_tokens": len(effective_tokens),
        "generated_tokens": trace_audit["generated_tokens"],
        "reward": float(reward),
        "record_sha256": record["record_sha256"],
        "project_commit": project_commit,
    }


def write_areal_joint_bridge_record(
    record: Mapping[str, object],
    *,
    trace_dir: str | Path,
    allowed_root: str | Path,
) -> Path:
    audit = validate_areal_joint_bridge_record(record)
    root = Path(allowed_root).expanduser().resolve()
    directory = Path(trace_dir).expanduser().resolve()
    _require(root.is_dir(), f"configured root does not exist: {root}")
    try:
        common = Path(os.path.commonpath((directory, root)))
    except ValueError as exc:
        raise ArealJointBridgeError("cannot compare bridge path with root") from exc
    _require(common == root, f"path escapes configured root: {directory}")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    request_id = str(audit["request_id"])
    _require(
        request_id
        and all(character.isalnum() or character in "._-" for character in request_id),
        "unsafe request ID",
    )
    path = directory / f"bridge-task{audit['task_id']}-{request_id}.json"
    _require(
        Path(os.path.commonpath((path.resolve(), root))) == root, "output escapes root"
    )
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
