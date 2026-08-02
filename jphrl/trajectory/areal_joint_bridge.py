from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jphrl.harness.controller import HarnessDecision, HarnessState
from jphrl.harness.learning import TabularHarnessController
from jphrl.harness.spec import HarnessAction

from .areal_trace_contract import (
    ArealTraceContractError,
    build_areal_trace_record,
    validate_areal_trace_record,
)
from .schema import EpisodeTrace, JointVersion


SCHEMA_VERSION = "jph.areal-joint-interaction-bridge.v1"
CONTEXT_BUILDER_VERSION = "gsm8k-harness-prompt-v1"
HARNESS_ARTIFACT_VERSION = "gsm8k-bounded-prompt-actions-v1"
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


def prompt_context_chars(messages: Sequence[Mapping[str, str]]) -> int:
    """Count the UTF-8 prompt representation used by the bridge state encoder."""

    return len(_canonical_json([dict(message) for message in messages]).decode("utf-8"))


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
) -> JointVersion:
    for name, value in (
        ("policy_release_id", policy_release_id),
        ("harness_controller_version", harness_controller_version),
        ("areal_commit", areal_commit),
        ("behavior_revision", behavior_revision),
        ("dataset_revision", dataset_revision),
    ):
        _require(isinstance(value, str) and value, f"{name} cannot be empty")
    artifact = harness_artifact_payload()
    return JointVersion(
        policy=policy_release_id,
        harness_controller=harness_controller_version,
        harness_artifact=f"{HARNESS_ARTIFACT_VERSION}@{artifact['sha256']}",
        tool_schema="no-tools-single-turn-v1",
        parser=f"areal-gsm8k-reward@{areal_commit}",
        environment=f"gsm8k-test@{dataset_revision}",
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
        isinstance(decision.get("decision_id"), str)
        and bool(decision["decision_id"]),
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
            pre_mask_logits=tuple(float(value) for value in decision["pre_mask_logits"]),
            harness_loss_mask=decision["harness_loss_mask"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArealJointBridgeError("invalid Harness state or decision payload") from exc
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
    _require(
        len(project_commit) == 40
        and all(character in "0123456789abcdef" for character in project_commit),
        "project commit must be a lowercase 40-character Git object ID",
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
        "joint_version": asdict(joint_version),
        "joint_version_id": joint_version.version_id,
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
        },
        "credit_binding": {
            "status": "raw-terminal-outcome-only",
            "raw_terminal_reward": float(interaction.reward),
            "policy_target_model_call_id": request_id,
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
    _require(record.get("record_sha256") == _record_sha256(record), "record hash mismatch")

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
    _require(dict(scope) == expected_scope, "evidence scope differs from bridge contract")
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
        restored_controller = TabularHarnessController.from_checkpoint(
            controller_checkpoint
        )
        replayed_decision = restored_controller.choose(HarnessState(**state))
    except (KeyError, TypeError, ValueError) as exc:
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
    _require(effective_messages == expected_effective, "Harness prompt transform mismatch")
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
        all(type(token) is int and token >= 0 for token in base_tokens + effective_tokens),
        "prompt token IDs must be non-negative integers",
    )
    _require(base_tokens != effective_tokens, "Harness action did not change prompt tokens")
    _require(prompt.get("prompt_tokens_changed") is True, "prompt change flag is false")
    _require(instruction == harness["applied_instruction"], "instruction binding mismatch")
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
        areal_trace["model_response"]["input_tokens"] == effective_tokens,
        "AReaL ModelResponse did not consume the Harness-effective prompt",
    )

    policy = record.get("policy_binding")
    _require(isinstance(policy, Mapping), "policy_binding must be an object")
    _require(
        policy.get("policy_release_id") == joint_version.policy,
        "policy release ID differs from JointVersion",
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
        credit.get("policy_target_model_call_id") == record.get("request_id")
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
        "joint_version_id": joint_version.version_id,
        "policy_release_id": joint_version.policy,
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
        request_id and all(character.isalnum() or character in "._-" for character in request_id),
        "unsafe request ID",
    )
    path = directory / f"bridge-task{audit['task_id']}-{request_id}.json"
    _require(Path(os.path.commonpath((path.resolve(), root))) == root, "output escapes root")
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
