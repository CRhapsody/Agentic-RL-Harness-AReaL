from __future__ import annotations

"""Private exactly-once entry from Hermes Agent Service into AA/M0 P/Q/R/S."""

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from jphrl.experiments.m0_joint_input import (
    M0JointInputError,
    prepare_m0_joint_training_input_from_receipts,
)
from jphrl.harness.controller import HarnessDecision, HarnessState
from jphrl.harness.spec import HarnessAction
from jphrl.paths import require_outside_repository
from jphrl.trajectory.areal_agent_service_adapter import (
    SECRET_FIELD_NAMES,
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    session_receipt_from_start_response,
    trajectory_receipt_from_set_reward_response,
    validate_agent_service_training_record,
)
from jphrl.trajectory.areal_data_proxy_pre_batch import (
    HOOK_STAGE,
    VerifiedDataProxyPreBatchHook,
    validate_pre_batch_trajectory_export,
)
from jphrl.trajectory.areal_interaction_sidecar import (
    InteractionBinding,
    build_interaction_adapter_sidecar,
)
from jphrl.trajectory.areal_joint_bridge import (
    build_areal_joint_bridge_record,
    deterministic_bridge_request_id,
    inference_runtime_contract_sha256,
    inject_harness_instruction,
    prompt_context_chars,
    validate_areal_joint_bridge_record,
)
from jphrl.trajectory.areal_policy_admission import (
    validate_policy_training_admission,
)
from jphrl.trajectory.harness_action_admission import (
    validate_harness_action_admission_record,
)
from jphrl.trajectory.hermes_model_call_receipts import (
    HermesModelCallReceipt,
    receipts_from_public_dicts,
    validate_hermes_model_call_receipts,
)
from jphrl.trajectory.joint_credit_alignment import (
    DualCreditEstimatorSpec,
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.schema import JointVersion

STAGED_SCHEMA_VERSION = "jph.m0-agent-service-staged-envelope.v1"
FINALIZED_SCHEMA_VERSION = "jph.m0-agent-service-finalized-envelope.v1"
EXPORT_STYLES = frozenset({"individual", "concat"})
_SECRET_NAMES = frozenset({*SECRET_FIELD_NAMES, "token"})
_SECRET_SUFFIXES = (
    "_api_key",
    "_access_key",
    "_private_key",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)


class M0AgentServiceEntryError(ValueError):
    """Raised when real Agent Service evidence cannot enter AA/M0."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0AgentServiceEntryError(message)


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
        raise M0AgentServiceEntryError(
            "M0 Agent Service envelope is not finite canonical JSON"
        ) from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical_json(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )
    ).hexdigest()


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            _require(
                normalized not in _SECRET_NAMES
                and not normalized.endswith(_SECRET_SUFFIXES),
                f"credential field cannot enter M0 Agent Service envelope: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _exact_mapping(
    value: object, fields: set[str], label: str
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == fields, f"{label} field set differs from schema")
    return value


def _public_id(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} is missing")
    result = value.strip()
    _require(
        not result.lower().startswith(("sk-", "bearer ")),
        f"{label} looks like a credential",
    )
    return result


def _resolve_root(journal_root: str | Path) -> Path:
    try:
        root = require_outside_repository(journal_root)
    except ValueError as exc:
        raise M0AgentServiceEntryError(str(exc)) from exc
    created = not root.exists()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = root.resolve()
    _require(root.is_dir(), f"M0 Agent Service journal is not a directory: {root}")
    if created:
        root.chmod(0o700)
    _require(
        root.stat().st_mode & 0o077 == 0,
        "M0 Agent Service journal must be private to its owner",
    )
    return root


def _require_within(path: Path, root: Path) -> None:
    try:
        common = Path(os.path.commonpath((path.parent.resolve(), root)))
    except ValueError as exc:
        raise M0AgentServiceEntryError("cannot compare private journal paths") from exc
    _require(common == root, f"private journal path escapes its root: {path}")


def _write_new(record: Mapping[str, object], path: Path, root: Path) -> Path:
    _assert_no_secrets(record)
    _canonical_json(record)
    _require_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def _read_private(path: Path, root: Path) -> dict[str, object]:
    _require_within(path, root)
    _require(path.is_file() and not path.is_symlink(), f"private record is missing: {path}")
    resolved = path.resolve(strict=True)
    _require(root in resolved.parents, "private record escapes its journal root")
    _require(
        resolved.stat().st_mode & 0o077 == 0,
        "private record is accessible to group or other users",
    )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M0AgentServiceEntryError(f"cannot read private record: {path}") from exc
    _require(isinstance(value, dict), "private record must be an object")
    return value


def _route_key(session_id: str, trajectory_id: int) -> str:
    payload = {
        "schema_version": "jph.m0-agent-service-route-key.v1",
        "session_id": session_id,
        "trajectory_id": trajectory_id,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _decision_record(decision: HarnessDecision) -> dict[str, object]:
    _require(type(decision) is HarnessDecision, "Harness decision must use its exact type")
    record = asdict(decision)
    record["action"] = decision.action.value
    record["action_ids"] = list(decision.action_ids)
    record["action_mask"] = list(decision.action_mask)
    record["pre_mask_logits"] = list(decision.pre_mask_logits)
    return record


def _parse_decision(raw: object) -> HarnessDecision:
    value = _exact_mapping(
        raw,
        set(HarnessDecision.__dataclass_fields__),
        "staged Harness decision",
    )
    try:
        return HarnessDecision(
            decision_id=value["decision_id"],
            action=HarnessAction(str(value["action"])),
            old_harness_logprob=float(value["old_harness_logprob"]),
            controller_version=value["controller_version"],
            action_ids=tuple(value["action_ids"]),
            action_mask=tuple(value["action_mask"]),
            pre_mask_logits=tuple(float(item) for item in value["pre_mask_logits"]),
            harness_loss_mask=value["harness_loss_mask"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M0AgentServiceEntryError("invalid staged Harness decision") from exc


def _parse_state(raw: object) -> HarnessState:
    value = _exact_mapping(
        raw,
        set(HarnessState.__dataclass_fields__),
        "staged Harness state",
    )
    try:
        return HarnessState(**dict(value))
    except TypeError as exc:
        raise M0AgentServiceEntryError("invalid staged Harness state") from exc


def _parse_joint_version(raw: object) -> JointVersion:
    value = _exact_mapping(
        raw,
        set(JointVersion.__dataclass_fields__),
        "staged JointVersion",
    )
    try:
        result = JointVersion(**dict(value))
    except TypeError as exc:
        raise M0AgentServiceEntryError("invalid staged JointVersion") from exc
    _require(
        all(_public_id(getattr(result, field), f"JointVersion {field}") for field in value),
        "JointVersion fields are invalid",
    )
    return result


def _parse_estimator(raw: object) -> DualCreditEstimatorSpec:
    value = _exact_mapping(
        raw,
        set(DualCreditEstimatorSpec.__dataclass_fields__),
        "staged dual-credit estimator",
    )
    policy_baselines = value.get("policy_baselines")
    harness_baselines = value.get("harness_baselines")
    _require(
        isinstance(policy_baselines, Mapping)
        and isinstance(harness_baselines, Mapping),
        "staged baseline maps are missing",
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
        raise M0AgentServiceEntryError("invalid staged dual-credit estimator") from exc


def _validate_harness_replay(
    *,
    state: HarnessState,
    decision: HarnessDecision,
    checkpoint: Mapping[str, object],
    joint_version: JointVersion,
) -> None:
    try:
        from jphrl.harness.torch_learning import (
            load_torch_harness_rollout_checkpoint,
        )

        replayed = load_torch_harness_rollout_checkpoint(checkpoint).choose(state)
    except (ImportError, ValueError) as exc:
        raise M0AgentServiceEntryError(
            f"Torch Harness sampling-before checkpoint is invalid: {exc}"
        ) from exc
    _require(
        replayed.controller_version == decision.controller_version
        == joint_version.harness_controller,
        "Harness checkpoint, decision, and JointVersion controller differ",
    )
    for field in (
        "action",
        "old_harness_logprob",
        "action_ids",
        "action_mask",
        "pre_mask_logits",
        "harness_loss_mask",
    ):
        _require(
            getattr(replayed, field) == getattr(decision, field),
            f"Harness decision field {field} differs from sampling-before replay",
        )


def _validate_prompt_and_runtime(
    *,
    task_id: int,
    state: HarnessState,
    decision: HarnessDecision,
    base_messages: Sequence[Mapping[str, str]],
    effective_messages: Sequence[Mapping[str, str]],
    base_input_tokens: Sequence[int],
    effective_input_tokens: Sequence[int],
    dataset_selection: str,
    inference_runtime_contract: Mapping[str, object],
) -> str:
    _require(type(task_id) is int and task_id >= 0, "M0 task ID is invalid")
    expected_messages, _ = inject_harness_instruction(base_messages, decision.action)
    _require(
        [dict(item) for item in effective_messages] == expected_messages,
        "effective prompt differs from the sampled Harness action",
    )
    _require(
        bool(base_input_tokens)
        and bool(effective_input_tokens)
        and all(
            type(token) is int and token >= 0
            for token in (*base_input_tokens, *effective_input_tokens)
        )
        and list(base_input_tokens) != list(effective_input_tokens),
        "base/effective prompt tokens are invalid or unchanged",
    )
    _require(
        state.turn == 0
        and state.remaining_tool_calls == 0
        and state.remaining_model_retries == 0
        and state.context_chars == prompt_context_chars(base_messages)
        and state.last_error is None
        and state.retrieval_hit is False
        and state.verifier_status == "not-run"
        and state.task_domain == "gsm8k",
        "Harness state differs from the frozen M0 prompt observation",
    )
    inference_runtime_contract_sha256(inference_runtime_contract)
    return deterministic_bridge_request_id(
        task_id=task_id,
        dataset_selection=dataset_selection,
        base_messages=base_messages,
    )


def _staged_paths(root: Path, session_id: str, trajectory_id: int) -> tuple[Path, Path]:
    key = _route_key(session_id, trajectory_id)
    return root / "pending" / f"{key}.json", root / "finalized" / f"{key}.json"


def validate_staged_m0_agent_service_envelope(
    record: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        "schema_version",
        "identity",
        "receipts",
        "joint_version",
        "harness",
        "prompt",
        "runtime",
        "estimator",
        "export",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == fields, "staged M0 envelope field set differs")
    _require(
        record.get("schema_version") == STAGED_SCHEMA_VERSION,
        "unknown staged M0 envelope schema",
    )
    _require(record.get("record_sha256") == _record_sha256(record), "staged M0 hash mismatch")
    _assert_no_secrets(record)
    identity = _exact_mapping(
        record.get("identity"),
        {
            "episode_id",
            "task_id",
            "request_id",
            "session_id",
            "trajectory_id",
            "model_call_id",
            "interaction_id",
            "joint_version_id",
        },
        "staged M0 identity",
    )
    receipts = _exact_mapping(
        record.get("receipts"),
        {"session", "hermes_model_call", "trajectory"},
        "staged M0 receipts",
    )
    session_raw = _exact_mapping(
        receipts.get("session"),
        set(AgentServiceSessionReceipt.__dataclass_fields__),
        "staged session receipt",
    )
    hermes_raw = _exact_mapping(
        receipts.get("hermes_model_call"),
        set(HermesModelCallReceipt.__dataclass_fields__),
        "staged Hermes receipt",
    )
    trajectory_raw = _exact_mapping(
        receipts.get("trajectory"),
        set(AgentServiceTrajectoryReceipt.__dataclass_fields__),
        "staged trajectory receipt",
    )
    try:
        session = AgentServiceSessionReceipt(**dict(session_raw))
        hermes = HermesModelCallReceipt(**dict(hermes_raw))
        trajectory = AgentServiceTrajectoryReceipt(**dict(trajectory_raw))
    except TypeError as exc:
        raise M0AgentServiceEntryError("invalid staged Agent Service receipts") from exc
    validate_hermes_model_call_receipts([hermes], expected_session_id=session.session_id)
    _require(
        hermes.ordinal == 0
        and hermes.parent_model_call_id is None
        and trajectory.interaction_count == 1
        and trajectory.session_id == session.session_id,
        "staged M0 receipts are not one root Agent Service call",
    )
    joint_version = _parse_joint_version(record.get("joint_version"))
    harness = _exact_mapping(
        record.get("harness"),
        {"state", "decision", "sampling_before_checkpoint"},
        "staged M0 Harness",
    )
    state = _parse_state(harness.get("state"))
    decision = _parse_decision(harness.get("decision"))
    checkpoint = harness.get("sampling_before_checkpoint")
    _require(isinstance(checkpoint, Mapping), "sampling-before checkpoint is missing")
    _validate_harness_replay(
        state=state,
        decision=decision,
        checkpoint=checkpoint,
        joint_version=joint_version,
    )
    prompt = _exact_mapping(
        record.get("prompt"),
        {
            "base_messages",
            "effective_messages",
            "base_input_tokens",
            "effective_input_tokens",
        },
        "staged M0 prompt",
    )
    runtime = _exact_mapping(
        record.get("runtime"),
        {
            "expected_policy_version",
            "project_commit",
            "areal_commit",
            "behavior_snapshot_path",
            "behavior_revision",
            "dataset_selection",
            "sglang_version",
            "generation_logprob_mode",
            "inference_runtime_contract",
        },
        "staged M0 runtime",
    )
    for commit_name in ("project_commit", "areal_commit"):
        commit = runtime.get(commit_name)
        _require(
            isinstance(commit, str)
            and len(commit) == 40
            and all(character in "0123456789abcdef" for character in commit),
            f"staged {commit_name} is not a full lowercase Git object ID",
        )
    expected_policy_version = runtime.get("expected_policy_version")
    _require(
        type(expected_policy_version) is int and expected_policy_version >= 0,
        "staged expected policy version is invalid",
    )
    dataset_selection = _public_id(runtime.get("dataset_selection"), "dataset selection")
    request_id = _validate_prompt_and_runtime(
        task_id=identity.get("task_id"),
        state=state,
        decision=decision,
        base_messages=prompt.get("base_messages"),
        effective_messages=prompt.get("effective_messages"),
        base_input_tokens=prompt.get("base_input_tokens"),
        effective_input_tokens=prompt.get("effective_input_tokens"),
        dataset_selection=dataset_selection,
        inference_runtime_contract=runtime.get("inference_runtime_contract"),
    )
    estimator = _parse_estimator(record.get("estimator"))
    estimator.validate(
        joint_version=joint_version,
        policy_model_call_ids=(hermes.model_call_id,),
        harness_decision_ids=(decision.decision_id,),
    )
    export = _exact_mapping(
        record.get("export"),
        {"style", "turn_discount"},
        "staged M0 export",
    )
    discount = export.get("turn_discount")
    _require(
        export.get("style") in EXPORT_STYLES
        and type(discount) in (int, float)
        and math.isfinite(float(discount))
        and 0.0 <= float(discount) <= 1.0,
        "staged M0 export contract is invalid",
    )
    expected_identity = {
        "episode_id": _public_id(identity.get("episode_id"), "episode ID"),
        "task_id": identity.get("task_id"),
        "request_id": request_id,
        "session_id": session.session_id,
        "trajectory_id": trajectory.trajectory_id,
        "model_call_id": hermes.model_call_id,
        "interaction_id": hermes.interaction_id,
        "joint_version_id": joint_version.version_id,
    }
    _require(dict(identity) == expected_identity, "staged M0 identity differs from its evidence")
    _require(
        record.get("evidence_scope")
        == {
            "agent_service_receipts_captured": True,
            "torch_harness_sampling_replayed": True,
            "prompt_runtime_provenance_bound": True,
            "pre_batch_interaction_binding": False,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "staged M0 evidence scope differs from contract",
    )
    return {
        "session": session,
        "hermes": hermes,
        "trajectory": trajectory,
        "joint_version": joint_version,
        "state": state,
        "decision": decision,
        "checkpoint": deepcopy(dict(checkpoint)),
        "estimator": estimator,
        "request_id": request_id,
        "record_sha256": record["record_sha256"],
    }


def stage_m0_agent_service_rollout(
    *,
    journal_root: str | Path,
    start_session_response: Mapping[str, object],
    set_reward_response: Mapping[str, object],
    hermes_receipt_payload: object,
    episode_id: str,
    task_id: int,
    joint_version: JointVersion,
    harness_state: HarnessState,
    harness_decision: HarnessDecision,
    harness_sampling_before_checkpoint: Mapping[str, object],
    base_messages: Sequence[Mapping[str, str]],
    effective_messages: Sequence[Mapping[str, str]],
    base_input_tokens: Sequence[int],
    effective_input_tokens: Sequence[int],
    expected_policy_version: int,
    project_commit: str,
    areal_commit: str,
    behavior_snapshot_path: str,
    behavior_revision: str,
    dataset_selection: str,
    sglang_version: str,
    generation_logprob_mode: str,
    inference_runtime_contract: Mapping[str, object],
    estimator: DualCreditEstimatorSpec,
    export_style: str,
    turn_discount: float,
    session_index: int = 0,
) -> Path:
    """Strip HTTP credentials and privately stage all pre-export AA evidence."""

    try:
        session = session_receipt_from_start_response(
            start_session_response,
            session_index=session_index,
        )
        trajectory = trajectory_receipt_from_set_reward_response(set_reward_response)
        receipts = receipts_from_public_dicts(
            hermes_receipt_payload,
            expected_session_id=session.session_id,
        )
    except ValueError as exc:
        raise M0AgentServiceEntryError(str(exc)) from exc
    _require(len(receipts) == 1, "AA M0 requires exactly one Hermes model-call receipt")
    hermes = receipts[0]
    _require(
        hermes.ordinal == 0 and hermes.parent_model_call_id is None,
        "AA M0 requires one root Hermes model call with ordinal zero",
    )
    _require(
        trajectory.session_id == session.session_id
        and trajectory.interaction_count == 1,
        "set-reward trajectory differs from the routed one-call session",
    )
    _public_id(session.group_id, "Agent Service group ID")
    _public_id(session.session_id, "Agent Service session ID")
    _public_id(behavior_snapshot_path, "behavior snapshot path")
    _public_id(behavior_revision, "behavior revision")
    _public_id(sglang_version, "SGLang version")
    _public_id(generation_logprob_mode, "generation logprob mode")
    request_id = _validate_prompt_and_runtime(
        task_id=task_id,
        state=harness_state,
        decision=harness_decision,
        base_messages=base_messages,
        effective_messages=effective_messages,
        base_input_tokens=base_input_tokens,
        effective_input_tokens=effective_input_tokens,
        dataset_selection=dataset_selection,
        inference_runtime_contract=inference_runtime_contract,
    )
    _validate_harness_replay(
        state=harness_state,
        decision=harness_decision,
        checkpoint=harness_sampling_before_checkpoint,
        joint_version=joint_version,
    )
    estimator.validate(
        joint_version=joint_version,
        policy_model_call_ids=(hermes.model_call_id,),
        harness_decision_ids=(harness_decision.decision_id,),
    )
    record: dict[str, object] = {
        "schema_version": STAGED_SCHEMA_VERSION,
        "identity": {
            "episode_id": _public_id(episode_id, "episode ID"),
            "task_id": task_id,
            "request_id": request_id,
            "session_id": session.session_id,
            "trajectory_id": trajectory.trajectory_id,
            "model_call_id": hermes.model_call_id,
            "interaction_id": hermes.interaction_id,
            "joint_version_id": joint_version.version_id,
        },
        "receipts": {
            "session": asdict(session),
            "hermes_model_call": asdict(hermes),
            "trajectory": asdict(trajectory),
        },
        "joint_version": asdict(joint_version),
        "harness": {
            "state": asdict(harness_state),
            "decision": _decision_record(harness_decision),
            "sampling_before_checkpoint": deepcopy(
                dict(harness_sampling_before_checkpoint)
            ),
        },
        "prompt": {
            "base_messages": [dict(item) for item in base_messages],
            "effective_messages": [dict(item) for item in effective_messages],
            "base_input_tokens": list(base_input_tokens),
            "effective_input_tokens": list(effective_input_tokens),
        },
        "runtime": {
            "expected_policy_version": expected_policy_version,
            "project_commit": project_commit,
            "areal_commit": areal_commit,
            "behavior_snapshot_path": behavior_snapshot_path,
            "behavior_revision": behavior_revision,
            "dataset_selection": dataset_selection,
            "sglang_version": sglang_version,
            "generation_logprob_mode": generation_logprob_mode,
            "inference_runtime_contract": deepcopy(dict(inference_runtime_contract)),
        },
        "estimator": estimator.to_record(),
        "export": {"style": export_style, "turn_discount": turn_discount},
        "evidence_scope": {
            "agent_service_receipts_captured": True,
            "torch_harness_sampling_replayed": True,
            "prompt_runtime_provenance_bound": True,
            "pre_batch_interaction_binding": False,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    _assert_no_secrets(record)
    record["record_sha256"] = _record_sha256(record)
    validate_staged_m0_agent_service_envelope(record)
    root = _resolve_root(journal_root)
    pending, finalized = _staged_paths(
        root,
        session.session_id,
        trajectory.trajectory_id,
    )
    _require(not finalized.exists(), "M0 Agent Service route is already finalized")
    try:
        return _write_new(record, pending, root)
    except FileExistsError as exc:
        raise M0AgentServiceEntryError("M0 Agent Service route is already staged") from exc


def validate_finalized_m0_agent_service_envelope(
    record: Mapping[str, object],
) -> dict[str, object]:
    _require(
        set(record)
        == {
            "schema_version",
            "identity",
            "staged_envelope_sha256",
            "bridge_record",
            "p_training_record",
            "q_policy_admission",
            "r_harness_admission",
            "s_joint_credit",
            "evidence_scope",
            "record_sha256",
        },
        "finalized M0 envelope field set differs",
    )
    _require(
        record.get("schema_version") == FINALIZED_SCHEMA_VERSION,
        "unknown finalized M0 envelope schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "finalized M0 envelope hash mismatch",
    )
    _assert_no_secrets(record)
    bridge = record.get("bridge_record")
    p_record = record.get("p_training_record")
    q_record = record.get("q_policy_admission")
    r_record = record.get("r_harness_admission")
    s_record = record.get("s_joint_credit")
    _require(
        all(isinstance(item, Mapping) for item in (bridge, p_record, q_record, r_record, s_record)),
        "finalized M0 bridge or P/Q/R/S record is missing",
    )
    expected_version = bridge["policy_binding"]["expected_inference_engine_version"]
    bridge_audit = validate_areal_joint_bridge_record(
        bridge,
        expected_policy_version=expected_version,
    )
    joint_version = _parse_joint_version(bridge["joint_version"])
    p_audit = validate_agent_service_training_record(p_record)
    q_audit = validate_policy_training_admission(
        q_record,
        active_joint_version=joint_version,
    )
    r_audit = validate_harness_action_admission_record(
        r_record,
        active_joint_version=joint_version,
    )
    s_audit = validate_frozen_joint_credit_alignment(
        s_record,
        active_joint_version=joint_version,
    )
    identity = _exact_mapping(
        record.get("identity"),
        {
            "episode_id",
            "session_id",
            "trajectory_id",
            "model_call_id",
            "interaction_id",
            "joint_version_id",
        },
        "finalized M0 identity",
    )
    _require(
        identity
        == {
            "episode_id": bridge_audit["episode_id"],
            "session_id": p_audit["session_id"],
            "trajectory_id": p_audit["trajectory_id"],
            "model_call_id": bridge_audit["model_call_id"],
            "interaction_id": bridge_audit["interaction_id"],
            "joint_version_id": joint_version.version_id,
        },
        "finalized M0 identity differs from bridge or P/Q/R/S",
    )
    _require(
        isinstance(record.get("staged_envelope_sha256"), str)
        and len(record["staged_envelope_sha256"]) == 64,
        "staged envelope hash is invalid",
    )
    _require(
        record.get("evidence_scope")
        == {
            "agent_service_receipts_captured": True,
            "torch_harness_sampling_replayed": True,
            "prompt_runtime_provenance_bound": True,
            "pre_batch_interaction_binding": True,
            "p_q_r_s_admitted": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "finalized M0 evidence scope differs from contract",
    )
    return {
        "bridge": bridge_audit,
        "p": p_audit,
        "q": q_audit,
        "r": r_audit,
        "s": s_audit,
        "record_sha256": record["record_sha256"],
    }


class PersistentM0AgentServicePreBatchEntry:
    """Persist exactly one bridge-plus-P/Q/R/S finalization for a route.

    Concurrent callers may both reach optimizer-free validation and construction,
    but the exclusive final write admits exactly one record.  This component has
    no optimizer side effect, so a losing caller cannot duplicate an update.
    """

    def __init__(self, journal_root: str | Path) -> None:
        self._root = _resolve_root(journal_root)

    def __call__(self, event: object) -> dict[str, object]:
        session_id = getattr(event, "session_id", None)
        trajectory_id = getattr(event, "trajectory_id", None)
        interactions = getattr(event, "exported_interactions", None)
        style = getattr(event, "export_style", None)
        discount = getattr(event, "turn_discount", None)
        _require(getattr(event, "hook_stage", None) == HOOK_STAGE, "M0 entry is not at pre-batch hook stage")
        _require(isinstance(session_id, str) and bool(session_id), "pre-batch session ID is missing")
        _require(type(trajectory_id) is int and trajectory_id >= 0, "pre-batch trajectory ID is invalid")
        _require(isinstance(interactions, Mapping), "pre-batch interaction mapping is missing")
        try:
            validate_pre_batch_trajectory_export(event)
        except ValueError as exc:
            raise M0AgentServiceEntryError(str(exc)) from exc
        pending, finalized = _staged_paths(self._root, session_id, trajectory_id)
        _require(not finalized.exists(), "M0 Agent Service route is already finalized")
        _require(pending.is_file(), "no staged M0 envelope exists for pre-batch route")
        staged_record = _read_private(pending, self._root)
        staged = validate_staged_m0_agent_service_envelope(staged_record)
        export = staged_record["export"]
        _require(
            staged["session"].session_id == session_id
            and staged["trajectory"].trajectory_id == trajectory_id
            and export["style"] == style
            and float(export["turn_discount"]) == float(discount),
            "pre-batch route or export contract differs from staged M0 evidence",
        )
        runtime = staged_record["runtime"]
        prompt = staged_record["prompt"]
        identity = staged_record["identity"]
        interaction_id = staged["hermes"].interaction_id
        _require(list(interactions) == [interaction_id], "pre-batch interaction differs from Hermes receipt")
        interaction = interactions[interaction_id]
        sidecar = build_interaction_adapter_sidecar(
            [
                InteractionBinding(
                    episode_id=identity["episode_id"],
                    model_call_id=staged["hermes"].model_call_id,
                    session_id=session_id,
                    trajectory_id=trajectory_id,
                    interaction_id=interaction_id,
                    parent_interaction_id=None,
                    ordinal=0,
                    joint_version_id=staged["joint_version"].version_id,
                    route_kind="agent-service-session",
                )
            ]
        )
        try:
            bridge = build_areal_joint_bridge_record(
                task_id=identity["task_id"],
                request_id=staged["request_id"],
                joint_version=staged["joint_version"],
                expected_policy_version=runtime["expected_policy_version"],
                harness_state=staged["state"],
                harness_decision=staged["decision"],
                harness_controller_checkpoint=staged["checkpoint"],
                base_messages=prompt["base_messages"],
                effective_messages=prompt["effective_messages"],
                base_input_tokens=prompt["base_input_tokens"],
                effective_input_tokens=prompt["effective_input_tokens"],
                model_response=interaction.model_response,
                interaction=interaction,
                tensor_dict=interaction.to_tensor_dict(),
                project_commit=runtime["project_commit"],
                areal_commit=runtime["areal_commit"],
                behavior_snapshot_path=runtime["behavior_snapshot_path"],
                behavior_revision=runtime["behavior_revision"],
                dataset_selection=runtime["dataset_selection"],
                sglang_version=runtime["sglang_version"],
                generation_logprob_mode=runtime["generation_logprob_mode"],
                inference_runtime_contract=runtime["inference_runtime_contract"],
                interaction_adapter_sidecar=sidecar,
            )
            result = prepare_m0_joint_training_input_from_receipts(
                bridge,
                session_receipt=staged["session"],
                hermes_model_call_receipt=staged["hermes"],
                trajectory_receipt=staged["trajectory"],
                pre_batch_exported_interactions=interactions,
                estimator=staged["estimator"],
                export_style=style,
                turn_discount=float(discount),
            )
        except (AttributeError, KeyError, TypeError, ValueError, M0JointInputError) as exc:
            raise M0AgentServiceEntryError(str(exc)) from exc
        final_record: dict[str, object] = {
            "schema_version": FINALIZED_SCHEMA_VERSION,
            "identity": {
                "episode_id": identity["episode_id"],
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "model_call_id": staged["hermes"].model_call_id,
                "interaction_id": interaction_id,
                "joint_version_id": staged["joint_version"].version_id,
            },
            "staged_envelope_sha256": staged["record_sha256"],
            "bridge_record": bridge,
            "p_training_record": result.p_training_record,
            "q_policy_admission": result.q_policy_admission,
            "r_harness_admission": result.r_harness_admission,
            "s_joint_credit": result.s_joint_credit,
            "evidence_scope": {
                "agent_service_receipts_captured": True,
                "torch_harness_sampling_replayed": True,
                "prompt_runtime_provenance_bound": True,
                "pre_batch_interaction_binding": True,
                "p_q_r_s_admitted": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        _assert_no_secrets(final_record)
        final_record["record_sha256"] = _record_sha256(final_record)
        validate_finalized_m0_agent_service_envelope(final_record)
        try:
            _write_new(final_record, finalized, self._root)
        except FileExistsError as exc:
            raise M0AgentServiceEntryError(
                "M0 Agent Service route is already finalized"
            ) from exc
        return final_record


async def pre_batch_finalize_m0_agent_service_input(
    *,
    session_id: str,
    trajectory_id: int,
    interactions: Mapping[str, object],
    discount: float,
    style: str,
) -> None:
    """Deployment callback for the patched AReaL pre-batch hook.

    The callback deliberately accepts no RLVR route and no credential.  Its
    journal root is an out-of-repository runtime path configured by the launcher.
    """

    journal_root = os.environ.get("JPH_M0_AGENT_SERVICE_JOURNAL_ROOT", "")
    _require(
        bool(journal_root),
        "JPH_M0_AGENT_SERVICE_JOURNAL_ROOT is required",
    )
    hook = VerifiedDataProxyPreBatchHook(
        PersistentM0AgentServicePreBatchEntry(journal_root)
    )
    await hook(
        session_id=session_id,
        trajectory_id=trajectory_id,
        interactions=interactions,
        discount=discount,
        style=style,
    )
