from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .areal_interaction_sidecar import (
    ArealInteractionAdapterError,
    InteractionBinding,
    archive_premerged_exported_interactions,
    build_interaction_adapter_sidecar,
    export_bound_training_sample_archive,
    validate_bound_training_sample_archive,
)
from .schema import EpisodeTrace


SCHEMA_VERSION = "jph.areal-agent-service-training-record.v1"
SECRET_FIELD_NAMES = frozenset(
    {
        "admin_api_key",
        "api_key",
        "authorization",
        "session_api_key",
    }
)


class ArealAgentServiceAdapterError(ArealInteractionAdapterError):
    """Raised when an Agent Service trajectory cannot safely enter training."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealAgentServiceAdapterError(message)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArealAgentServiceAdapterError(
            "Agent Service record is not finite canonical JSON"
        ) from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _assert_no_secret_fields(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_name = str(key).lower()
            _require(
                key_name not in SECRET_FIELD_NAMES,
                f"credential field cannot enter training record: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


@dataclass(frozen=True)
class AgentServiceSessionReceipt:
    """Non-secret identity returned by AReaL ``rl/start_session``."""

    group_id: str
    session_id: str


@dataclass(frozen=True)
class AgentServiceModelCallReceipt:
    """Client-side receipt linking one trace call to one OpenAI response ID."""

    model_call_id: str
    interaction_id: str
    ordinal: int
    parent_model_call_id: str | None


@dataclass(frozen=True)
class AgentServiceTrajectoryReceipt:
    """Ready trajectory identity returned by AReaL ``rl/set_reward``."""

    session_id: str
    trajectory_id: int
    interaction_count: int
    ready_transition: bool


def session_receipt_from_start_response(
    response: Mapping[str, object], *, session_index: int = 0
) -> AgentServiceSessionReceipt:
    """Extract session identity without retaining the session API credential."""

    _require(_is_int(session_index) and session_index >= 0, "invalid session index")
    group_id = response.get("group_id")
    sessions = response.get("sessions")
    _require(_is_non_empty_string(group_id), "start-session group ID is missing")
    _require(isinstance(sessions, list) and bool(sessions), "session list is missing")
    _require(session_index < len(sessions), "session index is out of range")
    selected = sessions[session_index]
    _require(isinstance(selected, Mapping), "session credential must be an object")
    session_id = selected.get("session_id")
    session_api_key = selected.get("session_api_key")
    _require(_is_non_empty_string(session_id), "start-session session ID is missing")
    _require(
        _is_non_empty_string(session_api_key),
        "start-session response is missing its routing credential",
    )
    return AgentServiceSessionReceipt(
        group_id=str(group_id),
        session_id=str(session_id),
    )


def model_call_receipt_from_response(
    *,
    model_call_id: str,
    response: Mapping[str, object] | object,
    ordinal: int,
    parent_model_call_id: str | None,
) -> AgentServiceModelCallReceipt:
    """Capture an OpenAI completion/response ID as the AReaL interaction ID."""

    interaction_id = (
        response.get("id")
        if isinstance(response, Mapping)
        else getattr(response, "id", None)
    )
    _require(_is_non_empty_string(model_call_id), "model call ID is missing")
    _require(
        _is_non_empty_string(interaction_id),
        "OpenAI response does not expose an interaction ID",
    )
    _require(_is_int(ordinal) and ordinal >= 0, "model call ordinal is invalid")
    _require(
        parent_model_call_id is None or _is_non_empty_string(parent_model_call_id),
        "parent model call ID must be null or non-empty",
    )
    return AgentServiceModelCallReceipt(
        model_call_id=model_call_id,
        interaction_id=str(interaction_id),
        ordinal=ordinal,
        parent_model_call_id=parent_model_call_id,
    )


def trajectory_receipt_from_set_reward_response(
    response: Mapping[str, object],
) -> AgentServiceTrajectoryReceipt:
    """Require a reward-bounded trajectory rather than an in-progress session."""

    session_id = response.get("session_id")
    trajectory_id = response.get("trajectory_id")
    interaction_count = response.get("interaction_count")
    trajectory_ready = response.get("trajectory_ready")
    ready_transition = response.get("ready_transition")
    _require(_is_non_empty_string(session_id), "reward receipt session ID is missing")
    _require(
        _is_int(trajectory_id) and trajectory_id >= 0,
        "reward receipt has no ready trajectory ID",
    )
    _require(
        _is_int(interaction_count) and interaction_count > 0,
        "reward receipt interaction count is invalid",
    )
    _require(trajectory_ready is True, "reward receipt trajectory is not ready")
    _require(
        type(ready_transition) is bool,
        "reward receipt ready-transition flag is invalid",
    )
    return AgentServiceTrajectoryReceipt(
        session_id=str(session_id),
        trajectory_id=int(trajectory_id),
        interaction_count=int(interaction_count),
        ready_transition=bool(ready_transition),
    )


def validate_agent_service_training_trace(
    trace: EpisodeTrace,
) -> dict[str, object]:
    """Validate that every policy call in a closed trace can receive credit."""

    try:
        trace_payload = trace.to_dict()
    except ValueError as exc:
        raise ArealAgentServiceAdapterError(str(exc)) from exc
    _require(
        trace.validity_class in {"valid", "policy_failure"},
        "invalid infrastructure or trace-contract episode cannot enter training",
    )
    _require(_is_finite_number(trace.reward), "trainable trace requires finite reward")
    _require(trace.events[-1].kind == "episode_ended", "trace is not closed")

    request_ids: list[str] = []
    response_ids: list[str] = []
    reward_events = []
    for event in trace.events:
        if event.kind == "model_request":
            model_call_id = event.payload.get("model_call_id")
            _require(
                _is_non_empty_string(model_call_id),
                "model request has no model call ID",
            )
            request_ids.append(str(model_call_id))
        elif event.kind == "model_response":
            model_call_id = event.payload.get("model_call_id")
            _require(
                _is_non_empty_string(model_call_id),
                "model response has no model call ID",
            )
            _require(
                event.payload.get("token_metadata_status") == "available",
                "Agent Service training requires real token metadata",
            )
            output_token_ids = event.payload.get("output_token_ids")
            output_versions = event.payload.get("output_versions")
            _require(
                isinstance(output_token_ids, list)
                and isinstance(output_versions, list)
                and len(output_token_ids) == len(output_versions)
                and bool(output_versions),
                "output policy versions do not align with output tokens",
            )
            _require(
                all(_is_int(version) and version >= 0 for version in output_versions),
                "output policy versions must be non-negative integers",
            )
            _require(
                len(set(output_versions)) == 1,
                "one model response cannot mix inference policy versions",
            )
            response_ids.append(str(model_call_id))
        elif event.kind == "reward_assigned":
            reward_events.append(event)

    _require(bool(request_ids), "trace contains no policy model calls")
    _require(
        len(set(request_ids)) == len(request_ids),
        "trace contains duplicate model request IDs",
    )
    _require(
        response_ids == request_ids,
        "model request and response IDs are incomplete or out of order",
    )
    _require(len(reward_events) == 1, "trace must contain exactly one reward event")
    reward_payload = reward_events[0].payload
    _require(
        reward_payload.get("target_model_call_ids") == request_ids,
        "reward targets do not cover the ordered policy model calls",
    )
    _require(
        _is_finite_number(reward_payload.get("reward"))
        and math.isclose(
            float(reward_payload["reward"]),
            float(trace.reward),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "reward event differs from trace reward",
    )
    return {
        "ok": True,
        "episode_id": trace.episode_id,
        "task_id": trace.task_id,
        "joint_version_id": trace.joint_version.version_id,
        "validity_class": trace.validity_class,
        "reward": float(trace.reward),
        "model_call_ids": request_ids,
        "trace_sha256": hashlib.sha256(_canonical_json(trace_payload)).hexdigest(),
    }


def build_agent_service_interaction_sidecar(
    *,
    trace: EpisodeTrace,
    session: AgentServiceSessionReceipt,
    model_calls: Sequence[AgentServiceModelCallReceipt],
    trajectory: AgentServiceTrajectoryReceipt,
) -> dict[str, object]:
    """Bind a closed Agent trace to the pre-export AReaL interaction cache."""

    trace_audit = validate_agent_service_training_trace(trace)
    _require(_is_non_empty_string(session.group_id), "session group ID is missing")
    _require(_is_non_empty_string(session.session_id), "session ID is missing")
    _require(
        _is_int(trajectory.trajectory_id) and trajectory.trajectory_id >= 0,
        "trajectory ID must be a non-negative integer",
    )
    _require(
        _is_int(trajectory.interaction_count) and trajectory.interaction_count > 0,
        "trajectory interaction count must be a positive integer",
    )
    _require(
        type(trajectory.ready_transition) is bool,
        "trajectory ready-transition flag must be boolean",
    )
    _require(
        trajectory.session_id == session.session_id,
        "reward receipt session differs from start-session receipt",
    )
    _require(
        trajectory.interaction_count == len(model_calls),
        "reward receipt interaction count differs from captured model calls",
    )
    _require(bool(model_calls), "Agent Service sidecar requires model calls")
    _require(
        all(_is_int(call.ordinal) and call.ordinal >= 0 for call in model_calls),
        "captured model call ordinal must be a non-negative integer",
    )
    _require(
        [call.ordinal for call in model_calls] == list(range(len(model_calls))),
        "captured model call ordinals must be contiguous and ordered",
    )
    _require(
        [call.model_call_id for call in model_calls]
        == trace_audit["model_call_ids"],
        "captured model calls differ from EpisodeTrace",
    )
    interaction_ids = [call.interaction_id for call in model_calls]
    _require(
        len(set(interaction_ids)) == len(interaction_ids),
        "captured AReaL interaction ID is bound more than once",
    )

    by_model_call: dict[str, AgentServiceModelCallReceipt] = {}
    bindings: list[InteractionBinding] = []
    for call in model_calls:
        _require(
            _is_non_empty_string(call.model_call_id)
            and _is_non_empty_string(call.interaction_id),
            "captured model call identity is incomplete",
        )
        parent_interaction_id: str | None = None
        if call.parent_model_call_id is not None:
            _require(
                call.parent_model_call_id in by_model_call,
                "parent model call must precede its child",
            )
            parent_interaction_id = by_model_call[
                call.parent_model_call_id
            ].interaction_id
        by_model_call[call.model_call_id] = call
        bindings.append(
            InteractionBinding(
                episode_id=str(trace_audit["episode_id"]),
                model_call_id=call.model_call_id,
                session_id=session.session_id,
                trajectory_id=trajectory.trajectory_id,
                interaction_id=call.interaction_id,
                parent_interaction_id=parent_interaction_id,
                ordinal=call.ordinal,
                joint_version_id=str(trace_audit["joint_version_id"]),
                route_kind="agent-service-session",
            )
        )
    try:
        return build_interaction_adapter_sidecar(bindings)
    except ArealInteractionAdapterError as exc:
        raise ArealAgentServiceAdapterError(str(exc)) from exc


def prepare_agent_service_training_record(
    *,
    trace: EpisodeTrace,
    session: AgentServiceSessionReceipt,
    model_calls: Sequence[AgentServiceModelCallReceipt],
    trajectory: AgentServiceTrajectoryReceipt,
    interaction_cache: Any | None = None,
    exported_interactions: Mapping[str, Any] | None = None,
    export_style: str,
    turn_discount: float | None,
) -> dict[str, object]:
    """Prepare a record before AReaL merges interactions into a batch tensor.

    The preferred hook receives the styled interaction mapping returned by
    ``SessionData.export_trajectory()``. A raw ``InteractionCache`` is also
    supported. Both inputs precede ``concat_padded_tensors()``; the public
    ``/export_trajectories`` response is too late because AReaL v2.0.0 has
    already replaced interaction identities with padded batch rows.
    """

    _require(
        (interaction_cache is None) != (exported_interactions is None),
        "provide exactly one pre-merge AReaL interaction source",
    )
    trace_audit = validate_agent_service_training_trace(trace)
    sidecar = build_agent_service_interaction_sidecar(
        trace=trace,
        session=session,
        model_calls=model_calls,
        trajectory=trajectory,
    )
    try:
        if exported_interactions is not None:
            archive = archive_premerged_exported_interactions(
                exported_interactions=exported_interactions,
                interaction_sidecar=sidecar,
                export_style=export_style,
                turn_discount=turn_discount,
            )
        else:
            _require(
                hasattr(interaction_cache, "export_interactions"),
                "adapter must run before AReaL merges interactions into batch tensors",
            )
            archive = export_bound_training_sample_archive(
                interaction_cache=interaction_cache,
                interaction_sidecar=sidecar,
                export_style=export_style,
                turn_discount=turn_discount,
            )
    except ArealAgentServiceAdapterError:
        raise
    except ArealInteractionAdapterError as exc:
        raise ArealAgentServiceAdapterError(str(exc)) from exc
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "episode_id": trace_audit["episode_id"],
            "task_id": trace_audit["task_id"],
            "group_id": session.group_id,
            "session_id": session.session_id,
            "trajectory_id": trajectory.trajectory_id,
            "joint_version_id": trace_audit["joint_version_id"],
        },
        "trace": {
            "trace_sha256": trace_audit["trace_sha256"],
            "validity_class": trace_audit["validity_class"],
            "reward": trace_audit["reward"],
            "model_call_ids": trace_audit["model_call_ids"],
        },
        "ready_transition": trajectory.ready_transition,
        "training_archive": archive,
        "evidence_scope": {
            "pre_batch_interaction_binding": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    validate_agent_service_training_record(record)
    return record


def validate_agent_service_training_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    _require(record.get("schema_version") == SCHEMA_VERSION, "unknown schema")
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "Agent Service training record hash mismatch",
    )
    _assert_no_secret_fields(record)
    identity = record.get("identity")
    trace = record.get("trace")
    archive = record.get("training_archive")
    _require(isinstance(identity, Mapping), "training record identity is missing")
    _require(isinstance(trace, Mapping), "training record trace summary is missing")
    _require(isinstance(archive, Mapping), "training archive is missing")
    archive_audit = validate_bound_training_sample_archive(archive)
    sidecar = archive["interaction_sidecar"]
    bindings = sidecar["bindings"]
    _require(
        identity.get("episode_id") == archive_audit["episode_id"],
        "record episode differs from training archive",
    )
    _require(
        identity.get("session_id") == bindings[0]["session_id"]
        and identity.get("trajectory_id") == bindings[0]["trajectory_id"]
        and identity.get("joint_version_id") == bindings[0]["joint_version_id"],
        "record route identity differs from interaction sidecar",
    )
    _require(
        _is_non_empty_string(identity.get("task_id"))
        and _is_non_empty_string(identity.get("group_id")),
        "record task or group identity is missing",
    )
    _require(
        _is_sha256(trace.get("trace_sha256")),
        "trace SHA-256 is invalid",
    )
    _require(
        trace.get("validity_class") in {"valid", "policy_failure"}
        and _is_finite_number(trace.get("reward")),
        "record contains an ineligible trace",
    )
    _require(
        trace.get("model_call_ids")
        == [binding["model_call_id"] for binding in bindings],
        "record model calls differ from interaction sidecar",
    )
    _require(
        type(record.get("ready_transition")) is bool,
        "ready-transition flag is invalid",
    )
    _require(
        record.get("evidence_scope")
        == {
            "pre_batch_interaction_binding": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "Agent Service evidence scope differs from contract",
    )
    return {
        "ok": True,
        "episode_id": identity["episode_id"],
        "session_id": identity["session_id"],
        "trajectory_id": identity["trajectory_id"],
        "model_call_count": len(bindings),
        "sample_count": archive_audit["sample_count"],
        "export_style": archive_audit["export_style"],
        "record_sha256": record["record_sha256"],
    }


def write_agent_service_training_record(
    record: Mapping[str, object],
    *,
    destination: str | Path,
    allowed_root: str | Path,
) -> Path:
    validate_agent_service_training_record(record)
    root = Path(allowed_root).expanduser().resolve()
    path = Path(destination).expanduser().resolve()
    _require(root.is_dir(), f"configured root does not exist: {root}")
    try:
        common = Path(os.path.commonpath((path, root)))
    except ValueError as exc:
        raise ArealAgentServiceAdapterError(
            "cannot compare output path with configured root"
        ) from exc
    _require(common == root, f"path escapes configured root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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
