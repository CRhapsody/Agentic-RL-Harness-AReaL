from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .areal_agent_service_adapter import (
    SECRET_FIELD_NAMES,
    AgentServiceModelCallReceipt,
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    ArealAgentServiceAdapterError,
    build_agent_service_interaction_sidecar,
    prepare_agent_service_training_record,
    validate_agent_service_training_record,
    validate_agent_service_training_trace,
    write_agent_service_training_record,
)
from .areal_data_proxy_pre_batch import (
    HOOK_STAGE as PRE_BATCH_HOOK_STAGE,
)
from .areal_data_proxy_pre_batch import (
    VerifiedDataProxyPreBatchHook,
)
from .hermes_model_call_receipts import (
    HermesModelCallReceipt,
    HermesModelCallReceiptError,
    validate_hermes_model_call_receipts,
)
from .schema import EpisodeTrace, JointVersion, TraceEvent

STAGED_BINDING_SCHEMA_VERSION = "jph.areal-agent-service-staged-binding.v1"
FINALIZED_BINDING_SCHEMA_VERSION = "jph.areal-agent-service-finalized-binding.v1"
EXPORT_STYLES = frozenset({"individual", "concat"})


class ArealOnlineBindingError(ArealAgentServiceAdapterError):
    """Raised when an online identity journal cannot be joined safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealOnlineBindingError(message)


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
        raise ArealOnlineBindingError(
            "online binding record is not finite canonical JSON"
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
                f"credential field cannot enter online binding: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


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


def _route_key(session_id: str, trajectory_id: int) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return f"{digest}-{trajectory_id}"


def _resolve_root(journal_root: str | Path) -> Path:
    configured = Path(journal_root).expanduser()
    created = not configured.exists()
    configured.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = configured.resolve()
    _require(root.is_dir(), f"online binding root is not a directory: {root}")
    if created:
        try:
            root.chmod(0o700)
        except OSError as exc:
            raise ArealOnlineBindingError(
                f"cannot protect online binding root: {root}"
            ) from exc
    _require(
        root.stat().st_mode & 0o077 == 0,
        f"online binding root must not be accessible to group or other users: {root}",
    )
    return root


def _require_within_root(path: Path, root: Path) -> None:
    resolved_parent = path.parent.resolve()
    try:
        common = Path(os.path.commonpath((resolved_parent, root)))
    except ValueError as exc:
        raise ArealOnlineBindingError(
            "cannot compare online binding path with configured root"
        ) from exc
    _require(common == root, f"online binding path escapes configured root: {path}")


def _private_write_new(
    record: Mapping[str, object], *, path: Path, root: Path
) -> Path:
    _require_within_root(path, root)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
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


def _private_read(path: Path, *, root: Path) -> dict[str, object]:
    _require_within_root(path, root)
    resolved = path.resolve(strict=True)
    try:
        common = Path(os.path.commonpath((resolved, root)))
    except ValueError as exc:
        raise ArealOnlineBindingError(
            "cannot compare online binding record with configured root"
        ) from exc
    _require(common == root, f"online binding record escapes configured root: {path}")
    _require(
        resolved.stat().st_mode & 0o077 == 0,
        f"online binding record is accessible to group or other users: {path}",
    )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArealOnlineBindingError(
            f"cannot read online binding record: {path}"
        ) from exc
    _require(isinstance(value, dict), "online binding record must be an object")
    return value


def _instantiate_exact(dataclass_type: type[Any], raw: object, label: str) -> Any:
    _require(isinstance(raw, Mapping), f"{label} must be an object")
    expected_fields = set(dataclass_type.__dataclass_fields__)
    _require(set(raw) == expected_fields, f"{label} field set differs from schema")
    try:
        return dataclass_type(**dict(raw))
    except TypeError as exc:
        raise ArealOnlineBindingError(f"invalid {label}") from exc


def _episode_trace_from_dict(raw: object) -> EpisodeTrace:
    _require(isinstance(raw, Mapping), "staged EpisodeTrace must be an object")
    expected_fields = set(EpisodeTrace.__dataclass_fields__)
    _require(
        set(raw) == expected_fields,
        "staged EpisodeTrace field set differs from schema",
    )
    joint_version = _instantiate_exact(
        JointVersion,
        raw.get("joint_version"),
        "staged JointVersion",
    )
    raw_events = raw.get("events")
    _require(isinstance(raw_events, list), "staged trace events must be a list")
    events = [
        _instantiate_exact(TraceEvent, event, "staged TraceEvent")
        for event in raw_events
    ]
    try:
        trace = EpisodeTrace(
            episode_id=raw["episode_id"],
            task_id=raw["task_id"],
            seed=raw["seed"],
            joint_version=joint_version,
            harness_spec_hash=raw["harness_spec_hash"],
            events=events,
            reward=raw["reward"],
            success=raw["success"],
            validity_class=raw["validity_class"],
            failure_category=raw["failure_category"],
        )
        trace.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ArealOnlineBindingError(f"invalid staged EpisodeTrace: {exc}") from exc
    return trace


@dataclass(frozen=True)
class ValidatedStagedBinding:
    trace: EpisodeTrace
    session: AgentServiceSessionReceipt
    model_calls: tuple[AgentServiceModelCallReceipt, ...]
    trajectory: AgentServiceTrajectoryReceipt
    export_style: str
    turn_discount: float | None
    interaction_sidecar: Mapping[str, object]


def _agent_service_receipts_from_hermes(
    receipts: Sequence[HermesModelCallReceipt],
    *,
    expected_session_id: str,
) -> tuple[AgentServiceModelCallReceipt, ...]:
    try:
        validate_hermes_model_call_receipts(
            receipts,
            expected_session_id=expected_session_id,
        )
    except HermesModelCallReceiptError as exc:
        raise ArealOnlineBindingError(str(exc)) from exc
    return tuple(
        AgentServiceModelCallReceipt(
            model_call_id=receipt.model_call_id,
            interaction_id=receipt.interaction_id,
            ordinal=receipt.ordinal,
            parent_model_call_id=receipt.parent_model_call_id,
        )
        for receipt in receipts
    )


def validate_staged_agent_service_binding(
    record: Mapping[str, object],
) -> ValidatedStagedBinding:
    """Rehydrate and revalidate an immutable pre-export identity record."""

    required_fields = {
        "schema_version",
        "identity",
        "trace",
        "session_receipt",
        "hermes_model_call_receipts",
        "trajectory_receipt",
        "interaction_sidecar",
        "export",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == required_fields, "staged binding field set differs")
    _require(
        record.get("schema_version") == STAGED_BINDING_SCHEMA_VERSION,
        "unknown staged binding schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "staged binding hash mismatch",
    )
    _assert_no_secret_fields(record)

    trace = _episode_trace_from_dict(record.get("trace"))
    session = _instantiate_exact(
        AgentServiceSessionReceipt,
        record.get("session_receipt"),
        "session receipt",
    )
    raw_model_calls = record.get("hermes_model_call_receipts")
    _require(
        isinstance(raw_model_calls, list) and bool(raw_model_calls),
        "staged binding requires model-call receipts",
    )
    hermes_model_calls = tuple(
        _instantiate_exact(
            HermesModelCallReceipt,
            raw,
            "Hermes model-call receipt",
        )
        for raw in raw_model_calls
    )
    trajectory = _instantiate_exact(
        AgentServiceTrajectoryReceipt,
        record.get("trajectory_receipt"),
        "trajectory receipt",
    )
    model_calls = _agent_service_receipts_from_hermes(
        hermes_model_calls,
        expected_session_id=session.session_id,
    )

    try:
        trace_audit = validate_agent_service_training_trace(trace)
        expected_sidecar = build_agent_service_interaction_sidecar(
            trace=trace,
            session=session,
            model_calls=model_calls,
            trajectory=trajectory,
        )
    except ArealAgentServiceAdapterError as exc:
        raise ArealOnlineBindingError(str(exc)) from exc
    _require(
        record.get("interaction_sidecar") == expected_sidecar,
        "staged interaction sidecar differs from receipts or trace",
    )

    identity = record.get("identity")
    _require(isinstance(identity, Mapping), "staged identity is missing")
    _require(
        set(identity)
        == {
            "episode_id",
            "task_id",
            "group_id",
            "session_id",
            "trajectory_id",
            "joint_version_id",
        },
        "staged identity field set differs from schema",
    )
    _require(
        identity
        == {
            "episode_id": trace_audit["episode_id"],
            "task_id": trace_audit["task_id"],
            "group_id": session.group_id,
            "session_id": session.session_id,
            "trajectory_id": trajectory.trajectory_id,
            "joint_version_id": trace_audit["joint_version_id"],
        },
        "staged identity differs from receipts or EpisodeTrace",
    )

    export = record.get("export")
    _require(isinstance(export, Mapping), "staged export contract is missing")
    _require(
        set(export) == {"style", "turn_discount"},
        "staged export field set differs from schema",
    )
    export_style = export.get("style")
    turn_discount = export.get("turn_discount")
    _require(export_style in EXPORT_STYLES, "unknown staged export style")
    _require(
        turn_discount is None
        or (
            _is_finite_number(turn_discount)
            and 0.0 <= float(turn_discount) <= 1.0
        ),
        "staged turn discount must be null or in [0, 1]",
    )
    _require(
        record.get("evidence_scope")
        == {
            "hermes_receipts_captured": True,
            "pre_batch_interaction_binding": False,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "staged evidence scope differs from contract",
    )
    return ValidatedStagedBinding(
        trace=trace,
        session=session,
        model_calls=model_calls,
        trajectory=trajectory,
        export_style=str(export_style),
        turn_discount=(
            float(turn_discount) if turn_discount is not None else None
        ),
        interaction_sidecar=expected_sidecar,
    )


def stage_agent_service_training_binding(
    *,
    journal_root: str | Path,
    trace: EpisodeTrace,
    session: AgentServiceSessionReceipt,
    model_calls: Sequence[HermesModelCallReceipt],
    trajectory: AgentServiceTrajectoryReceipt,
    export_style: str,
    turn_discount: float | None,
) -> Path:
    """Persist the non-secret route identity before DataProxy export begins."""

    try:
        trace_audit = validate_agent_service_training_trace(trace)
    except ArealAgentServiceAdapterError as exc:
        raise ArealOnlineBindingError(str(exc)) from exc
    adapter_model_calls = _agent_service_receipts_from_hermes(
        model_calls,
        expected_session_id=session.session_id,
    )
    try:
        sidecar = build_agent_service_interaction_sidecar(
            trace=trace,
            session=session,
            model_calls=adapter_model_calls,
            trajectory=trajectory,
        )
    except ArealAgentServiceAdapterError as exc:
        raise ArealOnlineBindingError(str(exc)) from exc
    _require(export_style in EXPORT_STYLES, "unknown staged export style")
    _require(
        turn_discount is None
        or (
            _is_finite_number(turn_discount)
            and 0.0 <= float(turn_discount) <= 1.0
        ),
        "staged turn discount must be null or in [0, 1]",
    )
    record: dict[str, object] = {
        "schema_version": STAGED_BINDING_SCHEMA_VERSION,
        "identity": {
            "episode_id": trace_audit["episode_id"],
            "task_id": trace_audit["task_id"],
            "group_id": session.group_id,
            "session_id": session.session_id,
            "trajectory_id": trajectory.trajectory_id,
            "joint_version_id": trace_audit["joint_version_id"],
        },
        "trace": trace.to_dict(),
        "session_receipt": asdict(session),
        "hermes_model_call_receipts": [asdict(call) for call in model_calls],
        "trajectory_receipt": asdict(trajectory),
        "interaction_sidecar": sidecar,
        "export": {
            "style": export_style,
            "turn_discount": turn_discount,
        },
        "evidence_scope": {
            "hermes_receipts_captured": True,
            "pre_batch_interaction_binding": False,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    validate_staged_agent_service_binding(record)

    root = _resolve_root(journal_root)
    route_key = _route_key(session.session_id, trajectory.trajectory_id)
    return _private_write_new(
        record,
        path=root / "pending" / f"{route_key}.json",
        root=root,
    )


class PersistentAgentServicePreBatchBinder:
    """Join one staged route to its real pre-batch AReaL interaction mapping."""

    def __init__(self, journal_root: str | Path) -> None:
        self._root = _resolve_root(journal_root)

    def __call__(self, event: object) -> dict[str, object]:
        hook_stage = getattr(event, "hook_stage", None)
        session_id = getattr(event, "session_id", None)
        trajectory_id = getattr(event, "trajectory_id", None)
        exported_interactions = getattr(event, "exported_interactions", None)
        export_style = getattr(event, "export_style", None)
        turn_discount = getattr(event, "turn_discount", None)

        _require(
            hook_stage == PRE_BATCH_HOOK_STAGE,
            "online binding was invoked outside the supported pre-batch stage",
        )
        _require(_is_non_empty_string(session_id), "pre-batch session ID is missing")
        _require(
            _is_int(trajectory_id) and trajectory_id >= 0,
            "pre-batch trajectory ID is invalid",
        )
        _require(
            isinstance(exported_interactions, Mapping),
            "pre-batch exported interactions must be a mapping",
        )

        route_key = _route_key(str(session_id), int(trajectory_id))
        pending_path = self._root / "pending" / f"{route_key}.json"
        record_path = self._root / "records" / f"{route_key}.json"
        finalized_path = self._root / "finalized" / f"{route_key}.json"
        _require(
            not finalized_path.exists(),
            "pre-batch route has already been finalized",
        )
        _require(pending_path.is_file(), "no staged binding exists for pre-batch route")

        staged_record = _private_read(pending_path, root=self._root)
        staged = validate_staged_agent_service_binding(staged_record)
        _require(
            staged.session.session_id == session_id
            and staged.trajectory.trajectory_id == trajectory_id,
            "pre-batch route differs from staged session or trajectory",
        )
        _require(
            staged.export_style == export_style,
            "pre-batch export style differs from staged contract",
        )
        normalized_discount = (
            float(turn_discount) if turn_discount is not None else None
        )
        _require(
            staged.turn_discount == normalized_discount,
            "pre-batch turn discount differs from staged contract",
        )

        try:
            training_record = prepare_agent_service_training_record(
                trace=staged.trace,
                session=staged.session,
                model_calls=staged.model_calls,
                trajectory=staged.trajectory,
                exported_interactions=exported_interactions,
                export_style=staged.export_style,
                turn_discount=staged.turn_discount,
            )
            validate_agent_service_training_record(training_record)
        except ArealAgentServiceAdapterError as exc:
            raise ArealOnlineBindingError(str(exc)) from exc
        _require(
            training_record["training_archive"]["interaction_sidecar"]
            == staged.interaction_sidecar,
            "pre-batch interaction mapping differs from staged Hermes receipts",
        )

        try:
            write_agent_service_training_record(
                training_record,
                destination=record_path,
                allowed_root=self._root,
            )
        except FileExistsError:
            existing = _private_read(record_path, root=self._root)
            validate_agent_service_training_record(existing)
            _require(
                existing == training_record,
                "existing finalized training record differs from current binding",
            )

        marker: dict[str, object] = {
            "schema_version": FINALIZED_BINDING_SCHEMA_VERSION,
            "identity": dict(staged_record["identity"]),
            "training_record_path": str(record_path.relative_to(self._root)),
            "staged_binding_sha256": staged_record["record_sha256"],
            "training_record_sha256": training_record["record_sha256"],
            "evidence_scope": {
                "hermes_receipts_captured": True,
                "pre_batch_interaction_binding": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        marker["record_sha256"] = _record_sha256(marker)
        _assert_no_secret_fields(marker)
        _private_write_new(marker, path=finalized_path, root=self._root)
        return training_record


async def pre_batch_bind_agent_service_training_record(
    *,
    session_id: str,
    trajectory_id: int,
    interactions: Mapping[str, Any],
    discount: float,
    style: str,
) -> None:
    """Deployment callback loaded by the patched AReaL DataProxy entrypoint."""

    journal_root = os.environ.get("JPH_AREAL_AGENT_SERVICE_JOURNAL_ROOT", "")
    _require(
        _is_non_empty_string(journal_root),
        "JPH_AREAL_AGENT_SERVICE_JOURNAL_ROOT is required",
    )
    hook = VerifiedDataProxyPreBatchHook(
        PersistentAgentServicePreBatchBinder(journal_root)
    )
    await hook(
        session_id=session_id,
        trajectory_id=trajectory_id,
        interactions=interactions,
        discount=discount,
        style=style,
    )
