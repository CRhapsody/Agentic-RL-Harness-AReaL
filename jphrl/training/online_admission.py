from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass

from jphrl.trajectory.areal_agent_service_adapter import (
    validate_agent_service_training_trace,
)
from jphrl.trajectory.joint_credit_alignment import (
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.schema import EpisodeTrace, JointVersion

LEASE_SCHEMA_VERSION = "jph.release-lease.v1"
DECISION_SCHEMA_VERSION = "jph.lag-zero-admission-decision.v1"
RESAMPLE_SCHEMA_VERSION = "jph.lag-zero-resample-request.v1"

_ACCEPTED = "accepted"
_POLICY_STALE = "policy_stale"
_HARNESS_STALE = "harness_stale"
_BOTH_STALE = "both_stale"
_CONTRACT_MISMATCH = "contract_mismatch"
_PUBLISH_STRADDLE = "publish_straddle"
_STALE_REASONS = {_POLICY_STALE, _HARNESS_STALE, _BOTH_STALE}
_REASONS = _STALE_REASONS | {
    _ACCEPTED,
    _CONTRACT_MISMATCH,
    _PUBLISH_STRADDLE,
}
_NON_DUAL_AXIS_FIELDS = (
    "harness_artifact",
    "tool_schema",
    "parser",
    "environment",
    "evaluator",
    "tokenizer",
    "context_builder",
)
_SECRET_FIELD_NAMES = {
    "admin_api_key",
    "api_key",
    "authorization",
    "session_api_key",
}
_EVIDENCE_SCOPE = {
    "lag_zero_admission": True,
    "source_episode_revalidated": True,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}


class OnlineAdmissionError(ValueError):
    """Raised when online training or resampling cannot fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OnlineAdmissionError(message)


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
        raise OnlineAdmissionError(
            "online admission is not finite canonical JSON"
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


def _public_identifier(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    identifier = str(value)
    _require(
        not identifier.lower().startswith(("sk-", "bearer ")),
        f"{label} looks like a credential",
    )
    return identifier


def _assert_no_secret_fields(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                str(key).lower() not in _SECRET_FIELD_NAMES,
                f"credential field cannot enter online admission: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


def _joint_version_from_record(raw: object, label: str) -> JointVersion:
    _require(isinstance(raw, Mapping), f"{label} must be an object")
    _require(
        set(raw) == set(JointVersion.__dataclass_fields__),
        f"{label} field set differs from schema",
    )
    try:
        version = JointVersion(**dict(raw))
    except TypeError as exc:
        raise OnlineAdmissionError(f"invalid {label}") from exc
    _require(
        all(
            isinstance(getattr(version, field), str) and bool(getattr(version, field))
            for field in JointVersion.__dataclass_fields__
        ),
        f"{label} fields must be non-empty strings",
    )
    return version


@dataclass(frozen=True)
class ReleaseLease:
    """One atomically-read active release and both of its update generations."""

    release_id: str
    joint_version: JointVersion
    policy_generation: int
    harness_generation: int
    macro_step: int

    def validate(self) -> None:
        _public_identifier(self.release_id, "release ID")
        _require(
            type(self.joint_version) is JointVersion,
            "release lease JointVersion has the wrong type",
        )
        _joint_version_from_record(asdict(self.joint_version), "release JointVersion")
        for field in ("policy_generation", "harness_generation", "macro_step"):
            value = getattr(self, field)
            _require(
                type(value) is int and value >= 0,
                f"release lease {field} must be a non-negative integer",
            )

    def to_record(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": LEASE_SCHEMA_VERSION,
            "release_id": self.release_id,
            "joint_version": asdict(self.joint_version),
            "policy_generation": self.policy_generation,
            "harness_generation": self.harness_generation,
            "macro_step": self.macro_step,
        }
        payload["record_sha256"] = _record_sha256(payload)
        return payload

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ReleaseLease:
        expected = {
            "schema_version",
            "release_id",
            "joint_version",
            "policy_generation",
            "harness_generation",
            "macro_step",
            "record_sha256",
        }
        _require(set(record) == expected, "release lease field set differs")
        _require(
            record.get("schema_version") == LEASE_SCHEMA_VERSION,
            "unknown release lease schema",
        )
        _require(
            record.get("record_sha256") == _record_sha256(record),
            "release lease hash mismatch",
        )
        _assert_no_secret_fields(record)
        version = _joint_version_from_record(
            record.get("joint_version"), "release JointVersion"
        )
        lease = cls(
            release_id=record.get("release_id"),
            joint_version=version,
            policy_generation=record.get("policy_generation"),
            harness_generation=record.get("harness_generation"),
            macro_step=record.get("macro_step"),
        )
        lease.validate()
        _require(
            lease.to_record() == dict(record),
            "release lease record is not canonical",
        )
        return lease


def _lease(value: ReleaseLease | Mapping[str, object], label: str) -> ReleaseLease:
    if type(value) is ReleaseLease:
        value.validate()
        return value
    _require(isinstance(value, Mapping), f"{label} must be a ReleaseLease record")
    return ReleaseLease.from_record(value)


@dataclass(frozen=True)
class LagAdmissionDecision:
    """Auditable lag-zero decision; false optimizer evidence is intentional."""

    accepted: bool
    reason: str
    policy_lag: int | None
    harness_lag: int | None
    behavior_lease: ReleaseLease
    active_lease: ReleaseLease
    source_episode_id: str | None = None
    source_record_sha256: str | None = None
    source_input_sha256: str | None = None
    resample_request: Mapping[str, object] | None = None

    def validate(self) -> None:
        self.behavior_lease.validate()
        self.active_lease.validate()
        _require(type(self.accepted) is bool, "admission flag must be boolean")
        _require(self.reason in _REASONS, "unknown lag admission reason")
        _require(
            self.accepted == (self.reason == _ACCEPTED),
            "lag decision result differs from its reason",
        )
        for label, lag in (("Policy", self.policy_lag), ("Harness", self.harness_lag)):
            _require(
                lag is None or (type(lag) is int and lag >= 0),
                f"{label} lag must be a non-negative integer or null",
            )
        expected = _classify_release_leases(
            self.behavior_lease,
            self.active_lease,
        )
        _require(
            (self.accepted, self.reason, self.policy_lag, self.harness_lag) == expected,
            "lag decision differs from its release leases",
        )
        if self.source_episode_id is None:
            _require(
                self.source_record_sha256 is None
                and self.source_input_sha256 is None
                and self.resample_request is None,
                "lease-only decision cannot contain source training fields",
            )
        else:
            _public_identifier(self.source_episode_id, "source episode ID")
            _require(
                _is_sha256(self.source_record_sha256),
                "source S record SHA-256 is invalid",
            )
            _require(
                _is_sha256(self.source_input_sha256),
                "source input SHA-256 is invalid",
            )
        if self.resample_request is not None:
            _require(
                not self.accepted and self.reason in _STALE_REASONS,
                "only a stale episode may carry a resample request",
            )
            validate_resample_request(self.resample_request)
            request_source = self.resample_request["rejected_source_identity"]
            _require(
                self.resample_request["reason"] == self.reason
                and ReleaseLease.from_record(self.resample_request["target_lease"])
                == self.active_lease
                and request_source["episode_id"] == self.source_episode_id
                and _sha256(self.resample_request["task"]["input"])
                == self.source_input_sha256,
                "resample request differs from its lag decision",
            )
        elif self.source_episode_id is not None and self.reason in _STALE_REASONS:
            raise OnlineAdmissionError(
                "stale source episode requires a resample request"
            )

    def to_record(self) -> dict[str, object]:
        self.validate()
        record: dict[str, object] = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "accepted": self.accepted,
            "reason": self.reason,
            "lag": {
                "policy": self.policy_lag,
                "harness": self.harness_lag,
            },
            "behavior_lease": self.behavior_lease.to_record(),
            "active_lease": self.active_lease.to_record(),
            "source": None
            if self.source_episode_id is None
            else {
                "episode_id": self.source_episode_id,
                "s_record_sha256": self.source_record_sha256,
                "input_sha256": self.source_input_sha256,
            },
            "resample_request": deepcopy(self.resample_request),
            "evidence_scope": dict(_EVIDENCE_SCOPE),
        }
        _assert_no_secret_fields(record)
        record["record_sha256"] = _record_sha256(record)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> LagAdmissionDecision:
        expected = {
            "schema_version",
            "accepted",
            "reason",
            "lag",
            "behavior_lease",
            "active_lease",
            "source",
            "resample_request",
            "evidence_scope",
            "record_sha256",
        }
        _require(set(record) == expected, "lag decision field set differs")
        _require(
            record.get("schema_version") == DECISION_SCHEMA_VERSION,
            "unknown lag decision schema",
        )
        _require(
            record.get("record_sha256") == _record_sha256(record),
            "lag decision hash mismatch",
        )
        _assert_no_secret_fields(record)
        _require(
            record.get("evidence_scope") == _EVIDENCE_SCOPE,
            "lag decision evidence scope differs",
        )
        lag = record.get("lag")
        _require(
            isinstance(lag, Mapping) and set(lag) == {"policy", "harness"},
            "lag field set differs",
        )
        source = record.get("source")
        _require(
            source is None
            or (
                isinstance(source, Mapping)
                and set(source) == {"episode_id", "s_record_sha256", "input_sha256"}
            ),
            "lag decision source field set differs",
        )
        behavior_lease = record.get("behavior_lease")
        active_lease = record.get("active_lease")
        _require(
            isinstance(behavior_lease, Mapping) and isinstance(active_lease, Mapping),
            "lag decision leases must be objects",
        )
        decision = cls(
            accepted=record.get("accepted"),
            reason=record.get("reason"),
            policy_lag=lag.get("policy"),
            harness_lag=lag.get("harness"),
            behavior_lease=ReleaseLease.from_record(behavior_lease),
            active_lease=ReleaseLease.from_record(active_lease),
            source_episode_id=None if source is None else source.get("episode_id"),
            source_record_sha256=None
            if source is None
            else source.get("s_record_sha256"),
            source_input_sha256=None if source is None else source.get("input_sha256"),
            resample_request=deepcopy(record.get("resample_request")),
        )
        decision.validate()
        _require(
            decision.to_record() == dict(record),
            "lag decision record is not canonical",
        )
        return decision


def _classify_release_leases(
    behavior: ReleaseLease,
    active: ReleaseLease,
) -> tuple[bool, str, int | None, int | None]:
    behavior.validate()
    active.validate()
    if any(
        getattr(behavior.joint_version, field) != getattr(active.joint_version, field)
        for field in _NON_DUAL_AXIS_FIELDS
    ):
        return False, _CONTRACT_MISMATCH, None, None

    policy_delta = active.policy_generation - behavior.policy_generation
    harness_delta = active.harness_generation - behavior.harness_generation
    if policy_delta < 0 or harness_delta < 0:
        return False, _PUBLISH_STRADDLE, None, None

    policy_same = behavior.joint_version.policy == active.joint_version.policy
    harness_same = (
        behavior.joint_version.harness_controller
        == active.joint_version.harness_controller
    )
    if policy_same != (policy_delta == 0) or harness_same != (harness_delta == 0):
        return False, _PUBLISH_STRADDLE, policy_delta, harness_delta

    if policy_delta or harness_delta:
        if (
            behavior.release_id == active.release_id
            or behavior.macro_step >= active.macro_step
        ):
            return False, _PUBLISH_STRADDLE, policy_delta, harness_delta
        if policy_delta and harness_delta:
            return False, _BOTH_STALE, policy_delta, harness_delta
        if policy_delta:
            return False, _POLICY_STALE, policy_delta, 0
        return False, _HARNESS_STALE, 0, harness_delta

    if (
        behavior.joint_version != active.joint_version
        or behavior.release_id != active.release_id
        or behavior.macro_step != active.macro_step
    ):
        return False, _PUBLISH_STRADDLE, 0, 0
    return True, _ACCEPTED, 0, 0


def decide_lag_zero_admission(
    behavior_lease: ReleaseLease | Mapping[str, object],
    active_lease: ReleaseLease | Mapping[str, object],
) -> LagAdmissionDecision:
    """Classify a behavior release against one atomically-read active lease."""

    behavior = _lease(behavior_lease, "behavior lease")
    active = _lease(active_lease, "active lease")
    accepted, reason, policy_lag, harness_lag = _classify_release_leases(
        behavior, active
    )
    decision = LagAdmissionDecision(
        accepted=accepted,
        reason=reason,
        policy_lag=policy_lag,
        harness_lag=harness_lag,
        behavior_lease=behavior,
        active_lease=active,
    )
    decision.validate()
    return decision


def _source_fields(
    s_record: Mapping[str, object],
    source_episode: EpisodeTrace,
    behavior: ReleaseLease,
) -> tuple[dict[str, object], str, tuple[str, ...]]:
    _assert_no_secret_fields(s_record)
    _require(type(source_episode) is EpisodeTrace, "source episode has the wrong type")
    try:
        source_episode_record = source_episode.to_dict()
    except ValueError as exc:
        raise OnlineAdmissionError(str(exc)) from exc
    _assert_no_secret_fields(source_episode_record, "source_episode")
    _require(
        isinstance(s_record.get("joint_version"), Mapping),
        "S record JointVersion is missing",
    )
    record_version = _joint_version_from_record(
        s_record["joint_version"], "S record JointVersion"
    )
    _require(
        record_version == behavior.joint_version,
        "S record full JointVersion differs from behavior lease",
    )
    try:
        s_audit = validate_frozen_joint_credit_alignment(
            s_record,
            active_joint_version=behavior.joint_version,
        )
        trace_audit = validate_agent_service_training_trace(source_episode)
    except ValueError as exc:
        raise OnlineAdmissionError(str(exc)) from exc
    _require(
        source_episode.joint_version == behavior.joint_version,
        "source episode full JointVersion differs from behavior lease",
    )
    _require(
        s_audit.get("joint_version_id") == behavior.joint_version.version_id,
        "S audit JointVersion differs from behavior lease",
    )
    identity = s_record.get("identity")
    admissions = s_record.get("admissions")
    _require(isinstance(identity, Mapping), "S record identity is missing")
    _require(isinstance(admissions, Mapping), "S admissions are missing")
    policy_record = admissions.get("policy_admission_record")
    _require(
        isinstance(policy_record, Mapping),
        "S Policy admission record is missing",
    )
    policy_identity = policy_record.get("identity")
    _require(
        isinstance(policy_identity, Mapping),
        "S Policy route identity is missing",
    )
    episode_id = _public_identifier(identity.get("episode_id"), "S episode ID")
    _require(
        episode_id
        == source_episode.episode_id
        == trace_audit.get("episode_id")
        == s_audit.get("episode_id"),
        "S record and source episode identity differ",
    )
    _require(
        source_episode.task_id
        == trace_audit.get("task_id")
        == policy_identity.get("task_id"),
        "S record and source task identity differ",
    )
    _require(
        identity.get("trace_sha256") == trace_audit.get("trace_sha256"),
        "S record and source episode hash differ",
    )
    model_call_ids = identity.get("model_call_ids")
    _require(
        isinstance(model_call_ids, list)
        and bool(model_call_ids)
        and model_call_ids == trace_audit.get("model_call_ids"),
        "S record and source episode model-call identities differ",
    )
    session_id = _public_identifier(
        policy_identity.get("session_id"), "source session ID"
    )
    _require(
        type(source_episode.seed) is int,
        "source episode seed must be an integer",
    )
    return s_audit, session_id, tuple(str(value) for value in model_call_ids)


def _build_resample_request(
    *,
    reason: str,
    source_episode: EpisodeTrace,
    source_input: object,
    source_session_id: str,
    source_model_call_ids: tuple[str, ...],
    active_lease: ReleaseLease,
) -> dict[str, object]:
    _require(reason in _STALE_REASONS, "only stale work may be resampled")
    semantic_input = json.loads(_canonical_json(source_input).decode("utf-8"))
    request: dict[str, object] = {
        "schema_version": RESAMPLE_SCHEMA_VERSION,
        "reason": reason,
        "task": {
            "task_id": source_episode.task_id,
            "seed": source_episode.seed,
            "input": semantic_input,
        },
        "target_lease": active_lease.to_record(),
        "rejected_source_identity": {
            "episode_id": source_episode.episode_id,
            "session_id": source_session_id,
            "model_call_ids": list(source_model_call_ids),
        },
        "fresh_identity_required": {
            "episode_id": True,
            "session_id": True,
            "all_model_call_ids": True,
        },
        "evidence_scope": {
            "resample_requested": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    _assert_no_secret_fields(request)
    request["record_sha256"] = _record_sha256(request)
    validate_resample_request(request)
    return request


def validate_resample_request(record: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema_version",
        "reason",
        "task",
        "target_lease",
        "rejected_source_identity",
        "fresh_identity_required",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == expected, "resample request field set differs")
    _require(
        record.get("schema_version") == RESAMPLE_SCHEMA_VERSION,
        "unknown resample request schema",
    )
    _require(record.get("reason") in _STALE_REASONS, "invalid resample reason")
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "resample request hash mismatch",
    )
    _assert_no_secret_fields(record)
    task = record.get("task")
    _require(
        isinstance(task, Mapping) and set(task) == {"task_id", "seed", "input"},
        "resample task field set differs",
    )
    _public_identifier(task.get("task_id"), "resample task ID")
    _require(type(task.get("seed")) is int, "resample seed must be an integer")
    _canonical_json(task.get("input"))
    target_record = record.get("target_lease")
    _require(
        isinstance(target_record, Mapping), "resample target lease must be an object"
    )
    target = ReleaseLease.from_record(target_record)
    source = record.get("rejected_source_identity")
    _require(
        isinstance(source, Mapping)
        and set(source) == {"episode_id", "session_id", "model_call_ids"},
        "resample source identity field set differs",
    )
    _public_identifier(source.get("episode_id"), "rejected episode ID")
    _public_identifier(source.get("session_id"), "rejected session ID")
    call_ids = source.get("model_call_ids")
    _require(
        isinstance(call_ids, list)
        and bool(call_ids)
        and len(set(call_ids)) == len(call_ids)
        and all(isinstance(value, str) and bool(value) for value in call_ids),
        "rejected model-call IDs are invalid",
    )
    _require(
        record.get("fresh_identity_required")
        == {
            "episode_id": True,
            "session_id": True,
            "all_model_call_ids": True,
        },
        "resample request does not require fresh identities",
    )
    _require(
        record.get("evidence_scope")
        == {
            "resample_requested": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "resample evidence scope differs",
    )
    return {
        "ok": True,
        "reason": record["reason"],
        "task_id": task["task_id"],
        "seed": task["seed"],
        "target_release_id": target.release_id,
        "source_episode_id": source["episode_id"],
        "record_sha256": record["record_sha256"],
    }


def admit_lag_zero_training_record(
    *,
    s_record: Mapping[str, object],
    source_episode: EpisodeTrace,
    source_input: object,
    behavior_lease: ReleaseLease | Mapping[str, object],
    active_lease: ReleaseLease | Mapping[str, object],
) -> LagAdmissionDecision:
    """Validate S and its source before accepting, or resample valid stale work."""

    behavior = _lease(behavior_lease, "behavior lease")
    active = _lease(active_lease, "active lease")
    _assert_no_secret_fields(source_input, "source_input")
    source_record_sha256 = s_record.get("record_sha256")
    _require(_is_sha256(source_record_sha256), "source S record SHA-256 is invalid")
    source_input_sha256 = _sha256(source_input)
    accepted, reason, policy_lag, harness_lag = _classify_release_leases(
        behavior, active
    )
    _, source_session_id, source_model_call_ids = _source_fields(
        s_record, source_episode, behavior
    )
    resample_request = None
    if reason in _STALE_REASONS:
        resample_request = _build_resample_request(
            reason=reason,
            source_episode=source_episode,
            source_input=source_input,
            source_session_id=source_session_id,
            source_model_call_ids=source_model_call_ids,
            active_lease=active,
        )
    decision = LagAdmissionDecision(
        accepted=accepted,
        reason=reason,
        policy_lag=policy_lag,
        harness_lag=harness_lag,
        behavior_lease=behavior,
        active_lease=active,
        source_episode_id=source_episode.episode_id,
        source_record_sha256=str(source_record_sha256),
        source_input_sha256=source_input_sha256,
        resample_request=resample_request,
    )
    decision.validate()
    return decision


def revalidate_before_training(
    admission: LagAdmissionDecision | Mapping[str, object],
    *,
    s_record: Mapping[str, object],
    source_episode: EpisodeTrace,
    source_input: object,
    active_lease: ReleaseLease | Mapping[str, object],
) -> LagAdmissionDecision:
    """Re-read active state and repeat the full S/source gate immediately pre-step."""

    initial = (
        admission
        if type(admission) is LagAdmissionDecision
        else LagAdmissionDecision.from_record(admission)
    )
    initial.validate()
    _require(initial.accepted, "initial lag-zero admission was not accepted")
    _require(
        initial.source_episode_id == source_episode.episode_id
        and initial.source_record_sha256 == s_record.get("record_sha256")
        and initial.source_input_sha256 == _sha256(source_input),
        "training inputs differ from the initial admission",
    )
    return admit_lag_zero_training_record(
        s_record=s_record,
        source_episode=source_episode,
        source_input=source_input,
        behavior_lease=initial.behavior_lease,
        active_lease=active_lease,
    )


def validate_resampled_identity(
    request: Mapping[str, object],
    *,
    episode_id: str,
    session_id: str,
    model_call_ids: Sequence[str],
) -> None:
    """Enforce that a resampler did not reuse any rejected rollout identity."""

    validate_resample_request(request)
    new_episode_id = _public_identifier(episode_id, "resampled episode ID")
    new_session_id = _public_identifier(session_id, "resampled session ID")
    _require(
        isinstance(model_call_ids, Sequence)
        and not isinstance(model_call_ids, (str, bytes))
        and bool(model_call_ids),
        "resampled model-call IDs must be a non-empty sequence",
    )
    new_call_ids = tuple(
        _public_identifier(value, "resampled model-call ID") for value in model_call_ids
    )
    _require(
        len(set(new_call_ids)) == len(new_call_ids),
        "resampled model-call IDs must be unique",
    )
    source = request["rejected_source_identity"]
    old_ids = {
        source["episode_id"],
        source["session_id"],
        *source["model_call_ids"],
    }
    new_ids = {new_episode_id, new_session_id, *new_call_ids}
    _require(
        len(new_ids) == 2 + len(new_call_ids),
        "resampled episode, session, and model-call IDs must be mutually distinct",
    )
    _require(
        old_ids.isdisjoint(new_ids),
        "resample reused an old episode, session, or model-call ID",
    )
