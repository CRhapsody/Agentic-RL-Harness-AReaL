from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jphrl.harness.controller import HarnessState
from jphrl.harness.spec import HarnessAction

from .areal_agent_service_adapter import (
    SECRET_FIELD_NAMES,
    ArealAgentServiceAdapterError,
    validate_agent_service_training_record,
    validate_agent_service_training_trace,
)
from .joint_batch import StaleJointVersionError, require_lag_zero_admission
from .schema import EpisodeTrace, JointVersion

SCHEMA_VERSION = "jph.real-harness-action-admission.v1"
EVIDENCE_SCOPE = {
    "pre_batch_interaction_binding": True,
    "harness_action_samples_admitted": True,
    "harness_advantages_attached": False,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}
_ACTION_IDS = tuple(action.value for action in HarnessAction)
_DECISION_PAYLOAD_FIELDS = {
    "decision_id",
    "action",
    "old_harness_logprob",
    "controller_version",
    "action_ids",
    "action_mask",
    "pre_mask_logits",
    "harness_loss_mask",
    "state",
}
_STATE_FIELDS = set(HarnessState.__dataclass_fields__)


class HarnessActionAdmissionError(ValueError):
    """Raised when Harness behavior evidence is not safe to train on."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessActionAdmissionError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


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
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HarnessActionAdmissionError(
            "Harness admission record is not finite canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _assert_no_secret_fields(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                str(key).lower() not in SECRET_FIELD_NAMES,
                f"credential field cannot enter Harness admission: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


def _masked_logprob(
    *,
    action: str,
    action_ids: Sequence[str],
    action_mask: Sequence[bool],
    pre_mask_logits: Sequence[float],
) -> float:
    selected_index = action_ids.index(action)
    allowed_logits = [
        float(logit) for logit, allowed in zip(pre_mask_logits, action_mask) if allowed
    ]
    maximum = max(allowed_logits)
    log_normalizer = maximum + math.log(
        sum(math.exp(logit - maximum) for logit in allowed_logits)
    )
    return float(pre_mask_logits[selected_index]) - log_normalizer


def _validate_state(state: HarnessState) -> None:
    _require(type(state) is HarnessState, "Harness state must use the frozen schema")
    for field_name in (
        "turn",
        "remaining_tool_calls",
        "remaining_model_retries",
        "context_chars",
    ):
        value = getattr(state, field_name)
        _require(
            _is_int(value) and value >= 0,
            f"Harness state {field_name} must be a non-negative integer",
        )
    _require(
        state.last_error is None or isinstance(state.last_error, str),
        "Harness state last_error must be null or a string",
    )
    _require(
        type(state.retrieval_hit) is bool,
        "Harness state retrieval_hit must be boolean",
    )
    _require(
        _is_non_empty_string(state.verifier_status),
        "Harness state verifier_status must be non-empty",
    )
    _require(
        _is_non_empty_string(state.task_domain),
        "Harness state task_domain must be non-empty",
    )


@dataclass(frozen=True)
class AdmittedHarnessAction:
    """One behavior-time Harness action, before any credit is attached."""

    episode_id: str
    joint_version_id: str
    event_id: str
    event_index: int
    decision_ordinal: int
    decision_id: str
    state: HarnessState
    action: str
    action_ids: tuple[str, ...]
    action_mask: tuple[bool, ...]
    pre_mask_logits: tuple[float, ...]
    old_harness_logprob: float
    harness_loss_mask: int
    harness_behavior_version: str

    def validate(self, joint_version: JointVersion) -> None:
        _require(
            _is_non_empty_string(self.episode_id),
            "Harness sample episode ID must be non-empty",
        )
        _require(
            self.joint_version_id == joint_version.version_id,
            "Harness sample JointVersion ID differs from the batch",
        )
        _require(
            _is_non_empty_string(self.event_id),
            "Harness sample event ID must be non-empty",
        )
        _require(
            _is_int(self.event_index) and self.event_index >= 0,
            "Harness sample event index must be a non-negative integer",
        )
        _require(
            _is_int(self.decision_ordinal) and self.decision_ordinal >= 0,
            "Harness decision ordinal must be a non-negative integer",
        )
        _require(
            _is_non_empty_string(self.decision_id),
            "Harness decision ID must be non-empty",
        )
        _validate_state(self.state)
        _require(
            type(self.action_ids) is tuple and self.action_ids == _ACTION_IDS,
            "Harness action IDs differ from the fixed five-action schema",
        )
        _require(
            type(self.action_mask) is tuple
            and len(self.action_mask) == len(self.action_ids)
            and all(type(value) is bool for value in self.action_mask)
            and any(self.action_mask),
            "Harness action mask must be a non-empty boolean five-action mask",
        )
        _require(
            self.action in self.action_ids
            and self.action_mask[self.action_ids.index(self.action)],
            "chosen Harness action is absent or masked out",
        )
        _require(
            type(self.pre_mask_logits) is tuple
            and len(self.pre_mask_logits) == len(self.action_ids)
            and all(_is_finite_number(value) for value in self.pre_mask_logits),
            "Harness pre-mask logits must be one finite value per action",
        )
        _require(
            _is_finite_number(self.old_harness_logprob)
            and float(self.old_harness_logprob) <= 0.0,
            "old Harness log-prob must be finite and non-positive",
        )
        expected_logprob = _masked_logprob(
            action=self.action,
            action_ids=self.action_ids,
            action_mask=self.action_mask,
            pre_mask_logits=self.pre_mask_logits,
        )
        _require(
            math.isclose(
                float(self.old_harness_logprob),
                expected_logprob,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "old Harness log-prob does not match masked behavior logits",
        )
        _require(
            type(self.harness_loss_mask) is int and self.harness_loss_mask in (0, 1),
            "Harness loss mask must be integer 0 or 1",
        )
        _require(
            self.harness_behavior_version == joint_version.harness_controller,
            "Harness behavior version differs from the admitted JointVersion",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AdmittedHarnessActionBatch:
    """A stable, credit-free Harness sidecar admitted from one P-bound episode."""

    joint_version: JointVersion
    episode_id: str
    trace_sha256: str
    source_training_record_sha256: str
    terminal_reward: float
    validity_class: str
    bound_model_call_ids: tuple[str, ...]
    actions: tuple[AdmittedHarnessAction, ...]

    def validate(self) -> None:
        _require(type(self.joint_version) is JointVersion, "invalid JointVersion")
        _require(
            all(
                _is_non_empty_string(getattr(self.joint_version, field_name))
                for field_name in JointVersion.__dataclass_fields__
            ),
            "JointVersion fields must be non-empty strings",
        )
        _require(
            _is_non_empty_string(self.episode_id),
            "Harness batch episode ID must be non-empty",
        )
        _require(_is_sha256(self.trace_sha256), "trace SHA-256 is invalid")
        _require(
            _is_sha256(self.source_training_record_sha256),
            "source training-record SHA-256 is invalid",
        )
        _require(
            _is_finite_number(self.terminal_reward),
            "Harness batch terminal reward must be finite",
        )
        _require(
            self.validity_class in {"valid", "policy_failure"},
            "Harness batch validity class is not trainable",
        )
        _require(
            (self.validity_class == "valid" and float(self.terminal_reward) == 1.0)
            or (
                self.validity_class == "policy_failure"
                and float(self.terminal_reward) == 0.0
            ),
            "Harness batch terminal outcome differs from EpisodeTrace semantics",
        )
        _require(
            type(self.bound_model_call_ids) is tuple
            and bool(self.bound_model_call_ids)
            and len(set(self.bound_model_call_ids)) == len(self.bound_model_call_ids)
            and all(_is_non_empty_string(value) for value in self.bound_model_call_ids),
            "bound model-call IDs must be unique non-empty strings",
        )
        _require(
            type(self.actions) is tuple and bool(self.actions),
            "Harness batch requires at least one real decision",
        )
        decision_ids: set[str] = set()
        event_ids: set[str] = set()
        event_indexes: list[int] = []
        for expected_ordinal, sample in enumerate(self.actions):
            _require(
                type(sample) is AdmittedHarnessAction,
                "Harness batch contains an unknown sample type",
            )
            sample.validate(self.joint_version)
            _require(
                sample.episode_id == self.episode_id,
                "Harness sample episode differs from the batch",
            )
            _require(
                sample.decision_ordinal == expected_ordinal,
                "Harness decision ordinals must be contiguous and ordered",
            )
            _require(
                sample.decision_id not in decision_ids,
                "duplicate Harness decision ID",
            )
            _require(sample.event_id not in event_ids, "duplicate Harness event ID")
            _require(
                sample.decision_id not in self.bound_model_call_ids,
                "Harness decision ID collides with a policy model-call ID",
            )
            decision_ids.add(sample.decision_id)
            event_ids.add(sample.event_id)
            event_indexes.append(sample.event_index)
        _require(
            event_indexes == sorted(event_indexes)
            and len(set(event_indexes)) == len(event_indexes),
            "Harness decision events must preserve trace order",
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "episode_id": self.episode_id,
                "joint_version_id": self.joint_version.version_id,
            },
            "joint_version": asdict(self.joint_version),
            "source": {
                "trace_sha256": self.trace_sha256,
                "pre_batch_training_record_sha256": (
                    self.source_training_record_sha256
                ),
                "bound_model_call_ids": list(self.bound_model_call_ids),
            },
            "terminal_outcome": {
                "reward": float(self.terminal_reward),
                "validity_class": self.validity_class,
            },
            "actions": [sample.to_dict() for sample in self.actions],
            "evidence_scope": dict(EVIDENCE_SCOPE),
        }

    @property
    def digest(self) -> str:
        self.validate()
        return _sha256(self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        self.validate()
        record = self._unsigned_record()
        _assert_no_secret_fields(record)
        record["record_sha256"] = _sha256(record)
        return record


def _state_from_payload(raw: object) -> HarnessState:
    _require(isinstance(raw, Mapping), "Harness decision state must be an object")
    _require(set(raw) == _STATE_FIELDS, "Harness state field set differs from schema")
    try:
        state = HarnessState(**dict(raw))
    except TypeError as exc:
        raise HarnessActionAdmissionError("invalid Harness state") from exc
    _validate_state(state)
    return state


def _sample_from_event(
    *,
    trace: EpisodeTrace,
    event: object,
    decision_ordinal: int,
) -> AdmittedHarnessAction:
    producer = getattr(event, "producer", None)
    payload = getattr(event, "payload", None)
    _require(
        producer == "harness",
        "Harness decision event must be emitted by the Harness producer",
    )
    _require(isinstance(payload, Mapping), "Harness decision payload must be an object")
    _require(
        set(payload) == _DECISION_PAYLOAD_FIELDS,
        "Harness decision payload field set differs from the live decision schema",
    )
    try:
        sample = AdmittedHarnessAction(
            episode_id=trace.episode_id,
            joint_version_id=trace.joint_version.version_id,
            event_id=event.event_id,
            event_index=event.index,
            decision_ordinal=decision_ordinal,
            decision_id=payload["decision_id"],
            state=_state_from_payload(payload["state"]),
            action=payload["action"],
            action_ids=tuple(payload["action_ids"]),
            action_mask=tuple(payload["action_mask"]),
            pre_mask_logits=tuple(float(value) for value in payload["pre_mask_logits"]),
            old_harness_logprob=float(payload["old_harness_logprob"]),
            harness_loss_mask=payload["harness_loss_mask"],
            harness_behavior_version=payload["controller_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessActionAdmissionError("invalid Harness decision payload") from exc
    sample.validate(trace.joint_version)
    return sample


def admit_real_harness_action_samples(
    *,
    trace: EpisodeTrace,
    active_joint_version: JointVersion,
    pre_batch_training_record: Mapping[str, object],
) -> AdmittedHarnessActionBatch:
    """Admit Harness behavior samples only after P bound the full episode.

    This deliberately accepts neither AReaL policy tensor dictionaries nor a
    post-batch row.  The pre-batch training record supplies the persistent P
    identity, while the complete ``EpisodeTrace`` supplies the original Harness
    state, categorical behavior logits, action mask, and old log-probability.
    Credit and optimizer evidence belong to later DAG stages.
    """

    _require(type(trace) is EpisodeTrace, "admission requires a full EpisodeTrace")
    _require(
        type(active_joint_version) is JointVersion,
        "admission requires an active JointVersion",
    )
    _require(
        isinstance(pre_batch_training_record, Mapping),
        "admission requires a P pre-batch training record",
    )
    try:
        trace_payload = trace.to_dict()
        _assert_no_secret_fields(trace_payload, "trace")
        trace_audit = validate_agent_service_training_trace(trace)
        require_lag_zero_admission(trace, active_joint_version)
        validate_agent_service_training_record(pre_batch_training_record)
    except (
        ArealAgentServiceAdapterError,
        StaleJointVersionError,
        ValueError,
    ) as exc:
        raise HarnessActionAdmissionError(str(exc)) from exc

    identity = pre_batch_training_record["identity"]
    trace_summary = pre_batch_training_record["trace"]
    _require(
        identity["episode_id"] == trace.episode_id,
        "P training record episode differs from EpisodeTrace",
    )
    _require(
        identity["joint_version_id"] == trace.joint_version.version_id,
        "P training record JointVersion differs from EpisodeTrace",
    )
    _require(
        trace_summary["trace_sha256"] == trace_audit["trace_sha256"],
        "P training record trace hash differs from EpisodeTrace",
    )
    _require(
        trace_summary["validity_class"] == trace.validity_class
        and float(trace_summary["reward"]) == float(trace.reward),
        "P training record terminal outcome differs from EpisodeTrace",
    )
    _require(
        trace_summary["model_call_ids"] == trace_audit["model_call_ids"],
        "P training record model calls differ from EpisodeTrace",
    )

    actions = tuple(
        _sample_from_event(
            trace=trace,
            event=event,
            decision_ordinal=decision_ordinal,
        )
        for decision_ordinal, event in enumerate(
            event for event in trace.events if event.kind == "harness_decision"
        )
    )
    batch = AdmittedHarnessActionBatch(
        joint_version=trace.joint_version,
        episode_id=trace.episode_id,
        trace_sha256=str(trace_audit["trace_sha256"]),
        source_training_record_sha256=str(pre_batch_training_record["record_sha256"]),
        terminal_reward=float(trace.reward),
        validity_class=trace.validity_class,
        bound_model_call_ids=tuple(trace_audit["model_call_ids"]),
        actions=actions,
    )
    batch.validate()
    return batch


def validate_harness_action_admission_record(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
) -> AdmittedHarnessActionBatch:
    """Rehydrate a persisted R record and re-run every semantic check."""

    expected_fields = {
        "schema_version",
        "identity",
        "joint_version",
        "source",
        "terminal_outcome",
        "actions",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == expected_fields, "Harness admission field set differs")
    _require(
        record.get("schema_version") == SCHEMA_VERSION,
        "unknown Harness admission schema",
    )
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    _require(
        record.get("record_sha256") == _sha256(unsigned),
        "Harness admission record hash mismatch",
    )
    _assert_no_secret_fields(record)
    identity = record.get("identity")
    source = record.get("source")
    outcome = record.get("terminal_outcome")
    joint_version_raw = record.get("joint_version")
    actions_raw = record.get("actions")
    _require(
        isinstance(identity, Mapping)
        and set(identity) == {"episode_id", "joint_version_id"},
        "Harness admission identity differs from schema",
    )
    _require(
        isinstance(source, Mapping)
        and set(source)
        == {
            "trace_sha256",
            "pre_batch_training_record_sha256",
            "bound_model_call_ids",
        },
        "Harness admission source differs from schema",
    )
    _require(
        isinstance(outcome, Mapping) and set(outcome) == {"reward", "validity_class"},
        "Harness admission terminal outcome differs from schema",
    )
    _require(isinstance(joint_version_raw, Mapping), "JointVersion must be an object")
    _require(
        set(joint_version_raw) == set(JointVersion.__dataclass_fields__),
        "JointVersion field set differs from schema",
    )
    _require(isinstance(actions_raw, list), "Harness admission actions must be a list")
    try:
        joint_version = JointVersion(**dict(joint_version_raw))
        actions = tuple(
            AdmittedHarnessAction(
                episode_id=raw["episode_id"],
                joint_version_id=raw["joint_version_id"],
                event_id=raw["event_id"],
                event_index=raw["event_index"],
                decision_ordinal=raw["decision_ordinal"],
                decision_id=raw["decision_id"],
                state=_state_from_payload(raw["state"]),
                action=raw["action"],
                action_ids=tuple(raw["action_ids"]),
                action_mask=tuple(raw["action_mask"]),
                pre_mask_logits=tuple(raw["pre_mask_logits"]),
                old_harness_logprob=raw["old_harness_logprob"],
                harness_loss_mask=raw["harness_loss_mask"],
                harness_behavior_version=raw["harness_behavior_version"],
            )
            for raw in actions_raw
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessActionAdmissionError("invalid Harness admission record") from exc
    _require(
        all(
            isinstance(raw, Mapping)
            and set(raw) == set(AdmittedHarnessAction.__dataclass_fields__)
            for raw in actions_raw
        ),
        "Harness admission action field set differs from schema",
    )
    batch = AdmittedHarnessActionBatch(
        joint_version=joint_version,
        episode_id=identity.get("episode_id"),
        trace_sha256=source.get("trace_sha256"),
        source_training_record_sha256=source.get("pre_batch_training_record_sha256"),
        terminal_reward=outcome.get("reward"),
        validity_class=outcome.get("validity_class"),
        bound_model_call_ids=tuple(source.get("bound_model_call_ids", ())),
        actions=actions,
    )
    batch.validate()
    if active_joint_version is not None:
        _require(
            batch.joint_version == active_joint_version,
            "Harness admission JointVersion differs from lag-zero active version",
        )
    _require(
        identity.get("joint_version_id") == joint_version.version_id,
        "Harness admission identity JointVersion mismatch",
    )
    _require(
        _canonical_json(batch.to_record()) == _canonical_json(dict(record)),
        "Harness admission record is not canonical",
    )
    return batch
