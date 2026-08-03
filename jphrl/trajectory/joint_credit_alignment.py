from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass

from jphrl.harness.controller import HarnessState

from .areal_interaction_sidecar import REQUIRED_TENSOR_FIELDS
from .areal_policy_admission import (
    ValidatedPolicyTrainingAdmission,
    validate_policy_training_admission,
)
from .harness_action_admission import (
    AdmittedHarnessAction,
    AdmittedHarnessActionBatch,
    validate_harness_action_admission_record,
)
from .schema import JointVersion

SCHEMA_VERSION = "jph.frozen-joint-credit-alignment.v1"
ESTIMATOR_VERSION = "jph.terminal-return-minus-frozen-dual-baseline.v1"
_FORBIDDEN_PROVENANCE_MARKERS = (
    "fixture",
    "placeholder",
    "raw-reward-only",
    "synthetic",
)
_EVIDENCE_SCOPE = {
    "policy_samples_admitted": True,
    "harness_action_samples_admitted": True,
    "policy_advantages_aligned": True,
    "harness_advantages_aligned": True,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}


class JointCreditAlignmentError(ValueError):
    """Raised when real Q/R samples cannot share one frozen credit batch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointCreditAlignmentError(message)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


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
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JointCreditAlignmentError(
            "joint credit alignment is not finite canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _record_sha256(record: Mapping[str, object]) -> str:
    return _sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _assert_no_secret_fields(value: object, path: str = "record") -> None:
    secret_names = {
        "admin_api_key",
        "api_key",
        "authorization",
        "session_api_key",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                str(key).lower() not in secret_names,
                f"credential field cannot enter joint credit alignment: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


def _exact_mapping(
    value: object,
    expected_fields: set[str],
    label: str,
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == expected_fields, f"{label} field set differs from schema")
    return value


def _public_identifier(value: object, label: str) -> str:
    _require(_is_non_empty_string(value), f"{label} must be non-empty")
    identifier = str(value)
    _require(
        not identifier.lower().startswith(("sk-", "bearer ")),
        f"{label} looks like a credential",
    )
    return identifier


@dataclass(frozen=True)
class DualCreditEstimatorSpec:
    """Frozen decision-level baselines for two distinct credit streams."""

    estimator_version: str
    parent_joint_version_id: str
    policy_source: str
    harness_source: str
    policy_baseline_snapshot_id: str
    harness_baseline_snapshot_id: str
    policy_baselines: Mapping[str, float]
    harness_baselines: Mapping[str, float]

    def validate(
        self,
        *,
        joint_version: JointVersion,
        policy_model_call_ids: tuple[str, ...],
        harness_decision_ids: tuple[str, ...],
    ) -> None:
        _require(
            self.estimator_version == ESTIMATOR_VERSION,
            "unknown dual-credit estimator version",
        )
        _require(
            self.parent_joint_version_id == joint_version.version_id,
            "credit estimator parent differs from the admitted JointVersion",
        )
        policy_source = _public_identifier(self.policy_source, "Policy credit source")
        harness_source = _public_identifier(
            self.harness_source,
            "Harness credit source",
        )
        _require(
            policy_source != harness_source,
            "Policy and Harness credit sources must remain distinct",
        )
        for source in (policy_source, harness_source):
            normalized = source.lower()
            _require(
                not any(
                    marker in normalized for marker in _FORBIDDEN_PROVENANCE_MARKERS
                ),
                "synthetic or placeholder credit cannot enter a real frozen batch",
            )
        _public_identifier(
            self.policy_baseline_snapshot_id,
            "Policy baseline snapshot ID",
        )
        _public_identifier(
            self.harness_baseline_snapshot_id,
            "Harness baseline snapshot ID",
        )
        _require(
            self.policy_baseline_snapshot_id != self.harness_baseline_snapshot_id,
            "Policy and Harness baseline snapshots must remain distinct",
        )
        _require(
            set(self.policy_baselines) == set(policy_model_call_ids),
            "Policy baseline targets differ from admitted model calls",
        )
        _require(
            set(self.harness_baselines) == set(harness_decision_ids),
            "Harness baseline targets differ from admitted decisions",
        )
        _require(
            all(_is_finite_number(value) for value in self.policy_baselines.values()),
            "Policy baselines must be finite",
        )
        _require(
            all(_is_finite_number(value) for value in self.harness_baselines.values()),
            "Harness baselines must be finite",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "estimator_version": self.estimator_version,
            "parent_joint_version_id": self.parent_joint_version_id,
            "policy_source": self.policy_source,
            "harness_source": self.harness_source,
            "policy_baseline_snapshot_id": self.policy_baseline_snapshot_id,
            "harness_baseline_snapshot_id": self.harness_baseline_snapshot_id,
            "policy_baselines": {
                key: float(value) for key, value in self.policy_baselines.items()
            },
            "harness_baselines": {
                key: float(value) for key, value in self.harness_baselines.items()
            },
        }


def _estimator_from_record(raw: object) -> DualCreditEstimatorSpec:
    fields = set(DualCreditEstimatorSpec.__dataclass_fields__)
    value = _exact_mapping(raw, fields, "dual-credit estimator")
    policy_baselines = value.get("policy_baselines")
    harness_baselines = value.get("harness_baselines")
    _require(
        isinstance(policy_baselines, Mapping)
        and isinstance(harness_baselines, Mapping),
        "dual-credit baseline maps are missing",
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
        raise JointCreditAlignmentError("invalid dual-credit estimator") from exc


def _joint_version_from_record(raw: object) -> JointVersion:
    value = _exact_mapping(
        raw,
        set(JointVersion.__dataclass_fields__),
        "JointVersion",
    )
    try:
        joint_version = JointVersion(**dict(value))
    except TypeError as exc:
        raise JointCreditAlignmentError("invalid JointVersion") from exc
    _require(
        all(
            _is_non_empty_string(getattr(joint_version, field))
            for field in JointVersion.__dataclass_fields__
        ),
        "JointVersion fields must be non-empty",
    )
    return joint_version


def _harness_action_from_record(raw: object) -> AdmittedHarnessAction:
    value = _exact_mapping(
        raw,
        set(AdmittedHarnessAction.__dataclass_fields__),
        "aligned Harness action",
    )
    state_raw = value.get("state")
    _require(isinstance(state_raw, Mapping), "aligned Harness state is missing")
    _require(
        set(state_raw) == set(HarnessState.__dataclass_fields__),
        "aligned Harness state field set differs",
    )
    try:
        return AdmittedHarnessAction(
            episode_id=value["episode_id"],
            joint_version_id=value["joint_version_id"],
            event_id=value["event_id"],
            event_index=value["event_index"],
            decision_ordinal=value["decision_ordinal"],
            decision_id=value["decision_id"],
            state=HarnessState(**dict(state_raw)),
            action=value["action"],
            action_ids=tuple(value["action_ids"]),
            action_mask=tuple(value["action_mask"]),
            pre_mask_logits=tuple(value["pre_mask_logits"]),
            old_harness_logprob=value["old_harness_logprob"],
            harness_loss_mask=value["harness_loss_mask"],
            harness_behavior_version=value["harness_behavior_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JointCreditAlignmentError("invalid aligned Harness action") from exc


def _aligned_policy_samples(
    policy: ValidatedPolicyTrainingAdmission,
    harness: AdmittedHarnessActionBatch,
    estimator: DualCreditEstimatorSpec,
) -> list[dict[str, object]]:
    aligned: list[dict[str, object]] = []
    for sample in policy.samples:
        _require(
            math.isclose(
                sample.reward,
                harness.terminal_reward,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "AReaL sample reward differs from the Harness terminal outcome; "
            "discounted returns require a dedicated estimator",
        )
        advantages = [0.0] * len(sample.input_ids)
        credit_mask = [0] * len(sample.input_ids)
        decision_credits: list[dict[str, object]] = []
        for span in sample.decision_spans:
            baseline = float(estimator.policy_baselines[span.model_call_id])
            advantage = sample.reward - baseline
            for position in range(span.start, span.end):
                _require(
                    sample.loss_mask[position] == 1,
                    "Policy credit span reaches a masked token",
                )
                advantages[position] = advantage
                credit_mask[position] = 1
            decision_credits.append(
                {
                    "model_call_id": span.model_call_id,
                    "interaction_id": span.interaction_id,
                    "start": span.start,
                    "end": span.end,
                    "inference_engine_version": span.inference_engine_version,
                    "raw_return": sample.reward,
                    "frozen_baseline": baseline,
                    "advantage": advantage,
                    "source": estimator.policy_source,
                }
            )
        _require(
            tuple(credit_mask) == sample.loss_mask,
            "Policy credit mask differs from the admitted loss mask",
        )
        aligned.append(
            {
                "sample_id": sample.sample_id,
                "tensor_dict": sample.areal_tensor_dict(),
                "decision_credits": decision_credits,
                "advantage_tensor": [advantages],
                "credit_mask": [credit_mask],
            }
        )
    return aligned


def _aligned_harness_samples(
    harness: AdmittedHarnessActionBatch,
    estimator: DualCreditEstimatorSpec,
) -> list[dict[str, object]]:
    aligned: list[dict[str, object]] = []
    for action in harness.actions:
        baseline = float(estimator.harness_baselines[action.decision_id])
        advantage = harness.terminal_reward - baseline
        aligned.append(
            {
                "action": action.to_dict(),
                "raw_return": harness.terminal_reward,
                "frozen_baseline": baseline,
                "advantage": advantage,
                "masked_advantage": advantage * action.harness_loss_mask,
                "source": estimator.harness_source,
            }
        )
    return aligned


def build_frozen_joint_credit_alignment(
    *,
    policy_admission: Mapping[str, object],
    harness_admission: Mapping[str, object] | AdmittedHarnessActionBatch,
    active_joint_version: JointVersion,
    estimator: DualCreditEstimatorSpec,
) -> dict[str, object]:
    """Join Q and R only after both streams pass lag-zero admission."""

    try:
        policy = validate_policy_training_admission(
            policy_admission,
            active_joint_version=active_joint_version,
        )
        if isinstance(harness_admission, Mapping):
            harness_admission = validate_harness_action_admission_record(
                harness_admission,
                active_joint_version=active_joint_version,
            )
        else:
            _require(
                type(harness_admission) is AdmittedHarnessActionBatch,
                "S requires an admitted real Harness action batch or record",
            )
            harness_admission.validate()
    except ValueError as exc:
        raise JointCreditAlignmentError(str(exc)) from exc
    policy_admission_record = deepcopy(dict(policy_admission))
    harness_admission_record = harness_admission.to_record()
    _require(
        harness_admission.joint_version == active_joint_version,
        "Harness admission JointVersion differs from lag-zero active version",
    )
    _require(
        policy.episode_id == harness_admission.episode_id,
        "Policy and Harness admissions refer to different episodes",
    )
    _require(
        policy.source_training_record_sha256
        == harness_admission.source_training_record_sha256,
        "Policy and Harness admissions do not share one P training record",
    )
    _require(
        policy.model_call_ids == harness_admission.bound_model_call_ids,
        "Policy and Harness admissions disagree on model-call identity",
    )
    decision_ids = tuple(action.decision_id for action in harness_admission.actions)
    estimator.validate(
        joint_version=active_joint_version,
        policy_model_call_ids=policy.model_call_ids,
        harness_decision_ids=decision_ids,
    )
    policy_samples = _aligned_policy_samples(policy, harness_admission, estimator)
    harness_samples = _aligned_harness_samples(harness_admission, estimator)
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "episode_id": policy.episode_id,
            "joint_version_id": active_joint_version.version_id,
            "trace_sha256": harness_admission.trace_sha256,
            "source_training_record_sha256": policy.source_training_record_sha256,
            "model_call_ids": list(policy.model_call_ids),
            "harness_decision_ids": list(decision_ids),
            "terminal_reward": harness_admission.terminal_reward,
        },
        "joint_version": asdict(active_joint_version),
        "admissions": {
            "policy_admission_sha256": policy.digest,
            "harness_admission_sha256": harness_admission.digest,
            "policy_export_style": policy.export_style,
            "policy_admission_record": policy_admission_record,
            "harness_admission_record": harness_admission_record,
        },
        "estimator": estimator.to_record(),
        "policy_samples": policy_samples,
        "harness_samples": harness_samples,
        "summary": {
            "policy_sample_count": len(policy_samples),
            "policy_trainable_token_count": sum(
                sum(sample["credit_mask"][0]) for sample in policy_samples
            ),
            "policy_decision_span_count": sum(
                len(sample["decision_credits"]) for sample in policy_samples
            ),
            "harness_action_count": len(harness_samples),
            "harness_trainable_action_count": sum(
                action.harness_loss_mask for action in harness_admission.actions
            ),
        },
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    validate_frozen_joint_credit_alignment(
        record,
        active_joint_version=active_joint_version,
    )
    return record


def _single_row(tensors: Mapping[str, object], field: str) -> list[object]:
    raw = tensors.get(field)
    _require(isinstance(raw, list) and len(raw) == 1, f"{field} must have one row")
    row = raw[0]
    _require(isinstance(row, list), f"{field} row must be a list")
    return row


def validate_frozen_joint_credit_alignment(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
) -> dict[str, object]:
    """Revalidate a persisted S batch without executing either optimizer."""

    expected_fields = {
        "schema_version",
        "identity",
        "joint_version",
        "admissions",
        "estimator",
        "policy_samples",
        "harness_samples",
        "summary",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == expected_fields, "joint credit field set differs")
    _require(record.get("schema_version") == SCHEMA_VERSION, "unknown S schema")
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "joint credit alignment hash mismatch",
    )
    _assert_no_secret_fields(record)
    joint_version = _joint_version_from_record(record.get("joint_version"))
    if active_joint_version is not None:
        _require(
            joint_version == active_joint_version,
            "frozen credit JointVersion differs from lag-zero active version",
        )
    identity = _exact_mapping(
        record.get("identity"),
        {
            "episode_id",
            "joint_version_id",
            "trace_sha256",
            "source_training_record_sha256",
            "model_call_ids",
            "harness_decision_ids",
            "terminal_reward",
        },
        "joint credit identity",
    )
    episode_id = _public_identifier(identity.get("episode_id"), "episode ID")
    _require(
        identity.get("joint_version_id") == joint_version.version_id,
        "joint credit identity differs from JointVersion",
    )
    _require(_is_sha256(identity.get("trace_sha256")), "trace SHA-256 is invalid")
    _require(
        _is_sha256(identity.get("source_training_record_sha256")),
        "source training-record SHA-256 is invalid",
    )
    model_call_ids_raw = identity.get("model_call_ids")
    decision_ids_raw = identity.get("harness_decision_ids")
    _require(
        isinstance(model_call_ids_raw, list)
        and bool(model_call_ids_raw)
        and len(set(model_call_ids_raw)) == len(model_call_ids_raw)
        and all(_is_non_empty_string(value) for value in model_call_ids_raw),
        "joint credit model-call IDs are invalid",
    )
    _require(
        isinstance(decision_ids_raw, list)
        and bool(decision_ids_raw)
        and len(set(decision_ids_raw)) == len(decision_ids_raw)
        and all(_is_non_empty_string(value) for value in decision_ids_raw),
        "joint credit Harness decision IDs are invalid",
    )
    model_call_ids = tuple(str(value) for value in model_call_ids_raw)
    decision_ids = tuple(str(value) for value in decision_ids_raw)
    _require(
        set(model_call_ids).isdisjoint(decision_ids),
        "Policy and Harness credit targets overlap",
    )
    terminal_reward = identity.get("terminal_reward")
    _require(_is_finite_number(terminal_reward), "terminal reward must be finite")
    terminal_reward = float(terminal_reward)

    admissions = _exact_mapping(
        record.get("admissions"),
        {
            "policy_admission_sha256",
            "harness_admission_sha256",
            "policy_export_style",
            "policy_admission_record",
            "harness_admission_record",
        },
        "joint admissions",
    )
    _require(
        _is_sha256(admissions.get("policy_admission_sha256"))
        and _is_sha256(admissions.get("harness_admission_sha256")),
        "Q/R admission hashes are invalid",
    )
    _require(
        admissions.get("policy_export_style") in {"individual", "concat"},
        "joint Policy export style is invalid",
    )
    policy_admission_record = admissions.get("policy_admission_record")
    harness_admission_record = admissions.get("harness_admission_record")
    _require(
        isinstance(policy_admission_record, Mapping)
        and isinstance(harness_admission_record, Mapping),
        "persisted Q/R admission records are missing",
    )
    try:
        policy_admission = validate_policy_training_admission(
            policy_admission_record,
            active_joint_version=joint_version,
        )
        harness_admission = validate_harness_action_admission_record(
            harness_admission_record,
            active_joint_version=joint_version,
        )
    except ValueError as exc:
        raise JointCreditAlignmentError(str(exc)) from exc
    _require(
        admissions.get("policy_admission_sha256") == policy_admission.digest
        and admissions.get("harness_admission_sha256") == harness_admission.digest,
        "persisted Q/R admission hashes differ from their records",
    )
    _require(
        admissions.get("policy_export_style") == policy_admission.export_style,
        "persisted Policy export style differs from Q admission",
    )
    _require(
        policy_admission.episode_id == episode_id
        and harness_admission.episode_id == episode_id,
        "persisted Q/R admission episode differs from S identity",
    )
    _require(
        policy_admission.source_training_record_sha256
        == identity.get("source_training_record_sha256")
        == harness_admission.source_training_record_sha256,
        "persisted Q/R admissions do not share the S training record",
    )
    _require(
        policy_admission.model_call_ids
        == model_call_ids
        == harness_admission.bound_model_call_ids,
        "persisted Q/R model-call identities differ from S identity",
    )
    _require(
        harness_admission.trace_sha256 == identity.get("trace_sha256")
        and math.isclose(
            harness_admission.terminal_reward,
            terminal_reward,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "persisted Harness outcome differs from S identity",
    )
    _require(
        tuple(action.decision_id for action in harness_admission.actions)
        == decision_ids,
        "persisted Harness decisions differ from S identity",
    )
    estimator = _estimator_from_record(record.get("estimator"))
    estimator.validate(
        joint_version=joint_version,
        policy_model_call_ids=model_call_ids,
        harness_decision_ids=decision_ids,
    )

    policy_samples = record.get("policy_samples")
    _require(
        isinstance(policy_samples, list) and bool(policy_samples),
        "joint credit requires Policy samples",
    )
    _require(
        _canonical_json(policy_samples)
        == _canonical_json(
            _aligned_policy_samples(
                policy_admission,
                harness_admission,
                estimator,
            )
        ),
        "aligned Policy samples differ from the persisted Q admission",
    )
    represented_policy: set[str] = set()
    policy_call_counts = {model_call_id: 0 for model_call_id in model_call_ids}
    represented_interactions: set[str] = set()
    policy_token_count = 0
    policy_span_count = 0
    sample_ids: set[str] = set()
    for sample in policy_samples:
        sample = _exact_mapping(
            sample,
            {
                "sample_id",
                "tensor_dict",
                "decision_credits",
                "advantage_tensor",
                "credit_mask",
            },
            "aligned Policy sample",
        )
        sample_id = _public_identifier(sample.get("sample_id"), "Policy sample ID")
        _require(sample_id not in sample_ids, "duplicate aligned Policy sample")
        sample_ids.add(sample_id)
        tensors = _exact_mapping(
            sample.get("tensor_dict"),
            set(REQUIRED_TENSOR_FIELDS),
            "aligned Policy tensor_dict",
        )
        input_ids = _single_row(tensors, "input_ids")
        loss_mask = _single_row(tensors, "loss_mask")
        logprobs = _single_row(tensors, "logprobs")
        versions = _single_row(tensors, "versions")
        attention_mask = _single_row(tensors, "attention_mask")
        rewards = tensors.get("rewards")
        length = len(input_ids)
        _require(
            length > 0
            and all(
                len(row) == length
                for row in (loss_mask, logprobs, versions, attention_mask)
            ),
            "aligned Policy tensor lengths differ",
        )
        _require(
            isinstance(rewards, list)
            and len(rewards) == 1
            and _is_finite_number(rewards[0])
            and math.isclose(
                float(rewards[0]), terminal_reward, rel_tol=0.0, abs_tol=1e-9
            ),
            "aligned Policy reward differs from terminal outcome",
        )
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in input_ids
            ),
            "aligned Policy token IDs must be integers",
        )
        _require(
            all(value in (0, 1) and type(value) is int for value in loss_mask),
            "aligned Policy loss mask must be binary integers",
        )
        _require(
            all(_is_finite_number(value) for value in logprobs)
            and all(type(value) is int and value >= -1 for value in versions)
            and all(type(value) is bool for value in attention_mask),
            "aligned Policy behavior tensors are invalid",
        )
        _require(
            all(attention_mask),
            "unpadded admitted Policy samples must have active attention masks",
        )
        for position, trainable in enumerate(loss_mask):
            if trainable:
                _require(
                    float(logprobs[position]) <= 0.0 and versions[position] >= 0,
                    "trainable Policy behavior metadata is invalid",
                )
            else:
                _require(
                    float(logprobs[position]) == 0.0 and versions[position] == -1,
                    "masked Policy behavior metadata must use neutral sentinels",
                )
        advantages = _single_row(
            {"advantages": sample.get("advantage_tensor")},
            "advantages",
        )
        credit_mask = _single_row(
            {"mask": sample.get("credit_mask")},
            "mask",
        )
        _require(
            len(advantages) == length
            and len(credit_mask) == length
            and credit_mask == loss_mask,
            "Policy advantage or credit mask is misaligned",
        )
        _require(
            all(_is_finite_number(value) for value in advantages),
            "Policy advantages must be finite",
        )
        credits = sample.get("decision_credits")
        _require(isinstance(credits, list) and bool(credits), "Policy credits missing")
        expected_advantages = [0.0] * length
        expected_mask = [0] * length
        previous_end = -1
        for credit in credits:
            credit = _exact_mapping(
                credit,
                {
                    "model_call_id",
                    "interaction_id",
                    "start",
                    "end",
                    "inference_engine_version",
                    "raw_return",
                    "frozen_baseline",
                    "advantage",
                    "source",
                },
                "Policy decision credit",
            )
            call_id = _public_identifier(
                credit.get("model_call_id"),
                "Policy credit target",
            )
            _require(call_id in model_call_ids, "unknown Policy credit target")
            represented_policy.add(call_id)
            policy_call_counts[call_id] += 1
            interaction_id = _public_identifier(
                credit.get("interaction_id"),
                "interaction ID",
            )
            _require(
                interaction_id not in represented_interactions,
                "one interaction cannot back multiple Policy credit spans",
            )
            represented_interactions.add(interaction_id)
            start = credit.get("start")
            end = credit.get("end")
            _require(
                type(start) is int
                and type(end) is int
                and previous_end <= start < end <= length,
                "Policy credit span is invalid or overlaps",
            )
            version = credit.get("inference_engine_version")
            _require(
                type(version) is int
                and version >= 0
                and versions[start:end] == [version] * (end - start),
                "Policy credit inference version is misaligned",
            )
            raw_return = credit.get("raw_return")
            baseline = credit.get("frozen_baseline")
            advantage = credit.get("advantage")
            _require(
                _is_finite_number(raw_return)
                and _is_finite_number(baseline)
                and _is_finite_number(advantage),
                "Policy credit values must be finite",
            )
            _require(
                math.isclose(
                    float(raw_return), terminal_reward, rel_tol=0.0, abs_tol=1e-9
                )
                and math.isclose(
                    float(baseline),
                    float(estimator.policy_baselines[call_id]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(advantage),
                    terminal_reward - float(baseline),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and credit.get("source") == estimator.policy_source,
                "Policy advantage differs from its frozen estimator",
            )
            for position in range(start, end):
                expected_advantages[position] = float(advantage)
                expected_mask[position] = 1
            previous_end = end
            policy_span_count += 1
        _require(
            expected_mask == loss_mask
            and all(
                math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(advantages, expected_advantages)
            ),
            "Policy advantages do not follow decision spans and loss mask",
        )
        policy_token_count += sum(expected_mask)
    _require(
        represented_policy == set(model_call_ids),
        "aligned Policy credits do not cover every admitted model call",
    )
    _require(
        all(count == 1 for count in policy_call_counts.values()),
        "every admitted model call must have exactly one Policy credit span",
    )
    if admissions["policy_export_style"] == "individual":
        _require(
            len(policy_samples) == len(model_call_ids),
            "individual export requires one Policy sample per model call",
        )
    else:
        _require(
            len(policy_samples) == 1,
            "concat export requires exactly one Policy sample",
        )

    harness_samples = record.get("harness_samples")
    _require(
        isinstance(harness_samples, list) and bool(harness_samples),
        "joint credit requires Harness samples",
    )
    _require(
        _canonical_json(harness_samples)
        == _canonical_json(_aligned_harness_samples(harness_admission, estimator)),
        "aligned Harness samples differ from the persisted R admission",
    )
    represented_harness: list[str] = []
    harness_trainable = 0
    for sample in harness_samples:
        sample = _exact_mapping(
            sample,
            {
                "action",
                "raw_return",
                "frozen_baseline",
                "advantage",
                "masked_advantage",
                "source",
            },
            "aligned Harness sample",
        )
        action = _harness_action_from_record(sample.get("action"))
        try:
            action.validate(joint_version)
        except ValueError as exc:
            raise JointCreditAlignmentError(str(exc)) from exc
        _require(action.episode_id == episode_id, "Harness action episode mismatch")
        represented_harness.append(action.decision_id)
        raw_return = sample.get("raw_return")
        baseline = sample.get("frozen_baseline")
        advantage = sample.get("advantage")
        masked_advantage = sample.get("masked_advantage")
        _require(
            all(
                _is_finite_number(value)
                for value in (raw_return, baseline, advantage, masked_advantage)
            ),
            "Harness credit values must be finite",
        )
        expected_baseline = float(estimator.harness_baselines[action.decision_id])
        expected_advantage = terminal_reward - expected_baseline
        _require(
            math.isclose(float(raw_return), terminal_reward, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                float(baseline), expected_baseline, rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(
                float(advantage), expected_advantage, rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(
                float(masked_advantage),
                expected_advantage * action.harness_loss_mask,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and sample.get("source") == estimator.harness_source,
            "Harness advantage differs from its frozen estimator or loss mask",
        )
        harness_trainable += action.harness_loss_mask
    _require(
        tuple(represented_harness) == decision_ids,
        "aligned Harness credits differ from admitted decision order",
    )

    summary = _exact_mapping(
        record.get("summary"),
        {
            "policy_sample_count",
            "policy_trainable_token_count",
            "policy_decision_span_count",
            "harness_action_count",
            "harness_trainable_action_count",
        },
        "joint credit summary",
    )
    expected_summary = {
        "policy_sample_count": len(policy_samples),
        "policy_trainable_token_count": policy_token_count,
        "policy_decision_span_count": policy_span_count,
        "harness_action_count": len(harness_samples),
        "harness_trainable_action_count": harness_trainable,
    }
    _require(summary == expected_summary, "joint credit summary differs from samples")
    _require(
        record.get("evidence_scope") == _EVIDENCE_SCOPE,
        "joint credit evidence scope differs from contract",
    )
    return {
        "ok": True,
        "episode_id": episode_id,
        "joint_version_id": joint_version.version_id,
        "policy_model_call_ids": list(model_call_ids),
        "harness_decision_ids": list(decision_ids),
        **expected_summary,
        "record_sha256": record["record_sha256"],
    }
