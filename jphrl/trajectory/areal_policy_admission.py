from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .areal_agent_service_adapter import (
    SECRET_FIELD_NAMES,
    ArealAgentServiceAdapterError,
    validate_agent_service_training_record,
)
from .areal_interaction_sidecar import (
    REQUIRED_TENSOR_FIELDS,
    ArealInteractionAdapterError,
    validate_bound_training_sample_archive,
)
from .schema import JointVersion

POLICY_ADMISSION_SCHEMA_VERSION = "jph.areal-policy-training-admission.v1"

_SOURCE_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "trace",
        "ready_transition",
        "training_archive",
        "evidence_scope",
        "record_sha256",
    }
)
_SOURCE_IDENTITY_FIELDS = frozenset(
    {
        "episode_id",
        "task_id",
        "group_id",
        "session_id",
        "trajectory_id",
        "joint_version_id",
    }
)
_SOURCE_TRACE_FIELDS = frozenset(
    {"trace_sha256", "validity_class", "reward", "model_call_ids"}
)
_SOURCE_ARCHIVE_FIELDS = frozenset(
    {
        "schema_version",
        "interaction_sidecar",
        "source_sidecar_sha256",
        "export_style",
        "turn_discount",
        "sample_count",
        "samples",
        "evidence_scope",
        "record_sha256",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "leaf_interaction_id",
        "leaf_model_call_id",
        "included_interaction_ids",
        "included_model_call_ids",
        "decision_spans",
        "sequence_length",
        "tensor_dict",
    }
)
_SPAN_FIELDS = frozenset({"model_call_id", "interaction_id", "start", "end"})
_ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "joint_version",
        "model_call_ids",
        "source",
        "export_style",
        "samples",
        "summary",
        "evidence_scope",
        "record_sha256",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "agent_service_training_record_sha256",
        "training_archive_sha256",
        "interaction_sidecar_sha256",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "sample_count",
        "token_count",
        "trainable_token_count",
        "decision_span_count",
        "inference_engine_versions",
    }
)
_EVIDENCE_SCOPE = {
    "pre_batch_interaction_binding": True,
    "policy_samples_admitted": True,
    "policy_advantages_attached": False,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}


class ArealPolicyAdmissionError(ValueError):
    """Raised when a P training record cannot become a Policy sample batch."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealPolicyAdmissionError(message)


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
        raise ArealPolicyAdmissionError(
            "Policy admission is not finite canonical JSON"
        ) from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _assert_no_secret_fields(value: object, path: str = "admission") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                str(key).lower() not in SECRET_FIELD_NAMES,
                f"credential field cannot enter Policy admission: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


def _require_exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == expected, f"{label} field set differs from contract")
    return value


def _joint_version_from_dict(raw: object) -> JointVersion:
    value = _require_exact_fields(
        raw,
        frozenset(JointVersion.__dataclass_fields__),
        "Policy admission JointVersion",
    )
    _require(
        all(_is_non_empty_string(item) for item in value.values()),
        "Policy admission JointVersion fields must be non-empty strings",
    )
    try:
        return JointVersion(**dict(value))
    except TypeError as exc:
        raise ArealPolicyAdmissionError(
            "invalid Policy admission JointVersion"
        ) from exc


@dataclass(frozen=True)
class PolicyDecisionSpan:
    model_call_id: str
    interaction_id: str
    start: int
    end: int
    inference_engine_version: int


@dataclass(frozen=True)
class AdmittedPolicyTrainingSample:
    """One AReaL sequence plus exact identity for every trainable token span."""

    sample_id: str
    leaf_interaction_id: str
    leaf_model_call_id: str
    included_interaction_ids: tuple[str, ...]
    included_model_call_ids: tuple[str, ...]
    decision_spans: tuple[PolicyDecisionSpan, ...]
    input_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    versions: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    reward: float

    def areal_tensor_dict(self) -> dict[str, object]:
        """Return the six unpadded AReaL fields without inventing advantages."""

        return {
            "input_ids": [list(self.input_ids)],
            "loss_mask": [list(self.loss_mask)],
            "logprobs": [list(self.logprobs)],
            "versions": [list(self.versions)],
            "attention_mask": [list(self.attention_mask)],
            "rewards": [self.reward],
        }


@dataclass(frozen=True)
class ValidatedPolicyTrainingAdmission:
    joint_version: JointVersion
    episode_id: str
    model_call_ids: tuple[str, ...]
    export_style: str
    samples: tuple[AdmittedPolicyTrainingSample, ...]
    source_training_record_sha256: str
    source_archive_sha256: str
    record_sha256: str

    @property
    def digest(self) -> str:
        return self.record_sha256


def _single_row(tensors: Mapping[str, object], field: str) -> list[object]:
    raw = tensors.get(field)
    _require(isinstance(raw, list), f"Policy tensor {field} must be a list")
    _require(len(raw) == 1, f"Policy tensor {field} must have batch size one")
    row = raw[0]
    _require(isinstance(row, list), f"Policy tensor {field} row must be a list")
    return row


def _validate_and_materialize_samples(
    raw_samples: object,
    *,
    expected_model_call_ids: tuple[str, ...],
) -> tuple[
    tuple[AdmittedPolicyTrainingSample, ...],
    dict[str, object],
]:
    _require(isinstance(raw_samples, list) and bool(raw_samples), "no Policy samples")
    expected_call_set = set(expected_model_call_ids)
    represented_calls: set[str] = set()
    sample_ids: set[str] = set()
    call_signatures: dict[str, tuple[object, ...]] = {}
    admitted: list[AdmittedPolicyTrainingSample] = []
    token_count = 0
    trainable_token_count = 0
    decision_span_count = 0
    inference_versions: set[int] = set()

    for raw_sample in raw_samples:
        sample = _require_exact_fields(
            raw_sample, _SAMPLE_FIELDS, "Policy admission sample"
        )
        sample_id = sample.get("sample_id")
        leaf_interaction_id = sample.get("leaf_interaction_id")
        leaf_model_call_id = sample.get("leaf_model_call_id")
        _require(_is_non_empty_string(sample_id), "Policy sample ID is missing")
        _require(sample_id not in sample_ids, "duplicate Policy sample ID")
        sample_ids.add(str(sample_id))
        _require(
            _is_non_empty_string(leaf_interaction_id)
            and _is_non_empty_string(leaf_model_call_id),
            "Policy sample leaf identity is missing",
        )

        included_interactions = sample.get("included_interaction_ids")
        included_calls = sample.get("included_model_call_ids")
        _require(
            isinstance(included_interactions, list)
            and isinstance(included_calls, list)
            and bool(included_calls)
            and len(included_interactions) == len(included_calls),
            "Policy sample included identity lists differ",
        )
        _require(
            all(_is_non_empty_string(item) for item in included_interactions)
            and all(_is_non_empty_string(item) for item in included_calls),
            "Policy sample included identities must be non-empty strings",
        )
        _require(
            len(set(included_interactions)) == len(included_interactions)
            and len(set(included_calls)) == len(included_calls),
            "Policy sample included identities must be unique within a chain",
        )
        _require(
            leaf_interaction_id == included_interactions[-1]
            and leaf_model_call_id == included_calls[-1],
            "Policy sample leaf differs from its included decision chain",
        )
        _require(
            set(included_calls).issubset(expected_call_set),
            "Policy sample contains a model call outside the P record",
        )
        represented_calls.update(str(item) for item in included_calls)

        tensors = _require_exact_fields(
            sample.get("tensor_dict"),
            frozenset(REQUIRED_TENSOR_FIELDS),
            "Policy sample tensor_dict",
        )
        input_ids = _single_row(tensors, "input_ids")
        loss_mask = _single_row(tensors, "loss_mask")
        logprobs = _single_row(tensors, "logprobs")
        versions = _single_row(tensors, "versions")
        attention_mask = _single_row(tensors, "attention_mask")
        rewards = tensors.get("rewards")
        sequence_length = len(input_ids)
        _require(
            _is_int(sample.get("sequence_length"))
            and sample.get("sequence_length") == sequence_length
            and sequence_length > 0,
            "Policy sample sequence length differs from tensors",
        )
        _require(
            all(
                len(row) == sequence_length
                for row in (
                    loss_mask,
                    logprobs,
                    versions,
                    attention_mask,
                )
            ),
            "Policy sample tensor lengths differ",
        )
        _require(
            all(_is_int(token) and token >= 0 for token in input_ids),
            "Policy input token IDs must be non-negative integers",
        )
        _require(
            all(_is_int(mask) and mask in (0, 1) for mask in loss_mask),
            "Policy loss mask must contain integer zero or one",
        )
        _require(
            all(_is_finite_number(logprob) for logprob in logprobs),
            "Policy old logprobs must be finite",
        )
        _require(
            all(_is_int(version) and version >= -1 for version in versions),
            "Policy inference versions must be integers greater than or equal to -1",
        )
        _require(
            all(value is True or value == 1 for value in attention_mask),
            "Policy attention mask must activate every archived token",
        )
        _require(
            isinstance(rewards, list)
            and len(rewards) == 1
            and _is_finite_number(rewards[0]),
            "Policy sample must retain one finite AReaL reward",
        )

        raw_spans = sample.get("decision_spans")
        _require(
            isinstance(raw_spans, list) and len(raw_spans) == len(included_calls),
            "Policy decision spans differ from included model calls",
        )
        spans: list[PolicyDecisionSpan] = []
        expected_mask = [0] * sequence_length
        previous_end = -1
        for raw_span, model_call_id, interaction_id in zip(
            raw_spans, included_calls, included_interactions
        ):
            span = _require_exact_fields(raw_span, _SPAN_FIELDS, "Policy decision span")
            _require(
                span.get("model_call_id") == model_call_id
                and span.get("interaction_id") == interaction_id,
                "Policy decision span identity differs from its chain",
            )
            start = span.get("start")
            end = span.get("end")
            _require(
                _is_int(start)
                and _is_int(end)
                and previous_end <= start < end <= sequence_length,
                "Policy decision span is invalid or overlaps",
            )
            assert isinstance(start, int) and isinstance(end, int)
            span_versions = versions[start:end]
            _require(
                len(set(span_versions)) == 1 and span_versions[0] >= 0,
                "one Policy model call must have one inference engine version",
            )
            _require(
                loss_mask[start:end] == [1] * (end - start),
                "Policy decision span must be fully trainable",
            )
            _require(
                all(float(logprob) <= 0.0 for logprob in logprobs[start:end]),
                "Policy trainable old logprob cannot be positive",
            )
            expected_mask[start:end] = [1] * (end - start)
            version = int(span_versions[0])
            inference_versions.add(version)
            signature = (
                tuple(input_ids[start:end]),
                tuple(float(value) for value in logprobs[start:end]),
                tuple(int(value) for value in span_versions),
            )
            previous_signature = call_signatures.setdefault(
                str(model_call_id), signature
            )
            _require(
                previous_signature == signature,
                "a Policy model call has inconsistent duplicated token data",
            )
            spans.append(
                PolicyDecisionSpan(
                    model_call_id=str(model_call_id),
                    interaction_id=str(interaction_id),
                    start=start,
                    end=end,
                    inference_engine_version=version,
                )
            )
            previous_end = end

        _require(
            loss_mask == expected_mask,
            "Policy loss mask contains tokens outside admitted decision spans",
        )
        for index, mask in enumerate(loss_mask):
            if mask == 0:
                _require(
                    float(logprobs[index]) == 0.0 and versions[index] == -1,
                    "Policy prompt positions require logprob=0 and version=-1",
                )
        admitted.append(
            AdmittedPolicyTrainingSample(
                sample_id=str(sample_id),
                leaf_interaction_id=str(leaf_interaction_id),
                leaf_model_call_id=str(leaf_model_call_id),
                included_interaction_ids=tuple(
                    str(item) for item in included_interactions
                ),
                included_model_call_ids=tuple(str(item) for item in included_calls),
                decision_spans=tuple(spans),
                input_ids=tuple(int(item) for item in input_ids),
                loss_mask=tuple(int(item) for item in loss_mask),
                logprobs=tuple(float(item) for item in logprobs),
                versions=tuple(int(item) for item in versions),
                attention_mask=tuple(bool(item) for item in attention_mask),
                reward=float(rewards[0]),
            )
        )
        token_count += sequence_length
        trainable_token_count += sum(int(mask) for mask in loss_mask)
        decision_span_count += len(spans)

    _require(
        represented_calls == expected_call_set,
        "Policy samples do not cover every P model call",
    )
    summary = {
        "sample_count": len(admitted),
        "token_count": token_count,
        "trainable_token_count": trainable_token_count,
        "decision_span_count": decision_span_count,
        "inference_engine_versions": sorted(inference_versions),
    }
    return tuple(admitted), summary


def validate_policy_training_admission(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
) -> ValidatedPolicyTrainingAdmission:
    """Validate a stable Q admission record without claiming an optimizer step."""

    _require_exact_fields(record, _ADMISSION_FIELDS, "Policy admission")
    _require(
        record.get("schema_version") == POLICY_ADMISSION_SCHEMA_VERSION,
        "unknown Policy admission schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "Policy admission hash mismatch",
    )
    _assert_no_secret_fields(record)
    joint_version = _joint_version_from_dict(record.get("joint_version"))
    if active_joint_version is not None:
        _require(
            joint_version == active_joint_version,
            "Policy admission JointVersion differs from lag-zero active version",
        )

    identity = _require_exact_fields(
        record.get("identity"), _SOURCE_IDENTITY_FIELDS, "Policy admission identity"
    )
    _require(
        all(
            _is_non_empty_string(identity.get(field))
            for field in (
                "episode_id",
                "task_id",
                "group_id",
                "session_id",
            )
        ),
        "Policy admission route identity is incomplete",
    )
    _require(
        _is_int(identity.get("trajectory_id")) and int(identity["trajectory_id"]) >= 0,
        "Policy admission trajectory ID is invalid",
    )
    _require(
        identity.get("joint_version_id") == joint_version.version_id,
        "Policy admission identity differs from its JointVersion",
    )
    raw_model_calls = record.get("model_call_ids")
    _require(
        isinstance(raw_model_calls, list)
        and bool(raw_model_calls)
        and all(_is_non_empty_string(item) for item in raw_model_calls)
        and len(set(raw_model_calls)) == len(raw_model_calls),
        "Policy admission model call IDs must be ordered and unique",
    )
    model_call_ids = tuple(str(item) for item in raw_model_calls)

    source = _require_exact_fields(
        record.get("source"), _SOURCE_FIELDS, "Policy admission source"
    )
    _require(
        all(_is_sha256(value) for value in source.values()),
        "Policy admission source hashes are invalid",
    )
    export_style = record.get("export_style")
    _require(
        export_style in {"individual", "concat"},
        "unknown Policy admission export style",
    )
    samples, summary = _validate_and_materialize_samples(
        record.get("samples"), expected_model_call_ids=model_call_ids
    )
    raw_summary = _require_exact_fields(
        record.get("summary"), _SUMMARY_FIELDS, "Policy admission summary"
    )
    _require(raw_summary == summary, "Policy admission summary differs from samples")
    _require(
        record.get("evidence_scope") == _EVIDENCE_SCOPE,
        "Policy admission evidence scope differs from contract",
    )
    return ValidatedPolicyTrainingAdmission(
        joint_version=joint_version,
        episode_id=str(identity["episode_id"]),
        model_call_ids=model_call_ids,
        export_style=str(export_style),
        samples=samples,
        source_training_record_sha256=str(
            source["agent_service_training_record_sha256"]
        ),
        source_archive_sha256=str(source["training_archive_sha256"]),
        record_sha256=str(record["record_sha256"]),
    )


def build_policy_training_admission(
    training_record: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
) -> dict[str, object]:
    """Admit P's verified pre-batch archive as real AReaL Policy samples.

    Q intentionally stops before advantage construction and before any optimizer
    call. The returned samples retain the six AReaL tensor fields verbatim.
    """

    _require_exact_fields(
        training_record, _SOURCE_RECORD_FIELDS, "P Agent Service training record"
    )
    identity = _require_exact_fields(
        training_record.get("identity"),
        _SOURCE_IDENTITY_FIELDS,
        "P training record identity",
    )
    trace = _require_exact_fields(
        training_record.get("trace"), _SOURCE_TRACE_FIELDS, "P trace summary"
    )
    archive = _require_exact_fields(
        training_record.get("training_archive"),
        _SOURCE_ARCHIVE_FIELDS,
        "P training archive",
    )
    try:
        validate_agent_service_training_record(training_record)
        validate_bound_training_sample_archive(archive)
    except (ArealAgentServiceAdapterError, ArealInteractionAdapterError) as exc:
        raise ArealPolicyAdmissionError(str(exc)) from exc
    _require(
        isinstance(active_joint_version, JointVersion),
        "active JointVersion has the wrong type",
    )
    _require(
        identity.get("joint_version_id") == active_joint_version.version_id,
        "P training record JointVersion differs from lag-zero active version",
    )
    _require(
        all(
            _is_non_empty_string(getattr(active_joint_version, field))
            for field in JointVersion.__dataclass_fields__
        ),
        "active JointVersion fields must be non-empty strings",
    )
    raw_calls = trace.get("model_call_ids")
    _require(
        isinstance(raw_calls, list) and bool(raw_calls),
        "P trace summary has no Policy model calls",
    )
    model_call_ids = tuple(str(item) for item in raw_calls)

    # Materialize once before hashing so Q's stricter token/version checks apply
    # even when every nested P hash is internally consistent.
    _, summary = _validate_and_materialize_samples(
        archive.get("samples"), expected_model_call_ids=model_call_ids
    )
    samples = json.loads(_canonical_json(archive["samples"]).decode("utf-8"))
    record: dict[str, object] = {
        "schema_version": POLICY_ADMISSION_SCHEMA_VERSION,
        "identity": dict(identity),
        "joint_version": asdict(active_joint_version),
        "model_call_ids": list(model_call_ids),
        "source": {
            "agent_service_training_record_sha256": training_record["record_sha256"],
            "training_archive_sha256": archive["record_sha256"],
            "interaction_sidecar_sha256": archive["source_sidecar_sha256"],
        },
        "export_style": archive["export_style"],
        "samples": samples,
        "summary": summary,
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    validate_policy_training_admission(
        record, active_joint_version=active_joint_version
    )
    return record


def areal_policy_tensor_batch(
    admission: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
) -> tuple[dict[str, object], ...]:
    """Return fresh six-field AReaL mappings from one validated admission."""

    validated = validate_policy_training_admission(
        admission, active_joint_version=active_joint_version
    )
    return tuple(sample.areal_tensor_dict() for sample in validated.samples)
