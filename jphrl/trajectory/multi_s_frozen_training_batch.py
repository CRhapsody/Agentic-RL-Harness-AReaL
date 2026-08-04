from __future__ import annotations

"""Canonical, fail-closed batches of independently persisted S records.

The batch is an ordering and provenance envelope.  It does not merge tensors and
it never reconstructs an identity from a padded/post-batch representation.  T
and U consume ``members[i].s_record`` one member at a time and carry that
member's claim SHA-256 into their receipts.  V must claim the complete ordered
set returned by :func:`required_v_member_claims`.
"""

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from jphrl.paths import require_outside_repository

from .joint_credit_alignment import validate_frozen_joint_credit_alignment
from .schema import JointVersion


MULTI_S_FROZEN_TRAINING_BATCH_SCHEMA = "jph.multi-s-frozen-training-batch.v1"
MULTI_S_SOURCE_BINDING_SCHEMA = "jph.multi-s-source-binding.v1"
MINIMUM_MEMBER_COUNT = 4
MEMBER_ORDERING = (
    "episode-id,source-training-record-sha256,s-record-sha256,source-path-v1"
)
_MAX_RECORD_BYTES = 128 * 1024 * 1024
_CONSUMER_CONTRACT = {
    "t_policy": {
        "input": "members[].s_record.policy_samples",
        "iteration": "member_index_then_policy_sample_order",
        "receipt_claim": "members[].member_claim_sha256",
        "merged_batch_identity_allowed": False,
    },
    "u_harness": {
        "input": "members[].s_record.harness_samples",
        "iteration": "member_index_then_harness_sample_order",
        "receipt_claim": "members[].member_claim_sha256",
        "merged_batch_identity_allowed": False,
    },
    "v_joint_barrier": {
        "claim_unit": "members[].s_record_sha256",
        "member_envelope_binding": "members[].member_claim_sha256",
        "requires_every_member_exactly_once": True,
        "requires_no_prior_batch_claim": True,
        "requires_ordered_aggregate_sha256": True,
    },
}
_EVIDENCE_SCOPE = {
    "members_individually_validated": True,
    "lag_zero_joint_version_validated": True,
    "individual_export_validated": True,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
    "joint_optimizer_barrier": False,
}


class MultiSFrozenTrainingBatchError(ValueError):
    """Raised when persisted S records cannot form one training transaction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiSFrozenTrainingBatchError(message)


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
        raise MultiSFrozenTrainingBatchError(
            "multi-S batch is not finite canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


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


def _non_empty_string(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    return value


def _exact_mapping(
    value: object,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == fields, f"{label} field set differs from schema")
    return value


def _assert_no_secrets(value: object, path: str = "record") -> None:
    exact_names = {
        "access_token",
        "admin_api_key",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "github_token",
        "password",
        "secret",
        "session_api_key",
        "token",
    }
    forbidden_suffixes = (
        "_access_token",
        "_api_key",
        "_authorization",
        "_cookie",
        "_credential",
        "_password",
        "_secret",
        "_token",
    )
    credential_prefixes = ("bearer ", "github_pat_", "ghp_", "sk-")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            _require(
                normalized not in exact_names
                and not normalized.endswith(forbidden_suffixes),
                f"credential field cannot enter multi-S batch: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str):
        normalized_value = value.strip().lower()
        _require(
            not normalized_value.startswith(credential_prefixes),
            f"credential-looking value cannot enter multi-S batch: {path}",
        )


def _strict_json(raw: bytes, label: str) -> Mapping[str, object]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MultiSFrozenTrainingBatchError(f"{label} is not strict JSON") from exc
    _require(isinstance(value, Mapping), f"{label} must contain one JSON object")
    _canonical_json(value)
    return value


def _external_path(path: str | Path, label: str) -> Path:
    unresolved = Path(path).expanduser()
    _require(not unresolved.is_symlink(), f"{label} cannot be a symlink")
    try:
        return require_outside_repository(unresolved)
    except (RuntimeError, ValueError) as exc:
        raise MultiSFrozenTrainingBatchError(str(exc)) from exc


def _read_external_json(path: str | Path, label: str) -> tuple[Path, bytes, Mapping[str, object]]:
    resolved = _external_path(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise MultiSFrozenTrainingBatchError(f"cannot open {label}: {resolved}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            _require(size <= _MAX_RECORD_BYTES, f"{label} exceeds size limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    _require(bool(raw), f"{label} is empty")
    return resolved, raw, _strict_json(raw, label)


def _joint_version_from_record(value: object) -> JointVersion:
    raw = _exact_mapping(
        value,
        set(JointVersion.__dataclass_fields__),
        "multi-S JointVersion",
    )
    try:
        joint_version = JointVersion(**dict(raw))
    except TypeError as exc:
        raise MultiSFrozenTrainingBatchError("invalid multi-S JointVersion") from exc
    _require(
        all(
            isinstance(getattr(joint_version, field), str)
            and bool(getattr(joint_version, field))
            for field in JointVersion.__dataclass_fields__
        ),
        "multi-S JointVersion fields must be non-empty",
    )
    return joint_version


def _identity_from_s_record(record: Mapping[str, object]) -> dict[str, object]:
    identity = record["identity"]
    _require(isinstance(identity, Mapping), "S identity is missing")
    policy_samples = record["policy_samples"]
    _require(isinstance(policy_samples, list), "S Policy samples are missing")
    sample_ids: list[str] = []
    for sample in policy_samples:
        _require(isinstance(sample, Mapping), "S Policy sample is invalid")
        sample_ids.append(_non_empty_string(sample.get("sample_id"), "sample ID"))
    return {
        "episode_id": identity["episode_id"],
        "joint_version_id": identity["joint_version_id"],
        "trace_sha256": identity["trace_sha256"],
        "source_training_record_sha256": identity[
            "source_training_record_sha256"
        ],
        "model_call_ids": deepcopy(identity["model_call_ids"]),
        "harness_decision_ids": deepcopy(identity["harness_decision_ids"]),
        "policy_sample_ids": sample_ids,
    }


def _member_claim_sha256(member: Mapping[str, object]) -> str:
    return _sha256(
        {
            key: value
            for key, value in member.items()
            if key != "member_claim_sha256"
        }
    )


def _aggregate_sha256(
    joint_version: JointVersion,
    members: Sequence[Mapping[str, object]],
) -> str:
    return _sha256(
        {
            "joint_version": asdict(joint_version),
            "ordering": MEMBER_ORDERING,
            "member_claim_sha256s": [
                member["member_claim_sha256"] for member in members
            ],
        }
    )


def _member_sort_key(member: Mapping[str, object]) -> tuple[str, str, str, str]:
    identity = member["identity"]
    _require(isinstance(identity, Mapping), "multi-S member identity is missing")
    return (
        str(identity["episode_id"]),
        str(identity["source_training_record_sha256"]),
        str(member["s_record_sha256"]),
        str(member["source_path"]),
    )


@dataclass(frozen=True)
class MultiSFrozenTrainingMember:
    member_index: int
    member_claim_sha256: str
    source_path: Path
    source_file_sha256: str
    s_record_sha256: str
    episode_id: str
    policy_sample_ids: tuple[str, ...]
    harness_decision_ids: tuple[str, ...]
    s_record: Mapping[str, object]


@dataclass(frozen=True)
class ValidatedMultiSFrozenTrainingBatch:
    joint_version: JointVersion
    aggregate_sha256: str
    record_sha256: str
    members: tuple[MultiSFrozenTrainingMember, ...]
    policy_sample_count: int
    harness_action_count: int


def multi_s_source_binding(
    batch: ValidatedMultiSFrozenTrainingBatch,
) -> dict[str, object]:
    """Return the one canonical source identity that T, U, and V must share.

    The batch record digest binds the persisted path/file envelope, while the
    aggregate and ordered stable S digests prevent either optimizer from
    silently training on a different member set.  V claims the stable digests
    individually; the binding digest is not a substitute for those claims.
    """

    _require(
        type(batch) is ValidatedMultiSFrozenTrainingBatch,
        "source binding requires a validated multi-S batch",
    )
    record: dict[str, object] = {
        "schema_version": MULTI_S_SOURCE_BINDING_SCHEMA,
        "joint_version_id": batch.joint_version.version_id,
        "batch_record_sha256": batch.record_sha256,
        "batch_aggregate_sha256": batch.aggregate_sha256,
        "member_claim_sha256s": [
            member.member_claim_sha256 for member in batch.members
        ],
        "s_record_sha256s": [member.s_record_sha256 for member in batch.members],
        "policy_sample_count": batch.policy_sample_count,
        "harness_action_count": batch.harness_action_count,
    }
    record["record_sha256"] = _record_sha256(record)
    return validate_multi_s_source_binding(record, batch=batch)


def validate_multi_s_source_binding(
    record: Mapping[str, object],
    *,
    batch: ValidatedMultiSFrozenTrainingBatch | None = None,
) -> dict[str, object]:
    """Validate a persisted/embedded T-U-V source binding exactly."""

    fields = {
        "schema_version",
        "joint_version_id",
        "batch_record_sha256",
        "batch_aggregate_sha256",
        "member_claim_sha256s",
        "s_record_sha256s",
        "policy_sample_count",
        "harness_action_count",
        "record_sha256",
    }
    binding = _exact_mapping(record, fields, "multi-S source binding")
    _require(
        binding.get("schema_version") == MULTI_S_SOURCE_BINDING_SCHEMA
        and binding.get("record_sha256") == _record_sha256(binding),
        "multi-S source binding schema or hash differs",
    )
    joint_version_id = binding.get("joint_version_id")
    _require(
        isinstance(joint_version_id, str)
        and len(joint_version_id) == 16
        and all(character in "0123456789abcdef" for character in joint_version_id),
        "multi-S source binding joint_version_id is invalid",
    )
    for field in (
        "batch_record_sha256",
        "batch_aggregate_sha256",
        "record_sha256",
    ):
        _require(_is_sha256(binding.get(field)), f"multi-S source binding {field} is invalid")
    member_claims = binding.get("member_claim_sha256s")
    stable_claims = binding.get("s_record_sha256s")
    _require(
        isinstance(member_claims, list)
        and isinstance(stable_claims, list)
        and len(member_claims) >= MINIMUM_MEMBER_COUNT
        and len(member_claims) == len(stable_claims)
        and len(set(member_claims)) == len(member_claims)
        and len(set(stable_claims)) == len(stable_claims)
        and all(_is_sha256(value) for value in member_claims)
        and all(_is_sha256(value) for value in stable_claims),
        "multi-S source binding member claims are invalid",
    )
    _require(
        type(binding.get("policy_sample_count")) is int
        and binding["policy_sample_count"] >= MINIMUM_MEMBER_COUNT
        and type(binding.get("harness_action_count")) is int
        and binding["harness_action_count"] >= MINIMUM_MEMBER_COUNT,
        "multi-S source binding sample counts are invalid",
    )
    if batch is not None:
        _require(
            type(batch) is ValidatedMultiSFrozenTrainingBatch,
            "source binding comparison requires a validated multi-S batch",
        )
        expected = {
            "schema_version": MULTI_S_SOURCE_BINDING_SCHEMA,
            "joint_version_id": batch.joint_version.version_id,
            "batch_record_sha256": batch.record_sha256,
            "batch_aggregate_sha256": batch.aggregate_sha256,
            "member_claim_sha256s": [
                member.member_claim_sha256 for member in batch.members
            ],
            "s_record_sha256s": [
                member.s_record_sha256 for member in batch.members
            ],
            "policy_sample_count": batch.policy_sample_count,
            "harness_action_count": batch.harness_action_count,
        }
        expected["record_sha256"] = _record_sha256(expected)
        _require(dict(binding) == expected, "multi-S source binding differs from batch")
    return deepcopy(dict(binding))


def prepare_multi_s_frozen_training_batch(
    s_record_paths: Sequence[str | Path],
    *,
    active_joint_version: JointVersion,
) -> dict[str, object]:
    """Load at least four persisted individual S records into one envelope."""

    _require(
        not isinstance(s_record_paths, (str, bytes, Path)),
        "multi-S input must be a sequence of persisted S paths",
    )
    paths = tuple(s_record_paths)
    _require(
        len(paths) >= MINIMUM_MEMBER_COUNT,
        f"multi-S batch requires at least {MINIMUM_MEMBER_COUNT} persisted S records",
    )
    members: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for source_path in paths:
        _require(
            isinstance(source_path, (str, Path)),
            "multi-S input accepts persisted S paths only, not post-batch objects",
        )
        resolved, raw, s_record = _read_external_json(source_path, "S record")
        _require(resolved not in seen_paths, "same persisted S path appears twice")
        seen_paths.add(resolved)
        _assert_no_secrets(s_record, f"S record {resolved}")
        try:
            validate_frozen_joint_credit_alignment(
                s_record,
                active_joint_version=active_joint_version,
            )
        except ValueError as exc:
            raise MultiSFrozenTrainingBatchError(
                f"persisted S record failed frozen-credit validation: {resolved}: {exc}"
            ) from exc
        admissions = s_record.get("admissions")
        _require(isinstance(admissions, Mapping), "S admissions are missing")
        _require(
            admissions.get("policy_export_style") == "individual",
            "multi-S batch rejects concat/post-batch Policy exports",
        )
        member: dict[str, object] = {
            "member_index": -1,
            "source_path": str(resolved),
            "source_file_sha256": _sha256_bytes(raw),
            "s_record_sha256": s_record["record_sha256"],
            "identity": _identity_from_s_record(s_record),
            "s_record": deepcopy(dict(s_record)),
        }
        members.append(member)

    members.sort(key=_member_sort_key)
    for index, member in enumerate(members):
        member["member_index"] = index
        member["member_claim_sha256"] = _member_claim_sha256(member)

    policy_sample_count = sum(
        len(member["identity"]["policy_sample_ids"])  # type: ignore[index]
        for member in members
    )
    harness_action_count = sum(
        len(member["identity"]["harness_decision_ids"])  # type: ignore[index]
        for member in members
    )
    record: dict[str, object] = {
        "schema_version": MULTI_S_FROZEN_TRAINING_BATCH_SCHEMA,
        "joint_version": asdict(active_joint_version),
        "batch": {
            "export_style": "individual",
            "minimum_member_count": MINIMUM_MEMBER_COUNT,
            "member_count": len(members),
            "policy_sample_count": policy_sample_count,
            "harness_action_count": harness_action_count,
            "ordering": MEMBER_ORDERING,
        },
        "members": members,
        "aggregate_sha256": _aggregate_sha256(active_joint_version, members),
        "consumer_contract": deepcopy(_CONSUMER_CONTRACT),
        "evidence_scope": deepcopy(_EVIDENCE_SCOPE),
    }
    _assert_no_secrets(record)
    record["record_sha256"] = _record_sha256(record)
    validate_multi_s_frozen_training_batch(
        record,
        active_joint_version=active_joint_version,
        verify_source_files=True,
    )
    return record


def validate_multi_s_frozen_training_batch(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
    verify_source_files: bool = True,
) -> ValidatedMultiSFrozenTrainingBatch:
    """Validate the envelope and every embedded S record without batching tensors."""

    fields = {
        "schema_version",
        "joint_version",
        "batch",
        "members",
        "aggregate_sha256",
        "consumer_contract",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == fields, "multi-S batch field set differs")
    _require(
        record.get("schema_version") == MULTI_S_FROZEN_TRAINING_BATCH_SCHEMA,
        "unknown multi-S batch schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "multi-S batch record hash mismatch",
    )
    _assert_no_secrets(record)
    joint_version = _joint_version_from_record(record.get("joint_version"))
    _require(
        joint_version == active_joint_version,
        "multi-S batch JointVersion differs from lag-zero active version",
    )
    _require(
        record.get("consumer_contract") == _CONSUMER_CONTRACT,
        "multi-S consumer contract differs",
    )
    _require(
        record.get("evidence_scope") == _EVIDENCE_SCOPE,
        "multi-S evidence scope differs",
    )
    batch = _exact_mapping(
        record.get("batch"),
        {
            "export_style",
            "minimum_member_count",
            "member_count",
            "policy_sample_count",
            "harness_action_count",
            "ordering",
        },
        "multi-S batch summary",
    )
    _require(
        batch.get("export_style") == "individual"
        and batch.get("minimum_member_count") == MINIMUM_MEMBER_COUNT
        and batch.get("ordering") == MEMBER_ORDERING,
        "multi-S batch summary contract differs",
    )
    raw_members = record.get("members")
    _require(
        isinstance(raw_members, list) and len(raw_members) >= MINIMUM_MEMBER_COUNT,
        f"multi-S batch requires at least {MINIMUM_MEMBER_COUNT} members",
    )
    _require(
        batch.get("member_count") == len(raw_members),
        "multi-S member count differs from members",
    )

    member_fields = {
        "member_index",
        "source_path",
        "source_file_sha256",
        "s_record_sha256",
        "member_claim_sha256",
        "identity",
        "s_record",
    }
    identity_fields = {
        "episode_id",
        "joint_version_id",
        "trace_sha256",
        "source_training_record_sha256",
        "model_call_ids",
        "harness_decision_ids",
        "policy_sample_ids",
    }
    parsed_members: list[MultiSFrozenTrainingMember] = []
    canonical_members: list[Mapping[str, object]] = []
    episode_ids: set[str] = set()
    source_training_sha256s: set[str] = set()
    s_record_sha256s: set[str] = set()
    member_claim_sha256s: set[str] = set()
    source_paths: set[Path] = set()
    model_call_ids: set[str] = set()
    policy_sample_ids: set[str] = set()
    harness_decision_ids: set[str] = set()
    policy_sample_count = 0
    harness_action_count = 0
    for expected_index, raw_member in enumerate(raw_members):
        member = _exact_mapping(raw_member, member_fields, "multi-S member")
        _require(
            member.get("member_index") == expected_index,
            "multi-S member indices must be contiguous and ordered",
        )
        source_path = _external_path(
            _non_empty_string(member.get("source_path"), "S source path"),
            "S source path",
        )
        _require(
            str(source_path) == member.get("source_path"),
            "S source path must be canonical and absolute",
        )
        _require(source_path not in source_paths, "same S source path appears twice")
        source_paths.add(source_path)
        _require(
            _is_sha256(member.get("source_file_sha256"))
            and _is_sha256(member.get("s_record_sha256"))
            and _is_sha256(member.get("member_claim_sha256")),
            "multi-S member hashes are invalid",
        )
        s_record = member.get("s_record")
        _require(isinstance(s_record, Mapping), "embedded S record is missing")
        _assert_no_secrets(s_record, f"members[{expected_index}].s_record")
        try:
            validate_frozen_joint_credit_alignment(
                s_record,
                active_joint_version=active_joint_version,
            )
        except ValueError as exc:
            raise MultiSFrozenTrainingBatchError(
                f"embedded S member {expected_index} failed validation: {exc}"
            ) from exc
        admissions = s_record.get("admissions")
        _require(isinstance(admissions, Mapping), "embedded S admissions are missing")
        _require(
            admissions.get("policy_export_style") == "individual",
            "multi-S batch rejects concat/post-batch Policy exports",
        )
        _require(
            member.get("s_record_sha256") == s_record.get("record_sha256"),
            "member S digest differs from embedded S record",
        )
        identity = _exact_mapping(
            member.get("identity"), identity_fields, "multi-S member identity"
        )
        expected_identity = _identity_from_s_record(s_record)
        _require(identity == expected_identity, "member identity differs from embedded S")
        _require(
            member.get("member_claim_sha256") == _member_claim_sha256(member),
            "multi-S member claim hash mismatch",
        )
        _require(
            _joint_version_from_record(s_record.get("joint_version")) == joint_version,
            "multi-S members do not share the complete JointVersion",
        )

        episode_id = _non_empty_string(identity.get("episode_id"), "episode ID")
        source_training_sha256 = identity.get("source_training_record_sha256")
        s_record_sha256 = member.get("s_record_sha256")
        member_claim_sha256 = member.get("member_claim_sha256")
        _require(
            _is_sha256(source_training_sha256),
            "source training-record SHA-256 is invalid",
        )
        _require(episode_id not in episode_ids, "duplicate episode in multi-S batch")
        _require(
            source_training_sha256 not in source_training_sha256s,
            "same persisted S member/source training record appears twice",
        )
        _require(
            s_record_sha256 not in s_record_sha256s,
            "same persisted S member digest appears twice",
        )
        _require(
            member_claim_sha256 not in member_claim_sha256s,
            "same multi-S member claim appears twice",
        )
        episode_ids.add(episode_id)
        source_training_sha256s.add(str(source_training_sha256))
        s_record_sha256s.add(str(s_record_sha256))
        member_claim_sha256s.add(str(member_claim_sha256))

        raw_sample_ids = identity.get("policy_sample_ids")
        raw_model_call_ids = identity.get("model_call_ids")
        raw_decision_ids = identity.get("harness_decision_ids")
        _require(
            isinstance(raw_model_call_ids, list) and bool(raw_model_call_ids),
            "multi-S member requires model-call IDs",
        )
        _require(
            isinstance(raw_sample_ids, list) and bool(raw_sample_ids),
            "multi-S member requires Policy sample IDs",
        )
        _require(
            isinstance(raw_decision_ids, list) and bool(raw_decision_ids),
            "multi-S member requires Harness decision IDs",
        )
        for model_call_id in raw_model_call_ids:
            value = _non_empty_string(model_call_id, "model-call ID")
            _require(
                value not in model_call_ids,
                "duplicate model-call ID across multi-S members",
            )
            model_call_ids.add(value)
        for sample_id in raw_sample_ids:
            value = _non_empty_string(sample_id, "Policy sample ID")
            _require(
                value not in policy_sample_ids,
                "duplicate Policy sample ID across multi-S members",
            )
            policy_sample_ids.add(value)
        for decision_id in raw_decision_ids:
            value = _non_empty_string(decision_id, "Harness decision ID")
            _require(
                value not in harness_decision_ids,
                "duplicate Harness decision ID across multi-S members",
            )
            harness_decision_ids.add(value)

        if verify_source_files:
            persisted_path, persisted_raw, persisted_s = _read_external_json(
                source_path,
                "persisted S source",
            )
            _require(persisted_path == source_path, "S source path changed")
            _require(
                _sha256_bytes(persisted_raw) == member.get("source_file_sha256"),
                "persisted S source file hash mismatch",
            )
            _require(
                _canonical_json(persisted_s) == _canonical_json(s_record),
                "persisted S source differs from embedded S record",
            )

        policy_sample_count += len(raw_sample_ids)
        harness_action_count += len(raw_decision_ids)
        parsed_members.append(
            MultiSFrozenTrainingMember(
                member_index=expected_index,
                member_claim_sha256=str(member_claim_sha256),
                source_path=source_path,
                source_file_sha256=str(member["source_file_sha256"]),
                s_record_sha256=str(s_record_sha256),
                episode_id=episode_id,
                policy_sample_ids=tuple(str(value) for value in raw_sample_ids),
                harness_decision_ids=tuple(str(value) for value in raw_decision_ids),
                s_record=deepcopy(dict(s_record)),
            )
        )
        canonical_members.append(member)

    _require(
        list(canonical_members) == sorted(canonical_members, key=_member_sort_key),
        "multi-S members differ from canonical ordering",
    )
    _require(
        batch.get("policy_sample_count") == policy_sample_count
        and batch.get("harness_action_count") == harness_action_count,
        "multi-S sample summary differs from members",
    )
    _require(
        record.get("aggregate_sha256")
        == _aggregate_sha256(joint_version, canonical_members),
        "multi-S aggregate hash mismatch",
    )
    return ValidatedMultiSFrozenTrainingBatch(
        joint_version=joint_version,
        aggregate_sha256=str(record["aggregate_sha256"]),
        record_sha256=str(record["record_sha256"]),
        members=tuple(parsed_members),
        policy_sample_count=policy_sample_count,
        harness_action_count=harness_action_count,
    )


def persist_multi_s_frozen_training_batch(
    s_record_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    active_joint_version: JointVersion,
) -> ValidatedMultiSFrozenTrainingBatch:
    """Atomically create a canonical batch file; an existing path is never replaced."""

    record = prepare_multi_s_frozen_training_batch(
        s_record_paths,
        active_joint_version=active_joint_version,
    )
    target = _external_path(output_path, "multi-S output")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(target, flags, 0o600)
        created = True
        payload = _canonical_json(record) + b"\n"
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                target.unlink()
            except OSError:
                pass
        raise MultiSFrozenTrainingBatchError(
            f"cannot exclusively persist multi-S batch: {target}"
        ) from exc
    return load_multi_s_frozen_training_batch(
        target,
        active_joint_version=active_joint_version,
    )


def load_multi_s_frozen_training_batch(
    path: str | Path,
    *,
    active_joint_version: JointVersion,
    verify_source_files: bool = True,
) -> ValidatedMultiSFrozenTrainingBatch:
    """Load a canonical external batch record and revalidate every member."""

    _, raw, record = _read_external_json(path, "multi-S batch")
    _require(
        raw == _canonical_json(record) + b"\n",
        "persisted multi-S batch must use canonical JSON encoding",
    )
    return validate_multi_s_frozen_training_batch(
        record,
        active_joint_version=active_joint_version,
        verify_source_files=verify_source_files,
    )


def iter_member_s_records(
    batch: ValidatedMultiSFrozenTrainingBatch,
) -> Iterator[tuple[str, Mapping[str, object]]]:
    """Yield ``(member_envelope_claim, exact_S)`` for T/U, in batch order.

    The first value binds the file/path envelope used by T and U.  It is not V's
    stable exactly-once key; V must use :func:`required_v_member_claims`, which
    returns the original path-independent ``s_record_sha256`` values.
    """

    _require(
        type(batch) is ValidatedMultiSFrozenTrainingBatch,
        "T/U input must be a validated multi-S batch",
    )
    for member in batch.members:
        yield member.member_claim_sha256, deepcopy(dict(member.s_record))


def required_v_member_claims(
    batch: ValidatedMultiSFrozenTrainingBatch,
) -> tuple[str, ...]:
    """Return stable ordered S digests that V must cover once across all batches."""

    _require(
        type(batch) is ValidatedMultiSFrozenTrainingBatch,
        "V input must be a validated multi-S batch",
    )
    return tuple(member.s_record_sha256 for member in batch.members)


def validate_v_member_claim_coverage(
    batch: ValidatedMultiSFrozenTrainingBatch,
    claims: Sequence[str],
    *,
    already_claimed_s_record_sha256s: Sequence[str] = (),
) -> tuple[str, ...]:
    """Fail unless V claims every ordered S exactly once, including across batches."""

    expected = required_v_member_claims(batch)
    actual = tuple(claims)
    prior = tuple(already_claimed_s_record_sha256s)
    _require(
        all(_is_sha256(value) for value in prior) and len(set(prior)) == len(prior),
        "V prior-claim ledger must contain unique S record SHA-256 values",
    )
    _require(
        actual == expected and len(set(actual)) == len(actual),
        "V member claims must cover the ordered multi-S batch exactly once",
    )
    _require(
        set(actual).isdisjoint(prior),
        "V cannot reuse an S record SHA-256 claimed by a prior batch",
    )
    return actual
