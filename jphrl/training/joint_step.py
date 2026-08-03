from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from jphrl.paths import require_within_configured_root
from jphrl.trajectory.schema import JointVersion

from .areal_policy_candidate import validate_areal_policy_candidate

JOINT_CANDIDATE_BUNDLE_SCHEMA = "jph.joint-candidate-bundle.v1"
_EVIDENCE_SCOPE = {
    "policy_candidate_receipt_revalidated": True,
    "harness_candidate_receipt_revalidated": True,
    "source_joint_credit_shared": True,
    "logical_dual_optimizer_barrier_sealed": True,
    "policy_weights_published": False,
    "rollout_weights_synchronized": False,
    "harness_candidate_activated": False,
    "active_joint_version_changed": False,
    "joint_publish": False,
}


class JointStepError(RuntimeError):
    """Raised when two component candidates cannot cross the joint barrier."""


class JointStepRollbackError(JointStepError):
    """Raised when barrier failure is followed by incomplete parent restore."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointStepError(message)


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
        raise JointStepError(
            "joint candidate bundle is not finite canonical JSON"
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


def _exact_mapping(
    value: object,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == fields, f"{label} field set differs from schema")
    return value


def _joint_version(value: object, label: str) -> JointVersion:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(
        set(value) == set(JointVersion.__dataclass_fields__),
        f"{label} field set differs",
    )
    try:
        return JointVersion(**dict(value))
    except TypeError as exc:
        raise JointStepError(f"{label} is invalid") from exc


def _validate_harness_receipt(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
    require_checkpoint: bool,
) -> Any:
    try:
        from jphrl.harness.torch_learning import (
            validate_torch_harness_update_evidence,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
        raise JointStepError("torch is required to validate a Harness receipt") from exc
    return validate_torch_harness_update_evidence(
        record,
        active_joint_version=active_joint_version,
        require_checkpoint=require_checkpoint,
    )


@dataclass(frozen=True)
class ValidatedJointCandidateBundle:
    macro_step_id: str
    parent_release_id: str
    parent_joint_version: JointVersion
    candidate_joint_version: JointVersion
    source_joint_credit_sha256: str
    policy_engine_version: int
    candidate_policy_engine_version: int
    policy_receipt_sha256: str
    harness_receipt_sha256: str
    record_sha256: str

    @property
    def digest(self) -> str:
        return self.record_sha256


def build_joint_candidate_bundle(
    *,
    policy_receipt: Mapping[str, object],
    harness_receipt: Mapping[str, object],
    active_joint_version: JointVersion,
    parent_release_id: str,
    macro_step_id: str,
    actor_public_version: int,
    harness_public_version: str,
    require_receipt_files: bool = True,
) -> dict[str, object]:
    """Revalidate two unpublished candidates and form one immutable pair."""

    _require(
        isinstance(parent_release_id, str) and parent_release_id,
        "parent release ID is missing",
    )
    _require(
        isinstance(macro_step_id, str) and macro_step_id,
        "macro step ID is missing",
    )
    policy = validate_areal_policy_candidate(
        policy_receipt,
        active_joint_version=active_joint_version,
        require_checkpoints=require_receipt_files,
    )
    harness = _validate_harness_receipt(
        harness_receipt,
        active_joint_version=active_joint_version,
        require_checkpoint=require_receipt_files,
    )
    _require(
        policy.transaction_id == macro_step_id,
        "Policy transaction differs from joint macro step",
    )
    _require(
        policy.source_joint_credit_sha256 == harness.source_joint_credit_sha256,
        "Policy and Harness candidates use different S records",
    )
    _require(
        policy.trainable_token_count > 0 and harness.effective_batch_size > 0,
        "joint M0 requires trainable samples for both optimizers",
    )
    _require(
        actor_public_version == policy.parent_engine_version,
        "Policy actor was published before the joint barrier",
    )
    _require(
        harness_public_version == active_joint_version.harness_controller,
        "Harness candidate was activated before the joint barrier",
    )
    candidate_joint_version = replace(
        active_joint_version,
        policy=policy.candidate_policy_version,
        harness_controller=harness.candidate_version,
    )
    record: dict[str, object] = {
        "schema_version": JOINT_CANDIDATE_BUNDLE_SCHEMA,
        "macro_step": {
            "macro_step_id": macro_step_id,
            "parent_release_id": parent_release_id,
            "source_joint_credit_sha256": policy.source_joint_credit_sha256,
        },
        "parent": {
            "joint_version": asdict(active_joint_version),
            "joint_version_id": active_joint_version.version_id,
            "policy_engine_version": policy.parent_engine_version,
            "harness_controller_version": active_joint_version.harness_controller,
        },
        "candidate": {
            "joint_version": asdict(candidate_joint_version),
            "joint_version_id": candidate_joint_version.version_id,
            "policy_engine_version": policy.reserved_candidate_engine_version,
            "policy_version": policy.candidate_policy_version,
            "harness_controller_version": harness.candidate_version,
        },
        "receipts": {
            "policy": dict(policy_receipt),
            "policy_sha256": policy.record_sha256,
            "harness": dict(harness_receipt),
            "harness_sha256": str(harness_receipt["record_sha256"]),
        },
        "summary": {
            "policy_trainable_token_count": policy.trainable_token_count,
            "harness_trainable_action_count": harness.effective_batch_size,
            "component_candidate_count": 2,
        },
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }
    record["record_sha256"] = _record_sha256(record)
    validate_joint_candidate_bundle(
        record,
        active_joint_version=active_joint_version,
        actor_public_version=actor_public_version,
        harness_public_version=harness_public_version,
        require_receipt_files=require_receipt_files,
    )
    return record


def validate_joint_candidate_bundle(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
    actor_public_version: int | None = None,
    harness_public_version: str | None = None,
    require_receipt_files: bool = True,
) -> ValidatedJointCandidateBundle:
    _require(
        set(record)
        == {
            "schema_version",
            "macro_step",
            "parent",
            "candidate",
            "receipts",
            "summary",
            "evidence_scope",
            "record_sha256",
        },
        "joint candidate field set differs from schema",
    )
    _require(
        record.get("schema_version") == JOINT_CANDIDATE_BUNDLE_SCHEMA,
        "joint candidate schema differs",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "joint candidate hash mismatch",
    )
    macro_step = _exact_mapping(
        record.get("macro_step"),
        {"macro_step_id", "parent_release_id", "source_joint_credit_sha256"},
        "joint candidate macro step",
    )
    macro_step_id = macro_step.get("macro_step_id")
    parent_release_id = macro_step.get("parent_release_id")
    source_sha256 = macro_step.get("source_joint_credit_sha256")
    _require(
        isinstance(macro_step_id, str)
        and macro_step_id
        and isinstance(parent_release_id, str)
        and parent_release_id
        and _is_sha256(source_sha256),
        "joint candidate macro identity is invalid",
    )
    parent = _exact_mapping(
        record.get("parent"),
        {
            "joint_version",
            "joint_version_id",
            "policy_engine_version",
            "harness_controller_version",
        },
        "joint candidate parent",
    )
    parent_joint_version = _joint_version(
        parent.get("joint_version"),
        "joint candidate parent JointVersion",
    )
    parent_engine_version = parent.get("policy_engine_version")
    _require(
        parent.get("joint_version_id") == parent_joint_version.version_id
        and parent.get("harness_controller_version")
        == parent_joint_version.harness_controller
        and type(parent_engine_version) is int
        and parent_engine_version >= 0,
        "joint candidate parent identity differs",
    )
    if active_joint_version is not None:
        _require(
            parent_joint_version == active_joint_version,
            "joint candidate differs from lag-zero active JointVersion",
        )
    if actor_public_version is not None:
        _require(
            actor_public_version == parent_engine_version,
            "Policy actor public version crossed the barrier early",
        )
    if harness_public_version is not None:
        _require(
            harness_public_version == parent_joint_version.harness_controller,
            "Harness public version crossed the barrier early",
        )
    candidate = _exact_mapping(
        record.get("candidate"),
        {
            "joint_version",
            "joint_version_id",
            "policy_engine_version",
            "policy_version",
            "harness_controller_version",
        },
        "joint candidate identity",
    )
    candidate_joint_version = _joint_version(
        candidate.get("joint_version"),
        "candidate JointVersion",
    )
    expected_candidate = replace(
        parent_joint_version,
        policy=str(candidate.get("policy_version")),
        harness_controller=str(candidate.get("harness_controller_version")),
    )
    candidate_engine_version = candidate.get("policy_engine_version")
    _require(
        candidate_joint_version == expected_candidate
        and candidate.get("joint_version_id") == candidate_joint_version.version_id
        and candidate_joint_version != parent_joint_version
        and candidate_engine_version == parent_engine_version + 1,
        "candidate pair does not advance exactly from its parent",
    )
    receipts = _exact_mapping(
        record.get("receipts"),
        {"policy", "policy_sha256", "harness", "harness_sha256"},
        "joint candidate receipts",
    )
    policy_record = receipts.get("policy")
    harness_record = receipts.get("harness")
    _require(
        isinstance(policy_record, Mapping) and isinstance(harness_record, Mapping),
        "joint candidate component receipts are missing",
    )
    policy = validate_areal_policy_candidate(
        policy_record,
        active_joint_version=parent_joint_version,
        require_checkpoints=require_receipt_files,
    )
    harness = _validate_harness_receipt(
        harness_record,
        active_joint_version=parent_joint_version,
        require_checkpoint=require_receipt_files,
    )
    _require(
        receipts.get("policy_sha256")
        == policy.record_sha256
        == policy_record.get("record_sha256")
        and receipts.get("harness_sha256") == harness_record.get("record_sha256"),
        "joint candidate receipt hashes differ",
    )
    _require(
        policy.transaction_id == macro_step_id
        and policy.source_joint_credit_sha256
        == source_sha256
        == harness.source_joint_credit_sha256,
        "joint candidate receipt lineage differs",
    )
    _require(
        policy.candidate_policy_version == candidate_joint_version.policy
        and policy.reserved_candidate_engine_version == candidate_engine_version
        and harness.candidate_version == candidate_joint_version.harness_controller,
        "joint candidate receipt outputs differ from candidate JointVersion",
    )
    summary = _exact_mapping(
        record.get("summary"),
        {
            "policy_trainable_token_count",
            "harness_trainable_action_count",
            "component_candidate_count",
        },
        "joint candidate summary",
    )
    _require(
        summary
        == {
            "policy_trainable_token_count": policy.trainable_token_count,
            "harness_trainable_action_count": harness.effective_batch_size,
            "component_candidate_count": 2,
        }
        and policy.trainable_token_count > 0
        and harness.effective_batch_size > 0,
        "joint candidate summary differs from component receipts",
    )
    _require(
        record.get("evidence_scope") == _EVIDENCE_SCOPE,
        "joint candidate evidence scope differs from contract",
    )
    return ValidatedJointCandidateBundle(
        macro_step_id=str(macro_step_id),
        parent_release_id=str(parent_release_id),
        parent_joint_version=parent_joint_version,
        candidate_joint_version=candidate_joint_version,
        source_joint_credit_sha256=str(source_sha256),
        policy_engine_version=parent_engine_version,
        candidate_policy_engine_version=candidate_engine_version,
        policy_receipt_sha256=policy.record_sha256,
        harness_receipt_sha256=str(harness_record["record_sha256"]),
        record_sha256=str(record["record_sha256"]),
    )


def _seal_marker(root: Path, bundle: Mapping[str, object]) -> Path:
    macro_step = bundle["macro_step"]
    macro_step_id = str(macro_step["macro_step_id"])
    marker_name = hashlib.sha256(macro_step_id.encode()).hexdigest() + ".json"
    markers = root / "sealed-macro-steps"
    markers.mkdir(parents=True, exist_ok=True)
    marker = markers / marker_name
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                _canonical_json(
                    {
                        "schema_version": "jph.joint-candidate-seal.v1",
                        "macro_step_id": macro_step_id,
                        "bundle_sha256": bundle["record_sha256"],
                    }
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(markers, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    return marker


def seal_joint_candidate_bundle(
    *,
    seal_root: str | Path,
    project_root: str | Path,
    policy_receipt: Mapping[str, object],
    harness_receipt: Mapping[str, object],
    active_joint_version: JointVersion,
    parent_release_id: str,
    macro_step_id: str,
    actor_public_version: int,
    harness_public_version: str,
    restore_policy_parent: Callable[[], object] | None = None,
    restore_harness_parent: Callable[[], object] | None = None,
    require_receipt_files: bool = True,
) -> dict[str, object]:
    """Seal one macro step, restoring both parents on any failure."""

    callbacks = (restore_harness_parent, restore_policy_parent)
    _require(
        all(callback is None for callback in callbacks)
        or all(callable(callback) for callback in callbacks),
        "both parent restore callbacks must be supplied together",
    )
    try:
        root = require_within_configured_root(seal_root)
        project = Path(project_root).expanduser().resolve()
        _require(
            root != project and project not in root.parents,
            "joint barrier journal must be outside the Git project",
        )
        root.mkdir(parents=True, exist_ok=True)
        bundle = build_joint_candidate_bundle(
            policy_receipt=policy_receipt,
            harness_receipt=harness_receipt,
            active_joint_version=active_joint_version,
            parent_release_id=parent_release_id,
            macro_step_id=macro_step_id,
            actor_public_version=actor_public_version,
            harness_public_version=harness_public_version,
            require_receipt_files=require_receipt_files,
        )
        _seal_marker(root, bundle)
        return bundle
    except BaseException:
        rollback_errors: list[BaseException] = []
        for callback in callbacks:
            if callback is None:
                continue
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - both restores must be attempted
                rollback_errors.append(exc)
        if rollback_errors:
            raise JointStepRollbackError(
                "joint barrier failed and parent rollback was incomplete"
            ) from rollback_errors[0]
        raise


__all__ = [
    "JOINT_CANDIDATE_BUNDLE_SCHEMA",
    "JointStepError",
    "JointStepRollbackError",
    "ValidatedJointCandidateBundle",
    "build_joint_candidate_bundle",
    "seal_joint_candidate_bundle",
    "validate_joint_candidate_bundle",
]
