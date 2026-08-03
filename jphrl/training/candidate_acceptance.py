from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from jphrl.joint_release import CandidateArtifact, JointReleaseStore, ReleaseManifest
from jphrl.paths import require_outside_repository
from jphrl.trajectory.schema import JointVersion

from .joint_step import ValidatedJointCandidateBundle, validate_joint_candidate_bundle
from .production_checkpoint import (
    ValidatedProductionCheckpoint,
    validate_exact_joint_recovery_evidence,
    validate_production_joint_checkpoint,
)

CANDIDATE_ACCEPTANCE_SCHEMA = "jph.joint-candidate-acceptance.v2"
CANDIDATE_PROBE_SCHEMA = "jph.joint-candidate-probe.v3"
PRODUCTION_POLICY_ARTIFACT_SCHEMA = "jph.production-policy-release-artifact.v2"
_LEGACY_POLICY_ARTIFACT_SCHEMA = "jph.production-policy-release-artifact.v1"
PRODUCTION_HARNESS_ARTIFACT_SCHEMA = "jph.production-harness-release-artifact.v1"
# Compatibility aliases for the earlier X draft.  Production validation is
# exact and therefore accepts only the values above.
POLICY_CANDIDATE_ARTIFACT_SCHEMA = PRODUCTION_POLICY_ARTIFACT_SCHEMA
HARNESS_CANDIDATE_ARTIFACT_SCHEMA = PRODUCTION_HARNESS_ARTIFACT_SCHEMA
REQUIRED_SUITE_KINDS = (
    "policy_heldout",
    "harness_heldout",
    "joint_safety",
    "historical_regression",
)
_SECRET_FIELDS = {
    "access_token",
    "admin_api_key",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "github_token",
    "password",
    "refresh_token",
    "secret",
    "secret_key",
    "session_api_key",
    "token",
}
_EVIDENCE_SCOPE = {
    "joint_candidate_revalidated": True,
    "production_checkpoint_revalidated": True,
    "live_exact_recovery_required_to_emit": True,
    "acceptance_spec_externally_frozen": True,
    "probe_outputs_framework_scored": True,
    "release_store_lineage_revalidated": True,
    "production_artifacts_revalidated": True,
    "all_critical_suites_passed": True,
    "parent_remained_active": True,
    "candidate_accepted": True,
    "persisted_report_regrants_live_acceptance": False,
    "performance_improvement_claimed": False,
    "policy_weights_published": False,
    "rollout_weights_synchronized": False,
    "harness_candidate_activated": False,
    "active_joint_version_changed": False,
    "joint_publish": False,
}
_LIVE_ACCEPTANCE_TOKEN = object()


class CandidateAcceptanceError(RuntimeError):
    """Raised when a Policy/Harness candidate pair is unsafe to activate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateAcceptanceError(message)


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
        raise CandidateAcceptanceError(
            "candidate acceptance record is not finite canonical JSON"
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


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            _require(
                normalized not in _SECRET_FIELDS
                and not normalized.endswith(
                    ("_api_key", "_password", "_secret", "_token")
                ),
                f"credential field cannot enter candidate acceptance: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == fields, f"{label} field set differs from schema")
    return value


def _joint_version(value: object, label: str) -> JointVersion:
    mapping = _exact(value, set(JointVersion.__dataclass_fields__), label)
    try:
        return JointVersion(**dict(mapping))
    except TypeError as exc:
        raise CandidateAcceptanceError(f"{label} is invalid") from exc


@dataclass(frozen=True)
class CandidateAcceptanceSuite:
    kind: str
    suite_id: str
    fixture_sha256: str
    metric_name: str
    minimum_score: float
    minimum_sample_count: int = 1

    def validate(self) -> None:
        _require(self.kind in REQUIRED_SUITE_KINDS, "unknown critical suite kind")
        _require(
            isinstance(self.suite_id, str) and bool(self.suite_id),
            "critical suite ID is missing",
        )
        _require(_is_sha256(self.fixture_sha256), "suite fixture hash is invalid")
        _require(
            isinstance(self.metric_name, str) and bool(self.metric_name),
            "suite metric name is missing",
        )
        _require(_finite_number(self.minimum_score), "suite threshold is not finite")
        _require(
            type(self.minimum_sample_count) is int and self.minimum_sample_count > 0,
            "suite minimum sample count must be positive",
        )

    def to_record(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CandidateAcceptanceSpec:
    suites: tuple[CandidateAcceptanceSuite, ...]

    def validate(self) -> None:
        _require(
            len(self.suites) == len(REQUIRED_SUITE_KINDS),
            "all four critical candidate suites are required",
        )
        for suite in self.suites:
            _require(
                type(suite) is CandidateAcceptanceSuite,
                "critical suite has the wrong type",
            )
            suite.validate()
        _require(
            tuple(suite.kind for suite in self.suites) == REQUIRED_SUITE_KINDS,
            "critical suites must be complete and in the frozen order",
        )
        _require(
            len({suite.suite_id for suite in self.suites}) == len(self.suites),
            "critical suite IDs must be unique",
        )
        joint_safety = self.suites[REQUIRED_SUITE_KINDS.index("joint_safety")]
        _require(
            joint_safety.minimum_sample_count == 1,
            "joint_safety freezes exactly one production probe observation",
        )

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {"suites": [suite.to_record() for suite in self.suites]}

    @property
    def digest(self) -> str:
        return _sha256(self.to_record())


@dataclass(frozen=True)
class CandidateProbeObservation:
    """One raw evaluator measurement; it carries no pass/fail decision."""

    sample_id: str
    metric_value: float | None
    output: object
    production_probe_output: bytes | None = None


@dataclass(frozen=True)
class CandidateAcceptanceReportAudit:
    parent_release_id: str
    parent_joint_version: JointVersion
    candidate_joint_version: JointVersion
    candidate_release_id: str
    macro_step_id: str
    bundle_sha256: str
    checkpoint_manifest_sha256: str
    exact_recovery_sha256: str
    policy_receipt_sha256: str
    harness_receipt_sha256: str
    policy_artifact_sha256: str
    harness_artifact_sha256: str
    spec_sha256: str
    critical_suite_count: int
    record_sha256: str
    live_acceptance: bool = False


# Kept as a source-compatible name.  The explicit ``live_acceptance=False``
# prevents a persisted report audit from being confused with the live token.
ValidatedCandidateAcceptance = CandidateAcceptanceReportAudit


@dataclass(frozen=True, init=False)
class LiveCandidateAcceptance:
    """Process-local X attestation.  It cannot be reconstructed from JSON."""

    _token: object
    _report: dict[str, object]
    _audit: CandidateAcceptanceReportAudit

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("LiveCandidateAcceptance is emitted only by the X runner")

    @classmethod
    def _create(
        cls,
        *,
        report: Mapping[str, object],
        audit: CandidateAcceptanceReportAudit,
        token: object,
    ) -> LiveCandidateAcceptance:
        _require(token is _LIVE_ACCEPTANCE_TOKEN, "live acceptance token is invalid")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_token", token)
        object.__setattr__(instance, "_report", deepcopy(dict(report)))
        object.__setattr__(instance, "_audit", audit)
        return instance

    @property
    def report(self) -> dict[str, object]:
        return deepcopy(self._report)

    @property
    def record_sha256(self) -> str:
        return self._audit.record_sha256

    @property
    def parent_release_id(self) -> str:
        return self._audit.parent_release_id

    @property
    def candidate_release_id(self) -> str:
        return self._audit.candidate_release_id

    @property
    def parent_joint_version(self) -> JointVersion:
        return self._audit.parent_joint_version

    @property
    def candidate_joint_version(self) -> JointVersion:
        return self._audit.candidate_joint_version

    @property
    def bundle_sha256(self) -> str:
        return self._audit.bundle_sha256

    @property
    def checkpoint_manifest_sha256(self) -> str:
        return self._audit.checkpoint_manifest_sha256

    @property
    def exact_recovery_sha256(self) -> str:
        return self._audit.exact_recovery_sha256

    @property
    def policy_receipt_sha256(self) -> str:
        return self._audit.policy_receipt_sha256

    @property
    def harness_receipt_sha256(self) -> str:
        return self._audit.harness_receipt_sha256

    @property
    def policy_artifact_sha256(self) -> str:
        return self._audit.policy_artifact_sha256

    @property
    def harness_artifact_sha256(self) -> str:
        return self._audit.harness_artifact_sha256

    @property
    def spec_sha256(self) -> str:
        return self._audit.spec_sha256

    @property
    def live_acceptance(self) -> bool:
        return True


def _component_records(
    joint_candidate_bundle: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    receipts = _exact(
        joint_candidate_bundle.get("receipts"),
        {"policy", "policy_sha256", "harness", "harness_sha256"},
        "joint candidate receipts",
    )
    policy = receipts.get("policy")
    harness = receipts.get("harness")
    _require(
        isinstance(policy, Mapping) and isinstance(harness, Mapping),
        "joint candidate component receipts are missing",
    )
    return policy, harness


def _expected_production_artifacts(
    *,
    joint_candidate_bundle: Mapping[str, object],
    bundle: ValidatedJointCandidateBundle,
    checkpoint: ValidatedProductionCheckpoint,
    live_serving_exports: object | None = None,
) -> tuple[CandidateArtifact, CandidateArtifact]:
    policy_receipt, harness_receipt = _component_records(joint_candidate_bundle)
    checkpoints = _exact(
        policy_receipt.get("checkpoints"),
        {"parent_path", "parent_manifest", "candidate_path", "candidate_manifest"},
        "Policy candidate checkpoints",
    )
    candidate_manifest = _exact(
        checkpoints.get("candidate_manifest"),
        {"files", "manifest_sha256"},
        "Policy candidate checkpoint manifest",
    )
    policy_checkpoint_sha256 = candidate_manifest.get("manifest_sha256")
    harness_checkpoint_sha256 = harness_receipt.get("checkpoint_sha256")
    harness_parameter_sha256 = harness_receipt.get("parameter_digest_after")
    _require(
        _is_sha256(policy_checkpoint_sha256)
        and _is_sha256(harness_checkpoint_sha256)
        and _is_sha256(harness_parameter_sha256),
        "production candidate checkpoint digests are invalid",
    )
    policy_payload: dict[str, object] = {
        "schema_version": _LEGACY_POLICY_ARTIFACT_SCHEMA,
        "candidate_joint_version_id": bundle.candidate_joint_version.version_id,
        "joint_candidate_bundle_sha256": bundle.record_sha256,
        "production_checkpoint_manifest_sha256": checkpoint.record_sha256,
        "policy_update_receipt_sha256": bundle.policy_receipt_sha256,
        "policy_checkpoint_manifest_sha256": policy_checkpoint_sha256,
        "policy_engine_version": bundle.candidate_policy_engine_version,
    }
    if live_serving_exports is not None:
        try:
            from .areal_production_worker import (
                require_live_areal_serving_export_pair,
            )

            serving = require_live_areal_serving_export_pair(live_serving_exports)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise CandidateAcceptanceError(str(exc)) from exc
        candidate_serving = serving.candidate
        _require(
            candidate_serving.joint_version == bundle.candidate_joint_version
            and candidate_serving.policy_engine_version
            == bundle.candidate_policy_engine_version
            and candidate_serving.source_dcp_manifest_sha256
            == policy_checkpoint_sha256
            and candidate_serving.policy_candidate_record_sha256
            == bundle.policy_receipt_sha256,
            "live serving export differs from T/V/W candidate lineage",
        )
        policy_payload.update(
            {
                "schema_version": PRODUCTION_POLICY_ARTIFACT_SCHEMA,
                "policy_serving_export_manifest_sha256": (
                    candidate_serving.serving_export_manifest_sha256
                ),
                "policy_serving_parameter_sha256": (
                    candidate_serving.serving_parameter_sha256
                ),
                "policy_serving_lineage_record_sha256": (
                    candidate_serving.record_sha256
                ),
            }
        )
    policy = CandidateArtifact(
        component="policy",
        version=bundle.candidate_joint_version.policy,
        payload=policy_payload,
    )
    harness = CandidateArtifact(
        component="harness",
        version=bundle.candidate_joint_version.harness_controller,
        payload={
            "schema_version": HARNESS_CANDIDATE_ARTIFACT_SCHEMA,
            "candidate_joint_version_id": bundle.candidate_joint_version.version_id,
            "joint_candidate_bundle_sha256": bundle.record_sha256,
            "production_checkpoint_manifest_sha256": checkpoint.record_sha256,
            "harness_update_receipt_sha256": bundle.harness_receipt_sha256,
            "harness_checkpoint_sha256": harness_checkpoint_sha256,
            "harness_parameter_sha256": harness_parameter_sha256,
        },
    )
    policy.validate()
    harness.validate()
    return policy, harness


def build_production_candidate_artifacts(
    *,
    joint_candidate_bundle: Mapping[str, object],
    checkpoint_manifest: str | Path,
    live_serving_exports: object | None = None,
    require_component_files: bool = True,
) -> tuple[CandidateArtifact, CandidateArtifact]:
    """Build the only candidate object payloads accepted by production X/Y."""

    _assert_no_secrets(joint_candidate_bundle, "joint_candidate_bundle")
    _require(
        live_serving_exports is not None or not require_component_files,
        "production candidate artifacts require native live serving-export lineage",
    )
    try:
        bundle = validate_joint_candidate_bundle(
            joint_candidate_bundle,
            require_receipt_files=require_component_files,
        )
        checkpoint = validate_production_joint_checkpoint(
            checkpoint_manifest,
            require_component_files=require_component_files,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CandidateAcceptanceError(str(exc)) from exc
    _require(
        checkpoint.bundle_sha256 == bundle.record_sha256
        and checkpoint.parent_joint_version == bundle.parent_joint_version
        and checkpoint.candidate_joint_version == bundle.candidate_joint_version
        and checkpoint.macro_step_id == bundle.macro_step_id,
        "production checkpoint differs from the sealed joint candidate",
    )
    return _expected_production_artifacts(
        joint_candidate_bundle=joint_candidate_bundle,
        bundle=bundle,
        checkpoint=checkpoint,
        live_serving_exports=live_serving_exports,
    )


def _validate_store_binding(
    *,
    release_store: JointReleaseStore,
    candidate_release_id: str,
    joint_candidate_bundle: Mapping[str, object],
    bundle: ValidatedJointCandidateBundle,
    checkpoint: ValidatedProductionCheckpoint,
    live_serving_exports: object | None = None,
) -> tuple[ReleaseManifest, ReleaseManifest, CandidateArtifact, CandidateArtifact]:
    _require(
        type(release_store) is JointReleaseStore,
        "candidate acceptance requires the native JointReleaseStore",
    )
    try:
        active = release_store.read_active()
        candidate = release_store.read_manifest(candidate_release_id)
        policy, harness = release_store.read_artifacts(candidate_release_id)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CandidateAcceptanceError(str(exc)) from exc
    _require(active is not None, "release store has no active parent")
    _require(
        active.release_id == bundle.parent_release_id
        and active.joint_version == bundle.parent_joint_version,
        "active parent differs from the sealed joint candidate",
    )
    _require(
        candidate.parent_release_id == active.release_id
        and candidate.joint_version == bundle.candidate_joint_version,
        "candidate release is not the active parent's direct staged child",
    )
    expected_policy, expected_harness = _expected_production_artifacts(
        joint_candidate_bundle=joint_candidate_bundle,
        bundle=bundle,
        checkpoint=checkpoint,
        live_serving_exports=live_serving_exports,
    )
    policy_matches = policy.to_dict() == expected_policy.to_dict()
    if live_serving_exports is None and isinstance(policy.payload, Mapping):
        payload = dict(policy.payload)
        serving_fields = {
            "policy_serving_export_manifest_sha256",
            "policy_serving_parameter_sha256",
            "policy_serving_lineage_record_sha256",
        }
        base = dict(expected_policy.payload)
        if payload.get("schema_version") == PRODUCTION_POLICY_ARTIFACT_SCHEMA:
            _require(
                set(payload) == set(base) | serving_fields,
                "production Policy serving artifact field set differs",
            )
            for field in serving_fields:
                _require(
                    _is_sha256(payload.get(field)),
                    f"production Policy {field} is invalid",
                )
            payload_without_serving = {
                key: value for key, value in payload.items() if key not in serving_fields
            }
            payload_without_serving["schema_version"] = _LEGACY_POLICY_ARTIFACT_SCHEMA
            policy_matches = payload_without_serving == base
    _require(
        policy_matches and harness.to_dict() == expected_harness.to_dict(),
        "candidate release artifacts differ from V/W/T/U evidence",
    )
    return active, candidate, policy, harness


def _active_observation(active: ReleaseManifest) -> dict[str, object]:
    return {
        "release_id": active.release_id,
        "joint_version": asdict(active.joint_version),
        "joint_version_id": active.joint_version.version_id,
    }


def _compute_probe(
    value: object,
    *,
    suite: CandidateAcceptanceSuite,
    candidate_joint_version: JointVersion,
) -> dict[str, object]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)),
        f"{suite.kind} probe must return raw observations",
    )
    sanitized: list[dict[str, object]] = []
    production_probe_outputs: list[bytes] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        _require(
            type(item) is CandidateProbeObservation,
            f"{suite.kind} probe item {index} is not a raw observation",
        )
        _require(
            isinstance(item.sample_id, str)
            and bool(item.sample_id)
            and item.sample_id not in seen_ids,
            f"{suite.kind} probe sample ID is missing or duplicated",
        )
        seen_ids.add(item.sample_id)
        _assert_no_secrets(item.output, f"{suite.kind}[{item.sample_id}].output")
        production_probe_output = item.production_probe_output
        if production_probe_output is not None:
            _require(
                type(production_probe_output) is bytes
                and bool(production_probe_output),
                f"{suite.kind} production probe output must be non-empty raw bytes",
            )
            production_probe_outputs.append(production_probe_output)
        output_sha256 = _sha256(item.output)
        metric_value: float | None
        if item.metric_value is None:
            metric_value = None
        else:
            _require(
                _finite_number(item.metric_value),
                f"{suite.kind} probe metric is not finite",
            )
            metric_value = float(item.metric_value)
        sanitized.append(
            {
                "sample_id": item.sample_id,
                "metric_value": metric_value,
                "output_sha256": output_sha256,
                "production_probe_output_sha256": (
                    None
                    if production_probe_output is None
                    else hashlib.sha256(production_probe_output).hexdigest()
                ),
                "production_probe_output_size_bytes": (
                    None
                    if production_probe_output is None
                    else len(production_probe_output)
                ),
            }
        )
    valid_values = [
        float(item["metric_value"])
        for item in sanitized
        if item["metric_value"] is not None
    ]
    sample_count = len(sanitized)
    invalid_count = sample_count - len(valid_values)
    score = sum(valid_values) / len(valid_values) if valid_values else None
    passed = (
        sample_count >= suite.minimum_sample_count
        and invalid_count == 0
        and score is not None
        and score >= float(suite.minimum_score)
    )
    if suite.kind == "joint_safety":
        _require(
            sample_count == 1 and len(production_probe_outputs) == 1,
            "joint_safety requires exactly one raw production probe output",
        )
        production_probe_output_sha256: str | None = hashlib.sha256(
            production_probe_outputs[0]
        ).hexdigest()
        production_probe_output_size_bytes: int | None = len(
            production_probe_outputs[0]
        )
    else:
        _require(
            not production_probe_outputs,
            "only joint_safety may carry a raw production probe output",
        )
        production_probe_output_sha256 = None
        production_probe_output_size_bytes = None
    return {
        "schema_version": CANDIDATE_PROBE_SCHEMA,
        "suite_kind": suite.kind,
        "suite_id": suite.suite_id,
        "fixture_sha256": suite.fixture_sha256,
        "candidate_joint_version_id": candidate_joint_version.version_id,
        "metric_name": suite.metric_name,
        "minimum_score": float(suite.minimum_score),
        "minimum_sample_count": suite.minimum_sample_count,
        "observations": sanitized,
        "sample_count": sample_count,
        "invalid_count": invalid_count,
        "score": score,
        "passed": passed,
        "output_sha256": _sha256(sanitized),
        "production_probe_output_sha256": production_probe_output_sha256,
        "production_probe_output_size_bytes": production_probe_output_size_bytes,
    }


def _validate_computed_probe(
    value: object,
    *,
    suite: CandidateAcceptanceSuite,
    candidate_joint_version: JointVersion,
) -> dict[str, object]:
    record = _exact(
        value,
        {
            "schema_version",
            "suite_kind",
            "suite_id",
            "fixture_sha256",
            "candidate_joint_version_id",
            "metric_name",
            "minimum_score",
            "minimum_sample_count",
            "observations",
            "sample_count",
            "invalid_count",
            "score",
            "passed",
            "output_sha256",
            "production_probe_output_sha256",
            "production_probe_output_size_bytes",
        },
        f"{suite.kind} computed probe",
    )
    _assert_no_secrets(record, f"{suite.kind}_computed_probe")
    _require(
        record.get("schema_version") == CANDIDATE_PROBE_SCHEMA
        and record.get("suite_kind") == suite.kind
        and record.get("suite_id") == suite.suite_id
        and record.get("fixture_sha256") == suite.fixture_sha256
        and record.get("candidate_joint_version_id")
        == candidate_joint_version.version_id
        and record.get("metric_name") == suite.metric_name
        and record.get("minimum_score") == float(suite.minimum_score)
        and record.get("minimum_sample_count") == suite.minimum_sample_count,
        f"{suite.kind} computed probe differs from the external frozen suite",
    )
    observations = record.get("observations")
    _require(isinstance(observations, list), f"{suite.kind} observations are missing")
    seen_ids: set[str] = set()
    valid_values: list[float] = []
    for item in observations:
        observation = _exact(
            item,
            {
                "sample_id",
                "metric_value",
                "output_sha256",
                "production_probe_output_sha256",
                "production_probe_output_size_bytes",
            },
            f"{suite.kind} sanitized observation",
        )
        sample_id = observation.get("sample_id")
        _require(
            isinstance(sample_id, str)
            and bool(sample_id)
            and sample_id not in seen_ids
            and _is_sha256(observation.get("output_sha256")),
            f"{suite.kind} sanitized observation identity is invalid",
        )
        seen_ids.add(sample_id)
        metric = observation.get("metric_value")
        production_sha256 = observation.get("production_probe_output_sha256")
        production_size = observation.get("production_probe_output_size_bytes")
        _require(
            (production_sha256 is None and production_size is None)
            or (
                _is_sha256(production_sha256)
                and type(production_size) is int
                and production_size > 0
            ),
            f"{suite.kind} production probe identity is invalid",
        )
        if metric is not None:
            _require(
                _finite_number(metric), f"{suite.kind} observation metric is not finite"
            )
            valid_values.append(float(metric))
    sample_count = len(observations)
    invalid_count = sample_count - len(valid_values)
    score = sum(valid_values) / len(valid_values) if valid_values else None
    passed = (
        sample_count >= suite.minimum_sample_count
        and invalid_count == 0
        and score is not None
        and score >= float(suite.minimum_score)
    )
    production_observations = [
        item
        for item in observations
        if item["production_probe_output_sha256"] is not None
    ]
    if suite.kind == "joint_safety":
        _require(
            sample_count == 1 and len(production_observations) == 1,
            "joint_safety requires exactly one raw production probe identity",
        )
        expected_production_sha256 = production_observations[0][
            "production_probe_output_sha256"
        ]
        expected_production_size = production_observations[0][
            "production_probe_output_size_bytes"
        ]
    else:
        _require(
            not production_observations,
            "only joint_safety may carry a production probe identity",
        )
        expected_production_sha256 = None
        expected_production_size = None
    _require(
        record.get("sample_count") == sample_count
        and record.get("invalid_count") == invalid_count
        and record.get("score") == score
        and record.get("passed") is passed
        and record.get("output_sha256") == _sha256(observations),
        f"{suite.kind} framework-computed score or digest differs",
    )
    _require(
        record.get("production_probe_output_sha256")
        == expected_production_sha256
        and record.get("production_probe_output_size_bytes")
        == expected_production_size,
        f"{suite.kind} production probe digest differs",
    )
    _require(passed, f"{suite.kind} critical gate did not pass")
    return dict(record)


def _live_recovery_record(live_recovery: object) -> Mapping[str, object]:
    record = getattr(live_recovery, "record", None)
    _require(
        isinstance(record, Mapping),
        "live exact recovery does not expose its sealed record",
    )
    return record


def _require_live_recovery(
    live_recovery: object,
    *,
    checkpoint_manifest: str | Path,
    checkpoint: ValidatedProductionCheckpoint,
) -> object:
    # Imported lazily so W and X can be developed independently without a
    # partial-module import cycle.  W owns the exact-type/private-token check.
    try:
        from .production_checkpoint import require_live_exact_joint_recovery

        recovery = require_live_exact_joint_recovery(
            live_recovery,
            manifest=checkpoint_manifest,
            current_topology=checkpoint.topology,
        )
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
        raise CandidateAcceptanceError(str(exc)) from exc
    _require(
        getattr(recovery, "checkpoint_manifest_sha256", None)
        == checkpoint.record_sha256
        and getattr(recovery, "candidate_joint_version", None)
        == checkpoint.candidate_joint_version
        and getattr(recovery, "macro_step_id", None) == checkpoint.macro_step_id
        and _is_sha256(getattr(recovery, "record_sha256", None)),
        "live exact recovery differs from the candidate checkpoint",
    )
    return recovery


def _write_new_report(
    root: str | Path,
    *,
    project_root: str | Path,
    bundle_sha256: str,
    report: Mapping[str, object],
) -> Path:
    del project_root
    destination_root = require_outside_repository(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    os.chmod(destination_root, 0o700)
    destination = destination_root / f"acceptance-{bundle_sha256}.json"
    _require(not destination.exists(), "candidate acceptance report already exists")
    temporary = destination_root / f".{destination.name}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(report))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        os.chmod(destination, 0o600)
        directory = os.open(destination_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _run_joint_candidate_acceptance_under_lease(
    *,
    joint_candidate_bundle: Mapping[str, object],
    checkpoint_manifest: str | Path,
    live_exact_recovery: object,
    candidate_release_id: str,
    expected_spec: CandidateAcceptanceSpec,
    probes: Mapping[
        str,
        Callable[
            [JointVersion, CandidateAcceptanceSuite],
            Sequence[CandidateProbeObservation],
        ],
    ],
    release_store: JointReleaseStore,
    report_root: str | Path,
    project_root: str | Path,
    live_serving_exports: object | None = None,
    require_component_files: bool = True,
) -> LiveCandidateAcceptance:
    """Run frozen shadow gates and emit a non-serializable live attestation."""

    _assert_no_secrets(joint_candidate_bundle, "joint_candidate_bundle")
    _require(
        live_serving_exports is not None or not require_component_files,
        "production X requires native live DCP-to-serving export lineage",
    )
    _require(
        type(expected_spec) is CandidateAcceptanceSpec,
        "candidate acceptance requires an external frozen spec",
    )
    expected_spec.validate()
    _require(
        set(probes) == set(REQUIRED_SUITE_KINDS),
        "critical candidate probe set is incomplete or contains a skipped gate",
    )
    try:
        bundle = validate_joint_candidate_bundle(
            joint_candidate_bundle,
            require_receipt_files=require_component_files,
        )
        checkpoint = validate_production_joint_checkpoint(
            checkpoint_manifest,
            require_component_files=require_component_files,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CandidateAcceptanceError(str(exc)) from exc
    _require(
        checkpoint.bundle_sha256 == bundle.record_sha256
        and checkpoint.parent_joint_version == bundle.parent_joint_version
        and checkpoint.candidate_joint_version == bundle.candidate_joint_version
        and checkpoint.macro_step_id == bundle.macro_step_id
        and checkpoint.source_joint_credit_sha256 == bundle.source_joint_credit_sha256,
        "production checkpoint differs from the sealed joint candidate",
    )
    recovery = _require_live_recovery(
        live_exact_recovery,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint=checkpoint,
    )
    active, _, policy_artifact, harness_artifact = _validate_store_binding(
        release_store=release_store,
        candidate_release_id=candidate_release_id,
        joint_candidate_bundle=joint_candidate_bundle,
        bundle=bundle,
        checkpoint=checkpoint,
        live_serving_exports=live_serving_exports,
    )

    observations = [_active_observation(active)]
    suite_records: list[dict[str, object]] = []
    for suite in expected_spec.suites:
        callback = probes[suite.kind]
        _require(callable(callback), f"{suite.kind} probe is not callable")
        first = _compute_probe(
            callback(bundle.candidate_joint_version, suite),
            suite=suite,
            candidate_joint_version=bundle.candidate_joint_version,
        )
        _validate_computed_probe(
            first,
            suite=suite,
            candidate_joint_version=bundle.candidate_joint_version,
        )
        active, _, _, _ = _validate_store_binding(
            release_store=release_store,
            candidate_release_id=candidate_release_id,
            joint_candidate_bundle=joint_candidate_bundle,
            bundle=bundle,
            checkpoint=checkpoint,
            live_serving_exports=live_serving_exports,
        )
        observations.append(_active_observation(active))
        second = _compute_probe(
            callback(bundle.candidate_joint_version, suite),
            suite=suite,
            candidate_joint_version=bundle.candidate_joint_version,
        )
        _validate_computed_probe(
            second,
            suite=suite,
            candidate_joint_version=bundle.candidate_joint_version,
        )
        _require(
            _canonical_json(first) == _canonical_json(second),
            f"{suite.kind} probe is not deterministic",
        )
        active, _, _, _ = _validate_store_binding(
            release_store=release_store,
            candidate_release_id=candidate_release_id,
            joint_candidate_bundle=joint_candidate_bundle,
            bundle=bundle,
            checkpoint=checkpoint,
            live_serving_exports=live_serving_exports,
        )
        observations.append(_active_observation(active))
        suite_records.append(
            {
                "spec": suite.to_record(),
                "probe": first,
                "deterministic_replay_count": 2,
                "critical": True,
                "passed": True,
            }
        )

    report: dict[str, object] = {
        "schema_version": CANDIDATE_ACCEPTANCE_SCHEMA,
        "identity": {
            "parent_release_id": bundle.parent_release_id,
            "candidate_release_id": candidate_release_id,
            "parent_joint_version": asdict(bundle.parent_joint_version),
            "parent_joint_version_id": bundle.parent_joint_version.version_id,
            "candidate_joint_version": asdict(bundle.candidate_joint_version),
            "candidate_joint_version_id": bundle.candidate_joint_version.version_id,
            "macro_step_id": bundle.macro_step_id,
            "source_joint_credit_sha256": bundle.source_joint_credit_sha256,
            "joint_candidate_bundle_sha256": bundle.record_sha256,
            "checkpoint_manifest_sha256": checkpoint.record_sha256,
            "exact_recovery_sha256": recovery.record_sha256,
            "policy_update_receipt_sha256": bundle.policy_receipt_sha256,
            "harness_update_receipt_sha256": bundle.harness_receipt_sha256,
            "policy_artifact_sha256": policy_artifact.digest,
            "harness_artifact_sha256": harness_artifact.digest,
            "acceptance_spec_sha256": expected_spec.digest,
        },
        "critical_suites": suite_records,
        "active_parent_observations": observations,
        "decision": {
            "accepted": True,
            "critical_gate_count": len(REQUIRED_SUITE_KINDS),
            "passed_critical_gate_count": len(REQUIRED_SUITE_KINDS),
            "skipped_critical_gate_count": 0,
        },
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }
    report["record_sha256"] = _record_sha256(report)
    audit = validate_candidate_acceptance_report(
        report,
        expected_spec=expected_spec,
        release_store=release_store,
        joint_candidate_bundle=joint_candidate_bundle,
        checkpoint_manifest=checkpoint_manifest,
        exact_recovery_evidence=_live_recovery_record(recovery),
        require_component_files=require_component_files,
    )
    _write_new_report(
        report_root,
        project_root=project_root,
        bundle_sha256=bundle.record_sha256,
        report=report,
    )
    return LiveCandidateAcceptance._create(
        report=report,
        audit=audit,
        token=_LIVE_ACCEPTANCE_TOKEN,
    )


def run_joint_candidate_acceptance(
    *,
    joint_candidate_bundle: Mapping[str, object],
    checkpoint_manifest: str | Path,
    live_exact_recovery: object,
    candidate_release_id: str,
    expected_spec: CandidateAcceptanceSpec,
    probes: Mapping[
        str,
        Callable[
            [JointVersion, CandidateAcceptanceSuite],
            Sequence[CandidateProbeObservation],
        ],
    ],
    release_store: JointReleaseStore,
    report_root: str | Path,
    project_root: str | Path,
    live_serving_exports: object | None = None,
    require_component_files: bool = True,
) -> LiveCandidateAcceptance:
    """Hold the store-global activation lease across the complete X run."""

    _require(
        type(release_store) is JointReleaseStore,
        "candidate acceptance requires the native JointReleaseStore",
    )
    with release_store.activation_lease():
        return _run_joint_candidate_acceptance_under_lease(
            joint_candidate_bundle=joint_candidate_bundle,
            checkpoint_manifest=checkpoint_manifest,
            live_exact_recovery=live_exact_recovery,
            candidate_release_id=candidate_release_id,
            expected_spec=expected_spec,
            probes=probes,
            release_store=release_store,
            report_root=report_root,
            project_root=project_root,
            live_serving_exports=live_serving_exports,
            require_component_files=require_component_files,
        )


def validate_candidate_acceptance_report(
    record: Mapping[str, object],
    *,
    expected_spec: CandidateAcceptanceSpec,
    release_store: JointReleaseStore,
    joint_candidate_bundle: Mapping[str, object],
    checkpoint_manifest: str | Path,
    exact_recovery_evidence: Mapping[str, object],
    require_component_files: bool = True,
) -> CandidateAcceptanceReportAudit:
    """Validate persisted integrity; this never grants live acceptance."""

    _require(
        set(record)
        == {
            "schema_version",
            "identity",
            "critical_suites",
            "active_parent_observations",
            "decision",
            "evidence_scope",
            "record_sha256",
        },
        "candidate acceptance field set differs from schema",
    )
    _require(
        record.get("schema_version") == CANDIDATE_ACCEPTANCE_SCHEMA,
        "candidate acceptance schema differs",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "candidate acceptance hash mismatch",
    )
    _assert_no_secrets(record)
    _require(
        type(expected_spec) is CandidateAcceptanceSpec,
        "candidate acceptance requires an external frozen spec",
    )
    expected_spec.validate()
    try:
        bundle = validate_joint_candidate_bundle(
            joint_candidate_bundle,
            require_receipt_files=require_component_files,
        )
        checkpoint = validate_production_joint_checkpoint(
            checkpoint_manifest,
            require_component_files=require_component_files,
        )
        recovery = validate_exact_joint_recovery_evidence(
            exact_recovery_evidence,
            manifest=checkpoint_manifest,
            current_topology=checkpoint.topology,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CandidateAcceptanceError(str(exc)) from exc
    active, _, policy_artifact, harness_artifact = _validate_store_binding(
        release_store=release_store,
        candidate_release_id=str(
            record.get("identity", {}).get("candidate_release_id", "")
        )
        if isinstance(record.get("identity"), Mapping)
        else "",
        joint_candidate_bundle=joint_candidate_bundle,
        bundle=bundle,
        checkpoint=checkpoint,
    )
    identity = _exact(
        record.get("identity"),
        {
            "parent_release_id",
            "candidate_release_id",
            "parent_joint_version",
            "parent_joint_version_id",
            "candidate_joint_version",
            "candidate_joint_version_id",
            "macro_step_id",
            "source_joint_credit_sha256",
            "joint_candidate_bundle_sha256",
            "checkpoint_manifest_sha256",
            "exact_recovery_sha256",
            "policy_update_receipt_sha256",
            "harness_update_receipt_sha256",
            "policy_artifact_sha256",
            "harness_artifact_sha256",
            "acceptance_spec_sha256",
        },
        "candidate acceptance identity",
    )
    parent = _joint_version(
        identity.get("parent_joint_version"),
        "candidate acceptance parent JointVersion",
    )
    candidate = _joint_version(
        identity.get("candidate_joint_version"),
        "candidate acceptance candidate JointVersion",
    )
    candidate_release_id = identity.get("candidate_release_id")
    recovery_record_sha = recovery.get("record_sha256")
    _require(
        identity.get("acceptance_spec_sha256") == expected_spec.digest,
        "candidate acceptance differs from the external frozen spec",
    )
    _require(
        identity.get("parent_release_id") == bundle.parent_release_id
        and isinstance(candidate_release_id, str)
        and bool(candidate_release_id)
        and parent == bundle.parent_joint_version
        and candidate == bundle.candidate_joint_version
        and identity.get("parent_joint_version_id") == parent.version_id
        and identity.get("candidate_joint_version_id") == candidate.version_id
        and identity.get("macro_step_id") == bundle.macro_step_id
        and identity.get("source_joint_credit_sha256")
        == bundle.source_joint_credit_sha256
        and identity.get("joint_candidate_bundle_sha256") == bundle.record_sha256
        and identity.get("checkpoint_manifest_sha256") == checkpoint.record_sha256
        and identity.get("exact_recovery_sha256") == recovery_record_sha
        and identity.get("policy_update_receipt_sha256") == bundle.policy_receipt_sha256
        and identity.get("harness_update_receipt_sha256")
        == bundle.harness_receipt_sha256
        and identity.get("policy_artifact_sha256") == policy_artifact.digest
        and identity.get("harness_artifact_sha256") == harness_artifact.digest
        and checkpoint.bundle_sha256 == bundle.record_sha256,
        "candidate acceptance lineage differs from store/V/W/T/U/spec evidence",
    )
    suites = record.get("critical_suites")
    _require(
        isinstance(suites, list) and len(suites) == len(REQUIRED_SUITE_KINDS),
        "candidate acceptance critical suite count differs",
    )
    for expected_suite, item in zip(expected_spec.suites, suites):
        suite = _exact(
            item,
            {"spec", "probe", "deterministic_replay_count", "critical", "passed"},
            f"{expected_suite.kind} acceptance suite",
        )
        _require(
            suite.get("spec") == expected_suite.to_record(),
            f"{expected_suite.kind} suite differs from external frozen spec",
        )
        _require(
            suite.get("deterministic_replay_count") == 2
            and suite.get("critical") is True
            and suite.get("passed") is True,
            f"{expected_suite.kind} acceptance suite was skipped or failed",
        )
        _validate_computed_probe(
            suite.get("probe"),
            suite=expected_suite,
            candidate_joint_version=candidate,
        )
    observations = record.get("active_parent_observations")
    _require(
        isinstance(observations, list)
        and len(observations) == 1 + 2 * len(REQUIRED_SUITE_KINDS),
        "active parent observation coverage is incomplete",
    )
    expected_observation = _active_observation(active)
    _require(
        all(observation == expected_observation for observation in observations),
        "active parent straddled candidate acceptance",
    )
    _require(
        record.get("decision")
        == {
            "accepted": True,
            "critical_gate_count": len(REQUIRED_SUITE_KINDS),
            "passed_critical_gate_count": len(REQUIRED_SUITE_KINDS),
            "skipped_critical_gate_count": 0,
        },
        "candidate acceptance decision contains a failed or skipped gate",
    )
    _require(
        record.get("evidence_scope") == _EVIDENCE_SCOPE,
        "candidate acceptance evidence scope differs",
    )
    return CandidateAcceptanceReportAudit(
        parent_release_id=bundle.parent_release_id,
        parent_joint_version=parent,
        candidate_joint_version=candidate,
        candidate_release_id=str(candidate_release_id),
        macro_step_id=bundle.macro_step_id,
        bundle_sha256=bundle.record_sha256,
        checkpoint_manifest_sha256=checkpoint.record_sha256,
        exact_recovery_sha256=str(recovery_record_sha),
        policy_receipt_sha256=bundle.policy_receipt_sha256,
        harness_receipt_sha256=bundle.harness_receipt_sha256,
        policy_artifact_sha256=policy_artifact.digest,
        harness_artifact_sha256=harness_artifact.digest,
        spec_sha256=expected_spec.digest,
        critical_suite_count=len(REQUIRED_SUITE_KINDS),
        record_sha256=str(record["record_sha256"]),
        live_acceptance=False,
    )


def require_live_candidate_acceptance(
    value: object,
    *,
    expected_spec: CandidateAcceptanceSpec,
    release_store: JointReleaseStore,
    joint_candidate_bundle: Mapping[str, object],
    checkpoint_manifest: str | Path,
    live_exact_recovery: object,
    require_component_files: bool = True,
) -> LiveCandidateAcceptance:
    """Consume X only after exact-type/token and all live bindings revalidate."""

    _require(
        type(value) is LiveCandidateAcceptance
        and getattr(value, "_token", None) is _LIVE_ACCEPTANCE_TOKEN,
        "production activation requires a live candidate acceptance attestation",
    )
    checkpoint = validate_production_joint_checkpoint(
        checkpoint_manifest,
        require_component_files=require_component_files,
    )
    recovery = _require_live_recovery(
        live_exact_recovery,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint=checkpoint,
    )
    audit = validate_candidate_acceptance_report(
        value.report,
        expected_spec=expected_spec,
        release_store=release_store,
        joint_candidate_bundle=joint_candidate_bundle,
        checkpoint_manifest=checkpoint_manifest,
        exact_recovery_evidence=_live_recovery_record(recovery),
        require_component_files=require_component_files,
    )
    expected_fields = {
        "record_sha256": audit.record_sha256,
        "parent_release_id": audit.parent_release_id,
        "candidate_release_id": audit.candidate_release_id,
        "parent_joint_version": audit.parent_joint_version,
        "candidate_joint_version": audit.candidate_joint_version,
        "bundle_sha256": audit.bundle_sha256,
        "checkpoint_manifest_sha256": audit.checkpoint_manifest_sha256,
        "exact_recovery_sha256": audit.exact_recovery_sha256,
        "policy_receipt_sha256": audit.policy_receipt_sha256,
        "harness_receipt_sha256": audit.harness_receipt_sha256,
        "policy_artifact_sha256": audit.policy_artifact_sha256,
        "harness_artifact_sha256": audit.harness_artifact_sha256,
        "spec_sha256": audit.spec_sha256,
    }
    _require(
        all(
            getattr(value, field) == expected
            for field, expected in expected_fields.items()
        ),
        "live candidate acceptance fields differ from its sealed report",
    )
    return value


__all__ = [
    "CANDIDATE_ACCEPTANCE_SCHEMA",
    "CANDIDATE_PROBE_SCHEMA",
    "HARNESS_CANDIDATE_ARTIFACT_SCHEMA",
    "POLICY_CANDIDATE_ARTIFACT_SCHEMA",
    "PRODUCTION_HARNESS_ARTIFACT_SCHEMA",
    "PRODUCTION_POLICY_ARTIFACT_SCHEMA",
    "REQUIRED_SUITE_KINDS",
    "CandidateAcceptanceError",
    "CandidateAcceptanceReportAudit",
    "CandidateAcceptanceSpec",
    "CandidateAcceptanceSuite",
    "CandidateProbeObservation",
    "LiveCandidateAcceptance",
    "ValidatedCandidateAcceptance",
    "build_production_candidate_artifacts",
    "require_live_candidate_acceptance",
    "run_joint_candidate_acceptance",
    "validate_candidate_acceptance_report",
]
