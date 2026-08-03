from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from jphrl.joint_release import JointReleaseStore, ReleaseManifest
from jphrl.paths import (
    repository_root,
    require_outside_repository,
)
from jphrl.trajectory.schema import JointVersion

ACTIVATION_STAGES = (
    "PREPARED",
    "QUIESCED",
    "POLICY_SYNCED",
    "HARNESS_STAGED",
    "SYNC_VERIFIED",
    "ACTIVE_POINTER_SWITCHED",
    "VERSIONS_SET",
    "POST_PUBLISH_VERIFIED",
    "RESUMED",
)

ACCEPTED_REPORT_SCHEMA = "jph.joint-activation-accepted-report.v1"
CALLBACK_RECEIPT_SCHEMA = "jph.joint-activation-callback-receipt.v1"
JOURNAL_SCHEMA = "jph.joint-activation-journal.v1"
PRODUCTION_ATTESTATION_SCHEMA = "jph.production-joint-activation.v1"
PRODUCTION_PROBE_SCHEMA = "jph.production-activation-probe.v1"
PRODUCTION_ROLLBACK_SCHEMA = "jph.production-activation-rollback-only.v1"
_PRODUCTION_POLICY_ARTIFACT_SCHEMA = "jph.production-policy-release-artifact.v2"
_PRODUCTION_HARNESS_ARTIFACT_SCHEMA = "jph.production-harness-release-artifact.v1"
_PRODUCTION_BRIDGE_TOKEN = object()
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
    "set_cookie",
    "token",
}

_EVIDENCE_SCOPE = {
    "control_plane_only": True,
    "production_activation_authorized": False,
    "production_bridge_required": False,
    "release_store_cas_exercised": True,
    "private_journal_state_machine_exercised": True,
    "callback_receipts_strictly_validated": True,
    "real_inference_weight_sync_verified": False,
    "real_harness_install_verified": False,
    "production_distributed_probe_verified": False,
}

_PRODUCTION_EVIDENCE_SCOPE = {
    "native_live_x_authorization": True,
    "native_live_w_authorization": True,
    "release_store_artifacts_revalidated": True,
    "worker_state_actively_observed": True,
    "policy_engine_version_verified": True,
    "policy_checkpoint_sha256_verified": True,
    "policy_checkpoint_identity_is_serving_parameter_sha256": True,
    "policy_dcp_recovery_lineage_separately_bound": True,
    "harness_checkpoint_sha256_verified": True,
    "harness_parameter_digest_verified": True,
    "probe_output_computed_by_framework": True,
    "static_callback_receipts_authoritative": False,
    "static_accepted_report_receipts_authoritative": False,
}

_PRODUCTION_ROLLBACK_EVIDENCE_SCOPE = {
    "rollback_only": True,
    "forward_activation_authorized": False,
    "parent_target_frozen_before_side_effects": True,
    "parent_worker_states_frozen": True,
    "candidate_target_bound_for_journal_identity_only": True,
    "probe_specs_bound": True,
    "worker_set_bound": True,
    "active_parent_reverified_after_restore": True,
}

_PRODUCTION_ROLLBACK_FIELDS = {
    "schema_version",
    "mode",
    "activation_id",
    "store_root_sha256",
    "worker_ids",
    "parent_worker_states",
    "parent_target",
    "candidate_target",
    "probe_specs",
    "probe_specs_sha256",
    "authorization_record_sha256",
    "acceptance_record_sha256",
    "exact_recovery_record_sha256",
    "expected_journal_name",
    "evidence_scope",
    "record_sha256",
}

_JOURNAL_FIELDS = {
    "schema_version",
    "activation_id",
    "parent_release_id",
    "candidate_release_id",
    "worker_ids",
    "accepted_x_report",
    "stage",
    "stage_index",
    "receipts",
    "status",
    "rollback",
    "evidence_scope",
    "record_sha256",
}
_ROLLBACK_FIELDS = {"status", "receipts", "errors"}
_CALLBACK_RECEIPT_FIELDS = {
    "schema_version",
    "scope",
    "operation",
    "success",
    "release_id",
    "joint_version",
    "observations",
    "receipt_sha256",
}
_ACCEPTED_REPORT_FIELDS = {
    "schema_version",
    "scope",
    "accepted",
    "parent_release_id",
    "candidate_release_id",
    "report_sha256",
    "receipt_sha256",
}

_TERMINAL_STATUSES = {"candidate_active", "parent_restored"}
_ROLLBACK_RECEIPT_OPERATIONS = {
    "QUIESCED": "rollback_quiesce",
    "ACTIVE_POINTER_RESTORED": "restore_active_pointer",
    "POLICY_RESTORED": "restore_policy",
    "HARNESS_RESTORED": "restore_harness",
    "PARENT_VERSIONS_SET": "set_parent_versions",
    "PARENT_PROBE_VERIFIED": "parent_probe",
    "PARENT_RESUMED": "resume_parent",
}


class JointActivationError(RuntimeError):
    """Raised after an activation fails and the parent is safely restored."""


class JointActivationRollbackError(JointActivationError):
    """Raised when an activation rollback is incomplete and remains fail closed."""


class JointActivationRecoveryRequired(JointActivationError):
    """Raised when an incomplete journal must be recovered before new work."""


class InjectedActivationFailure(JointActivationError):
    """A deterministic recoverable fault used to exercise stage boundaries."""


class InjectedActivationCrash(BaseException):
    """A simulated process death that intentionally bypasses in-process rollback."""


ActivationCallback = Callable[[str, ReleaseManifest], Mapping[str, object]]
AcceptedReportValidator = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class JointActivationCallbacks:
    quiesce: ActivationCallback
    sync_policy: ActivationCallback
    stage_harness: ActivationCallback
    verify_sync: ActivationCallback
    set_versions: ActivationCallback
    probe: ActivationCallback
    resume: ActivationCallback
    restore_policy: ActivationCallback
    restore_harness: ActivationCallback


@dataclass(frozen=True)
class JointActivationResult:
    activation_id: str
    parent_release_id: str
    candidate_release_id: str
    active_release_id: str
    stage: str
    outcome: str
    journal_path: Path
    evidence_scope: Mapping[str, bool]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointActivationError(message)


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
        raise JointActivationError(
            "activation evidence is not finite canonical JSON"
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


def _safe_error(label: str, exc: BaseException) -> str:
    """Persist only an exception type; external messages may contain secrets."""

    exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
    return f"{label}: {exception_type}"


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            _require(
                normalized not in _SECRET_FIELDS
                and not normalized.endswith(
                    ("_api_key", "_password", "_secret", "_token")
                ),
                f"credential field cannot enter activation evidence: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _exact_mapping(
    value: object,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == fields, f"{label} field set differs from schema")
    return value


def _private_atomic_write(path: Path, record: Mapping[str, object]) -> None:
    payload = dict(record)
    payload["record_sha256"] = _record_sha256(payload)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _private_exclusive_write(path: Path, record: Mapping[str, object]) -> None:
    payload = dict(record)
    payload["record_sha256"] = _record_sha256(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if path.exists():
            path.unlink()
        raise


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise JointActivationRecoveryRequired(
            f"activation journal is missing or unsafe: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointActivationRecoveryRequired(
            f"activation journal cannot be read: {path}"
        ) from exc
    _require(isinstance(value, dict), "activation journal must be an object")
    return value


class JointActivationController:
    """Fail-closed *control-plane-only* activation state machine.

    This controller proves CAS/journal/rollback ordering only.  Its injected
    callbacks and accepted-report validator are deliberately not a production
    trust boundary: even perfectly formatted callback receipts do not prove a
    real inference weight sync, Harness install, or distributed probe.  A
    caller must use :class:`ProductionJointActivationController` for a release
    that will serve or train real traffic.

    Callbacks are expected to be idempotent because a process can die after an
    external side effect but before its receipt reaches the journal.  A fresh
    process must call :meth:`recover_pending` before starting another
    activation; incomplete journals deliberately prohibit a normal resume.
    """

    def __init__(
        self,
        *,
        store: JointReleaseStore,
        journal_root: str | Path,
        project_root: str | Path,
        worker_ids: Sequence[str],
        callbacks: JointActivationCallbacks,
        validate_accepted_report: AcceptedReportValidator,
        control_plane_only: bool,
        _production_bridge_token: object | None = None,
    ) -> None:
        if control_plane_only is not True:
            raise ValueError(
                "generic activation callbacks are control-plane-only and cannot "
                "authorize production activation"
            )
        self.store = store
        if (
            _production_bridge_token is not None
            and _production_bridge_token is not _PRODUCTION_BRIDGE_TOKEN
        ):
            raise ValueError("production activation bridge token is invalid")
        self._production_bridge_authorized = (
            _production_bridge_token is _PRODUCTION_BRIDGE_TOKEN
        )
        self._journal_evidence_scope = dict(_EVIDENCE_SCOPE)
        self._journal_evidence_scope["production_bridge_required"] = (
            self._production_bridge_authorized
        )
        self.project_root = Path(project_root).expanduser().resolve()
        self._git_project_root = repository_root()
        if self.project_root != self._git_project_root:
            raise ValueError("activation project root differs from the actual checkout")
        self.journal_root = require_outside_repository(journal_root)
        if self.journal_root != self.store.activation_journal_root:
            raise ValueError(
                "activation journal root must be the release store's canonical root"
            )
        normalized_workers = tuple(worker_ids)
        if (
            not normalized_workers
            or any(
                not isinstance(worker, str) or not worker
                for worker in normalized_workers
            )
            or len(set(normalized_workers)) != len(normalized_workers)
        ):
            raise ValueError("worker IDs must be unique non-empty strings")
        self.worker_ids = normalized_workers
        self.callbacks = callbacks
        self.validate_accepted_report = validate_accepted_report
        self.journal_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.journal_root, 0o700)
        self.lock_path = self.journal_root / ".activation.lock"

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(descriptor, "r+") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            if self.lock_path.exists():
                os.chmod(self.lock_path, 0o600)

    def _journal_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.journal_root.glob("activation-*.json")))

    def _read_journal(self, path: Path) -> dict[str, object]:
        record = _read_json(path)
        self._validate_journal(record)
        return record

    def pending_journals(self) -> tuple[Path, ...]:
        """Return valid non-terminal journal paths without changing state."""

        pending: list[Path] = []
        for path in self._journal_paths():
            record = self._read_journal(path)
            if record["status"] not in _TERMINAL_STATUSES:
                pending.append(path)
        return tuple(pending)

    def _validate_accepted_report_receipt(
        self,
        value: object,
        *,
        parent_release_id: str,
        candidate_release_id: str,
        report_sha256: str | None = None,
    ) -> dict[str, object]:
        _assert_no_secrets(value, "accepted_x_report_receipt")
        receipt = _exact_mapping(
            value,
            _ACCEPTED_REPORT_FIELDS,
            "accepted X report receipt",
        )
        _require(
            receipt["schema_version"] == ACCEPTED_REPORT_SCHEMA,
            "accepted X report receipt schema differs",
        )
        _require(
            receipt["scope"] == "control_plane_only",
            "accepted X report receipt cannot authorize production activation",
        )
        _require(receipt["accepted"] is True, "X report was not accepted")
        _require(
            receipt["parent_release_id"] == parent_release_id,
            "accepted X report parent release differs",
        )
        _require(
            receipt["candidate_release_id"] == candidate_release_id,
            "accepted X report candidate release differs",
        )
        _require(
            _is_sha256(receipt["report_sha256"]),
            "accepted X report digest is invalid",
        )
        if report_sha256 is not None:
            _require(
                receipt["report_sha256"] == report_sha256,
                "accepted X report digest differs from the supplied report",
            )
        _require(
            _is_sha256(receipt["receipt_sha256"]),
            "accepted X report receipt digest is invalid",
        )
        _require(
            receipt["receipt_sha256"]
            == _sha256(
                {key: receipt[key] for key in receipt if key != "receipt_sha256"}
            ),
            "accepted X report receipt digest differs",
        )
        return dict(receipt)

    def _expected_observations(
        self,
        operation: str,
        release: ReleaseManifest,
    ) -> Mapping[str, object] | None:
        if operation in {"quiesce", "rollback_quiesce"}:
            return {"quiesced": True}
        if operation in {"sync_policy", "restore_policy"}:
            return {"policy_version": release.joint_version.policy}
        if operation in {"stage_harness", "restore_harness"}:
            return {
                "harness_controller_version": (release.joint_version.harness_controller)
            }
        if operation in {
            "verify_sync",
            "set_versions",
            "set_parent_versions",
        }:
            return None
        if operation in {"post_publish_probe", "parent_probe"}:
            return None
        if operation in {"resume", "resume_parent"}:
            return {"resumed": True}
        raise JointActivationError(
            f"unknown activation callback operation: {operation}"
        )

    def _validate_worker_versions(
        self,
        value: object,
        *,
        release: ReleaseManifest,
    ) -> None:
        _require(isinstance(value, Mapping), "worker versions must be an object")
        _require(
            set(value) == set(self.worker_ids),
            "worker version receipt is partial or has unknown workers",
        )
        expected_fields = {"policy", "harness_controller"}
        for worker_id in self.worker_ids:
            pair = _exact_mapping(
                value[worker_id],
                expected_fields,
                f"worker {worker_id} version pair",
            )
            _require(
                pair["policy"] == release.joint_version.policy,
                f"worker {worker_id} has a partial Policy/Harness version pair",
            )
            _require(
                pair["harness_controller"] == release.joint_version.harness_controller,
                f"worker {worker_id} has a partial Policy/Harness version pair",
            )

    def _validate_callback_receipt(
        self,
        value: object,
        *,
        operation: str,
        release: ReleaseManifest,
    ) -> dict[str, object]:
        _assert_no_secrets(value, f"{operation}_receipt")
        receipt = _exact_mapping(
            value,
            _CALLBACK_RECEIPT_FIELDS,
            f"{operation} receipt",
        )
        _require(
            receipt["schema_version"] == CALLBACK_RECEIPT_SCHEMA,
            f"{operation} receipt schema differs",
        )
        _require(
            receipt["scope"] == "control_plane_only",
            f"{operation} receipt cannot authorize production activation",
        )
        _require(
            receipt["operation"] == operation, f"{operation} receipt is mislabeled"
        )
        _require(receipt["success"] is True, f"{operation} did not succeed")
        _require(
            receipt["release_id"] == release.release_id,
            f"{operation} release differs",
        )
        _require(
            receipt["joint_version"] == asdict(release.joint_version),
            f"{operation} JointVersion differs",
        )
        observations = receipt["observations"]
        expected = self._expected_observations(operation, release)
        if expected is not None:
            _require(
                observations == expected,
                f"{operation} observations differ from the strict contract",
            )
        else:
            required_fields = {"worker_versions"}
            if operation in {"post_publish_probe", "parent_probe"}:
                required_fields.add("probe_passed")
            observed = _exact_mapping(
                observations,
                required_fields,
                f"{operation} observations",
            )
            if "probe_passed" in required_fields:
                _require(
                    observed["probe_passed"] is True,
                    f"{operation} did not pass",
                )
            self._validate_worker_versions(
                observed["worker_versions"],
                release=release,
            )
        _require(
            _is_sha256(receipt["receipt_sha256"]),
            f"{operation} receipt digest is invalid",
        )
        _require(
            receipt["receipt_sha256"]
            == _sha256(
                {key: receipt[key] for key in receipt if key != "receipt_sha256"}
            ),
            f"{operation} receipt digest differs",
        )
        return dict(receipt)

    def _pointer_receipt(
        self,
        *,
        operation: str,
        release: ReleaseManifest,
        pointer_action: str,
        expected_active_release_id: str,
        active_release_id: str,
    ) -> dict[str, object]:
        receipt: dict[str, object] = {
            "schema_version": CALLBACK_RECEIPT_SCHEMA,
            "scope": "control_plane_only",
            "operation": operation,
            "success": True,
            "release_id": release.release_id,
            "joint_version": asdict(release.joint_version),
            "observations": {
                "pointer_action": pointer_action,
                "expected_active_release_id": expected_active_release_id,
                "active_release_id": active_release_id,
            },
        }
        receipt["receipt_sha256"] = _sha256(receipt)
        return receipt

    def _validate_pointer_receipt(
        self,
        value: object,
        *,
        operation: str,
        release: ReleaseManifest,
        pointer_action: str,
        expected_active_release_id: str,
        active_release_id: str,
    ) -> None:
        receipt = _exact_mapping(
            value,
            _CALLBACK_RECEIPT_FIELDS,
            f"{operation} receipt",
        )
        expected = self._pointer_receipt(
            operation=operation,
            release=release,
            pointer_action=pointer_action,
            expected_active_release_id=expected_active_release_id,
            active_release_id=active_release_id,
        )
        _require(receipt == expected, f"{operation} receipt differs")

    def _validate_journal(self, record: Mapping[str, object]) -> None:
        _assert_no_secrets(record, "activation_journal")
        _exact_mapping(record, _JOURNAL_FIELDS, "activation journal")
        _require(
            record["schema_version"] == JOURNAL_SCHEMA,
            "activation journal schema differs",
        )
        _require(
            isinstance(record["activation_id"], str)
            and len(record["activation_id"]) == 32
            and all(
                character in "0123456789abcdef" for character in record["activation_id"]
            ),
            "activation ID is invalid",
        )
        parent_release_id = record["parent_release_id"]
        candidate_release_id = record["candidate_release_id"]
        _require(
            isinstance(parent_release_id, str) and parent_release_id,
            "journal parent release ID is invalid",
        )
        _require(
            isinstance(candidate_release_id, str) and candidate_release_id,
            "journal candidate release ID is invalid",
        )
        _require(
            record["worker_ids"] == list(self.worker_ids),
            "journal worker set differs from controller configuration",
        )
        self._validate_accepted_report_receipt(
            record["accepted_x_report"],
            parent_release_id=parent_release_id,
            candidate_release_id=candidate_release_id,
        )
        stage = record["stage"]
        _require(stage in ACTIVATION_STAGES, "journal activation stage is invalid")
        stage_index = ACTIVATION_STAGES.index(stage)
        _require(
            record["stage_index"] == stage_index,
            "journal stage index differs from its stage",
        )
        _require(
            record["status"]
            in {
                "activating",
                "candidate_active",
                "rollback_in_progress",
                "fail_closed",
                "parent_restored",
            },
            "journal status is invalid",
        )
        _require(
            record["evidence_scope"] == self._journal_evidence_scope,
            "journal evidence scope differs",
        )
        receipts = record["receipts"]
        _require(isinstance(receipts, Mapping), "journal receipts must be an object")
        expected_stage_keys = set(ACTIVATION_STAGES[: stage_index + 1])
        _require(
            set(receipts) == expected_stage_keys,
            "journal stages are skipped or out of order",
        )
        parent = self.store.read_manifest(parent_release_id)
        candidate = self.store.read_manifest(candidate_release_id)
        _require(
            candidate.parent_release_id == parent.release_id,
            "journal candidate is not a child of its parent",
        )
        for completed_stage in ACTIVATION_STAGES[: stage_index + 1]:
            receipt = receipts[completed_stage]
            if completed_stage == "PREPARED":
                self._validate_accepted_report_receipt(
                    receipt,
                    parent_release_id=parent.release_id,
                    candidate_release_id=candidate.release_id,
                )
            elif completed_stage == "ACTIVE_POINTER_SWITCHED":
                self._validate_pointer_receipt(
                    receipt,
                    operation="activate_pointer",
                    release=candidate,
                    pointer_action="cas",
                    expected_active_release_id=parent.release_id,
                    active_release_id=candidate.release_id,
                )
            else:
                operation, release = self._stage_contract(
                    completed_stage,
                    parent=parent,
                    candidate=candidate,
                )
                self._validate_callback_receipt(
                    receipt,
                    operation=operation,
                    release=release,
                )
        rollback = _exact_mapping(
            record["rollback"],
            _ROLLBACK_FIELDS,
            "journal rollback",
        )
        _require(
            rollback["status"] in {"not_started", "in_progress", "failed", "complete"},
            "journal rollback status is invalid",
        )
        _require(
            isinstance(rollback["receipts"], Mapping),
            "journal rollback receipts must be an object",
        )
        _require(
            set(rollback["receipts"]) <= set(_ROLLBACK_RECEIPT_OPERATIONS),
            "journal rollback has an unknown receipt",
        )
        _require(
            isinstance(rollback["errors"], list)
            and all(isinstance(error, str) and error for error in rollback["errors"]),
            "journal rollback errors are invalid",
        )
        self._validate_rollback_receipts(
            rollback["receipts"],
            parent=parent,
            candidate=candidate,
        )
        if record["status"] == "candidate_active":
            _require(stage == "RESUMED", "candidate-active journal is incomplete")
            _require(
                rollback["status"] == "not_started",
                "candidate-active journal unexpectedly contains rollback state",
            )
        if record["status"] == "parent_restored":
            _require(
                rollback["status"] == "complete",
                "parent-restored journal has incomplete rollback",
            )
            _require(
                set(rollback["receipts"]) == set(_ROLLBACK_RECEIPT_OPERATIONS),
                "parent-restored journal skipped rollback operations",
            )
        _require(
            _is_sha256(record["record_sha256"]),
            "activation journal digest is invalid",
        )
        _require(
            record["record_sha256"] == _record_sha256(record),
            "activation journal digest differs",
        )

    def _validate_rollback_receipts(
        self,
        receipts: Mapping[str, object],
        *,
        parent: ReleaseManifest,
        candidate: ReleaseManifest,
    ) -> None:
        for name, receipt in receipts.items():
            operation = _ROLLBACK_RECEIPT_OPERATIONS[name]
            if name == "ACTIVE_POINTER_RESTORED":
                possible_receipts = (
                    self._pointer_receipt(
                        operation=operation,
                        release=parent,
                        pointer_action="cas",
                        expected_active_release_id=candidate.release_id,
                        active_release_id=parent.release_id,
                    ),
                    self._pointer_receipt(
                        operation=operation,
                        release=parent,
                        pointer_action="confirmed",
                        expected_active_release_id=parent.release_id,
                        active_release_id=parent.release_id,
                    ),
                )
                _require(
                    receipt in possible_receipts,
                    "restore_active_pointer receipt differs",
                )
                continue
            release = candidate if name == "QUIESCED" else parent
            self._validate_callback_receipt(
                receipt,
                operation=operation,
                release=release,
            )

    @staticmethod
    def _stage_contract(
        stage: str,
        *,
        parent: ReleaseManifest,
        candidate: ReleaseManifest,
    ) -> tuple[str, ReleaseManifest]:
        contracts = {
            "QUIESCED": ("quiesce", parent),
            "POLICY_SYNCED": ("sync_policy", candidate),
            "HARNESS_STAGED": ("stage_harness", candidate),
            "SYNC_VERIFIED": ("verify_sync", candidate),
            "VERSIONS_SET": ("set_versions", candidate),
            "POST_PUBLISH_VERIFIED": ("post_publish_probe", candidate),
            "RESUMED": ("resume", candidate),
        }
        try:
            return contracts[stage]
        except KeyError as exc:
            raise JointActivationError(
                f"stage {stage} has no callback contract"
            ) from exc

    def _invoke(
        self,
        callback: ActivationCallback,
        *,
        operation: str,
        release: ReleaseManifest,
    ) -> dict[str, object]:
        return self._validate_callback_receipt(
            callback(operation, release),
            operation=operation,
            release=release,
        )

    def _write_journal(self, path: Path, record: dict[str, object]) -> None:
        record["record_sha256"] = _record_sha256(record)
        self._validate_journal(record)
        _private_atomic_write(path, record)

    def _advance(
        self,
        path: Path,
        record: dict[str, object],
        *,
        next_stage: str,
        receipt: Mapping[str, object],
    ) -> None:
        current_index = ACTIVATION_STAGES.index(record["stage"])
        _require(
            current_index + 1 < len(ACTIVATION_STAGES)
            and ACTIVATION_STAGES[current_index + 1] == next_stage,
            "activation journal transition attempted to skip a stage",
        )
        receipts = record["receipts"]
        _require(isinstance(receipts, dict), "activation receipt journal is immutable")
        receipts[next_stage] = dict(receipt)
        record["stage"] = next_stage
        record["stage_index"] = current_index + 1
        if next_stage == "RESUMED":
            record["status"] = "candidate_active"
        self._write_journal(path, record)

    @staticmethod
    def _maybe_fault(
        stage: str,
        *,
        fault_after_stage: str | None,
        fault_mode: str,
    ) -> None:
        if stage != fault_after_stage:
            return
        if fault_mode == "crash":
            raise InjectedActivationCrash(stage)
        raise InjectedActivationFailure(stage)

    def activate(
        self,
        *,
        accepted_x_report: Mapping[str, object],
        parent_release_id: str,
        candidate_release_id: str,
        fault_after_stage: str | None = None,
        fault_mode: str = "raise",
        _activation_id: str | None = None,
    ) -> JointActivationResult:
        """Activate a pre-staged child release or restore its parent on error."""

        if fault_after_stage is not None and fault_after_stage not in ACTIVATION_STAGES:
            raise ValueError(f"unknown activation stage: {fault_after_stage}")
        if fault_mode not in {"raise", "crash"}:
            raise ValueError("activation fault mode must be raise or crash")
        if _activation_id is not None:
            _require(
                self._production_bridge_authorized,
                "only the production bridge may reserve an activation ID",
            )
            _require(
                len(_activation_id) == 32
                and all(
                    character in "0123456789abcdef" for character in _activation_id
                ),
                "reserved production activation ID is invalid",
            )
        with self.store.activation_lease(), self._exclusive_lock():
            pending = self.pending_journals()
            if pending:
                raise JointActivationRecoveryRequired(
                    "incomplete activation journal must be recovered before activation"
                )
            active = self.store.read_active()
            _require(active is not None, "release store has no active parent")
            _require(
                active.release_id == parent_release_id,
                "active release differs from the accepted report parent",
            )
            parent = self.store.read_manifest(parent_release_id)
            candidate = self.store.read_manifest(candidate_release_id)
            _require(
                candidate.parent_release_id == parent.release_id,
                "candidate was staged from a different parent",
            )
            policy_artifact, harness_artifact = self.store.read_artifacts(
                candidate.release_id
            )
            policy_schema = policy_artifact.payload.get("schema_version")
            harness_schema = harness_artifact.payload.get("schema_version")
            is_production_candidate = (
                policy_schema == _PRODUCTION_POLICY_ARTIFACT_SCHEMA
                or harness_schema == _PRODUCTION_HARNESS_ARTIFACT_SCHEMA
            )
            if is_production_candidate:
                _require(
                    policy_schema == _PRODUCTION_POLICY_ARTIFACT_SCHEMA
                    and harness_schema == _PRODUCTION_HARNESS_ARTIFACT_SCHEMA,
                    "candidate has a partial production artifact pair",
                )
                _require(
                    self._production_bridge_authorized,
                    "control-plane receipts cannot activate production artifacts",
                )
            _require(
                isinstance(accepted_x_report, Mapping),
                "accepted X report must be an object",
            )
            validated_report = self._validate_accepted_report_receipt(
                self.validate_accepted_report(accepted_x_report),
                parent_release_id=parent.release_id,
                candidate_release_id=candidate.release_id,
                report_sha256=_sha256(accepted_x_report),
            )
            activation_id = _activation_id or uuid.uuid4().hex
            journal_path = self.journal_root / f"activation-{activation_id}.json"
            record: dict[str, object] = {
                "schema_version": JOURNAL_SCHEMA,
                "activation_id": activation_id,
                "parent_release_id": parent.release_id,
                "candidate_release_id": candidate.release_id,
                "worker_ids": list(self.worker_ids),
                "accepted_x_report": validated_report,
                "stage": "PREPARED",
                "stage_index": 0,
                "receipts": {"PREPARED": validated_report},
                "status": "activating",
                "rollback": {
                    "status": "not_started",
                    "receipts": {},
                    "errors": [],
                },
                "evidence_scope": dict(self._journal_evidence_scope),
                "record_sha256": "",
            }
            record["record_sha256"] = _record_sha256(record)
            self._validate_journal(record)
            _private_exclusive_write(journal_path, record)
            try:
                self._maybe_fault(
                    "PREPARED",
                    fault_after_stage=fault_after_stage,
                    fault_mode=fault_mode,
                )
                transitions = (
                    (
                        "QUIESCED",
                        self.callbacks.quiesce,
                        "quiesce",
                        parent,
                    ),
                    (
                        "POLICY_SYNCED",
                        self.callbacks.sync_policy,
                        "sync_policy",
                        candidate,
                    ),
                    (
                        "HARNESS_STAGED",
                        self.callbacks.stage_harness,
                        "stage_harness",
                        candidate,
                    ),
                    (
                        "SYNC_VERIFIED",
                        self.callbacks.verify_sync,
                        "verify_sync",
                        candidate,
                    ),
                )
                for stage, callback, operation, release in transitions:
                    receipt = self._invoke(
                        callback,
                        operation=operation,
                        release=release,
                    )
                    self._advance(
                        journal_path,
                        record,
                        next_stage=stage,
                        receipt=receipt,
                    )
                    self._maybe_fault(
                        stage,
                        fault_after_stage=fault_after_stage,
                        fault_mode=fault_mode,
                    )

                switched = self.store.activate(
                    release_id=candidate.release_id,
                    expected_active_release_id=parent.release_id,
                )
                _require(
                    switched.release_id == candidate.release_id,
                    "active pointer switch returned a different release",
                )
                pointer_receipt = self._pointer_receipt(
                    operation="activate_pointer",
                    release=candidate,
                    pointer_action="cas",
                    expected_active_release_id=parent.release_id,
                    active_release_id=candidate.release_id,
                )
                self._advance(
                    journal_path,
                    record,
                    next_stage="ACTIVE_POINTER_SWITCHED",
                    receipt=pointer_receipt,
                )
                self._maybe_fault(
                    "ACTIVE_POINTER_SWITCHED",
                    fault_after_stage=fault_after_stage,
                    fault_mode=fault_mode,
                )

                final_transitions = (
                    (
                        "VERSIONS_SET",
                        self.callbacks.set_versions,
                        "set_versions",
                    ),
                    (
                        "POST_PUBLISH_VERIFIED",
                        self.callbacks.probe,
                        "post_publish_probe",
                    ),
                    ("RESUMED", self.callbacks.resume, "resume"),
                )
                for stage, callback, operation in final_transitions:
                    receipt = self._invoke(
                        callback,
                        operation=operation,
                        release=candidate,
                    )
                    self._advance(
                        journal_path,
                        record,
                        next_stage=stage,
                        receipt=receipt,
                    )
                    self._maybe_fault(
                        stage,
                        fault_after_stage=fault_after_stage,
                        fault_mode=fault_mode,
                    )
            except Exception as activation_error:
                try:
                    self._rollback(
                        journal_path,
                        record,
                        parent=parent,
                        candidate=candidate,
                    )
                except JointActivationRollbackError as rollback_error:
                    raise JointActivationRollbackError(
                        "activation failed and parent recovery is incomplete; "
                        f"journal={journal_path}: {activation_error}"
                    ) from rollback_error
                raise JointActivationError(
                    "activation failed; the parent pair was restored and resumed; "
                    f"journal={journal_path}: {activation_error}"
                ) from activation_error

            return JointActivationResult(
                activation_id=activation_id,
                parent_release_id=parent.release_id,
                candidate_release_id=candidate.release_id,
                active_release_id=candidate.release_id,
                stage="RESUMED",
                outcome="candidate_active",
                journal_path=journal_path,
                evidence_scope=dict(self._journal_evidence_scope),
            )

    def _append_rollback_receipt(
        self,
        path: Path,
        record: dict[str, object],
        *,
        name: str,
        receipt: Mapping[str, object],
    ) -> None:
        rollback = record["rollback"]
        _require(isinstance(rollback, dict), "rollback journal is immutable")
        receipts = rollback["receipts"]
        _require(isinstance(receipts, dict), "rollback receipts are immutable")
        receipts[name] = dict(receipt)
        self._write_journal(path, record)

    def _rollback(
        self,
        path: Path,
        record: dict[str, object],
        *,
        parent: ReleaseManifest,
        candidate: ReleaseManifest,
    ) -> JointActivationResult:
        rollback = record["rollback"]
        _require(isinstance(rollback, dict), "rollback journal is immutable")
        receipts = rollback["receipts"]
        _require(isinstance(receipts, dict), "rollback receipts are immutable")
        rollback["status"] = "in_progress"
        rollback["errors"] = []
        record["status"] = "rollback_in_progress"
        self._write_journal(path, record)

        errors: list[str] = []
        if "QUIESCED" not in receipts:
            try:
                receipt = self._invoke(
                    self.callbacks.quiesce,
                    operation="rollback_quiesce",
                    release=candidate,
                )
                self._append_rollback_receipt(
                    path,
                    record,
                    name="QUIESCED",
                    receipt=receipt,
                )
            except Exception as exc:  # noqa: BLE001 - recovery must persist failure
                errors.append(_safe_error("rollback quiesce failed", exc))

        if not errors and "ACTIVE_POINTER_RESTORED" not in receipts:
            try:
                active = self.store.read_active()
                _require(active is not None, "release store lost its active pointer")
                if active.release_id == candidate.release_id:
                    restored = self.store.rollback(
                        target_release_id=parent.release_id,
                        expected_active_release_id=candidate.release_id,
                    )
                    _require(
                        restored.release_id == parent.release_id,
                        "rollback returned a different parent",
                    )
                    pointer_action = "cas"
                    expected_active_release_id = candidate.release_id
                elif active.release_id != parent.release_id:
                    raise JointActivationError(
                        "active pointer is neither activation parent nor candidate"
                    )
                else:
                    pointer_action = "confirmed"
                    expected_active_release_id = parent.release_id
                pointer_receipt = self._pointer_receipt(
                    operation="restore_active_pointer",
                    release=parent,
                    pointer_action=pointer_action,
                    expected_active_release_id=expected_active_release_id,
                    active_release_id=parent.release_id,
                )
                self._append_rollback_receipt(
                    path,
                    record,
                    name="ACTIVE_POINTER_RESTORED",
                    receipt=pointer_receipt,
                )
            except Exception as exc:  # noqa: BLE001 - recovery must persist failure
                errors.append(_safe_error("active pointer rollback failed", exc))

        if not errors:
            restore_operations = (
                (
                    "POLICY_RESTORED",
                    self.callbacks.restore_policy,
                    "restore_policy",
                ),
                (
                    "HARNESS_RESTORED",
                    self.callbacks.restore_harness,
                    "restore_harness",
                ),
            )
            for name, callback, operation in restore_operations:
                if name in receipts:
                    continue
                try:
                    receipt = self._invoke(
                        callback,
                        operation=operation,
                        release=parent,
                    )
                    self._append_rollback_receipt(
                        path,
                        record,
                        name=name,
                        receipt=receipt,
                    )
                except Exception as exc:  # noqa: BLE001 - try both component restores
                    errors.append(_safe_error(f"{operation} failed", exc))

        if not errors and "PARENT_VERSIONS_SET" not in receipts:
            try:
                receipt = self._invoke(
                    self.callbacks.set_versions,
                    operation="set_parent_versions",
                    release=parent,
                )
                self._append_rollback_receipt(
                    path,
                    record,
                    name="PARENT_VERSIONS_SET",
                    receipt=receipt,
                )
            except Exception as exc:  # noqa: BLE001 - recovery must persist failure
                errors.append(_safe_error("set_parent_versions failed", exc))

        if not errors and "PARENT_PROBE_VERIFIED" not in receipts:
            try:
                receipt = self._invoke(
                    self.callbacks.probe,
                    operation="parent_probe",
                    release=parent,
                )
                self._append_rollback_receipt(
                    path,
                    record,
                    name="PARENT_PROBE_VERIFIED",
                    receipt=receipt,
                )
            except Exception as exc:  # noqa: BLE001 - recovery must persist failure
                errors.append(_safe_error("parent_probe failed", exc))

        if not errors and "PARENT_RESUMED" not in receipts:
            try:
                receipt = self._invoke(
                    self.callbacks.resume,
                    operation="resume_parent",
                    release=parent,
                )
                self._append_rollback_receipt(
                    path,
                    record,
                    name="PARENT_RESUMED",
                    receipt=receipt,
                )
            except Exception as exc:  # noqa: BLE001 - recovery must persist failure
                errors.append(_safe_error("resume_parent failed", exc))

        if errors:
            rollback["status"] = "failed"
            rollback["errors"] = errors
            record["status"] = "fail_closed"
            self._write_journal(path, record)
            raise JointActivationRollbackError(
                "parent recovery is incomplete and workers remain fail closed: "
                + "; ".join(errors)
            )

        rollback["status"] = "complete"
        rollback["errors"] = []
        record["status"] = "parent_restored"
        self._write_journal(path, record)
        return JointActivationResult(
            activation_id=str(record["activation_id"]),
            parent_release_id=parent.release_id,
            candidate_release_id=candidate.release_id,
            active_release_id=parent.release_id,
            stage=str(record["stage"]),
            outcome="parent_restored",
            journal_path=path,
            evidence_scope=dict(self._journal_evidence_scope),
        )

    def recover_pending(self) -> JointActivationResult:
        """Roll an interrupted activation back and resume only after parent probe."""

        with self.store.activation_lease(), self._exclusive_lock():
            pending = self.pending_journals()
            if not pending:
                raise JointActivationRecoveryRequired(
                    "there is no incomplete activation journal to recover"
                )
            if len(pending) != 1:
                raise JointActivationRecoveryRequired(
                    "multiple incomplete activation journals require operator review"
                )
            path = pending[0]
            record = self._read_journal(path)
            parent = self.store.read_manifest(str(record["parent_release_id"]))
            candidate = self.store.read_manifest(str(record["candidate_release_id"]))
            return self._rollback(
                path,
                record,
                parent=parent,
                candidate=candidate,
            )


def accepted_report_receipt(
    *,
    accepted_x_report: Mapping[str, object],
    parent_release_id: str,
    candidate_release_id: str,
    control_plane_only: bool,
) -> dict[str, object]:
    """Build a non-authoritative receipt for control-plane state-machine tests.

    This helper cannot consume X's live token and its output is explicitly
    scoped away from :func:`authorize_production_activation`.
    """

    if control_plane_only is not True:
        raise JointActivationError(
            "static accepted-report receipts cannot authorize production activation"
        )
    _assert_no_secrets(accepted_x_report, "accepted_x_report")
    receipt: dict[str, object] = {
        "schema_version": ACCEPTED_REPORT_SCHEMA,
        "scope": "control_plane_only",
        "accepted": True,
        "parent_release_id": parent_release_id,
        "candidate_release_id": candidate_release_id,
        "report_sha256": _sha256(accepted_x_report),
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    return receipt


def callback_receipt(
    *,
    operation: str,
    release: ReleaseManifest,
    observations: Mapping[str, object],
    control_plane_only: bool,
) -> dict[str, object]:
    """Hash a non-authoritative callback receipt for control-plane tests."""

    if control_plane_only is not True:
        raise JointActivationError(
            "static callback receipts cannot authorize production activation"
        )
    _assert_no_secrets(observations, f"{operation}_observations")
    receipt: dict[str, object] = {
        "schema_version": CALLBACK_RECEIPT_SCHEMA,
        "scope": "control_plane_only",
        "operation": operation,
        "success": True,
        "release_id": release.release_id,
        "joint_version": asdict(release.joint_version),
        "observations": dict(observations),
    }
    receipt["receipt_sha256"] = _sha256(receipt)
    return receipt


@dataclass(frozen=True)
class ProductionReleaseIdentity:
    """Release identity that does not invent bootstrap checkpoint evidence."""

    release_id: str
    joint_version: JointVersion

    def validate(self) -> None:
        _require(
            isinstance(self.release_id, str) and bool(self.release_id),
            "production release identity is missing",
        )

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {
            "release_id": self.release_id,
            "joint_version": asdict(self.joint_version),
            "joint_version_id": self.joint_version.version_id,
        }


@dataclass(frozen=True)
class ProductionReleaseTarget:
    """Framework-owned expected state for one content-addressed release.

    In production Y, ``policy_checkpoint_sha256`` identifies the canonical
    serving-parameter set actively observed from inference.  The distinct DCP
    recovery manifest remains bound by the v2 Policy release artifact.
    """

    release_id: str
    joint_version: JointVersion
    policy_engine_version: int
    policy_checkpoint_sha256: str
    harness_checkpoint_sha256: str
    harness_parameter_digest: str

    def validate(self) -> None:
        _require(
            isinstance(self.release_id, str) and bool(self.release_id),
            "production release target ID is missing",
        )
        _require(
            type(self.policy_engine_version) is int and self.policy_engine_version >= 0,
            "production Policy engine version is invalid",
        )
        for label, digest in (
            ("Policy checkpoint", self.policy_checkpoint_sha256),
            ("Harness checkpoint", self.harness_checkpoint_sha256),
            ("Harness parameter", self.harness_parameter_digest),
        ):
            _require(_is_sha256(digest), f"production {label} digest is invalid")

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {
            "release_id": self.release_id,
            "joint_version": asdict(self.joint_version),
            "joint_version_id": self.joint_version.version_id,
            "policy_engine_version": self.policy_engine_version,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "harness_checkpoint_sha256": self.harness_checkpoint_sha256,
            "harness_parameter_digest": self.harness_parameter_digest,
        }

    @classmethod
    def from_record(cls, value: object) -> ProductionReleaseTarget:
        record = _exact_mapping(
            value,
            {
                "release_id",
                "joint_version",
                "joint_version_id",
                "policy_engine_version",
                "policy_checkpoint_sha256",
                "harness_checkpoint_sha256",
                "harness_parameter_digest",
            },
            "production release target",
        )
        joint_version_record = _exact_mapping(
            record["joint_version"],
            set(JointVersion.__dataclass_fields__),
            "production target JointVersion",
        )
        try:
            joint_version = JointVersion(**dict(joint_version_record))
        except TypeError as exc:
            raise JointActivationError(
                "production target JointVersion is invalid"
            ) from exc
        _require(
            record["joint_version_id"] == joint_version.version_id,
            "production target JointVersion digest differs",
        )
        target = cls(
            release_id=str(record["release_id"]),
            joint_version=joint_version,
            policy_engine_version=record["policy_engine_version"],
            policy_checkpoint_sha256=str(record["policy_checkpoint_sha256"]),
            harness_checkpoint_sha256=str(record["harness_checkpoint_sha256"]),
            harness_parameter_digest=str(record["harness_parameter_digest"]),
        )
        target.validate()
        return target


_PRODUCTION_AUTHORIZATION_TOKEN = object()


@dataclass(frozen=True, init=False)
class ProductionActivationAuthorization:
    """Unforgeable-in-process authorization issued from native live W/X gates."""

    parent: ProductionReleaseTarget
    candidate: ProductionReleaseTarget
    acceptance_record_sha256: str
    exact_recovery_record_sha256: str
    probe_specs_sha256: str
    _token: object

    @classmethod
    def _create(
        cls,
        *,
        parent: ProductionReleaseTarget,
        candidate: ProductionReleaseTarget,
        acceptance_record_sha256: str,
        exact_recovery_record_sha256: str,
        probe_specs_sha256: str,
        token: object,
    ) -> ProductionActivationAuthorization:
        _require(
            token is _PRODUCTION_AUTHORIZATION_TOKEN,
            "production authorization token is invalid",
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "parent", parent)
        object.__setattr__(instance, "candidate", candidate)
        object.__setattr__(
            instance,
            "acceptance_record_sha256",
            acceptance_record_sha256,
        )
        object.__setattr__(
            instance,
            "exact_recovery_record_sha256",
            exact_recovery_record_sha256,
        )
        object.__setattr__(instance, "probe_specs_sha256", probe_specs_sha256)
        object.__setattr__(instance, "_token", token)
        return instance

    def to_record(self) -> dict[str, object]:
        require_production_activation_authorization(self)
        return {
            "parent": self.parent.to_record(),
            "candidate": self.candidate.to_record(),
            "acceptance_record_sha256": self.acceptance_record_sha256,
            "exact_recovery_record_sha256": self.exact_recovery_record_sha256,
            "probe_specs_sha256": self.probe_specs_sha256,
        }


def require_production_activation_authorization(
    value: object,
    *,
    store: JointReleaseStore | None = None,
) -> ProductionActivationAuthorization:
    """Reject persisted dictionaries and user-constructed authorization objects."""

    _require(
        type(value) is ProductionActivationAuthorization
        and value._token is _PRODUCTION_AUTHORIZATION_TOKEN,
        "production activation requires a native live W/X authorization",
    )
    value.parent.validate()
    value.candidate.validate()
    _require(
        value.candidate.policy_engine_version == value.parent.policy_engine_version + 1,
        "production candidate Policy engine version is not the next version",
    )
    _require(
        _is_sha256(value.acceptance_record_sha256)
        and _is_sha256(value.exact_recovery_record_sha256)
        and _is_sha256(value.probe_specs_sha256),
        "production live authorization digests are invalid",
    )
    if store is not None:
        active = store.read_active()
        _require(active is not None, "production release store has no active parent")
        parent = store.read_manifest(value.parent.release_id)
        candidate = store.read_manifest(value.candidate.release_id)
        _require(
            active == parent
            and parent.joint_version == value.parent.joint_version
            and candidate.joint_version == value.candidate.joint_version
            and candidate.parent_release_id == parent.release_id,
            "production live authorization differs from the release store",
        )
    return value


@dataclass(frozen=True)
class ProductionWorkerState:
    """Raw worker state sampled by the controller after a side effect.

    ``policy_checkpoint_sha256`` has the same serving-parameter semantics as
    :class:`ProductionReleaseTarget`; it is not the training DCP manifest.
    """

    worker_id: str
    lifecycle_phase: str
    active_release_id: str
    joint_version: JointVersion
    policy_engine_version: int
    policy_checkpoint_sha256: str
    harness_controller_version: str
    harness_checkpoint_sha256: str
    harness_parameter_digest: str

    def validate(self) -> None:
        _require(
            isinstance(self.worker_id, str) and bool(self.worker_id),
            "production worker ID is missing",
        )
        _require(
            self.lifecycle_phase in {"quiesced", "serving"},
            "production worker lifecycle phase is invalid",
        )
        _require(
            isinstance(self.active_release_id, str) and bool(self.active_release_id),
            "production worker active release is missing",
        )
        _require(
            type(self.policy_engine_version) is int and self.policy_engine_version >= 0,
            "production worker Policy engine version is invalid",
        )
        for label, digest in (
            ("Policy checkpoint", self.policy_checkpoint_sha256),
            ("Harness checkpoint", self.harness_checkpoint_sha256),
            ("Harness parameter", self.harness_parameter_digest),
        ):
            _require(_is_sha256(digest), f"production worker {label} hash is invalid")

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {
            "worker_id": self.worker_id,
            "lifecycle_phase": self.lifecycle_phase,
            "active_release_id": self.active_release_id,
            "joint_version": asdict(self.joint_version),
            "joint_version_id": self.joint_version.version_id,
            "policy_engine_version": self.policy_engine_version,
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "harness_controller_version": self.harness_controller_version,
            "harness_checkpoint_sha256": self.harness_checkpoint_sha256,
            "harness_parameter_digest": self.harness_parameter_digest,
        }

    @classmethod
    def from_record(cls, value: object) -> ProductionWorkerState:
        record = _exact_mapping(
            value,
            {
                "worker_id",
                "lifecycle_phase",
                "active_release_id",
                "joint_version",
                "joint_version_id",
                "policy_engine_version",
                "policy_checkpoint_sha256",
                "harness_controller_version",
                "harness_checkpoint_sha256",
                "harness_parameter_digest",
            },
            "production worker state",
        )
        joint_version_record = _exact_mapping(
            record["joint_version"],
            set(JointVersion.__dataclass_fields__),
            "production worker JointVersion",
        )
        try:
            joint_version = JointVersion(**dict(joint_version_record))
        except TypeError as exc:
            raise JointActivationError(
                "production worker JointVersion is invalid"
            ) from exc
        _require(
            record["joint_version_id"] == joint_version.version_id,
            "production worker JointVersion digest differs",
        )
        state = cls(
            worker_id=str(record["worker_id"]),
            lifecycle_phase=str(record["lifecycle_phase"]),
            active_release_id=str(record["active_release_id"]),
            joint_version=joint_version,
            policy_engine_version=record["policy_engine_version"],
            policy_checkpoint_sha256=str(record["policy_checkpoint_sha256"]),
            harness_controller_version=str(record["harness_controller_version"]),
            harness_checkpoint_sha256=str(record["harness_checkpoint_sha256"]),
            harness_parameter_digest=str(record["harness_parameter_digest"]),
        )
        state.validate()
        return state


@dataclass(frozen=True)
class ProductionProbeSpec:
    """Frozen bytes and expected output digest for an active worker probe."""

    probe_id: str
    fixture: bytes
    fixture_sha256: str
    expected_output_sha256: str

    def validate(self) -> None:
        _require(
            isinstance(self.probe_id, str) and bool(self.probe_id),
            "production probe ID is missing",
        )
        _require(
            type(self.fixture) is bytes and bool(self.fixture),
            "production probe fixture must be non-empty bytes",
        )
        _require(
            self.fixture_sha256 == hashlib.sha256(self.fixture).hexdigest(),
            "production probe fixture digest differs from its bytes",
        )
        _require(
            _is_sha256(self.expected_output_sha256),
            "production probe expected output digest is invalid",
        )

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": PRODUCTION_PROBE_SCHEMA,
            "probe_id": self.probe_id,
            "fixture_sha256": self.fixture_sha256,
            "expected_output_sha256": self.expected_output_sha256,
        }


class ProductionActivationWorker(ABC):
    """Typed side-effect boundary for one real rollout worker.

    Mutation methods return ``None`` and therefore cannot mint evidence.  The
    controller calls :meth:`read_state` after each mutation, validates the
    exact typed state against content-addressed release targets, and computes
    probe output hashes itself.
    """

    @property
    @abstractmethod
    def worker_id(self) -> str:
        """Return the stable orchestrator worker ID."""

    @abstractmethod
    def quiesce(self) -> None:
        """Stop admitting new work and drain in-flight work."""

    @abstractmethod
    def install_policy(self, target: ProductionReleaseTarget) -> None:
        """Synchronize the exact Policy checkpoint in ``target``."""

    @abstractmethod
    def install_harness(self, target: ProductionReleaseTarget) -> None:
        """Install the exact Harness checkpoint in ``target``."""

    @abstractmethod
    def bind_release(self, target: ProductionReleaseTarget) -> None:
        """Publish the complete JointVersion to this quiesced worker."""

    @abstractmethod
    def run_probe(self, fixture: bytes) -> bytes:
        """Return raw deterministic probe output, never a passed boolean."""

    @abstractmethod
    def resume(self) -> None:
        """Resume admission after a framework-verified probe."""

    @abstractmethod
    def read_state(self) -> ProductionWorkerState:
        """Read current engine/version/checkpoint state from the live worker."""


def _freeze_parent_worker_target(
    *,
    workers: Sequence[ProductionActivationWorker],
    parent: ProductionReleaseIdentity,
    policy_engine_version: int,
) -> ProductionReleaseTarget:
    """Measure a bootstrap parent instead of inventing release artifact hashes."""

    parent.validate()
    states: list[ProductionWorkerState] = []
    for worker in workers:
        _require(
            isinstance(worker, ProductionActivationWorker),
            "production parent freeze requires typed workers",
        )
        state = worker.read_state()
        _require(
            type(state) is ProductionWorkerState,
            f"production worker {worker.worker_id} returned an untyped state mapping",
        )
        state.validate()
        _require(
            state.worker_id == worker.worker_id
            and state.lifecycle_phase == "serving"
            and state.active_release_id == parent.release_id
            and state.joint_version == parent.joint_version
            and state.policy_engine_version == policy_engine_version
            and state.harness_controller_version
            == parent.joint_version.harness_controller,
            f"production worker {worker.worker_id} is not serving the authorized parent",
        )
        states.append(state)
    _require(bool(states), "production parent freeze has no workers")
    consensus = {
        (
            state.policy_checkpoint_sha256,
            state.harness_checkpoint_sha256,
            state.harness_parameter_digest,
        )
        for state in states
    }
    _require(
        len(consensus) == 1,
        "production workers disagree on the parent checkpoint hashes",
    )
    policy_sha256, harness_sha256, harness_parameter_digest = consensus.pop()
    return ProductionReleaseTarget(
        release_id=parent.release_id,
        joint_version=parent.joint_version,
        policy_engine_version=policy_engine_version,
        policy_checkpoint_sha256=policy_sha256,
        harness_checkpoint_sha256=harness_sha256,
        harness_parameter_digest=harness_parameter_digest,
    )


def _probe_specs_sha256(
    probes: Mapping[str, ProductionProbeSpec],
    *,
    release_ids: set[str],
) -> str:
    _require(
        set(probes) == release_ids,
        "production probes must cover exactly the authorized release pair",
    )
    records: dict[str, object] = {}
    for release_id in sorted(release_ids):
        spec = probes[release_id]
        _require(
            type(spec) is ProductionProbeSpec,
            "production probe must use the native typed specification",
        )
        records[release_id] = spec.to_record()
    return _sha256(records)


def authorize_production_activation(
    *,
    store: JointReleaseStore,
    live_candidate_acceptance: object,
    live_exact_recovery: object,
    expected_acceptance_spec: object,
    joint_candidate_bundle: Mapping[str, object],
    checkpoint_manifest: str | Path,
    workers: Sequence[ProductionActivationWorker],
    probes: Mapping[str, ProductionProbeSpec],
    require_component_files: bool = True,
) -> ProductionActivationAuthorization:
    """Bind live W/X capabilities, release objects, and observed parent state."""

    try:
        from .candidate_acceptance import (
            PRODUCTION_HARNESS_ARTIFACT_SCHEMA,
            PRODUCTION_POLICY_ARTIFACT_SCHEMA,
            require_live_candidate_acceptance,
        )
        from .production_checkpoint import require_live_exact_joint_recovery
    except (
        ImportError,
        ModuleNotFoundError,
    ) as exc:  # pragma: no cover - install guard
        raise JointActivationError(
            "native live W/X production gates are unavailable"
        ) from exc

    acceptance = require_live_candidate_acceptance(
        live_candidate_acceptance,
        expected_spec=expected_acceptance_spec,
        release_store=store,
        joint_candidate_bundle=joint_candidate_bundle,
        checkpoint_manifest=checkpoint_manifest,
        live_exact_recovery=live_exact_recovery,
        require_component_files=require_component_files,
    )
    recovery = require_live_exact_joint_recovery(
        live_exact_recovery,
        manifest=checkpoint_manifest,
    )
    _require(
        acceptance.exact_recovery_sha256 == recovery.record_sha256
        and acceptance.candidate_joint_version == recovery.candidate_joint_version,
        "live X acceptance differs from live W exact recovery",
    )
    active = store.read_active()
    _require(active is not None, "production release store has no active parent")
    parent = store.read_manifest(acceptance.parent_release_id)
    candidate = store.read_manifest(acceptance.candidate_release_id)
    _require(
        active == parent
        and parent.joint_version == acceptance.parent_joint_version
        and candidate.parent_release_id == parent.release_id
        and candidate.joint_version == acceptance.candidate_joint_version,
        "live X acceptance differs from the release store lineage",
    )
    policy_artifact, harness_artifact = store.read_artifacts(candidate.release_id)
    _require(
        policy_artifact.digest == acceptance.policy_artifact_sha256
        and harness_artifact.digest == acceptance.harness_artifact_sha256,
        "live X acceptance differs from content-addressed release artifacts",
    )
    policy_payload = _exact_mapping(
        policy_artifact.payload,
        {
            "schema_version",
            "candidate_joint_version_id",
            "joint_candidate_bundle_sha256",
            "production_checkpoint_manifest_sha256",
            "policy_update_receipt_sha256",
            "policy_checkpoint_manifest_sha256",
            "policy_engine_version",
            "policy_serving_export_manifest_sha256",
            "policy_serving_parameter_sha256",
            "policy_serving_lineage_record_sha256",
        },
        "production Policy release artifact",
    )
    harness_payload = _exact_mapping(
        harness_artifact.payload,
        {
            "schema_version",
            "candidate_joint_version_id",
            "joint_candidate_bundle_sha256",
            "production_checkpoint_manifest_sha256",
            "harness_update_receipt_sha256",
            "harness_checkpoint_sha256",
            "harness_parameter_sha256",
        },
        "production Harness release artifact",
    )
    candidate_joint_version_id = candidate.joint_version.version_id
    _require(
        policy_payload["schema_version"] == PRODUCTION_POLICY_ARTIFACT_SCHEMA
        and harness_payload["schema_version"] == PRODUCTION_HARNESS_ARTIFACT_SCHEMA
        and policy_payload["candidate_joint_version_id"] == candidate_joint_version_id
        and harness_payload["candidate_joint_version_id"] == candidate_joint_version_id
        and policy_payload["joint_candidate_bundle_sha256"] == acceptance.bundle_sha256
        and harness_payload["joint_candidate_bundle_sha256"] == acceptance.bundle_sha256
        and policy_payload["production_checkpoint_manifest_sha256"]
        == acceptance.checkpoint_manifest_sha256
        and harness_payload["production_checkpoint_manifest_sha256"]
        == acceptance.checkpoint_manifest_sha256
        and policy_payload["policy_update_receipt_sha256"]
        == acceptance.policy_receipt_sha256
        and harness_payload["harness_update_receipt_sha256"]
        == acceptance.harness_receipt_sha256,
        "production release artifacts differ from live X lineage",
    )
    policy_engine_version = policy_payload["policy_engine_version"]
    policy_dcp_checkpoint_sha256 = policy_payload[
        "policy_checkpoint_manifest_sha256"
    ]
    policy_checkpoint_sha256 = policy_payload["policy_serving_parameter_sha256"]
    harness_checkpoint_sha256 = harness_payload["harness_checkpoint_sha256"]
    harness_parameter_digest = harness_payload["harness_parameter_sha256"]
    _require(
        type(policy_engine_version) is int
        and policy_engine_version > 0
        and _is_sha256(policy_checkpoint_sha256)
        and _is_sha256(policy_dcp_checkpoint_sha256)
        and _is_sha256(policy_payload["policy_serving_export_manifest_sha256"])
        and _is_sha256(policy_payload["policy_serving_lineage_record_sha256"])
        and _is_sha256(harness_checkpoint_sha256)
        and _is_sha256(harness_parameter_digest),
        "production release artifact checkpoint identity is invalid",
    )
    parent_target = _freeze_parent_worker_target(
        workers=workers,
        parent=ProductionReleaseIdentity(
            release_id=parent.release_id,
            joint_version=parent.joint_version,
        ),
        policy_engine_version=policy_engine_version - 1,
    )
    candidate_target = ProductionReleaseTarget(
        release_id=candidate.release_id,
        joint_version=candidate.joint_version,
        policy_engine_version=policy_engine_version,
        policy_checkpoint_sha256=str(policy_checkpoint_sha256),
        harness_checkpoint_sha256=str(harness_checkpoint_sha256),
        harness_parameter_digest=str(harness_parameter_digest),
    )
    probe_specs_sha256 = _probe_specs_sha256(
        probes,
        release_ids={parent.release_id, candidate.release_id},
    )
    acceptance_report = live_candidate_acceptance.report
    _require(
        isinstance(acceptance_report, Mapping),
        "live X acceptance report is unavailable",
    )
    critical_suites = acceptance_report.get("critical_suites")
    _require(
        isinstance(critical_suites, list),
        "live X acceptance has no critical suite evidence",
    )
    joint_safety_records = [
        suite
        for suite in critical_suites
        if isinstance(suite, Mapping)
        and isinstance(suite.get("spec"), Mapping)
        and suite["spec"].get("kind") == "joint_safety"
    ]
    _require(
        len(joint_safety_records) == 1,
        "live X acceptance has no unique joint-safety probe",
    )
    joint_safety = joint_safety_records[0]
    joint_safety_spec = joint_safety["spec"]
    joint_safety_probe = joint_safety.get("probe")
    _require(
        isinstance(joint_safety_probe, Mapping),
        "live X joint-safety probe evidence is invalid",
    )
    candidate_probe = probes[candidate.release_id]
    _require(
        candidate_probe.fixture_sha256 == joint_safety_spec.get("fixture_sha256")
        and candidate_probe.expected_output_sha256
        == joint_safety_probe.get("production_probe_output_sha256"),
        "candidate active probe differs from the frozen X joint-safety probe",
    )
    authorization = ProductionActivationAuthorization._create(
        parent=parent_target,
        candidate=candidate_target,
        acceptance_record_sha256=acceptance.record_sha256,
        exact_recovery_record_sha256=recovery.record_sha256,
        probe_specs_sha256=probe_specs_sha256,
        token=_PRODUCTION_AUTHORIZATION_TOKEN,
    )
    return require_production_activation_authorization(authorization, store=store)


@dataclass(frozen=True)
class ProductionJointActivationResult:
    activation_id: str
    parent_release_id: str
    candidate_release_id: str
    active_release_id: str
    outcome: str
    journal_path: Path
    attestation_path: Path
    attestation_sha256: str
    rollback_record_path: Path
    rollback_record_sha256: str
    evidence_scope: Mapping[str, bool]


@dataclass(frozen=True)
class ProductionRollbackRecoveryResult:
    activation_id: str
    parent_release_id: str
    candidate_release_id: str
    active_release_id: str
    outcome: str
    rollback_record_path: Path
    journal_path: Path | None
    evidence_scope: Mapping[str, bool]


class _ProductionWorkerBridge:
    """Convert active observations to non-authoritative journal receipts."""

    def __init__(
        self,
        *,
        workers: Sequence[ProductionActivationWorker],
        parent: ProductionReleaseTarget,
        candidate: ProductionReleaseTarget,
        probes: Mapping[str, ProductionProbeSpec],
    ) -> None:
        self.workers = tuple(workers)
        self.targets = {
            parent.release_id: parent,
            candidate.release_id: candidate,
        }
        self.probes = dict(probes)
        worker_ids = tuple(worker.worker_id for worker in self.workers)
        _require(
            bool(worker_ids)
            and all(
                isinstance(worker_id, str) and worker_id for worker_id in worker_ids
            )
            and len(set(worker_ids)) == len(worker_ids),
            "production workers must have unique non-empty IDs",
        )
        _require(
            set(self.probes) == set(self.targets),
            "production probes must cover the parent and candidate releases",
        )
        for probe in self.probes.values():
            _require(
                type(probe) is ProductionProbeSpec,
                "production probe must use the native typed specification",
            )
            probe.validate()
        self.worker_ids = worker_ids
        self.evidence: list[dict[str, object]] = []

    @staticmethod
    def _expect_none(value: object, operation: str, worker_id: str) -> None:
        _require(
            value is None,
            f"production worker {worker_id} {operation} returned self-reported evidence",
        )

    def _read_states(self) -> tuple[ProductionWorkerState, ...]:
        states: list[ProductionWorkerState] = []
        for worker in self.workers:
            state = worker.read_state()
            _require(
                type(state) is ProductionWorkerState,
                f"production worker {worker.worker_id} returned an untyped state mapping",
            )
            state.validate()
            _require(
                state.worker_id == worker.worker_id,
                "production worker state is attributed to another worker",
            )
            states.append(state)
        _require(
            tuple(state.worker_id for state in states) == self.worker_ids,
            "production worker state coverage differs from the frozen worker set",
        )
        return tuple(states)

    @staticmethod
    def _validate_state(
        state: ProductionWorkerState,
        *,
        target: ProductionReleaseTarget,
        lifecycle_phase: str,
        policy: bool,
        harness: bool,
        binding: bool,
    ) -> None:
        _require(
            state.lifecycle_phase == lifecycle_phase,
            f"production worker {state.worker_id} lifecycle phase differs",
        )
        if policy:
            _require(
                state.policy_engine_version == target.policy_engine_version
                and state.policy_checkpoint_sha256 == target.policy_checkpoint_sha256,
                f"production worker {state.worker_id} Policy sync differs from target",
            )
        if harness:
            _require(
                state.harness_controller_version
                == target.joint_version.harness_controller
                and state.harness_checkpoint_sha256 == target.harness_checkpoint_sha256
                and state.harness_parameter_digest == target.harness_parameter_digest,
                f"production worker {state.worker_id} Harness install differs from target",
            )
        if binding:
            _require(
                state.active_release_id == target.release_id
                and state.joint_version == target.joint_version,
                f"production worker {state.worker_id} release binding differs from target",
            )

    def _observe(
        self,
        *,
        operation: str,
        target: ProductionReleaseTarget,
        lifecycle_phase: str,
        policy: bool,
        harness: bool,
        binding: bool,
        probe_outputs: Mapping[str, str] | None = None,
    ) -> tuple[ProductionWorkerState, ...]:
        states = self._read_states()
        for state in states:
            self._validate_state(
                state,
                target=target,
                lifecycle_phase=lifecycle_phase,
                policy=policy,
                harness=harness,
                binding=binding,
            )
        entry: dict[str, object] = {
            "operation": operation,
            "target_release_id": target.release_id,
            "target_joint_version_id": target.joint_version.version_id,
            "worker_states": [state.to_record() for state in states],
            "probe_output_sha256": (
                {} if probe_outputs is None else dict(probe_outputs)
            ),
        }
        entry["evidence_sha256"] = _sha256(entry)
        self.evidence.append(entry)
        return states

    @staticmethod
    def _journal_worker_versions(
        states: Sequence[ProductionWorkerState],
        target: ProductionReleaseTarget,
    ) -> dict[str, object]:
        return {
            state.worker_id: {
                "policy": target.joint_version.policy,
                "harness_controller": target.joint_version.harness_controller,
            }
            for state in states
        }

    def callback(
        self,
        operation: str,
        release: ReleaseManifest,
    ) -> dict[str, object]:
        try:
            target = self.targets[release.release_id]
        except KeyError as exc:
            raise JointActivationError(
                "production callback target is outside the authorized release pair"
            ) from exc
        _require(
            release.joint_version == target.joint_version,
            "production callback release differs from its authorized target",
        )

        if operation in {"quiesce", "rollback_quiesce"}:
            if operation == "quiesce":
                self._observe(
                    operation="pre_quiesce_parent_observation",
                    target=target,
                    lifecycle_phase="serving",
                    policy=True,
                    harness=True,
                    binding=True,
                )
            for worker in self.workers:
                self._expect_none(worker.quiesce(), operation, worker.worker_id)
            self._observe(
                operation=operation,
                target=target,
                lifecycle_phase="quiesced",
                policy=False,
                harness=False,
                binding=False,
            )
            observations: dict[str, object] = {"quiesced": True}
        elif operation in {"sync_policy", "restore_policy"}:
            for worker in self.workers:
                self._expect_none(
                    worker.install_policy(target), operation, worker.worker_id
                )
            self._observe(
                operation=operation,
                target=target,
                lifecycle_phase="quiesced",
                policy=True,
                harness=False,
                binding=False,
            )
            observations = {"policy_version": target.joint_version.policy}
        elif operation in {"stage_harness", "restore_harness"}:
            for worker in self.workers:
                self._expect_none(
                    worker.install_harness(target), operation, worker.worker_id
                )
            self._observe(
                operation=operation,
                target=target,
                lifecycle_phase="quiesced",
                policy=True,
                harness=True,
                binding=False,
            )
            observations = {
                "harness_controller_version": target.joint_version.harness_controller
            }
        elif operation == "verify_sync":
            states = self._observe(
                operation=operation,
                target=target,
                lifecycle_phase="quiesced",
                policy=True,
                harness=True,
                binding=False,
            )
            observations = {
                "worker_versions": self._journal_worker_versions(states, target)
            }
        elif operation in {"set_versions", "set_parent_versions"}:
            for worker in self.workers:
                self._expect_none(
                    worker.bind_release(target), operation, worker.worker_id
                )
            states = self._observe(
                operation=operation,
                target=target,
                lifecycle_phase="quiesced",
                policy=True,
                harness=True,
                binding=True,
            )
            observations = {
                "worker_versions": self._journal_worker_versions(states, target)
            }
        elif operation in {"post_publish_probe", "parent_probe"}:
            spec = self.probes[target.release_id]
            outputs: dict[str, str] = {}
            for worker in self.workers:
                raw = worker.run_probe(spec.fixture)
                _require(
                    type(raw) is bytes,
                    f"production worker {worker.worker_id} probe returned a verdict instead of raw bytes",
                )
                output_sha256 = hashlib.sha256(raw).hexdigest()
                _require(
                    output_sha256 == spec.expected_output_sha256,
                    f"production worker {worker.worker_id} probe output differs",
                )
                outputs[worker.worker_id] = output_sha256
            states = self._observe(
                operation=operation,
                target=target,
                lifecycle_phase="quiesced",
                policy=True,
                harness=True,
                binding=True,
                probe_outputs=outputs,
            )
            observations = {
                "worker_versions": self._journal_worker_versions(states, target),
                "probe_passed": True,
            }
        elif operation in {"resume", "resume_parent"}:
            for worker in self.workers:
                self._expect_none(worker.resume(), operation, worker.worker_id)
            self._observe(
                operation=operation,
                target=target,
                lifecycle_phase="serving",
                policy=True,
                harness=True,
                binding=True,
            )
            observations = {"resumed": True}
        else:
            raise JointActivationError(
                f"unsupported production activation operation: {operation}"
            )
        return callback_receipt(
            operation=operation,
            release=release,
            observations=observations,
            control_plane_only=True,
        )

    def callbacks(self) -> JointActivationCallbacks:
        return JointActivationCallbacks(
            quiesce=self.callback,
            sync_policy=self.callback,
            stage_harness=self.callback,
            verify_sync=self.callback,
            set_versions=self.callback,
            probe=self.callback,
            resume=self.callback,
            restore_policy=self.callback,
            restore_harness=self.callback,
        )


class ProductionJointActivationController:
    """Production entry point gated by native live W/X authorization.

    The callback-receipt state machine remains useful for crash recovery, but
    none of its mappings are trusted.  This wrapper owns the callbacks, samples
    exact typed worker state after every side effect, computes probe hashes from
    raw bytes, and emits a separate production attestation only after the
    candidate is active and serving on every worker.
    """

    def __init__(
        self,
        *,
        store: JointReleaseStore,
        workers: Sequence[ProductionActivationWorker],
        probes: Mapping[str, ProductionProbeSpec],
        project_root: str | Path,
    ) -> None:
        self.store = store
        self.workers = tuple(workers)
        _require(bool(self.workers), "production activation has no workers")
        _require(
            all(
                isinstance(worker, ProductionActivationWorker)
                for worker in self.workers
            ),
            "production activation workers must use the typed adapter",
        )
        self.probes = dict(probes)
        self.project_root = Path(project_root).expanduser().resolve()
        if self.project_root != repository_root():
            raise ValueError("activation project root differs from the actual checkout")

    @property
    def rollback_record_root(self) -> Path:
        return self.store.root / "activation-rollback-only"

    def _seal_rollback_only_record(
        self,
        *,
        activation_id: str,
        authorization: ProductionActivationAuthorization,
    ) -> tuple[Path, dict[str, object]]:
        probe_records = {
            release_id: self.probes[release_id].to_record()
            for release_id in sorted(self.probes)
        }
        authorization_record = authorization.to_record()
        parent_worker_states: list[dict[str, object]] = []
        for worker in self.workers:
            state = worker.read_state()
            _require(
                type(state) is ProductionWorkerState
                and state.worker_id == worker.worker_id,
                "production rollback seal requires exact typed worker state",
            )
            state.validate()
            _ProductionWorkerBridge._validate_state(
                state,
                target=authorization.parent,
                lifecycle_phase="serving",
                policy=True,
                harness=True,
                binding=True,
            )
            parent_worker_states.append(state.to_record())
        record: dict[str, object] = {
            "schema_version": PRODUCTION_ROLLBACK_SCHEMA,
            "mode": "rollback_only",
            "activation_id": activation_id,
            "store_root_sha256": hashlib.sha256(
                str(self.store.root).encode("utf-8")
            ).hexdigest(),
            "worker_ids": [worker.worker_id for worker in self.workers],
            "parent_worker_states": parent_worker_states,
            "parent_target": authorization.parent.to_record(),
            "candidate_target": authorization.candidate.to_record(),
            "probe_specs": probe_records,
            "probe_specs_sha256": authorization.probe_specs_sha256,
            "authorization_record_sha256": _sha256(authorization_record),
            "acceptance_record_sha256": authorization.acceptance_record_sha256,
            "exact_recovery_record_sha256": (
                authorization.exact_recovery_record_sha256
            ),
            "expected_journal_name": f"activation-{activation_id}.json",
            "evidence_scope": dict(_PRODUCTION_ROLLBACK_EVIDENCE_SCOPE),
            "record_sha256": "",
        }
        record["record_sha256"] = _record_sha256(record)
        _assert_no_secrets(record)
        root = require_outside_repository(self.rollback_record_root)
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        path = root / f"rollback-{activation_id}.json"
        _private_exclusive_write(path, record)
        self._validate_rollback_only_record(path, require_active_parent=True)
        return path, record

    def _validate_rollback_only_record(
        self,
        path: str | Path,
        *,
        require_active_parent: bool,
    ) -> tuple[
        dict[str, object],
        ProductionReleaseTarget,
        ProductionReleaseTarget,
    ]:
        unresolved_source = Path(path).expanduser()
        _require(
            not unresolved_source.is_symlink(),
            "rollback-only record cannot be a symlink",
        )
        source = unresolved_source.resolve()
        root = require_outside_repository(self.rollback_record_root)
        _require(
            source.parent == root,
            "rollback-only record is outside the canonical store root",
        )
        record = _read_json(source)
        _assert_no_secrets(record)
        _exact_mapping(
            record,
            _PRODUCTION_ROLLBACK_FIELDS,
            "production rollback-only record",
        )
        _require(
            record["schema_version"] == PRODUCTION_ROLLBACK_SCHEMA
            and record["mode"] == "rollback_only"
            and record["evidence_scope"] == _PRODUCTION_ROLLBACK_EVIDENCE_SCOPE,
            "production rollback-only record scope differs",
        )
        _require(
            record["record_sha256"] == _record_sha256(record),
            "production rollback-only record hash mismatch",
        )
        activation_id = record["activation_id"]
        _require(
            isinstance(activation_id, str)
            and len(activation_id) == 32
            and all(character in "0123456789abcdef" for character in activation_id)
            and source.name == f"rollback-{activation_id}.json"
            and record["expected_journal_name"] == f"activation-{activation_id}.json",
            "production rollback-only activation identity differs",
        )
        _require(
            record["store_root_sha256"]
            == hashlib.sha256(str(self.store.root).encode("utf-8")).hexdigest(),
            "production rollback-only store binding differs",
        )
        worker_ids = [worker.worker_id for worker in self.workers]
        _require(
            record["worker_ids"] == worker_ids
            and len(set(worker_ids)) == len(worker_ids),
            "production rollback-only worker binding differs",
        )
        parent = ProductionReleaseTarget.from_record(record["parent_target"])
        candidate = ProductionReleaseTarget.from_record(record["candidate_target"])
        parent_state_records = record["parent_worker_states"]
        _require(
            isinstance(parent_state_records, list)
            and len(parent_state_records) == len(worker_ids),
            "production rollback parent state coverage differs",
        )
        parent_states = tuple(
            ProductionWorkerState.from_record(item) for item in parent_state_records
        )
        _require(
            [state.worker_id for state in parent_states] == worker_ids,
            "production rollback parent worker state order differs",
        )
        for state in parent_states:
            _ProductionWorkerBridge._validate_state(
                state,
                target=parent,
                lifecycle_phase="serving",
                policy=True,
                harness=True,
                binding=True,
            )
        _require(
            candidate.policy_engine_version == parent.policy_engine_version + 1,
            "production rollback-only Policy versions differ",
        )
        parent_manifest = self.store.read_manifest(parent.release_id)
        candidate_manifest = self.store.read_manifest(candidate.release_id)
        _require(
            parent_manifest.joint_version == parent.joint_version
            and candidate_manifest.joint_version == candidate.joint_version
            and candidate_manifest.parent_release_id == parent.release_id,
            "production rollback-only release lineage differs",
        )
        policy_artifact, harness_artifact = self.store.read_artifacts(
            candidate.release_id
        )
        _require(
            policy_artifact.payload.get("schema_version")
            == _PRODUCTION_POLICY_ARTIFACT_SCHEMA
            and harness_artifact.payload.get("schema_version")
            == _PRODUCTION_HARNESS_ARTIFACT_SCHEMA,
            "production rollback-only candidate artifacts differ",
        )
        _require(
            policy_artifact.payload.get("policy_engine_version")
            == candidate.policy_engine_version
            and policy_artifact.payload.get("policy_serving_parameter_sha256")
            == candidate.policy_checkpoint_sha256
            and harness_artifact.payload.get("harness_checkpoint_sha256")
            == candidate.harness_checkpoint_sha256
            and harness_artifact.payload.get("harness_parameter_sha256")
            == candidate.harness_parameter_digest,
            "production rollback-only candidate checkpoint binding differs",
        )
        active = self.store.read_active()
        _require(active is not None, "production rollback store has no active release")
        allowed_active_ids = {parent.release_id, candidate.release_id}
        _require(
            active.release_id in allowed_active_ids,
            "production rollback store advanced to an unrelated release",
        )
        if require_active_parent:
            _require(
                active.release_id == parent.release_id,
                "production rollback record was not sealed against active parent",
            )
        probe_specs = _exact_mapping(
            record["probe_specs"],
            allowed_active_ids,
            "production rollback probe specs",
        )
        expected_probe_records = {
            release_id: self.probes[release_id].to_record()
            for release_id in sorted(allowed_active_ids)
        }
        _require(
            dict(probe_specs) == expected_probe_records
            and record["probe_specs_sha256"]
            == _probe_specs_sha256(
                self.probes,
                release_ids=allowed_active_ids,
            ),
            "production rollback probe binding differs",
        )
        authorization_record = {
            "parent": parent.to_record(),
            "candidate": candidate.to_record(),
            "acceptance_record_sha256": record["acceptance_record_sha256"],
            "exact_recovery_record_sha256": record["exact_recovery_record_sha256"],
            "probe_specs_sha256": record["probe_specs_sha256"],
        }
        _require(
            _is_sha256(record["acceptance_record_sha256"])
            and _is_sha256(record["exact_recovery_record_sha256"])
            and record["authorization_record_sha256"] == _sha256(authorization_record),
            "production rollback authorization binding differs",
        )
        return record, parent, candidate

    def _controller(
        self,
        *,
        authorization: ProductionActivationAuthorization,
        bridge: _ProductionWorkerBridge,
    ) -> JointActivationController:
        authorization_record = authorization.to_record()

        def validate(record: Mapping[str, object]) -> dict[str, object]:
            current = require_production_activation_authorization(
                authorization,
                store=self.store,
            )
            _require(
                dict(record) == authorization_record,
                "production activation authorization changed before prepare",
            )
            return accepted_report_receipt(
                accepted_x_report=record,
                parent_release_id=current.parent.release_id,
                candidate_release_id=current.candidate.release_id,
                control_plane_only=True,
            )

        return JointActivationController(
            store=self.store,
            journal_root=self.store.activation_journal_root,
            project_root=self.project_root,
            worker_ids=bridge.worker_ids,
            callbacks=bridge.callbacks(),
            validate_accepted_report=validate,
            control_plane_only=True,
            _production_bridge_token=_PRODUCTION_BRIDGE_TOKEN,
        )

    def _rollback_controller(
        self,
        *,
        bridge: _ProductionWorkerBridge,
    ) -> JointActivationController:
        def reject_forward_validation(
            _: Mapping[str, object],
        ) -> Mapping[str, object]:
            raise JointActivationError(
                "rollback-only recovery cannot validate forward activation"
            )

        return JointActivationController(
            store=self.store,
            journal_root=self.store.activation_journal_root,
            project_root=self.project_root,
            worker_ids=bridge.worker_ids,
            callbacks=bridge.callbacks(),
            validate_accepted_report=reject_forward_validation,
            control_plane_only=True,
            _production_bridge_token=_PRODUCTION_BRIDGE_TOKEN,
        )

    @staticmethod
    def _restore_parent_without_journal(
        *,
        bridge: _ProductionWorkerBridge,
        parent: ReleaseManifest,
    ) -> None:
        for operation in (
            "rollback_quiesce",
            "restore_policy",
            "restore_harness",
            "set_parent_versions",
            "parent_probe",
            "resume_parent",
        ):
            bridge.callback(operation, parent)

    def _verify_restored_parent(
        self,
        *,
        bridge: _ProductionWorkerBridge,
        parent: ProductionReleaseTarget,
    ) -> None:
        active = self.store.read_active()
        _require(
            active is not None and active.release_id == parent.release_id,
            "rollback-only recovery did not restore the active parent",
        )
        for state in bridge._read_states():
            bridge._validate_state(
                state,
                target=parent,
                lifecycle_phase="serving",
                policy=True,
                harness=True,
                binding=True,
            )

    def _attest_completed_activation(
        self,
        *,
        result: JointActivationResult,
        authorization: ProductionActivationAuthorization,
        bridge: _ProductionWorkerBridge,
        rollback_record_path: Path,
        rollback_record: Mapping[str, object],
    ) -> tuple[Path, dict[str, object]]:
        active = self.store.read_active()
        _require(
            active is not None
            and active.release_id == authorization.candidate.release_id,
            "production candidate is not active after the activation journal completed",
        )
        final_states = bridge._read_states()
        for state in final_states:
            bridge._validate_state(
                state,
                target=authorization.candidate,
                lifecycle_phase="serving",
                policy=True,
                harness=True,
                binding=True,
            )
        journal = _read_json(result.journal_path)
        attestation: dict[str, object] = {
            "schema_version": PRODUCTION_ATTESTATION_SCHEMA,
            "activation_id": result.activation_id,
            "authorization": authorization.to_record(),
            "journal_path": str(result.journal_path),
            "journal_record_sha256": journal["record_sha256"],
            "rollback_record_path": str(rollback_record_path),
            "rollback_record_sha256": rollback_record["record_sha256"],
            "worker_ids": list(bridge.worker_ids),
            "operation_evidence": bridge.evidence,
            "final_worker_states": [state.to_record() for state in final_states],
            "evidence_scope": dict(_PRODUCTION_EVIDENCE_SCOPE),
            "record_sha256": "",
        }
        attestation["record_sha256"] = _record_sha256(attestation)
        attestation_root = self.store.root / "activation-attestations"
        require_outside_repository(attestation_root).mkdir(parents=True, exist_ok=True)
        os.chmod(attestation_root, 0o700)
        attestation_path = attestation_root / f"activation-{result.activation_id}.json"
        _private_exclusive_write(attestation_path, attestation)
        return attestation_path, attestation

    def _rollback_completed_activation(
        self,
        *,
        controller: JointActivationController,
        result: JointActivationResult,
        authorization: ProductionActivationAuthorization,
    ) -> None:
        with self.store.activation_lease(), controller._exclusive_lock():
            record = controller._read_journal(result.journal_path)
            parent = self.store.read_manifest(authorization.parent.release_id)
            candidate = self.store.read_manifest(authorization.candidate.release_id)
            controller._rollback(
                result.journal_path,
                record,
                parent=parent,
                candidate=candidate,
            )

    def activate(
        self,
        authorization: ProductionActivationAuthorization,
        *,
        fault_after_stage: str | None = None,
        fault_mode: str = "raise",
    ) -> ProductionJointActivationResult:
        authorization = require_production_activation_authorization(
            authorization,
            store=self.store,
        )
        _require(
            _probe_specs_sha256(
                self.probes,
                release_ids={
                    authorization.parent.release_id,
                    authorization.candidate.release_id,
                },
            )
            == authorization.probe_specs_sha256,
            "production probe specifications differ from live authorization",
        )
        activation_id = uuid.uuid4().hex
        rollback_record_path, rollback_record = self._seal_rollback_only_record(
            activation_id=activation_id,
            authorization=authorization,
        )
        bridge = _ProductionWorkerBridge(
            workers=self.workers,
            parent=authorization.parent,
            candidate=authorization.candidate,
            probes=self.probes,
        )
        controller = self._controller(authorization=authorization, bridge=bridge)
        result = controller.activate(
            accepted_x_report=authorization.to_record(),
            parent_release_id=authorization.parent.release_id,
            candidate_release_id=authorization.candidate.release_id,
            fault_after_stage=fault_after_stage,
            fault_mode=fault_mode,
            _activation_id=activation_id,
        )
        try:
            attestation_path, attestation = self._attest_completed_activation(
                result=result,
                authorization=authorization,
                bridge=bridge,
                rollback_record_path=rollback_record_path,
                rollback_record=rollback_record,
            )
        except Exception as attestation_error:
            self._rollback_completed_activation(
                controller=controller,
                result=result,
                authorization=authorization,
            )
            raise JointActivationError(
                "production attestation failed; the parent pair was restored"
            ) from attestation_error
        return ProductionJointActivationResult(
            activation_id=result.activation_id,
            parent_release_id=result.parent_release_id,
            candidate_release_id=result.candidate_release_id,
            active_release_id=result.active_release_id,
            outcome=result.outcome,
            journal_path=result.journal_path,
            attestation_path=attestation_path,
            attestation_sha256=str(attestation["record_sha256"]),
            rollback_record_path=rollback_record_path,
            rollback_record_sha256=str(rollback_record["record_sha256"]),
            evidence_scope=dict(_PRODUCTION_EVIDENCE_SCOPE),
        )

    def recover_pending(
        self,
        rollback_record_path: str | Path,
    ) -> ProductionRollbackRecoveryResult:
        """Fresh-process recovery that can only restore and verify the parent."""

        unresolved_source = Path(rollback_record_path).expanduser()
        with self.store.activation_lease():
            record, parent, candidate = self._validate_rollback_only_record(
                unresolved_source,
                require_active_parent=False,
            )
            source = unresolved_source.resolve()
            bridge = _ProductionWorkerBridge(
                workers=self.workers,
                parent=parent,
                candidate=candidate,
                probes=self.probes,
            )
            controller = self._rollback_controller(bridge=bridge)
            journal_path = self.store.activation_journal_root / str(
                record["expected_journal_name"]
            )
            parent_manifest = self.store.read_manifest(parent.release_id)
            candidate_manifest = self.store.read_manifest(candidate.release_id)
            if not journal_path.exists():
                active = self.store.read_active()
                _require(
                    active is not None and active.release_id == parent.release_id,
                    "missing production journal cannot authorize candidate rollback",
                )
                self._restore_parent_without_journal(
                    bridge=bridge,
                    parent=parent_manifest,
                )
                outcome = "parent_restored_without_journal"
                result_journal_path: Path | None = None
            else:
                with controller._exclusive_lock():
                    journal = controller._read_journal(journal_path)
                    _require(
                        journal["activation_id"] == record["activation_id"]
                        and journal["parent_release_id"] == parent.release_id
                        and journal["candidate_release_id"] == candidate.release_id
                        and journal["worker_ids"] == record["worker_ids"],
                        "production rollback journal binding differs",
                    )
                    accepted = _exact_mapping(
                        journal["accepted_x_report"],
                        _ACCEPTED_REPORT_FIELDS,
                        "production rollback accepted receipt",
                    )
                    _require(
                        accepted["report_sha256"]
                        == record["authorization_record_sha256"],
                        "production rollback journal authorization differs",
                    )
                    if journal["status"] == "parent_restored":
                        self._restore_parent_without_journal(
                            bridge=bridge,
                            parent=parent_manifest,
                        )
                    else:
                        controller._rollback(
                            journal_path,
                            journal,
                            parent=parent_manifest,
                            candidate=candidate_manifest,
                        )
                outcome = "parent_restored_from_journal"
                result_journal_path = journal_path
            self._verify_restored_parent(bridge=bridge, parent=parent)
            return ProductionRollbackRecoveryResult(
                activation_id=str(record["activation_id"]),
                parent_release_id=parent.release_id,
                candidate_release_id=candidate.release_id,
                active_release_id=parent.release_id,
                outcome=outcome,
                rollback_record_path=source,
                journal_path=result_journal_path,
                evidence_scope=dict(_PRODUCTION_ROLLBACK_EVIDENCE_SCOPE),
            )
