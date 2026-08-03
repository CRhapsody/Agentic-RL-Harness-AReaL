from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from .paths import require_outside_repository, require_within_configured_root
from .trajectory.schema import JointVersion


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


_SECRET_FIELDS = {
    "access_token",
    "admin_api_key",
    "api_key",
    "authorization",
    "credential",
    "github_token",
    "password",
    "refresh_token",
    "secret_key",
    "session_api_key",
    "token",
}


def _assert_no_secrets(value: object, path: str = "release") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _SECRET_FIELDS or normalized.endswith(
                ("_api_key", "_password", "_secret", "_token")
            ):
                raise ValueError(f"credential field cannot enter release: {path}.{key}")
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = require_within_configured_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"release JSON path is missing or unsafe: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"release JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return value


def _validate_joint_version(value: JointVersion, label: str) -> None:
    if type(value) is not JointVersion:
        raise TypeError(f"{label} must be an exact JointVersion")
    if not all(
        isinstance(getattr(value, field), str) and bool(getattr(value, field))
        for field in JointVersion.__dataclass_fields__
    ):
        raise ValueError(f"{label} fields must be non-empty strings")


@dataclass(frozen=True)
class ComponentCheckpoint:
    component: str
    version: str
    parameters: tuple[float, ...]
    optimizer_momentum: tuple[float, ...]
    optimizer_step: int
    rng_state: int
    sample_count: int
    state_sha256: str

    @staticmethod
    def compute_state_sha256(
        *,
        component: str,
        version: str,
        parameters: tuple[float, ...],
        optimizer_momentum: tuple[float, ...],
        optimizer_step: int,
        rng_state: int,
        sample_count: int,
    ) -> str:
        return _sha256(
            {
                "component": component,
                "version": version,
                "parameters": parameters,
                "optimizer_momentum": optimizer_momentum,
                "optimizer_step": optimizer_step,
                "rng_state": rng_state,
                "sample_count": sample_count,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        component: str,
        version: str,
        parameters: tuple[float, ...],
        optimizer_momentum: tuple[float, ...],
        optimizer_step: int,
        rng_state: int,
        sample_count: int,
    ) -> ComponentCheckpoint:
        return cls(
            component=component,
            version=version,
            parameters=parameters,
            optimizer_momentum=optimizer_momentum,
            optimizer_step=optimizer_step,
            rng_state=rng_state,
            sample_count=sample_count,
            state_sha256=cls.compute_state_sha256(
                component=component,
                version=version,
                parameters=parameters,
                optimizer_momentum=optimizer_momentum,
                optimizer_step=optimizer_step,
                rng_state=rng_state,
                sample_count=sample_count,
            ),
        )

    def validate(self) -> None:
        if self.component not in {"policy", "harness"}:
            raise ValueError("checkpoint component must be policy or harness")
        if not self.version:
            raise ValueError("component version must be non-empty")
        if not self.parameters or len(self.parameters) != len(self.optimizer_momentum):
            raise ValueError(
                "parameters and optimizer momentum must have equal non-zero length"
            )
        if not all(
            math.isfinite(value)
            for value in (*self.parameters, *self.optimizer_momentum)
        ):
            raise ValueError("checkpoint tensors must be finite")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer step must be a non-negative integer")
        if type(self.rng_state) is not int or not 0 <= self.rng_state < 2**31:
            raise ValueError(
                "component RNG state must be a 31-bit non-negative integer"
            )
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("component sample count must be a non-negative integer")
        expected_state_sha256 = self.compute_state_sha256(
            component=self.component,
            version=self.version,
            parameters=self.parameters,
            optimizer_momentum=self.optimizer_momentum,
            optimizer_step=self.optimizer_step,
            rng_state=self.rng_state,
            sample_count=self.sample_count,
        )
        if self.state_sha256 != expected_state_sha256:
            raise ValueError(
                "component checkpoint state hash does not match its contents"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComponentCheckpoint:
        expected_fields = set(cls.__dataclass_fields__)
        if set(payload) != expected_fields:
            raise ValueError("component checkpoint field set differs")
        if not isinstance(payload["component"], str) or not isinstance(
            payload["version"], str
        ):
            raise TypeError("component checkpoint identity must be strings")
        if not isinstance(payload["parameters"], (list, tuple)) or not isinstance(
            payload["optimizer_momentum"], (list, tuple)
        ):
            raise TypeError("component checkpoint tensors must be arrays")
        for field in ("optimizer_step", "rng_state", "sample_count"):
            if type(payload[field]) is not int:
                raise TypeError(f"component checkpoint {field} must be an integer")
        if not isinstance(payload["state_sha256"], str):
            raise TypeError("component checkpoint state hash must be a string")
        checkpoint = cls(
            component=payload["component"],
            version=payload["version"],
            parameters=tuple(float(value) for value in payload["parameters"]),
            optimizer_momentum=tuple(
                float(value) for value in payload["optimizer_momentum"]
            ),
            optimizer_step=payload["optimizer_step"],
            rng_state=payload["rng_state"],
            sample_count=payload["sample_count"],
            state_sha256=payload["state_sha256"],
        )
        checkpoint.validate()
        return checkpoint


@dataclass(frozen=True)
class JointCheckpoint:
    schema_version: str
    joint_version: JointVersion
    active_release_id: str
    macro_step: int
    policy: ComponentCheckpoint
    harness: ComponentCheckpoint
    rng_state: int
    rollout_cursor: int

    def validate(self) -> None:
        if self.schema_version != "jph.joint-checkpoint.v1":
            raise ValueError("joint checkpoint schema version differs")
        _validate_joint_version(self.joint_version, "checkpoint JointVersion")
        self.policy.validate()
        self.harness.validate()
        if not isinstance(self.active_release_id, str) or not self.active_release_id:
            raise ValueError("active release ID must be a non-empty string")
        if self.policy.component != "policy" or self.harness.component != "harness":
            raise ValueError("joint checkpoint components are swapped")
        if self.policy.version != self.joint_version.policy:
            raise ValueError("policy checkpoint version differs from JointVersion")
        if self.harness.version != self.joint_version.harness_controller:
            raise ValueError("Harness checkpoint version differs from JointVersion")
        if type(self.macro_step) is not int or self.macro_step < 0:
            raise ValueError("macro step must be a non-negative integer")
        component_steps = (
            self.policy.optimizer_step,
            self.harness.optimizer_step,
        )
        if any(step > self.macro_step for step in component_steps):
            raise ValueError("component optimizer step exceeds joint macro step")
        if max(component_steps) != self.macro_step:
            raise ValueError("joint macro step has no corresponding component update")
        if type(self.rng_state) is not int or not 0 <= self.rng_state < 2**31:
            raise ValueError("RNG state must be a 31-bit non-negative integer")
        if type(self.rollout_cursor) is not int or self.rollout_cursor < 0:
            raise ValueError("rollout cursor must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "joint_version": asdict(self.joint_version),
            "schema_version": self.schema_version,
            "active_release_id": self.active_release_id,
            "macro_step": self.macro_step,
            "policy": self.policy.to_dict(),
            "harness": self.harness.to_dict(),
            "rng_state": self.rng_state,
            "rollout_cursor": self.rollout_cursor,
        }

    @property
    def digest(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JointCheckpoint:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("joint checkpoint field set differs")
        version_payload = payload["joint_version"]
        if not isinstance(version_payload, Mapping):
            raise TypeError("joint_version must be an object")
        if set(version_payload) != set(JointVersion.__dataclass_fields__):
            raise ValueError("checkpoint JointVersion field set differs")
        for field in ("schema_version", "active_release_id"):
            if not isinstance(payload[field], str):
                raise TypeError(f"joint checkpoint {field} must be a string")
        for field in ("macro_step", "rng_state", "rollout_cursor"):
            if type(payload[field]) is not int:
                raise TypeError(f"joint checkpoint {field} must be an integer")
        checkpoint = cls(
            schema_version=payload["schema_version"],
            joint_version=JointVersion(**version_payload),
            active_release_id=payload["active_release_id"],
            macro_step=payload["macro_step"],
            policy=ComponentCheckpoint.from_dict(payload["policy"]),
            harness=ComponentCheckpoint.from_dict(payload["harness"]),
            rng_state=payload["rng_state"],
            rollout_cursor=payload["rollout_cursor"],
        )
        checkpoint.validate()
        return checkpoint


def write_joint_checkpoint(path: str | Path, checkpoint: JointCheckpoint) -> None:
    checkpoint.validate()
    _write_atomic_json(
        Path(path).expanduser().resolve(),
        {
            "envelope_schema": "jph.joint-checkpoint-envelope.v1",
            "checkpoint": checkpoint.to_dict(),
            "checkpoint_sha256": checkpoint.digest,
        },
    )


def read_joint_checkpoint(path: str | Path) -> JointCheckpoint:
    source = require_within_configured_root(path)
    envelope = _read_json(source)
    if set(envelope) != {
        "envelope_schema",
        "checkpoint",
        "checkpoint_sha256",
    }:
        raise ValueError("joint checkpoint envelope field set differs")
    if envelope.get("envelope_schema") != "jph.joint-checkpoint-envelope.v1":
        raise ValueError("joint checkpoint envelope schema differs")
    checkpoint_payload = envelope.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("joint checkpoint envelope has no checkpoint object")
    checkpoint = JointCheckpoint.from_dict(checkpoint_payload)
    if envelope.get("checkpoint_sha256") != checkpoint.digest:
        raise ValueError("joint checkpoint digest does not match its contents")
    return checkpoint


@dataclass(frozen=True)
class CandidateArtifact:
    component: str
    version: str
    payload: Mapping[str, Any]

    def validate(self) -> None:
        if not isinstance(self.component, str) or self.component not in {
            "policy",
            "harness",
        }:
            raise ValueError("candidate component must be policy or harness")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("candidate version must be non-empty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("candidate payload must be an object")
        _assert_no_secrets(self.payload, f"{self.component}_candidate")
        _canonical_json(self.to_unsigned_dict())

    def to_unsigned_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "version": self.version,
            "payload": dict(self.payload),
        }

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return self.to_unsigned_dict()

    @property
    def digest(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateArtifact:
        if set(payload) != {"component", "version", "payload"}:
            raise ValueError("candidate artifact field set differs")
        raw_payload = payload.get("payload")
        if not isinstance(raw_payload, Mapping):
            raise TypeError("candidate artifact payload must be an object")
        if not isinstance(payload.get("component"), str) or not isinstance(
            payload.get("version"), str
        ):
            raise TypeError("candidate artifact identity must be strings")
        artifact = cls(
            component=payload["component"],
            version=payload["version"],
            payload=dict(raw_payload),
        )
        artifact.validate()
        return artifact


@dataclass(frozen=True)
class ReleaseManifest:
    release_id: str
    parent_release_id: str | None
    joint_version: JointVersion
    policy_object: str
    harness_object: str

    def validate(self) -> None:
        if (
            not isinstance(self.release_id, str)
            or len(self.release_id) != 20
            or any(character not in "0123456789abcdef" for character in self.release_id)
        ):
            raise ValueError("release ID must be 20 lowercase hexadecimal characters")
        if self.parent_release_id is not None and (
            not isinstance(self.parent_release_id, str)
            or len(self.parent_release_id) != 20
            or any(
                character not in "0123456789abcdef"
                for character in self.parent_release_id
            )
        ):
            raise ValueError(
                "parent release ID must be 20 lowercase hexadecimal characters"
            )
        _validate_joint_version(self.joint_version, "release JointVersion")
        for field in ("policy_object", "harness_object"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value.startswith("objects/")
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise ValueError(f"release {field} is unsafe")
        _assert_no_secrets(self.to_dict(), "release_manifest")
        _canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "parent_release_id": self.parent_release_id,
            "joint_version": asdict(self.joint_version),
            "policy_object": self.policy_object,
            "harness_object": self.harness_object,
        }

    @classmethod
    def create(
        cls,
        *,
        parent_release_id: str | None,
        joint_version: JointVersion,
        policy_object: str,
        harness_object: str,
    ) -> ReleaseManifest:
        _validate_joint_version(joint_version, "release JointVersion")
        unsigned = {
            "parent_release_id": parent_release_id,
            "joint_version": asdict(joint_version),
            "policy_object": policy_object,
            "harness_object": harness_object,
        }
        manifest = cls(
            release_id=_sha256(unsigned)[:20],
            parent_release_id=parent_release_id,
            joint_version=joint_version,
            policy_object=policy_object,
            harness_object=harness_object,
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReleaseManifest:
        if set(payload) != {
            "release_id",
            "parent_release_id",
            "joint_version",
            "policy_object",
            "harness_object",
        }:
            raise ValueError("release manifest field set differs")
        version_payload = payload["joint_version"]
        if not isinstance(version_payload, Mapping):
            raise TypeError("manifest joint_version must be an object")
        if set(version_payload) != set(JointVersion.__dataclass_fields__):
            raise ValueError("release JointVersion field set differs")
        for field in ("release_id", "policy_object", "harness_object"):
            if not isinstance(payload[field], str):
                raise TypeError(f"release {field} must be a string")
        if payload.get("parent_release_id") is not None and not isinstance(
            payload["parent_release_id"], str
        ):
            raise TypeError("parent release ID must be a string or null")
        manifest = cls(
            release_id=payload["release_id"],
            parent_release_id=(
                None
                if payload.get("parent_release_id") is None
                else payload["parent_release_id"]
            ),
            joint_version=JointVersion(**version_payload),
            policy_object=payload["policy_object"],
            harness_object=payload["harness_object"],
        )
        manifest.validate()
        expected = cls.create(
            parent_release_id=manifest.parent_release_id,
            joint_version=manifest.joint_version,
            policy_object=manifest.policy_object,
            harness_object=manifest.harness_object,
        )
        if manifest.release_id != expected.release_id:
            raise ValueError("release manifest ID does not match its contents")
        return manifest


class InjectedPublishFailure(RuntimeError):
    pass


class ConcurrentPublishError(RuntimeError):
    pass


class JointReleaseStore:
    """Content-addressed pair publication with one atomic active manifest."""

    _FAULT_PHASES: ClassVar[set[str]] = {
        "after_policy_object",
        "after_harness_object",
        "before_active_switch",
        "after_active_switch",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = require_outside_repository(root)
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests"
        self.active_path = self.root / "active.json"
        self.lock_path = self.root / ".publish.lock"
        self.activation_lock_path = self.root / ".activation.lock"
        self.activation_journal_root = self.root / "activation-journal"

    def _ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.objects.mkdir(parents=True, exist_ok=True)
        self.manifests.mkdir(parents=True, exist_ok=True)
        os.chmod(self.objects, 0o700)
        os.chmod(self.manifests, 0o700)

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self._ensure_layout()
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(descriptor, "r+") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            if self.lock_path.exists():
                os.chmod(self.lock_path, 0o600)

    @contextmanager
    def activation_lease(self) -> Iterator[None]:
        """Serialize all worker side effects for this release store."""

        self._ensure_layout()
        descriptor = os.open(
            self.activation_lock_path,
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "r+") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            if self.activation_lock_path.exists():
                os.chmod(self.activation_lock_path, 0o600)

    @staticmethod
    def _validate_candidate_pair(
        *,
        joint_version: JointVersion,
        policy: CandidateArtifact,
        harness: CandidateArtifact,
    ) -> None:
        _validate_joint_version(joint_version, "candidate JointVersion")
        policy.validate()
        harness.validate()
        if policy.component != "policy" or harness.component != "harness":
            raise ValueError("publish candidates are swapped")
        if policy.version != joint_version.policy:
            raise ValueError("policy candidate differs from JointVersion")
        if harness.version != joint_version.harness_controller:
            raise ValueError("Harness candidate differs from JointVersion")

    def _manifest_path(self, release_id: str) -> Path:
        if (
            not isinstance(release_id, str)
            or len(release_id) != 20
            or any(character not in "0123456789abcdef" for character in release_id)
        ):
            raise ValueError("release ID must be 20 lowercase hexadecimal characters")
        return self.manifests / f"{release_id}.json"

    def _write_manifest(self, manifest: ReleaseManifest) -> None:
        destination = self._manifest_path(manifest.release_id)
        payload = manifest.to_dict()
        if destination.exists():
            if _read_json(destination) != payload:
                raise ValueError("content-addressed release manifest changed")
            return
        _write_atomic_json(destination, payload)

    def _stage_locked(
        self,
        *,
        joint_version: JointVersion,
        policy: CandidateArtifact,
        harness: CandidateArtifact,
        parent_release_id: str | None,
    ) -> ReleaseManifest:
        policy_object = self._write_object(policy)
        harness_object = self._write_object(harness)
        manifest = ReleaseManifest.create(
            parent_release_id=parent_release_id,
            joint_version=joint_version,
            policy_object=policy_object,
            harness_object=harness_object,
        )
        self._write_manifest(manifest)
        self._validate_manifest_objects(manifest)
        return manifest

    def _write_object(self, artifact: CandidateArtifact) -> str:
        artifact.validate()
        relative = f"objects/{artifact.component}-{artifact.digest}.json"
        destination = self.root / relative
        if destination.exists():
            existing = _read_json(destination)
            if existing != artifact.to_dict():
                raise ValueError("content-addressed candidate object changed")
        else:
            _write_atomic_json(destination, artifact.to_dict())
        return relative

    def _read_object(self, relative: str, expected_component: str) -> CandidateArtifact:
        path = (self.root / relative).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError("release object path escapes the store")
        payload = _read_json(path)
        artifact = CandidateArtifact.from_dict(payload)
        if artifact.component != expected_component:
            raise ValueError("release manifest references the wrong component type")
        expected_name = f"{artifact.component}-{artifact.digest}.json"
        if path.name != expected_name:
            raise ValueError("release object hash does not match its filename")
        return artifact

    def _validate_manifest_objects(self, manifest: ReleaseManifest) -> None:
        policy = self._read_object(manifest.policy_object, "policy")
        harness = self._read_object(manifest.harness_object, "harness")
        if policy.version != manifest.joint_version.policy:
            raise ValueError("active policy object differs from JointVersion")
        if harness.version != manifest.joint_version.harness_controller:
            raise ValueError("active Harness object differs from JointVersion")

    def read_active(self) -> ReleaseManifest | None:
        if not self.active_path.exists():
            return None
        manifest = ReleaseManifest.from_dict(_read_json(self.active_path))
        self._validate_manifest_objects(manifest)
        return manifest

    def read_manifest(self, release_id: str) -> ReleaseManifest:
        """Read and validate an active or staged content-addressed manifest."""

        manifest = ReleaseManifest.from_dict(
            _read_json(self._manifest_path(release_id))
        )
        if manifest.release_id != release_id:
            raise ValueError("release manifest filename differs from its contents")
        self._validate_manifest_objects(manifest)
        return manifest

    def read_artifacts(
        self,
        release_id: str,
    ) -> tuple[CandidateArtifact, CandidateArtifact]:
        """Read the exact content-addressed Policy and Harness objects."""

        manifest = self.read_manifest(release_id)
        return (
            self._read_object(manifest.policy_object, "policy"),
            self._read_object(manifest.harness_object, "harness"),
        )

    def stage(
        self,
        *,
        joint_version: JointVersion,
        policy: CandidateArtifact,
        harness: CandidateArtifact,
        expected_active_release_id: str | None,
    ) -> ReleaseManifest:
        """Persist a candidate pair without changing the active pointer.

        The expected active release is compared while holding the publication
        lock.  A successfully staged manifest is therefore bound to the parent
        observed by the caller, but it is not visible as active until
        :meth:`activate` performs a second compare-and-swap.
        """

        self._validate_candidate_pair(
            joint_version=joint_version,
            policy=policy,
            harness=harness,
        )
        with self._exclusive_lock():
            active = self.read_active()
            active_id = active.release_id if active else None
            if active_id != expected_active_release_id:
                raise ConcurrentPublishError(
                    "active release changed since candidate preparation"
                )
            return self._stage_locked(
                joint_version=joint_version,
                policy=policy,
                harness=harness,
                parent_release_id=active_id,
            )

    def activate(
        self,
        *,
        release_id: str,
        expected_active_release_id: str | None,
    ) -> ReleaseManifest:
        """CAS the active pointer from the staged manifest's parent to it."""

        with self._exclusive_lock():
            active = self.read_active()
            active_id = active.release_id if active else None
            if active_id != expected_active_release_id:
                raise ConcurrentPublishError(
                    "active release changed before candidate activation"
                )
            candidate = self.read_manifest(release_id)
            if candidate.parent_release_id != expected_active_release_id:
                raise ConcurrentPublishError(
                    "staged candidate was prepared from a different parent"
                )
            _write_atomic_json(self.active_path, candidate.to_dict())
            return candidate

    def rollback(
        self,
        *,
        target_release_id: str,
        expected_active_release_id: str,
    ) -> ReleaseManifest:
        """CAS the active pointer from a candidate back to its direct parent."""

        with self._exclusive_lock():
            active = self.read_active()
            active_id = active.release_id if active else None
            if active_id != expected_active_release_id:
                raise ConcurrentPublishError("active release changed before rollback")
            if active is None or active.parent_release_id != target_release_id:
                raise ConcurrentPublishError(
                    "rollback target is not the active release's direct parent"
                )
            parent = self.read_manifest(target_release_id)
            _write_atomic_json(self.active_path, parent.to_dict())
            return parent

    def validate_checkpoint(self, checkpoint: JointCheckpoint) -> ReleaseManifest:
        checkpoint.validate()
        active = self.read_active()
        if active is None:
            raise ValueError("checkpoint release store has no active manifest")
        if active.release_id != checkpoint.active_release_id:
            raise ValueError("checkpoint active release ID differs from the store")
        if active.joint_version != checkpoint.joint_version:
            raise ValueError("checkpoint JointVersion differs from the active manifest")
        policy = self._read_object(active.policy_object, "policy")
        harness = self._read_object(active.harness_object, "harness")
        if _canonical_json(policy.payload) != _canonical_json(
            checkpoint.policy.to_dict()
        ):
            raise ValueError("checkpoint policy state differs from the active object")
        if _canonical_json(harness.payload) != _canonical_json(
            checkpoint.harness.to_dict()
        ):
            raise ValueError("checkpoint Harness state differs from the active object")
        return active

    def publish(
        self,
        *,
        joint_version: JointVersion,
        policy: CandidateArtifact,
        harness: CandidateArtifact,
        expected_active_release_id: str | None,
        fault_at: str | None = None,
        fault_mode: str = "raise",
    ) -> ReleaseManifest:
        if fault_at is not None and fault_at not in self._FAULT_PHASES:
            raise ValueError(f"unknown publish fault phase: {fault_at}")
        if fault_mode not in {"raise", "exit"}:
            raise ValueError("fault mode must be raise or exit")
        self._validate_candidate_pair(
            joint_version=joint_version,
            policy=policy,
            harness=harness,
        )

        with self._exclusive_lock():
            active = self.read_active()
            active_id = active.release_id if active else None
            if active_id != expected_active_release_id:
                raise ConcurrentPublishError(
                    "active release changed since candidate preparation"
                )

            policy_object = self._write_object(policy)
            if fault_at == "after_policy_object":
                self._fail(fault_at, fault_mode)
            harness_object = self._write_object(harness)
            if fault_at == "after_harness_object":
                self._fail(fault_at, fault_mode)

            manifest = ReleaseManifest.create(
                parent_release_id=active_id,
                joint_version=joint_version,
                policy_object=policy_object,
                harness_object=harness_object,
            )
            self._write_manifest(manifest)
            self._validate_manifest_objects(manifest)
            if fault_at == "before_active_switch":
                self._fail(fault_at, fault_mode)
            _write_atomic_json(self.active_path, manifest.to_dict())
            if fault_at == "after_active_switch":
                self._fail(fault_at, fault_mode)
            return manifest

    @staticmethod
    def _fail(phase: str, mode: str) -> None:
        if mode == "exit":
            os._exit(91)
        raise InjectedPublishFailure(phase)
