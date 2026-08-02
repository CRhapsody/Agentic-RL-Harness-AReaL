from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

from .paths import require_within_configured_root
from .trajectory.schema import JointVersion


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


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
            raise ValueError("parameters and optimizer momentum must have equal non-zero length")
        if not all(math.isfinite(value) for value in (*self.parameters, *self.optimizer_momentum)):
            raise ValueError("checkpoint tensors must be finite")
        if type(self.optimizer_step) is not int or self.optimizer_step < 0:
            raise ValueError("optimizer step must be a non-negative integer")
        if type(self.rng_state) is not int or not 0 <= self.rng_state < 2**31:
            raise ValueError("component RNG state must be a 31-bit non-negative integer")
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
            raise ValueError("component checkpoint state hash does not match its contents")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComponentCheckpoint:
        checkpoint = cls(
            component=str(payload["component"]),
            version=str(payload["version"]),
            parameters=tuple(float(value) for value in payload["parameters"]),
            optimizer_momentum=tuple(
                float(value) for value in payload["optimizer_momentum"]
            ),
            optimizer_step=int(payload["optimizer_step"]),
            rng_state=int(payload["rng_state"]),
            sample_count=int(payload["sample_count"]),
            state_sha256=str(payload["state_sha256"]),
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
        version_payload = payload["joint_version"]
        if not isinstance(version_payload, Mapping):
            raise ValueError("joint_version must be an object")
        checkpoint = cls(
            schema_version=str(payload["schema_version"]),
            joint_version=JointVersion(**version_payload),
            active_release_id=str(payload["active_release_id"]),
            macro_step=int(payload["macro_step"]),
            policy=ComponentCheckpoint.from_dict(payload["policy"]),
            harness=ComponentCheckpoint.from_dict(payload["harness"]),
            rng_state=int(payload["rng_state"]),
            rollout_cursor=int(payload["rollout_cursor"]),
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
    if envelope.get("envelope_schema") != "jph.joint-checkpoint-envelope.v1":
        raise ValueError("joint checkpoint envelope schema differs")
    checkpoint_payload = envelope.get("checkpoint")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("joint checkpoint envelope has no checkpoint object")
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
        if self.component not in {"policy", "harness"}:
            raise ValueError("candidate component must be policy or harness")
        if not self.version:
            raise ValueError("candidate version must be non-empty")
        if not isinstance(self.payload, Mapping):
            raise ValueError("candidate payload must be an object")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "component": self.component,
            "version": self.version,
            "payload": dict(self.payload),
        }

    @property
    def digest(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class ReleaseManifest:
    release_id: str
    parent_release_id: str | None
    joint_version: JointVersion
    policy_object: str
    harness_object: str

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
        unsigned = {
            "parent_release_id": parent_release_id,
            "joint_version": asdict(joint_version),
            "policy_object": policy_object,
            "harness_object": harness_object,
        }
        return cls(
            release_id=_sha256(unsigned)[:20],
            parent_release_id=parent_release_id,
            joint_version=joint_version,
            policy_object=policy_object,
            harness_object=harness_object,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReleaseManifest:
        version_payload = payload["joint_version"]
        if not isinstance(version_payload, Mapping):
            raise ValueError("manifest joint_version must be an object")
        manifest = cls(
            release_id=str(payload["release_id"]),
            parent_release_id=(
                None
                if payload.get("parent_release_id") is None
                else str(payload["parent_release_id"])
            ),
            joint_version=JointVersion(**version_payload),
            policy_object=str(payload["policy_object"]),
            harness_object=str(payload["harness_object"]),
        )
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

    _FAULT_PHASES = {
        "after_policy_object",
        "after_harness_object",
        "before_active_switch",
        "after_active_switch",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = require_within_configured_root(root)
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests"
        self.active_path = self.root / "active.json"
        self.lock_path = self.root / ".publish.lock"

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
        artifact = CandidateArtifact(
            component=str(payload["component"]),
            version=str(payload["version"]),
            payload=payload["payload"],
        )
        artifact.validate()
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
        if _canonical_json(policy.payload) != _canonical_json(checkpoint.policy.to_dict()):
            raise ValueError("checkpoint policy state differs from the active object")
        if _canonical_json(harness.payload) != _canonical_json(checkpoint.harness.to_dict()):
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
        policy.validate()
        harness.validate()
        if policy.component != "policy" or harness.component != "harness":
            raise ValueError("publish candidates are swapped")
        if policy.version != joint_version.policy:
            raise ValueError("policy candidate differs from JointVersion")
        if harness.version != joint_version.harness_controller:
            raise ValueError("Harness candidate differs from JointVersion")

        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.objects.mkdir(parents=True, exist_ok=True)
        self.manifests.mkdir(parents=True, exist_ok=True)
        os.chmod(self.objects, 0o700)
        os.chmod(self.manifests, 0o700)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(descriptor, "r+") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
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
                manifest_path = self.manifests / f"{manifest.release_id}.json"
                _write_atomic_json(manifest_path, manifest.to_dict())
                self._validate_manifest_objects(manifest)
                if fault_at == "before_active_switch":
                    self._fail(fault_at, fault_mode)
                _write_atomic_json(self.active_path, manifest.to_dict())
                if fault_at == "after_active_switch":
                    self._fail(fault_at, fault_mode)
                return manifest
        finally:
            os.chmod(self.lock_path, 0o600)

    @staticmethod
    def _fail(phase: str, mode: str) -> None:
        if mode == "exit":
            os._exit(91)
        raise InjectedPublishFailure(phase)
