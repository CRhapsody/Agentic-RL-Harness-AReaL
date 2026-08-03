from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import socket
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from jphrl.paths import (
    configured_root,
    require_outside_repository,
    require_within_configured_root,
)
from jphrl.trajectory.schema import JointVersion

from .areal_policy_candidate import checkpoint_manifest
from .joint_step import validate_joint_candidate_bundle

CHECKPOINT_SCHEMA_VERSION = "jph.production-joint-checkpoint.v1"
RANK_STATE_SCHEMA_VERSION = "jph.production-rank-runtime-state.v1"
SCHEDULER_SCHEMA_VERSION = "jph.production-lr-scheduler-state.v1"
CURSOR_SCHEMA_VERSION = "jph.production-runtime-cursor.v1"
RECOVERY_EVIDENCE_SCHEMA_VERSION = "jph.production-joint-recovery-evidence.v2"
DRY_LOAD_PROBE_SCHEMA_VERSION = "jph.production-joint-dry-load-probe.v1"

_LIVE_EXACT_RECOVERY_TOKEN = object()

_SECRET_FIELDS = {
    "access_token",
    "admin_api_key",
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "session_api_key",
    "set_cookie",
    "token",
}
_CHECKPOINT_EVIDENCE = {
    "policy_dcp_with_optimizer_referenced": True,
    "harness_checkpoint_with_optimizer_referenced": True,
    "lr_scheduler_state_saved": True,
    "all_rank_rng_state_saved": True,
    "cursor_state_saved": True,
    "dry_load_probe_passed": False,
    "continuous_next_step_verified": False,
    "exact_joint_recovery": False,
}


class ProductionCheckpointError(RuntimeError):
    """Raised when a production joint checkpoint cannot fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionCheckpointError(message)


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
        raise ProductionCheckpointError(
            "production checkpoint is not finite canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _record_sha256(record: Mapping[str, object]) -> str:
    return _sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                str(key).lower() not in _SECRET_FIELDS,
                f"credential field cannot enter production checkpoint: {path}.{key}",
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
    value = _exact(value, set(JointVersion.__dataclass_fields__), label)
    try:
        version = JointVersion(**dict(value))
    except TypeError as exc:
        raise ProductionCheckpointError(f"{label} is invalid") from exc
    _require(
        all(
            isinstance(getattr(version, field), str) and bool(getattr(version, field))
            for field in JointVersion.__dataclass_fields__
        ),
        f"{label} fields must be non-empty",
    )
    return version


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_json(record))
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, object]:
    _require(
        path.is_file() and not path.is_symlink(), f"checkpoint file is missing: {path}"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionCheckpointError(f"cannot read checkpoint file: {path}") from exc
    _require(isinstance(value, dict), f"checkpoint file is not an object: {path}")
    return value


def _file_reference(
    path: Path, *, root: Path, schema_version: str
) -> dict[str, object]:
    _require(
        path.is_file() and not path.is_symlink(), "checkpoint state file is unsafe"
    )
    return {
        "path": path.relative_to(root).as_posix(),
        "schema_version": schema_version,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _external_file_reference(path: str | Path) -> dict[str, object]:
    source = require_within_configured_root(path)
    _require(
        source.is_file() and not source.is_symlink(), "component checkpoint is unsafe"
    )
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _file_sha256(source),
    }


def _prepare_target(
    checkpoint_root: str | Path,
    project_root: str | Path,
) -> tuple[Path, Path]:
    del project_root  # safety is derived from this module's actual Git checkout
    configured = configured_root()
    _require(configured is not None, "JPH_ROOT must be set for production checkpoints")
    target = require_outside_repository(checkpoint_root)
    _require(not target.exists(), "production checkpoint path already exists")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    ).resolve()
    os.chmod(temporary, 0o700)
    return target, temporary


@dataclass(frozen=True)
class RuntimeTopology:
    world_size: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    rank_to_device: tuple[str, ...]

    def validate(self) -> None:
        for field in (
            "world_size",
            "data_parallel_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
        ):
            value = getattr(self, field)
            _require(type(value) is int and value > 0, f"topology {field} is invalid")
        _require(
            self.world_size
            == self.data_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size,
            "topology parallel sizes do not multiply to world size",
        )
        _require(
            len(self.rank_to_device) == self.world_size
            and all(isinstance(item, str) and item for item in self.rank_to_device),
            "topology rank/device mapping is incomplete",
        )

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {
            "world_size": self.world_size,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "rank_to_device": list(self.rank_to_device),
        }

    @classmethod
    def from_record(cls, value: object) -> RuntimeTopology:
        record = _exact(
            value,
            {
                "world_size",
                "data_parallel_size",
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "rank_to_device",
            },
            "runtime topology",
        )
        mapping = record.get("rank_to_device")
        _require(
            isinstance(mapping, list), "topology rank/device mapping must be a list"
        )
        topology = cls(
            world_size=record.get("world_size"),
            data_parallel_size=record.get("data_parallel_size"),
            tensor_parallel_size=record.get("tensor_parallel_size"),
            pipeline_parallel_size=record.get("pipeline_parallel_size"),
            rank_to_device=tuple(mapping),
        )
        topology.validate()
        return topology


@dataclass(frozen=True)
class RuntimeCursorState:
    """Identity-bearing position of one checkpointable M0 input stream.

    ``source_sha256`` binds the immutable stream and ``pending_item_sha256``
    binds the exact item that the first continuation step must consume.  The
    nullable form exists only to read checkpoints created through the old
    integer-only API; it is deliberately ineligible for live exact recovery.
    """

    name: str
    position: int
    source_sha256: str | None
    pending_item_sha256: str | None

    @property
    def live_exact_eligible(self) -> bool:
        return _is_sha256(self.source_sha256) and _is_sha256(self.pending_item_sha256)

    def validate(self) -> None:
        _require(
            self.name in {"rollout", "dataloader"},
            "runtime cursor name is invalid",
        )
        _require(
            type(self.position) is int and self.position >= 0,
            f"{self.name} cursor position is invalid",
        )
        _require(
            (self.source_sha256 is None and self.pending_item_sha256 is None)
            or self.live_exact_eligible,
            f"{self.name} cursor identity hashes are invalid",
        )

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": CURSOR_SCHEMA_VERSION,
            "name": self.name,
            "position": self.position,
            "source_sha256": self.source_sha256,
            "pending_item_sha256": self.pending_item_sha256,
        }

    @classmethod
    def from_record(cls, value: object, *, expected_name: str) -> RuntimeCursorState:
        record = _exact(
            value,
            {
                "schema_version",
                "name",
                "position",
                "source_sha256",
                "pending_item_sha256",
            },
            f"{expected_name} runtime cursor",
        )
        _require(
            record.get("schema_version") == CURSOR_SCHEMA_VERSION,
            f"{expected_name} cursor schema differs",
        )
        cursor = cls(
            name=record.get("name"),
            position=record.get("position"),
            source_sha256=record.get("source_sha256"),
            pending_item_sha256=record.get("pending_item_sha256"),
        )
        cursor.validate()
        _require(cursor.name == expected_name, f"{expected_name} cursor name differs")
        return cursor

    @classmethod
    def legacy(cls, *, name: str, position: int) -> RuntimeCursorState:
        cursor = cls(
            name=name,
            position=position,
            source_sha256=None,
            pending_item_sha256=None,
        )
        cursor.validate()
        return cursor


@dataclass(frozen=True)
class RankRuntimeState:
    rank: int
    local_rank: int
    device: str
    hostname: str
    python_random_state: object
    numpy_rng: Mapping[str, object]
    torch_rng: Mapping[str, object]
    harness_generator_state: list[int] | None

    def to_record(self) -> dict[str, object]:
        record = {
            "schema_version": RANK_STATE_SCHEMA_VERSION,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": self.device,
            "hostname": self.hostname,
            "python_random_state": deepcopy(self.python_random_state),
            "numpy_rng": deepcopy(dict(self.numpy_rng)),
            "torch_rng": deepcopy(dict(self.torch_rng)),
            "harness_generator_state": deepcopy(self.harness_generator_state),
        }
        _validate_rank_state_record(record)
        return record


def _to_json_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_to_json_state(item) for item in value]
    if isinstance(value, list):
        return [_to_json_state(item) for item in value]
    if isinstance(value, Mapping):
        _require(
            all(isinstance(key, str) for key in value),
            "runtime state mapping keys must be strings",
        )
        return {str(key): _to_json_state(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        tensor = value.detach().cpu()
        return {
            "__torch_tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data": tensor.tolist(),
        }
    raise ProductionCheckpointError(
        f"runtime state contains unsupported value {type(value).__qualname__}"
    )


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def capture_rank_runtime_state(
    *,
    rank: int,
    local_rank: int,
    device: str,
    harness_policy: object | None = None,
) -> RankRuntimeState:
    """Capture process and owned Harness RNG on the rank that owns them."""

    _require(type(rank) is int and rank >= 0, "rank must be non-negative")
    _require(
        type(local_rank) is int and local_rank >= 0, "local rank must be non-negative"
    )
    _require(isinstance(device, str) and device, "rank device is missing")
    try:
        import numpy as np
    except ModuleNotFoundError:  # optional dependency, recorded explicitly
        numpy_rng: dict[str, object] = {"available": False, "state": None}
    else:
        state = np.random.get_state()
        numpy_rng = {
            "available": True,
            "state": {
                "bit_generator": str(state[0]),
                "keys": state[1].tolist(),
                "position": int(state[2]),
                "has_gauss": int(state[3]),
                "cached_gaussian": float(state[4]),
            },
        }
    try:
        import torch
    except ModuleNotFoundError:
        torch_rng: dict[str, object] = {
            "available": False,
            "cpu_state": None,
            "cuda_states": [],
        }
    else:
        cuda_states = (
            [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )
        torch_rng = {
            "available": True,
            "cpu_state": torch.get_rng_state().cpu().tolist(),
            "cuda_states": cuda_states,
        }
    generator_state = None
    if harness_policy is not None:
        generator = getattr(harness_policy, "_generator", None)
        get_state = getattr(generator, "get_state", None)
        _require(callable(get_state), "Harness policy generator is unavailable")
        generator_state = get_state().cpu().tolist()
    state = RankRuntimeState(
        rank=rank,
        local_rank=local_rank,
        device=device,
        hostname=socket.gethostname(),
        python_random_state=_to_json_state(random.getstate()),
        numpy_rng=numpy_rng,
        torch_rng=torch_rng,
        harness_generator_state=generator_state,
    )
    state.to_record()
    return state


def _validate_byte_state(value: object, label: str) -> None:
    _require(
        isinstance(value, list)
        and bool(value)
        and all(type(item) is int and 0 <= item <= 255 for item in value),
        f"{label} is invalid",
    )


def _validate_rank_state_record(record: Mapping[str, object]) -> None:
    _require(
        set(record)
        == {
            "schema_version",
            "rank",
            "local_rank",
            "device",
            "hostname",
            "python_random_state",
            "numpy_rng",
            "torch_rng",
            "harness_generator_state",
        },
        "rank runtime state field set differs",
    )
    _require(
        record.get("schema_version") == RANK_STATE_SCHEMA_VERSION,
        "rank runtime state schema differs",
    )
    _require(
        type(record.get("rank")) is int
        and record["rank"] >= 0
        and type(record.get("local_rank")) is int
        and record["local_rank"] >= 0
        and isinstance(record.get("device"), str)
        and bool(record["device"])
        and isinstance(record.get("hostname"), str)
        and bool(record["hostname"]),
        "rank runtime identity is invalid",
    )
    python_state = record.get("python_random_state")
    _require(
        isinstance(python_state, list) and bool(python_state),
        "Python RNG state is missing",
    )
    try:
        random.Random().setstate(_nested_tuple(python_state))
    except (TypeError, ValueError) as exc:
        raise ProductionCheckpointError("Python RNG state is invalid") from exc
    numpy_rng = _exact(record.get("numpy_rng"), {"available", "state"}, "NumPy RNG")
    _require(
        type(numpy_rng.get("available")) is bool, "NumPy availability must be boolean"
    )
    if numpy_rng["available"]:
        state = _exact(
            numpy_rng.get("state"),
            {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"},
            "NumPy RNG state",
        )
        _require(
            isinstance(state.get("bit_generator"), str)
            and isinstance(state.get("keys"), list)
            and bool(state["keys"]),
            "NumPy RNG state is invalid",
        )
    else:
        _require(
            numpy_rng.get("state") is None, "unavailable NumPy must not carry state"
        )
    torch_rng = _exact(
        record.get("torch_rng"),
        {"available", "cpu_state", "cuda_states"},
        "Torch RNG",
    )
    _require(
        type(torch_rng.get("available")) is bool, "Torch availability must be boolean"
    )
    _require(
        isinstance(torch_rng.get("cuda_states"), list), "CUDA RNG states must be a list"
    )
    if torch_rng["available"]:
        _validate_byte_state(torch_rng.get("cpu_state"), "Torch CPU RNG state")
        for state in torch_rng["cuda_states"]:
            _validate_byte_state(state, "Torch CUDA RNG state")
    else:
        _require(
            torch_rng.get("cpu_state") is None and torch_rng.get("cuda_states") == [],
            "unavailable Torch must not carry RNG state",
        )
    generator = record.get("harness_generator_state")
    if generator is not None:
        _validate_byte_state(generator, "Harness generator state")
    _assert_no_secrets(record)
    _canonical_json(record)


def _scheduler_state(actor: object) -> dict[str, object]:
    scheduler = getattr(actor, "lr_scheduler", None)
    state_dict = getattr(scheduler, "state_dict", None)
    _require(callable(state_dict), "AReaL lr scheduler state is unavailable")
    state = _to_json_state(state_dict())
    _assert_no_secrets(state, "lr_scheduler")
    record = {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "scheduler_class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "state": state,
        "state_sha256": _sha256(state),
    }
    return record


def _real_areal_actor(actor: object) -> bool:
    actor_type = type(actor)
    if (
        actor_type.__module__ != "areal.engine.fsdp_engine"
        or actor_type.__qualname__ != "FSDPPPOActor"
    ):
        return False
    try:
        from areal.engine.fsdp_engine import FSDPPPOActor
    except (ImportError, ModuleNotFoundError):
        return False
    return type(actor) is FSDPPPOActor


def _real_harness_policy(policy: object) -> bool:
    try:
        from jphrl.harness.torch_learning import TorchHarnessPolicy
    except ModuleNotFoundError:
        return False
    return type(policy) is TorchHarnessPolicy


def _extract_component_records(
    bundle: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    receipts = _exact(
        bundle.get("receipts"),
        {"policy", "policy_sha256", "harness", "harness_sha256"},
        "joint bundle receipts",
    )
    policy = receipts.get("policy")
    harness = receipts.get("harness")
    _require(
        isinstance(policy, Mapping) and isinstance(harness, Mapping),
        "joint bundle component receipts are missing",
    )
    return policy, harness


@dataclass(frozen=True)
class ValidatedProductionCheckpoint:
    manifest_path: str
    parent_joint_version: JointVersion
    candidate_joint_version: JointVersion
    macro_step_id: str
    macro_step: int
    source_joint_credit_sha256: str
    bundle_sha256: str
    topology: RuntimeTopology
    rollout_cursor: int
    dataloader_cursor: int
    rollout_cursor_state: RuntimeCursorState
    dataloader_cursor_state: RuntimeCursorState
    actor_public_version: int
    actor_reserved_version: int
    real_areal_checkpoint: bool
    real_harness_checkpoint: bool
    exact_joint_recovery: bool
    record_sha256: str


def save_production_joint_checkpoint(
    *,
    checkpoint_root: str | Path,
    project_root: str | Path,
    joint_candidate_bundle: Mapping[str, object],
    actor: object,
    harness_policy: object | None,
    topology: RuntimeTopology,
    rank_states: Sequence[RankRuntimeState],
    macro_step: int,
    rollout_cursor: int | RuntimeCursorState,
    dataloader_cursor: int | RuntimeCursorState,
) -> Path:
    """Atomically save the missing state around real V component checkpoints."""

    topology.validate()
    _require(type(macro_step) is int and macro_step >= 0, "macro step is invalid")
    rollout_state = (
        RuntimeCursorState.legacy(name="rollout", position=rollout_cursor)
        if type(rollout_cursor) is int
        else rollout_cursor
    )
    dataloader_state = (
        RuntimeCursorState.legacy(name="dataloader", position=dataloader_cursor)
        if type(dataloader_cursor) is int
        else dataloader_cursor
    )
    _require(
        type(rollout_state) is RuntimeCursorState
        and type(dataloader_state) is RuntimeCursorState,
        "production cursors must use RuntimeCursorState or legacy integers",
    )
    rollout_state.validate()
    dataloader_state.validate()
    _require(
        rollout_state.name == "rollout" and dataloader_state.name == "dataloader",
        "production cursor roles differ",
    )
    _assert_no_secrets(joint_candidate_bundle)
    try:
        bundle = validate_joint_candidate_bundle(
            joint_candidate_bundle,
            actor_public_version=actor.get_version(),
            harness_public_version=(
                joint_candidate_bundle["parent"]["harness_controller_version"]
            ),
            require_receipt_files=True,
        )
    except (ValueError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
        raise ProductionCheckpointError(str(exc)) from exc
    _require(
        len(rank_states) == topology.world_size,
        "rank RNG count differs from topology",
    )
    ordered_states = sorted(rank_states, key=lambda state: state.rank)
    _require(
        [state.rank for state in ordered_states] == list(range(topology.world_size)),
        "rank RNG states are not contiguous",
    )
    for state in ordered_states:
        _require(
            state.device == topology.rank_to_device[state.rank],
            "rank RNG device differs from topology",
        )

    policy_receipt, harness_receipt = _extract_component_records(joint_candidate_bundle)
    policy_checkpoints = _exact(
        policy_receipt.get("checkpoints"),
        {"parent_path", "parent_manifest", "candidate_path", "candidate_manifest"},
        "Policy candidate checkpoints",
    )
    for kind in ("parent", "candidate"):
        _require(
            checkpoint_manifest(policy_checkpoints[f"{kind}_path"])
            == policy_checkpoints[f"{kind}_manifest"],
            f"Policy {kind} DCP differs from its real manifest",
        )
    harness_path = harness_receipt.get("checkpoint_path")
    harness_sha256 = harness_receipt.get("checkpoint_sha256")
    _require(
        isinstance(harness_path, str)
        and _is_sha256(harness_sha256)
        and _file_sha256(require_within_configured_root(harness_path))
        == harness_sha256,
        "Harness checkpoint reference is invalid",
    )
    public_version = actor.get_version()
    _require(
        public_version == bundle.policy_engine_version,
        "actor public version differs from V parent",
    )
    scheduler_record = _scheduler_state(actor)

    target, temporary = _prepare_target(checkpoint_root, project_root)
    try:
        state_dir = temporary / "rank-state"
        state_dir.mkdir(mode=0o700)
        rank_references: list[dict[str, object]] = []
        for state in ordered_states:
            path = state_dir / f"rank-{state.rank:05d}.json"
            _write_json(path, state.to_record())
            rank_references.append(
                {
                    "rank": state.rank,
                    "local_rank": state.local_rank,
                    "device": state.device,
                    "file": _file_reference(
                        path,
                        root=temporary,
                        schema_version=RANK_STATE_SCHEMA_VERSION,
                    ),
                }
            )
        scheduler_path = temporary / "policy-lr-scheduler.json"
        _write_json(scheduler_path, scheduler_record)
        scheduler_reference = _file_reference(
            scheduler_path,
            root=temporary,
            schema_version=SCHEDULER_SCHEMA_VERSION,
        )
        harness_reference = _external_file_reference(harness_path)
        real_areal = _real_areal_actor(actor)
        real_harness = _real_harness_policy(harness_policy)
        if real_harness:
            from jphrl.harness.torch_learning import (
                load_torch_harness_checkpoint,
            )

            loaded_policy, loaded_optimizer, _ = load_torch_harness_checkpoint(
                harness_path,
                map_location="cpu",
            )
            _require(
                getattr(harness_policy, "version", None)
                == bundle.candidate_joint_version.harness_controller,
                "Harness candidate object differs from V candidate",
            )
            _require(
                loaded_policy.version == harness_policy.version
                and loaded_policy.parameter_digest == harness_policy.parameter_digest
                and bool(loaded_optimizer.state),
                "real Harness checkpoint dry load differs from candidate",
            )
            rank_zero_generator = ordered_states[0].harness_generator_state
            current_generator = harness_policy._generator.get_state().cpu().tolist()
            _require(
                rank_zero_generator == current_generator,
                "Harness generator differs from rank-zero runtime state",
            )
        manifest: dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": {
                "parent_joint_version": asdict(bundle.parent_joint_version),
                "parent_joint_version_id": bundle.parent_joint_version.version_id,
                "candidate_joint_version": asdict(bundle.candidate_joint_version),
                "candidate_joint_version_id": bundle.candidate_joint_version.version_id,
                "macro_step_id": bundle.macro_step_id,
                "macro_step": macro_step,
                "source_joint_credit_sha256": bundle.source_joint_credit_sha256,
                "joint_candidate_bundle_sha256": bundle.record_sha256,
            },
            "joint_candidate_bundle": deepcopy(dict(joint_candidate_bundle)),
            "topology": topology.to_record(),
            "cursors": {
                "rollout": rollout_state.to_record(),
                "dataloader": dataloader_state.to_record(),
            },
            "policy": {
                "actor_class": f"{type(actor).__module__}.{type(actor).__qualname__}",
                "real_areal_fsdp_actor": real_areal,
                "actor_public_version": public_version,
                "actor_reserved_version": bundle.candidate_policy_engine_version,
                "parent_dcp_path": policy_checkpoints["parent_path"],
                "parent_dcp_manifest": policy_checkpoints["parent_manifest"],
                "candidate_dcp_path": policy_checkpoints["candidate_path"],
                "candidate_dcp_manifest": policy_checkpoints["candidate_manifest"],
                "dcp_with_optimizer": True,
                "lr_scheduler": scheduler_reference,
            },
            "harness": {
                "policy_class": (
                    None
                    if harness_policy is None
                    else f"{type(harness_policy).__module__}.{type(harness_policy).__qualname__}"
                ),
                "real_torch_harness_policy": real_harness,
                "behavior_version": bundle.parent_joint_version.harness_controller,
                "candidate_version": bundle.candidate_joint_version.harness_controller,
                "checkpoint": harness_reference,
                "checkpoint_schema_version": harness_receipt.get("schema_version"),
                "checkpoint_contains_optimizer": True,
                "checkpoint_contains_generator": True,
            },
            "rng": {
                "rank_states": rank_references,
                "python_saved_for_all_ranks": True,
                "numpy_optional_dependency_explicit": True,
                "torch_cpu_saved_for_all_available_ranks": True,
                "torch_cuda_saved_for_all_visible_devices_per_rank": True,
                "harness_generator_saved_on_rank_zero": (
                    ordered_states[0].harness_generator_state is not None
                ),
            },
            "evidence_scope": dict(_CHECKPOINT_EVIDENCE),
        }
        _assert_no_secrets(manifest)
        manifest["record_sha256"] = _record_sha256(manifest)
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    manifest_path = target / "manifest.json"
    validate_production_joint_checkpoint(
        manifest_path,
        current_topology=topology,
        require_component_files=True,
    )
    return manifest_path


def _validate_file_reference(
    value: object,
    *,
    checkpoint_root: Path,
    expected_schema: str,
) -> Path:
    reference = _exact(
        value,
        {"path", "schema_version", "size_bytes", "sha256"},
        "checkpoint file reference",
    )
    relative = reference.get("path")
    _require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        "checkpoint file path is invalid",
    )
    path = (checkpoint_root / relative).resolve()
    _require(checkpoint_root in path.parents, "checkpoint file escapes its root")
    _require(
        reference.get("schema_version") == expected_schema,
        "checkpoint file schema differs",
    )
    _require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == reference.get("size_bytes")
        and _file_sha256(path) == reference.get("sha256"),
        "checkpoint file hash or size mismatch",
    )
    return path


def validate_production_joint_checkpoint(
    manifest_path: str | Path,
    *,
    current_topology: RuntimeTopology | None = None,
    require_component_files: bool = True,
) -> ValidatedProductionCheckpoint:
    path = require_within_configured_root(manifest_path)
    root = path.parent
    record = _read_json(path)
    _require(
        set(record)
        == {
            "schema_version",
            "identity",
            "joint_candidate_bundle",
            "topology",
            "cursors",
            "policy",
            "harness",
            "rng",
            "evidence_scope",
            "record_sha256",
        },
        "production checkpoint field set differs",
    )
    _require(
        record.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "checkpoint schema differs",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "checkpoint manifest hash mismatch",
    )
    _assert_no_secrets(record)
    topology = RuntimeTopology.from_record(record.get("topology"))
    if current_topology is not None:
        current_topology.validate()
        _require(topology == current_topology, "checkpoint topology mismatch")
    identity = _exact(
        record.get("identity"),
        {
            "parent_joint_version",
            "parent_joint_version_id",
            "candidate_joint_version",
            "candidate_joint_version_id",
            "macro_step_id",
            "macro_step",
            "source_joint_credit_sha256",
            "joint_candidate_bundle_sha256",
        },
        "checkpoint identity",
    )
    parent_version = _joint_version(
        identity.get("parent_joint_version"), "parent JointVersion"
    )
    candidate_version = _joint_version(
        identity.get("candidate_joint_version"), "candidate JointVersion"
    )
    _require(
        identity.get("parent_joint_version_id") == parent_version.version_id
        and identity.get("candidate_joint_version_id") == candidate_version.version_id
        and parent_version != candidate_version,
        "checkpoint JointVersion identity differs",
    )
    _require(
        isinstance(identity.get("macro_step_id"), str)
        and bool(identity["macro_step_id"])
        and type(identity.get("macro_step")) is int
        and identity["macro_step"] >= 0
        and _is_sha256(identity.get("source_joint_credit_sha256"))
        and _is_sha256(identity.get("joint_candidate_bundle_sha256")),
        "checkpoint macro/source identity is invalid",
    )
    bundle_record = record.get("joint_candidate_bundle")
    _require(isinstance(bundle_record, Mapping), "checkpoint V bundle is missing")
    try:
        bundle = validate_joint_candidate_bundle(
            bundle_record,
            require_receipt_files=require_component_files,
        )
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise ProductionCheckpointError(str(exc)) from exc
    _require(
        identity.get("joint_candidate_bundle_sha256")
        == bundle.record_sha256
        == bundle_record.get("record_sha256")
        and bundle.parent_joint_version == parent_version
        and bundle.candidate_joint_version == candidate_version
        and bundle.macro_step_id == identity.get("macro_step_id")
        and bundle.source_joint_credit_sha256
        == identity.get("source_joint_credit_sha256"),
        "checkpoint differs from its V bundle",
    )
    cursors = _exact(
        record.get("cursors"),
        {"rollout", "dataloader"},
        "checkpoint cursors",
    )
    rollout_cursor_state = RuntimeCursorState.from_record(
        cursors.get("rollout"), expected_name="rollout"
    )
    dataloader_cursor_state = RuntimeCursorState.from_record(
        cursors.get("dataloader"), expected_name="dataloader"
    )
    policy = _exact(
        record.get("policy"),
        {
            "actor_class",
            "real_areal_fsdp_actor",
            "actor_public_version",
            "actor_reserved_version",
            "parent_dcp_path",
            "parent_dcp_manifest",
            "candidate_dcp_path",
            "candidate_dcp_manifest",
            "dcp_with_optimizer",
            "lr_scheduler",
        },
        "checkpoint Policy state",
    )
    _require(
        type(policy.get("real_areal_fsdp_actor")) is bool
        and policy.get("actor_public_version") == bundle.policy_engine_version
        and policy.get("actor_reserved_version")
        == bundle.candidate_policy_engine_version
        and policy.get("dcp_with_optimizer") is True,
        "checkpoint Policy version or optimizer reference differs",
    )
    for kind in ("parent", "candidate"):
        dcp_path = policy.get(f"{kind}_dcp_path")
        dcp_manifest = policy.get(f"{kind}_dcp_manifest")
        _require(
            isinstance(dcp_path, str) and isinstance(dcp_manifest, Mapping),
            f"Policy {kind} DCP reference is missing",
        )
        if require_component_files:
            _require(
                checkpoint_manifest(dcp_path) == dcp_manifest,
                f"Policy {kind} DCP manifest mismatch",
            )
    scheduler_path = _validate_file_reference(
        policy.get("lr_scheduler"),
        checkpoint_root=root,
        expected_schema=SCHEDULER_SCHEMA_VERSION,
    )
    scheduler = _read_json(scheduler_path)
    _require(
        set(scheduler) == {"schema_version", "scheduler_class", "state", "state_sha256"}
        and scheduler.get("schema_version") == SCHEDULER_SCHEMA_VERSION
        and scheduler.get("state_sha256") == _sha256(scheduler.get("state")),
        "lr scheduler state differs from schema or hash",
    )
    harness = _exact(
        record.get("harness"),
        {
            "real_torch_harness_policy",
            "policy_class",
            "behavior_version",
            "candidate_version",
            "checkpoint",
            "checkpoint_schema_version",
            "checkpoint_contains_optimizer",
            "checkpoint_contains_generator",
        },
        "checkpoint Harness state",
    )
    _require(
        type(harness.get("real_torch_harness_policy")) is bool
        and harness.get("behavior_version") == parent_version.harness_controller
        and harness.get("candidate_version") == candidate_version.harness_controller
        and harness.get("checkpoint_contains_optimizer") is True
        and harness.get("checkpoint_contains_generator") is True,
        "Harness checkpoint version or optimizer reference differs",
    )
    if policy["real_areal_fsdp_actor"]:
        _require(
            policy.get("actor_class") == "areal.engine.fsdp_engine.FSDPPPOActor",
            "real Policy checkpoint actor class differs",
        )
    if harness["real_torch_harness_policy"]:
        _require(
            harness.get("policy_class")
            == "jphrl.harness.torch_learning.TorchHarnessPolicy",
            "real Harness checkpoint policy class differs",
        )
    harness_reference = _exact(
        harness.get("checkpoint"),
        {"path", "size_bytes", "sha256"},
        "Harness checkpoint reference",
    )
    harness_path = harness_reference.get("path")
    _require(
        isinstance(harness_path, str) and _is_sha256(harness_reference.get("sha256")),
        "Harness checkpoint reference is invalid",
    )
    if require_component_files:
        source = require_within_configured_root(harness_path)
        _require(
            source.is_file()
            and not source.is_symlink()
            and source.stat().st_size == harness_reference.get("size_bytes")
            and _file_sha256(source) == harness_reference.get("sha256"),
            "Harness checkpoint file hash mismatch",
        )
        if harness["real_torch_harness_policy"]:
            from jphrl.harness.torch_learning import (
                load_torch_harness_checkpoint,
            )

            restored_harness, restored_optimizer, _ = load_torch_harness_checkpoint(
                source,
                map_location="cpu",
            )
            _require(
                restored_harness.version == candidate_version.harness_controller
                and bool(restored_optimizer.state),
                "real Harness checkpoint dry load differs from JointVersion",
            )
    rng = _exact(
        record.get("rng"),
        {
            "rank_states",
            "python_saved_for_all_ranks",
            "numpy_optional_dependency_explicit",
            "torch_cpu_saved_for_all_available_ranks",
            "torch_cuda_saved_for_all_visible_devices_per_rank",
            "harness_generator_saved_on_rank_zero",
        },
        "checkpoint RNG state",
    )
    rank_states = rng.get("rank_states")
    _require(
        isinstance(rank_states, list) and len(rank_states) == topology.world_size,
        "checkpoint rank RNG count differs from topology",
    )
    seen: set[int] = set()
    for item in rank_states:
        item = _exact(
            item, {"rank", "local_rank", "device", "file"}, "rank RNG reference"
        )
        rank = item.get("rank")
        _require(
            type(rank) is int and rank not in seen and 0 <= rank < topology.world_size,
            "rank RNG identity is invalid",
        )
        seen.add(rank)
        _require(
            item.get("device") == topology.rank_to_device[rank],
            "rank RNG device differs from topology",
        )
        rank_path = _validate_file_reference(
            item.get("file"),
            checkpoint_root=root,
            expected_schema=RANK_STATE_SCHEMA_VERSION,
        )
        state = _read_json(rank_path)
        _validate_rank_state_record(state)
        _require(
            state.get("rank") == rank
            and state.get("local_rank") == item.get("local_rank")
            and state.get("device") == item.get("device"),
            "rank RNG file identity differs",
        )
    _require(
        seen == set(range(topology.world_size)),
        "checkpoint rank RNG coverage is incomplete",
    )
    _require(
        all(
            rng.get(field) is True
            for field in (
                "python_saved_for_all_ranks",
                "numpy_optional_dependency_explicit",
                "torch_cpu_saved_for_all_available_ranks",
                "torch_cuda_saved_for_all_visible_devices_per_rank",
            )
        ),
        "checkpoint RNG completeness flags differ",
    )
    _require(
        record.get("evidence_scope") == _CHECKPOINT_EVIDENCE,
        "checkpoint evidence scope differs",
    )
    return ValidatedProductionCheckpoint(
        manifest_path=str(path),
        parent_joint_version=parent_version,
        candidate_joint_version=candidate_version,
        macro_step_id=str(identity["macro_step_id"]),
        macro_step=int(identity["macro_step"]),
        source_joint_credit_sha256=str(identity["source_joint_credit_sha256"]),
        bundle_sha256=str(identity["joint_candidate_bundle_sha256"]),
        topology=topology,
        rollout_cursor=rollout_cursor_state.position,
        dataloader_cursor=dataloader_cursor_state.position,
        rollout_cursor_state=rollout_cursor_state,
        dataloader_cursor_state=dataloader_cursor_state,
        actor_public_version=int(policy["actor_public_version"]),
        actor_reserved_version=int(policy["actor_reserved_version"]),
        real_areal_checkpoint=bool(policy["real_areal_fsdp_actor"]),
        real_harness_checkpoint=bool(harness["real_torch_harness_policy"]),
        exact_joint_recovery=False,
        record_sha256=str(record["record_sha256"]),
    )


def _restore_rank_rng(
    record: Mapping[str, object], *, harness_policy: object | None
) -> None:
    random.setstate(_nested_tuple(record["python_random_state"]))
    numpy_rng = record["numpy_rng"]
    if numpy_rng["available"]:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise ProductionCheckpointError(
                "NumPy RNG was saved but NumPy is unavailable"
            ) from exc
        state = numpy_rng["state"]
        np.random.set_state(
            (
                state["bit_generator"],
                np.asarray(state["keys"], dtype="uint32"),
                state["position"],
                state["has_gauss"],
                state["cached_gaussian"],
            )
        )
    torch_rng = record["torch_rng"]
    if torch_rng["available"]:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ProductionCheckpointError(
                "Torch RNG was saved but Torch is unavailable"
            ) from exc
        torch.set_rng_state(torch.tensor(torch_rng["cpu_state"], dtype=torch.uint8))
        if torch_rng["cuda_states"]:
            _require(
                torch.cuda.is_available(), "CUDA RNG was saved but CUDA is unavailable"
            )
            _require(
                len(torch_rng["cuda_states"]) == torch.cuda.device_count(),
                "saved CUDA RNG count differs from visible devices",
            )
            torch.cuda.set_rng_state_all(
                [
                    torch.tensor(state, dtype=torch.uint8)
                    for state in torch_rng["cuda_states"]
                ]
            )
        if record["harness_generator_state"] is not None:
            _require(
                harness_policy is not None,
                "Harness generator restore target is missing",
            )
            harness_policy._generator.set_state(
                torch.tensor(record["harness_generator_state"], dtype=torch.uint8)
            )


def _assert_rank_rng_restored(
    record: Mapping[str, object],
    *,
    harness_policy: object | None,
) -> None:
    _require(
        _to_json_state(random.getstate()) == record["python_random_state"],
        "Python RNG restore differs from checkpoint",
    )
    numpy_rng = record["numpy_rng"]
    if numpy_rng["available"]:
        import numpy as np

        current = np.random.get_state()
        expected = numpy_rng["state"]
        _require(
            str(current[0]) == expected["bit_generator"]
            and current[1].tolist() == expected["keys"]
            and int(current[2]) == expected["position"]
            and int(current[3]) == expected["has_gauss"]
            and float(current[4]) == expected["cached_gaussian"],
            "NumPy RNG restore differs from checkpoint",
        )
    torch_rng = record["torch_rng"]
    if torch_rng["available"]:
        import torch

        _require(
            torch.get_rng_state().cpu().tolist() == torch_rng["cpu_state"],
            "Torch CPU RNG restore differs from checkpoint",
        )
        if torch_rng["cuda_states"]:
            _require(
                [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
                == torch_rng["cuda_states"],
                "Torch CUDA RNG restore differs from checkpoint",
            )
        if record["harness_generator_state"] is not None:
            _require(
                harness_policy is not None
                and harness_policy._generator.get_state().cpu().tolist()
                == record["harness_generator_state"],
                "Harness generator restore differs from checkpoint",
            )


def _scheduler_from_json(value: object) -> object:
    if isinstance(value, list):
        return [_scheduler_from_json(item) for item in value]
    if isinstance(value, Mapping):
        if value.get("__torch_tensor__") is True:
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise ProductionCheckpointError(
                    "scheduler tensor requires Torch"
                ) from exc
            dtype_name = str(value["dtype"]).removeprefix("torch.")
            dtype = getattr(torch, dtype_name, None)
            _require(dtype is not None, "scheduler tensor dtype is unknown")
            tensor = torch.tensor(value["data"], dtype=dtype)
            return tensor.reshape(value["shape"])
        return {key: _scheduler_from_json(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class RestoredProductionCheckpoint:
    audit: ValidatedProductionCheckpoint
    harness_policy: object
    harness_optimizer: object
    rollout_cursor: int
    dataloader_cursor: int
    rollout_cursor_state: RuntimeCursorState
    dataloader_cursor_state: RuntimeCursorState
    exact_joint_recovery: bool = False


def restore_production_joint_checkpoint(
    manifest_path: str | Path,
    *,
    actor: object,
    current_topology: RuntimeTopology,
    rank: int,
    restore_rng: bool = True,
    require_real_components: bool = True,
) -> RestoredProductionCheckpoint:
    audit = validate_production_joint_checkpoint(
        manifest_path,
        current_topology=current_topology,
        require_component_files=True,
    )
    _require(
        type(rank) is int and 0 <= rank < audit.topology.world_size,
        "restore rank is invalid",
    )
    if require_real_components:
        _require(
            _real_areal_actor(actor) and audit.real_areal_checkpoint,
            "exact restore requires real AReaL FSDPPPOActor",
        )
        _require(
            audit.real_harness_checkpoint,
            "exact restore requires real Torch Harness checkpoint",
        )
    manifest = _read_json(Path(audit.manifest_path))
    policy = manifest["policy"]
    try:
        from areal.api import SaveLoadMeta
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProductionCheckpointError(
            "pinned AReaL SaveLoadMeta is unavailable"
        ) from exc
    actor.load(
        meta=SaveLoadMeta(
            path=policy["candidate_dcp_path"],
            weight_format="dcp",
            with_optim=True,
        )
    )
    scheduler_path = Path(audit.manifest_path).parent / policy["lr_scheduler"]["path"]
    scheduler_record = _read_json(scheduler_path)
    scheduler_class = (
        f"{type(actor.lr_scheduler).__module__}.{type(actor.lr_scheduler).__qualname__}"
    )
    _require(
        scheduler_class == scheduler_record["scheduler_class"],
        "lr scheduler restore target class differs from checkpoint",
    )
    actor.lr_scheduler.load_state_dict(_scheduler_from_json(scheduler_record["state"]))
    _require(
        _sha256(_to_json_state(actor.lr_scheduler.state_dict()))
        == scheduler_record["state_sha256"],
        "lr scheduler state restore differs from checkpoint",
    )
    actor.set_version(audit.actor_public_version)
    _require(
        actor.get_version() == audit.actor_public_version,
        "actor public version restore failed",
    )
    try:
        from jphrl.harness.torch_learning import load_torch_harness_checkpoint
    except ModuleNotFoundError as exc:
        raise ProductionCheckpointError("Torch Harness loader is unavailable") from exc
    harness_policy, harness_optimizer, _ = load_torch_harness_checkpoint(
        manifest["harness"]["checkpoint"]["path"], map_location="cpu"
    )
    _require(
        harness_policy.version == audit.candidate_joint_version.harness_controller,
        "restored Harness version differs from candidate JointVersion",
    )
    if restore_rng:
        reference = next(
            item for item in manifest["rng"]["rank_states"] if item["rank"] == rank
        )
        rank_record = _read_json(
            Path(audit.manifest_path).parent / reference["file"]["path"]
        )
        _restore_rank_rng(
            rank_record,
            harness_policy=harness_policy if rank == 0 else None,
        )
        _assert_rank_rng_restored(
            rank_record,
            harness_policy=harness_policy if rank == 0 else None,
        )
    return RestoredProductionCheckpoint(
        audit=audit,
        harness_policy=harness_policy,
        harness_optimizer=harness_optimizer,
        rollout_cursor=audit.rollout_cursor,
        dataloader_cursor=audit.dataloader_cursor,
        rollout_cursor_state=audit.rollout_cursor_state,
        dataloader_cursor_state=audit.dataloader_cursor_state,
    )


def dry_load_production_joint_checkpoint(
    manifest_path: str | Path,
    *,
    current_topology: RuntimeTopology,
    actor: object | None = None,
    rank: int = 0,
) -> dict[str, object]:
    """Validate every reference; optionally perform real component load without RNG."""

    audit = validate_production_joint_checkpoint(
        manifest_path,
        current_topology=current_topology,
        require_component_files=True,
    )
    real_probe = False
    if actor is not None:
        restore_production_joint_checkpoint(
            manifest_path,
            actor=actor,
            current_topology=current_topology,
            rank=rank,
            restore_rng=False,
            require_real_components=True,
        )
        real_probe = True
    record: dict[str, object] = {
        "schema_version": DRY_LOAD_PROBE_SCHEMA_VERSION,
        "ok": True,
        "manifest_sha256": audit.record_sha256,
        "component_files_verified": True,
        "real_component_dry_load": real_probe,
        "evidence_scope": {
            "dry_load_probe_passed": True,
            "real_component_dry_load": real_probe,
            "continuous_next_step_verified": False,
            "exact_joint_recovery": False,
        },
    }
    record["record_sha256"] = _record_sha256(record)
    return record


def _optimizer_step_from_state(state: object, label: str) -> int:
    _require(isinstance(state, Mapping), f"{label} state is unavailable")
    steps: list[int] = []
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping) or "step" not in parameter_state:
            continue
        raw_step = parameter_state["step"]
        if hasattr(raw_step, "item"):
            raw_step = raw_step.item()
        _require(
            isinstance(raw_step, (int, float))
            and not isinstance(raw_step, bool)
            and math.isfinite(float(raw_step))
            and float(raw_step).is_integer()
            and int(raw_step) >= 0,
            f"{label} contains an invalid optimizer step",
        )
        steps.append(int(raw_step))
    if not steps:
        return 0
    _require(len(set(steps)) == 1, f"{label} optimizer steps differ")
    return steps[0]


def _policy_optimizer_step(actor: object) -> int:
    optimizer = getattr(actor, "optimizer", None)
    return _optimizer_step_from_state(
        getattr(optimizer, "state", None), "AReaL optimizer"
    )


def _require_real_harness_optimizer(policy: object, optimizer: object) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ProductionCheckpointError(
            "Torch Harness optimizer is unavailable"
        ) from exc
    _require(_real_harness_policy(policy), "live recovery requires TorchHarnessPolicy")
    _require(
        type(optimizer) is torch.optim.Adam,
        "live recovery requires the real Harness Adam optimizer",
    )
    policy_parameters = tuple(policy.parameters())
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    _require(
        len(policy_parameters) == len(optimizer_parameters)
        and all(
            left is right
            for left, right in zip(policy_parameters, optimizer_parameters, strict=True)
        ),
        "Harness Adam optimizer is not bound to the restored policy",
    )
    _require(bool(optimizer.state), "Harness Adam optimizer state is empty")


def _harness_optimizer_step(policy: object, optimizer: object) -> int:
    _require_real_harness_optimizer(policy, optimizer)
    step = _optimizer_step_from_state(optimizer.state, "Harness Adam optimizer")
    _require(
        type(policy.update_step) is int and policy.update_step == step,
        "Harness policy update step differs from its Adam state",
    )
    return step


def _policy_probe_sha256(
    actor: object,
    admission_record: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
    device: str | object,
) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ProductionCheckpointError("Policy probe requires Torch") from exc
    from .areal_policy_optimizer import materialize_areal_ppo_update_tensors

    prepared = materialize_areal_ppo_update_tensors(
        admission_record,
        actor=actor,
        active_joint_version=active_joint_version,
        device=device,
    )
    compute_logp = getattr(actor, "compute_logp", None)
    _require(callable(compute_logp), "AReaL actor compute_logp is unavailable")
    outputs = compute_logp(prepared)
    _require(isinstance(outputs, list) and bool(outputs), "Policy probe is missing")
    values: list[list[float]] = []
    for output in outputs:
        _require(isinstance(output, torch.Tensor), "Policy probe is not a tensor")
        row = [float(item) for item in output.detach().float().cpu().reshape(-1)]
        _require(
            bool(row) and all(math.isfinite(item) for item in row),
            "Policy probe contains no finite log-probabilities",
        )
        values.append(row)
    return _sha256(values)


def _cursor_advance_record(cursor: RuntimeCursorState) -> dict[str, object]:
    _require(
        cursor.live_exact_eligible,
        f"legacy {cursor.name} cursor cannot prove live exact recovery",
    )
    return {
        "name": cursor.name,
        "source_sha256": cursor.source_sha256,
        "consumed_item_sha256": cursor.pending_item_sha256,
        "position": cursor.position + 1,
    }


def _validate_cursor_advance(
    value: object,
    *,
    cursor: RuntimeCursorState,
    label: str,
) -> Mapping[str, object]:
    record = _exact(
        value,
        {"name", "source_sha256", "consumed_item_sha256", "position"},
        label,
    )
    _require(
        cursor.live_exact_eligible
        and record.get("name") == cursor.name
        and record.get("source_sha256") == cursor.source_sha256
        and record.get("consumed_item_sha256") == cursor.pending_item_sha256
        and record.get("position") == cursor.position + 1,
        f"{label} does not consume the checkpoint pending item exactly once",
    )
    return record


def _validate_next_step(
    value: object,
    label: str,
    *,
    checkpoint: ValidatedProductionCheckpoint,
) -> Mapping[str, object]:
    record = _exact(
        value,
        {
            "policy_state_sha256",
            "harness_state_sha256",
            "harness_optimizer_state_sha256",
            "harness_sample_count",
            "harness_training",
            "policy_optimizer_step",
            "harness_optimizer_step",
            "lr_scheduler_state_sha256",
            "runtime_rng_state_sha256",
            "actor_public_version",
            "rollout_cursor",
            "dataloader_cursor",
        },
        label,
    )
    _require(
        all(
            _is_sha256(record.get(field))
            for field in (
                "policy_state_sha256",
                "harness_state_sha256",
                "harness_optimizer_state_sha256",
                "lr_scheduler_state_sha256",
                "runtime_rng_state_sha256",
            )
        ),
        f"{label} state digests are invalid",
    )
    _require(
        all(
            type(record.get(field)) is int and record[field] >= 0
            for field in (
                "policy_optimizer_step",
                "harness_optimizer_step",
                "harness_sample_count",
                "actor_public_version",
            )
        ),
        f"{label} counters are invalid",
    )
    _require(
        type(record.get("harness_training")) is bool,
        f"{label} Harness training mode is invalid",
    )
    _validate_cursor_advance(
        record.get("rollout_cursor"),
        cursor=checkpoint.rollout_cursor_state,
        label=f"{label} rollout cursor",
    )
    _validate_cursor_advance(
        record.get("dataloader_cursor"),
        cursor=checkpoint.dataloader_cursor_state,
        label=f"{label} dataloader cursor",
    )
    return record


def _saved_optimizer_steps(
    checkpoint: ValidatedProductionCheckpoint,
) -> tuple[int, int]:
    manifest_record = _read_json(Path(checkpoint.manifest_path))
    receipts = manifest_record["joint_candidate_bundle"]["receipts"]
    try:
        policy_step = receipts["policy"]["optimizer"]["stats"]["optimizer_step_after"]
        harness_step = receipts["harness"]["optimizer_step_after"]
    except (KeyError, TypeError) as exc:
        raise ProductionCheckpointError(
            "exact recovery component optimizer references are missing"
        ) from exc
    _require(
        type(policy_step) is int
        and policy_step >= 0
        and type(harness_step) is int
        and harness_step >= 0,
        "exact recovery component optimizer steps are invalid",
    )
    return policy_step, harness_step


def _execute_live_continuation(
    restored: RestoredProductionCheckpoint,
    *,
    actor: object,
    admission_record: Mapping[str, object],
    device: str | object,
    rank: int,
    saved_policy_step: int,
    saved_harness_step: int,
    run_policy_optimizer_step: Callable[[object], object],
    run_harness_optimizer_step: Callable[[object, object], object],
) -> dict[str, object]:
    checkpoint = restored.audit
    _require(_real_areal_actor(actor), "live recovery requires real FSDPPPOActor")
    _require_real_harness_optimizer(restored.harness_policy, restored.harness_optimizer)
    _require(
        checkpoint.rollout_cursor_state.live_exact_eligible
        and checkpoint.dataloader_cursor_state.live_exact_eligible,
        "legacy integer cursors cannot prove live exact recovery",
    )
    admission_sha256 = admission_record.get("record_sha256")
    _require(
        _is_sha256(admission_sha256)
        and checkpoint.rollout_cursor_state.pending_item_sha256
        == checkpoint.source_joint_credit_sha256
        and checkpoint.dataloader_cursor_state.pending_item_sha256 == admission_sha256,
        "checkpoint pending cursors do not bind the live W continuation inputs",
    )
    _assert_no_secrets(admission_record, "admission_record")
    _canonical_json(admission_record)

    actor_version = actor.get_version()
    _require(
        actor_version == checkpoint.actor_public_version,
        "restored actor public version differs from checkpoint",
    )
    policy_step_before = _policy_optimizer_step(actor)
    harness_step_before = _harness_optimizer_step(
        restored.harness_policy, restored.harness_optimizer
    )
    _require(
        policy_step_before == saved_policy_step
        and harness_step_before == saved_harness_step,
        "restored optimizer steps differ from the V component receipts",
    )
    policy_probe_before = _policy_probe_sha256(
        actor,
        admission_record,
        active_joint_version=checkpoint.parent_joint_version,
        device=device,
    )
    harness_digest_before = restored.harness_policy.parameter_digest
    scheduler_before = _scheduler_state(actor)

    policy_result = run_policy_optimizer_step(actor)
    _require(
        policy_result is None,
        "Policy continuation callback must not self-report recovery evidence",
    )
    _require(
        _scheduler_state(actor) == scheduler_before,
        "Policy callback must leave scheduler advancement to the W framework",
    )
    policy_step_after = _policy_optimizer_step(actor)
    _require(
        policy_step_after == policy_step_before + 1,
        "live Policy optimizer did not advance exactly once",
    )
    _require(
        actor.get_version() == actor_version,
        "Policy optimizer changed the public inference version before activation",
    )
    step_scheduler = getattr(actor, "step_lr_scheduler", None)
    _require(callable(step_scheduler), "AReaL lr scheduler step is unavailable")
    step_scheduler()
    scheduler_after = _scheduler_state(actor)
    _require(
        scheduler_after["scheduler_class"] == scheduler_before["scheduler_class"]
        and scheduler_after["state_sha256"] != scheduler_before["state_sha256"],
        "framework-owned AReaL scheduler step did not advance state",
    )
    policy_probe_after = _policy_probe_sha256(
        actor,
        admission_record,
        active_joint_version=checkpoint.parent_joint_version,
        device=device,
    )
    _require(
        policy_probe_after != policy_probe_before,
        "live Policy optimizer did not change the framework-owned probe",
    )

    harness_result = run_harness_optimizer_step(
        restored.harness_policy, restored.harness_optimizer
    )
    _require(
        harness_result is None,
        "Harness continuation callback must not self-report recovery evidence",
    )
    _require(
        restored.harness_policy.update_step == harness_step_before,
        "Harness callback must leave version advancement to the W framework",
    )
    harness_optimizer_step = _optimizer_step_from_state(
        restored.harness_optimizer.state, "Harness Adam optimizer"
    )
    _require(
        harness_optimizer_step == harness_step_before + 1,
        "live Harness Adam optimizer did not advance exactly once",
    )
    restored.harness_policy.update_step = harness_optimizer_step
    harness_digest_after = restored.harness_policy.parameter_digest
    _require(
        harness_digest_after != harness_digest_before,
        "live Harness optimizer did not change the parameter digest",
    )
    harness_optimizer_state_sha256 = _sha256(
        _to_json_state(restored.harness_optimizer.state_dict())
    )
    runtime_rng = capture_rank_runtime_state(
        rank=rank,
        local_rank=rank,
        device=checkpoint.topology.rank_to_device[rank],
        harness_policy=restored.harness_policy,
    )
    return {
        "policy_state_sha256": policy_probe_after,
        "harness_state_sha256": harness_digest_after,
        "harness_optimizer_state_sha256": harness_optimizer_state_sha256,
        "harness_sample_count": restored.harness_policy.sample_count,
        "harness_training": restored.harness_policy.training,
        "policy_optimizer_step": policy_step_after,
        "harness_optimizer_step": harness_optimizer_step,
        "lr_scheduler_state_sha256": scheduler_after["state_sha256"],
        "runtime_rng_state_sha256": _sha256(runtime_rng.to_record()),
        "actor_public_version": actor.get_version(),
        "rollout_cursor": _cursor_advance_record(checkpoint.rollout_cursor_state),
        "dataloader_cursor": _cursor_advance_record(checkpoint.dataloader_cursor_state),
    }


@dataclass(frozen=True, init=False)
class LiveExactJointRecovery:
    """Unforgeable-by-serialization handle for one live W recovery run."""

    _record: dict[str, object] = dataclass_field(repr=False)
    _checkpoint: ValidatedProductionCheckpoint = dataclass_field(repr=False)
    _final_restored: RestoredProductionCheckpoint = dataclass_field(repr=False)
    _token: object = dataclass_field(repr=False)

    @classmethod
    def _create(
        cls,
        *,
        record: Mapping[str, object],
        checkpoint: ValidatedProductionCheckpoint,
        final_restored: RestoredProductionCheckpoint,
        token: object,
    ) -> LiveExactJointRecovery:
        _require(token is _LIVE_EXACT_RECOVERY_TOKEN, "live recovery token is invalid")
        instance = object.__new__(cls)
        object.__setattr__(instance, "_record", deepcopy(dict(record)))
        object.__setattr__(instance, "_checkpoint", checkpoint)
        object.__setattr__(instance, "_final_restored", final_restored)
        object.__setattr__(instance, "_token", token)
        return instance

    @property
    def record(self) -> dict[str, object]:
        return deepcopy(self._record)

    @property
    def record_sha256(self) -> str:
        return str(self._record["record_sha256"])

    @property
    def checkpoint_manifest_sha256(self) -> str:
        return self._checkpoint.record_sha256

    @property
    def candidate_joint_version(self) -> JointVersion:
        return self._checkpoint.candidate_joint_version

    @property
    def macro_step_id(self) -> str:
        return self._checkpoint.macro_step_id

    @property
    def restored_harness_policy(self) -> object:
        return self._final_restored.harness_policy

    @property
    def restored_harness_optimizer(self) -> object:
        return self._final_restored.harness_optimizer

    @property
    def exact_joint_recovery(self) -> bool:
        return self._token is _LIVE_EXACT_RECOVERY_TOKEN


def verify_exact_joint_recovery(
    manifest_path: str | Path,
    *,
    actor: object,
    current_topology: RuntimeTopology,
    rank: int,
    admission_record: Mapping[str, object],
    device: str | object,
    run_policy_optimizer_step: Callable[[object], object],
    run_harness_optimizer_step: Callable[[object, object], object],
) -> LiveExactJointRecovery:
    """Execute and measure both branches; callbacks cannot supply evidence."""

    _require(
        _real_areal_actor(actor),
        "exact recovery requires real AReaL FSDPPPOActor",
    )
    first = restore_production_joint_checkpoint(
        manifest_path,
        actor=actor,
        current_topology=current_topology,
        rank=rank,
        restore_rng=True,
        require_real_components=True,
    )
    _require(
        first.audit.topology.world_size == 1 and rank == 0,
        "multi-rank exact recovery requires aggregated per-rank continuation evidence",
    )
    saved_policy_step, saved_harness_step = _saved_optimizer_steps(first.audit)
    uninterrupted = _execute_live_continuation(
        first,
        actor=actor,
        admission_record=admission_record,
        device=device,
        rank=rank,
        saved_policy_step=saved_policy_step,
        saved_harness_step=saved_harness_step,
        run_policy_optimizer_step=run_policy_optimizer_step,
        run_harness_optimizer_step=run_harness_optimizer_step,
    )
    second = restore_production_joint_checkpoint(
        manifest_path,
        actor=actor,
        current_topology=current_topology,
        rank=rank,
        restore_rng=True,
        require_real_components=True,
    )
    _require(
        second.audit.record_sha256 == first.audit.record_sha256,
        "checkpoint changed between exact recovery branches",
    )
    recovered = _execute_live_continuation(
        second,
        actor=actor,
        admission_record=admission_record,
        device=device,
        rank=rank,
        saved_policy_step=saved_policy_step,
        saved_harness_step=saved_harness_step,
        run_policy_optimizer_step=run_policy_optimizer_step,
        run_harness_optimizer_step=run_harness_optimizer_step,
    )
    _require(
        _canonical_json(recovered) == _canonical_json(uninterrupted),
        "recovered next step differs from uninterrupted live execution",
    )
    final_restored = restore_production_joint_checkpoint(
        manifest_path,
        actor=actor,
        current_topology=current_topology,
        rank=rank,
        restore_rng=True,
        require_real_components=True,
    )
    _require(
        final_restored.audit.record_sha256 == first.audit.record_sha256
        and _policy_optimizer_step(actor) == saved_policy_step
        and _harness_optimizer_step(
            final_restored.harness_policy, final_restored.harness_optimizer
        )
        == saved_harness_step
        and actor.get_version() == first.audit.actor_public_version,
        "final candidate restore after exact recovery differs from V",
    )
    record: dict[str, object] = {
        "schema_version": RECOVERY_EVIDENCE_SCHEMA_VERSION,
        "checkpoint_manifest_sha256": first.audit.record_sha256,
        "actor_class": f"{type(actor).__module__}.{type(actor).__qualname__}",
        "harness_policy_class": (
            f"{type(second.harness_policy).__module__}."
            f"{type(second.harness_policy).__qualname__}"
        ),
        "harness_optimizer_class": (
            f"{type(second.harness_optimizer).__module__}."
            f"{type(second.harness_optimizer).__qualname__}"
        ),
        "uninterrupted_next_step": uninterrupted,
        "recovered_next_step": recovered,
        "evidence_scope": {
            "framework_owned_state_measurement": True,
            "real_areal_fsdp_actor_loaded": True,
            "real_harness_checkpoint_loaded": True,
            "scheduler_rng_cursor_restored": True,
            "continuous_next_step_verified": True,
            "persisted_record_regrants_live_exact": False,
            "exact_joint_recovery": True,
        },
    }
    record["record_sha256"] = _record_sha256(record)
    validate_exact_joint_recovery_evidence(
        record,
        manifest=manifest_path,
        current_topology=current_topology,
    )
    return LiveExactJointRecovery._create(
        record=record,
        checkpoint=final_restored.audit,
        final_restored=final_restored,
        token=_LIVE_EXACT_RECOVERY_TOKEN,
    )


def validate_exact_joint_recovery_evidence(
    record: Mapping[str, object],
    *,
    manifest: str | Path,
    current_topology: RuntimeTopology | None = None,
) -> dict[str, object]:
    """Validate persisted integrity without recreating live exact authority."""

    _require(
        set(record)
        == {
            "schema_version",
            "checkpoint_manifest_sha256",
            "actor_class",
            "harness_policy_class",
            "harness_optimizer_class",
            "uninterrupted_next_step",
            "recovered_next_step",
            "evidence_scope",
            "record_sha256",
        },
        "exact recovery evidence field set differs",
    )
    _require(
        record.get("schema_version") == RECOVERY_EVIDENCE_SCHEMA_VERSION,
        "exact recovery evidence schema differs",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "exact recovery evidence hash mismatch",
    )
    _assert_no_secrets(record)
    checkpoint = validate_production_joint_checkpoint(
        manifest,
        current_topology=current_topology,
        require_component_files=True,
    )
    _require(
        checkpoint.real_areal_checkpoint and checkpoint.real_harness_checkpoint,
        "exact recovery evidence requires real AReaL and Harness checkpoints",
    )
    _require(
        checkpoint.topology.world_size == 1,
        "multi-rank exact recovery requires aggregated per-rank continuation evidence",
    )
    _require(
        checkpoint.rollout_cursor_state.live_exact_eligible
        and checkpoint.dataloader_cursor_state.live_exact_eligible,
        "legacy integer cursors cannot support persisted exact recovery evidence",
    )
    _require(
        record.get("checkpoint_manifest_sha256") == checkpoint.record_sha256,
        "exact recovery evidence refers to another checkpoint",
    )
    _require(
        record.get("actor_class") == "areal.engine.fsdp_engine.FSDPPPOActor"
        and record.get("harness_policy_class")
        == "jphrl.harness.torch_learning.TorchHarnessPolicy"
        and record.get("harness_optimizer_class") == "torch.optim.adam.Adam",
        "exact recovery evidence component classes are not the real production types",
    )
    uninterrupted = _validate_next_step(
        record.get("uninterrupted_next_step"),
        "uninterrupted next step",
        checkpoint=checkpoint,
    )
    recovered = _validate_next_step(
        record.get("recovered_next_step"),
        "recovered next step",
        checkpoint=checkpoint,
    )
    _require(
        _canonical_json(uninterrupted) == _canonical_json(recovered),
        "exact recovery next-step records differ",
    )
    saved_policy_step, saved_harness_step = _saved_optimizer_steps(checkpoint)
    _require(
        recovered["policy_optimizer_step"] == saved_policy_step + 1
        and recovered["harness_optimizer_step"] == saved_harness_step + 1
        and recovered["actor_public_version"] == checkpoint.actor_public_version,
        "exact recovery did not advance both optimizer steps exactly once",
    )
    expected_scope = {
        "framework_owned_state_measurement": True,
        "real_areal_fsdp_actor_loaded": True,
        "real_harness_checkpoint_loaded": True,
        "scheduler_rng_cursor_restored": True,
        "continuous_next_step_verified": True,
        "persisted_record_regrants_live_exact": False,
        "exact_joint_recovery": True,
    }
    _require(
        record.get("evidence_scope") == expected_scope,
        "exact recovery evidence scope differs",
    )
    return {
        "ok": True,
        "integrity_valid": True,
        "checkpoint_manifest_sha256": checkpoint.record_sha256,
        "parent_joint_version_id": checkpoint.parent_joint_version.version_id,
        "candidate_joint_version_id": checkpoint.candidate_joint_version.version_id,
        "macro_step_id": checkpoint.macro_step_id,
        "macro_step": checkpoint.macro_step,
        "persisted_exact_claim": True,
        "live_exact_joint_recovery": False,
        "exact_joint_recovery": False,
        "record_sha256": record["record_sha256"],
    }


def require_live_exact_joint_recovery(
    value: object,
    *,
    manifest: str | Path | None = None,
    current_topology: RuntimeTopology | None = None,
) -> LiveExactJointRecovery:
    """Require the in-memory capability emitted by the live two-branch run."""

    _require(
        type(value) is LiveExactJointRecovery
        and getattr(value, "_token", None) is _LIVE_EXACT_RECOVERY_TOKEN,
        "live exact joint recovery capability is required",
    )
    manifest_path = manifest or value._checkpoint.manifest_path
    audit = validate_exact_joint_recovery_evidence(
        value._record,
        manifest=manifest_path,
        current_topology=current_topology,
    )
    _require(
        audit["integrity_valid"] is True
        and audit["exact_joint_recovery"] is False
        and value._checkpoint.record_sha256 == audit["checkpoint_manifest_sha256"],
        "live exact recovery capability differs from persisted integrity",
    )
    return value


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CURSOR_SCHEMA_VERSION",
    "LiveExactJointRecovery",
    "ProductionCheckpointError",
    "RankRuntimeState",
    "RestoredProductionCheckpoint",
    "RuntimeCursorState",
    "RuntimeTopology",
    "ValidatedProductionCheckpoint",
    "capture_rank_runtime_state",
    "dry_load_production_joint_checkpoint",
    "require_live_exact_joint_recovery",
    "restore_production_joint_checkpoint",
    "save_production_joint_checkpoint",
    "validate_exact_joint_recovery_evidence",
    "validate_production_joint_checkpoint",
    "verify_exact_joint_recovery",
]
