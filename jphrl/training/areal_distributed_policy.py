from __future__ import annotations

"""Worker-authenticated AReaL ``fsdp:d4`` Policy candidate updates.

The controller in this module owns orchestration only.  Optimizer, scheduler,
device, rank, and DCP observations are made inside the four remote engine
processes.  A scalar AReaL controller call normally keeps only the first
data-parallel result, so this controller deliberately retains all four raw RPC
results and validates them as one receipt.
"""

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import uuid4

from jphrl.experiments.m0_eight_gpu_topology import (
    M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA,
    M0_REMOTE_OPTIMIZER_RECEIPT_SCHEMA,
    M0EightGPUTopology,
    assert_controller_has_no_local_optimizer,
    validate_remote_optimizer_receipt,
)
from jphrl.paths import require_outside_repository, require_within_configured_root
from jphrl.trajectory.schema import JointVersion
from jphrl.trajectory.multi_s_frozen_training_batch import (
    ValidatedMultiSFrozenTrainingBatch,
    multi_s_source_binding,
    validate_multi_s_source_binding,
)

from .areal_policy_candidate import checkpoint_manifest
from .areal_policy_optimizer import (
    materialize_areal_ppo_update_tensors,
    validate_areal_policy_optimizer_source,
    validate_m0_areal_actor_config,
)

try:  # Imported lazily enough for CPU-only contract tests to load this module.
    from areal.api import SaveLoadMeta
    from areal.engine.fsdp_engine import FSDPPPOActor as _ArealFSDPPPOActor
    from areal.infra.utils.concurrent import run_async_task as _run_async_task
    from areal.trainer.ppo.actor import PPOActorController as _ArealPPOActorController

    _AREAL_IMPORT_ERROR: BaseException | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - local gate
    SaveLoadMeta = None  # type: ignore[assignment,misc]
    _ArealFSDPPPOActor = object  # type: ignore[assignment,misc]
    _ArealPPOActorController = object  # type: ignore[assignment,misc]
    _run_async_task = None
    _AREAL_IMPORT_ERROR = exc


JPH_AREAL_DISTRIBUTED_ACTOR_CLASS = (
    "jphrl.training.areal_distributed_policy.JPHFSDPPPOActor"
)
JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS = (
    "jphrl.training.areal_distributed_policy.JPHPPOActorController"
)
AREAL_DISTRIBUTED_POLICY_CANDIDATE_SCHEMA = (
    "jph.areal-distributed-policy-candidate.v1"
)
M0_DISTRIBUTED_POLICY_RESTORE_RANK_SCHEMA = (
    "jph.m0-distributed-policy-restore-rank.v1"
)
M0_DISTRIBUTED_POLICY_RESTORE_SCHEMA = "jph.m0-distributed-policy-restore.v1"
M0_DISTRIBUTED_POLICY_CONTINUATION_RANK_SCHEMA = (
    "jph.m0-distributed-policy-continuation-rank.v1"
)
M0_DISTRIBUTED_POLICY_CONTINUATION_SCHEMA = (
    "jph.m0-distributed-policy-continuation.v1"
)
M0_DISTRIBUTED_SERVING_EXPORT_RANK_SCHEMA = (
    "jph.m0-distributed-serving-export-rank.v1"
)
M0_DISTRIBUTED_SERVING_EXPORT_SCHEMA = "jph.m0-distributed-serving-export.v1"
M0_DISTRIBUTED_POLICY_CURRENT_STATE_RANK_SCHEMA = (
    "jph.m0-distributed-policy-current-state-rank.v1"
)
M0_DISTRIBUTED_POLICY_CURRENT_STATE_SCHEMA = (
    "jph.m0-distributed-policy-current-state.v1"
)
PINNED_AREAL_COMMIT = "fee938eada49208a5aabdbc1095730a13076a349"
PINNED_OPTIMIZER_CLASS = "torch.optim.adamw.AdamW"
_RANK_EVIDENCE_SCOPE = {
    "remote_worker_optimizer_observed": True,
    "controller_read_worker_optimizer": False,
    "policy_optimizer_update": True,
    "harness_optimizer_update": False,
}
_AGGREGATE_EVIDENCE_SCOPE = {
    "all_actor_ranks_attested": True,
    "remote_worker_optimizer_observed": True,
    "controller_read_worker_optimizer": False,
    "policy_optimizer_update": True,
    "harness_optimizer_update": False,
}
_T = TypeVar("_T")
_LIVE_CANDIDATE_TOKEN = object()
_W_BRANCH_IDS = {"uninterrupted", "recovered", "final-restore"}


class ArealDistributedPolicyError(RuntimeError):
    """Raised when a four-rank Policy transition cannot be proven atomically."""


@dataclass(frozen=True, init=False)
class LiveArealDistributedPolicyCandidate:
    """In-process capability carrying worker-exported inputs required by W."""

    receipt: Mapping[str, object]
    worker_states: tuple[Mapping[str, object], ...]
    _token: object

    @classmethod
    def _create(
        cls,
        *,
        receipt: Mapping[str, object],
        worker_states: Sequence[Mapping[str, object]],
        token: object,
    ) -> LiveArealDistributedPolicyCandidate:
        _require(token is _LIVE_CANDIDATE_TOKEN, "live Policy candidate token is invalid")
        instance = object.__new__(cls)
        object.__setattr__(instance, "receipt", deepcopy(dict(receipt)))
        object.__setattr__(
            instance,
            "worker_states",
            tuple(deepcopy(dict(item)) for item in worker_states),
        )
        object.__setattr__(instance, "_token", token)
        return instance


@dataclass(frozen=True)
class ValidatedDistributedPolicyCandidate:
    transaction_id: str
    episode_id: str
    parent_joint_version: JointVersion
    source_admission_sha256: str
    source_joint_credit_sha256: str
    source_binding: Mapping[str, object]
    source_batch_identity_sha256: str
    trainable_token_count: int
    parent_engine_version: int
    reserved_candidate_engine_version: int
    candidate_policy_version: str
    remote_optimizer_receipt_sha256: str
    record_sha256: str

    @property
    def digest(self) -> str:
        return self.record_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealDistributedPolicyError(message)


def _validated_pending_w_candidate_path(
    *,
    pending: Mapping[str, object],
    candidate_path: str,
    candidate_dcp_manifest_sha256: str,
) -> Path:
    """Bind W to the exact candidate path and bytes persisted by T."""

    path = require_within_configured_root(candidate_path)
    _require(
        path.is_dir() and not path.is_symlink(),
        "distributed W candidate DCP is missing or unsafe",
    )
    _require(
        isinstance(pending.get("candidate_path"), str)
        and str(path) == pending["candidate_path"],
        "distributed W candidate DCP path differs from T",
    )
    observed_manifest_sha256 = checkpoint_manifest(path)["manifest_sha256"]
    _require(
        observed_manifest_sha256 == candidate_dcp_manifest_sha256,
        "distributed W candidate DCP contents differ from T",
    )
    _require(
        candidate_dcp_manifest_sha256
        == pending.get("candidate_manifest_sha256"),
        "distributed W candidate DCP manifest binding differs from T",
    )
    return path


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
        raise ArealDistributedPolicyError(
            "distributed Policy evidence is not finite canonical JSON"
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


def _validated_pending_m0_candidate_state(
    pending: Mapping[str, object],
) -> Mapping[str, object]:
    """Require the complete worker-private T state before later mutations."""

    rank_receipt = pending.get("rank_receipt")
    scheduler_state_after = pending.get("scheduler_state_after")
    rank_runtime_state = pending.get("rank_runtime_state")
    optimizer_step_after = pending.get("optimizer_step_after")
    _require(
        isinstance(pending.get("parent_path"), str)
        and bool(pending["parent_path"])
        and isinstance(pending.get("candidate_path"), str)
        and bool(pending["candidate_path"])
        and _is_sha256(pending.get("candidate_manifest_sha256"))
        and isinstance(optimizer_step_after, int)
        and not isinstance(optimizer_step_after, bool)
        and optimizer_step_after >= 0
        and isinstance(scheduler_state_after, Mapping)
        and isinstance(rank_runtime_state, Mapping)
        and isinstance(rank_receipt, Mapping),
        "pending M0 candidate state is incomplete",
    )
    _require(
        rank_receipt.get("candidate_dcp_manifest_sha256")
        == pending["candidate_manifest_sha256"]
        and rank_receipt.get("optimizer_step_after") == optimizer_step_after
        and rank_receipt.get("lr_scheduler_state_after_sha256")
        == _sha256(scheduler_state_after),
        "pending M0 candidate state differs from its rank receipt",
    )
    return rank_receipt


def _checkpoint_dcp_payload_sha256(manifest: Mapping[str, object]) -> str:
    """Digest DCP tensor/optimizer shards without save-instance metadata."""

    files = manifest.get("files")
    _require(
        isinstance(files, list)
        and len(files) >= 2
        and sum(
            isinstance(item, Mapping) and item.get("path") == ".metadata"
            for item in files
        )
        == 1,
        "distributed DCP manifest has no unique metadata boundary",
    )
    payload_files = []
    for item in files:
        _require(
            isinstance(item, Mapping)
            and set(item) == {"path", "size_bytes", "sha256"}
            and isinstance(item.get("path"), str)
            and isinstance(item.get("size_bytes"), int)
            and not isinstance(item.get("size_bytes"), bool)
            and item["size_bytes"] >= 0
            and _is_sha256(item.get("sha256")),
            "distributed DCP manifest file entry is invalid",
        )
        if item["path"] != ".metadata":
            _require(
                str(item["path"]).endswith(".distcp"),
                "distributed DCP payload contains an unexpected file",
            )
            payload_files.append(deepcopy(dict(item)))
    _require(payload_files, "distributed DCP payload is empty")
    return _sha256({"files": payload_files})


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write_json(path: Path, record: Mapping[str, object]) -> None:
    destination = require_within_configured_root(path)
    _require(destination.parent.is_dir(), "receipt parent directory is missing")
    _require(not destination.exists(), "receipt path already exists")
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _joint_version(value: object) -> JointVersion:
    fields = set(JointVersion.__dataclass_fields__)
    _require(
        isinstance(value, Mapping) and set(value) == fields,
        "distributed Policy JointVersion fields differ",
    )
    try:
        version = JointVersion(**dict(value))
    except TypeError as exc:
        raise ArealDistributedPolicyError(
            "distributed Policy JointVersion is invalid"
        ) from exc
    _require(
        all(
            isinstance(getattr(version, field), str) and getattr(version, field)
            for field in fields
        ),
        "distributed Policy JointVersion contains an empty identity",
    )
    return version


def _partition_indices(sample_count: int) -> tuple[tuple[int, ...], ...]:
    _require(
        type(sample_count) is int and sample_count == 4,
        "first fsdp:d4 M0 transaction requires exactly four Policy samples",
    )
    partitions = tuple(
        tuple(range(rank, sample_count, 4)) for rank in range(4)
    )
    _require(
        all(partitions)
        and sorted(index for part in partitions for index in part)
        == list(range(sample_count)),
        "fsdp:d4 sample partition is not disjoint and exhaustive",
    )
    return partitions


def _validate_ordered_sources(
    admission_records: Sequence[Mapping[str, object]],
    source_joint_credit_records: Sequence[Mapping[str, object]],
    *,
    active_joint_version: JointVersion,
    source_binding: Mapping[str, object],
) -> tuple[tuple[object, ...], tuple[tuple[int, int, Mapping[str, object]], ...]]:
    binding = validate_multi_s_source_binding(source_binding)
    _require(
        isinstance(admission_records, Sequence)
        and not isinstance(admission_records, (str, bytes, bytearray))
        and isinstance(source_joint_credit_records, Sequence)
        and not isinstance(source_joint_credit_records, (str, bytes, bytearray))
        and len(admission_records) == len(source_joint_credit_records)
        and bool(admission_records),
        "ordered Policy admissions and S records must be non-empty and one-to-one",
    )
    _require(
        binding["joint_version_id"] == active_joint_version.version_id
        and len(admission_records) == len(binding["s_record_sha256s"])
        and [record.get("record_sha256") for record in source_joint_credit_records]
        == binding["s_record_sha256s"],
        "ordered Policy records differ from the canonical multi-S source binding",
    )
    admissions: list[object] = []
    flattened: list[tuple[int, int, Mapping[str, object]]] = []
    for member_index, (admission_record, source_record) in enumerate(
        zip(admission_records, source_joint_credit_records)
    ):
        _require(
            isinstance(admission_record, Mapping)
            and isinstance(source_record, Mapping),
            f"ordered Policy member {member_index} is not a record pair",
        )
        admission = validate_areal_policy_optimizer_source(
            admission_record,
            source_joint_credit_record=source_record,
            active_joint_version=active_joint_version,
        )
        _require(
            admission.export_style == "individual" and bool(admission.samples),
            f"ordered Policy member {member_index} has no individual samples",
        )
        samples = tuple(admission.samples)
        admissions.append(admission)
        flattened.extend(
            (member_index, sample_index, sample)
            for sample_index, sample in enumerate(samples)
        )
    _require(
        len(flattened) == binding["policy_sample_count"],
        "ordered Policy sample count differs from the multi-S source binding",
    )
    inference_versions = {
        admission.inference_engine_version for admission in admissions
    }
    _require(
        len(inference_versions) == 1,
        "ordered Policy admissions disagree on inference engine version",
    )
    return tuple(admissions), tuple(flattened)


def _optimizer_step(actor: object) -> int:
    optimizer = getattr(actor, "optimizer", None)
    state = getattr(optimizer, "state", None)
    _require(isinstance(state, Mapping), "rank-local optimizer state is unavailable")
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
            "rank-local optimizer step is invalid",
        )
        steps.append(int(raw_step))
    if not steps:
        return 0
    _require(
        len(set(steps)) == 1,
        "rank-local optimizer parameter steps disagree",
    )
    return steps[0]


def _require_post_parent_optimizer_baseline(
    *,
    pre_parent_dcp_step: int,
    post_parent_dcp_step: int,
) -> None:
    """Accept only an unchanged or one-step DCP lazy-state initialization."""

    _require(
        post_parent_dcp_step
        in {pre_parent_dcp_step, pre_parent_dcp_step + 1},
        "parent DCP changed the rank-local AdamW parameter-state step by "
        "more than its optional one-step lazy initialization",
    )


def _lr_scheduler_state(actor: object) -> dict[str, object]:
    scheduler = getattr(actor, "lr_scheduler", None)
    state_dict = getattr(scheduler, "state_dict", None)
    _require(callable(state_dict), "rank-local lr scheduler is unavailable")
    value = state_dict()
    _require(isinstance(value, Mapping), "rank-local lr scheduler state is invalid")
    state = deepcopy(dict(value))
    _canonical_json(state)
    return state


def _restore_lr_scheduler_state(
    actor: object,
    state: Mapping[str, object],
) -> None:
    scheduler = getattr(actor, "lr_scheduler", None)
    load_state_dict = getattr(scheduler, "load_state_dict", None)
    _require(callable(load_state_dict), "rank-local scheduler rollback is unavailable")
    load_state_dict(deepcopy(dict(state)))
    _require(
        _lr_scheduler_state(actor) == state,
        "rank-local scheduler rollback is incomplete",
    )


def _require_one_scheduler_step(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> None:
    _require(
        set(before) == set(after)
        and type(before.get("last_epoch")) is int
        and after.get("last_epoch") == before["last_epoch"] + 1
        and type(before.get("_step_count")) is int
        and after.get("_step_count") == before["_step_count"] + 1,
        "rank-local lr scheduler did not advance exactly once",
    )
    before_rest = dict(before)
    after_rest = dict(after)
    before_rest.pop("last_epoch")
    before_rest.pop("_step_count")
    after_rest.pop("last_epoch")
    after_rest.pop("_step_count")
    _require(
        before_rest == after_rest,
        "constant lr scheduler changed outside its step counters",
    )


def _normalized_update_stats(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "rank-local optimizer result is missing")
    successful = value.get("update_successful")
    grad_norm = value.get("grad_norm")
    learning_rate = value.get("lr")
    _require(
        isinstance(successful, (int, float))
        and not isinstance(successful, bool)
        and float(successful) == 1.0
        and isinstance(grad_norm, (int, float))
        and not isinstance(grad_norm, bool)
        and math.isfinite(float(grad_norm))
        and float(grad_norm) > 0.0
        and isinstance(learning_rate, (int, float))
        and not isinstance(learning_rate, bool)
        and math.isfinite(float(learning_rate))
        and float(learning_rate) > 0.0,
        "rank-local optimizer result does not prove one successful update",
    )
    return {
        "update_successful": 1.0,
        "update_successful_count": 1,
        "grad_norm": float(grad_norm),
        "grad_norm_count": 1,
        "learning_rate": float(learning_rate),
        "learning_rate_count": 1,
    }


def _physical_gpu_id() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",") if visible else []
    _require(
        len(devices) == 1 and devices[0].strip().isdigit(),
        "worker must observe exactly one numeric CUDA_VISIBLE_DEVICES entry",
    )
    return int(devices[0].strip())


def _require_new_transaction(pending: object) -> None:
    _require(
        pending is None,
        "actor still retains an earlier M0 candidate transaction",
    )


def _exception_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:1000]}"


def build_remote_optimizer_aggregate(
    rank_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build and validate one aggregate without reading controller state."""

    _require(
        len(rank_receipts) == 4,
        "remote Policy aggregate requires exactly four rank receipts",
    )
    ranks = [dict(item) for item in rank_receipts]
    first = ranks[0]
    record: dict[str, object] = {
        "schema_version": M0_REMOTE_OPTIMIZER_RECEIPT_SCHEMA,
        "transaction_id": first.get("transaction_id"),
        "joint_version_id": first.get("joint_version_id"),
        "source_admission_sha256": first.get("source_admission_sha256"),
        "actor_backend": "fsdp:d4",
        "controller_class": JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS,
        "global_sample_count": first.get("global_sample_count"),
        "global_sample_sha256": first.get("global_sample_sha256"),
        "rank0_lr_scheduler_state_after": first.get(
            "lr_scheduler_state_after"
        ),
        "rank_receipts": ranks,
        "evidence_scope": dict(_AGGREGATE_EVIDENCE_SCOPE),
    }
    record["record_sha256"] = _record_sha256(record)
    validate_remote_optimizer_receipt(record)
    return record


def _validate_manifest(value: object, label: str) -> Mapping[str, object]:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"files", "manifest_sha256"}
        and isinstance(value["files"], list)
        and bool(value["files"])
        and value["manifest_sha256"] == _sha256({"files": value["files"]}),
        f"distributed Policy {label} manifest is invalid",
    )
    for item in value["files"]:
        _require(
            isinstance(item, Mapping)
            and set(item) == {"path", "size_bytes", "sha256"}
            and isinstance(item["path"], str)
            and bool(item["path"])
            and not Path(item["path"]).is_absolute()
            and ".." not in Path(item["path"]).parts
            and type(item["size_bytes"]) is int
            and item["size_bytes"] >= 0
            and _is_sha256(item["sha256"]),
            f"distributed Policy {label} manifest file is invalid",
        )
    return value


def build_distributed_policy_candidate(
    *,
    transaction_id: str,
    admissions: Sequence[object],
    source_joint_credit_records: Sequence[Mapping[str, object]],
    source_binding: Mapping[str, object],
    flattened_samples: Sequence[tuple[int, int, Mapping[str, object]]],
    active_joint_version: JointVersion,
    remote_optimizer_receipt: Mapping[str, object],
    candidate_root: str | Path,
    project_commit: str,
    areal_commit: str,
) -> dict[str, object]:
    """Seal the distributed T result without masquerading as the d1 schema."""

    audit = validate_remote_optimizer_receipt(remote_optimizer_receipt)
    binding = validate_multi_s_source_binding(source_binding)
    _require(
        isinstance(transaction_id, str)
        and transaction_id
        and transaction_id == audit.transaction_id,
        "distributed Policy candidate transaction differs from worker receipts",
    )
    _require(
        len(admissions) == len(source_joint_credit_records) and bool(admissions),
        "distributed Policy candidate source members are incomplete",
    )
    _require(
        areal_commit == PINNED_AREAL_COMMIT and _is_git_commit(project_commit),
        "distributed Policy candidate provenance is invalid",
    )
    members: list[dict[str, object]] = []
    trainable_token_count = 0
    for member_index, (admission, source_record) in enumerate(
        zip(admissions, source_joint_credit_records)
    ):
        source_sha256 = source_record.get("record_sha256")
        _require(
            _is_sha256(source_sha256)
            and admission.source_joint_credit_sha256 == source_sha256,
            f"distributed Policy source member {member_index} differs from S",
        )
        sample_ids = [sample["sample_id"] for sample in admission.samples]
        member = {
            "member_index": member_index,
            "member_claim_sha256": binding["member_claim_sha256s"][member_index],
            "episode_id": admission.episode_id,
            "admission_record_sha256": admission.record_sha256,
            "source_joint_credit_sha256": source_sha256,
            "sample_ids": sample_ids,
            "trainable_token_count": admission.trainable_token_count,
        }
        members.append(member)
        trainable_token_count += admission.trainable_token_count
    source_batch_identity_sha256 = _sha256(
        {
            "schema_version": "jph.m0-ordered-policy-source-batch.v1",
            "members": members,
        }
    )
    global_samples = [
        {
            "member_index": member_index,
            "sample_index": sample_index,
            "sample_id": sample["sample_id"],
            "tensor_dict": sample["tensor_dict"],
        }
        for member_index, sample_index, sample in flattened_samples
    ]
    _require(
        len(global_samples) == audit.global_sample_count
        and _sha256(global_samples) == audit.global_sample_sha256,
        "distributed Policy candidate source batch differs from rank receipts",
    )
    _require(
        binding["joint_version_id"] == active_joint_version.version_id
        and binding["record_sha256"] == audit.source_admission_sha256
        and binding["policy_sample_count"] == len(global_samples)
        and binding["s_record_sha256s"]
        == [member["source_joint_credit_sha256"] for member in members],
        "distributed Policy candidate differs from canonical multi-S binding",
    )
    root = require_outside_repository(candidate_root)
    parent_path = root / "policy-parent.dcp"
    candidate_path = root / "policy-candidate.dcp"
    parent_manifest = checkpoint_manifest(parent_path)
    candidate_manifest = checkpoint_manifest(candidate_path)
    rank_zero = remote_optimizer_receipt["rank_receipts"][0]
    _require(
        parent_manifest["manifest_sha256"]
        == rank_zero["parent_dcp_manifest_sha256"]
        and candidate_manifest["manifest_sha256"]
        == rank_zero["candidate_dcp_manifest_sha256"],
        "distributed Policy candidate DCPs differ from worker receipts",
    )
    parent_engine_version = rank_zero["inference_engine_version"]
    candidate_policy_version = (
        "areal-distributed-policy-"
        + str(candidate_manifest["manifest_sha256"])[:20]
    )
    record: dict[str, object] = {
        "schema_version": AREAL_DISTRIBUTED_POLICY_CANDIDATE_SCHEMA,
        "source_binding": deepcopy(dict(binding)),
        "transaction": {
            "transaction_id": transaction_id,
            "source_admission_sha256": audit.source_admission_sha256,
            "source_batch_identity_sha256": source_batch_identity_sha256,
            "members": members,
            "global_sample_count": audit.global_sample_count,
            "global_sample_sha256": audit.global_sample_sha256,
            "trainable_token_count": trainable_token_count,
        },
        "parent": {
            "joint_version": asdict(active_joint_version),
            "joint_version_id": active_joint_version.version_id,
            "policy_engine_version": parent_engine_version,
            "policy_version": active_joint_version.policy,
        },
        "candidate": {
            "policy_version": candidate_policy_version,
            "reserved_policy_engine_version": parent_engine_version + 1,
        },
        "optimizer": {
            "remote_optimizer_receipt": deepcopy(
                dict(remote_optimizer_receipt)
            ),
        },
        "checkpoints": {
            "parent_path": str(parent_path),
            "parent_manifest": parent_manifest,
            "candidate_path": str(candidate_path),
            "candidate_manifest": candidate_manifest,
        },
        "provenance": {
            "areal_commit": areal_commit,
            "project_commit": project_commit,
            "actor_class": JPH_AREAL_DISTRIBUTED_ACTOR_CLASS,
            "controller_class": JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS,
        },
        "evidence_scope": {
            "all_actor_ranks_attested": True,
            "policy_optimizer_update": True,
            "policy_candidate_created": True,
            "policy_weights_published": False,
            "active_joint_version_changed": False,
            "harness_optimizer_update": False,
            "joint_publish": False,
        },
    }
    record["record_sha256"] = _record_sha256(record)
    validate_distributed_policy_candidate(
        record,
        active_joint_version=active_joint_version,
        require_checkpoints=True,
    )
    return record


def validate_distributed_policy_candidate(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
    require_checkpoints: bool = False,
) -> ValidatedDistributedPolicyCandidate:
    """Validate the only Policy candidate schema accepted from ``fsdp:d4``."""

    _require(
        set(record)
        == {
            "schema_version",
            "source_binding",
            "transaction",
            "parent",
            "candidate",
            "optimizer",
            "checkpoints",
            "provenance",
            "evidence_scope",
            "record_sha256",
        }
        and record.get("schema_version")
        == AREAL_DISTRIBUTED_POLICY_CANDIDATE_SCHEMA
        and record.get("record_sha256") == _record_sha256(record),
        "distributed Policy candidate schema or digest differs",
    )
    transaction = record["transaction"]
    top_level_binding = record["source_binding"]
    _require(
        isinstance(top_level_binding, Mapping),
        "distributed Policy top-level source_binding is missing",
    )
    top_level_binding = validate_multi_s_source_binding(top_level_binding)
    _require(
        isinstance(transaction, Mapping)
        and set(transaction)
        == {
            "transaction_id",
            "source_admission_sha256",
            "source_batch_identity_sha256",
            "members",
            "global_sample_count",
            "global_sample_sha256",
            "trainable_token_count",
        },
        "distributed Policy candidate transaction fields differ",
    )
    transaction_id = transaction["transaction_id"]
    members = transaction["members"]
    binding = top_level_binding
    _require(
        isinstance(transaction_id, str)
        and transaction_id
        and _is_sha256(transaction["source_admission_sha256"])
        and _is_sha256(transaction["source_batch_identity_sha256"])
        and isinstance(members, list)
        and bool(members),
        "distributed Policy candidate source identity is invalid",
    )
    expected_member_fields = {
        "member_index",
        "member_claim_sha256",
        "episode_id",
        "admission_record_sha256",
        "source_joint_credit_sha256",
        "sample_ids",
        "trainable_token_count",
    }
    for member_index, member in enumerate(members):
        _require(
            isinstance(member, Mapping)
            and set(member) == expected_member_fields
            and member["member_index"] == member_index
            and _is_sha256(member["member_claim_sha256"])
            and isinstance(member["episode_id"], str)
            and bool(member["episode_id"])
            and _is_sha256(member["admission_record_sha256"])
            and _is_sha256(member["source_joint_credit_sha256"])
            and isinstance(member["sample_ids"], list)
            and bool(member["sample_ids"])
            and all(
                isinstance(sample_id, str) and sample_id
                for sample_id in member["sample_ids"]
            )
            and type(member["trainable_token_count"]) is int
            and member["trainable_token_count"] > 0,
            f"distributed Policy source member {member_index} is invalid",
        )
    _require(
        transaction["source_batch_identity_sha256"]
        == _sha256(
            {
                "schema_version": "jph.m0-ordered-policy-source-batch.v1",
                "members": members,
            }
        )
        and transaction["trainable_token_count"]
        == sum(member["trainable_token_count"] for member in members),
        "distributed Policy source batch digest or token count differs",
    )
    _require(
        transaction["source_admission_sha256"] == binding["record_sha256"]
        and binding["member_claim_sha256s"]
        == [member["member_claim_sha256"] for member in members]
        and binding["s_record_sha256s"]
        == [member["source_joint_credit_sha256"] for member in members]
        and binding["policy_sample_count"] == transaction["global_sample_count"]
        and transaction["global_sample_count"] == 4
        and sum(len(member["sample_ids"]) for member in members) == 4
        and len(
            {
                sample_id
                for member in members
                for sample_id in member["sample_ids"]
            }
        )
        == 4,
        "distributed Policy members differ from canonical multi-S binding",
    )
    parent = record["parent"]
    _require(
        isinstance(parent, Mapping)
        and set(parent)
        == {
            "joint_version",
            "joint_version_id",
            "policy_engine_version",
            "policy_version",
        },
        "distributed Policy parent fields differ",
    )
    joint_version = _joint_version(parent["joint_version"])
    _require(
        parent["joint_version_id"] == joint_version.version_id
        and binding["joint_version_id"] == joint_version.version_id
        and parent["policy_version"] == joint_version.policy
        and type(parent["policy_engine_version"]) is int
        and parent["policy_engine_version"] >= 0,
        "distributed Policy parent identity is invalid",
    )
    if active_joint_version is not None:
        _require(
            joint_version == active_joint_version,
            "distributed Policy candidate differs from active JointVersion",
        )
    candidate = record["candidate"]
    _require(
        isinstance(candidate, Mapping)
        and set(candidate)
        == {"policy_version", "reserved_policy_engine_version"}
        and isinstance(candidate["policy_version"], str)
        and candidate["policy_version"].startswith(
            "areal-distributed-policy-"
        )
        and candidate["reserved_policy_engine_version"]
        == parent["policy_engine_version"] + 1,
        "distributed Policy candidate version is invalid",
    )
    optimizer = record["optimizer"]
    _require(
        isinstance(optimizer, Mapping)
        and set(optimizer) == {"remote_optimizer_receipt"},
        "distributed Policy optimizer fields differ",
    )
    remote = optimizer["remote_optimizer_receipt"]
    _require(isinstance(remote, Mapping), "remote optimizer receipt is missing")
    remote_audit = validate_remote_optimizer_receipt(remote)
    rank_zero = remote["rank_receipts"][0]
    _require(
        remote_audit.transaction_id == transaction_id
        and remote_audit.joint_version_id == joint_version.version_id
        and remote_audit.source_admission_sha256
        == transaction["source_admission_sha256"]
        and remote_audit.global_sample_count == transaction["global_sample_count"]
        and remote_audit.global_sample_sha256 == transaction["global_sample_sha256"]
        and rank_zero["inference_engine_version"]
        == parent["policy_engine_version"],
        "distributed Policy optimizer differs from source or parent",
    )
    checkpoints = record["checkpoints"]
    _require(
        isinstance(checkpoints, Mapping)
        and set(checkpoints)
        == {
            "parent_path",
            "parent_manifest",
            "candidate_path",
            "candidate_manifest",
        }
        and all(
            isinstance(checkpoints[f"{kind}_path"], str)
            and checkpoints[f"{kind}_path"]
            for kind in ("parent", "candidate")
        ),
        "distributed Policy checkpoint fields differ",
    )
    parent_manifest = _validate_manifest(
        checkpoints["parent_manifest"], "parent"
    )
    candidate_manifest = _validate_manifest(
        checkpoints["candidate_manifest"], "candidate"
    )
    _require(
        parent_manifest["manifest_sha256"]
        == rank_zero["parent_dcp_manifest_sha256"]
        and candidate_manifest["manifest_sha256"]
        == rank_zero["candidate_dcp_manifest_sha256"]
        and parent_manifest["manifest_sha256"]
        != candidate_manifest["manifest_sha256"]
        and candidate["policy_version"]
        == "areal-distributed-policy-"
        + str(candidate_manifest["manifest_sha256"])[:20],
        "distributed Policy checkpoint lineage differs from worker receipts",
    )
    if require_checkpoints:
        _require(
            checkpoint_manifest(checkpoints["parent_path"]) == parent_manifest
            and checkpoint_manifest(checkpoints["candidate_path"])
            == candidate_manifest,
            "distributed Policy DCP contents differ from their manifests",
        )
    provenance = record["provenance"]
    _require(
        isinstance(provenance, Mapping)
        and set(provenance)
        == {
            "areal_commit",
            "project_commit",
            "actor_class",
            "controller_class",
        }
        and provenance["areal_commit"] == PINNED_AREAL_COMMIT
        and _is_git_commit(provenance["project_commit"])
        and provenance["actor_class"] == JPH_AREAL_DISTRIBUTED_ACTOR_CLASS
        and provenance["controller_class"]
        == JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS,
        "distributed Policy provenance differs",
    )
    _require(
        record["evidence_scope"]
        == {
            "all_actor_ranks_attested": True,
            "policy_optimizer_update": True,
            "policy_candidate_created": True,
            "policy_weights_published": False,
            "active_joint_version_changed": False,
            "harness_optimizer_update": False,
            "joint_publish": False,
        },
        "distributed Policy candidate evidence scope differs",
    )
    return ValidatedDistributedPolicyCandidate(
        transaction_id=transaction_id,
        episode_id="multi-s:" + str(binding["record_sha256"]),
        parent_joint_version=joint_version,
        source_admission_sha256=str(transaction["source_admission_sha256"]),
        source_joint_credit_sha256=str(binding["record_sha256"]),
        source_binding=deepcopy(dict(binding)),
        source_batch_identity_sha256=str(
            transaction["source_batch_identity_sha256"]
        ),
        trainable_token_count=int(transaction["trainable_token_count"]),
        parent_engine_version=int(parent["policy_engine_version"]),
        reserved_candidate_engine_version=int(
            candidate["reserved_policy_engine_version"]
        ),
        candidate_policy_version=str(candidate["policy_version"]),
        remote_optimizer_receipt_sha256=remote_audit.record_sha256,
        record_sha256=str(record["record_sha256"]),
    )


def require_live_remote_policy_candidate(
    value: object,
) -> LiveArealDistributedPolicyCandidate:
    """Validate the opaque worker exports against the sealed public receipt."""

    _require(
        type(value) is LiveArealDistributedPolicyCandidate
        and value._token is _LIVE_CANDIDATE_TOKEN,
        "W requires a native live distributed Policy candidate capability",
    )
    candidate_audit = validate_distributed_policy_candidate(value.receipt)
    remote_receipt = value.receipt["optimizer"]["remote_optimizer_receipt"]
    audit = validate_remote_optimizer_receipt(remote_receipt)
    _require(
        len(value.worker_states) == 4,
        "live Policy candidate requires four worker state exports",
    )
    rank_receipts = remote_receipt["rank_receipts"]
    for rank, (worker_state, rank_receipt) in enumerate(
        zip(value.worker_states, rank_receipts)
    ):
        _require(
            set(worker_state)
            == {
                "schema_version",
                "transaction_id",
                "worker_rank",
                "aggregate_sha256",
                "policy_candidate_sha256",
                "rank_receipt_sha256",
                "rank_runtime_state",
                "rank0_lr_scheduler_class",
                "rank0_lr_scheduler_state_after",
                "record_sha256",
            }
            and worker_state["schema_version"]
            == "jph.m0-live-policy-worker-state.v1"
            and worker_state["record_sha256"] == _record_sha256(worker_state)
            and worker_state["transaction_id"] == audit.transaction_id
            and worker_state["worker_rank"] == rank
            and worker_state["aggregate_sha256"] == audit.record_sha256
            and worker_state["policy_candidate_sha256"]
            == candidate_audit.record_sha256
            and worker_state["rank_receipt_sha256"]
            == rank_receipt["record_sha256"],
            f"live Policy worker state {rank} differs from T receipt",
        )
        full_runtime = worker_state["rank_runtime_state"]
        _require(
            isinstance(full_runtime, Mapping),
            f"live Policy worker {rank} runtime record is missing",
        )
        from .production_checkpoint import RankRuntimeState

        runtime_fields = set(RankRuntimeState.__dataclass_fields__)
        _require(
            set(full_runtime) == {"schema_version"} | runtime_fields,
            f"live Policy worker {rank} runtime fields differ",
        )
        runtime = RankRuntimeState(
            **{field: deepcopy(full_runtime[field]) for field in runtime_fields}
        )
        _require(
            runtime.to_record() == full_runtime,
            f"live Policy worker {rank} runtime record is invalid",
        )
        torch_rng = full_runtime["torch_rng"]
        compact = rank_receipt["runtime_state"]
        _require(
            isinstance(torch_rng, Mapping)
            and compact["hostname"] == full_runtime["hostname"]
            and compact["local_rank"] == full_runtime["local_rank"]
            and compact["device"] == full_runtime["device"]
            and compact["python_rng_state_sha256"]
            == _sha256(full_runtime["python_random_state"])
            and compact["torch_cpu_rng_state_sha256"]
            == _sha256(torch_rng["cpu_state"])
            and compact["torch_cuda_rng_state_sha256"]
            == _sha256(torch_rng["cuda_states"]),
            f"live Policy worker {rank} RNG state differs from its T digests",
        )
        expected_scheduler = (
            audit.rank0_lr_scheduler_state_after if rank == 0 else None
        )
        scheduler_class = worker_state["rank0_lr_scheduler_class"]
        _require(
            worker_state["rank0_lr_scheduler_state_after"] == expected_scheduler
            and (
                isinstance(scheduler_class, str) and bool(scheduler_class)
                if rank == 0
                else scheduler_class is None
            ),
            f"live Policy worker {rank} scheduler export differs from T",
        )
    return value


def _require_w_branch_id(branch_id: object) -> str:
    _require(
        isinstance(branch_id, str) and branch_id in _W_BRANCH_IDS,
        "distributed W branch ID is invalid",
    )
    return branch_id


def _w_candidate_context(
    candidate: LiveArealDistributedPolicyCandidate,
) -> tuple[
    LiveArealDistributedPolicyCandidate,
    ValidatedDistributedPolicyCandidate,
    Mapping[str, object],
    Sequence[Mapping[str, object]],
]:
    live = require_live_remote_policy_candidate(candidate)
    audit = validate_distributed_policy_candidate(live.receipt)
    remote = live.receipt["optimizer"]["remote_optimizer_receipt"]
    _require(isinstance(remote, Mapping), "distributed optimizer receipt is missing")
    rank_receipts = remote.get("rank_receipts")
    _require(
        isinstance(rank_receipts, list)
        and len(rank_receipts) == 4
        and all(isinstance(item, Mapping) for item in rank_receipts),
        "distributed W requires four rank optimizer receipts",
    )
    return live, audit, remote, rank_receipts


def _restore_rank_state_payload(
    record: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in {"schema_version", "branch_id", "record_sha256"}
    }


def _continuation_rank_state_payload(
    record: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: deepcopy(value)
        for key, value in record.items()
        if key
        not in {
            "schema_version",
            "branch_id",
            "restore_receipt_sha256",
            # PyTorch DCP metadata contains save-instance details that need
            # not be byte-identical across an uninterrupted and a restored
            # save.  Exact state is instead bound by the payload digest.
            "continuation_dcp_manifest_sha256",
            "record_sha256",
        }
    }


def build_distributed_policy_restore_receipt(
    rank_records: Sequence[Mapping[str, object]],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
    branch_id: str,
) -> dict[str, object]:
    """Validate four worker-created restore receipts without granting exact W."""

    branch = _require_w_branch_id(branch_id)
    live, audit, _remote, rank_receipts = _w_candidate_context(candidate)
    records = tuple(rank_records)
    _require(
        len(records) == 4 and all(isinstance(item, Mapping) for item in records),
        "distributed W restore requires exactly four rank receipts",
    )
    expected_fields = {
        "schema_version",
        "branch_id",
        "transaction_id",
        "policy_candidate_sha256",
        "worker_rank",
        "world_size",
        "engine_name",
        "physical_gpu_id",
        "candidate_dcp_manifest_sha256",
        "optimizer_step",
        "lr_scheduler_state_sha256",
        "runtime_rng_state_sha256",
        "actor_public_version",
        "evidence_scope",
        "record_sha256",
    }
    load_observed = branch != "uninterrupted"
    expected_scope = {
        "live_candidate_state_attested": branch == "uninterrupted",
        "candidate_dcp_loaded": load_observed,
        "optimizer_state_loaded": load_observed,
        "lr_scheduler_state_loaded": load_observed,
        "rank_rng_state_loaded": load_observed,
        "rank_scheduler_rng_state_attested": True,
        "continuation_executed": False,
        "exact_joint_recovery": False,
    }
    scheduler_sha256 = rank_receipts[0]["lr_scheduler_state_after_sha256"]
    candidate_manifest_sha256 = live.receipt["checkpoints"][
        "candidate_manifest"
    ]["manifest_sha256"]
    normalized: list[dict[str, object]] = []
    for rank, (record, worker_state, rank_receipt) in enumerate(
        zip(records, live.worker_states, rank_receipts)
    ):
        _require(
            set(record) == expected_fields
            and record.get("schema_version")
            == M0_DISTRIBUTED_POLICY_RESTORE_RANK_SCHEMA
            and record.get("record_sha256") == _record_sha256(record)
            and record.get("branch_id") == branch
            and record.get("transaction_id") == audit.transaction_id
            and record.get("policy_candidate_sha256") == audit.record_sha256
            and record.get("worker_rank") == rank
            and record.get("world_size") == 4
            and record.get("engine_name") == f"actor/{rank}"
            and record.get("physical_gpu_id")
            == rank_receipt["physical_gpu_id"]
            and record.get("candidate_dcp_manifest_sha256")
            == candidate_manifest_sha256
            and record.get("optimizer_step")
            == rank_receipt["optimizer_step_after"]
            and record.get("lr_scheduler_state_sha256") == scheduler_sha256
            and record.get("runtime_rng_state_sha256")
            == _sha256(worker_state["rank_runtime_state"])
            and record.get("actor_public_version") == audit.parent_engine_version
            and record.get("evidence_scope") == expected_scope,
            f"distributed W restore rank {rank} differs from T worker state",
        )
        normalized.append(_restore_rank_state_payload(record))
    restore_state_sha256 = _sha256(normalized)
    aggregate: dict[str, object] = {
        "schema_version": M0_DISTRIBUTED_POLICY_RESTORE_SCHEMA,
        "branch_id": branch,
        "transaction_id": audit.transaction_id,
        "policy_candidate_sha256": audit.record_sha256,
        "restore_state_sha256": restore_state_sha256,
        "rank_receipts": [deepcopy(dict(record)) for record in records],
        "evidence_scope": {
            "all_four_actor_ranks_participated": True,
            "live_candidate_state_attested": branch == "uninterrupted",
            "candidate_dcp_loaded": load_observed,
            "optimizer_state_loaded": load_observed,
            "lr_scheduler_state_loaded": load_observed,
            "rank_rng_state_loaded": load_observed,
            "rank_scheduler_rng_state_attested": True,
            "continuation_executed": False,
            "exact_joint_recovery": False,
        },
    }
    aggregate["record_sha256"] = _record_sha256(aggregate)
    return aggregate


def validate_distributed_policy_restore_receipt(
    record: Mapping[str, object],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
    branch_id: str,
) -> Mapping[str, object]:
    _require(
        set(record)
        == {
            "schema_version",
            "branch_id",
            "transaction_id",
            "policy_candidate_sha256",
            "restore_state_sha256",
            "rank_receipts",
            "evidence_scope",
            "record_sha256",
        }
        and record.get("schema_version") == M0_DISTRIBUTED_POLICY_RESTORE_SCHEMA
        and record.get("record_sha256") == _record_sha256(record)
        and _is_sha256(record.get("restore_state_sha256")),
        "distributed W restore aggregate schema or digest differs",
    )
    rebuilt = build_distributed_policy_restore_receipt(
        record["rank_receipts"],  # type: ignore[arg-type]
        candidate=candidate,
        branch_id=branch_id,
    )
    _require(
        _canonical_json(record) == _canonical_json(rebuilt),
        "distributed W restore aggregate differs from four rank receipts",
    )
    return record


def build_distributed_policy_continuation_receipt(
    rank_records: Sequence[Mapping[str, object]],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
    restore_receipt: Mapping[str, object],
    branch_id: str,
) -> dict[str, object]:
    """Aggregate one diagnostic continuation from all four restored ranks."""

    branch = _require_w_branch_id(branch_id)
    _require(
        branch != "final-restore",
        "final distributed W restore cannot execute a continuation",
    )
    validate_distributed_policy_restore_receipt(
        restore_receipt,
        candidate=candidate,
        branch_id=branch,
    )
    live, audit, _remote, rank_receipts = _w_candidate_context(candidate)
    records = tuple(rank_records)
    _require(
        len(records) == 4 and all(isinstance(item, Mapping) for item in records),
        "distributed W continuation requires exactly four rank receipts",
    )
    expected_fields = {
        "schema_version",
        "branch_id",
        "transaction_id",
        "policy_candidate_sha256",
        "restore_receipt_sha256",
        "worker_rank",
        "world_size",
        "engine_name",
        "physical_gpu_id",
        "source_binding_sha256",
        "candidate_dcp_manifest_sha256",
        "continuation_dcp_manifest_sha256",
        "continuation_dcp_payload_sha256",
        "local_sample_indices",
        "optimizer_step_before",
        "optimizer_step_after",
        "lr_scheduler_state_before_sha256",
        "lr_scheduler_state_after_sha256",
        "runtime_rng_state_after_sha256",
        "actor_public_version",
        "evidence_scope",
        "record_sha256",
    }
    expected_scope = {
        "bound_pre_batch_sources_reused": True,
        "diagnostic_policy_optimizer_step_observed": True,
        "continuation_dcp_with_optimizer_observed": True,
        "exact_joint_recovery": False,
    }
    candidate_manifest_sha256 = live.receipt["checkpoints"][
        "candidate_manifest"
    ]["manifest_sha256"]
    normalized: list[dict[str, object]] = []
    common_continuation_manifest: str | None = None
    common_continuation_payload: str | None = None
    for rank, (record, rank_receipt) in enumerate(zip(records, rank_receipts)):
        continuation_manifest = record.get("continuation_dcp_manifest_sha256")
        continuation_payload = record.get("continuation_dcp_payload_sha256")
        _require(
            set(record) == expected_fields
            and record.get("schema_version")
            == M0_DISTRIBUTED_POLICY_CONTINUATION_RANK_SCHEMA
            and record.get("record_sha256") == _record_sha256(record)
            and record.get("branch_id") == branch
            and record.get("transaction_id") == audit.transaction_id
            and record.get("policy_candidate_sha256") == audit.record_sha256
            and record.get("restore_receipt_sha256")
            == restore_receipt["record_sha256"]
            and record.get("worker_rank") == rank
            and record.get("world_size") == 4
            and record.get("engine_name") == f"actor/{rank}"
            and record.get("physical_gpu_id")
            == rank_receipt["physical_gpu_id"]
            and record.get("source_binding_sha256")
            == live.receipt["source_binding"]["record_sha256"]
            and record.get("candidate_dcp_manifest_sha256")
            == candidate_manifest_sha256
            and _is_sha256(continuation_manifest)
            and continuation_manifest != candidate_manifest_sha256
            and _is_sha256(continuation_payload)
            and record.get("local_sample_indices")
            == rank_receipt["local_sample_indices"]
            and record.get("optimizer_step_before")
            == rank_receipt["optimizer_step_after"]
            and record.get("optimizer_step_after")
            == rank_receipt["optimizer_step_after"] + 1
            and record.get("lr_scheduler_state_before_sha256")
            == rank_receipt["lr_scheduler_state_after_sha256"]
            and _is_sha256(record.get("lr_scheduler_state_after_sha256"))
            and record.get("lr_scheduler_state_after_sha256")
            != rank_receipt["lr_scheduler_state_after_sha256"]
            and _is_sha256(record.get("runtime_rng_state_after_sha256"))
            and record.get("actor_public_version") == audit.parent_engine_version
            and record.get("evidence_scope") == expected_scope,
            f"distributed W continuation rank {rank} differs from restored T state",
        )
        if common_continuation_manifest is None:
            common_continuation_manifest = str(continuation_manifest)
        _require(
            continuation_manifest == common_continuation_manifest,
            "distributed W continuation ranks disagree on DCP manifest",
        )
        if common_continuation_payload is None:
            common_continuation_payload = str(continuation_payload)
        _require(
            continuation_payload == common_continuation_payload,
            "distributed W continuation ranks disagree on DCP payload",
        )
        normalized.append(_continuation_rank_state_payload(record))
    continuation_state_sha256 = _sha256(normalized)
    aggregate: dict[str, object] = {
        "schema_version": M0_DISTRIBUTED_POLICY_CONTINUATION_SCHEMA,
        "branch_id": branch,
        "transaction_id": audit.transaction_id,
        "policy_candidate_sha256": audit.record_sha256,
        "restore_receipt_sha256": restore_receipt["record_sha256"],
        "continuation_state_sha256": continuation_state_sha256,
        "rank_receipts": [deepcopy(dict(record)) for record in records],
        "evidence_scope": {
            "all_four_actor_ranks_continued": True,
            "bound_pre_batch_sources_reused": True,
            "diagnostic_policy_optimizer_step_observed": True,
            "continuation_dcp_with_optimizer_observed": True,
            "exact_joint_recovery": False,
        },
    }
    aggregate["record_sha256"] = _record_sha256(aggregate)
    return aggregate


def validate_distributed_policy_continuation_receipt(
    record: Mapping[str, object],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
    restore_receipt: Mapping[str, object],
    branch_id: str,
) -> Mapping[str, object]:
    _require(
        set(record)
        == {
            "schema_version",
            "branch_id",
            "transaction_id",
            "policy_candidate_sha256",
            "restore_receipt_sha256",
            "continuation_state_sha256",
            "rank_receipts",
            "evidence_scope",
            "record_sha256",
        }
        and record.get("schema_version")
        == M0_DISTRIBUTED_POLICY_CONTINUATION_SCHEMA
        and record.get("record_sha256") == _record_sha256(record)
        and _is_sha256(record.get("continuation_state_sha256")),
        "distributed W continuation aggregate schema or digest differs",
    )
    rebuilt = build_distributed_policy_continuation_receipt(
        record["rank_receipts"],  # type: ignore[arg-type]
        candidate=candidate,
        restore_receipt=restore_receipt,
        branch_id=branch_id,
    )
    _require(
        _canonical_json(record) == _canonical_json(rebuilt),
        "distributed W continuation aggregate differs from four rank receipts",
    )
    return record


def build_distributed_serving_export_receipt(
    rank_receipts: Sequence[Mapping[str, object]],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
) -> dict[str, object]:
    """Aggregate four worker-authenticated collective HF export receipts."""

    live = require_live_remote_policy_candidate(candidate)
    audit = validate_distributed_policy_candidate(
        live.receipt,
        require_checkpoints=True,
    )
    _require(len(rank_receipts) == 4, "serving export requires four rank receipts")
    ranks = [deepcopy(dict(receipt)) for receipt in rank_receipts]
    shared: dict[str, object] | None = None
    for rank, receipt in enumerate(ranks):
        _require(
            set(receipt)
            == {
                "schema_version",
                "transaction_id",
                "policy_candidate_sha256",
                "worker_rank",
                "world_size",
                "parent",
                "candidate",
                "evidence_scope",
                "record_sha256",
            }
            and receipt["schema_version"]
            == M0_DISTRIBUTED_SERVING_EXPORT_RANK_SCHEMA
            and receipt["record_sha256"] == _record_sha256(receipt)
            and receipt["transaction_id"] == audit.transaction_id
            and receipt["policy_candidate_sha256"] == audit.record_sha256
            and receipt["worker_rank"] == rank
            and receipt["world_size"] == 4
            and receipt["evidence_scope"]
            == {
                "collective_hf_save_executed": True,
                "candidate_dcp_restored": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
            f"serving export rank {rank} receipt differs",
        )
        current = {
            "parent": receipt["parent"],
            "candidate": receipt["candidate"],
        }
        if shared is None:
            shared = current
        else:
            _require(
                _canonical_json(current) == _canonical_json(shared),
                "serving export ranks disagree on manifests or parameters",
            )
    _require(shared is not None, "serving export shared receipt is missing")
    checkpoints = live.receipt["checkpoints"]
    for kind in ("parent", "candidate"):
        value = shared[kind]
        _require(
            isinstance(value, Mapping)
            and set(value)
            == {
                "dcp_path",
                "dcp_manifest_sha256",
                "export_path",
                "export_manifest",
                "parameter_sha256",
            }
            and value["dcp_path"] == checkpoints[f"{kind}_path"]
            and value["dcp_manifest_sha256"]
            == checkpoints[f"{kind}_manifest"]["manifest_sha256"]
            and _is_sha256(value["parameter_sha256"]),
            f"serving export {kind} differs from T checkpoints",
        )
        _validate_manifest(value["export_manifest"], f"{kind} HF export")
    _require(
        shared["parent"]["parameter_sha256"]
        != shared["candidate"]["parameter_sha256"],
        "serving export parent and candidate parameters are identical",
    )
    record: dict[str, object] = {
        "schema_version": M0_DISTRIBUTED_SERVING_EXPORT_SCHEMA,
        "transaction_id": audit.transaction_id,
        "policy_candidate_sha256": audit.record_sha256,
        "parent": deepcopy(shared["parent"]),
        "candidate": deepcopy(shared["candidate"]),
        "rank_receipts": ranks,
        "evidence_scope": {
            "all_four_actor_ranks_attested": True,
            "collective_hf_save_executed": True,
            "candidate_dcp_restored": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    record["record_sha256"] = _record_sha256(record)
    return record


def validate_distributed_serving_export_receipt(
    record: Mapping[str, object],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
) -> Mapping[str, object]:
    """Rebuild a collective export receipt from its four worker records."""

    _require(
        set(record)
        == {
            "schema_version",
            "transaction_id",
            "policy_candidate_sha256",
            "parent",
            "candidate",
            "rank_receipts",
            "evidence_scope",
            "record_sha256",
        }
        and record.get("schema_version") == M0_DISTRIBUTED_SERVING_EXPORT_SCHEMA
        and record.get("record_sha256") == _record_sha256(record),
        "distributed serving export aggregate schema or digest differs",
    )
    rank_receipts = record.get("rank_receipts")
    _require(
        isinstance(rank_receipts, list)
        and len(rank_receipts) == 4
        and all(isinstance(item, Mapping) for item in rank_receipts),
        "distributed serving export aggregate lacks four rank receipts",
    )
    rebuilt = build_distributed_serving_export_receipt(
        rank_receipts,
        candidate=candidate,
    )
    _require(
        _canonical_json(record) == _canonical_json(rebuilt),
        "distributed serving export aggregate differs from rank receipts",
    )
    return record


def build_distributed_policy_current_state_receipt(
    rank_receipts: Sequence[Mapping[str, object]],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
    distributed_serving_export_receipt: Mapping[str, object],
    candidate_serving_export_lineage_sha256: str,
) -> dict[str, object]:
    """Bind the post-export live actor state back to T and all export ranks."""

    live, audit, _remote, optimizer_rank_receipts = _w_candidate_context(candidate)
    export = validate_distributed_serving_export_receipt(
        distributed_serving_export_receipt,
        candidate=live,
    )
    _require(
        _is_sha256(candidate_serving_export_lineage_sha256),
        "candidate serving-export lineage digest is invalid",
    )
    candidate_manifest_sha256 = live.receipt["checkpoints"][
        "candidate_manifest"
    ]["manifest_sha256"]
    candidate_serving_parameter_sha256 = export["candidate"]["parameter_sha256"]
    scheduler_sha256 = optimizer_rank_receipts[0][
        "lr_scheduler_state_after_sha256"
    ]
    records = tuple(rank_receipts)
    _require(
        len(records) == 4 and all(isinstance(item, Mapping) for item in records),
        "current Policy state requires exactly four rank receipts",
    )
    expected_fields = {
        "schema_version",
        "transaction_id",
        "policy_candidate_sha256",
        "distributed_serving_export_receipt_sha256",
        "collective_export_rank_receipt_sha256",
        "candidate_serving_export_lineage_sha256",
        "candidate_serving_parameter_sha256",
        "worker_rank",
        "world_size",
        "engine_name",
        "physical_gpu_id",
        "candidate_dcp_manifest_sha256",
        "optimizer_step",
        "lr_scheduler_state_sha256",
        "runtime_rng_state_sha256",
        "actor_public_version",
        "evidence_scope",
        "record_sha256",
    }
    expected_scope = {
        "candidate_dcp_restored_after_collective_export": True,
        "optimizer_state_attested": True,
        "lr_scheduler_state_attested": True,
        "rank_rng_state_restored_and_attested": True,
        "candidate_serving_parameter_digest_bound": True,
        "live_weight_parameter_digest_observed": False,
        "policy_optimizer_update": False,
        "harness_optimizer_update": False,
    }
    normalized: list[dict[str, object]] = []
    for rank, (record, worker_state, optimizer_receipt, export_receipt) in enumerate(
        zip(
            records,
            live.worker_states,
            optimizer_rank_receipts,
            export["rank_receipts"],
        )
    ):
        _require(
            set(record) == expected_fields
            and record.get("schema_version")
            == M0_DISTRIBUTED_POLICY_CURRENT_STATE_RANK_SCHEMA
            and record.get("record_sha256") == _record_sha256(record)
            and record.get("transaction_id") == audit.transaction_id
            and record.get("policy_candidate_sha256") == audit.record_sha256
            and record.get("distributed_serving_export_receipt_sha256")
            == export["record_sha256"]
            and record.get("collective_export_rank_receipt_sha256")
            == export_receipt["record_sha256"]
            and record.get("candidate_serving_export_lineage_sha256")
            == candidate_serving_export_lineage_sha256
            and record.get("candidate_serving_parameter_sha256")
            == candidate_serving_parameter_sha256
            and record.get("worker_rank") == rank
            and record.get("world_size") == 4
            and record.get("engine_name") == f"actor/{rank}"
            and record.get("physical_gpu_id")
            == optimizer_receipt["physical_gpu_id"]
            and record.get("candidate_dcp_manifest_sha256")
            == candidate_manifest_sha256
            and record.get("optimizer_step")
            == optimizer_receipt["optimizer_step_after"]
            and record.get("lr_scheduler_state_sha256") == scheduler_sha256
            and record.get("runtime_rng_state_sha256")
            == _sha256(worker_state["rank_runtime_state"])
            and record.get("actor_public_version") == audit.parent_engine_version
            and record.get("evidence_scope") == expected_scope,
            f"current Policy state rank {rank} differs from T/export state",
        )
        normalized.append(
            {
                key: deepcopy(value)
                for key, value in record.items()
                if key not in {"schema_version", "record_sha256"}
            }
        )
    aggregate: dict[str, object] = {
        "schema_version": M0_DISTRIBUTED_POLICY_CURRENT_STATE_SCHEMA,
        "transaction_id": audit.transaction_id,
        "policy_candidate_sha256": audit.record_sha256,
        "distributed_serving_export_receipt_sha256": export["record_sha256"],
        "candidate_serving_export_lineage_sha256": (
            candidate_serving_export_lineage_sha256
        ),
        "candidate_serving_parameter_sha256": candidate_serving_parameter_sha256,
        "current_state_sha256": _sha256(normalized),
        "rank_receipts": [deepcopy(dict(item)) for item in records],
        "evidence_scope": {
            "all_four_actor_ranks_attested": True,
            **expected_scope,
        },
    }
    aggregate["record_sha256"] = _record_sha256(aggregate)
    return aggregate


def validate_distributed_policy_current_state_receipt(
    record: Mapping[str, object],
    *,
    candidate: LiveArealDistributedPolicyCandidate,
    distributed_serving_export_receipt: Mapping[str, object],
    candidate_serving_export_lineage_sha256: str,
) -> Mapping[str, object]:
    """Fail closed unless X consumes the exact four-rank post-export state."""

    _require(
        set(record)
        == {
            "schema_version",
            "transaction_id",
            "policy_candidate_sha256",
            "distributed_serving_export_receipt_sha256",
            "candidate_serving_export_lineage_sha256",
            "candidate_serving_parameter_sha256",
            "current_state_sha256",
            "rank_receipts",
            "evidence_scope",
            "record_sha256",
        }
        and record.get("schema_version")
        == M0_DISTRIBUTED_POLICY_CURRENT_STATE_SCHEMA
        and record.get("record_sha256") == _record_sha256(record)
        and _is_sha256(record.get("current_state_sha256")),
        "current Policy state aggregate schema or digest differs",
    )
    rank_receipts = record.get("rank_receipts")
    _require(
        isinstance(rank_receipts, list)
        and len(rank_receipts) == 4
        and all(isinstance(item, Mapping) for item in rank_receipts),
        "current Policy state aggregate lacks four rank receipts",
    )
    rebuilt = build_distributed_policy_current_state_receipt(
        rank_receipts,
        candidate=candidate,
        distributed_serving_export_receipt=distributed_serving_export_receipt,
        candidate_serving_export_lineage_sha256=(
            candidate_serving_export_lineage_sha256
        ),
    )
    _require(
        _canonical_json(record) == _canonical_json(rebuilt),
        "current Policy state aggregate differs from four rank receipts",
    )
    return record


class JPHFSDPPPOActor(_ArealFSDPPPOActor):  # type: ignore[misc,valid-type]
    """Pinned AReaL actor with a worker-local atomic M0 update RPC."""

    def __init__(self, config: object):
        if _AREAL_IMPORT_ERROR is not None:  # pragma: no cover - local gate
            raise ArealDistributedPolicyError(
                "pinned AReaL FSDPPPOActor is unavailable"
            ) from _AREAL_IMPORT_ERROR
        super().__init__(config)  # type: ignore[misc]
        self._jph_capture_optimizer_step = False
        self._jph_optimizer_step_count = 0
        self._jph_last_optimizer_result: dict[str, object] | None = None
        self._jph_pending_m0_transaction: dict[str, object] | None = None
        self._jph_w_restore: dict[str, object] | None = None

    def optimizer_step(self):
        """Capture the real FSDP optimizer result at its worker-side source."""

        result = super().optimizer_step()
        if self._jph_capture_optimizer_step:
            _require(
                isinstance(result, Mapping),
                "AReaL optimizer_step returned no worker-local statistics",
            )
            self._jph_optimizer_step_count += 1
            self._jph_last_optimizer_result = deepcopy(dict(result))
        return result

    @classmethod
    def as_controller(cls, config: object, scheduler: object):
        _require(
            _AREAL_IMPORT_ERROR is None,
            "pinned AReaL controller dependencies are unavailable",
        )
        _require(
            getattr(config, "_version", None) != "v2",
            "M0 distributed receipt does not support the gateway controller",
        )
        _require(
            getattr(config, "backend", None) == "fsdp:d4",
            "M0 distributed actor backend must be fsdp:d4",
        )
        return JPHPPOActorController(
            train_engine=cls,
            config=config,
            scheduler=scheduler,
        )

    def _gather_objects(self, value: object) -> list[object]:
        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        gathered: list[object] = [None] * 4
        dist.all_gather_object(gathered, value, group=self.cpu_group)
        return gathered

    def _run_group_phase(self, name: str, operation: Callable[[], _T]) -> _T:
        result: _T | None = None
        local_error: str | None = None
        try:
            result = operation()
        except BaseException as exc:
            local_error = _exception_summary(exc)
        errors = self._gather_objects(local_error)
        failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
        if failures:
            raise ArealDistributedPolicyError(
                f"distributed Policy phase {name} failed: " + "; ".join(failures)
            )
        return result  # type: ignore[return-value]

    def _require_group_common(self, name: str, value: object) -> None:
        gathered = self._gather_objects(value)
        encoded = [_canonical_json(item) for item in gathered]
        _require(
            all(item == encoded[0] for item in encoded),
            f"actor ranks disagree on {name}",
        )

    def _collectively_require_new_checkpoint_path(
        self,
        *,
        name: str,
        path: Path,
    ) -> Path:
        """Barrier all ranks after observing a shared DCP path as new."""

        observation = {
            "path": str(path),
            "exists": path.exists(),
            "is_symlink": path.is_symlink(),
        }
        # No rank may create the shared directory until every rank has made
        # and exchanged its pre-save observation.  Rechecking after this
        # barrier would itself race with a faster rank entering DCP save.
        self._require_group_common(f"{name} path preflight", observation)
        _require(
            observation == {
                "path": str(path),
                "exists": False,
                "is_symlink": False,
            },
            f"{name} path must be collectively new",
        )
        return path

    def _rollback_pending_group(self, transaction_id: str) -> bool:
        pending = self._jph_pending_m0_transaction
        descriptor = None
        if pending is not None:
            descriptor = {
                "transaction_id": pending.get("transaction_id"),
                "parent_path": pending.get("parent_path"),
                "optimizer_step_before": pending.get("optimizer_step_before"),
                "scheduler_state_before_sha256": _sha256(
                    pending.get("scheduler_state_before")
                ),
            }
        descriptors = self._gather_objects(descriptor)
        if all(item is None for item in descriptors):
            return False
        _require(
            all(item == descriptors[0] and item is not None for item in descriptors)
            and descriptors[0]["transaction_id"] == transaction_id,
            "actor ranks disagree on the pending rollback transaction",
        )
        _require(pending is not None, "rank-local rollback state is missing")

        def _rollback() -> None:
            _require(SaveLoadMeta is not None, "AReaL SaveLoadMeta is unavailable")
            self.load(
                meta=SaveLoadMeta(
                    path=str(pending["parent_path"]),
                    weight_format="dcp",
                    with_optim=True,
                )
            )
            _restore_lr_scheduler_state(
                self,
                pending["scheduler_state_before"],  # type: ignore[arg-type]
            )
            restored_optimizer_step = _optimizer_step(self)
            _require(
                restored_optimizer_step == pending["optimizer_step_before"],
                "parent DCP did not restore the rank-local optimizer step "
                f"(parent baseline={pending['optimizer_step_before']}, "
                f"observed={restored_optimizer_step})",
            )

        self._run_group_phase("parent rollback", _rollback)
        self._jph_pending_m0_transaction = None
        self._jph_w_restore = None
        return True

    def rollback_m0_policy_candidate(self, *, transaction_id: str) -> dict[str, object]:
        _require(
            isinstance(transaction_id, str) and transaction_id,
            "rollback transaction ID is missing",
        )
        rolled_back = self._rollback_pending_group(transaction_id)
        return {"transaction_id": transaction_id, "rolled_back": rolled_back}

    def bind_m0_policy_aggregate(
        self,
        *,
        transaction_id: str,
        aggregate_path: str,
        aggregate_sha256: str,
        policy_candidate_path: str,
        policy_candidate_sha256: str,
    ) -> dict[str, object]:
        """Bind the persisted four-rank aggregate back into every worker."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        rank = dist.get_rank()

        def _bind() -> dict[str, object]:
            pending = self._jph_pending_m0_transaction
            _require(pending is not None, "worker has no pending M0 candidate")
            _require(
                pending.get("transaction_id") == transaction_id,
                "aggregate transaction differs from the pending worker state",
            )
            rank_receipt = _validated_pending_m0_candidate_state(pending)
            path = require_within_configured_root(aggregate_path)
            _require(path.is_file() and not path.is_symlink(), "aggregate is missing")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArealDistributedPolicyError(
                    "persisted remote optimizer aggregate is unreadable"
                ) from exc
            _require(isinstance(raw, Mapping), "persisted aggregate is not an object")
            audit = validate_remote_optimizer_receipt(raw)
            _require(
                audit.transaction_id == transaction_id
                and audit.record_sha256 == aggregate_sha256
                and raw["rank_receipts"][rank] == pending.get("rank_receipt"),
                "persisted aggregate differs from the rank-local receipt",
            )
            candidate_path = require_within_configured_root(policy_candidate_path)
            _require(
                candidate_path.is_file() and not candidate_path.is_symlink(),
                "persisted distributed Policy candidate is missing",
            )
            try:
                candidate_record = json.loads(
                    candidate_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArealDistributedPolicyError(
                    "persisted distributed Policy candidate is unreadable"
                ) from exc
            _require(
                isinstance(candidate_record, Mapping),
                "persisted distributed Policy candidate is not an object",
            )
            candidate_audit = validate_distributed_policy_candidate(
                candidate_record,
                require_checkpoints=True,
            )
            _require(
                candidate_audit.transaction_id == transaction_id
                and candidate_audit.record_sha256 == policy_candidate_sha256
                and candidate_record["optimizer"]["remote_optimizer_receipt"]
                == raw,
                "persisted Policy candidate differs from its worker aggregate",
            )
            checkpoints = candidate_record["checkpoints"]
            _require(
                checkpoints["parent_path"] == pending["parent_path"]
                and checkpoints["parent_manifest"]["manifest_sha256"]
                == rank_receipt["parent_dcp_manifest_sha256"]
                and checkpoints["candidate_path"] == pending["candidate_path"]
                and checkpoints["candidate_manifest"]["manifest_sha256"]
                == pending["candidate_manifest_sha256"],
                "persisted Policy candidate checkpoints differ from T",
            )
            before = {
                "actor_version": self.get_version(),
                "optimizer_step": _optimizer_step(self),
                "scheduler_state_sha256": _sha256(_lr_scheduler_state(self)),
            }
            pending.update(
                {
                    "aggregate_path": str(path),
                    "aggregate_sha256": aggregate_sha256,
                    "policy_candidate_path": str(candidate_path),
                    "policy_candidate_sha256": policy_candidate_sha256,
                }
            )
            after = {
                "actor_version": self.get_version(),
                "optimizer_step": _optimizer_step(self),
                "scheduler_state_sha256": _sha256(_lr_scheduler_state(self)),
            }
            _require(before == after, "aggregate binding changed worker training state")
            return {
                "transaction_id": transaction_id,
                "worker_rank": rank,
                "aggregate_sha256": aggregate_sha256,
                "policy_candidate_sha256": policy_candidate_sha256,
            }

        return self._run_group_phase("aggregate binding", _bind)

    def export_m0_policy_candidate_live_state(
        self,
        *,
        transaction_id: str,
    ) -> dict[str, object]:
        """Export worker-owned W inputs while the rollback capability is live."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        rank = dist.get_rank()
        pending = self._jph_pending_m0_transaction
        _require(
            pending is not None
            and pending.get("transaction_id") == transaction_id
            and isinstance(pending.get("aggregate_sha256"), str)
            and isinstance(pending.get("policy_candidate_sha256"), str),
            "worker has no aggregate-bound pending M0 candidate",
        )
        rank_receipt = _validated_pending_m0_candidate_state(pending)
        record: dict[str, object] = {
            "schema_version": "jph.m0-live-policy-worker-state.v1",
            "transaction_id": transaction_id,
            "worker_rank": rank,
            "aggregate_sha256": pending["aggregate_sha256"],
            "policy_candidate_sha256": pending["policy_candidate_sha256"],
            "rank_receipt_sha256": rank_receipt["record_sha256"],
            "rank_runtime_state": deepcopy(pending["rank_runtime_state"]),
            "rank0_lr_scheduler_class": (
                f"{type(self.lr_scheduler).__module__}."
                f"{type(self.lr_scheduler).__qualname__}"
                if rank == 0
                else None
            ),
            "rank0_lr_scheduler_state_after": (
                deepcopy(pending["scheduler_state_after"]) if rank == 0 else None
            ),
        }
        record["record_sha256"] = _record_sha256(record)
        return record

    def materialize_m0_serving_export_pair(
        self,
        *,
        transaction_id: str,
        policy_candidate_sha256: str,
        parent_dcp_path: str,
        parent_dcp_manifest_sha256: str,
        candidate_dcp_path: str,
        candidate_dcp_manifest_sha256: str,
        parent_export_path: str,
        candidate_export_path: str,
    ) -> dict[str, object]:
        """Collectively export parent/candidate HF weights on all four ranks."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        rank = dist.get_rank()
        pending = self._jph_pending_m0_transaction
        _require(
            pending is not None
            and pending.get("transaction_id") == transaction_id
            and pending.get("policy_candidate_sha256")
            == policy_candidate_sha256
            and pending.get("post_export_candidate_state") is None,
            "serving export has no matching pending Policy candidate",
        )
        rank_receipt = _validated_pending_m0_candidate_state(pending)
        request = {
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "parent_dcp_path": parent_dcp_path,
            "parent_dcp_manifest_sha256": parent_dcp_manifest_sha256,
            "candidate_dcp_path": candidate_dcp_path,
            "candidate_dcp_manifest_sha256": candidate_dcp_manifest_sha256,
            "parent_export_path": parent_export_path,
            "candidate_export_path": candidate_export_path,
        }
        self._require_group_common("serving export request", request)
        _require(SaveLoadMeta is not None, "AReaL SaveLoadMeta is unavailable")
        parent_dcp = require_within_configured_root(parent_dcp_path)
        candidate_dcp = _validated_pending_w_candidate_path(
            pending=pending,
            candidate_path=candidate_dcp_path,
            candidate_dcp_manifest_sha256=candidate_dcp_manifest_sha256,
        )
        parent_export = require_within_configured_root(parent_export_path)
        candidate_export = require_within_configured_root(candidate_export_path)
        _require(
            str(parent_dcp) == pending["parent_path"]
            and parent_dcp_manifest_sha256
            == rank_receipt.get("parent_dcp_manifest_sha256"),
            "serving export parent DCP binding differs from T",
        )
        _require(
            checkpoint_manifest(parent_dcp)["manifest_sha256"]
            == parent_dcp_manifest_sha256,
            "serving export parent DCP contents differ from T",
        )
        tokenizer = getattr(self, "tokenizer", None)
        _require(
            tokenizer is not None
            and callable(getattr(tokenizer, "save_pretrained", None)),
            "serving export requires the actor's initialized tokenizer",
        )

        def _export_one(source: Path, destination: Path) -> dict[str, object]:
            from .areal_production_worker import (
                _directory_manifest,
                _load_safetensor_export,
                _parameter_digest,
            )

            self.load(
                meta=SaveLoadMeta(
                    path=str(source),
                    weight_format="dcp",
                    with_optim=True,
                )
            )
            self.save(
                meta=SaveLoadMeta(
                    path=str(destination),
                    weight_format="hf",
                    with_optim=False,
                    tokenizer=tokenizer,
                    processor=getattr(self, "processor", None),
                )
            )
            return {
                "manifest": _directory_manifest(destination),
                "parameter_sha256": _parameter_digest(
                    _load_safetensor_export(destination)
                ),
            }

        export_error: BaseException | None = None
        parent_result: dict[str, object] | None = None
        candidate_result: dict[str, object] | None = None
        try:
            parent_result = self._run_group_phase(
                "parent collective HF serving export",
                lambda: _export_one(parent_dcp, parent_export),
            )
            self._require_group_common("parent HF export", parent_result)
            candidate_result = self._run_group_phase(
                "candidate collective HF serving export",
                lambda: _export_one(candidate_dcp, candidate_export),
            )
            self._require_group_common("candidate HF export", candidate_result)
        except BaseException as exc:
            export_error = exc
        finally:

            def _restore_candidate() -> None:
                from .production_checkpoint import (
                    _assert_rank_rng_restored,
                    _restore_rank_rng,
                )

                self.load(
                    meta=SaveLoadMeta(
                        path=str(candidate_dcp),
                        weight_format="dcp",
                        with_optim=True,
                    )
                )
                _restore_lr_scheduler_state(
                    self,
                    pending["scheduler_state_after"],  # type: ignore[arg-type]
                )
                rank_runtime_state = pending["rank_runtime_state"]
                _require(
                    isinstance(rank_runtime_state, Mapping),
                    "serving export candidate RNG source is missing",
                )
                _restore_rank_rng(rank_runtime_state, harness_policy=None)
                _assert_rank_rng_restored(rank_runtime_state, harness_policy=None)
                _require(
                    isinstance(candidate_result, Mapping)
                    and _optimizer_step(self) == pending["optimizer_step_after"]
                    and _sha256(_lr_scheduler_state(self))
                    == rank_receipt[
                        "lr_scheduler_state_after_sha256"
                    ],
                    "serving export did not restore the complete candidate state",
                )
                pending["post_export_candidate_state"] = {
                    "candidate_dcp_manifest_sha256": (
                        candidate_dcp_manifest_sha256
                    ),
                    "candidate_serving_parameter_sha256": candidate_result[
                        "parameter_sha256"
                    ],
                    "optimizer_step": pending["optimizer_step_after"],
                    "lr_scheduler_state_sha256": _sha256(
                        pending["scheduler_state_after"]
                    ),
                    "runtime_rng_state_sha256": _sha256(rank_runtime_state),
                    "collective_export_rank_receipt_sha256": None,
                }

            self._run_group_phase(
                "candidate restore after serving export",
                _restore_candidate,
            )
        if export_error is not None:
            raise export_error
        _require(
            isinstance(parent_result, Mapping)
            and isinstance(candidate_result, Mapping),
            "serving export did not produce both manifests",
        )
        record: dict[str, object] = {
            "schema_version": M0_DISTRIBUTED_SERVING_EXPORT_RANK_SCHEMA,
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "worker_rank": rank,
            "world_size": 4,
            "parent": {
                "dcp_path": str(parent_dcp),
                "dcp_manifest_sha256": parent_dcp_manifest_sha256,
                "export_path": str(parent_export),
                "export_manifest": deepcopy(dict(parent_result["manifest"])),
                "parameter_sha256": parent_result["parameter_sha256"],
            },
            "candidate": {
                "dcp_path": str(candidate_dcp),
                "dcp_manifest_sha256": candidate_dcp_manifest_sha256,
                "export_path": str(candidate_export),
                "export_manifest": deepcopy(dict(candidate_result["manifest"])),
                "parameter_sha256": candidate_result["parameter_sha256"],
            },
            "evidence_scope": {
                "collective_hf_save_executed": True,
                "candidate_dcp_restored": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        marker = pending.get("post_export_candidate_state")
        _require(
            isinstance(marker, dict)
            and marker.get("collective_export_rank_receipt_sha256") is None,
            "serving export candidate restore marker is missing",
        )
        marker["collective_export_rank_receipt_sha256"] = record["record_sha256"]
        return record

    def attest_m0_policy_candidate_current_state(
        self,
        *,
        transaction_id: str,
        policy_candidate_sha256: str,
        distributed_serving_export_receipt_sha256: str,
        collective_export_rank_receipt_sha256: str,
        candidate_serving_export_lineage_sha256: str,
        candidate_serving_parameter_sha256: str,
        candidate_dcp_manifest_sha256: str,
        actor_public_version: int,
        expected_optimizer_step: int,
        scheduler_state: Mapping[str, object],
        rank_runtime_state: Mapping[str, object],
    ) -> dict[str, object]:
        """Attest the live candidate immediately after collective HF export."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        common = {
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "distributed_serving_export_receipt_sha256": (
                distributed_serving_export_receipt_sha256
            ),
            "candidate_serving_export_lineage_sha256": (
                candidate_serving_export_lineage_sha256
            ),
            "candidate_serving_parameter_sha256": (
                candidate_serving_parameter_sha256
            ),
            "candidate_dcp_manifest_sha256": candidate_dcp_manifest_sha256,
            "actor_public_version": actor_public_version,
            "expected_optimizer_step": expected_optimizer_step,
            "scheduler_state_sha256": _sha256(scheduler_state),
        }
        self._require_group_common("X current Policy state request", common)

        def _attest() -> dict[str, object]:
            pending = self._jph_pending_m0_transaction
            _require(
                pending is not None
                and pending.get("transaction_id") == transaction_id
                and pending.get("policy_candidate_sha256")
                == policy_candidate_sha256,
                "X current state differs from the pending T candidate",
            )
            marker = pending.get("post_export_candidate_state")
            _require(
                isinstance(marker, Mapping)
                and marker
                == {
                    "candidate_dcp_manifest_sha256": (
                        candidate_dcp_manifest_sha256
                    ),
                    "candidate_serving_parameter_sha256": (
                        candidate_serving_parameter_sha256
                    ),
                    "optimizer_step": expected_optimizer_step,
                    "lr_scheduler_state_sha256": _sha256(scheduler_state),
                    "runtime_rng_state_sha256": _sha256(rank_runtime_state),
                    "collective_export_rank_receipt_sha256": (
                        collective_export_rank_receipt_sha256
                    ),
                },
                "X current state is not the completed collective export restore",
            )
            _require(
                checkpoint_manifest(pending["candidate_path"])["manifest_sha256"]
                == candidate_dcp_manifest_sha256
                == pending["candidate_manifest_sha256"],
                "X current state candidate DCP differs from T",
            )
            from .production_checkpoint import (
                _assert_rank_rng_restored,
                RankRuntimeState,
            )

            runtime_fields = set(RankRuntimeState.__dataclass_fields__)
            _require(
                set(rank_runtime_state) == {"schema_version"} | runtime_fields,
                "X current state rank runtime fields differ",
            )
            runtime = RankRuntimeState(
                **{
                    field: deepcopy(rank_runtime_state[field])
                    for field in runtime_fields
                }
            )
            _require(
                runtime.to_record() == rank_runtime_state and runtime.rank == rank,
                "X current state rank runtime record is invalid",
            )
            _assert_rank_rng_restored(rank_runtime_state, harness_policy=None)
            _require(
                world_size == 4
                and rank in range(4)
                and _optimizer_step(self) == expected_optimizer_step
                and _sha256(_lr_scheduler_state(self))
                == _sha256(scheduler_state)
                and self.get_version() == actor_public_version,
                "X current state optimizer/scheduler/version differs from T",
            )
            return {"physical_gpu_id": _physical_gpu_id()}

        observed = self._run_group_phase("X current Policy state", _attest)
        record: dict[str, object] = {
            "schema_version": M0_DISTRIBUTED_POLICY_CURRENT_STATE_RANK_SCHEMA,
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "distributed_serving_export_receipt_sha256": (
                distributed_serving_export_receipt_sha256
            ),
            "collective_export_rank_receipt_sha256": (
                collective_export_rank_receipt_sha256
            ),
            "candidate_serving_export_lineage_sha256": (
                candidate_serving_export_lineage_sha256
            ),
            "candidate_serving_parameter_sha256": (
                candidate_serving_parameter_sha256
            ),
            "worker_rank": rank,
            "world_size": world_size,
            "engine_name": f"actor/{rank}",
            "physical_gpu_id": observed["physical_gpu_id"],
            "candidate_dcp_manifest_sha256": candidate_dcp_manifest_sha256,
            "optimizer_step": expected_optimizer_step,
            "lr_scheduler_state_sha256": _sha256(scheduler_state),
            "runtime_rng_state_sha256": _sha256(rank_runtime_state),
            "actor_public_version": actor_public_version,
            "evidence_scope": {
                "candidate_dcp_restored_after_collective_export": True,
                "optimizer_state_attested": True,
                "lr_scheduler_state_attested": True,
                "rank_rng_state_restored_and_attested": True,
                "candidate_serving_parameter_digest_bound": True,
                "live_weight_parameter_digest_observed": False,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        return record

    def attest_m0_live_policy_candidate_for_w(
        self,
        *,
        transaction_id: str,
        policy_candidate_sha256: str,
        candidate_path: str,
        candidate_dcp_manifest_sha256: str,
        actor_public_version: int,
        expected_optimizer_step: int,
        scheduler_state: Mapping[str, object],
        rank_runtime_state: Mapping[str, object],
    ) -> dict[str, object]:
        """Collectively attest the still-live T state without loading its DCP."""

        return self._attest_or_restore_m0_policy_candidate_for_w(
            branch_id="uninterrupted",
            transaction_id=transaction_id,
            policy_candidate_sha256=policy_candidate_sha256,
            candidate_path=candidate_path,
            candidate_dcp_manifest_sha256=candidate_dcp_manifest_sha256,
            actor_public_version=actor_public_version,
            expected_optimizer_step=expected_optimizer_step,
            scheduler_state=scheduler_state,
            rank_runtime_state=rank_runtime_state,
            load_candidate=False,
        )

    def restore_m0_policy_candidate_for_w(
        self,
        *,
        branch_id: str,
        transaction_id: str,
        policy_candidate_sha256: str,
        candidate_path: str,
        candidate_dcp_manifest_sha256: str,
        actor_public_version: int,
        expected_optimizer_step: int,
        scheduler_state: Mapping[str, object],
        rank_runtime_state: Mapping[str, object],
    ) -> dict[str, object]:
        """Collectively restore candidate DCP+optimizer and rank-owned W state."""

        branch = _require_w_branch_id(branch_id)
        _require(
            branch in {"recovered", "final-restore"},
            "uninterrupted W must attest the live candidate without loading DCP",
        )
        return self._attest_or_restore_m0_policy_candidate_for_w(
            branch_id=branch,
            transaction_id=transaction_id,
            policy_candidate_sha256=policy_candidate_sha256,
            candidate_path=candidate_path,
            candidate_dcp_manifest_sha256=candidate_dcp_manifest_sha256,
            actor_public_version=actor_public_version,
            expected_optimizer_step=expected_optimizer_step,
            scheduler_state=scheduler_state,
            rank_runtime_state=rank_runtime_state,
            load_candidate=True,
        )

    def _attest_or_restore_m0_policy_candidate_for_w(
        self,
        *,
        branch_id: str,
        transaction_id: str,
        policy_candidate_sha256: str,
        candidate_path: str,
        candidate_dcp_manifest_sha256: str,
        actor_public_version: int,
        expected_optimizer_step: int,
        scheduler_state: Mapping[str, object],
        rank_runtime_state: Mapping[str, object],
        load_candidate: bool,
    ) -> dict[str, object]:
        """Implement the rank-local W attestation or collective restore path."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        _require(dist.is_initialized(), "actor distributed group is not initialized")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        branch = _require_w_branch_id(branch_id)
        _require(
            world_size == 4
            and rank in range(4)
            and load_candidate == (branch != "uninterrupted"),
            "distributed W state action differs from the requested branch",
        )

        def _state_action() -> dict[str, object]:
            _require(
                type(self) is JPHFSDPPPOActor,
                "distributed W restore requires the exact project actor subclass",
            )
            pending = self._jph_pending_m0_transaction
            _require(
                pending is not None
                and pending.get("transaction_id") == transaction_id
                and pending.get("policy_candidate_sha256")
                == policy_candidate_sha256
                and isinstance(pending.get("w_source"), Mapping),
                "distributed W restore differs from the live T transaction",
            )
            path = _validated_pending_w_candidate_path(
                pending=pending,
                candidate_path=candidate_path,
                candidate_dcp_manifest_sha256=(
                    candidate_dcp_manifest_sha256
                ),
            )
            rank_receipt = _validated_pending_m0_candidate_state(pending)
            from .production_checkpoint import (
                RankRuntimeState,
                _assert_rank_rng_restored,
                _restore_rank_rng,
            )

            runtime_fields = set(RankRuntimeState.__dataclass_fields__)
            _require(
                set(rank_runtime_state) == {"schema_version"} | runtime_fields,
                "distributed W rank runtime fields differ",
            )
            runtime = RankRuntimeState(
                **{
                    field: deepcopy(rank_runtime_state[field])
                    for field in runtime_fields
                }
            )
            _require(
                runtime.to_record() == rank_runtime_state
                and runtime.rank == rank
                and runtime.harness_generator_state is None,
                "distributed W rank runtime state is invalid",
            )
            _require(
                rank_runtime_state == pending["rank_runtime_state"],
                "distributed W rank runtime state differs from T",
            )
            _require(
                _sha256(scheduler_state)
                == rank_receipt[
                    "lr_scheduler_state_after_sha256"
                ],
                "distributed W scheduler state differs from T",
            )
            _require(
                expected_optimizer_step
                == rank_receipt["optimizer_step_after"]
                and actor_public_version
                == rank_receipt["inference_engine_version"],
                "distributed W optimizer/version identity differs from T",
            )
            if load_candidate:
                _require(
                    SaveLoadMeta is not None,
                    "AReaL SaveLoadMeta is unavailable",
                )
                self.load(
                    meta=SaveLoadMeta(
                        path=str(path),
                        weight_format="dcp",
                        with_optim=True,
                    )
                )
                _restore_lr_scheduler_state(self, scheduler_state)
                set_version = getattr(self, "set_version", None)
                _require(
                    callable(set_version),
                    "actor version restore API is unavailable",
                )
                set_version(actor_public_version)
                _restore_rank_rng(rank_runtime_state, harness_policy=None)
            _assert_rank_rng_restored(rank_runtime_state, harness_policy=None)
            _require(
                _optimizer_step(self) == expected_optimizer_step
                == rank_receipt["optimizer_step_after"]
                and _sha256(_lr_scheduler_state(self))
                == rank_receipt[
                    "lr_scheduler_state_after_sha256"
                ]
                and self.get_version() == actor_public_version,
                "distributed W restored optimizer/scheduler/version differs from T",
            )
            return {
                "physical_gpu_id": _physical_gpu_id(),
                "runtime_rng_state_sha256": _sha256(rank_runtime_state),
            }

        common = {
            "branch_id": branch,
            "load_candidate": load_candidate,
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "candidate_path": candidate_path,
            "candidate_dcp_manifest_sha256": candidate_dcp_manifest_sha256,
            "actor_public_version": actor_public_version,
            "expected_optimizer_step": expected_optimizer_step,
            "scheduler_state_sha256": _sha256(scheduler_state),
        }
        self._require_group_common("W state request", common)
        phase = "W candidate restore" if load_candidate else "W live candidate attest"
        observed = self._run_group_phase(phase, _state_action)
        record: dict[str, object] = {
            "schema_version": M0_DISTRIBUTED_POLICY_RESTORE_RANK_SCHEMA,
            "branch_id": branch,
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "worker_rank": rank,
            "world_size": world_size,
            "engine_name": f"actor/{rank}",
            "physical_gpu_id": observed["physical_gpu_id"],
            "candidate_dcp_manifest_sha256": candidate_dcp_manifest_sha256,
            "optimizer_step": expected_optimizer_step,
            "lr_scheduler_state_sha256": _sha256(scheduler_state),
            "runtime_rng_state_sha256": observed[
                "runtime_rng_state_sha256"
            ],
            "actor_public_version": actor_public_version,
            "evidence_scope": {
                "live_candidate_state_attested": not load_candidate,
                "candidate_dcp_loaded": load_candidate,
                "optimizer_state_loaded": load_candidate,
                "lr_scheduler_state_loaded": load_candidate,
                "rank_rng_state_loaded": load_candidate,
                "rank_scheduler_rng_state_attested": True,
                "continuation_executed": False,
                "exact_joint_recovery": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        self._jph_w_restore = {
            "branch_id": branch,
            "rank_receipt": deepcopy(record),
        }
        return record

    def run_m0_policy_recovery_continuation(
        self,
        *,
        branch_id: str,
        transaction_id: str,
        policy_candidate_sha256: str,
        restore_receipt: Mapping[str, object],
    ) -> dict[str, object]:
        """Run one diagnostic step from worker-private, T-bound sources only."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        _require(dist.is_initialized(), "actor distributed group is not initialized")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        branch = _require_w_branch_id(branch_id)
        _require(
            branch != "final-restore" and world_size == 4 and rank in range(4),
            "distributed W continuation requires an executable four-rank branch",
        )

        def _continue() -> dict[str, object]:
            pending = self._jph_pending_m0_transaction
            restored = getattr(self, "_jph_w_restore", None)
            _require(
                pending is not None
                and pending.get("transaction_id") == transaction_id
                and pending.get("policy_candidate_sha256")
                == policy_candidate_sha256
                and isinstance(pending.get("w_source"), Mapping)
                and isinstance(restored, Mapping)
                and restored.get("branch_id") == branch,
                "distributed W continuation has no matching collective restore",
            )
            _require(
                set(restore_receipt)
                == {
                    "schema_version",
                    "branch_id",
                    "transaction_id",
                    "policy_candidate_sha256",
                    "restore_state_sha256",
                    "rank_receipts",
                    "evidence_scope",
                    "record_sha256",
                }
                and restore_receipt.get("schema_version")
                == M0_DISTRIBUTED_POLICY_RESTORE_SCHEMA
                and restore_receipt.get("record_sha256")
                == _record_sha256(restore_receipt)
                and restore_receipt.get("branch_id") == branch
                and restore_receipt.get("transaction_id") == transaction_id
                and restore_receipt.get("policy_candidate_sha256")
                == policy_candidate_sha256
                and isinstance(restore_receipt.get("rank_receipts"), list)
                and len(restore_receipt["rank_receipts"]) == 4
                and restore_receipt["rank_receipts"][rank]
                == restored["rank_receipt"],
                "distributed W continuation restore aggregate is invalid",
            )
            source = pending["w_source"]
            version = _joint_version(source["active_joint_version"])
            admissions, flattened_samples = _validate_ordered_sources(
                source["admission_records"],
                source["source_joint_credit_records"],
                active_joint_version=version,
                source_binding=source["source_binding"],
            )
            indices = tuple(source["local_sample_indices"])
            _require(
                indices == _partition_indices(len(flattened_samples))[rank]
                == tuple(pending["rank_receipt"]["local_sample_indices"]),
                "distributed W source partition differs from T",
            )
            selected = set(indices)
            update_batch: list[dict[str, Any]] = []
            for member_index, (admission_record, _admission) in enumerate(
                zip(source["admission_records"], admissions)
            ):
                member_indices = [
                    sample_index
                    for global_index, (
                        flattened_member_index,
                        sample_index,
                        _sample,
                    ) in enumerate(flattened_samples)
                    if global_index in selected
                    and flattened_member_index == member_index
                ]
                if member_indices:
                    update_batch.extend(
                        materialize_areal_ppo_update_tensors(
                            admission_record,
                            actor=self,
                            active_joint_version=version,
                            device=self.device,
                            sample_indices=member_indices,
                        )
                    )
            _require(
                len(update_batch) == len(indices),
                "distributed W pre-batch source re-materialization lost a sample",
            )
            optimizer_step_before = _optimizer_step(self)
            scheduler_before = _lr_scheduler_state(self)
            _require(
                optimizer_step_before
                == pending["rank_receipt"]["optimizer_step_after"]
                and _sha256(scheduler_before)
                == pending["rank_receipt"][
                    "lr_scheduler_state_after_sha256"
                ],
                "distributed W continuation did not begin at the restored candidate",
            )
            self._jph_optimizer_step_count = 0
            self._jph_last_optimizer_result = None
            self._jph_capture_optimizer_step = True
            try:
                self.ppo_update(update_batch)
            finally:
                self._jph_capture_optimizer_step = False
            _require(
                self._jph_optimizer_step_count == 1,
                "distributed W continuation did not execute exactly one optimizer step",
            )
            # Invocation alone is not success: AReaL deliberately skips the
            # underlying AdamW step for a non-finite gradient norm.  Validate
            # its worker-local result before advancing the scheduler.
            _normalized_update_stats(self._jph_last_optimizer_result)
            optimizer_step_after = _optimizer_step(self)
            _require(
                optimizer_step_after == optimizer_step_before + 1,
                "distributed W continuation optimizer step differs",
            )
            self.step_lr_scheduler()
            scheduler_after = _lr_scheduler_state(self)
            _require_one_scheduler_step(scheduler_before, scheduler_after)
            root = Path(str(pending["candidate_path"])).parent
            continuation_path = self._collectively_require_new_checkpoint_path(
                name=f"distributed W {branch} continuation DCP",
                path=root / f"w-{branch}-continuation.dcp",
            )
            _require(SaveLoadMeta is not None, "AReaL SaveLoadMeta is unavailable")
            self.save(
                meta=SaveLoadMeta(
                    path=str(continuation_path),
                    weight_format="dcp",
                    with_optim=True,
                )
            )
            continuation_manifest = checkpoint_manifest(continuation_path)
            continuation_payload_sha256 = _checkpoint_dcp_payload_sha256(
                continuation_manifest
            )
            from .production_checkpoint import capture_rank_runtime_state

            runtime_after = capture_rank_runtime_state(
                rank=rank,
                local_rank=int(os.environ.get("LOCAL_RANK", "-1")),
                device=str(self.device),
            ).to_record()
            return {
                "physical_gpu_id": _physical_gpu_id(),
                "source_binding_sha256": source["source_binding"][
                    "record_sha256"
                ],
                "candidate_dcp_manifest_sha256": pending[
                    "candidate_manifest_sha256"
                ],
                "continuation_dcp_manifest_sha256": continuation_manifest[
                    "manifest_sha256"
                ],
                "continuation_dcp_payload_sha256": (
                    continuation_payload_sha256
                ),
                "local_sample_indices": list(indices),
                "optimizer_step_before": optimizer_step_before,
                "optimizer_step_after": optimizer_step_after,
                "lr_scheduler_state_before_sha256": _sha256(
                    scheduler_before
                ),
                "lr_scheduler_state_after_sha256": _sha256(scheduler_after),
                "runtime_rng_state_after_sha256": _sha256(runtime_after),
                "actor_public_version": self.get_version(),
            }

        common = {
            "branch_id": branch,
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "restore_receipt_sha256": restore_receipt.get("record_sha256"),
        }
        self._require_group_common("W continuation request", common)
        observed = self._run_group_phase("W diagnostic continuation", _continue)
        self._require_group_common(
            "W continuation common state",
            {
                key: value
                for key, value in observed.items()
                if key
                not in {
                    "physical_gpu_id",
                    "local_sample_indices",
                    "runtime_rng_state_after_sha256",
                }
            },
        )
        record: dict[str, object] = {
            "schema_version": M0_DISTRIBUTED_POLICY_CONTINUATION_RANK_SCHEMA,
            "branch_id": branch,
            "transaction_id": transaction_id,
            "policy_candidate_sha256": policy_candidate_sha256,
            "restore_receipt_sha256": restore_receipt["record_sha256"],
            "worker_rank": rank,
            "world_size": world_size,
            "engine_name": f"actor/{rank}",
            **observed,
            "evidence_scope": {
                "bound_pre_batch_sources_reused": True,
                "diagnostic_policy_optimizer_step_observed": True,
                "continuation_dcp_with_optimizer_observed": True,
                "exact_joint_recovery": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        self._jph_w_restore = None
        return record

    def commit_m0_policy_candidate(
        self,
        *,
        transaction_id: str,
        y_attestation_path: str,
        y_attestation_sha256: str,
        y_active_release_id: str,
    ) -> dict[str, object]:
        """Seal a Y-success terminal record, then drop rollback-only memory."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        rank = dist.get_rank()

        def _seal_commit() -> dict[str, object]:
            from jphrl.joint_release import JointReleaseStore
            from jphrl.training.joint_activation import (
                PRODUCTION_ATTESTATION_SCHEMA,
                ProductionWorkerState,
            )

            pending = self._jph_pending_m0_transaction
            _require(
                pending is not None
                and pending.get("transaction_id") == transaction_id
                and isinstance(pending.get("aggregate_sha256"), str)
                and isinstance(pending.get("policy_candidate_sha256"), str),
                "commit transaction differs from the aggregate-bound pending candidate",
            )
            path = require_within_configured_root(y_attestation_path)
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.parent.name == "activation-attestations",
                "Y attestation path is not a production activation attestation",
            )
            try:
                attestation = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArealDistributedPolicyError("Y attestation is unreadable") from exc
            _require(
                isinstance(attestation, Mapping)
                and attestation.get("schema_version") == PRODUCTION_ATTESTATION_SCHEMA
                and attestation.get("record_sha256") == _record_sha256(attestation)
                and attestation["record_sha256"] == y_attestation_sha256,
                "Y attestation schema or digest differs",
            )
            authorization = attestation.get("authorization")
            candidate = (
                authorization.get("candidate")
                if isinstance(authorization, Mapping)
                else None
            )
            _require(
                isinstance(candidate, Mapping)
                and candidate.get("release_id") == y_active_release_id,
                "Y attestation candidate differs from the active release",
            )
            final_states = attestation.get("final_worker_states")
            _require(
                isinstance(final_states, list) and bool(final_states),
                "Y attestation has no final production worker states",
            )
            for item in final_states:
                observed = ProductionWorkerState.from_record(item)
                _require(
                    observed.lifecycle_phase == "serving"
                    and observed.active_release_id == y_active_release_id,
                    "Y final worker is not serving the candidate release",
                )
            store = JointReleaseStore(path.parent.parent)
            active = store.read_active()
            _require(
                active is not None and active.release_id == y_active_release_id,
                "release store does not expose the Y candidate as active",
            )
            policy_artifact, _harness_artifact = store.read_artifacts(
                y_active_release_id
            )
            policy_payload = policy_artifact.payload
            _require(
                isinstance(policy_payload, Mapping)
                and policy_payload.get("policy_update_receipt_sha256")
                == pending["policy_candidate_sha256"],
                "Y Policy artifact does not bind the distributed Policy candidate",
            )
            state_before = {
                "actor_version": self.get_version(),
                "optimizer_step": _optimizer_step(self),
                "scheduler_state_sha256": _sha256(_lr_scheduler_state(self)),
            }
            commit_record: dict[str, object] = {
                "schema_version": "jph.m0-policy-worker-commit.v1",
                "transaction_id": transaction_id,
                "worker_rank": rank,
                "aggregate_sha256": pending["aggregate_sha256"],
                "policy_candidate_sha256": pending[
                    "policy_candidate_sha256"
                ],
                "rank_receipt_sha256": pending["rank_receipt"]["record_sha256"],
                "y_attestation_sha256": y_attestation_sha256,
                "y_active_release_id": y_active_release_id,
                "state_before": state_before,
                "state_after": deepcopy(state_before),
                "evidence_scope": {
                    "y_success_revalidated": True,
                    "training_state_changed": False,
                    "rollback_state_clear_authorized": True,
                    "policy_optimizer_update": False,
                    "harness_optimizer_update": False,
                },
            }
            commit_record["record_sha256"] = _record_sha256(commit_record)
            root = Path(str(pending["parent_path"])).parent
            _atomic_write_json(
                root / f"policy-rank-{rank}-commit.json",
                commit_record,
            )
            state_after = {
                "actor_version": self.get_version(),
                "optimizer_step": _optimizer_step(self),
                "scheduler_state_sha256": _sha256(_lr_scheduler_state(self)),
            }
            _require(
                state_after == state_before,
                "M0 commit changed worker weights, optimizer, or scheduler",
            )
            return commit_record

        commit_record = self._run_group_phase("Y terminal commit", _seal_commit)
        self._jph_pending_m0_transaction = None
        self._jph_w_restore = None
        terminal = self._gather_objects(
            {
                "transaction_id": transaction_id,
                "pending_cleared": self._jph_pending_m0_transaction is None,
            }
        )
        _require(
            all(
                item
                == {
                    "transaction_id": transaction_id,
                    "pending_cleared": True,
                }
                for item in terminal
            ),
            "actor ranks disagree on the terminal pending-state clear",
        )
        return commit_record

    def run_m0_policy_candidate_step(
        self,
        *,
        admission_records: Sequence[Mapping[str, object]],
        source_joint_credit_records: Sequence[Mapping[str, object]],
        source_binding: Mapping[str, object],
        active_joint_version: Mapping[str, object],
        transaction_id: str,
        candidate_root: str,
        local_sample_indices: Sequence[int],
    ) -> dict[str, object]:
        """Run one real Policy update and return one worker-created receipt."""

        try:
            import torch.distributed as dist
        except ModuleNotFoundError as exc:  # pragma: no cover - remote gate
            raise ArealDistributedPolicyError(
                "torch.distributed is unavailable on the actor worker"
            ) from exc
        _require(dist.is_initialized(), "actor distributed group is not initialized")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _require(
            world_size == 4 and rank in range(4),
            "M0 Policy worker must belong to one four-rank process group",
        )

        def _preflight() -> dict[str, object]:
            _require(
                type(self) is JPHFSDPPPOActor,
                "optimizer receipt requires the exact project AReaL actor subclass",
            )
            _require(
                f"{type(self).__module__}.{type(self).__qualname__}"
                == JPH_AREAL_DISTRIBUTED_ACTOR_CLASS,
                "worker actor class identity differs from the receipt contract",
            )
            _require(
                isinstance(transaction_id, str) and transaction_id,
                "distributed Policy transaction ID is missing",
            )
            _require_new_transaction(self._jph_pending_m0_transaction)
            validate_m0_areal_actor_config(self)
            _require(
                getattr(self.config, "backend", None) == "fsdp:d4",
                "worker actor backend must be fsdp:d4",
            )
            fsdp_config = getattr(self.config, "fsdp", None)
            _require(
                fsdp_config is not None
                and getattr(fsdp_config, "per_layer_optim_step", None) is False,
                "M0 receipts require one direct AdamW optimizer step",
            )
            optimizer = getattr(self, "optimizer", None)
            optimizer_class = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            _require(
                optimizer_class == PINNED_OPTIMIZER_CLASS,
                "worker optimizer is not the pinned torch AdamW implementation",
            )
            version = _joint_version(active_joint_version)
            admissions, flattened_samples = _validate_ordered_sources(
                admission_records,
                source_joint_credit_records,
                active_joint_version=version,
                source_binding=source_binding,
            )
            partitions = _partition_indices(len(flattened_samples))
            indices = tuple(local_sample_indices)
            _require(
                indices == partitions[rank],
                "worker sample indices differ from the deterministic rank partition",
            )
            physical_gpu_id = _physical_gpu_id()
            topology = M0EightGPUTopology()
            topology.validate()
            _require(
                physical_gpu_id == topology.actor_gpu_ids[rank],
                "worker physical GPU differs from the actor topology",
            )
            get_version = getattr(self, "get_version", None)
            _require(callable(get_version), "worker actor version API is unavailable")
            inference_engine_version = get_version()
            _require(
                inference_engine_version == admissions[0].inference_engine_version,
                "worker actor version differs from admitted Policy samples",
            )
            root = require_outside_repository(candidate_root)
            _require(root.is_dir() and not root.is_symlink(), "candidate root is invalid")
            parent_path = root / "policy-parent.dcp"
            candidate_path = root / "policy-candidate.dcp"
            rank_receipt_path = root / f"policy-rank-{rank}-receipt.json"
            _require(
                not parent_path.exists()
                and not candidate_path.exists()
                and not rank_receipt_path.exists()
                and not (root / "policy-remote-optimizer-receipt.json").exists(),
                "distributed Policy transaction paths must be new",
            )
            scheduler_state_before = _lr_scheduler_state(self)
            # This is diagnostic only.  On pinned PyTorch, the first DCP
            # get_state_dict() may lazily initialize an empty AdamW state with
            # a zero-LR step.  The formal update/rollback baseline therefore
            # has to be sampled after the parent DCP has been materialized.
            optimizer_step_pre_parent_dcp = _optimizer_step(self)
            selected = set(indices)
            update_batch: list[dict[str, Any]] = []
            for member_index, (admission_record, _admission) in enumerate(
                zip(admission_records, admissions)
            ):
                member_indices = [
                    sample_index
                    for global_index, (
                        flattened_member_index,
                        sample_index,
                        _sample,
                    ) in enumerate(flattened_samples)
                    if global_index in selected
                    and flattened_member_index == member_index
                ]
                if member_indices:
                    update_batch.extend(
                        materialize_areal_ppo_update_tensors(
                            admission_record,
                            actor=self,
                            active_joint_version=version,
                            device=self.device,
                            sample_indices=member_indices,
                        )
                    )
            _require(
                len(update_batch) == len(indices),
                "rank-local ordered Policy materialization lost a sample",
            )
            global_samples = [
                {
                    "member_index": member_index,
                    "sample_index": sample_index,
                    "sample_id": sample["sample_id"],
                    "tensor_dict": sample["tensor_dict"],
                }
                for member_index, sample_index, sample in flattened_samples
            ]
            return {
                "version": version,
                "admissions": admissions,
                "source_admission_sha256": source_binding["record_sha256"],
                "indices": indices,
                "physical_gpu_id": physical_gpu_id,
                "optimizer_class": optimizer_class,
                "inference_engine_version": inference_engine_version,
                "parent_path": parent_path,
                "candidate_path": candidate_path,
                "rank_receipt_path": rank_receipt_path,
                "scheduler_state_before": scheduler_state_before,
                "optimizer_step_pre_parent_dcp": optimizer_step_pre_parent_dcp,
                "update_batch": update_batch,
                "global_sample_count": len(flattened_samples),
                "global_sample_sha256": _sha256(global_samples),
            }

        state = self._run_group_phase("preflight", _preflight)
        preflight_common = {
            "transaction_id": transaction_id,
            "joint_version_id": state["version"].version_id,
            "source_admission_sha256": state["source_admission_sha256"],
            "optimizer_class": state["optimizer_class"],
            "inference_engine_version": state["inference_engine_version"],
            "optimizer_step_pre_parent_dcp": state[
                "optimizer_step_pre_parent_dcp"
            ],
            "scheduler_state_before_sha256": _sha256(
                state["scheduler_state_before"]
            ),
            "parent_path": str(state["parent_path"]),
            "candidate_path": str(state["candidate_path"]),
            "global_sample_count": state["global_sample_count"],
            "global_sample_sha256": state["global_sample_sha256"],
        }
        self._require_group_common("preflight identity", preflight_common)
        _require(SaveLoadMeta is not None, "AReaL SaveLoadMeta is unavailable")
        parent_meta = SaveLoadMeta(
            path=str(state["parent_path"]),
            weight_format="dcp",
            with_optim=True,
        )
        candidate_meta = SaveLoadMeta(
            path=str(state["candidate_path"]),
            weight_format="dcp",
            with_optim=True,
        )

        parent_manifest = self._run_group_phase(
            "parent DCP",
            lambda: (self.save(meta=parent_meta), checkpoint_manifest(state["parent_path"]))[1],
        )
        self._require_group_common("parent DCP manifest", parent_manifest)
        optimizer_step_before = self._run_group_phase(
            "post-parent optimizer baseline",
            lambda: _optimizer_step(self),
        )
        post_parent_common = {
            "optimizer_step_before": optimizer_step_before,
            "scheduler_state_before_sha256": _sha256(
                state["scheduler_state_before"]
            ),
            "parent_dcp_manifest_sha256": parent_manifest[
                "manifest_sha256"
            ],
        }
        self._require_group_common("post-parent baseline", post_parent_common)
        _require_post_parent_optimizer_baseline(
            pre_parent_dcp_step=state["optimizer_step_pre_parent_dcp"],
            post_parent_dcp_step=optimizer_step_before,
        )
        state["optimizer_step_before"] = optimizer_step_before
        self._jph_pending_m0_transaction = {
            "transaction_id": transaction_id,
            "parent_path": str(state["parent_path"]),
            "candidate_path": str(state["candidate_path"]),
            "optimizer_step_before": state["optimizer_step_before"],
            "scheduler_state_before": deepcopy(state["scheduler_state_before"]),
            # W re-materializes only the individual, pre-batch sources that T
            # already validated and partitioned.  A merged/post-batch payload
            # can never be supplied later by the controller as recovery input.
            "w_source": {
                "admission_records": [
                    deepcopy(dict(record)) for record in admission_records
                ],
                "source_joint_credit_records": [
                    deepcopy(dict(record))
                    for record in source_joint_credit_records
                ],
                "source_binding": deepcopy(dict(source_binding)),
                "active_joint_version": asdict(state["version"]),
                "local_sample_indices": list(state["indices"]),
            },
        }

        try:
            def _update() -> dict[str, object]:
                self._jph_optimizer_step_count = 0
                self._jph_last_optimizer_result = None
                self._jph_capture_optimizer_step = True
                try:
                    self.ppo_update(state["update_batch"])
                finally:
                    self._jph_capture_optimizer_step = False
                _require(
                    self._jph_optimizer_step_count == 1,
                    "worker did not execute exactly one optimizer_step",
                )
                update_stats = _normalized_update_stats(
                    self._jph_last_optimizer_result
                )
                optimizer_step_after = _optimizer_step(self)
                _require(
                    optimizer_step_after == state["optimizer_step_before"] + 1,
                    "worker optimizer step did not advance exactly once "
                    f"(parent baseline={state['optimizer_step_before']}, "
                    f"observed={optimizer_step_after})",
                )
                self.step_lr_scheduler()
                scheduler_state_after = _lr_scheduler_state(self)
                _require_one_scheduler_step(
                    state["scheduler_state_before"],
                    scheduler_state_after,
                )
                return {
                    "optimizer_step_after": optimizer_step_after,
                    "scheduler_state_after": scheduler_state_after,
                    "update_stats": update_stats,
                }

            update = self._run_group_phase("PPO optimizer update", _update)
            post_update_common = {
                "optimizer_step_after": update["optimizer_step_after"],
                "scheduler_state_after_sha256": _sha256(
                    update["scheduler_state_after"]
                ),
                "update_stats": update["update_stats"],
            }
            self._require_group_common("post-update evidence", post_update_common)

            candidate_manifest = self._run_group_phase(
                "candidate DCP",
                lambda: (
                    self.save(meta=candidate_meta),
                    checkpoint_manifest(state["candidate_path"]),
                )[1],
            )
            self._require_group_common("candidate DCP manifest", candidate_manifest)
            _require(
                parent_manifest["manifest_sha256"]
                != candidate_manifest["manifest_sha256"],
                "parent and candidate distributed Policy DCPs are identical",
            )

            def _capture_runtime() -> dict[str, object]:
                from .production_checkpoint import capture_rank_runtime_state

                local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
                runtime = capture_rank_runtime_state(
                    rank=rank,
                    local_rank=local_rank,
                    device=str(self.device),
                ).to_record()
                torch_rng = runtime["torch_rng"]
                _require(
                    isinstance(torch_rng, Mapping)
                    and torch_rng.get("available") is True
                    and isinstance(torch_rng.get("cuda_states"), list)
                    and len(torch_rng["cuda_states"]) == 1,
                    "actor worker CUDA RNG state is unavailable",
                )
                compact = {
                    "hostname": runtime["hostname"],
                    "local_rank": runtime["local_rank"],
                    "device": runtime["device"],
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES", ""
                    ),
                    "python_rng_state_sha256": _sha256(
                        runtime["python_random_state"]
                    ),
                    "torch_cpu_rng_state_sha256": _sha256(
                        torch_rng["cpu_state"]
                    ),
                    "torch_cuda_rng_state_sha256": _sha256(
                        torch_rng["cuda_states"]
                    ),
                }
                return {"full": runtime, "compact": compact}

            runtime_state = self._run_group_phase(
                "rank runtime capture", _capture_runtime
            )

            def _write_rank_receipt() -> dict[str, object]:
                record: dict[str, object] = {
                    "schema_version": M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA,
                    "transaction_id": transaction_id,
                    "joint_version_id": state["version"].version_id,
                    "source_admission_sha256": state["source_admission_sha256"],
                    "worker_rank": rank,
                    "world_size": world_size,
                    "engine_name": f"actor/{rank}",
                    "physical_gpu_id": state["physical_gpu_id"],
                    "engine_class": JPH_AREAL_DISTRIBUTED_ACTOR_CLASS,
                    "optimizer_class": state["optimizer_class"],
                    "inference_engine_version": state["inference_engine_version"],
                    "global_sample_count": state["global_sample_count"],
                    "global_sample_sha256": state["global_sample_sha256"],
                    "local_sample_indices": list(state["indices"]),
                    "optimizer_step_before": state["optimizer_step_before"],
                    "optimizer_step_after": update["optimizer_step_after"],
                    "lr_scheduler_state_before_sha256": _sha256(
                        state["scheduler_state_before"]
                    ),
                    "lr_scheduler_state_after_sha256": _sha256(
                        update["scheduler_state_after"]
                    ),
                    "lr_scheduler_state_after": (
                        deepcopy(update["scheduler_state_after"])
                        if rank == 0
                        else None
                    ),
                    "parent_dcp_manifest_sha256": parent_manifest[
                        "manifest_sha256"
                    ],
                    "candidate_dcp_manifest_sha256": candidate_manifest[
                        "manifest_sha256"
                    ],
                    "runtime_state": runtime_state["compact"],
                    "update_stats": update["update_stats"],
                    "evidence_scope": dict(_RANK_EVIDENCE_SCOPE),
                }
                record["record_sha256"] = _record_sha256(record)
                _atomic_write_json(state["rank_receipt_path"], record)
                return record

            receipt = self._run_group_phase(
                "rank receipt persistence", _write_rank_receipt
            )
            self._jph_pending_m0_transaction.update(
                {
                    "rank_receipt": deepcopy(receipt),
                    "rank_runtime_state": deepcopy(runtime_state["full"]),
                    "scheduler_state_after": deepcopy(
                        update["scheduler_state_after"]
                    ),
                    "optimizer_step_after": update["optimizer_step_after"],
                    "candidate_manifest_sha256": candidate_manifest[
                        "manifest_sha256"
                    ],
                }
            )
            return receipt
        except BaseException as update_error:
            try:
                self._rollback_pending_group(transaction_id)
            except BaseException as rollback_error:
                raise ArealDistributedPolicyError(
                    "distributed Policy update failed and four-rank rollback also failed"
                ) from rollback_error
            raise update_error


class JPHPPOActorController(_ArealPPOActorController):  # type: ignore[misc,valid-type]
    """Aggregate all four worker receipts without observing optimizer state."""

    async def _call_rank_specific(
        self,
        method: str,
        kwargs_by_rank: Sequence[Mapping[str, object]],
    ) -> list[object]:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method=method,
                engine_name=self._engine_name(rank),
                rpc_meta={"broadcast": False},
                max_retries=(
                    1
                    if method
                    in {
                        "run_m0_policy_candidate_step",
                        "bind_m0_policy_aggregate",
                        "rollback_m0_policy_candidate",
                        "commit_m0_policy_candidate",
                        "attest_m0_live_policy_candidate_for_w",
                        "restore_m0_policy_candidate_for_w",
                        "run_m0_policy_recovery_continuation",
                        "materialize_m0_serving_export_pair",
                        "attest_m0_policy_candidate_current_state",
                    }
                    else 3
                ),
                **dict(kwargs_by_rank[rank]),
            )
            for rank, worker in enumerate(self.workers)
        ]
        return list(await asyncio.gather(*tasks, return_exceptions=True))

    def _dispatch_rank_specific(
        self,
        method: str,
        kwargs_by_rank: Sequence[Mapping[str, object]],
    ) -> list[object]:
        _require(_run_async_task is not None, "AReaL async RPC runner is unavailable")
        return _run_async_task(self._call_rank_specific, method, kwargs_by_rank)

    def _require_controller_topology(self) -> None:
        _require(
            type(self) is JPHPPOActorController,
            "optimizer aggregate requires the exact project controller subclass",
        )
        assert_controller_has_no_local_optimizer(self)
        _require(
            getattr(self.config, "backend", None) == "fsdp:d4"
            and getattr(self, "_worker_role", None) == "actor"
            and len(self.workers) == 4
            and self.workers_is_dp_head == [True, True, True, True]
            and [worker.id for worker in self.workers]
            == [f"actor/{rank}" for rank in range(4)],
            "controller workers differ from the fsdp:d4 actor topology",
        )

    def _rollback_workers(self, transaction_id: str) -> None:
        results = self._dispatch_rank_specific(
            "rollback_m0_policy_candidate",
            [{"transaction_id": transaction_id} for _ in range(4)],
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        _require(
            not errors,
            "one or more actor workers failed distributed parent rollback",
        )

    def _bind_workers(
        self,
        *,
        transaction_id: str,
        aggregate_path: Path,
        aggregate_sha256: str,
        policy_candidate_path: Path,
        policy_candidate_sha256: str,
    ) -> None:
        results = self._dispatch_rank_specific(
            "bind_m0_policy_aggregate",
            [
                {
                    "transaction_id": transaction_id,
                    "aggregate_path": str(aggregate_path),
                    "aggregate_sha256": aggregate_sha256,
                    "policy_candidate_path": str(policy_candidate_path),
                    "policy_candidate_sha256": policy_candidate_sha256,
                }
                for _ in range(4)
            ],
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        _require(not errors, "one or more workers rejected the persisted aggregate")

    def run_m0_policy_candidate_step(
        self,
        admission_records: Sequence[Mapping[str, object]],
        *,
        multi_s_batch: ValidatedMultiSFrozenTrainingBatch,
        active_joint_version: JointVersion,
        transaction_id: str,
        candidate_root: str | Path,
        project_commit: str,
        areal_commit: str,
    ) -> dict[str, object]:
        """Run one four-rank step, validate every raw receipt, and persist it."""

        self._require_controller_topology()
        _require(
            isinstance(transaction_id, str) and transaction_id,
            "distributed Policy transaction ID is missing",
        )
        _require(
            type(multi_s_batch) is ValidatedMultiSFrozenTrainingBatch
            and multi_s_batch.joint_version == active_joint_version,
            "distributed Policy T requires the validated active multi-S batch",
        )
        source_binding = multi_s_source_binding(multi_s_batch)
        source_joint_credit_records = tuple(
            member.s_record for member in multi_s_batch.members
        )
        admissions, flattened_samples = _validate_ordered_sources(
            admission_records,
            source_joint_credit_records,
            active_joint_version=active_joint_version,
            source_binding=source_binding,
        )
        partitions = _partition_indices(len(flattened_samples))
        root = require_outside_repository(candidate_root)
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(root, 0o700)
        _require(root.is_dir() and not root.is_symlink(), "candidate root is invalid")
        aggregate_path = root / "policy-remote-optimizer-receipt.json"
        policy_candidate_path = root / "policy-distributed-candidate.json"
        common = {
            "admission_records": [dict(record) for record in admission_records],
            "source_joint_credit_records": [
                dict(record) for record in source_joint_credit_records
            ],
            "source_binding": dict(source_binding),
            "active_joint_version": asdict(active_joint_version),
            "transaction_id": transaction_id,
            "candidate_root": str(root),
        }
        kwargs_by_rank = [
            {**common, "local_sample_indices": list(partitions[rank])}
            for rank in range(4)
        ]
        results = self._dispatch_rank_specific(
            "run_m0_policy_candidate_step",
            kwargs_by_rank,
        )
        worker_errors = [
            f"rank {rank}: {_exception_summary(result)}"
            for rank, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        if worker_errors:
            try:
                self._rollback_workers(transaction_id)
            except BaseException as rollback_error:
                raise ArealDistributedPolicyError(
                    "worker RPC failed and distributed parent rollback also failed"
                ) from rollback_error
            raise ArealDistributedPolicyError(
                "one or more Policy worker RPCs failed: " + "; ".join(worker_errors)
            )
        try:
            _require(
                all(isinstance(result, Mapping) for result in results),
                "Policy worker RPC returned a non-receipt value",
            )
            aggregate = build_remote_optimizer_aggregate(
                [result for result in results if isinstance(result, Mapping)]
            )
            _require(
                aggregate["transaction_id"] == transaction_id
                and aggregate["joint_version_id"] == active_joint_version.version_id
                and aggregate["source_admission_sha256"]
                == source_binding["record_sha256"],
                "remote optimizer aggregate differs from the requested transaction",
            )
            _atomic_write_json(aggregate_path, aggregate)
            candidate_record = build_distributed_policy_candidate(
                transaction_id=transaction_id,
                admissions=admissions,
                source_joint_credit_records=source_joint_credit_records,
                source_binding=source_binding,
                flattened_samples=flattened_samples,
                active_joint_version=active_joint_version,
                remote_optimizer_receipt=aggregate,
                candidate_root=root,
                project_commit=project_commit,
                areal_commit=areal_commit,
            )
            _atomic_write_json(policy_candidate_path, candidate_record)
            self._bind_workers(
                transaction_id=transaction_id,
                aggregate_path=aggregate_path,
                aggregate_sha256=str(aggregate["record_sha256"]),
                policy_candidate_path=policy_candidate_path,
                policy_candidate_sha256=str(candidate_record["record_sha256"]),
            )
            return candidate_record
        except BaseException as aggregate_error:
            try:
                self._rollback_workers(transaction_id)
            except BaseException as rollback_error:
                raise ArealDistributedPolicyError(
                    "receipt aggregation failed and distributed parent rollback also failed"
                ) from rollback_error
            raise aggregate_error

    def export_m0_policy_candidate_live_state(
        self,
        receipt: Mapping[str, object],
    ) -> LiveArealDistributedPolicyCandidate:
        """Collect worker-owned W inputs after the aggregate is durably bound."""

        self._require_controller_topology()
        validate_distributed_policy_candidate(receipt)
        remote = receipt["optimizer"]["remote_optimizer_receipt"]
        audit = validate_remote_optimizer_receipt(remote)
        results = self._dispatch_rank_specific(
            "export_m0_policy_candidate_live_state",
            [{"transaction_id": audit.transaction_id} for _ in range(4)],
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        _require(not errors, "one or more workers failed to export live W state")
        _require(
            all(isinstance(result, Mapping) for result in results),
            "worker live state RPC returned a non-record value",
        )
        candidate = LiveArealDistributedPolicyCandidate._create(
            receipt=receipt,
            worker_states=[
                result for result in results if isinstance(result, Mapping)
            ],
            token=_LIVE_CANDIDATE_TOKEN,
        )
        return require_live_remote_policy_candidate(candidate)

    def materialize_m0_serving_export_pair(
        self,
        candidate: LiveArealDistributedPolicyCandidate,
        *,
        export_root: str | Path,
    ) -> dict[str, object]:
        """Dispatch one collective HF save pair and retain all rank receipts."""

        self._require_controller_topology()
        live = require_live_remote_policy_candidate(candidate)
        audit = validate_distributed_policy_candidate(
            live.receipt,
            require_checkpoints=True,
        )
        root = require_outside_repository(export_root)
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(root, 0o700)
        checkpoints = live.receipt["checkpoints"]
        common = {
            "transaction_id": audit.transaction_id,
            "policy_candidate_sha256": audit.record_sha256,
            "parent_dcp_path": checkpoints["parent_path"],
            "parent_dcp_manifest_sha256": checkpoints["parent_manifest"][
                "manifest_sha256"
            ],
            "candidate_dcp_path": checkpoints["candidate_path"],
            "candidate_dcp_manifest_sha256": checkpoints["candidate_manifest"][
                "manifest_sha256"
            ],
            "parent_export_path": str(root / "policy-parent-hf"),
            "candidate_export_path": str(root / "policy-candidate-hf"),
        }
        results = self._dispatch_rank_specific(
            "materialize_m0_serving_export_pair",
            [deepcopy(common) for _ in range(4)],
        )
        errors = [
            f"rank {rank}: {_exception_summary(result)}"
            for rank, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        _require(
            not errors,
            "one or more workers failed collective serving export: "
            + "; ".join(errors),
        )
        _require(
            len(results) == 4
            and all(isinstance(result, Mapping) for result in results),
            "collective serving export did not return four rank receipts",
        )
        aggregate = build_distributed_serving_export_receipt(
            [result for result in results if isinstance(result, Mapping)],
            candidate=live,
        )
        _atomic_write_json(root / "distributed-serving-export.json", aggregate)
        return aggregate

    def attest_m0_policy_candidate_current_state(
        self,
        candidate: LiveArealDistributedPolicyCandidate,
        *,
        distributed_serving_export_receipt: Mapping[str, object],
        candidate_serving_export_lineage_sha256: str,
    ) -> dict[str, object]:
        """Aggregate four rank states immediately before X ``compute_logp``."""

        self._require_controller_topology()
        live, audit, _remote, optimizer_rank_receipts = _w_candidate_context(
            candidate
        )
        export = validate_distributed_serving_export_receipt(
            distributed_serving_export_receipt,
            candidate=live,
        )
        _require(
            _is_sha256(candidate_serving_export_lineage_sha256),
            "candidate serving-export lineage digest is invalid",
        )
        scheduler_state = live.worker_states[0][
            "rank0_lr_scheduler_state_after"
        ]
        _require(
            isinstance(scheduler_state, Mapping),
            "X current state rank-zero scheduler state is missing",
        )
        candidate_manifest_sha256 = live.receipt["checkpoints"][
            "candidate_manifest"
        ]["manifest_sha256"]
        candidate_serving_parameter_sha256 = export["candidate"][
            "parameter_sha256"
        ]
        kwargs_by_rank = [
            {
                "transaction_id": audit.transaction_id,
                "policy_candidate_sha256": audit.record_sha256,
                "distributed_serving_export_receipt_sha256": export[
                    "record_sha256"
                ],
                "collective_export_rank_receipt_sha256": export[
                    "rank_receipts"
                ][rank]["record_sha256"],
                "candidate_serving_export_lineage_sha256": (
                    candidate_serving_export_lineage_sha256
                ),
                "candidate_serving_parameter_sha256": (
                    candidate_serving_parameter_sha256
                ),
                "candidate_dcp_manifest_sha256": candidate_manifest_sha256,
                "actor_public_version": audit.parent_engine_version,
                "expected_optimizer_step": optimizer_rank_receipts[rank][
                    "optimizer_step_after"
                ],
                "scheduler_state": deepcopy(dict(scheduler_state)),
                "rank_runtime_state": deepcopy(
                    dict(live.worker_states[rank]["rank_runtime_state"])
                ),
            }
            for rank in range(4)
        ]
        results = self._dispatch_rank_specific(
            "attest_m0_policy_candidate_current_state",
            kwargs_by_rank,
        )
        errors = [
            f"rank {rank}: {_exception_summary(result)}"
            for rank, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        _require(
            not errors,
            "one or more workers failed X current-state attestation: "
            + "; ".join(errors),
        )
        _require(
            len(results) == 4 and all(isinstance(item, Mapping) for item in results),
            "X current-state RPC did not return four rank receipts",
        )
        return build_distributed_policy_current_state_receipt(
            [item for item in results if isinstance(item, Mapping)],
            candidate=live,
            distributed_serving_export_receipt=export,
            candidate_serving_export_lineage_sha256=(
                candidate_serving_export_lineage_sha256
            ),
        )

    def run_m0_policy_candidate_step_live(
        self,
        admission_records: Sequence[Mapping[str, object]],
        *,
        multi_s_batch: ValidatedMultiSFrozenTrainingBatch,
        active_joint_version: JointVersion,
        transaction_id: str,
        candidate_root: str | Path,
        project_commit: str,
        areal_commit: str,
    ) -> LiveArealDistributedPolicyCandidate:
        receipt = self.run_m0_policy_candidate_step(
            admission_records,
            multi_s_batch=multi_s_batch,
            active_joint_version=active_joint_version,
            transaction_id=transaction_id,
            candidate_root=candidate_root,
            project_commit=project_commit,
            areal_commit=areal_commit,
        )
        return self.export_m0_policy_candidate_live_state(receipt)

    def attest_m0_live_policy_candidate_for_w(
        self,
        candidate: LiveArealDistributedPolicyCandidate,
    ) -> dict[str, object]:
        """Attest live post-T state on all ranks without invoking a DCP load."""

        return self._candidate_state_receipt_for_w(
            candidate,
            branch_id="uninterrupted",
            rpc_method="attest_m0_live_policy_candidate_for_w",
        )

    def restore_m0_policy_candidate_for_w(
        self,
        candidate: LiveArealDistributedPolicyCandidate,
        *,
        branch_id: str,
    ) -> dict[str, object]:
        """Collectively load the candidate DCP+optimizer and every rank's state."""

        branch = _require_w_branch_id(branch_id)
        _require(
            branch in {"recovered", "final-restore"},
            "uninterrupted W must attest the live candidate without loading DCP",
        )
        return self._candidate_state_receipt_for_w(
            candidate,
            branch_id=branch,
            rpc_method="restore_m0_policy_candidate_for_w",
        )

    def _candidate_state_receipt_for_w(
        self,
        candidate: LiveArealDistributedPolicyCandidate,
        *,
        branch_id: str,
        rpc_method: str,
    ) -> dict[str, object]:
        """Dispatch exactly one live-attest or checkpoint-restore rank action."""

        self._require_controller_topology()
        branch = _require_w_branch_id(branch_id)
        _require(
            (branch == "uninterrupted")
            == (rpc_method == "attest_m0_live_policy_candidate_for_w")
            and rpc_method
            in {
                "attest_m0_live_policy_candidate_for_w",
                "restore_m0_policy_candidate_for_w",
            },
            "distributed W RPC action differs from branch semantics",
        )
        live, audit, _remote, rank_receipts = _w_candidate_context(candidate)
        scheduler_state = live.worker_states[0][
            "rank0_lr_scheduler_state_after"
        ]
        _require(
            isinstance(scheduler_state, Mapping),
            "distributed W rank-zero scheduler state is missing",
        )
        checkpoints = live.receipt["checkpoints"]
        candidate_manifest_sha256 = checkpoints["candidate_manifest"][
            "manifest_sha256"
        ]
        kwargs_by_rank = [
            {
                "transaction_id": audit.transaction_id,
                "policy_candidate_sha256": audit.record_sha256,
                "candidate_path": checkpoints["candidate_path"],
                "candidate_dcp_manifest_sha256": candidate_manifest_sha256,
                "actor_public_version": audit.parent_engine_version,
                "expected_optimizer_step": rank_receipts[rank][
                    "optimizer_step_after"
                ],
                "scheduler_state": deepcopy(dict(scheduler_state)),
                "rank_runtime_state": deepcopy(
                    dict(live.worker_states[rank]["rank_runtime_state"])
                ),
            }
            for rank in range(4)
        ]
        if rpc_method == "restore_m0_policy_candidate_for_w":
            for kwargs in kwargs_by_rank:
                kwargs["branch_id"] = branch
        results = self._dispatch_rank_specific(
            rpc_method,
            kwargs_by_rank,
        )
        errors = [
            f"rank {rank}: {_exception_summary(result)}"
            for rank, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        _require(
            not errors,
            "one or more workers failed distributed W state action: "
            + "; ".join(errors),
        )
        _require(
            len(results) == 4 and all(isinstance(item, Mapping) for item in results),
            "distributed W state RPC did not return four rank receipts",
        )
        return build_distributed_policy_restore_receipt(
            [item for item in results if isinstance(item, Mapping)],
            candidate=live,
            branch_id=branch,
        )

    def run_m0_policy_recovery_continuation(
        self,
        candidate: LiveArealDistributedPolicyCandidate,
        *,
        restore_receipt: Mapping[str, object],
        branch_id: str,
    ) -> dict[str, object]:
        """Run W from T-bound sources; no caller-supplied batch is accepted."""

        self._require_controller_topology()
        branch = _require_w_branch_id(branch_id)
        _require(
            branch != "final-restore",
            "final distributed W restore cannot execute a continuation",
        )
        live, audit, _remote, _rank_receipts = _w_candidate_context(candidate)
        validate_distributed_policy_restore_receipt(
            restore_receipt,
            candidate=live,
            branch_id=branch,
        )
        kwargs = {
            "branch_id": branch,
            "transaction_id": audit.transaction_id,
            "policy_candidate_sha256": audit.record_sha256,
            "restore_receipt": deepcopy(dict(restore_receipt)),
        }
        results = self._dispatch_rank_specific(
            "run_m0_policy_recovery_continuation",
            [deepcopy(kwargs) for _ in range(4)],
        )
        errors = [
            f"rank {rank}: {_exception_summary(result)}"
            for rank, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        _require(
            not errors,
            "one or more workers failed distributed W continuation: "
            + "; ".join(errors),
        )
        _require(
            len(results) == 4 and all(isinstance(item, Mapping) for item in results),
            "distributed W continuation RPC did not return four rank receipts",
        )
        return build_distributed_policy_continuation_receipt(
            [item for item in results if isinstance(item, Mapping)],
            candidate=live,
            restore_receipt=restore_receipt,
            branch_id=branch,
        )

    def commit_m0_policy_candidate(
        self,
        candidate: LiveArealDistributedPolicyCandidate,
        *,
        production_activation: object,
    ) -> tuple[Mapping[str, object], ...]:
        """Clear pending rollback state only after a real completed Y result."""

        self._require_controller_topology()
        live = require_live_remote_policy_candidate(candidate)
        candidate_audit = validate_distributed_policy_candidate(live.receipt)
        from jphrl.training.joint_activation import ProductionJointActivationResult

        _require(
            type(production_activation) is ProductionJointActivationResult
            and production_activation.outcome == "candidate_active"
            and production_activation.active_release_id
            == production_activation.candidate_release_id
            and production_activation.attestation_path.is_file()
            and isinstance(production_activation.attestation_sha256, str)
            and len(production_activation.attestation_sha256) == 64,
            "M0 commit requires a completed native Y activation result",
        )
        results = self._dispatch_rank_specific(
            "commit_m0_policy_candidate",
            [
                {
                    "transaction_id": candidate_audit.transaction_id,
                    "y_attestation_path": str(
                        production_activation.attestation_path
                    ),
                    "y_attestation_sha256": (
                        production_activation.attestation_sha256
                    ),
                    "y_active_release_id": (
                        production_activation.active_release_id
                    ),
                }
                for _ in range(4)
            ],
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        _require(not errors, "one or more workers failed the Y terminal commit")
        _require(
            all(isinstance(result, Mapping) for result in results),
            "worker Y commit RPC returned a non-record value",
        )
        return tuple(result for result in results if isinstance(result, Mapping))


__all__ = [
    "AREAL_DISTRIBUTED_POLICY_CANDIDATE_SCHEMA",
    "ArealDistributedPolicyError",
    "JPH_AREAL_DISTRIBUTED_ACTOR_CLASS",
    "JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS",
    "M0_DISTRIBUTED_POLICY_CONTINUATION_RANK_SCHEMA",
    "M0_DISTRIBUTED_POLICY_CONTINUATION_SCHEMA",
    "M0_DISTRIBUTED_POLICY_CURRENT_STATE_RANK_SCHEMA",
    "M0_DISTRIBUTED_POLICY_CURRENT_STATE_SCHEMA",
    "M0_DISTRIBUTED_POLICY_RESTORE_RANK_SCHEMA",
    "M0_DISTRIBUTED_POLICY_RESTORE_SCHEMA",
    "M0_DISTRIBUTED_SERVING_EXPORT_RANK_SCHEMA",
    "M0_DISTRIBUTED_SERVING_EXPORT_SCHEMA",
    "JPHFSDPPPOActor",
    "JPHPPOActorController",
    "LiveArealDistributedPolicyCandidate",
    "ValidatedDistributedPolicyCandidate",
    "build_distributed_policy_continuation_receipt",
    "build_distributed_policy_current_state_receipt",
    "build_distributed_policy_restore_receipt",
    "build_distributed_serving_export_receipt",
    "build_distributed_policy_candidate",
    "build_remote_optimizer_aggregate",
    "require_live_remote_policy_candidate",
    "validate_distributed_policy_continuation_receipt",
    "validate_distributed_policy_current_state_receipt",
    "validate_distributed_policy_candidate",
    "validate_distributed_policy_restore_receipt",
    "validate_distributed_serving_export_receipt",
]
