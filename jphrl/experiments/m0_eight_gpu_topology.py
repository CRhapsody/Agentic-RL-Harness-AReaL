from __future__ import annotations

"""Fail-closed contracts for one distributed eight-GPU M0 transaction.

This module deliberately owns no optimizer implementation.  The controller may
schedule and dispatch work, but only rank-local worker receipts can attest to a
remote optimizer transition.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


M0_EIGHT_GPU_TOPOLOGY_SCHEMA = "jph.m0-eight-gpu-topology.v1"
M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA = (
    "jph.m0-areal-remote-optimizer-rank-receipt.v1"
)
M0_REMOTE_OPTIMIZER_RECEIPT_SCHEMA = "jph.m0-areal-remote-optimizer-receipt.v1"
PINNED_AREAL_ACTOR_CLASS = (
    "jphrl.training.areal_distributed_policy.JPHFSDPPPOActor"
)
PINNED_OPTIMIZER_CLASS = "torch.optim.adamw.AdamW"
PINNED_CONTROLLER_CLASS = (
    "jphrl.training.areal_distributed_policy.JPHPPOActorController"
)


class M0EightGPUTopologyError(ValueError):
    """Raised when an eight-GPU M0 contract is incomplete or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0EightGPUTopologyError(message)


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
        raise M0EightGPUTopologyError("record is not finite canonical JSON") from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


@dataclass(frozen=True)
class M0EightGPUTopology:
    scheduler_gpu_ids: tuple[int, ...] = tuple(range(8))
    actor_gpu_ids: tuple[int, ...] = (0, 1, 2, 3)
    rollout_gpu_ids: tuple[int, ...] = (4, 5, 6, 7)
    actor_backend: str = "fsdp:d4"
    rollout_backend: str = "sglang:d4"
    rollout_max_concurrent: int = 8
    rollout_consumer_batch_size: int = 4
    export_style: str = "individual"
    minimum_training_sample_count: int = 4
    memory_policy: str = "observation-only"
    prelaunch_memory_snapshot_count: int = 2
    runtime_memory_sample_interval_seconds: int = 1

    def validate(self) -> None:
        _require(
            self.scheduler_gpu_ids == tuple(range(8)),
            "M0 scheduler must own physical GPUs 0 through 7 exactly once",
        )
        _require(
            self.actor_gpu_ids == (0, 1, 2, 3)
            and self.rollout_gpu_ids == (4, 5, 6, 7)
            and set(self.actor_gpu_ids).isdisjoint(self.rollout_gpu_ids)
            and tuple(sorted(self.actor_gpu_ids + self.rollout_gpu_ids))
            == self.scheduler_gpu_ids,
            "M0 actor/rollout GPU partitions must be disjoint and exhaustive",
        )
        _require(
            self.actor_backend == "fsdp:d4"
            and self.rollout_backend == "sglang:d4"
            and self.rollout_max_concurrent == 8
            and self.rollout_consumer_batch_size == 4,
            "M0 distributed backends must remain fsdp:d4 and sglang:d4",
        )
        _require(
            self.export_style == "individual"
            and type(self.minimum_training_sample_count) is int
            and self.minimum_training_sample_count >= 4,
            "M0 requires at least four individual training samples",
        )
        _require(
            self.memory_policy == "observation-only"
            and self.prelaunch_memory_snapshot_count == 2
            and self.runtime_memory_sample_interval_seconds == 1,
            "M0 memory policy requires two prelaunch snapshots and 1s observation",
        )

    def record(self) -> dict[str, object]:
        self.validate()
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_TOPOLOGY_SCHEMA,
            "scheduler_gpu_ids": list(self.scheduler_gpu_ids),
            "actor": {
                "backend": self.actor_backend,
                "gpu_ids": list(self.actor_gpu_ids),
                "worker_count": len(self.actor_gpu_ids),
            },
            "rollout": {
                "backend": self.rollout_backend,
                "gpu_ids": list(self.rollout_gpu_ids),
                "worker_count": len(self.rollout_gpu_ids),
                "max_concurrent_rollouts": self.rollout_max_concurrent,
                "consumer_batch_size": self.rollout_consumer_batch_size,
            },
            "training_batch": {
                "export_style": self.export_style,
                "minimum_sample_count": self.minimum_training_sample_count,
            },
            "memory": {
                "policy": self.memory_policy,
                "prelaunch_snapshot_count": self.prelaunch_memory_snapshot_count,
                "runtime_sample_interval_seconds": (
                    self.runtime_memory_sample_interval_seconds
                ),
                "fixed_limit_mib": None,
                "other_user_process_action": "observe-only",
            },
            "ownership": {
                "single_local_scheduler": True,
                "actor_alive_through_y": True,
                "controller_reads_remote_optimizer": False,
            },
            "evidence_scope": {
                "topology_validated": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        return record

    def actor_config_overrides(self) -> dict[str, object]:
        self.validate()
        return {
            "backend": self.actor_backend,
            "worker_replicas": 4,
            "worker_gpu_per_replica": 1,
            "controller_class": PINNED_CONTROLLER_CLASS,
        }

    def rollout_config_overrides(self) -> dict[str, object]:
        self.validate()
        return {
            "backend": self.rollout_backend,
            "worker_replicas": 4,
            "worker_gpu_per_replica": 1,
            "max_concurrent_rollouts": self.rollout_max_concurrent,
            "consumer_batch_size": self.rollout_consumer_batch_size,
            "dump_to_file": False,
        }


@dataclass(frozen=True)
class M0WorkerPlacement:
    role: str
    rank: int
    physical_gpu_id: int
    worker_id: str

    def validate(self) -> None:
        expected_worker_id = (
            f"actor/{self.rank}"
            if self.role == "actor"
            else f"rollout-inf/{self.rank}"
        )
        _require(self.role in {"actor", "rollout"}, "unknown M0 worker role")
        _require(
            type(self.rank) is int
            and 0 <= self.rank < 4
            and type(self.physical_gpu_id) is int,
            "M0 worker rank or GPU ID is invalid",
        )
        _require(
            self.worker_id == expected_worker_id,
            "M0 worker ID differs from its role/rank",
        )


@dataclass
class M0EightGPUOwnershipLedger:
    """Enforce actor lifetime and placement ordering without owning optimizer state."""

    topology: M0EightGPUTopology = field(default_factory=M0EightGPUTopology)
    actor_alive: bool = False
    rollout_alive: bool = False
    y_completed: bool = False
    actor_placements: tuple[M0WorkerPlacement, ...] = ()
    rollout_placements: tuple[M0WorkerPlacement, ...] = ()

    @staticmethod
    def _validate_placements(
        placements: Sequence[M0WorkerPlacement],
        *,
        role: str,
        expected_gpu_ids: tuple[int, ...],
    ) -> tuple[M0WorkerPlacement, ...]:
        value = tuple(placements)
        _require(
            len(value) == 4 and all(type(item) is M0WorkerPlacement for item in value),
            f"M0 {role} requires four typed worker placements",
        )
        for item in value:
            item.validate()
        ordered = tuple(sorted(value, key=lambda item: item.rank))
        _require(
            tuple(item.rank for item in ordered) == tuple(range(4))
            and all(item.role == role for item in ordered)
            and tuple(item.physical_gpu_id for item in ordered) == expected_gpu_ids,
            f"M0 {role} worker placement differs from the frozen GPU partition",
        )
        return ordered

    def start_actor(self, placements: Sequence[M0WorkerPlacement]) -> None:
        self.topology.validate()
        _require(not self.actor_alive and not self.rollout_alive, "actor already started")
        self.actor_placements = self._validate_placements(
            placements,
            role="actor",
            expected_gpu_ids=self.topology.actor_gpu_ids,
        )
        self.actor_alive = True

    def start_rollout(self, placements: Sequence[M0WorkerPlacement]) -> None:
        _require(
            self.actor_alive and not self.rollout_alive and not self.y_completed,
            "rollout must start after actor and while actor remains alive",
        )
        self.rollout_placements = self._validate_placements(
            placements,
            role="rollout",
            expected_gpu_ids=self.topology.rollout_gpu_ids,
        )
        self.rollout_alive = True

    def complete_y(self) -> None:
        _require(
            self.actor_alive and self.rollout_alive and not self.y_completed,
            "Y requires both distributed actor and rollout to remain alive",
        )
        self.y_completed = True

    def stop_rollout(self) -> None:
        _require(self.y_completed and self.rollout_alive, "rollout cannot stop before Y")
        self.rollout_alive = False

    def stop_actor(self) -> None:
        _require(
            self.y_completed and not self.rollout_alive and self.actor_alive,
            "actor cannot stop before Y or before rollout cleanup",
        )
        self.actor_alive = False


def assert_controller_has_no_local_optimizer(controller: object) -> None:
    """Reject controller objects that mirror worker optimizer state locally."""

    state = vars(controller)
    forbidden = {"optimizer", "lr_scheduler"}.intersection(state)
    _require(
        not forbidden,
        "M0 distributed controller must not hold worker optimizer or scheduler state",
    )


def observe_local_scheduler_placements(
    scheduler: object,
    *,
    role: str,
) -> tuple[M0WorkerPlacement, ...]:
    """Read pinned LocalScheduler placement metadata, never worker optimizer state."""

    _require(role in {"actor", "rollout"}, "unknown scheduler placement role")
    workers_by_role = vars(scheduler).get("_workers")
    _require(
        isinstance(workers_by_role, Mapping),
        "LocalScheduler placement registry is unavailable",
    )
    scheduler_role = role if role == "actor" else "rollout-inf"
    raw_workers = workers_by_role.get(scheduler_role)
    _require(
        isinstance(raw_workers, list) and len(raw_workers) == 4,
        f"LocalScheduler did not create four {role} workers",
    )
    placements: list[M0WorkerPlacement] = []
    for rank, worker_info in enumerate(raw_workers):
        gpu_devices = getattr(worker_info, "gpu_devices", None)
        worker = getattr(worker_info, "worker", None)
        worker_id = getattr(worker, "id", None)
        _require(
            isinstance(gpu_devices, list)
            and len(gpu_devices) == 1
            and type(gpu_devices[0]) is int
            and isinstance(worker_id, str),
            f"LocalScheduler {role}/{rank} placement is invalid",
        )
        placements.append(
            M0WorkerPlacement(
                role=role,
                rank=rank,
                physical_gpu_id=gpu_devices[0],
                worker_id=worker_id,
            )
        )
    return tuple(placements)


def admit_individual_training_samples(
    samples: object,
    *,
    export_style: str,
    minimum_sample_count: int = 4,
) -> tuple[Mapping[str, object], ...]:
    _require(export_style == "individual", "M0 distributed training rejects concat")
    _require(
        isinstance(samples, Sequence)
        and not isinstance(samples, (str, bytes, bytearray))
        and len(samples) >= minimum_sample_count >= 4,
        "M0 distributed training requires at least four samples",
    )
    value = tuple(samples)
    _require(
        all(isinstance(sample, Mapping) for sample in value),
        "M0 distributed training sample is not an object",
    )
    sample_ids = [sample.get("sample_id") for sample in value]
    _require(
        all(isinstance(sample_id, str) and bool(sample_id) for sample_id in sample_ids)
        and len(set(sample_ids)) == len(sample_ids),
        "M0 distributed training sample IDs are missing or duplicated",
    )
    _canonical_json(value)
    return value


def validate_per_gpu_memory_envelope(
    *,
    baseline_used_mib: Mapping[int, int],
    peak_used_mib: Mapping[int, int],
    topology: M0EightGPUTopology | None = None,
) -> dict[int, int]:
    """Return observed per-GPU deltas without enforcing a fixed memory limit."""

    plan = topology or M0EightGPUTopology()
    plan.validate()
    expected = set(plan.scheduler_gpu_ids)
    _require(
        set(baseline_used_mib) == expected and set(peak_used_mib) == expected,
        "M0 memory audit must cover all eight physical GPUs",
    )
    deltas: dict[int, int] = {}
    for gpu_id in plan.scheduler_gpu_ids:
        baseline = baseline_used_mib[gpu_id]
        peak = peak_used_mib[gpu_id]
        _require(
            type(baseline) is int and baseline >= 0 and type(peak) is int and peak >= 0,
            f"GPU {gpu_id} memory sample is invalid",
        )
        delta = max(0, peak - baseline)
        deltas[gpu_id] = delta
    return deltas


@dataclass(frozen=True)
class ValidatedRemoteOptimizerReceipt:
    transaction_id: str
    joint_version_id: str
    source_admission_sha256: str
    global_sample_count: int
    global_sample_sha256: str
    optimizer_step_before: int
    optimizer_step_after: int
    rank0_lr_scheduler_state_after: Mapping[str, object]
    runtime_states: tuple[Mapping[str, object], ...]
    record_sha256: str


def _validate_rank_receipt(
    record: object,
    *,
    expected_rank: int,
    topology: M0EightGPUTopology,
) -> Mapping[str, object]:
    fields = {
        "schema_version",
        "transaction_id",
        "joint_version_id",
        "source_admission_sha256",
        "worker_rank",
        "world_size",
        "engine_name",
        "physical_gpu_id",
        "engine_class",
        "optimizer_class",
        "inference_engine_version",
        "global_sample_count",
        "global_sample_sha256",
        "local_sample_indices",
        "optimizer_step_before",
        "optimizer_step_after",
        "lr_scheduler_state_before_sha256",
        "lr_scheduler_state_after_sha256",
        "lr_scheduler_state_after",
        "parent_dcp_manifest_sha256",
        "candidate_dcp_manifest_sha256",
        "runtime_state",
        "update_stats",
        "evidence_scope",
        "record_sha256",
    }
    _require(isinstance(record, Mapping) and set(record) == fields, "rank receipt fields differ")
    _require(
        record["schema_version"] == M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA
        and record["record_sha256"] == _record_sha256(record),
        "rank receipt schema or hash differs",
    )
    rank = record["worker_rank"]
    _require(
        rank == expected_rank
        and record["world_size"] == 4
        and record["engine_name"] == f"actor/{expected_rank}"
        and record["physical_gpu_id"] == topology.actor_gpu_ids[expected_rank],
        "rank receipt worker identity differs from topology",
    )
    _require(
        record["engine_class"] == PINNED_AREAL_ACTOR_CLASS
        and record["optimizer_class"] == PINNED_OPTIMIZER_CLASS,
        "rank receipt engine or optimizer identity differs",
    )
    _require(
        type(record["inference_engine_version"]) is int
        and record["inference_engine_version"] >= 0
        and type(record["global_sample_count"]) is int
        and record["global_sample_count"] >= topology.minimum_training_sample_count
        and _is_sha256(record["global_sample_sha256"]),
        "rank receipt version or global batch identity is invalid",
    )
    local_indices = record["local_sample_indices"]
    _require(
        isinstance(local_indices, list)
        and bool(local_indices)
        and all(type(index) is int and index >= 0 for index in local_indices)
        and len(set(local_indices)) == len(local_indices),
        "rank receipt local sample partition is invalid",
    )
    before = record["optimizer_step_before"]
    after = record["optimizer_step_after"]
    _require(
        type(before) is int and before >= 0 and after == before + 1,
        "rank receipt does not prove exactly one optimizer step",
    )
    _require(
        _is_sha256(record["lr_scheduler_state_before_sha256"])
        and _is_sha256(record["lr_scheduler_state_after_sha256"])
        and record["lr_scheduler_state_before_sha256"]
        != record["lr_scheduler_state_after_sha256"]
        and _is_sha256(record["parent_dcp_manifest_sha256"])
        and _is_sha256(record["candidate_dcp_manifest_sha256"])
        and record["parent_dcp_manifest_sha256"]
        != record["candidate_dcp_manifest_sha256"],
        "rank receipt scheduler or DCP transition is invalid",
    )
    scheduler_state_after = record["lr_scheduler_state_after"]
    if expected_rank == 0:
        _require(
            isinstance(scheduler_state_after, Mapping)
            and bool(scheduler_state_after)
            and _json_sha256(scheduler_state_after)
            == record["lr_scheduler_state_after_sha256"],
            "rank 0 scheduler state_after is missing or differs from its digest",
        )
    else:
        _require(
            scheduler_state_after is None,
            "only rank 0 may carry the full scheduler state_after",
        )
    runtime_state = record["runtime_state"]
    runtime_fields = {
        "hostname",
        "local_rank",
        "device",
        "cuda_visible_devices",
        "python_rng_state_sha256",
        "torch_cpu_rng_state_sha256",
        "torch_cuda_rng_state_sha256",
    }
    _require(
        isinstance(runtime_state, Mapping)
        and set(runtime_state) == runtime_fields
        and isinstance(runtime_state["hostname"], str)
        and bool(runtime_state["hostname"])
        and runtime_state["local_rank"] == 0
        and runtime_state["device"] == "cuda:0"
        and runtime_state["cuda_visible_devices"]
        == str(topology.actor_gpu_ids[expected_rank])
        and all(
            _is_sha256(runtime_state[field_name])
            for field_name in (
                "python_rng_state_sha256",
                "torch_cpu_rng_state_sha256",
                "torch_cuda_rng_state_sha256",
            )
        ),
        "rank receipt runtime state differs from the AReaL one-process/one-GPU worker",
    )
    stats = record["update_stats"]
    _require(
        isinstance(stats, Mapping)
        and set(stats) == {
            "update_successful",
            "update_successful_count",
            "grad_norm",
            "grad_norm_count",
            "learning_rate",
            "learning_rate_count",
        }
        and stats["update_successful"] == 1.0
        and stats["update_successful_count"] == 1
        and _finite_positive(stats["grad_norm"])
        and stats["grad_norm_count"] == 1
        and _finite_positive(stats["learning_rate"])
        and stats["learning_rate_count"] == 1,
        "rank receipt update stats do not prove one successful update",
    )
    _require(
        record["evidence_scope"]
        == {
            "remote_worker_optimizer_observed": True,
            "controller_read_worker_optimizer": False,
            "policy_optimizer_update": True,
            "harness_optimizer_update": False,
        },
        "rank receipt evidence scope differs",
    )
    for field_name in (
        "transaction_id",
        "joint_version_id",
        "source_admission_sha256",
    ):
        _require(
            isinstance(record[field_name], str) and bool(record[field_name]),
            f"rank receipt {field_name} is missing",
        )
    _require(
        _is_sha256(record["source_admission_sha256"]),
        "rank receipt source admission digest is invalid",
    )
    return record


def validate_remote_optimizer_receipt(
    record: Mapping[str, object],
    *,
    topology: M0EightGPUTopology | None = None,
) -> ValidatedRemoteOptimizerReceipt:
    """Validate worker-created optimizer evidence without reading controller state."""

    plan = topology or M0EightGPUTopology()
    plan.validate()
    fields = {
        "schema_version",
        "transaction_id",
        "joint_version_id",
        "source_admission_sha256",
        "actor_backend",
        "controller_class",
        "global_sample_count",
        "global_sample_sha256",
        "rank0_lr_scheduler_state_after",
        "rank_receipts",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == fields, "remote optimizer receipt fields differ")
    _require(
        record["schema_version"] == M0_REMOTE_OPTIMIZER_RECEIPT_SCHEMA
        and record["record_sha256"] == _record_sha256(record),
        "remote optimizer receipt schema or hash differs",
    )
    _require(
        record["actor_backend"] == plan.actor_backend
        and record["controller_class"] == PINNED_CONTROLLER_CLASS,
        "remote optimizer receipt topology differs",
    )
    raw_ranks = record["rank_receipts"]
    _require(
        isinstance(raw_ranks, list) and len(raw_ranks) == 4,
        "remote optimizer receipt requires all four rank receipts",
    )
    ranks = tuple(
        _validate_rank_receipt(item, expected_rank=rank, topology=plan)
        for rank, item in enumerate(raw_ranks)
    )
    common_fields = (
        "transaction_id",
        "joint_version_id",
        "source_admission_sha256",
        "inference_engine_version",
        "global_sample_count",
        "global_sample_sha256",
        "optimizer_step_before",
        "optimizer_step_after",
        "lr_scheduler_state_before_sha256",
        "lr_scheduler_state_after_sha256",
        "parent_dcp_manifest_sha256",
        "candidate_dcp_manifest_sha256",
        "update_stats",
    )
    for name in common_fields:
        _require(
            all(rank[name] == ranks[0][name] for rank in ranks),
            f"remote optimizer rank receipts disagree on {name}",
        )
    rank0_scheduler_state = record["rank0_lr_scheduler_state_after"]
    _require(
        isinstance(rank0_scheduler_state, Mapping)
        and rank0_scheduler_state == ranks[0]["lr_scheduler_state_after"]
        and _json_sha256(rank0_scheduler_state)
        == ranks[0]["lr_scheduler_state_after_sha256"],
        "remote optimizer aggregate does not bind rank 0 scheduler state_after",
    )
    _require(
        record["transaction_id"] == ranks[0]["transaction_id"]
        and record["joint_version_id"] == ranks[0]["joint_version_id"]
        and record["source_admission_sha256"]
        == ranks[0]["source_admission_sha256"]
        and record["global_sample_count"] == ranks[0]["global_sample_count"]
        and record["global_sample_sha256"] == ranks[0]["global_sample_sha256"],
        "remote optimizer aggregate identity differs from rank receipts",
    )
    partitions = [
        index for rank in ranks for index in rank["local_sample_indices"]
    ]
    _require(
        sorted(partitions) == list(range(int(record["global_sample_count"]))),
        "remote optimizer rank sample partitions are not disjoint and exhaustive",
    )
    _require(
        record["evidence_scope"]
        == {
            "all_actor_ranks_attested": True,
            "remote_worker_optimizer_observed": True,
            "controller_read_worker_optimizer": False,
            "policy_optimizer_update": True,
            "harness_optimizer_update": False,
        },
        "remote optimizer aggregate evidence scope differs",
    )
    return ValidatedRemoteOptimizerReceipt(
        transaction_id=str(record["transaction_id"]),
        joint_version_id=str(record["joint_version_id"]),
        source_admission_sha256=str(record["source_admission_sha256"]),
        global_sample_count=int(record["global_sample_count"]),
        global_sample_sha256=str(record["global_sample_sha256"]),
        optimizer_step_before=int(ranks[0]["optimizer_step_before"]),
        optimizer_step_after=int(ranks[0]["optimizer_step_after"]),
        rank0_lr_scheduler_state_after=dict(rank0_scheduler_state),
        runtime_states=tuple(dict(rank["runtime_state"]) for rank in ranks),
        record_sha256=str(record["record_sha256"]),
    )


def build_local_scheduler(
    *,
    experiment_name: str,
    trial_name: str,
    fileroot: str,
    log_dir: str,
    name_resolve_root: str,
    topology: M0EightGPUTopology | None = None,
) -> object:
    """Construct the single pinned LocalScheduler; this does not launch workers."""

    plan = topology or M0EightGPUTopology()
    plan.validate()
    try:
        from areal.infra.scheduler.local import LocalScheduler
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote gate
        raise M0EightGPUTopologyError("pinned AReaL LocalScheduler is unavailable") from exc
    scheduler = LocalScheduler(
        gpu_devices=list(plan.scheduler_gpu_ids),
        log_dir=log_dir,
        startup_timeout=180.0,
        experiment_name=experiment_name,
        trial_name=trial_name,
        fileroot=fileroot,
        name_resolve_type="nfs",
        nfs_record_root=name_resolve_root,
    )
    _require(
        list(scheduler.gpu_devices) == list(plan.scheduler_gpu_ids),
        "LocalScheduler GPU devices differ from the M0 topology",
    )
    return scheduler


__all__ = [
    "M0_EIGHT_GPU_TOPOLOGY_SCHEMA",
    "M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA",
    "M0_REMOTE_OPTIMIZER_RECEIPT_SCHEMA",
    "M0EightGPUOwnershipLedger",
    "M0EightGPUTopology",
    "M0EightGPUTopologyError",
    "M0WorkerPlacement",
    "ValidatedRemoteOptimizerReceipt",
    "admit_individual_training_samples",
    "assert_controller_has_no_local_optimizer",
    "build_local_scheduler",
    "observe_local_scheduler_placements",
    "validate_per_gpu_memory_envelope",
    "validate_remote_optimizer_receipt",
]
