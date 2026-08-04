from __future__ import annotations

"""Real AReaL adapters for the formal eight-GPU M0 stage machine.

The module deliberately keeps imports of CUDA/AReaL components inside launch
methods.  Importing it on a CPU host therefore validates configuration and
negative contracts without pretending that a GPU transition took place.
"""

import hashlib
import json
import os
import platform
import subprocess
import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from uuid import uuid4

from jphrl.paths import (
    repository_root,
    require_outside_repository,
    require_within_configured_root,
)
from jphrl.trajectory.areal_joint_bridge import (
    DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
    inference_runtime_contract_sha256,
)
from jphrl.trajectory.multi_s_frozen_training_batch import (
    ValidatedMultiSFrozenTrainingBatch,
    required_v_member_claims,
)

from .m0_eight_gpu_integrated import (
    EightGPUMemoryObservation,
    GPUObservation,
    GPUProcessObservation,
    IntegratedCleanupReceipt,
    IntegratedSchedulerHandle,
    IntegratedStageReference,
    M0EightGPUIntegratedError,
)
from .m0_eight_gpu_topology import (
    M0EightGPUTopology,
    M0WorkerPlacement,
    assert_controller_has_no_local_optimizer,
    build_local_scheduler,
    observe_local_scheduler_placements,
)
from .m0_joint_runner import (
    M0ArealActorSpec,
    PINNED_AREAL_COMMIT,
    RLVRM0SourceRecords,
    build_pinned_areal_actor_config,
)


M0_EIGHT_GPU_REAL_ADAPTER_SCHEMA = "jph.m0-eight-gpu-real-adapter.v1"
M0_EIGHT_GPU_TUVW_STAGE_SCHEMA = "jph.m0-eight-gpu-tuvw-stage.v1"
M0_EIGHT_GPU_X_STAGE_SCHEMA = "jph.m0-eight-gpu-x-stage.v1"
M0_EIGHT_GPU_Y_STAGE_SCHEMA = "jph.m0-eight-gpu-y-stage.v1"


class M0EightGPURealAdapterError(M0EightGPUIntegratedError):
    """Raised when the real adapter cannot prove an AReaL operation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0EightGPURealAdapterError(message)


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
        raise M0EightGPURealAdapterError(
            "real adapter evidence is not finite canonical JSON"
        ) from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_acceptance_spec(gates: Sequence[object]) -> object:
    from jphrl.training.candidate_acceptance import (
        CandidateAcceptanceSpec,
        CandidateAcceptanceSuite,
    )

    kind_map = {
        "policy_heldout": "policy_heldout",
        "harness_offpolicy": "harness_heldout",
        "joint_safety": "joint_safety",
        "restart_recovery": "historical_regression",
    }
    _require(
        tuple(getattr(gate, "kind", None) for gate in gates) == tuple(kind_map),
        "distributed X gates are incomplete or unordered",
    )
    suites = tuple(
        CandidateAcceptanceSuite(
            kind=kind_map[gate.kind],
            suite_id=gate.suite_id,
            fixture_sha256=gate.fixture_sha256,
            metric_name=gate.metric_name,
            minimum_score=float(gate.minimum_score),
            minimum_sample_count=gate.minimum_sample_count,
        )
        for gate in gates
    )
    spec = CandidateAcceptanceSpec(suites=suites)
    spec.validate()
    return spec


def _attach_joint_safety_probe(
    observations: Sequence[object],
    *,
    suite_kind: str,
    production_probe_output: bytes,
) -> tuple[object, ...]:
    from jphrl.training.candidate_acceptance import CandidateProbeObservation

    values = tuple(observations)
    if suite_kind != "joint_safety":
        _require(
            all(
                getattr(observation, "production_probe_output", None) is None
                for observation in values
            ),
            "only joint_safety may carry production probe bytes",
        )
        return values
    _require(
        len(values) == 1 and type(values[0]) is CandidateProbeObservation,
        "joint_safety requires one native observation",
    )
    observation = values[0]
    _require(
        observation.production_probe_output in {None, production_probe_output},
        "joint_safety observation supplied crossed production bytes",
    )
    return (
        CandidateProbeObservation(
            sample_id=observation.sample_id,
            metric_value=observation.metric_value,
            output=observation.output,
            production_probe_output=production_probe_output,
        ),
    )


def _joint_safety_probe_sha256(report: Mapping[str, object]) -> str:
    suites = report.get("critical_suites")
    _require(isinstance(suites, list), "X acceptance suites are missing")
    matches = [
        suite
        for suite in suites
        if isinstance(suite, Mapping)
        and isinstance(suite.get("spec"), Mapping)
        and suite["spec"].get("kind") == "joint_safety"
    ]
    _require(len(matches) == 1, "X has no unique joint_safety suite")
    probe = matches[0].get("probe")
    digest = (
        probe.get("production_probe_output_sha256")
        if isinstance(probe, Mapping)
        else None
    )
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "X joint_safety production probe digest is invalid",
    )
    return digest


def _validate_y_actor_terminal_receipts(
    receipts: Sequence[object],
    *,
    live_policy_candidate: object,
    activation: object,
) -> tuple[dict[str, object], ...]:
    """Bind each terminal actor receipt to T and the completed Y attestation."""

    live_receipt = getattr(live_policy_candidate, "receipt", None)
    _require(isinstance(live_receipt, Mapping), "Y live Policy receipt is missing")
    transaction = live_receipt.get("transaction")
    optimizer = live_receipt.get("optimizer")
    remote = (
        optimizer.get("remote_optimizer_receipt")
        if isinstance(optimizer, Mapping)
        else None
    )
    rank_receipts = remote.get("rank_receipts") if isinstance(remote, Mapping) else None
    _require(
        isinstance(transaction, Mapping)
        and isinstance(remote, Mapping)
        and isinstance(rank_receipts, list)
        and len(rank_receipts) == 4,
        "Y live Policy distributed lineage is incomplete",
    )
    transaction_id = transaction.get("transaction_id")
    candidate_sha256 = live_receipt.get("record_sha256")
    aggregate_sha256 = remote.get("record_sha256")
    attestation_sha256 = getattr(activation, "attestation_sha256", None)
    active_release_id = getattr(activation, "active_release_id", None)
    candidate_release_id = getattr(activation, "candidate_release_id", None)
    _require(
        isinstance(transaction_id, str)
        and bool(transaction_id)
        and isinstance(candidate_sha256, str)
        and isinstance(aggregate_sha256, str)
        and isinstance(attestation_sha256, str)
        and active_release_id == candidate_release_id
        and isinstance(active_release_id, str)
        and bool(active_release_id),
        "Y terminal receipt lineage inputs are invalid",
    )
    _require(len(receipts) == 4, "Y requires exactly four actor terminal receipts")
    expected_fields = {
        "schema_version",
        "transaction_id",
        "worker_rank",
        "aggregate_sha256",
        "policy_candidate_sha256",
        "rank_receipt_sha256",
        "y_attestation_sha256",
        "y_active_release_id",
        "state_before",
        "state_after",
        "evidence_scope",
        "record_sha256",
    }
    expected_scope = {
        "y_success_revalidated": True,
        "training_state_changed": False,
        "rollback_state_clear_authorized": True,
        "policy_optimizer_update": False,
        "harness_optimizer_update": False,
    }
    summaries: list[dict[str, object]] = []
    for rank, value in enumerate(receipts):
        _require(isinstance(value, Mapping), f"Y actor terminal rank {rank} is untyped")
        receipt = dict(value)
        rank_source = rank_receipts[rank]
        state = receipt.get("state_before")
        _require(
            set(receipt) == expected_fields
            and receipt.get("schema_version") == "jph.m0-policy-worker-commit.v1"
            and receipt.get("record_sha256") == _record_sha256(receipt)
            and receipt.get("worker_rank") == rank
            and receipt.get("transaction_id") == transaction_id
            and receipt.get("aggregate_sha256") == aggregate_sha256
            and receipt.get("policy_candidate_sha256") == candidate_sha256
            and isinstance(rank_source, Mapping)
            and receipt.get("rank_receipt_sha256")
            == rank_source.get("record_sha256")
            and receipt.get("y_attestation_sha256") == attestation_sha256
            and receipt.get("y_active_release_id") == active_release_id
            and isinstance(state, Mapping)
            and set(state)
            == {"actor_version", "optimizer_step", "scheduler_state_sha256"}
            and type(state.get("actor_version")) is int
            and state["actor_version"] >= 0
            and type(state.get("optimizer_step")) is int
            and state["optimizer_step"] >= 0
            and isinstance(state.get("scheduler_state_sha256"), str)
            and len(state["scheduler_state_sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in state["scheduler_state_sha256"]
            )
            and receipt.get("state_after") == state
            and receipt.get("evidence_scope") == expected_scope,
            f"Y actor terminal rank {rank} receipt differs from T/Y",
        )
        summaries.append(
            {
                "worker_rank": rank,
                "transaction_id": transaction_id,
                "aggregate_sha256": aggregate_sha256,
                "policy_candidate_sha256": candidate_sha256,
                "rank_receipt_sha256": receipt["rank_receipt_sha256"],
                "y_attestation_sha256": attestation_sha256,
                "y_active_release_id": active_release_id,
                "record_sha256": receipt["record_sha256"],
            }
        )
    return tuple(summaries)


def _recover_y_parent_after_terminal_failure(
    *,
    activation_controller: object,
    activation: object,
    release_store: object,
    parent_release: object,
    candidate_release: object,
    worker: object,
    parent_probe: object,
) -> dict[str, object]:
    """Consume Y's rollback-only record and re-observe the restored parent."""

    from jphrl.training.joint_activation import (
        ProductionProbeSpec,
        ProductionRollbackRecoveryResult,
        ProductionWorkerState,
    )

    _require(
        type(parent_probe) is ProductionProbeSpec,
        "Y terminal compensation requires the frozen parent probe",
    )
    parent_probe.validate()
    recovery = activation_controller.recover_pending(
        getattr(activation, "rollback_record_path", None)
    )
    active = release_store.read_active()
    state = worker.read_state()
    fixture = getattr(parent_probe, "fixture", None)
    expected_probe_sha256 = getattr(parent_probe, "expected_output_sha256", None)
    raw_probe = worker.run_probe(fixture)
    _require(
        type(recovery) is ProductionRollbackRecoveryResult
        and recovery.activation_id == getattr(activation, "activation_id", None)
        and recovery.parent_release_id == getattr(parent_release, "release_id", None)
        and recovery.candidate_release_id
        == getattr(candidate_release, "release_id", None)
        and recovery.active_release_id == getattr(parent_release, "release_id", None)
        and recovery.outcome == "parent_restored_from_journal"
        and recovery.rollback_record_path
        == getattr(activation, "rollback_record_path", None)
        and active == parent_release
        and type(state) is ProductionWorkerState
        and state.lifecycle_phase == "serving"
        and state.active_release_id == getattr(parent_release, "release_id", None)
        and state.joint_version == getattr(parent_release, "joint_version", None)
        and type(fixture) is bytes
        and bool(fixture)
        and type(raw_probe) is bytes
        and hashlib.sha256(raw_probe).hexdigest() == expected_probe_sha256,
        "Y terminal compensation did not restore and probe the exact parent",
    )
    return {
        "activation_id": recovery.activation_id,
        "parent_release_id": recovery.parent_release_id,
        "candidate_release_id": recovery.candidate_release_id,
        "active_release_id": recovery.active_release_id,
        "rollback_record_path": str(recovery.rollback_record_path),
        "parent_worker_id": state.worker_id,
        "parent_probe_sha256": expected_probe_sha256,
    }


def _commit_actor_with_y_compensation(
    *,
    actor: object,
    live_policy_candidate: object,
    activation_controller: object,
    activation: object,
    release_store: object,
    parent_release: object,
    candidate_release: object,
    worker: object,
    parent_probe: object,
) -> tuple[dict[str, object], ...]:
    """Commit all actor ranks or synchronously restore the public parent pair."""

    try:
        receipts = actor.commit_m0_policy_candidate(
            live_policy_candidate,
            production_activation=activation,
        )
        return _validate_y_actor_terminal_receipts(
            receipts,
            live_policy_candidate=live_policy_candidate,
            activation=activation,
        )
    except BaseException as terminal_error:
        try:
            _recover_y_parent_after_terminal_failure(
                activation_controller=activation_controller,
                activation=activation,
                release_store=release_store,
                parent_release=parent_release,
                candidate_release=candidate_release,
                worker=worker,
                parent_probe=parent_probe,
            )
        except BaseException as recovery_error:
            raise M0EightGPURealAdapterError(
                "Y actor terminal failure and production parent compensation both failed"
            ) from recovery_error
        raise terminal_error


def _write_new_json(path: Path, record: Mapping[str, object]) -> Path:
    target = require_outside_repository(path)
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(target.parent, 0o700)
    payload = _canonical_json(record) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    created = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(target, 0o600)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


def _strict_json(path: Path, label: str) -> dict[str, object]:
    source = require_within_configured_root(path)
    _require(source.is_file() and not source.is_symlink(), f"{label} is unsafe")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise M0EightGPURealAdapterError(f"{label} is not strict JSON") from exc
    _require(isinstance(value, dict), f"{label} must contain one object")
    _canonical_json(value)
    return value


def _snapshot_metadata(path: Path, label: str) -> tuple[Path, str]:
    record = _strict_json(path, label)
    _require(
        set(record) >= {"snapshot_path", "resolved_commit"}
        and isinstance(record["snapshot_path"], str)
        and isinstance(record["resolved_commit"], str)
        and len(record["resolved_commit"]) == 40,
        f"{label} lacks a pinned snapshot identity",
    )
    snapshot = require_within_configured_root(record["snapshot_path"])
    _require(snapshot.is_dir() and not snapshot.is_symlink(), f"{label} snapshot is unsafe")
    return snapshot, str(record["resolved_commit"])


@dataclass(frozen=True)
class RealEightGPUAdapterConfig:
    jph_root: Path
    run_root: Path
    project_commit: str
    model_snapshot: Path
    behavior_revision: str
    dataset_snapshot: Path
    dataset_revision: str
    admin_api_key: str
    transaction_id: str
    experiment_name: str = "jph-m0-eight-gpu"
    trial_name: str = "formal"
    dataset_selection: str = "sequential-offset0-count8-v1"
    learning_rate: float = 1e-6
    rollout_mem_fraction_static: float = 0.29
    max_new_tokens: int = 512
    harness_seed: int = 1
    harness_hidden_size: int = 32

    @classmethod
    def from_snapshot_reports(
        cls,
        *,
        jph_root: str | Path,
        run_root: str | Path,
        project_commit: str,
        model_report: str | Path,
        dataset_report: str | Path,
        admin_api_key: str,
        transaction_id: str,
        trial_name: str,
    ) -> RealEightGPUAdapterConfig:
        root = require_within_configured_root(jph_root)
        model, behavior_revision = _snapshot_metadata(
            Path(model_report), "model snapshot report"
        )
        dataset, dataset_revision = _snapshot_metadata(
            Path(dataset_report), "dataset snapshot report"
        )
        value = cls(
            jph_root=root,
            run_root=require_outside_repository(run_root),
            project_commit=project_commit,
            model_snapshot=model,
            behavior_revision=behavior_revision,
            dataset_snapshot=dataset,
            dataset_revision=dataset_revision,
            admin_api_key=admin_api_key,
            transaction_id=transaction_id,
            trial_name=trial_name,
        )
        value.validate()
        return value

    def validate(self) -> None:
        root = require_within_configured_root(self.jph_root)
        run = require_outside_repository(self.run_root)
        _require(
            root in run.parents
            and self.model_snapshot.is_dir()
            and self.dataset_snapshot.is_dir()
            and all(
                root == path or root in path.parents
                for path in (self.model_snapshot, self.dataset_snapshot)
            ),
            "real adapter paths escape JPH_ROOT or snapshots are missing",
        )
        _require(
            len(self.project_commit) == 40
            and len(self.behavior_revision) == 40
            and len(self.dataset_revision) == 40
            and all(
                all(character in "0123456789abcdef" for character in value)
                for value in (
                    self.project_commit,
                    self.behavior_revision,
                    self.dataset_revision,
                )
            ),
            "real adapter commit identity is invalid",
        )
        _require(
            isinstance(self.admin_api_key, str) and len(self.admin_api_key) >= 32,
            "AReaL admin key is missing or too short",
        )
        _require(
            isinstance(self.transaction_id, str)
            and bool(self.transaction_id)
            and isinstance(self.trial_name, str)
            and bool(self.trial_name),
            "real adapter transaction/trial identity is missing",
        )
        _require(
            self.dataset_selection == "sequential-offset0-count8-v1"
            and 0.0 < float(self.rollout_mem_fraction_static) <= 0.95,
            "formal admission selection or SGLang memory fraction differs",
        )


def build_distributed_actor_config(config: RealEightGPUAdapterConfig) -> object:
    """Build the exact audited ``PPOActorConfig`` for actor ranks 0..3."""

    config.validate()
    spec = M0ArealActorSpec(
        model_path=str(config.model_snapshot),
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        learning_rate=config.learning_rate,
        dtype="bfloat16",
        optimizer_dtype="float32",
        attention_implementation="flash_attention_2",
        gradient_checkpointing=True,
        max_new_tokens=config.max_new_tokens,
    )
    actor_config = build_pinned_areal_actor_config(spec)
    try:
        from areal.api.cli_args import SchedulingSpec
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote gate
        raise M0EightGPURealAdapterError("pinned AReaL config is unavailable") from exc
    actor_config.backend = "fsdp:d4"
    actor_config.scheduling_spec = (
        SchedulingSpec(
            gpu=1,
            cpu=8,
            mem=32,
            port_count=2,
            cmd="python -m areal.infra.rpc.rpc_server",
        ),
    )
    actor_config.fsdp.per_layer_optim_step = False
    _require(
        actor_config.backend == "fsdp:d4"
        and len(actor_config.scheduling_spec) == 1
        and actor_config.scheduling_spec[0].gpu == 1
        and actor_config.fsdp.per_layer_optim_step is False,
        "distributed actor config differs from fsdp:d4 one-GPU workers",
    )
    return actor_config


def build_distributed_rollout_config(config: RealEightGPUAdapterConfig) -> object:
    """Build the pinned v2 DataProxy rollout config for ranks 4..7."""

    config.validate()
    try:
        from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote gate
        raise M0EightGPURealAdapterError("pinned AReaL rollout config is unavailable") from exc
    value = InferenceEngineConfig(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=str(config.run_root / "scheduler"),
        max_concurrent_rollouts=8,
        consumer_batch_size=4,
        max_head_offpolicyness=0,
        enable_rollout_tracing=False,
        check_trajectory_format=False,
        tokenizer_path=str(config.model_snapshot),
        dump_to_file=False,
        setup_timeout=900.0,
        workers_ready_timeout=180.0,
        request_timeout=180.0,
        request_retries=1,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cpu=8,
                mem=32,
                port_count=2,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        backend="sglang:d4",
        _version="v2",
        model=str(config.model_snapshot),
        admin_api_key=config.admin_api_key,
    )
    _require(
        value.backend == "sglang:d4"
        and value._version == "v2"
        and value.max_concurrent_rollouts == 8
        and value.consumer_batch_size == 4
        and len(value.scheduling_spec) == 1
        and value.scheduling_spec[0].gpu == 1,
        "distributed rollout config differs from sglang:d4 DataProxy",
    )
    return value


def build_distributed_server_args(config: RealEightGPUAdapterConfig) -> dict[str, object]:
    config.validate()
    try:
        from areal.api.cli_args import SGLangConfig
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote gate
        raise M0EightGPURealAdapterError("pinned AReaL SGLang config is unavailable") from exc
    sglang = SGLangConfig(
        model_path=str(config.model_snapshot),
        random_seed=config.harness_seed,
        disable_cuda_graph=False,
        disable_radix_cache=True,
        context_length=4096,
        mem_fraction_static=float(config.rollout_mem_fraction_static),
        max_running_requests=8,
        dtype="bfloat16",
    )
    value = SGLangConfig.build_args(
        sglang_config=sglang,
        tp_size=1,
        base_gpu_id=0,
    )
    value["tokenizer_path"] = str(config.model_snapshot)
    _canonical_json(value)
    return value


def build_distributed_inference_runtime_contract(
    config: RealEightGPUAdapterConfig,
    *,
    server_args: Mapping[str, object],
    gpu_uuids: Sequence[str],
    gpu_names: Sequence[str],
) -> dict[str, object]:
    """Bind every distributed rollout rank without persisting a credential."""

    config.validate()
    _require(
        len(gpu_uuids) == len(gpu_names) == 4,
        "distributed runtime requires four GPU identities",
    )
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote gate
        raise M0EightGPURealAdapterError("Torch runtime is unavailable") from exc
    generation = {
        "n_samples": 1,
        "max_new_tokens": config.max_new_tokens,
        "min_new_tokens": 0,
        "max_tokens": 4096,
        "greedy": False,
        "top_p": 1.0,
        "top_k": int(1e8),
        "temperature": 1.0,
        "stop_token_ids": [],
        "ignore_eos": False,
        "skip_special_tokens": True,
        "stop": None,
        "frequency_penalty": 0.0,
        "lora_name": "default_lora",
        "use_beam_search": False,
    }
    fixed: dict[str, object] = {
        "areal_commit": PINNED_AREAL_COMMIT,
        "areal_version": distribution_version("areal"),
        "behavior_revision": config.behavior_revision,
        "clean_environment_policy": "filtered-inherited-v1",
        "cuda_runtime_version": str(torch.version.cuda),
        "cuda_visible_devices_by_rank": ["4", "5", "6", "7"],
        "dataset_revision": config.dataset_revision,
        "dataset_selection": config.dataset_selection,
        "driver_version": _nvidia_driver_version(),
        "generation": generation,
        "gpu_names": list(gpu_names),
        "gpu_uuids": list(gpu_uuids),
        "physical_gpu_ids": [4, 5, 6, 7],
        "python_version": platform.python_version(),
        "project_commit": config.project_commit,
        "rollout": {
            "backend": "sglang:d4",
            "max_concurrent_rollouts": 8,
        },
        "seed": config.harness_seed,
        "server_args": deepcopy(dict(server_args)),
        "sglang_environment": {
            "SGLANG_CACHE_DIR": os.environ.get(
                "SGLANG_CACHE_DIR", str(config.jph_root / "cache" / "sglang")
            )
        },
        "sglang_version": distribution_version("sglang"),
        "torch_version": distribution_version("torch"),
        "transformers_version": distribution_version("transformers"),
    }
    contract: dict[str, object] = {
        "schema_version": DISTRIBUTED_INFERENCE_RUNTIME_CONTRACT_SCHEMA_VERSION,
        "identity": {"run_id": config.transaction_id, "screen_pair_id": None},
        "fixed": fixed,
        "treatment": {
            "disable_cuda_graph": bool(server_args.get("disable_cuda_graph")),
            "experimental_axis": "none-v1",
            "generation_logprob_mode": "standard-log-of-softmax-v1",
            "sglang_return_original_logprob": False,
        },
    }
    inference_runtime_contract_sha256(contract)
    return contract


def _run_nvidia_smi(arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M0EightGPURealAdapterError("cannot observe NVIDIA GPU state") from exc


def _nvidia_driver_version() -> str:
    values = {
        line.strip()
        for line in _run_nvidia_smi(
            ("--query-gpu=driver_version", "--format=csv,noheader")
        ).splitlines()
        if line.strip()
    }
    _require(len(values) == 1, "eight GPUs disagree on driver version")
    return values.pop()


def _gpu_static_identities(gpu_ids: Sequence[int]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    query = _run_nvidia_smi(
        (
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        )
    )
    by_id: dict[int, tuple[str, str]] = {}
    for line in query.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        _require(
            len(fields) == 3 and fields[0].isdigit() and fields[1] and fields[2],
            "NVIDIA static identity row is invalid",
        )
        by_id[int(fields[0])] = (fields[1], fields[2])
    _require(all(gpu_id in by_id for gpu_id in gpu_ids), "NVIDIA GPU identity is missing")
    return (
        tuple(by_id[gpu_id][0] for gpu_id in gpu_ids),
        tuple(by_id[gpu_id][1] for gpu_id in gpu_ids),
    )


class NvidiaSMIGPUStateProvider:
    """Read memory and process ownership; never mutate another process."""

    def snapshot(self) -> Sequence[GPUObservation]:
        memory_rows = _run_nvidia_smi(
            (
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            )
        )
        process_rows = _run_nvidia_smi(
            (
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        )
        identity_rows = _run_nvidia_smi(
            ("--query-gpu=index,uuid", "--format=csv,noheader,nounits")
        )
        uuid_to_id: dict[str, int] = {}
        for line in identity_rows.splitlines():
            index, gpu_uuid = [field.strip() for field in line.split(",", 1)]
            _require(index.isdigit() and gpu_uuid, "GPU UUID mapping is invalid")
            uuid_to_id[gpu_uuid] = int(index)
        processes: dict[int, list[GPUProcessObservation]] = {
            gpu_id: [] for gpu_id in range(8)
        }
        for line in process_rows.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",", 3)]
            _require(
                len(fields) == 4
                and fields[0] in uuid_to_id
                and fields[1].isdigit()
                and fields[3].isdigit(),
                "GPU compute process row is invalid",
            )
            pid = int(fields[1])
            try:
                raw = subprocess.run(
                    ["ps", "-o", "sid=,uid=,user=", "-p", str(pid)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                sid, uid, user = raw.split(maxsplit=2)
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                raise M0EightGPURealAdapterError(
                    "cannot bind GPU process to SID/UID/user"
                ) from exc
            processes[uuid_to_id[fields[0]]].append(
                GPUProcessObservation(
                    pid=pid,
                    session_id=int(sid),
                    uid=int(uid),
                    user=user,
                    process_name=fields[2],
                    used_memory_mib=int(fields[3]),
                )
            )
        observations: list[GPUObservation] = []
        for line in memory_rows.splitlines():
            fields = [field.strip() for field in line.split(",", 2)]
            _require(
                len(fields) == 3 and all(field.isdigit() for field in fields),
                "GPU memory row is invalid",
            )
            gpu_id, used, free = (int(field) for field in fields)
            if gpu_id in range(8):
                observations.append(
                    GPUObservation(
                        gpu_id=gpu_id,
                        memory_used_mib=used,
                        memory_free_mib=free,
                        processes=tuple(processes[gpu_id]),
                    )
                )
        return tuple(observations)


class ThreadedEightGPUMemoryRuntime:
    """One-second observation wrapper used by the integrated stage machine."""

    def __init__(self, *, audit_root: str | Path) -> None:
        self.observation = EightGPUMemoryObservation(
            provider=NvidiaSMIGPUStateProvider(),
            audit_root=audit_root,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def gate(self, stage: str) -> object:
        return self.observation.gate(stage)

    def start_watchdog(self, *, exact_session_id: int) -> None:
        _require(self._thread is None, "memory watchdog is already running")

        def observe() -> None:
            while not self._stop.wait(1.0):
                try:
                    self.observation.watchdog_sample(
                        exact_session_id=exact_session_id
                    )
                except BaseException as exc:  # saved for main-thread fail closed
                    self._error = exc
                    self._stop.set()
                    return

        self._thread = threading.Thread(
            target=observe,
            name="jph-m0-eight-gpu-memory-watchdog",
            daemon=False,
        )
        self._thread.start()

    def assert_watchdog_healthy(self) -> None:
        if self._error is not None:
            raise M0EightGPURealAdapterError("GPU memory watchdog failed") from self._error
        _require(self._thread is not None and self._thread.is_alive(), "GPU watchdog stopped")

    def stop_watchdog(self) -> None:
        _require(self._thread is not None, "GPU memory watchdog was not started")
        self._stop.set()
        self._thread.join(timeout=10.0)
        _require(not self._thread.is_alive(), "GPU memory watchdog did not stop")
        if self._error is not None:
            raise M0EightGPURealAdapterError("GPU memory watchdog failed") from self._error
        # Very short CPU tests may finish before the first interval.  A real run
        # still records a final fresh sample before sealing the audit.
        if self.observation._sample_count == 0:
            self.observation.watchdog_sample(exact_session_id=os.getsid(0))
        self.observation.final_audit()


class RealEightGPUIntegratedAdapters:
    """Own the one LocalScheduler and all live T--Y capabilities."""

    execution_mode = "real-gpu"

    def __init__(self, config: RealEightGPUAdapterConfig) -> None:
        config.validate()
        self.config = config
        self.actor: object | None = None
        self.rollout: object | None = None
        self.server_args: dict[str, object] | None = None
        self.runtime_contract: dict[str, object] | None = None
        self.live_policy_candidate: object | None = None
        self.harness_result: object | None = None
        self.bundle: Mapping[str, object] | None = None
        self.checkpoint_manifest: Path | None = None
        self.live_recovery: object | None = None
        self.serving_exports: object | None = None
        self.release_store: object | None = None
        self.parent_release: object | None = None
        self.candidate_release: object | None = None
        self.live_acceptance: object | None = None
        self.production_activation: object | None = None
        self.production_worker: object | None = None
        self.training_sources: tuple[RLVRM0SourceRecords, ...] = ()
        self.multi_s_batch: ValidatedMultiSFrozenTrainingBatch | None = None
        self.behavior_checkpoint_path: Path | None = None
        self.behavior_checkpoint_sha256: str | None = None
        self.acceptance_spec: object | None = None
        self.actor_candidate_committed = False
        self._joint_safety_fixture: bytes | None = None

    def create_scheduler(
        self,
        *,
        topology: M0EightGPUTopology,
        run_root: Path,
    ) -> IntegratedSchedulerHandle:
        _require(run_root == self.config.run_root, "scheduler run root differs")
        scheduler_root = run_root / "scheduler"
        for path in (
            scheduler_root,
            scheduler_root / "logs",
            scheduler_root / "name-resolve",
        ):
            path.mkdir(parents=True, mode=0o700, exist_ok=True)
            os.chmod(path, 0o700)
        native = build_local_scheduler(
            experiment_name=self.config.experiment_name,
            trial_name=self.config.trial_name,
            fileroot=str(scheduler_root),
            log_dir=str(scheduler_root / "logs"),
            name_resolve_root=str(scheduler_root / "name-resolve"),
            topology=topology,
        )
        return IntegratedSchedulerHandle(
            instance_id=f"local-scheduler-{uuid4().hex}",
            implementation_class="areal.infra.scheduler.local.LocalScheduler",
            gpu_ids=tuple(range(8)),
            native_scheduler=native,
            execution_mode=self.execution_mode,
        )

    def start_actor(
        self,
        scheduler: IntegratedSchedulerHandle,
    ) -> Sequence[M0WorkerPlacement]:
        scheduler.validate(self.execution_mode)
        try:
            from areal.api import FinetuneSpec
            from jphrl.training.areal_distributed_policy import JPHFSDPPPOActor
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote gate
            raise M0EightGPURealAdapterError("distributed AReaL actor is unavailable") from exc
        actor_config = build_distributed_actor_config(self.config)
        actor = JPHFSDPPPOActor.as_controller(
            actor_config,
            scheduler.native_scheduler,
        )
        assert_controller_has_no_local_optimizer(actor)
        try:
            actor.create_process_group()
            actor.initialize(
                "actor",
                FinetuneSpec(
                    total_train_epochs=1,
                    dataset_size=4,
                    train_batch_size=4,
                ),
            )
            actor.set_version(0)
            _require(actor.get_version() == 0, "actor public version did not initialize")
            placements = observe_local_scheduler_placements(
                scheduler.native_scheduler,
                role="actor",
            )
        except BaseException:
            try:
                actor.destroy()
            except Exception:
                pass
            raise
        self.actor = actor
        return placements

    def start_rollout(
        self,
        scheduler: IntegratedSchedulerHandle,
    ) -> Sequence[M0WorkerPlacement]:
        _require(self.actor is not None, "rollout cannot start before actor")
        try:
            from areal.v2.inference_service.controller.controller import (
                RolloutControllerV2,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote gate
            raise M0EightGPURealAdapterError("AReaL DataProxy rollout is unavailable") from exc
        rollout_config = build_distributed_rollout_config(self.config)
        server_args = build_distributed_server_args(self.config)
        gpu_uuids, gpu_names = _gpu_static_identities((4, 5, 6, 7))
        runtime_contract = build_distributed_inference_runtime_contract(
            self.config,
            server_args=server_args,
            gpu_uuids=gpu_uuids,
            gpu_names=gpu_names,
        )
        rollout = RolloutControllerV2(rollout_config, scheduler.native_scheduler)
        try:
            rollout.initialize(role="rollout", server_args=server_args, wait=True)
            placements = observe_local_scheduler_placements(
                scheduler.native_scheduler,
                role="rollout",
            )
        except BaseException:
            try:
                rollout.destroy()
            except Exception:
                pass
            raise
        self.rollout = rollout
        self.server_args = server_args
        self.runtime_contract = runtime_contract
        return placements

    def generate_eight_rlvr_admissions(
        self,
        scheduler: IntegratedSchedulerHandle,
    ) -> Path:
        del scheduler
        _require(
            self.rollout is not None
            and self.runtime_contract is not None
            and self.server_args is not None,
            "RLVR admissions require the live distributed rollout",
        )
        root = self.config.run_root / "rollout-admissions"
        bridge_root = root / "bridges"
        admission_root = root / "runner-admissions"
        score_root = root / "same-backend-scores"
        for path in (root, bridge_root, admission_root, score_root):
            path.mkdir(parents=True, mode=0o700, exist_ok=False)
            os.chmod(path, 0o700)
        from scripts.write_m0_rlvr_estimator_template import (
            write_m0_rlvr_estimator_template,
        )

        estimator_path = root / "frozen-estimator-template.json"
        write_m0_rlvr_estimator_template(
            output_path=estimator_path,
            allowed_root=self.config.jph_root,
        )
        runtime_hash = inference_runtime_contract_sha256(self.runtime_contract)
        env = {
            "JPH_AREAL_JOINT_BRIDGE_DIR": str(bridge_root),
            "JPH_AREAL_SAME_BACKEND_SCORE_DIR": str(score_root),
            "JPH_RLVR_RUNNER_ADMISSION_MODE": "m0-torch-joint-v1",
            "JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH": str(estimator_path),
            "JPH_RLVR_RUNNER_ADMISSION_DIR": str(admission_root),
            "JPH_ROOT": str(self.config.jph_root),
            "JPH_DATASET_SELECTION": self.config.dataset_selection,
            "JPH_EXPECTED_POLICY_VERSION": "0",
            "JPH_BEHAVIOR_REVISION": self.config.behavior_revision,
            "JPH_SGLANG_LOGPROB_MODE": "standard-log-of-softmax-v1",
            "JPH_INFERENCE_RUNTIME_CONTRACT": self.runtime_contract_json,
            "JPH_INFERENCE_RUNTIME_CONTRACT_SHA256": runtime_hash,
            "JPH_AREAL_COMMIT": PINNED_AREAL_COMMIT,
            "JPH_PROJECT_COMMIT": self.config.project_commit,
            "JPH_BEHAVIOR_SNAPSHOT": str(self.config.model_snapshot),
            "JPH_DATASET_REVISION": self.config.dataset_revision,
            "JPH_SGLANG_VERSION": distribution_version("sglang"),
            "JPH_RUN_ID": self.config.transaction_id,
        }
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            from areal.api.cli_args import GenerationHyperparameters, ValidDatasetConfig
            from areal.dataset import get_custom_dataset
            from areal.utils.dataloader import create_dataloader
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(str(self.config.model_snapshot))
            dataset_config = ValidDatasetConfig(
                path=str(self.config.dataset_snapshot),
                type="rl",
                split="test",
                batch_size=8,
                max_length=4096,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            dataset = get_custom_dataset(
                split="test",
                path=str(self.config.dataset_snapshot),
                type="rl",
                max_length=dataset_config.max_length,
                tokenizer=tokenizer,
            )
            dataloader = create_dataloader(
                dataset,
                rank=0,
                world_size=1,
                dataset_config=dataset_config,
            )
            gconfig = GenerationHyperparameters(
                n_samples=1,
                max_new_tokens=self.config.max_new_tokens,
                max_tokens=4096,
                temperature=1.0,
            ).new_with_stop_and_pad_token_ids(tokenizer)
            workflow_kwargs = {
                "reward_fn": "areal.reward.gsm8k.gsm8k_reward_fn",
                "gconfig": gconfig,
                "tokenizer": str(self.config.model_snapshot),
                "enable_thinking": False,
                "harness_seed": self.config.harness_seed,
                "harness_kind": "torch",
                "harness_hidden_size": self.config.harness_hidden_size,
            }
            submitted = 0
            for batch in dataloader:
                for item in batch:
                    self.rollout.submit(
                        item,
                        workflow=(
                            "jphrl.areal_joint_bridge_workflow."
                            "ArealJointBridgeWorkflow"
                        ),
                        workflow_kwargs=workflow_kwargs,
                        group_size=1,
                    )
                    submitted += 1
                    if submitted == 8:
                        break
                if submitted == 8:
                    break
            _require(submitted == 8, "GSM8K snapshot yielded fewer than eight tasks")
            results = self.rollout.wait(submitted, timeout=1800.0)
            _require(
                isinstance(results, list)
                and len(results) == 8
                and all(result is not None for result in results),
                "distributed rollout did not return eight successful results",
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        admission_files = tuple(admission_root.glob("rlvr-runner-admission-*.json"))
        bridge_files = tuple(bridge_root.glob("bridge-*.json"))
        _require(
            len(admission_files) == len(bridge_files) == 8,
            "rollout outputs and RLVR admission/bridge records are not one-to-one",
        )
        return admission_root

    @property
    def runtime_contract_json(self) -> str:
        _require(self.runtime_contract is not None, "runtime contract is unavailable")
        return _canonical_json(self.runtime_contract).decode("utf-8")

    def run_tuvw(
        self,
        scheduler: IntegratedSchedulerHandle,
        training_sources: Sequence[RLVRM0SourceRecords],
        multi_s_batch: ValidatedMultiSFrozenTrainingBatch,
    ) -> IntegratedStageReference:
        del scheduler
        _require(self.actor is not None, "TUVW requires the live actor")
        _require(
            len(training_sources) == 4
            and tuple(source.s_record_sha256 for source in training_sources)
            == required_v_member_claims(multi_s_batch),
            "TUVW sources differ from the ordered four-member multi-S batch",
        )
        # Imported here so CPU contract tests cannot accidentally claim an update.
        from jphrl.harness.torch_learning import (
            TorchHarnessOptimizer,
            load_torch_harness_rollout_checkpoint,
        )
        from jphrl.training.areal_distributed_policy import (
            require_live_remote_policy_candidate,
        )
        from jphrl.training.areal_policy_optimizer import (
            build_areal_external_advantage_batch,
        )

        active = multi_s_batch.joint_version
        admissions = [
            build_areal_external_advantage_batch(
                source.s_joint_credit,
                active_joint_version=active,
            )
            for source in training_sources
        ]
        checkpoints: list[Mapping[str, object]] = []
        for source in training_sources:
            checkpoint = source.runner_admission["bridge_record"]["harness"][
                "controller_checkpoint_before_decision"
            ]
            _require(isinstance(checkpoint, Mapping), "Harness rollout checkpoint is missing")
            checkpoints.append(checkpoint)
        checkpoint_hashes = {_record_sha256(checkpoint) for checkpoint in checkpoints}
        _require(
            len(checkpoint_hashes) == 1
            and all(
                checkpoint.get("record_sha256") in checkpoint_hashes
                for checkpoint in checkpoints
            ),
            "four training admissions do not share one Harness behavior checkpoint",
        )
        behavior_path = _write_new_json(
            self.config.run_root / "inputs" / "harness-behavior.json",
            checkpoints[0],
        )
        behavior_policy = load_torch_harness_rollout_checkpoint(
            checkpoints[0],
            device="cpu",
        )
        _require(
            behavior_policy.version == active.harness_controller,
            "Harness behavior checkpoint differs from active JointVersion",
        )
        trainer = TorchHarnessOptimizer(behavior_policy)
        live_policy = self.actor.run_m0_policy_candidate_step_live(
            admissions,
            multi_s_batch=multi_s_batch,
            active_joint_version=active,
            transaction_id=self.config.transaction_id,
            candidate_root=self.config.run_root / "policy-candidate",
            project_commit=self.config.project_commit,
            areal_commit=PINNED_AREAL_COMMIT,
        )
        require_live_remote_policy_candidate(live_policy)
        # Persist the process-local rollback capability immediately.  Any later
        # Harness/V/W exception is then visible to stage-machine cleanup.
        self.live_policy_candidate = live_policy
        harness_result = trainer.update_from_validated_multi_s_frozen_training_batch(
            multi_s_batch,
            transaction_id=self.config.transaction_id,
            active_joint_version=active,
            checkpoint_path=(
                self.config.run_root / "harness-candidate" / "harness-candidate.pt"
            ),
        )
        harness_receipt = harness_result.evidence.to_record()
        harness_receipt_path = _write_new_json(
            self.config.run_root
            / "harness-candidate"
            / "harness-candidate-evidence.json",
            harness_receipt,
        )
        self.harness_result = harness_result
        self.training_sources = tuple(training_sources)
        self.multi_s_batch = multi_s_batch
        self.behavior_checkpoint_path = behavior_path
        self.behavior_checkpoint_sha256 = str(checkpoints[0]["record_sha256"])
        # V/W are sealed by the dedicated continuation helper.  Keeping this call
        # isolated avoids ever substituting the legacy single-rank W verifier.
        tuv_w = self._seal_and_verify_distributed_w(
            active_joint_version=active,
            behavior_checkpoint_path=behavior_path,
            behavior_checkpoint_sha256=str(checkpoints[0]["record_sha256"]),
        )
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_TUVW_STAGE_SCHEMA,
            "stage": "tuvw",
            "input_s_record_sha256s": list(required_v_member_claims(multi_s_batch)),
            "policy_candidate_receipt_sha256": live_policy.receipt["record_sha256"],
            "harness_candidate_receipt_path": str(harness_receipt_path),
            "harness_candidate_receipt_sha256": harness_receipt["record_sha256"],
            **tuv_w,
            "evidence_scope": {
                "all_four_actor_ranks_attested": True,
                "policy_optimizer_update": True,
                "harness_optimizer_update": True,
                "joint_candidate_sealed": True,
                "exact_joint_recovery": True,
                "joint_version_publish": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        path = _write_new_json(self.config.run_root / "stages" / "tuvw.json", record)
        return IntegratedStageReference(
            stage="tuvw",
            record_path=path,
            record_sha256=str(record["record_sha256"]),
            input_s_record_sha256s=required_v_member_claims(multi_s_batch),
        )

    def _seal_and_verify_distributed_w(
        self,
        *,
        active_joint_version: object,
        behavior_checkpoint_path: Path,
        behavior_checkpoint_sha256: str,
    ) -> dict[str, object]:
        """Seal V, persist all four rank states, and run real two-branch W."""

        from jphrl.experiments.m0_joint_runner import (
            _run_harness_continuation_step,
        )
        from jphrl.joint_release import CandidateArtifact, JointReleaseStore
        from jphrl.training.joint_step import seal_joint_candidate_bundle
        from jphrl.training.production_checkpoint import (
            RuntimeCursorState,
            RuntimeTopology,
            distributed_policy_checkpoint_inputs,
            save_production_joint_checkpoint,
            verify_exact_distributed_joint_recovery,
        )
        from jphrl.trajectory.multi_s_frozen_training_batch import (
            multi_s_source_binding,
        )
        from jphrl.trajectory.schema import JointVersion

        _require(
            type(active_joint_version) is JointVersion
            and self.actor is not None
            and self.live_policy_candidate is not None
            and self.harness_result is not None
            and self.multi_s_batch is not None
            and len(self.training_sources) == 4,
            "distributed V/W live inputs are incomplete",
        )
        live_policy = self.live_policy_candidate
        harness_result = self.harness_result
        binding = multi_s_source_binding(self.multi_s_batch)
        store = JointReleaseStore(self.config.run_root / "release-store")
        _require(store.read_active() is None, "new M0 release store is already active")
        parent_policy = CandidateArtifact(
            component="policy",
            version=active_joint_version.policy,
            payload={
                "schema_version": "jph.m0-distributed-measured-parent-policy.v1",
                "joint_version_id": active_joint_version.version_id,
                "source_binding": binding,
                "training_runner_admission_sha256s": [
                    source.runner_admission_sha256 for source in self.training_sources
                ],
                "observed_policy_engine_version": self.actor.get_version(),
            },
        )
        parent_harness = CandidateArtifact(
            component="harness",
            version=active_joint_version.harness_controller,
            payload={
                "schema_version": "jph.m0-distributed-measured-parent-harness.v1",
                "joint_version_id": active_joint_version.version_id,
                "source_binding": binding,
                "harness_rollout_checkpoint_sha256": behavior_checkpoint_sha256,
            },
        )
        staged_parent = store.stage(
            joint_version=active_joint_version,
            policy=parent_policy,
            harness=parent_harness,
            expected_active_release_id=None,
        )
        parent_release = store.activate(
            release_id=staged_parent.release_id,
            expected_active_release_id=None,
        )
        harness_receipt = harness_result.evidence.to_record()

        def restore_policy_parent() -> None:
            self.actor._rollback_workers(self.config.transaction_id)

        def restore_harness_parent() -> None:
            _require(
                harness_result.candidate_policy.version
                != active_joint_version.harness_controller,
                "Harness candidate unexpectedly replaced the behavior object",
            )

        bundle = seal_joint_candidate_bundle(
            seal_root=self.config.run_root / "joint-candidate",
            transaction_journal_root=(
                self.config.run_root.parent / "joint-transaction-journal"
            ),
            project_root=repository_root(),
            policy_receipt=live_policy.receipt,
            harness_receipt=harness_receipt,
            active_joint_version=active_joint_version,
            parent_release_id=parent_release.release_id,
            macro_step_id=self.config.transaction_id,
            actor_public_version=self.actor.get_version(),
            harness_public_version=active_joint_version.harness_controller,
            restore_policy_parent=restore_policy_parent,
            restore_harness_parent=restore_harness_parent,
        )
        bundle_path = _write_new_json(
            self.config.run_root / "joint-candidate" / "bundle.json",
            bundle,
        )
        topology = RuntimeTopology(
            world_size=4,
            data_parallel_size=4,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            rank_to_device=("cuda:0",) * 4,
        )
        rank_states, policy_states, scheduler_state = (
            distributed_policy_checkpoint_inputs(
                live_policy,
                harness_policy=harness_result.candidate_policy,
            )
        )
        source_binding_sha256 = str(binding["record_sha256"])
        rollout_stream_sha256 = hashlib.sha256(
            _canonical_json(
                [
                    source.runner_admission_sha256
                    for source in self.training_sources
                ]
            )
        ).hexdigest()
        checkpoint_manifest = save_production_joint_checkpoint(
            checkpoint_root=self.config.run_root / "production-checkpoint",
            project_root=repository_root(),
            joint_candidate_bundle=bundle,
            actor=self.actor,
            harness_policy=harness_result.candidate_policy,
            topology=topology,
            rank_states=rank_states,
            policy_rank_states=policy_states,
            policy_scheduler_state=scheduler_state,
            macro_step=0,
            rollout_cursor=RuntimeCursorState(
                name="rollout",
                position=0,
                source_sha256=rollout_stream_sha256,
                pending_item_sha256=source_binding_sha256,
            ),
            dataloader_cursor=RuntimeCursorState(
                name="dataloader",
                position=0,
                source_sha256=self.multi_s_batch.record_sha256,
                pending_item_sha256=source_binding_sha256,
            ),
            live_policy_candidate=live_policy,
        )
        s_records = tuple(source.s_joint_credit for source in self.training_sources)

        def harness_continuation(policy: object, optimizer: object) -> None:
            _run_harness_continuation_step(policy, optimizer, s_records)

        live_recovery = verify_exact_distributed_joint_recovery(
            checkpoint_manifest,
            controller=self.actor,
            live_policy_candidate=live_policy,
            harness_policy=harness_result.candidate_policy,
            harness_optimizer=harness_result.candidate_optimizer.optimizer,
            current_topology=topology,
            run_harness_optimizer_step=harness_continuation,
        )
        recovery_path = _write_new_json(
            self.config.run_root
            / "production-checkpoint"
            / "w-live-distributed-recovery.json",
            live_recovery.record,
        )
        self.bundle = bundle
        self.checkpoint_manifest = checkpoint_manifest
        self.live_recovery = live_recovery
        self.release_store = store
        self.parent_release = parent_release
        return {
            "joint_candidate_bundle_path": str(bundle_path),
            "joint_candidate_bundle_sha256": bundle["record_sha256"],
            "production_checkpoint_manifest": str(checkpoint_manifest),
            "production_checkpoint_manifest_sha256": (
                live_recovery.checkpoint_manifest_sha256
            ),
            "distributed_w_record_path": str(recovery_path),
            "distributed_w_record_sha256": live_recovery.record_sha256,
        }

    def run_x(
        self,
        scheduler: IntegratedSchedulerHandle,
        holdout_sources: Sequence[RLVRM0SourceRecords],
        tuvw: IntegratedStageReference,
    ) -> IntegratedStageReference:
        del scheduler
        _require(
            len(holdout_sources) == 4
            and len(self.training_sources) == 4
            and self.actor is not None
            and self.live_policy_candidate is not None
            and self.bundle is not None
            and self.checkpoint_manifest is not None
            and self.live_recovery is not None
            and self.release_store is not None
            and self.parent_release is not None,
            "distributed X live inputs are incomplete",
        )
        from jphrl.experiments.m0_live_evaluator import (
            DistributedRealRlvrM0CandidateEvaluator,
        )
        from jphrl.training.areal_production_worker import (
            build_production_probe_output,
            materialize_areal_serving_export_pair,
        )
        from jphrl.training.candidate_acceptance import (
            build_production_candidate_artifacts,
            run_joint_candidate_acceptance,
        )

        evaluator = DistributedRealRlvrM0CandidateEvaluator(
            training_sources=self.training_sources,
            holdout_sources=holdout_sources,
        )
        gates = evaluator.acceptance_gates
        spec = _candidate_acceptance_spec(gates)
        serving_exports = materialize_areal_serving_export_pair(
            actor=self.actor,
            policy_candidate_record=self.live_policy_candidate.receipt,
            export_root=self.config.run_root / "serving-exports",
            parent_joint_version=self.training_sources[0].active_joint_version,
            candidate_joint_version=self.live_recovery.candidate_joint_version,
            live_policy_candidate=self.live_policy_candidate,
        )
        distributed_export_receipt = _strict_json(
            self.config.run_root
            / "serving-exports"
            / "distributed-serving-export.json",
            "distributed serving export receipt",
        )
        policy_current_state = (
            self.actor.attest_m0_policy_candidate_current_state(
                self.live_policy_candidate,
                distributed_serving_export_receipt=(
                    distributed_export_receipt
                ),
                candidate_serving_export_lineage_sha256=(
                    serving_exports.candidate.record_sha256
                ),
            )
        )
        policy_current_state_path = _write_new_json(
            self.config.run_root
            / "serving-exports"
            / "distributed-current-policy-state.json",
            policy_current_state,
        )
        policy_artifact, harness_artifact = build_production_candidate_artifacts(
            joint_candidate_bundle=self.bundle,
            checkpoint_manifest=self.checkpoint_manifest,
            live_serving_exports=serving_exports,
        )
        candidate_release = self.release_store.stage(
            joint_version=self.live_recovery.candidate_joint_version,
            policy=policy_artifact,
            harness=harness_artifact,
            expected_active_release_id=self.parent_release.release_id,
        )
        m0_gate_by_kind = {gate.kind: gate for gate in gates}
        joint_gate = m0_gate_by_kind["joint_safety"]
        harness_receipt = self.harness_result.evidence.to_record()
        candidate_probe_output = build_production_probe_output(
            fixture=joint_gate.fixture,
            target_release_id=candidate_release.release_id,
            target_joint_version=self.live_recovery.candidate_joint_version,
            policy_engine_version=serving_exports.candidate.policy_engine_version,
            policy_checkpoint_sha256=(
                serving_exports.candidate.source_dcp_manifest_sha256
            ),
            serving_parameter_sha256=(
                serving_exports.candidate.serving_parameter_sha256
            ),
            harness_checkpoint_sha256=str(harness_receipt["checkpoint_sha256"]),
            harness_parameter_sha256=str(
                harness_receipt["parameter_digest_after"]
            ),
        )
        candidate_kind_to_m0 = {
            "policy_heldout": "policy_heldout",
            "harness_heldout": "harness_offpolicy",
            "joint_safety": "joint_safety",
            "historical_regression": "restart_recovery",
        }
        probes: dict[str, object] = {}
        for candidate_kind, m0_kind in candidate_kind_to_m0.items():
            gate = m0_gate_by_kind[m0_kind]

            def observe(
                version: object,
                _suite: object,
                *,
                frozen_gate: object = gate,
                frozen_candidate_kind: str = candidate_kind,
            ) -> Sequence[object]:
                _require(
                    version == self.live_recovery.candidate_joint_version,
                    "X evaluator received a crossed candidate JointVersion",
                )
                observations = evaluator.observe(
                    joint_version=version,
                    gate=frozen_gate,
                    actor=self.actor,
                    harness_policy=self.live_recovery.restored_harness_policy,
                    live_policy_candidate=self.live_policy_candidate,
                    policy_current_state_attestation=policy_current_state,
                    distributed_serving_export_receipt=(
                        distributed_export_receipt
                    ),
                    live_serving_exports=serving_exports,
                )
                return _attach_joint_safety_probe(
                    observations,
                    suite_kind=frozen_candidate_kind,
                    production_probe_output=candidate_probe_output,
                )

            probes[candidate_kind] = observe
        live_acceptance = run_joint_candidate_acceptance(
            joint_candidate_bundle=self.bundle,
            checkpoint_manifest=self.checkpoint_manifest,
            live_exact_recovery=self.live_recovery,
            candidate_release_id=candidate_release.release_id,
            expected_spec=spec,
            probes=probes,
            release_store=self.release_store,
            report_root=self.config.run_root / "candidate-acceptance",
            project_root=repository_root(),
            live_serving_exports=serving_exports,
        )
        report_path = (
            self.config.run_root
            / "candidate-acceptance"
            / f"acceptance-{self.bundle['record_sha256']}.json"
        )
        _require(
            report_path.is_file() and not report_path.is_symlink(),
            "X acceptance report was not persisted",
        )
        report = live_acceptance.report
        expected_holdout_claims = tuple(
            source.s_record_sha256 for source in holdout_sources
        )
        _require(
            tuvw.stage == "tuvw"
            and set(tuvw.input_s_record_sha256s)
            == {source.s_record_sha256 for source in self.training_sources}
            and set(tuvw.input_s_record_sha256s).isdisjoint(expected_holdout_claims),
            "X holdouts overlap TUVW sources",
        )
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_X_STAGE_SCHEMA,
            "stage": "x",
            "input_s_record_sha256s": list(expected_holdout_claims),
            "training_runner_admission_sha256s": [
                source.runner_admission_sha256 for source in self.training_sources
            ],
            "holdout_runner_admission_sha256s": [
                source.runner_admission_sha256 for source in holdout_sources
            ],
            "candidate_acceptance_report_path": str(report_path),
            "candidate_acceptance_record_sha256": live_acceptance.record_sha256,
            "candidate_release_id": candidate_release.release_id,
            "joint_safety_production_probe_sha256": (
                _joint_safety_probe_sha256(report)
            ),
            "parent_serving_export_lineage_sha256": (
                serving_exports.parent.record_sha256
            ),
            "candidate_serving_export_lineage_sha256": (
                serving_exports.candidate.record_sha256
            ),
            "policy_current_state_attestation_path": str(
                policy_current_state_path
            ),
            "policy_current_state_attestation_sha256": policy_current_state[
                "record_sha256"
            ],
            "evidence_scope": {
                "four_training_admissions_frozen": True,
                "four_holdout_admissions_measured": True,
                "training_holdout_disjoint": True,
                "candidate_accepted": True,
                "all_four_actor_ranks_current_state_attested": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
                "joint_version_publish": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        path = _write_new_json(self.config.run_root / "stages" / "x.json", record)
        self.serving_exports = serving_exports
        self.candidate_release = candidate_release
        self.live_acceptance = live_acceptance
        self.acceptance_spec = spec
        self._joint_safety_fixture = joint_gate.fixture
        return IntegratedStageReference(
            stage="x",
            record_path=path,
            record_sha256=str(record["record_sha256"]),
            input_s_record_sha256s=expected_holdout_claims,
        )

    def run_y(
        self,
        scheduler: IntegratedSchedulerHandle,
        tuvw: IntegratedStageReference,
        x: IntegratedStageReference,
    ) -> IntegratedStageReference:
        del scheduler
        _require(
            self.rollout is not None
            and self.actor is not None
            and self.live_policy_candidate is not None
            and self.bundle is not None
            and self.checkpoint_manifest is not None
            and self.live_recovery is not None
            and self.serving_exports is not None
            and self.release_store is not None
            and self.parent_release is not None
            and self.candidate_release is not None
            and self.live_acceptance is not None
            and self.acceptance_spec is not None
            and self.behavior_checkpoint_path is not None
            and self.behavior_checkpoint_sha256 is not None,
            "distributed Y live inputs are incomplete",
        )
        from jphrl.training.areal_production_worker import (
            HarnessServingCheckpoint,
            PinnedArealSGLangActivationWorker,
        )
        from jphrl.training.joint_activation import (
            ProductionJointActivationController,
            ProductionProbeSpec,
            authorize_production_activation,
        )

        expected_training_claims = tuple(
            source.s_record_sha256 for source in self.training_sources
        )
        _require(
            tuvw.input_s_record_sha256s == expected_training_claims
            and x.stage == "x",
            "Y stage references differ from TUVW/X",
        )

        harness_receipt = self.harness_result.evidence.to_record()
        worker = PinnedArealSGLangActivationWorker.create(
            controller=self.rollout,
            serving_exports=self.serving_exports,
            harness_checkpoints={
                self.parent_release.release_id: HarnessServingCheckpoint(
                    path=str(self.behavior_checkpoint_path),
                    checkpoint_sha256=_file_sha256(self.behavior_checkpoint_path),
                    kind="rollout_json",
                ),
                self.candidate_release.release_id: HarnessServingCheckpoint(
                    path=str(harness_receipt["checkpoint_path"]),
                    checkpoint_sha256=str(harness_receipt["checkpoint_sha256"]),
                    kind="candidate_pt",
                ),
            },
            observation_root=self.config.run_root / "production-observations",
            parent_release_id=self.parent_release.release_id,
            candidate_release_id=self.candidate_release.release_id,
            request_timeout_seconds=180.0,
        )
        # The typed worker now owns the rollout controller.
        self.production_worker = worker
        self.rollout = None
        report = self.live_acceptance.report
        joint_suites = [
            suite
            for suite in report["critical_suites"]
            if suite["spec"]["kind"] == "joint_safety"
        ]
        _require(len(joint_suites) == 1, "Y has no unique accepted joint suite")
        fixture_sha256 = joint_suites[0]["spec"]["fixture_sha256"]
        # X's original bytes are retained by the adapter, not reconstructed from
        # their digest.
        joint_fixture = self._joint_safety_fixture
        _require(
            type(joint_fixture) is bytes
            and bool(joint_fixture)
            and
            hashlib.sha256(joint_fixture).hexdigest() == fixture_sha256,
            "Y joint fixture differs from X",
        )
        first_parent = worker.run_probe(joint_fixture)
        second_parent = worker.run_probe(joint_fixture)
        _require(
            type(first_parent) is bytes and first_parent == second_parent,
            "parent production probe is not deterministic raw bytes",
        )
        parent_probe_sha256 = hashlib.sha256(first_parent).hexdigest()
        candidate_probe_sha256 = _joint_safety_probe_sha256(report)
        probes = {
            self.parent_release.release_id: ProductionProbeSpec(
                probe_id="m0-distributed-parent-joint-safety",
                fixture=joint_fixture,
                fixture_sha256=fixture_sha256,
                expected_output_sha256=parent_probe_sha256,
            ),
            self.candidate_release.release_id: ProductionProbeSpec(
                probe_id="m0-distributed-candidate-joint-safety",
                fixture=joint_fixture,
                fixture_sha256=fixture_sha256,
                expected_output_sha256=candidate_probe_sha256,
            ),
        }
        authorization = authorize_production_activation(
            store=self.release_store,
            live_candidate_acceptance=self.live_acceptance,
            live_exact_recovery=self.live_recovery,
            expected_acceptance_spec=self.acceptance_spec,
            joint_candidate_bundle=self.bundle,
            checkpoint_manifest=self.checkpoint_manifest,
            workers=(worker,),
            probes=probes,
        )
        activation_controller = ProductionJointActivationController(
            store=self.release_store,
            workers=(worker,),
            probes=probes,
            project_root=repository_root(),
        )
        activation = activation_controller.activate(authorization)
        commit_summaries = _commit_actor_with_y_compensation(
            actor=self.actor,
            live_policy_candidate=self.live_policy_candidate,
            activation_controller=activation_controller,
            activation=activation,
            release_store=self.release_store,
            parent_release=self.parent_release,
            candidate_release=self.candidate_release,
            worker=worker,
            parent_probe=probes[self.parent_release.release_id],
        )
        live_receipt = self.live_policy_candidate.receipt
        remote_receipt = live_receipt["optimizer"]["remote_optimizer_receipt"]
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_Y_STAGE_SCHEMA,
            "stage": "y",
            "input_s_record_sha256s": list(expected_training_claims),
            "activation_id": activation.activation_id,
            "production_attestation_path": str(activation.attestation_path),
            "production_attestation_sha256": activation.attestation_sha256,
            "active_release_id": activation.active_release_id,
            "candidate_release_id": activation.candidate_release_id,
            "transaction_id": live_receipt["transaction"]["transaction_id"],
            "policy_candidate_sha256": live_receipt["record_sha256"],
            "remote_optimizer_aggregate_sha256": remote_receipt["record_sha256"],
            "actor_terminal_commits": list(commit_summaries),
            "actor_terminal_commit_receipt_sha256s": [
                summary["record_sha256"] for summary in commit_summaries
            ],
            "rollout_data_parallel_worker_ids": list(
                worker.data_parallel_worker_ids
            ),
            "evidence_scope": {
                "all_four_rollout_replicas_observed": True,
                "candidate_active": True,
                "all_four_actor_ranks_terminally_committed": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
                "joint_version_publish": True,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        path = _write_new_json(self.config.run_root / "stages" / "y.json", record)
        self.production_activation = activation
        self.actor_candidate_committed = True
        return IntegratedStageReference(
            stage="y",
            record_path=path,
            record_sha256=str(record["record_sha256"]),
            input_s_record_sha256s=expected_training_claims,
        )

    def stop_rollout(
        self,
        scheduler: IntegratedSchedulerHandle,
        placements: Sequence[M0WorkerPlacement],
        *,
        exact_session_id: int,
    ) -> IntegratedCleanupReceipt:
        _require(
            self.rollout is not None or self.production_worker is not None,
            "rollout cleanup has no owned controller",
        )
        if self.production_worker is not None:
            self.production_worker.close()
            self.production_worker = None
        else:
            self.rollout.destroy()
            self.rollout = None
        return IntegratedCleanupReceipt(
            role="rollout",
            scheduler_instance_id=scheduler.instance_id,
            exact_session_id=exact_session_id,
            worker_ids=tuple(item.worker_id for item in placements),
        )

    def stop_actor(
        self,
        scheduler: IntegratedSchedulerHandle,
        placements: Sequence[M0WorkerPlacement],
        *,
        exact_session_id: int,
    ) -> IntegratedCleanupReceipt:
        _require(self.actor is not None, "actor cleanup has no owned controller")
        actor = self.actor
        try:
            if self.live_policy_candidate is not None and not self.actor_candidate_committed:
                receipt = self.live_policy_candidate.receipt
                transaction_id = receipt["transaction"]["transaction_id"]
                actor._rollback_workers(transaction_id)
        finally:
            # Destruction must still run if a four-rank parent rollback reports
            # an error; otherwise the scheduler-owned CUDA workers leak.
            actor.destroy()
            self.actor = None
        return IntegratedCleanupReceipt(
            role="actor",
            scheduler_instance_id=scheduler.instance_id,
            exact_session_id=exact_session_id,
            worker_ids=tuple(item.worker_id for item in placements),
        )


__all__ = [
    "M0_EIGHT_GPU_REAL_ADAPTER_SCHEMA",
    "M0EightGPURealAdapterError",
    "NvidiaSMIGPUStateProvider",
    "RealEightGPUAdapterConfig",
    "RealEightGPUIntegratedAdapters",
    "ThreadedEightGPUMemoryRuntime",
    "build_distributed_actor_config",
    "build_distributed_inference_runtime_contract",
    "build_distributed_rollout_config",
    "build_distributed_server_args",
]
