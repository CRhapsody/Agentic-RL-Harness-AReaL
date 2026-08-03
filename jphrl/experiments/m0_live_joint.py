from __future__ import annotations

"""Remote-only entry point for one real RLVR M0 T -> Y transaction.

The four source admissions are created by the audited rollout launcher.  The
lowest task ID is frozen as the optimizer input and the remaining three are
held out for X.  This module does not manufacture observations or optimizer
receipts: it only assembles the real AReaL actor, Torch Harness, and pinned
SGLang production worker around those already validated records.
"""

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from jphrl.experiments.m0_joint_runner import (
    GPULaunchGuard,
    M0ActivationAssets,
    M0ArealActorSpec,
    M0JointRunConfig,
    M0JointRunnerError,
    ProductionWorkerFactory,
    RLVRM0JointUpdateRunner,
    RLVRM0SourceRecords,
    load_m0_rlvr_source_records,
)
from jphrl.experiments.m0_live_evaluator import RealRlvrM0CandidateEvaluator
from jphrl.paths import (
    assert_remote_environment,
    require_outside_repository,
    require_within_configured_root,
)
from jphrl.trajectory.schema import JointVersion

M0_LIVE_SELECTION_SCHEMA = "jph.m0-live-rlvr-selection.v1"
M0_GPU_LAUNCH_AUDIT_SCHEMA = "jph.m0-gpu-launch-audit.v1"
M0_RLVR_ADMISSION_COUNT = 4

_SECRET_NAMES = {
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


class M0LiveJointError(M0JointRunnerError):
    """Raised before a live M0 launch when deployment evidence is ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0LiveJointError(message)


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
        raise M0LiveJointError("M0 live deployment record is not canonical JSON") from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            _require(
                normalized not in _SECRET_NAMES
                and not normalized.endswith(
                    ("_api_key", "_credential", "_password", "_secret", "_token")
                ),
                f"credential field cannot enter M0 live evidence: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _write_new_json(path: Path, record: Mapping[str, object]) -> Path:
    _assert_no_secrets(record)
    payload = _canonical_json(record)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _read_strict_json(path: Path, *, label: str) -> dict[str, object]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise M0LiveJointError(f"{label} is not strict JSON") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    _assert_no_secrets(value, label)
    _canonical_json(value)
    return value


def _joint_version_from_admission(record: Mapping[str, object]) -> JointVersion:
    bridge = record.get("bridge_record")
    raw = bridge.get("joint_version") if isinstance(bridge, Mapping) else None
    _require(isinstance(raw, Mapping), "runner admission JointVersion is missing")
    _require(
        set(raw) == set(JointVersion.__dataclass_fields__)
        and all(isinstance(value, str) and bool(value) for value in raw.values()),
        "runner admission JointVersion field set differs",
    )
    try:
        return JointVersion(**dict(raw))
    except TypeError as exc:  # pragma: no cover - exact fields own the branch
        raise M0LiveJointError("runner admission JointVersion is invalid") from exc


def _source_task_id(source: RLVRM0SourceRecords) -> int:
    trace = source.runner_admission.get("episode_trace")
    task_id = trace.get("task_id") if isinstance(trace, Mapping) else None
    _require(
        isinstance(task_id, str) and task_id.isdigit(),
        "M0 runner admission task ID must be a non-negative decimal string",
    )
    return int(task_id)


@dataclass(frozen=True)
class LiveM0Selection:
    training_source: RLVRM0SourceRecords
    holdout_sources: tuple[RLVRM0SourceRecords, ...]
    harness_behavior_checkpoint: Path
    harness_behavior_record_sha256: str
    harness_behavior_file_sha256: str
    selection_record_path: Path


def prepare_live_m0_selection(
    *,
    runner_admission_dir: str | Path,
    selection_root: str | Path,
) -> LiveM0Selection:
    """Load exactly four immutable admissions and persist the frozen split."""

    source_root = require_outside_repository(runner_admission_dir)
    target_root = require_outside_repository(selection_root)
    _require(
        source_root.is_dir() and not source_root.is_symlink(),
        "RLVR runner admission directory is missing or unsafe",
    )
    _require(not target_root.exists(), "M0 selection root must be new")
    entries = tuple(sorted(source_root.iterdir()))
    _require(
        len(entries) == M0_RLVR_ADMISSION_COUNT
        and all(
            entry.is_file()
            and not entry.is_symlink()
            and entry.name.startswith("runner-admission-")
            and entry.suffix == ".json"
            for entry in entries
        ),
        "M0 requires exactly four safe runner-admission-*.json files",
    )
    first = _read_strict_json(entries[0], label="RLVR runner admission")
    active_joint_version = _joint_version_from_admission(first)
    loaded = tuple(
        load_m0_rlvr_source_records(
            runner_admission_path=entry,
            active_joint_version=active_joint_version,
        )
        for entry in entries
    )
    _require(
        all(source.active_joint_version == active_joint_version for source in loaded),
        "M0 runner admissions cross JointVersions",
    )
    ordered = tuple(
        sorted(loaded, key=lambda source: (_source_task_id(source), source.runner_admission_sha256))
    )
    task_ids = tuple(_source_task_id(source) for source in ordered)
    _require(
        len(set(task_ids)) == M0_RLVR_ADMISSION_COUNT,
        "M0 runner admission task IDs are not unique",
    )
    admission_digests = tuple(source.runner_admission_sha256 for source in ordered)
    _require(
        len(set(admission_digests)) == M0_RLVR_ADMISSION_COUNT,
        "M0 runner admissions are duplicated",
    )
    training = ordered[0]
    holdouts = ordered[1:]

    bridge = training.runner_admission["bridge_record"]
    harness = bridge.get("harness") if isinstance(bridge, Mapping) else None
    checkpoint = (
        harness.get("controller_checkpoint_before_decision")
        if isinstance(harness, Mapping)
        else None
    )
    _require(isinstance(checkpoint, Mapping), "training Harness checkpoint is missing")
    checkpoint_sha256 = checkpoint.get("record_sha256")
    binding = training.runner_admission["rlvr_pre_batch_record"].get(
        "harness_checkpoint_binding"
    )
    _require(
        isinstance(checkpoint_sha256, str)
        and len(checkpoint_sha256) == 64
        and checkpoint_sha256 == _record_sha256(checkpoint)
        and isinstance(binding, Mapping)
        and binding.get("checkpoint_sha256") == checkpoint_sha256,
        "training Harness checkpoint digest differs from the RLVR binding",
    )
    try:
        from jphrl.harness.torch_learning import (
            TorchHarnessPolicy,
            load_torch_harness_rollout_checkpoint,
        )
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote dep
        raise M0LiveJointError("Torch Harness is unavailable") from exc
    policy = load_torch_harness_rollout_checkpoint(checkpoint, device="cpu")
    _require(
        type(policy) is TorchHarnessPolicy
        and policy.version == active_joint_version.harness_controller,
        "training Harness checkpoint differs from the active JointVersion",
    )

    target_root.mkdir(parents=True, mode=0o700)
    os.chmod(target_root, 0o700)
    checkpoint_path = _write_new_json(
        target_root / "training-harness-behavior.json",
        checkpoint,
    )
    checkpoint_file_sha256 = _file_sha256(checkpoint_path)
    selection_record: dict[str, object] = {
        "schema_version": M0_LIVE_SELECTION_SCHEMA,
        "joint_version": asdict(active_joint_version),
        "joint_version_id": active_joint_version.version_id,
        "selection_rule": "lowest-numeric-task-id-trains-remaining-three-held-out-v1",
        "training": {
            "task_id": task_ids[0],
            "runner_admission_sha256": training.runner_admission_sha256,
        },
        "holdouts": [
            {
                "task_id": _source_task_id(source),
                "runner_admission_sha256": source.runner_admission_sha256,
            }
            for source in holdouts
        ],
        "harness_behavior_checkpoint": {
            "path": str(checkpoint_path),
            "record_sha256": checkpoint_sha256,
            "file_sha256": checkpoint_file_sha256,
        },
        "evidence_scope": {
            "four_real_rlvr_admissions_validated": True,
            "training_holdout_disjoint": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    selection_record["record_sha256"] = _record_sha256(selection_record)
    selection_path = _write_new_json(target_root / "selection.json", selection_record)
    return LiveM0Selection(
        training_source=training,
        holdout_sources=holdouts,
        harness_behavior_checkpoint=checkpoint_path,
        harness_behavior_record_sha256=str(checkpoint_sha256),
        harness_behavior_file_sha256=checkpoint_file_sha256,
        selection_record_path=selection_path,
    )


def _run_nvidia_smi(arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M0LiveJointError("cannot read live NVIDIA GPU state") from exc
    return result.stdout.strip()


@dataclass
class LiveM0GPULaunchGuard:
    """Persist and enforce a fresh resource snapshot before both GPU launches."""

    physical_gpu_id: int
    audit_root: str | Path
    max_used_memory_mib: int = 10240
    min_free_memory_mib: int = 65536

    def __post_init__(self) -> None:
        _require(self.physical_gpu_id >= 0, "physical GPU ID must be non-negative")
        _require(
            self.max_used_memory_mib >= 0 and self.min_free_memory_mib > 0,
            "GPU headroom thresholds are invalid",
        )
        require_outside_repository(self.audit_root)

    def __call__(self, phase: str) -> None:
        _require(
            phase in {"training-actor", "production-sglang"},
            "unknown M0 GPU launch phase",
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        _require(
            visible == str(self.physical_gpu_id),
            "CUDA_VISIBLE_DEVICES differs from the locked physical GPU",
        )
        raw_memory = _run_nvidia_smi(
            (
                "-i",
                str(self.physical_gpu_id),
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            )
        )
        fields = [field.strip() for field in raw_memory.split(",")]
        _require(
            len(fields) == 2 and all(field.isdigit() for field in fields),
            "live GPU memory snapshot is invalid",
        )
        used_mib, free_mib = (int(field) for field in fields)
        raw_processes = _run_nvidia_smi(
            (
                "-i",
                str(self.physical_gpu_id),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        )
        processes: list[dict[str, object]] = []
        for line in raw_processes.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",", 2)]
            _require(
                len(parts) == 3 and parts[0].isdigit(),
                "live GPU process snapshot is invalid",
            )
            memory = parts[2]
            _require(memory.isdigit(), "GPU process memory is invalid")
            processes.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "used_memory_mib": int(memory),
                }
            )
        allowed_pids: set[int] = set()
        if phase == "production-sglang":
            allowed_pids.add(os.getpid())
        unexpected = [
            process for process in processes if process["pid"] not in allowed_pids
        ]
        passed = (
            used_mib <= self.max_used_memory_mib
            and free_mib >= self.min_free_memory_mib
            and not unexpected
            and (phase != "training-actor" or not processes)
        )
        record: dict[str, object] = {
            "schema_version": M0_GPU_LAUNCH_AUDIT_SCHEMA,
            "phase": phase,
            "physical_gpu_id": self.physical_gpu_id,
            "memory_used_mib": used_mib,
            "memory_free_mib": free_mib,
            "max_used_memory_mib": self.max_used_memory_mib,
            "min_free_memory_mib": self.min_free_memory_mib,
            "current_process_pid": os.getpid(),
            "compute_processes": processes,
            "unexpected_compute_process_count": len(unexpected),
            "passed": passed,
            "evidence_scope": {
                "live_gpu_memory_observed": True,
                "live_compute_processes_observed": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        audit_root = require_outside_repository(self.audit_root)
        _write_new_json(audit_root / f"{phase}.json", record)
        _require(passed, f"GPU {self.physical_gpu_id} is not safe for {phase}")


def _recorded_server_args(source: RLVRM0SourceRecords) -> dict[str, object]:
    try:
        runtime = source.runner_admission["bridge_record"]["policy_binding"][
            "inference_runtime_contract"
        ]
        server_args = runtime["fixed"]["server_args"]
    except (KeyError, TypeError) as exc:
        raise M0LiveJointError("recorded rollout SGLang arguments are missing") from exc
    _require(isinstance(server_args, Mapping), "recorded SGLang arguments are invalid")
    value = deepcopy(dict(server_args))
    _assert_no_secrets(value, "recorded_server_args")
    _canonical_json(value)
    _require(
        float(value.get("mem_fraction_static", -1.0))
        == source.rollout_sglang_mem_fraction_static,
        "recorded SGLang memory fraction differs from the runner source",
    )
    return value


def build_live_production_worker_factory(
    *,
    selection: LiveM0Selection,
    artifact_root: str | Path,
    physical_gpu_id: int,
    admin_api_key: str,
    experiment_name: str,
    trial_name: str,
) -> ProductionWorkerFactory:
    """Build Y's factory without serializing the in-memory admin credential."""

    runtime_root = require_outside_repository(Path(artifact_root) / "production-runtime")
    _require(
        isinstance(admin_api_key, str) and len(admin_api_key) >= 32,
        "M0 production admin API key is missing or too short",
    )
    _require(physical_gpu_id >= 0, "physical GPU ID must be non-negative")
    original_server_args = _recorded_server_args(selection.training_source)

    def create(assets: M0ActivationAssets) -> Sequence[object]:
        _require(
            type(assets) is M0ActivationAssets,
            "production factory received untyped activation assets",
        )
        _require(
            assets.parent_joint_version == selection.training_source.active_joint_version
            and assets.parent_harness_checkpoint_sha256
            == selection.harness_behavior_record_sha256
            and _file_sha256(selection.harness_behavior_checkpoint)
            == selection.harness_behavior_file_sha256,
            "production parent assets differ from the frozen M0 selection",
        )
        candidate_checkpoint = require_within_configured_root(
            assets.candidate_harness_checkpoint
        )
        _require(
            candidate_checkpoint.is_file()
            and not candidate_checkpoint.is_symlink()
            and _file_sha256(candidate_checkpoint)
            == assets.candidate_harness_checkpoint_sha256,
            "production candidate Harness checkpoint differs from U",
        )
        parent_export = assets.serving_exports.parent.serving_export_path
        server_args = deepcopy(original_server_args)
        server_args["model_path"] = parent_export
        server_args["tokenizer_path"] = parent_export
        _assert_no_secrets(server_args, "production_server_args")
        _require(
            float(server_args.get("mem_fraction_static", -1.0))
            == assets.recorded_rollout_sglang_mem_fraction_static
            and 0.28
            <= float(server_args["mem_fraction_static"])
            <= 0.30,
            "production memory fraction left the recorded M0 envelope",
        )
        try:
            from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec
            from areal.infra.scheduler.local import LocalScheduler
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            raise M0LiveJointError("pinned AReaL production controller is unavailable") from exc
        from jphrl.training.areal_production_worker import (
            HarnessServingCheckpoint,
            launch_pinned_areal_sglang_activation_worker,
        )

        fileroot = runtime_root / "controller"
        name_resolve_root = runtime_root / "name-resolve"
        log_root = runtime_root / "logs"
        for path in (fileroot, name_resolve_root, log_root):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
        controller_config = InferenceEngineConfig(
            experiment_name=experiment_name,
            trial_name=trial_name,
            fileroot=str(fileroot),
            max_concurrent_rollouts=1,
            consumer_batch_size=1,
            max_head_offpolicyness=0,
            enable_rollout_tracing=False,
            check_trajectory_format=False,
            tokenizer_path=parent_export,
            dump_to_file=False,
            setup_timeout=900.0,
            workers_ready_timeout=180.0,
            request_timeout=120.0,
            request_retries=1,
            scheduling_spec=(
                SchedulingSpec(
                    gpu=1,
                    cpu=8,
                    mem=32,
                    cmd="python -m areal.v2.inference_service.guard",
                ),
            ),
            backend="sglang:d1",
            _version="v2",
            model=parent_export,
            admin_api_key=admin_api_key,
        )
        scheduler = LocalScheduler(
            gpu_devices=[physical_gpu_id],
            log_dir=str(log_root),
            startup_timeout=180.0,
            experiment_name=experiment_name,
            trial_name=trial_name,
            fileroot=str(fileroot),
            name_resolve_type="nfs",
            nfs_record_root=str(name_resolve_root),
        )
        harness_checkpoints = {
            assets.parent_release.release_id: HarnessServingCheckpoint(
                path=str(selection.harness_behavior_checkpoint),
                checkpoint_sha256=selection.harness_behavior_file_sha256,
                kind="rollout_json",
            ),
            assets.candidate_release.release_id: HarnessServingCheckpoint(
                path=str(candidate_checkpoint),
                checkpoint_sha256=assets.candidate_harness_checkpoint_sha256,
                kind="candidate_pt",
            ),
        }
        worker = launch_pinned_areal_sglang_activation_worker(
            controller_config=controller_config,
            scheduler=scheduler,
            server_args=server_args,
            serving_exports=assets.serving_exports,
            harness_checkpoints=harness_checkpoints,
            observation_root=runtime_root / "observations",
            parent_release_id=assets.parent_release.release_id,
            candidate_release_id=assets.candidate_release.release_id,
            recorded_mem_fraction_static=(
                assets.recorded_rollout_sglang_mem_fraction_static
            ),
            request_timeout_seconds=120.0,
        )
        return (worker,)

    return create


def run_live_m0_joint(
    *,
    runner_admission_dir: str | Path,
    selection_root: str | Path,
    artifact_root: str | Path,
    model_path: str,
    areal_root: str,
    project_commit: str,
    transaction_id: str,
    macro_step: int,
    physical_gpu_id: int,
    admin_api_key: str,
    experiment_name: str,
    trial_name: str,
) -> object:
    selection = prepare_live_m0_selection(
        runner_admission_dir=runner_admission_dir,
        selection_root=selection_root,
    )
    evaluator = RealRlvrM0CandidateEvaluator(
        training_source=selection.training_source,
        holdout_sources=selection.holdout_sources,
    )
    actor_spec = M0ArealActorSpec(
        model_path=model_path,
        experiment_name=experiment_name,
        trial_name=trial_name,
        learning_rate=1e-6,
        dtype="bfloat16",
        optimizer_dtype="float32",
        attention_implementation="flash_attention_2",
        gradient_checkpointing=True,
        max_new_tokens=512,
    )
    run_config = M0JointRunConfig(
        artifact_root=str(artifact_root),
        project_commit=project_commit,
        areal_root=areal_root,
        transaction_id=transaction_id,
        macro_step=macro_step,
        rollout_sglang_mem_fraction_static=0.29,
        max_new_gpu_memory_gib=26.0,
    )
    factory = build_live_production_worker_factory(
        selection=selection,
        artifact_root=artifact_root,
        physical_gpu_id=physical_gpu_id,
        admin_api_key=admin_api_key,
        experiment_name=experiment_name,
        trial_name=trial_name,
    )
    gpu_guard: GPULaunchGuard = LiveM0GPULaunchGuard(
        physical_gpu_id=physical_gpu_id,
        audit_root=Path(artifact_root) / "gpu-launch-audits",
    )
    runner = RLVRM0JointUpdateRunner(
        source=selection.training_source,
        actor_spec=actor_spec,
        run_config=run_config,
        harness_behavior_checkpoint=selection.harness_behavior_checkpoint,
        acceptance_gates=evaluator.acceptance_gates,
        evaluator=evaluator,
        production_worker_factory=factory,
        gpu_launch_guard=gpu_guard,
    )
    return runner.run()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-admission-dir", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--areal-root", required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--transaction-id", required=True)
    parser.add_argument("--macro-step", required=True, type=int)
    parser.add_argument("--physical-gpu-id", required=True, type=int)
    parser.add_argument("--experiment-name", default="jph-m0-live-joint")
    parser.add_argument("--trial-name", required=True)
    args = parser.parse_args(argv)
    assert_remote_environment()
    admin_api_key = os.environ.get("JPH_AREAL_ADMIN_API_KEY", "")
    result = run_live_m0_joint(
        runner_admission_dir=args.runner_admission_dir,
        selection_root=args.selection_root,
        artifact_root=args.artifact_root,
        model_path=args.model_path,
        areal_root=args.areal_root,
        project_commit=args.project_commit,
        transaction_id=args.transaction_id,
        macro_step=args.macro_step,
        physical_gpu_id=args.physical_gpu_id,
        admin_api_key=admin_api_key,
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
    )
    print(
        json.dumps(
            {
                "artifact_root": str(result.artifact_root),
                "summary_path": str(result.summary_path),
                "active_release_id": result.active_release_id,
                "candidate_joint_version_id": (
                    result.candidate_joint_version.version_id
                ),
                "production_attestation_path": str(
                    result.production_attestation_path
                ),
                "production_attestation_sha256": (
                    result.production_attestation_sha256
                ),
                "production_worker_cleanup_path": str(
                    result.production_worker_cleanup_path
                ),
                "peak_gpu_memory_gib": result.peak_gpu_memory_gib,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
