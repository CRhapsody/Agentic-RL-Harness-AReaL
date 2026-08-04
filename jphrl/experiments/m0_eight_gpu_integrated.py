from __future__ import annotations

"""Fail-closed orchestration contract for the formal eight-GPU M0 run.

This module owns selection, placement ordering, memory gates, lifecycle, and
receipt references.  It deliberately does not implement T/U/V/W/X/Y and does
not manufacture evidence for them.  A production execution is impossible
until an explicit adapter implementing :class:`IntegratedM0Adapters` is
provided by the real AReaL integration.
"""

import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from jphrl.paths import require_outside_repository, require_within_configured_root
from jphrl.trajectory.multi_s_frozen_training_batch import (
    ValidatedMultiSFrozenTrainingBatch,
    persist_multi_s_frozen_training_batch,
    required_v_member_claims,
)
from jphrl.trajectory.schema import JointVersion

from .m0_eight_gpu_topology import (
    M0EightGPUOwnershipLedger,
    M0EightGPUTopology,
    M0WorkerPlacement,
)
from .m0_joint_runner import RLVRM0SourceRecords, load_m0_rlvr_source_records


M0_EIGHT_GPU_INTEGRATED_PREFLIGHT_SCHEMA = (
    "jph.m0-eight-gpu-integrated-preflight.v1"
)
M0_EIGHT_GPU_ADMISSION_SELECTION_SCHEMA = (
    "jph.m0-eight-gpu-admission-selection.v1"
)
M0_EIGHT_GPU_MEMORY_GATE_SCHEMA = "jph.m0-eight-gpu-memory-gate.v1"
M0_EIGHT_GPU_MEMORY_AUDIT_SCHEMA = "jph.m0-eight-gpu-memory-audit.v1"
M0_EIGHT_GPU_STAGE_MACHINE_SCHEMA = "jph.m0-eight-gpu-stage-machine.v1"
PINNED_AREAL_COMMIT = "fee938eada49208a5aabdbc1095730a13076a349"
EXPECTED_GPU_IDS = tuple(range(8))
TRAINING_ADMISSION_COUNT = 4
HOLDOUT_ADMISSION_COUNT = 4
TOTAL_ADMISSION_COUNT = 8

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
_STAGE_ORDER = (
    "created",
    "preflight-gate-passed",
    "immediately-before-scheduler-gate-passed",
    "watchdog-started",
    "scheduler-created",
    "actor-started",
    "rollout-started",
    "eight-admissions-frozen",
    "tuvw-completed",
    "x-completed",
    "y-completed",
    "rollout-stopped",
    "actor-stopped",
    "watchdog-stopped",
    "completed",
)


class M0EightGPUIntegratedError(RuntimeError):
    """Raised when the formal eight-GPU orchestration cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0EightGPUIntegratedError(message)


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
        raise M0EightGPUIntegratedError(
            "integrated M0 evidence is not finite canonical JSON"
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_no_secrets(value: object, path: str = "record") -> None:
    forbidden_suffixes = (
        "_access_token",
        "_api_key",
        "_authorization",
        "_cookie",
        "_credential",
        "_password",
        "_secret",
        "_token",
    )
    credential_prefixes = ("bearer ", "github_pat_", "ghp_", "sk-")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            _require(
                normalized not in _SECRET_NAMES
                and not normalized.endswith(forbidden_suffixes),
                f"credential field cannot enter integrated M0 evidence: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str):
        _require(
            not value.strip().lower().startswith(credential_prefixes),
            f"credential-looking value cannot enter integrated M0 evidence: {path}",
        )


def _write_new_json(path: Path, record: Mapping[str, object]) -> Path:
    target = require_outside_repository(path)
    _assert_no_secrets(record)
    payload = _canonical_json(record) + b"\n"
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(target, flags, 0o600)
        created = True
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "integrated M0 record write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(target, 0o600)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


def _read_strict_json(path: Path, label: str) -> dict[str, object]:
    source = require_outside_repository(path)
    _require(source.is_file() and not source.is_symlink(), f"{label} is unsafe")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise M0EightGPUIntegratedError(f"{label} is not strict JSON") from exc
    _require(isinstance(value, dict), f"{label} must contain one object")
    _assert_no_secrets(value, label)
    _canonical_json(value)
    return value


@dataclass(frozen=True)
class RepositoryState:
    commit: str
    clean: bool


class RepositoryProbe(Protocol):
    def inspect(self, repository: Path) -> RepositoryState:
        """Return the full commit and porcelain cleanliness of one repository."""


class SubprocessRepositoryProbe:
    def inspect(self, repository: Path) -> RepositoryState:
        try:
            commit = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise M0EightGPUIntegratedError(
                f"cannot inspect repository {repository}"
            ) from exc
        return RepositoryState(commit=commit, clean=not bool(status))


@dataclass(frozen=True)
class IntegratedLaunchPreflight:
    run_root: Path
    project_commit: str
    areal_commit: str
    record_path: Path


def freeze_integrated_launch_preflight(
    *,
    jph_root: str | Path,
    project_repository: str | Path,
    areal_repository: str | Path,
    run_root: str | Path,
    expected_project_commit: str,
    tmux_connection: str,
    repository_probe: RepositoryProbe | None = None,
    runtime_import_paths: Mapping[str, str | Path] | None = None,
) -> IntegratedLaunchPreflight:
    """Freeze clean pinned sources and a new project-external tmux run root."""

    root = require_within_configured_root(jph_root)
    project = require_within_configured_root(project_repository)
    areal = require_within_configured_root(areal_repository)
    target = require_outside_repository(run_root)
    _require(root.is_dir() and not root.is_symlink(), "JPH_ROOT is missing or unsafe")
    _require(
        project.is_dir()
        and not project.is_symlink()
        and areal.is_dir()
        and not areal.is_symlink(),
        "project or AReaL repository is missing or unsafe",
    )
    _require(
        target != root and root in target.parents,
        "integrated run root must be a dedicated path below JPH_ROOT",
    )
    formal_root = root / "artifacts" / "m0-eight-gpu-integrated"
    _require(
        formal_root in target.parents,
        "formal integrated run root must use its own non-holder artifact namespace",
    )
    _require(not target.exists(), "integrated run root must be new")
    _require(
        isinstance(tmux_connection, str) and bool(tmux_connection),
        "formal integrated M0 must run inside tmux",
    )
    _require(
        _is_git_sha1(expected_project_commit),
        "expected project commit must be a full lowercase Git SHA-1",
    )
    probe = repository_probe or SubprocessRepositoryProbe()
    project_state = probe.inspect(project)
    areal_state = probe.inspect(areal)
    _require(
        project_state.clean and project_state.commit == expected_project_commit,
        "project repository is dirty or differs from the expected commit",
    )
    _require(
        areal_state.clean and areal_state.commit == PINNED_AREAL_COMMIT,
        "AReaL repository is dirty or differs from the pinned commit",
    )
    if runtime_import_paths is None:
        resolved_imports: dict[str, Path] = {}
        for module_name in ("jphrl", "areal"):
            try:
                module = importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError) as exc:
                raise M0EightGPUIntegratedError(
                    f"cannot import formal runtime module {module_name}"
                ) from exc
            module_file = getattr(module, "__file__", None)
            _require(
                isinstance(module_file, str) and bool(module_file),
                f"formal runtime module {module_name} has no source file",
            )
            resolved_imports[module_name] = Path(module_file).resolve()
    else:
        _require(
            set(runtime_import_paths) == {"jphrl", "areal"},
            "formal runtime import path coverage differs",
        )
        resolved_imports = {
            name: Path(path).expanduser().resolve()
            for name, path in runtime_import_paths.items()
        }
    expected_import_roots = {"jphrl": project, "areal": areal}
    for module_name, module_file in resolved_imports.items():
        checkout = expected_import_roots[module_name]
        _require(
            module_file.is_file()
            and not module_file.is_symlink()
            and checkout in module_file.parents,
            f"formal runtime module {module_name} was imported outside its checked checkout",
        )
    target.mkdir(mode=0o700, parents=True)
    os.chmod(target, 0o700)
    record: dict[str, object] = {
        "schema_version": M0_EIGHT_GPU_INTEGRATED_PREFLIGHT_SCHEMA,
        "project": {"path": str(project), "commit": project_state.commit},
        "areal": {"path": str(areal), "commit": areal_state.commit},
        "runtime_imports": {
            name: {"module_file": str(resolved_imports[name]), "checkout": str(root_path)}
            for name, root_path in expected_import_roots.items()
        },
        "run_root": str(target),
        "tmux_required": True,
        "gpu_ids": list(EXPECTED_GPU_IDS),
        "holder_control_reused": False,
        "evidence_scope": {
            "source_preflight_validated": True,
            "gpu_execution": False,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
            "joint_version_publish": False,
        },
    }
    record["record_sha256"] = _record_sha256(record)
    record_path = _write_new_json(target / "launch-preflight.json", record)
    return IntegratedLaunchPreflight(
        run_root=target,
        project_commit=project_state.commit,
        areal_commit=areal_state.commit,
        record_path=record_path,
    )


def _joint_version_from_runner_admission(record: Mapping[str, object]) -> JointVersion:
    bridge = record.get("bridge_record")
    raw = bridge.get("joint_version") if isinstance(bridge, Mapping) else None
    _require(isinstance(raw, Mapping), "RLVR admission JointVersion is missing")
    _require(
        set(raw) == set(JointVersion.__dataclass_fields__)
        and all(isinstance(value, str) and bool(value) for value in raw.values()),
        "RLVR admission JointVersion fields differ",
    )
    try:
        return JointVersion(**dict(raw))
    except TypeError as exc:
        raise M0EightGPUIntegratedError("RLVR admission JointVersion is invalid") from exc


def _source_task_id(source: RLVRM0SourceRecords) -> int:
    trace = source.runner_admission.get("episode_trace")
    task_id = trace.get("task_id") if isinstance(trace, Mapping) else None
    _require(
        isinstance(task_id, str) and task_id.isdigit(),
        "RLVR admission task ID must be a non-negative decimal string",
    )
    return int(task_id)


@dataclass(frozen=True)
class EightGPUAdmissionSelection:
    active_joint_version: JointVersion
    training_sources: tuple[RLVRM0SourceRecords, ...]
    holdout_sources: tuple[RLVRM0SourceRecords, ...]
    training_s_paths: tuple[Path, ...]
    multi_s_batch_path: Path
    multi_s_batch: ValidatedMultiSFrozenTrainingBatch
    selection_record_path: Path


def prepare_eight_gpu_admission_selection(
    *,
    runner_admission_dir: str | Path,
    selection_root: str | Path,
) -> EightGPUAdmissionSelection:
    """Freeze exactly eight real admissions into four training and four X-only."""

    source_root = require_outside_repository(runner_admission_dir)
    target_root = require_outside_repository(selection_root)
    _require(
        source_root.is_dir() and not source_root.is_symlink(),
        "eight-GPU RLVR admission directory is missing or unsafe",
    )
    _require(not target_root.exists(), "eight-GPU selection root must be new")
    entries = tuple(sorted(source_root.iterdir()))
    _require(
        len(entries) == TOTAL_ADMISSION_COUNT
        and all(
            entry.is_file()
            and not entry.is_symlink()
            and entry.suffix == ".json"
            and entry.name.startswith(("runner-admission-", "rlvr-runner-admission-"))
            for entry in entries
        ),
        "formal M0 requires exactly eight safe RLVR runner-admission JSON files",
    )
    first = _read_strict_json(entries[0], "RLVR runner admission")
    active = _joint_version_from_runner_admission(first)
    loaded = tuple(
        load_m0_rlvr_source_records(
            runner_admission_path=entry,
            active_joint_version=active,
        )
        for entry in entries
    )
    _require(
        all(source.active_joint_version == active for source in loaded),
        "eight RLVR admissions do not share one complete JointVersion",
    )
    ordered = tuple(
        sorted(
            loaded,
            key=lambda source: (
                _source_task_id(source),
                source.runner_admission_sha256,
            ),
        )
    )
    task_ids = tuple(_source_task_id(source) for source in ordered)
    admission_sha256s = tuple(source.runner_admission_sha256 for source in ordered)
    s_sha256s = tuple(source.s_record_sha256 for source in ordered)
    episode_sha256s = tuple(source.episode_trace_sha256 for source in ordered)
    _require(
        len(set(task_ids)) == TOTAL_ADMISSION_COUNT
        and len(set(admission_sha256s)) == TOTAL_ADMISSION_COUNT
        and len(set(s_sha256s)) == TOTAL_ADMISSION_COUNT
        and len(set(episode_sha256s)) == TOTAL_ADMISSION_COUNT,
        "eight RLVR admissions contain duplicate task/admission/episode/S identity",
    )
    training = ordered[:TRAINING_ADMISSION_COUNT]
    holdouts = ordered[TRAINING_ADMISSION_COUNT:]
    _require(
        len(training) == TRAINING_ADMISSION_COUNT
        and len(holdouts) == HOLDOUT_ADMISSION_COUNT,
        "formal M0 training/holdout split differs from four plus four",
    )

    target_root.mkdir(mode=0o700, parents=True)
    os.chmod(target_root, 0o700)
    training_s_paths: list[Path] = []
    for index, source in enumerate(training):
        path = target_root / "training-s" / (
            f"member-{index:02d}-{source.s_record_sha256}.json"
        )
        training_s_paths.append(_write_new_json(path, source.s_joint_credit))
    batch_path = target_root / "training-batch" / "multi-s.json"
    batch = persist_multi_s_frozen_training_batch(
        training_s_paths,
        batch_path,
        active_joint_version=active,
    )
    training_s_claims = tuple(source.s_record_sha256 for source in training)
    holdout_s_claims = tuple(source.s_record_sha256 for source in holdouts)
    _require(
        set(required_v_member_claims(batch)) == set(training_s_claims)
        and set(training_s_claims).isdisjoint(holdout_s_claims),
        "multi-S members differ from training admissions or overlap X holdouts",
    )

    by_digest = {
        source.runner_admission_sha256: entry
        for source, entry in zip(loaded, entries)
    }
    selection_record: dict[str, object] = {
        "schema_version": M0_EIGHT_GPU_ADMISSION_SELECTION_SCHEMA,
        "joint_version": asdict(active),
        "joint_version_id": active.version_id,
        "selection_rule": (
            "numeric-task-id-then-admission-sha256-first-four-train-last-four-x-v1"
        ),
        "admissions": [
            {
                "ordinal": ordinal,
                "role": "training" if ordinal < TRAINING_ADMISSION_COUNT else "x-holdout",
                "task_id": _source_task_id(source),
                "source_path": str(by_digest[source.runner_admission_sha256]),
                "source_file_sha256": _file_sha256(
                    by_digest[source.runner_admission_sha256]
                ),
                "runner_admission_sha256": source.runner_admission_sha256,
                "episode_trace_sha256": source.episode_trace_sha256,
                "s_record_sha256": source.s_record_sha256,
            }
            for ordinal, source in enumerate(ordered)
        ],
        "training_batch": {
            "path": str(batch_path),
            "record_sha256": batch.record_sha256,
            "aggregate_sha256": batch.aggregate_sha256,
            "ordered_s_record_sha256s": list(required_v_member_claims(batch)),
        },
        "holdout_s_record_sha256s": list(holdout_s_claims),
        "evidence_scope": {
            "eight_real_rlvr_admissions_validated": True,
            "training_holdout_disjoint": True,
            "multi_s_training_batch_validated": True,
            "holdouts_used_for_training": False,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
            "joint_version_publish": False,
        },
    }
    selection_record["record_sha256"] = _record_sha256(selection_record)
    selection_path = _write_new_json(target_root / "selection.json", selection_record)
    return EightGPUAdmissionSelection(
        active_joint_version=active,
        training_sources=training,
        holdout_sources=holdouts,
        training_s_paths=tuple(training_s_paths),
        multi_s_batch_path=batch_path,
        multi_s_batch=batch,
        selection_record_path=selection_path,
    )


@dataclass(frozen=True)
class GPUProcessObservation:
    pid: int
    session_id: int
    uid: int
    user: str
    process_name: str
    used_memory_mib: int

    def validate(self) -> None:
        _require(
            type(self.pid) is int
            and self.pid > 1
            and type(self.session_id) is int
            and self.session_id > 1
            and type(self.uid) is int
            and self.uid >= 0
            and isinstance(self.user, str)
            and bool(self.user)
            and isinstance(self.process_name, str)
            and bool(self.process_name)
            and type(self.used_memory_mib) is int
            and self.used_memory_mib >= 0,
            "GPU process observation is invalid",
        )


@dataclass(frozen=True)
class GPUObservation:
    gpu_id: int
    memory_used_mib: int
    memory_free_mib: int
    processes: tuple[GPUProcessObservation, ...] = ()

    def validate(self) -> None:
        _require(
            type(self.gpu_id) is int
            and self.gpu_id in EXPECTED_GPU_IDS
            and type(self.memory_used_mib) is int
            and self.memory_used_mib >= 0
            and type(self.memory_free_mib) is int
            and self.memory_free_mib >= 0,
            "GPU memory observation is invalid",
        )
        for process in self.processes:
            _require(
                type(process) is GPUProcessObservation,
                "GPU process observation must be typed",
            )
            process.validate()


class GPUStateProvider(Protocol):
    def snapshot(self) -> Sequence[GPUObservation]:
        """Return one fresh observation for each physical GPU 0 through 7."""


@dataclass
class EightGPUMemoryObservation:
    provider: GPUStateProvider
    audit_root: str | Path
    _baseline_used_mib: dict[int, int] | None = None
    _gate_count: int = 0
    _sample_count: int = 0
    _peak_used_mib: dict[int, int] | None = None
    _peak_adjusted_delta_mib: dict[int, int] | None = None
    _baseline_observations: dict[int, GPUObservation] | None = None

    def __post_init__(self) -> None:
        require_outside_repository(self.audit_root)

    def _snapshot(self) -> tuple[GPUObservation, ...]:
        raw = tuple(self.provider.snapshot())
        _require(
            len(raw) == 8 and all(type(item) is GPUObservation for item in raw),
            "GPU provider must return eight typed observations",
        )
        ordered = tuple(sorted(raw, key=lambda item: item.gpu_id))
        _require(
            tuple(item.gpu_id for item in ordered) == EXPECTED_GPU_IDS,
            "GPU provider did not cover physical GPUs 0 through 7 exactly once",
        )
        for item in ordered:
            item.validate()
        return ordered

    def gate(self, stage: str) -> Mapping[str, object]:
        expected_stage = (
            "preflight" if self._gate_count == 0 else "immediately-before-scheduler"
        )
        _require(
            self._gate_count < 2 and stage == expected_stage,
            "eight-GPU memory gates must run exactly twice in order",
        )
        observations = self._snapshot()
        process_changes: list[dict[str, object]] = []
        if self._baseline_observations is not None:
            for item in observations:
                baseline = {
                    process.pid: process
                    for process in self._baseline_observations[item.gpu_id].processes
                }
                for process in item.processes:
                    previous = baseline.get(process.pid)
                    if previous is None:
                        process_changes.append(
                            {"gpu_id": item.gpu_id, "kind": "new", **asdict(process)}
                        )
                    elif (
                        process.session_id != previous.session_id
                        or process.uid != previous.uid
                        or process.user != previous.user
                        or process.process_name != previous.process_name
                    ):
                        process_changes.append(
                            {
                                "gpu_id": item.gpu_id,
                                "kind": "identity-changed",
                                "baseline": asdict(previous),
                                "observed": asdict(process),
                            }
                        )
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_MEMORY_GATE_SCHEMA,
            "stage": stage,
            "gpu_ids": list(EXPECTED_GPU_IDS),
            "observation_policy": "no-fixed-memory-limit-v1",
            "fixed_memory_limit_mib": None,
            "other_user_process_action": "observe-only",
            "observations": [
                {
                    "gpu_id": item.gpu_id,
                    "memory_used_mib": item.memory_used_mib,
                    "memory_free_mib": item.memory_free_mib,
                    "compute_processes": [asdict(process) for process in item.processes],
                }
                for item in observations
            ],
            "baseline_process_changes": process_changes,
            "passed": True,
            "evidence_scope": {
                "live_gpu_state_observed": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        _write_new_json(
            require_outside_repository(self.audit_root) / f"gate-{stage}.json",
            record,
        )
        if self._baseline_used_mib is None:
            self._baseline_used_mib = {
                item.gpu_id: item.memory_used_mib for item in observations
            }
            self._peak_used_mib = dict(self._baseline_used_mib)
            self._peak_adjusted_delta_mib = {
                gpu_id: 0 for gpu_id in EXPECTED_GPU_IDS
            }
            self._baseline_observations = {
                item.gpu_id: item for item in observations
            }
        self._gate_count += 1
        return record

    def watchdog_sample(self, *, exact_session_id: int) -> Mapping[str, object]:
        _require(self._gate_count == 2, "watchdog requires both memory gates")
        _require(
            type(exact_session_id) is int and exact_session_id > 1,
            "watchdog exact session ID is invalid",
        )
        _require(
            self._baseline_used_mib is not None and self._peak_used_mib is not None,
            "watchdog baseline is missing",
        )
        observations = self._snapshot()
        deltas: dict[int, int] = {}
        released_baseline_mib: dict[int, int] = {}
        new_or_changed_non_run_processes: list[dict[str, object]] = []
        _require(
            self._baseline_observations is not None
            and self._peak_adjusted_delta_mib is not None,
            "watchdog baseline process roster is missing",
        )
        for item in observations:
            baseline_processes = {
                process.pid: process
                for process in self._baseline_observations[item.gpu_id].processes
            }
            observed_processes = {process.pid: process for process in item.processes}
            released = 0
            for pid, baseline_process in baseline_processes.items():
                observed = observed_processes.get(pid)
                same_identity = observed is not None and (
                    observed.session_id == baseline_process.session_id
                    and observed.uid == baseline_process.uid
                    and observed.user == baseline_process.user
                    and observed.process_name == baseline_process.process_name
                )
                observed_memory = observed.used_memory_mib if same_identity else 0
                released += max(
                    0,
                    baseline_process.used_memory_mib - observed_memory,
                )
            released_baseline_mib[item.gpu_id] = released
            delta = max(
                0,
                item.memory_used_mib
                - self._baseline_used_mib[item.gpu_id]
                + released,
            )
            deltas[item.gpu_id] = delta
            self._peak_adjusted_delta_mib[item.gpu_id] = max(
                self._peak_adjusted_delta_mib[item.gpu_id],
                delta,
            )
            self._peak_used_mib[item.gpu_id] = max(
                self._peak_used_mib[item.gpu_id], item.memory_used_mib
            )
            for process in item.processes:
                if process.session_id == exact_session_id:
                    continue
                baseline = baseline_processes.get(process.pid)
                if baseline is None or (
                    process.session_id != baseline.session_id
                    or process.uid != baseline.uid
                    or process.user != baseline.user
                    or process.process_name != baseline.process_name
                ):
                    new_or_changed_non_run_processes.append(
                        {"gpu_id": item.gpu_id, **asdict(process)}
                    )
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_MEMORY_AUDIT_SCHEMA,
            "sample_index": self._sample_count,
            "exact_session_id": exact_session_id,
            "observation_policy": "no-fixed-memory-limit-v1",
            "fixed_memory_limit_mib": None,
            "observed_adjusted_delta_mib": {
                str(key): value for key, value in deltas.items()
            },
            "released_baseline_mib": {
                str(key): value for key, value in released_baseline_mib.items()
            },
            "new_or_identity_changed_non_run_processes": (
                new_or_changed_non_run_processes
            ),
            "other_user_process_action": "observe-only",
            "passed": True,
            "evidence_scope": {
                "one_second_watchdog_sample": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        _write_new_json(
            require_outside_repository(self.audit_root)
            / "watchdog"
            / f"sample-{self._sample_count:08d}.json",
            record,
        )
        self._sample_count += 1
        return record

    def final_audit(self) -> Mapping[str, object]:
        _require(
            self._gate_count == 2
            and self._sample_count > 0
            and self._baseline_used_mib is not None
            and self._peak_used_mib is not None,
            "eight-GPU final memory audit lacks gates or watchdog samples",
        )
        _require(
            self._peak_adjusted_delta_mib is not None,
            "eight-GPU adjusted delta audit is missing",
        )
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_MEMORY_AUDIT_SCHEMA,
            "sample_interval_seconds": 1,
            "watchdog_sample_count": self._sample_count,
            "baseline_used_mib": {
                str(key): value for key, value in self._baseline_used_mib.items()
            },
            "peak_used_mib": {
                str(key): value for key, value in self._peak_used_mib.items()
            },
            "peak_adjusted_delta_mib": {
                str(key): value
                for key, value in self._peak_adjusted_delta_mib.items()
            },
            "observation_policy": "no-fixed-memory-limit-v1",
            "fixed_memory_limit_mib": None,
            "other_user_process_action": "observe-only",
            "passed": True,
            "evidence_scope": {
                "all_eight_gpu_memory_audited": True,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        }
        record["record_sha256"] = _record_sha256(record)
        _write_new_json(
            require_outside_repository(self.audit_root) / "gpu-memory-audit.json",
            record,
        )
        return record


@dataclass(frozen=True)
class IntegratedSchedulerHandle:
    instance_id: str
    implementation_class: str
    gpu_ids: tuple[int, ...]
    native_scheduler: object
    execution_mode: str

    def validate(self, expected_mode: str) -> None:
        _require(
            isinstance(self.instance_id, str)
            and bool(self.instance_id)
            and self.implementation_class
            == "areal.infra.scheduler.local.LocalScheduler"
            and self.gpu_ids == EXPECTED_GPU_IDS
            and self.execution_mode == expected_mode
            and self.native_scheduler is not None
            and expected_mode in {"cpu-contract-test", "real-gpu"},
            "integrated scheduler handle differs from one LocalScheduler[0..7]",
        )


@dataclass(frozen=True)
class IntegratedStageReference:
    stage: str
    record_path: Path
    record_sha256: str
    input_s_record_sha256s: tuple[str, ...]

    def validate(
        self,
        expected_stage: str,
        *,
        expected_s_record_sha256s: Sequence[str],
    ) -> None:
        path = require_outside_repository(self.record_path)
        record = _read_strict_json(path, f"integrated {expected_stage} reference")
        _require(
            self.stage == expected_stage
            and _is_sha256(self.record_sha256)
            and self.input_s_record_sha256s == tuple(expected_s_record_sha256s)
            and record.get("stage") == expected_stage
            and record.get("input_s_record_sha256s")
            == list(expected_s_record_sha256s)
            and record.get("record_sha256") == self.record_sha256
            and _record_sha256(record) == self.record_sha256,
            f"integrated {expected_stage} reference is missing or invalid",
        )


@dataclass(frozen=True)
class IntegratedCleanupReceipt:
    role: str
    scheduler_instance_id: str
    exact_session_id: int
    worker_ids: tuple[str, ...]

    def validate(
        self,
        *,
        role: str,
        scheduler_instance_id: str,
        exact_session_id: int,
        placements: Sequence[M0WorkerPlacement],
    ) -> None:
        _require(
            self.role == role
            and self.scheduler_instance_id == scheduler_instance_id
            and self.exact_session_id == exact_session_id
            and self.worker_ids == tuple(item.worker_id for item in placements),
            f"integrated {role} cleanup receipt differs from the exact scheduler/SID",
        )


class IntegratedMemoryRuntime(Protocol):
    def gate(self, stage: str) -> object:
        ...

    def start_watchdog(self, *, exact_session_id: int) -> None:
        ...

    def assert_watchdog_healthy(self) -> None:
        ...

    def stop_watchdog(self) -> None:
        ...


class IntegratedM0Adapters(Protocol):
    execution_mode: str

    def create_scheduler(
        self,
        *,
        topology: M0EightGPUTopology,
        run_root: Path,
    ) -> IntegratedSchedulerHandle:
        ...

    def start_actor(
        self,
        scheduler: IntegratedSchedulerHandle,
    ) -> Sequence[M0WorkerPlacement]:
        ...

    def start_rollout(
        self,
        scheduler: IntegratedSchedulerHandle,
    ) -> Sequence[M0WorkerPlacement]:
        ...

    def generate_eight_rlvr_admissions(
        self,
        scheduler: IntegratedSchedulerHandle,
    ) -> Path:
        ...

    def run_tuvw(
        self,
        scheduler: IntegratedSchedulerHandle,
        training_sources: Sequence[RLVRM0SourceRecords],
        multi_s_batch: ValidatedMultiSFrozenTrainingBatch,
    ) -> IntegratedStageReference:
        ...

    def run_x(
        self,
        scheduler: IntegratedSchedulerHandle,
        holdout_sources: Sequence[RLVRM0SourceRecords],
        tuvw: IntegratedStageReference,
    ) -> IntegratedStageReference:
        ...

    def run_y(
        self,
        scheduler: IntegratedSchedulerHandle,
        tuvw: IntegratedStageReference,
        x: IntegratedStageReference,
    ) -> IntegratedStageReference:
        ...

    def stop_rollout(
        self,
        scheduler: IntegratedSchedulerHandle,
        placements: Sequence[M0WorkerPlacement],
        *,
        exact_session_id: int,
    ) -> IntegratedCleanupReceipt:
        ...

    def stop_actor(
        self,
        scheduler: IntegratedSchedulerHandle,
        placements: Sequence[M0WorkerPlacement],
        *,
        exact_session_id: int,
    ) -> IntegratedCleanupReceipt:
        ...


@dataclass(frozen=True)
class IntegratedStageMachineResult:
    state_record_path: Path
    selection: EightGPUAdmissionSelection
    tuvw: IntegratedStageReference
    x: IntegratedStageReference
    y: IntegratedStageReference


class EightGPUIntegratedStageMachine:
    """Run only the lifecycle around injected real T/U/V/W/X/Y implementations."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        exact_session_id: int,
        execution_mode: str,
    ) -> None:
        self.run_root = require_outside_repository(run_root)
        _require(
            self.run_root.is_dir() and not self.run_root.is_symlink(),
            "integrated stage-machine run root is missing or unsafe",
        )
        _require(
            "m0-eight-gpu-integrated" in self.run_root.parts,
            "integrated stage machine cannot reuse a holder run namespace",
        )
        _require(
            type(exact_session_id) is int and exact_session_id > 1,
            "integrated stage-machine exact SID is invalid",
        )
        _require(
            execution_mode in {"cpu-contract-test", "real-gpu"},
            "unknown integrated execution mode",
        )
        self.exact_session_id = exact_session_id
        self.execution_mode = execution_mode

    def run(
        self,
        *,
        adapters: IntegratedM0Adapters,
        memory: IntegratedMemoryRuntime,
    ) -> IntegratedStageMachineResult:
        _require(
            getattr(adapters, "execution_mode", None) == self.execution_mode,
            "integrated adapters differ from the requested execution mode",
        )
        transitions = ["created"]
        topology = M0EightGPUTopology()
        topology.validate()
        ledger = M0EightGPUOwnershipLedger(topology=topology)
        scheduler: IntegratedSchedulerHandle | None = None
        actor: tuple[M0WorkerPlacement, ...] = ()
        rollout: tuple[M0WorkerPlacement, ...] = ()
        rollout_cleanup: IntegratedCleanupReceipt | None = None
        actor_cleanup: IntegratedCleanupReceipt | None = None
        watchdog_started = False
        main_error: BaseException | None = None
        selection: EightGPUAdmissionSelection | None = None
        tuvw: IntegratedStageReference | None = None
        x_reference: IntegratedStageReference | None = None
        y_reference: IntegratedStageReference | None = None
        try:
            memory.gate("preflight")
            transitions.append("preflight-gate-passed")
            memory.gate("immediately-before-scheduler")
            transitions.append("immediately-before-scheduler-gate-passed")
            memory.start_watchdog(exact_session_id=self.exact_session_id)
            watchdog_started = True
            transitions.append("watchdog-started")

            scheduler = adapters.create_scheduler(topology=topology, run_root=self.run_root)
            _require(
                type(scheduler) is IntegratedSchedulerHandle,
                "integrated adapter returned an untyped scheduler",
            )
            scheduler.validate(self.execution_mode)
            transitions.append("scheduler-created")

            actor = tuple(adapters.start_actor(scheduler))
            ledger.start_actor(actor)
            transitions.append("actor-started")
            memory.assert_watchdog_healthy()

            rollout = tuple(adapters.start_rollout(scheduler))
            ledger.start_rollout(rollout)
            transitions.append("rollout-started")
            memory.assert_watchdog_healthy()

            admission_root = adapters.generate_eight_rlvr_admissions(scheduler)
            selection = prepare_eight_gpu_admission_selection(
                runner_admission_dir=admission_root,
                selection_root=self.run_root / "admission-selection",
            )
            transitions.append("eight-admissions-frozen")
            memory.assert_watchdog_healthy()

            training_claims = required_v_member_claims(selection.multi_s_batch)
            holdout_claims = tuple(
                source.s_record_sha256 for source in selection.holdout_sources
            )
            _require(
                set(training_claims).isdisjoint(holdout_claims),
                "training and X holdout S claims overlap before TUVW",
            )
            tuvw = adapters.run_tuvw(
                scheduler,
                selection.training_sources,
                selection.multi_s_batch,
            )
            _require(type(tuvw) is IntegratedStageReference, "TUVW returned no receipt")
            tuvw.validate(
                "tuvw",
                expected_s_record_sha256s=training_claims,
            )
            transitions.append("tuvw-completed")
            memory.assert_watchdog_healthy()

            x_reference = adapters.run_x(
                scheduler,
                selection.holdout_sources,
                tuvw,
            )
            _require(
                type(x_reference) is IntegratedStageReference,
                "X returned no receipt",
            )
            x_reference.validate(
                "x",
                expected_s_record_sha256s=holdout_claims,
            )
            transitions.append("x-completed")
            memory.assert_watchdog_healthy()

            y_reference = adapters.run_y(scheduler, tuvw, x_reference)
            _require(
                type(y_reference) is IntegratedStageReference,
                "Y returned no receipt",
            )
            y_reference.validate(
                "y",
                expected_s_record_sha256s=training_claims,
            )
            ledger.complete_y()
            transitions.append("y-completed")
            memory.assert_watchdog_healthy()
        except BaseException as exc:
            main_error = exc
        finally:
            cleanup_errors: list[BaseException] = []
            if scheduler is not None and rollout:
                try:
                    rollout_cleanup = adapters.stop_rollout(
                        scheduler,
                        rollout,
                        exact_session_id=self.exact_session_id,
                    )
                    _require(
                        type(rollout_cleanup) is IntegratedCleanupReceipt,
                        "rollout cleanup returned no typed receipt",
                    )
                    rollout_cleanup.validate(
                        role="rollout",
                        scheduler_instance_id=scheduler.instance_id,
                        exact_session_id=self.exact_session_id,
                        placements=rollout,
                    )
                    if main_error is None:
                        ledger.stop_rollout()
                    transitions.append("rollout-stopped")
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if scheduler is not None and actor:
                try:
                    actor_cleanup = adapters.stop_actor(
                        scheduler,
                        actor,
                        exact_session_id=self.exact_session_id,
                    )
                    _require(
                        type(actor_cleanup) is IntegratedCleanupReceipt,
                        "actor cleanup returned no typed receipt",
                    )
                    actor_cleanup.validate(
                        role="actor",
                        scheduler_instance_id=scheduler.instance_id,
                        exact_session_id=self.exact_session_id,
                        placements=actor,
                    )
                    if main_error is None:
                        ledger.stop_actor()
                    transitions.append("actor-stopped")
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if watchdog_started:
                try:
                    memory.stop_watchdog()
                    transitions.append("watchdog-stopped")
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                raise M0EightGPUIntegratedError(
                    "integrated M0 exact-SID reverse cleanup failed"
                ) from cleanup_errors[0]
        if main_error is not None:
            raise M0EightGPUIntegratedError("integrated M0 stage failed closed") from main_error
        _require(
            selection is not None
            and tuvw is not None
            and x_reference is not None
            and y_reference is not None
            and scheduler is not None
            and rollout_cleanup is not None
            and actor_cleanup is not None,
            "integrated M0 completed without all lifecycle references",
        )
        transitions.append("completed")
        _require(
            tuple(transitions) == _STAGE_ORDER,
            "integrated M0 stage order differs from the frozen lifecycle",
        )
        record: dict[str, object] = {
            "schema_version": M0_EIGHT_GPU_STAGE_MACHINE_SCHEMA,
            "execution_mode": self.execution_mode,
            "exact_session_id": self.exact_session_id,
            "scheduler": {
                "instance_id": scheduler.instance_id,
                "implementation_class": scheduler.implementation_class,
                "gpu_ids": list(scheduler.gpu_ids),
            },
            "transitions": transitions,
            "selection_record_sha256": json.loads(
                selection.selection_record_path.read_text(encoding="utf-8")
            )["record_sha256"],
            "stage_receipts": {
                "tuvw": {
                    "path": str(tuvw.record_path),
                    "record_sha256": tuvw.record_sha256,
                },
                "x": {
                    "path": str(x_reference.record_path),
                    "record_sha256": x_reference.record_sha256,
                },
                "y": {
                    "path": str(y_reference.record_path),
                    "record_sha256": y_reference.record_sha256,
                },
            },
            "cleanup": {
                "rollout_worker_ids": list(rollout_cleanup.worker_ids),
                "actor_worker_ids": list(actor_cleanup.worker_ids),
                "exact_session_id": self.exact_session_id,
            },
            "evidence_scope": {
                "orchestration_contract_completed": True,
                "real_gpu_adapter_connected": self.execution_mode == "real-gpu",
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
                "joint_version_publish": False,
            },
        }
        _assert_no_secrets(record)
        record["record_sha256"] = _record_sha256(record)
        path = _write_new_json(self.run_root / "stage-machine.json", record)
        return IntegratedStageMachineResult(
            state_record_path=path,
            selection=selection,
            tuvw=tuvw,
            x=x_reference,
            y=y_reference,
        )


def assert_secret_values_absent(
    root: str | Path,
    secret_values: Sequence[str],
) -> None:
    """Final fail-closed check after the shell redactor has run."""

    target = require_outside_repository(root)
    _require(target.is_dir() and not target.is_symlink(), "secret audit root is unsafe")
    secrets = tuple(secret_values)
    _require(
        all(isinstance(value, str) and len(value) >= 16 for value in secrets),
        "secret absence audit received an invalid secret value",
    )
    encoded = tuple(value.encode("utf-8") for value in secrets)
    for path in target.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        _require(
            not any(secret in data for secret in encoded),
            f"runtime secret remains in integrated M0 artifact {path}",
        )
