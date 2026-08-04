from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from jphrl.experiments.m0_eight_gpu_integrated import (
    M0_EIGHT_GPU_ADMISSION_SELECTION_SCHEMA,
    M0_EIGHT_GPU_STAGE_MACHINE_SCHEMA,
    PINNED_AREAL_COMMIT,
    EightGPUIntegratedStageMachine,
    EightGPUMemoryObservation,
    GPUObservation,
    GPUProcessObservation,
    IntegratedCleanupReceipt,
    IntegratedSchedulerHandle,
    IntegratedStageReference,
    M0EightGPUIntegratedError,
    RepositoryState,
    assert_secret_values_absent,
    freeze_integrated_launch_preflight,
    prepare_eight_gpu_admission_selection,
)
from jphrl.experiments.m0_eight_gpu_topology import M0WorkerPlacement
from jphrl.experiments.m0_eight_gpu_real_adapter import (
    M0EightGPURealAdapterError,
    NvidiaSMIGPUStateProvider,
    RealEightGPUIntegratedAdapters,
    _activate_pinned_areal_child_overlay,
    _bind_pinned_areal_child_environment,
    _commit_actor_with_y_compensation,
    _validate_y_actor_terminal_receipts,
    build_distributed_inference_runtime_contract,
    build_distributed_rollout_config,
)
from jphrl.training.joint_activation import (
    ProductionProbeSpec,
    ProductionRollbackRecoveryResult,
    ProductionWorkerState,
)
from jphrl.trajectory.schema import JointVersion
from jphrl.trajectory.rlvr_workflow_admission import (
    prepare_rlvr_workflow_joint_admission,
)
from tests.test_rlvr_workflow_admission import _fake_areal_type_import, _source

def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _seal(record: dict[str, object]) -> dict[str, object]:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return record


def _write_real_admissions(root: Path, count: int = 8) -> Path:
    admission_root = root / "runner-admissions"
    admission_root.mkdir(parents=True, mode=0o700)
    for index in range(count):
        bridge, interaction, joint_version, estimator = _source(
            reward=float(index % 2),
            task_id=7 + index,
            controller_kind="tabular",
        )
        with _fake_areal_type_import():
            result = prepare_rlvr_workflow_joint_admission(
                bridge,
                pre_batch_interaction=interaction,
                estimator=estimator,
                active_joint_version=joint_version,
            )
        (admission_root / f"runner-admission-{index}.json").write_bytes(
            _canonical_json(result.runner_admission) + b"\n"
        )
    return admission_root


class _RepositoryProbe:
    def __init__(self, project: Path, *, project_commit: str, clean: bool = True):
        self.project = project.resolve()
        self.project_commit = project_commit
        self.clean = clean

    def inspect(self, repository: Path) -> RepositoryState:
        if repository == self.project:
            return RepositoryState(self.project_commit, self.clean)
        return RepositoryState(PINNED_AREAL_COMMIT, True)


def _runtime_import_paths(project: Path, areal: Path) -> dict[str, Path]:
    project_module = project / "jphrl" / "__init__.py"
    areal_module = areal / "areal" / "__init__.py"
    project_module.parent.mkdir(parents=True, exist_ok=True)
    areal_module.parent.mkdir(parents=True, exist_ok=True)
    project_module.write_text("# test module\n", encoding="utf-8")
    areal_module.write_text("# test module\n", encoding="utf-8")
    return {"jphrl": project_module, "areal": areal_module}


class _SnapshotProvider:
    def __init__(self, snapshots: list[tuple[GPUObservation, ...]]):
        self.snapshots = snapshots
        self.index = 0

    def snapshot(self):
        value = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        self.index += 1
        return value


class ArealChildOverlayTests(unittest.TestCase):
    def test_verified_overlay_is_bound_explicitly_without_parent_path_pollution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "artifacts" / "m0" / "run-1"
            overlay = root / "runtime" / "areal-overlays" / run_root.name
            data_proxy = overlay / "areal" / "v2" / "inference_service" / "data_proxy"
            data_proxy.mkdir(parents=True, mode=0o700)
            (overlay / "areal" / "__init__.py").write_text("", encoding="utf-8")
            app = data_proxy / "app.py"
            main = data_proxy / "__main__.py"
            app.write_text("pre_batch_export_hook = None\n", encoding="utf-8")
            main.write_text("AREAL_PRE_BATCH_HOOK = None\n", encoding="utf-8")
            patch_path = (
                Path(__file__).resolve().parents[1]
                / "patches"
                / "areal-v2.0.0-data-proxy-pre-batch-hook.patch"
            )
            patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()
            manifest = {
                "schema_version": "jph.areal-child-overlay.v1",
                "areal_base_commit": PINNED_AREAL_COMMIT,
                "hook_import_path": (
                    "jphrl.trajectory.rlvr_online_binding."
                    "pre_batch_finalize_rlvr_v2_agent_admission"
                ),
                "patch_sha256": patch_sha,
                "patched_files": {
                    "areal/v2/inference_service/data_proxy/__main__.py": (
                        hashlib.sha256(main.read_bytes()).hexdigest()
                    ),
                    "areal/v2/inference_service/data_proxy/app.py": (
                        hashlib.sha256(app.read_bytes()).hexdigest()
                    ),
                },
                "project_commit": "a" * 40,
            }
            manifest_path = overlay / "jph-overlay-manifest.json"
            manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
            overlay.chmod(0o700)
            manifest_path.chmod(0o600)
            config = SimpleNamespace(
                jph_root=root.resolve(),
                run_root=run_root.resolve(),
                project_commit="a" * 40,
            )
            rollout_config = SimpleNamespace(
                scheduling_spec=(SimpleNamespace(env_vars={}),)
            )
            environment = {
                "JPH_AREAL_CHILD_OVERLAY": str(overlay),
                "JPH_AREAL_PRE_BATCH_PATCH_SHA256": patch_sha,
                "PYTHONPATH": "/original/pythonpath",
            }
            with patch.dict(os.environ, environment, clear=False):
                _activate_pinned_areal_child_overlay(config)
                child_env = _bind_pinned_areal_child_environment(rollout_config)
                self.assertEqual(os.environ["PYTHONPATH"], "/original/pythonpath")
                self.assertEqual(
                    child_env["PYTHONPATH"],
                    f"{overlay.resolve()}{os.pathsep}/original/pythonpath",
                )
                self.assertEqual(
                    rollout_config.scheduling_spec[0].env_vars,
                    child_env,
                )
                self.assertEqual(
                    child_env["AREAL_PRE_BATCH_HOOK"],
                    manifest["hook_import_path"],
                )


def _snapshot(
    *,
    used_mib: int,
    free_mib: int = 80000,
    session_id: int | None = None,
) -> tuple[GPUObservation, ...]:
    return tuple(
        GPUObservation(
            gpu_id=gpu_id,
            memory_used_mib=used_mib,
            memory_free_mib=free_mib,
            processes=(
                ()
                if session_id is None
                else (
                    GPUProcessObservation(
                        pid=2000 + gpu_id,
                        session_id=session_id,
                        uid=1000,
                        user="shared-user",
                        process_name="python",
                        used_memory_mib=used_mib,
                    ),
                )
            ),
        )
        for gpu_id in range(8)
    )


class _MemoryRuntime:
    def __init__(self):
        self.calls: list[str] = []
        self.sid: int | None = None

    def gate(self, stage: str):
        self.calls.append(f"gate:{stage}")

    def start_watchdog(self, *, exact_session_id: int) -> None:
        self.sid = exact_session_id
        self.calls.append("watchdog:start")

    def assert_watchdog_healthy(self) -> None:
        self.calls.append("watchdog:healthy")

    def stop_watchdog(self) -> None:
        self.calls.append("watchdog:stop")


class _Adapters:
    execution_mode = "cpu-contract-test"

    def __init__(self, admission_root: Path, run_root: Path, *, fail_at: str | None = None):
        self.admission_root = admission_root
        self.run_root = run_root
        self.fail_at = fail_at
        self.calls: list[str] = []
        self.scheduler_creations = 0
        self.training_s: tuple[str, ...] = ()
        self.holdout_s: tuple[str, ...] = ()

    @staticmethod
    def _placements(role: str) -> tuple[M0WorkerPlacement, ...]:
        offset = 0 if role == "actor" else 4
        scheduler_role = role if role == "actor" else "rollout-inf"
        return tuple(
            M0WorkerPlacement(role, rank, rank + offset, f"{scheduler_role}/{rank}")
            for rank in range(4)
        )

    def _reference(self, stage: str, claims: tuple[str, ...]) -> IntegratedStageReference:
        path = self.run_root / "injected-receipts" / f"{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = _seal(
            {
                "schema_version": "jph.test-injected-stage-reference.v1",
                "stage": stage,
                "input_s_record_sha256s": list(claims),
                "internal_receipts_verified": False,
                "evidence_scope": {
                    "cpu_contract_test": True,
                    "policy_optimizer_update": False,
                    "harness_optimizer_update": False,
                    "joint_version_publish": False,
                },
            }
        )
        path.write_bytes(_canonical_json(record) + b"\n")
        return IntegratedStageReference(
            stage=stage,
            record_path=path,
            record_sha256=record["record_sha256"],
            input_s_record_sha256s=claims,
        )

    def create_scheduler(self, *, topology, run_root):
        self.calls.append("scheduler:create")
        self.scheduler_creations += 1
        self.assertEqualPath = run_root
        return IntegratedSchedulerHandle(
            instance_id="cpu-local-scheduler-1",
            implementation_class="areal.infra.scheduler.local.LocalScheduler",
            gpu_ids=tuple(topology.scheduler_gpu_ids),
            native_scheduler=object(),
            execution_mode=self.execution_mode,
        )

    def start_actor(self, scheduler):
        self.calls.append("actor:start")
        return self._placements("actor")

    def start_rollout(self, scheduler):
        self.calls.append("rollout:start")
        return self._placements("rollout")

    def generate_eight_rlvr_admissions(self, scheduler):
        self.calls.append("admissions:generate")
        return self.admission_root

    def run_tuvw(self, scheduler, training_sources, multi_s_batch):
        self.calls.append("tuvw")
        self.training_s = tuple(source.s_record_sha256 for source in training_sources)
        claims = tuple(member.s_record_sha256 for member in multi_s_batch.members)
        if self.fail_at == "tuvw-claims":
            claims = tuple(reversed(claims))
        return self._reference("tuvw", claims)

    def run_x(self, scheduler, holdout_sources, tuvw):
        self.calls.append("x")
        if self.fail_at == "x":
            raise RuntimeError("injected X failure")
        self.holdout_s = tuple(source.s_record_sha256 for source in holdout_sources)
        return self._reference("x", self.holdout_s)

    def run_y(self, scheduler, tuvw, x):
        self.calls.append("y")
        return self._reference("y", tuvw.input_s_record_sha256s)

    def stop_rollout(self, scheduler, placements, *, exact_session_id):
        self.calls.append("rollout:stop")
        return IntegratedCleanupReceipt(
            role="rollout",
            scheduler_instance_id=scheduler.instance_id,
            exact_session_id=exact_session_id,
            worker_ids=tuple(item.worker_id for item in placements),
        )

    def stop_actor(self, scheduler, placements, *, exact_session_id):
        self.calls.append("actor:stop")
        return IntegratedCleanupReceipt(
            role="actor",
            scheduler_instance_id=scheduler.instance_id,
            exact_session_id=exact_session_id,
            worker_ids=tuple(item.worker_id for item in placements),
        )


class EightGPUAdmissionSelectionTests(unittest.TestCase):
    def test_exact_eight_freeze_first_four_training_last_four_x_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions = _write_real_admissions(root)
            selection = prepare_eight_gpu_admission_selection(
                runner_admission_dir=admissions,
                selection_root=root / "selection",
            )

            self.assertEqual(
                [
                    source.runner_admission["episode_trace"]["task_id"]
                    for source in selection.training_sources
                ],
                ["7", "8", "9", "10"],
            )
            self.assertEqual(
                [
                    source.runner_admission["episode_trace"]["task_id"]
                    for source in selection.holdout_sources
                ],
                ["11", "12", "13", "14"],
            )
            training_s = {source.s_record_sha256 for source in selection.training_sources}
            holdout_s = {source.s_record_sha256 for source in selection.holdout_sources}
            self.assertEqual(
                {member.s_record_sha256 for member in selection.multi_s_batch.members},
                training_s,
            )
            self.assertTrue(training_s.isdisjoint(holdout_s))
            self.assertEqual(len(selection.training_s_paths), 4)
            self.assertEqual(len(selection.multi_s_batch.members), 4)
            record = json.loads(
                selection.selection_record_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                record["schema_version"],
                M0_EIGHT_GPU_ADMISSION_SELECTION_SCHEMA,
            )
            self.assertEqual(
                [item["role"] for item in record["admissions"]],
                ["training"] * 4 + ["x-holdout"] * 4,
            )
            self.assertFalse(record["evidence_scope"]["holdouts_used_for_training"])
            self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])

    def test_count_extra_duplicate_and_reuse_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seven = _write_real_admissions(root / "seven", count=7)
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "exactly eight"):
                prepare_eight_gpu_admission_selection(
                    runner_admission_dir=seven,
                    selection_root=root / "selection-seven",
                )

            nine = _write_real_admissions(root / "nine", count=8)
            (nine / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "exactly eight"):
                prepare_eight_gpu_admission_selection(
                    runner_admission_dir=nine,
                    selection_root=root / "selection-nine",
                )

            valid = _write_real_admissions(root / "valid", count=8)
            existing = root / "selection-existing"
            existing.mkdir()
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "must be new"):
                prepare_eight_gpu_admission_selection(
                    runner_admission_dir=valid,
                    selection_root=existing,
                )

            duplicate = _write_real_admissions(root / "duplicate", count=8)
            (duplicate / "runner-admission-7.json").write_bytes(
                (duplicate / "runner-admission-0.json").read_bytes()
            )
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "duplicate"):
                prepare_eight_gpu_admission_selection(
                    runner_admission_dir=duplicate,
                    selection_root=root / "selection-duplicate",
                )


class IntegratedPreflightTests(unittest.TestCase):
    def test_tmux_clean_pinned_repositories_and_formal_root_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "src" / "project"
            areal = root / "src" / "AReaL-v2.0.0"
            project.mkdir(parents=True)
            areal.mkdir(parents=True)
            runtime_imports = _runtime_import_paths(project, areal)
            project_commit = "4" * 40
            run_root = root / "artifacts" / "m0-eight-gpu-integrated" / "run-1"

            frozen = freeze_integrated_launch_preflight(
                jph_root=root,
                project_repository=project,
                areal_repository=areal,
                run_root=run_root,
                expected_project_commit=project_commit,
                tmux_connection="/tmp/tmux/default,1,0",
                repository_probe=_RepositoryProbe(
                    project,
                    project_commit=project_commit,
                ),
                runtime_import_paths=runtime_imports,
            )
            record = json.loads(frozen.record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["project"]["commit"], project_commit)
            self.assertEqual(record["areal"]["commit"], PINNED_AREAL_COMMIT)
            self.assertFalse(record["holder_control_reused"])
            self.assertFalse(record["evidence_scope"]["gpu_execution"])
            self.assertEqual(frozen.record_path.stat().st_mode & 0o777, 0o600)

    def test_non_tmux_dirty_and_holder_namespace_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "src" / "project"
            areal = root / "src" / "AReaL-v2.0.0"
            project.mkdir(parents=True)
            areal.mkdir(parents=True)
            runtime_imports = _runtime_import_paths(project, areal)
            commit = "4" * 40
            base = dict(
                jph_root=root,
                project_repository=project,
                areal_repository=areal,
                expected_project_commit=commit,
                runtime_import_paths=runtime_imports,
            )
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "inside tmux"):
                freeze_integrated_launch_preflight(
                    **base,
                    run_root=root / "artifacts/m0-eight-gpu-integrated/no-tmux",
                    tmux_connection="",
                    repository_probe=_RepositoryProbe(project, project_commit=commit),
                )
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "dirty"):
                freeze_integrated_launch_preflight(
                    **base,
                    run_root=root / "artifacts/m0-eight-gpu-integrated/dirty",
                    tmux_connection="tmux",
                    repository_probe=_RepositoryProbe(
                        project,
                        project_commit=commit,
                        clean=False,
                    ),
                )
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "non-holder"):
                freeze_integrated_launch_preflight(
                    **base,
                    run_root=root / "artifacts/areal-b0/holder-run",
                    tmux_connection="tmux",
                    repository_probe=_RepositoryProbe(project, project_commit=commit),
                )

            outside = root / "unverified-install" / "jphrl" / "__init__.py"
            outside.parent.mkdir(parents=True)
            outside.write_text("# crossed import\n", encoding="utf-8")
            with self.assertRaisesRegex(
                M0EightGPUIntegratedError,
                "imported outside",
            ):
                freeze_integrated_launch_preflight(
                    **{
                        **base,
                        "runtime_import_paths": {
                            "jphrl": outside,
                            "areal": runtime_imports["areal"],
                        },
                    },
                    run_root=root
                    / "artifacts"
                    / "m0-eight-gpu-integrated"
                    / "crossed-import",
                    tmux_connection="tmux",
                    repository_probe=_RepositoryProbe(project, project_commit=commit),
                )


class RealAdapterCleanupTests(unittest.TestCase):
    def test_runtime_contract_binds_rollout_prefetch_capacity(self) -> None:
        torch_module = ModuleType("torch")
        torch_module.version = SimpleNamespace(cuda="12.8")
        config = SimpleNamespace(
            validate=Mock(),
            max_new_tokens=512,
            behavior_revision="b" * 40,
            dataset_revision="c" * 40,
            dataset_selection="sequential-offset0-count8-v1",
            project_commit="d" * 40,
            harness_seed=1,
            jph_root=Path("/outside-repository"),
            transaction_id="capacity-contract",
        )
        with (
            patch.dict(sys.modules, {"torch": torch_module}),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "distribution_version",
                return_value="test-version",
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "_nvidia_driver_version",
                return_value="test-driver",
            ),
        ):
            contract = build_distributed_inference_runtime_contract(
                config,
                server_args={"disable_cuda_graph": False},
                gpu_uuids=[f"GPU-{rank}" for rank in range(4)],
                gpu_names=["test-gpu"] * 4,
            )

        rollout = contract["fixed"]["rollout"]
        self.assertEqual(
            rollout,
            {
                "backend": "sglang:d4",
                "consumer_batch_size": 4,
                "max_concurrent_rollouts": 8,
                "max_head_offpolicyness": 1,
            },
        )

    def test_rollout_capacity_admits_both_frozen_version_zero_batches(self) -> None:
        class ConfigRecord:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        areal_module = ModuleType("areal")
        areal_api_module = ModuleType("areal.api")
        cli_args_module = ModuleType("areal.api.cli_args")
        cli_args_module.InferenceEngineConfig = ConfigRecord
        cli_args_module.SchedulingSpec = ConfigRecord
        config = SimpleNamespace(
            validate=Mock(),
            experiment_name="m0",
            trial_name="capacity-regression",
            run_root=Path("/outside-repository/run"),
            model_snapshot=Path("/outside-repository/model"),
            admin_api_key="a" * 32,
        )
        with patch.dict(
            sys.modules,
            {
                "areal": areal_module,
                "areal.api": areal_api_module,
                "areal.api.cli_args": cli_args_module,
            },
        ):
            rollout = build_distributed_rollout_config(config)

        config.validate.assert_called_once_with()
        self.assertEqual(rollout.consumer_batch_size, 4)
        self.assertEqual(rollout.max_concurrent_rollouts, 8)
        self.assertEqual(rollout.max_head_offpolicyness, 1)
        self.assertEqual(
            (rollout.max_head_offpolicyness + 1)
            * rollout.consumer_batch_size,
            rollout.max_concurrent_rollouts,
        )

    def test_actor_initialize_forwards_required_engine_addr_by_keyword(self) -> None:
        class FinetuneSpec:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class Actor:
            def __init__(self) -> None:
                self.initialize_args: tuple[object, ...] | None = None
                self.initialize_kwargs: dict[str, object] | None = None
                self.version = -1

            def create_process_group(self) -> None:
                pass

            def initialize(self, *args: object, **kwargs: object) -> None:
                self.initialize_args = args
                self.initialize_kwargs = kwargs

            def set_version(self, version: int) -> None:
                self.version = version

            def get_version(self) -> int:
                return self.version

            def destroy(self) -> None:
                pass

        actor = Actor()

        class ActorType:
            @classmethod
            def as_controller(cls, config: object, scheduler: object) -> Actor:
                return actor

        areal_module = ModuleType("areal")
        areal_api_module = ModuleType("areal.api")
        areal_api_module.FinetuneSpec = FinetuneSpec
        adapter = object.__new__(RealEightGPUIntegratedAdapters)
        adapter.config = object()
        adapter.actor = None
        scheduler = IntegratedSchedulerHandle(
            instance_id="scheduler",
            implementation_class="areal.infra.scheduler.local.LocalScheduler",
            gpu_ids=tuple(range(8)),
            native_scheduler=object(),
            execution_mode="real-gpu",
        )
        placements = tuple(
            M0WorkerPlacement(
                role="actor",
                rank=rank,
                physical_gpu_id=rank,
                worker_id=f"actor/{rank}",
            )
            for rank in range(4)
        )
        with (
            patch.dict(
                sys.modules,
                {"areal": areal_module, "areal.api": areal_api_module},
            ),
            patch(
                "jphrl.training.areal_distributed_policy.JPHFSDPPPOActor",
                ActorType,
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "build_distributed_actor_config",
                return_value=object(),
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "assert_controller_has_no_local_optimizer",
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "observe_local_scheduler_placements",
                return_value=placements,
            ),
        ):
            observed = adapter.start_actor(scheduler)

        self.assertEqual(observed, placements)
        self.assertEqual(actor.initialize_args, ())
        self.assertIsNotNone(actor.initialize_kwargs)
        assert actor.initialize_kwargs is not None
        self.assertEqual(set(actor.initialize_kwargs), {"addr", "ft_spec", "role"})
        self.assertIsNone(actor.initialize_kwargs["addr"])
        self.assertEqual(actor.initialize_kwargs["role"], "actor")
        self.assertEqual(
            actor.initialize_kwargs["ft_spec"].kwargs,
            {
                "total_train_epochs": 1,
                "dataset_size": 4,
                "train_batch_size": 4,
            },
        )

    def test_actor_destroy_runs_even_when_pending_parent_rollback_fails(self) -> None:
        class Actor:
            def __init__(self) -> None:
                self.destroyed = False

            def _rollback_workers(self, transaction_id: str) -> None:
                self.transaction_id = transaction_id
                raise RuntimeError("injected rollback failure")

            def destroy(self) -> None:
                self.destroyed = True

        adapter = object.__new__(RealEightGPUIntegratedAdapters)
        actor = Actor()
        adapter.actor = actor
        adapter.live_policy_candidate = SimpleNamespace(
            receipt={"transaction": {"transaction_id": "pending-m0"}}
        )
        adapter.actor_candidate_committed = False
        scheduler = IntegratedSchedulerHandle(
            instance_id="scheduler",
            implementation_class="areal.infra.scheduler.local.LocalScheduler",
            gpu_ids=tuple(range(8)),
            native_scheduler=object(),
            execution_mode="cpu-contract-test",
        )
        with self.assertRaisesRegex(RuntimeError, "rollback failure"):
            adapter.stop_actor(
                scheduler,
                (),
                exact_session_id=4321,
            )
        self.assertTrue(actor.destroyed)
        self.assertIsNone(adapter.actor)

    def test_live_policy_capability_is_remembered_before_harness_update(self) -> None:
        source = Path(
            RealEightGPUIntegratedAdapters.run_tuvw.__code__.co_filename
        ).read_text(encoding="utf-8")
        method = source[
            source.index("    def run_tuvw(") : source.index(
                "    def _seal_and_verify_distributed_w("
            )
        ]
        self.assertLess(
            method.index("self.live_policy_candidate = live_policy"),
            method.index(
                "trainer.update_from_validated_multi_s_frozen_training_batch"
            ),
        )


def _terminal_commit_fixture() -> tuple[object, object, list[dict[str, object]]]:
    rank_receipts = [{"record_sha256": str(rank + 1) * 64} for rank in range(4)]
    live = SimpleNamespace(
        receipt={
            "transaction": {"transaction_id": "m0-terminal"},
            "optimizer": {
                "remote_optimizer_receipt": {
                    "record_sha256": "a" * 64,
                    "rank_receipts": rank_receipts,
                }
            },
            "record_sha256": "b" * 64,
        }
    )
    activation = SimpleNamespace(
        activation_id="c" * 32,
        attestation_sha256="c" * 64,
        active_release_id="candidate-release",
        candidate_release_id="candidate-release",
        rollback_record_path=Path("/outside/rollback.json"),
    )
    receipts: list[dict[str, object]] = []
    state = {
        "actor_version": 7,
        "optimizer_step": 3,
        "scheduler_state_sha256": "d" * 64,
    }
    for rank in range(4):
        receipts.append(
            _seal(
                {
                    "schema_version": "jph.m0-policy-worker-commit.v1",
                    "transaction_id": "m0-terminal",
                    "worker_rank": rank,
                    "aggregate_sha256": "a" * 64,
                    "policy_candidate_sha256": "b" * 64,
                    "rank_receipt_sha256": rank_receipts[rank]["record_sha256"],
                    "y_attestation_sha256": "c" * 64,
                    "y_active_release_id": "candidate-release",
                    "state_before": dict(state),
                    "state_after": dict(state),
                    "evidence_scope": {
                        "y_success_revalidated": True,
                        "training_state_changed": False,
                        "rollback_state_clear_authorized": True,
                        "policy_optimizer_update": False,
                        "harness_optimizer_update": False,
                    },
                }
            )
        )
    return live, activation, receipts


class YTerminalCompensationTests(unittest.TestCase):
    @staticmethod
    def _compensation_fixture(activation: object):
        version = JointVersion(
            policy="parent-policy",
            harness_controller="parent-harness",
            harness_artifact="artifact",
            tool_schema="tool",
            parser="parser",
            environment="environment",
            evaluator="evaluator",
            tokenizer="tokenizer",
            context_builder="context",
        )
        parent = SimpleNamespace(
            release_id="parent-release",
            joint_version=version,
        )
        candidate = SimpleNamespace(
            release_id="candidate-release",
        )
        raw_probe = b"restored-parent-probe"
        probe = ProductionProbeSpec(
            probe_id="parent-probe",
            fixture=b"fixture",
            fixture_sha256=hashlib.sha256(b"fixture").hexdigest(),
            expected_output_sha256=hashlib.sha256(raw_probe).hexdigest(),
        )
        state = ProductionWorkerState(
            worker_id="rollout-dp4",
            lifecycle_phase="serving",
            active_release_id=parent.release_id,
            joint_version=version,
            policy_engine_version=7,
            policy_checkpoint_sha256="1" * 64,
            harness_controller_version=version.harness_controller,
            harness_checkpoint_sha256="2" * 64,
            harness_parameter_digest="3" * 64,
        )

        class Controller:
            def __init__(self) -> None:
                self.calls: list[Path] = []

            def recover_pending(self, path: Path):
                self.calls.append(path)
                return ProductionRollbackRecoveryResult(
                    activation_id=activation.activation_id,
                    parent_release_id=parent.release_id,
                    candidate_release_id=candidate.release_id,
                    active_release_id=parent.release_id,
                    outcome="parent_restored_from_journal",
                    rollback_record_path=activation.rollback_record_path,
                    journal_path=Path("/outside/journal.json"),
                    evidence_scope={},
                )

        controller = Controller()
        store = SimpleNamespace(read_active=Mock(return_value=parent))
        worker = SimpleNamespace(
            read_state=Mock(return_value=state),
            run_probe=Mock(return_value=raw_probe),
        )
        return controller, store, parent, candidate, worker, probe

    def test_terminal_receipts_bind_rank_transaction_candidate_and_attestation(self) -> None:
        live, activation, receipts = _terminal_commit_fixture()
        summaries = _validate_y_actor_terminal_receipts(
            receipts,
            live_policy_candidate=live,
            activation=activation,
        )
        self.assertEqual([item["worker_rank"] for item in summaries], list(range(4)))
        self.assertTrue(
            all(item["transaction_id"] == "m0-terminal" for item in summaries)
        )
        self.assertTrue(
            all(item["policy_candidate_sha256"] == "b" * 64 for item in summaries)
        )
        self.assertTrue(
            all(item["y_attestation_sha256"] == "c" * 64 for item in summaries)
        )

        mutations = (
            ("worker_rank", 0, 2),
            ("transaction_id", 1, "crossed"),
            ("policy_candidate_sha256", 2, "e" * 64),
            ("y_attestation_sha256", 3, "f" * 64),
        )
        for field, rank, value in mutations:
            with self.subTest(field=field):
                tampered = [dict(receipt) for receipt in receipts]
                tampered[rank] = dict(tampered[rank])
                tampered[rank][field] = value
                _seal(tampered[rank])
                with self.assertRaisesRegex(
                    M0EightGPURealAdapterError,
                    f"rank {rank} receipt differs",
                ):
                    _validate_y_actor_terminal_receipts(
                        tampered,
                        live_policy_candidate=live,
                        activation=activation,
                    )

    def test_commit_failure_restores_and_reprobes_parent_before_reraising(self) -> None:
        live, activation, _receipts = _terminal_commit_fixture()
        controller, store, parent, candidate, worker, probe = (
            self._compensation_fixture(activation)
        )
        actor = SimpleNamespace(
            commit_m0_policy_candidate=Mock(
                side_effect=RuntimeError("injected terminal commit failure")
            )
        )
        with self.assertRaisesRegex(RuntimeError, "terminal commit failure"):
            _commit_actor_with_y_compensation(
                actor=actor,
                live_policy_candidate=live,
                activation_controller=controller,
                activation=activation,
                release_store=store,
                parent_release=parent,
                candidate_release=candidate,
                worker=worker,
                parent_probe=probe,
            )
        self.assertEqual(controller.calls, [activation.rollback_record_path])
        store.read_active.assert_called_once_with()
        worker.read_state.assert_called_once_with()
        worker.run_probe.assert_called_once_with(probe.fixture)

    def test_successful_terminal_validation_does_not_invoke_compensation(self) -> None:
        live, activation, receipts = _terminal_commit_fixture()
        controller, store, parent, candidate, worker, probe = (
            self._compensation_fixture(activation)
        )
        actor = SimpleNamespace(
            commit_m0_policy_candidate=Mock(return_value=receipts)
        )
        summaries = _commit_actor_with_y_compensation(
            actor=actor,
            live_policy_candidate=live,
            activation_controller=controller,
            activation=activation,
            release_store=store,
            parent_release=parent,
            candidate_release=candidate,
            worker=worker,
            parent_probe=probe,
        )
        self.assertEqual([item["worker_rank"] for item in summaries], list(range(4)))
        self.assertEqual(controller.calls, [])
        store.read_active.assert_not_called()
        worker.run_probe.assert_not_called()

    def test_bad_terminal_receipt_also_compensates_and_bad_recovery_fails_closed(self) -> None:
        live, activation, receipts = _terminal_commit_fixture()
        receipts[2]["transaction_id"] = "crossed"
        _seal(receipts[2])
        controller, store, parent, candidate, worker, probe = (
            self._compensation_fixture(activation)
        )
        actor = SimpleNamespace(
            commit_m0_policy_candidate=Mock(return_value=receipts)
        )
        with self.assertRaisesRegex(
            M0EightGPURealAdapterError,
            "rank 2 receipt differs",
        ):
            _commit_actor_with_y_compensation(
                actor=actor,
                live_policy_candidate=live,
                activation_controller=controller,
                activation=activation,
                release_store=store,
                parent_release=parent,
                candidate_release=candidate,
                worker=worker,
                parent_probe=probe,
            )
        self.assertEqual(len(controller.calls), 1)

        controller, store, parent, candidate, worker, probe = (
            self._compensation_fixture(activation)
        )
        worker.run_probe.return_value = b"wrong-parent-probe"
        failing_actor = SimpleNamespace(
            commit_m0_policy_candidate=Mock(
                side_effect=RuntimeError("terminal failure")
            )
        )
        with self.assertRaisesRegex(
            M0EightGPURealAdapterError,
            "compensation both failed",
        ):
            _commit_actor_with_y_compensation(
                actor=failing_actor,
                live_policy_candidate=live,
                activation_controller=controller,
                activation=activation,
                release_store=store,
                parent_release=parent,
                candidate_release=candidate,
                worker=worker,
                parent_probe=probe,
            )


class NvidiaSMIGPUStateProviderTests(unittest.TestCase):
    @staticmethod
    def _nvidia_smi_output(
        arguments: object,
        *,
        process_rows: str = "GPU-0, 946362, python, 128",
    ) -> str:
        query = tuple(arguments)
        if query[0] == "--query-gpu=index,memory.used,memory.free":
            return "\n".join(f"{gpu_id}, 1024, 80000" for gpu_id in range(8))
        if query[0] == "--query-compute-apps=gpu_uuid,pid,process_name,used_memory":
            return process_rows
        if query[0] == "--query-gpu=index,uuid":
            return "\n".join(f"{gpu_id}, GPU-{gpu_id}" for gpu_id in range(8))
        raise AssertionError(f"unexpected nvidia-smi query: {query!r}")

    def test_process_that_exits_before_identity_lookup_is_skipped(self) -> None:
        with (
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter._run_nvidia_smi",
                side_effect=self._nvidia_smi_output,
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "_linux_process_start_time",
                return_value=None,
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter.subprocess.run"
            ) as process_probe,
        ):
            observations = NvidiaSMIGPUStateProvider().snapshot()

        self.assertEqual(len(observations), 8)
        self.assertTrue(all(not item.processes for item in observations))
        process_probe.assert_not_called()

    def test_failed_identity_lookup_for_still_live_process_fails_closed(self) -> None:
        with (
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter._run_nvidia_smi",
                side_effect=self._nvidia_smi_output,
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "_linux_process_start_time",
                side_effect=(101, 101),
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ("ps",)),
            ),
        ):
            with self.assertRaisesRegex(
                M0EightGPURealAdapterError,
                "cannot bind GPU process",
            ):
                NvidiaSMIGPUStateProvider().snapshot()

    def test_pid_reuse_during_identity_lookup_fails_closed(self) -> None:
        with (
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter._run_nvidia_smi",
                side_effect=self._nvidia_smi_output,
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "_linux_process_start_time",
                side_effect=(101, 202),
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter.subprocess.run",
                return_value=SimpleNamespace(stdout="4321 1000 worker\n"),
            ),
        ):
            with self.assertRaisesRegex(
                M0EightGPURealAdapterError,
                "PID was reused",
            ):
                NvidiaSMIGPUStateProvider().snapshot()

    def test_process_that_exits_after_identity_lookup_is_skipped(self) -> None:
        with (
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter._run_nvidia_smi",
                side_effect=self._nvidia_smi_output,
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter."
                "_linux_process_start_time",
                side_effect=(101, None),
            ),
            patch(
                "jphrl.experiments.m0_eight_gpu_real_adapter.subprocess.run",
                return_value=SimpleNamespace(stdout="4321 1000 worker\n"),
            ),
        ):
            observations = NvidiaSMIGPUStateProvider().snapshot()

        self.assertTrue(all(not item.processes for item in observations))


class EightGPUMemoryObservationTests(unittest.TestCase):
    def test_two_gates_and_watchdog_are_observation_only_without_fixed_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = _snapshot(used_mib=3000, session_id=5555)
            provider = _SnapshotProvider(
                [
                    baseline,
                    baseline,
                    _snapshot(used_mib=50000, session_id=7777),
                ]
            )
            safety = EightGPUMemoryObservation(
                provider=provider,
                audit_root=Path(directory) / "memory",
            )
            safety.gate("preflight")
            safety.gate("immediately-before-scheduler")
            sample = safety.watchdog_sample(exact_session_id=7777)
            final = safety.final_audit()

            self.assertTrue(sample["passed"])
            self.assertTrue(final["passed"])
            self.assertEqual(final["sample_interval_seconds"], 1)
            self.assertIsNone(final["fixed_memory_limit_mib"])
            self.assertEqual(final["other_user_process_action"], "observe-only")
            self.assertEqual(
                set(final["peak_adjusted_delta_mib"]),
                {str(i) for i in range(8)},
            )
            # Baseline process used 3000 MiB and then disappeared. Its released
            # memory is added back, so the observed run delta is not understated.
            self.assertEqual(sample["released_baseline_mib"]["0"], 3000)
            self.assertGreater(sample["observed_adjusted_delta_mib"]["0"], 30000)

    def test_wrong_order_fails_but_new_other_user_process_is_observed_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrong = EightGPUMemoryObservation(
                provider=_SnapshotProvider([_snapshot(used_mib=1000)]),
                audit_root=Path(directory) / "wrong",
            )
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "exactly twice"):
                wrong.gate("immediately-before-scheduler")

            baseline = _snapshot(used_mib=3000, session_id=5555)
            other = EightGPUMemoryObservation(
                provider=_SnapshotProvider(
                    [
                        baseline,
                        _snapshot(used_mib=4000, session_id=6666),
                        _snapshot(used_mib=70000, session_id=9999),
                    ]
                ),
                audit_root=Path(directory) / "other-user",
            )
            other.gate("preflight")
            second = other.gate("immediately-before-scheduler")
            sample = other.watchdog_sample(exact_session_id=8888)
            self.assertTrue(second["passed"])
            self.assertTrue(second["baseline_process_changes"])
            self.assertTrue(sample["passed"])
            self.assertTrue(sample["new_or_identity_changed_non_run_processes"])
            self.assertIsNone(sample["fixed_memory_limit_mib"])


class EightGPUIntegratedStageMachineTests(unittest.TestCase):
    def test_one_scheduler_actor_lives_through_y_and_reverse_exact_sid_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions = _write_real_admissions(root / "source")
            run_root = root / "artifacts" / "m0-eight-gpu-integrated" / "run"
            run_root.mkdir(parents=True)
            adapters = _Adapters(admissions, run_root)
            memory = _MemoryRuntime()
            result = EightGPUIntegratedStageMachine(
                run_root=run_root,
                exact_session_id=4321,
                execution_mode="cpu-contract-test",
            ).run(adapters=adapters, memory=memory)

            self.assertEqual(adapters.scheduler_creations, 1)
            self.assertEqual(
                adapters.calls,
                [
                    "scheduler:create",
                    "actor:start",
                    "rollout:start",
                    "admissions:generate",
                    "tuvw",
                    "x",
                    "y",
                    "rollout:stop",
                    "actor:stop",
                ],
            )
            self.assertTrue(set(adapters.training_s).isdisjoint(adapters.holdout_s))
            self.assertEqual(len(adapters.training_s), 4)
            self.assertEqual(len(adapters.holdout_s), 4)
            record = json.loads(result.state_record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], M0_EIGHT_GPU_STAGE_MACHINE_SCHEMA)
            self.assertEqual(record["scheduler"]["gpu_ids"], list(range(8)))
            self.assertEqual(record["cleanup"]["exact_session_id"], 4321)
            self.assertFalse(record["evidence_scope"]["real_gpu_adapter_connected"])
            self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])
            self.assertLess(
                record["transitions"].index("y-completed"),
                record["transitions"].index("actor-stopped"),
            )

    def test_x_failure_and_crossed_tuvw_claims_cleanup_without_false_success(self) -> None:
        for fail_at, expected_cause in (("x", "stage failed"), ("tuvw-claims", "stage failed")):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                admissions = _write_real_admissions(root / "source")
                run_root = root / "artifacts" / "m0-eight-gpu-integrated" / "run"
                run_root.mkdir(parents=True)
                adapters = _Adapters(admissions, run_root, fail_at=fail_at)
                memory = _MemoryRuntime()
                with self.assertRaisesRegex(M0EightGPUIntegratedError, expected_cause):
                    EightGPUIntegratedStageMachine(
                        run_root=run_root,
                        exact_session_id=4321,
                        execution_mode="cpu-contract-test",
                    ).run(adapters=adapters, memory=memory)

                self.assertEqual(adapters.calls[-2:], ["rollout:stop", "actor:stop"])
                self.assertEqual(memory.calls[-1], "watchdog:stop")
                self.assertFalse((run_root / "stage-machine.json").exists())


class SecretAbsenceTests(unittest.TestCase):
    def test_secret_absence_audit_rejects_runtime_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = "runtime-secret-value-1234567890"
            (root / "safe.txt").write_text("redacted", encoding="utf-8")
            assert_secret_values_absent(root, (secret,))
            (root / "leak.txt").write_text(secret, encoding="utf-8")
            with self.assertRaisesRegex(M0EightGPUIntegratedError, "secret remains"):
                assert_secret_values_absent(root, (secret,))


class FormalLauncherStaticSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[1]
        cls.launcher = cls.repository / "scripts" / "run_m0_eight_gpu_integrated.sh"
        cls.entry = cls.repository / "scripts" / "run_m0_eight_gpu_integrated.py"
        cls.launcher_text = cls.launcher.read_text(encoding="utf-8")
        cls.entry_text = cls.entry.read_text(encoding="utf-8")

    def test_shell_parses_and_non_tmux_exits_before_remote_setup(self) -> None:
        parsed = subprocess.run(
            ["bash", "-n", str(self.launcher)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        environment = dict(os.environ)
        environment.pop("TMUX", None)
        refused = subprocess.run(
            ["bash", str(self.launcher)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(refused.returncode, 2)
        self.assertIn("must run inside tmux", refused.stderr)

    def test_two_all_gpu_snapshots_and_one_second_runtime_observation_are_literal(self) -> None:
        text = self.launcher_text
        self.assertEqual(text.count('snapshot_all_eight_gpus "preflight"'), 1)
        self.assertEqual(
            text.count('snapshot_all_eight_gpus "immediately-before-scheduler"'),
            1,
        )
        self.assertIn("for gpu_id in 0 1 2 3 4 5 6 7", text)
        self.assertIn("--query-gpu=memory.used,memory.free", text)
        self.assertIn(
            "--query-compute-apps=pid,process_name,used_memory",
            text,
        )
        self.assertIn("ps -o sid=,uid=,user=", text)
        self.assertIn("sleep 1", text)
        self.assertIn('watch_gpu_memory_every_second "${JOB_SID}"', text)
        for obsolete in (
            "26624",
            "30720",
            "MAX_IDLE",
            "MIN_IDLE",
            "require_all_eight_idle",
            "WATCHDOG_BREACH",
        ):
            self.assertNotIn(obsolete, text)

    def test_formal_namespace_exact_sid_cleanup_and_holder_separation_are_literal(self) -> None:
        text = self.launcher_text
        self.assertIn("artifacts/m0-eight-gpu-integrated", text)
        self.assertIn("m0-eight-gpu-integrated.lock", text)
        self.assertIn('session_pids "${JOB_SID}"', text)
        self.assertIn('process_start_time "${JOB_PID}"', text)
        self.assertIn('kill -TERM -- "${pids[@]}"', text)
        self.assertIn('kill -KILL -- "${pids[@]}"', text)
        self.assertNotIn("stop_areal_gpu_holder", text)
        self.assertNotIn("holder-control", text)
        self.assertNotIn("killall", text)
        self.assertNotIn("pkill", text)
        self.assertNotIn("lyy", text)

    def test_commits_clean_tree_secret_redaction_and_real_adapter_are_literal(self) -> None:
        text = self.launcher_text
        self.assertIn("fee938eada49208a5aabdbc1095730a13076a349", text)
        self.assertIn("status --porcelain=v1 --untracked-files=all", text)
        self.assertIn("redact_runtime_admin_key.py", text)
        self.assertIn("--verify-absent", text)
        self.assertIn("RealEightGPUAdapterConfig.from_snapshot_reports", self.entry_text)
        self.assertIn("RealEightGPUIntegratedAdapters(config)", self.entry_text)
        self.assertIn('execution_mode="real-gpu"', self.entry_text)
        self.assertIn('"gpu_execution": True', self.entry_text)
        self.assertNotIn("T/U/V/W/X/Y are not yet", self.entry_text)
        self.assertNotIn("no LocalScheduler or GPU worker", self.entry_text)
        self.assertNotIn('"policy_optimizer_update": True', self.entry_text)
        self.assertNotIn('"harness_optimizer_update": True', self.entry_text)

    def test_nofile_and_snapshot_reports_are_checked_before_scheduler(self) -> None:
        text = self.launcher_text
        scheduler_snapshot = text.index(
            'snapshot_all_eight_gpus "immediately-before-scheduler"'
        )
        self.assertLess(text.index('NOFILE_HARD_LIMIT="$(ulimit -Hn)"'), scheduler_snapshot)
        self.assertLess(text.index('ulimit -Sn "${NOFILE_HARD_LIMIT}"'), scheduler_snapshot)
        self.assertIn("NOFILE_SOFT_LIMIT < 65536", text)
        self.assertIn("qwen2.5-1.5b-snapshot.json", text)
        self.assertIn("gsm8k-snapshot.json", text)
        self.assertIn('--model-report "${MODEL_REPORT}"', text)
        self.assertIn('--dataset-report "${DATASET_REPORT}"', text)

    def test_sglang_child_python3_is_pinned_to_the_areal_venv(self) -> None:
        text = self.launcher_text
        path_export = (
            'export PATH="${AREAL_VENV}/bin:${CUDA_TOOLKIT}/bin:${PATH}"'
        )
        resolution_check = (
            '"$(command -v python3)" != "${AREAL_VENV}/bin/python3"'
        )
        scheduler_snapshot = text.index(
            'snapshot_all_eight_gpus "immediately-before-scheduler"'
        )
        self.assertIn('"${AREAL_VENV}/bin/python3"', text)
        self.assertIn(path_export, text)
        self.assertIn(resolution_check, text)
        self.assertLess(text.index(path_export), scheduler_snapshot)
        self.assertLess(text.index(resolution_check), scheduler_snapshot)

    def test_sglang_jit_is_pinned_to_cuda_twelve_six_before_scheduler(self) -> None:
        text = self.launcher_text
        scheduler_snapshot = text.index(
            'snapshot_all_eight_gpus "immediately-before-scheduler"'
        )
        for fragment in (
            'readonly CUDA_TOOLKIT="/usr/local/cuda-12.6"',
            'export CUDA_HOME="${CUDA_TOOLKIT}"',
            'export CUDA_PATH="${CUDA_TOOLKIT}"',
            '"$(command -v nvcc)" != "${CUDA_TOOLKIT}/bin/nvcc"',
            'nvcc --version | grep -Fq "release 12.6"',
        ):
            self.assertIn(fragment, text)
            self.assertLess(text.index(fragment), scheduler_snapshot)

    def test_execute_mode_constructs_and_runs_the_registered_real_adapter(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "jph_test_run_m0_eight_gpu_integrated",
            self.entry,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_root = Path("/outside-project/formal-run")
        preflight = SimpleNamespace(
            run_root=run_root,
            project_commit="4" * 40,
            record_path=run_root / "preflight.json",
        )
        result = SimpleNamespace(
            state_record_path=run_root / "stage-machine.json",
            selection=SimpleNamespace(
                selection_record_path=run_root / "selection.json"
            ),
            tuvw=SimpleNamespace(record_path=run_root / "tuvw.json"),
            x=SimpleNamespace(record_path=run_root / "x.json"),
            y=SimpleNamespace(record_path=run_root / "y.json"),
        )
        config = object()
        adapter = object()
        memory = object()
        stage_machine = Mock()
        stage_machine.run.return_value = result
        environment = {
            "JPH_ROOT": "/mnt/sdb/ljw/chizm",
            "JPH_PROJECT_DIR": "/mnt/sdb/ljw/chizm/src/project",
            "JPH_AREAL_ROOT": "/mnt/sdb/ljw/chizm/src/AReaL-v2.0.0",
            "JPH_AREAL_ADMIN_API_KEY": "ephemeral-test-secret",
            "TMUX": "/tmp/tmux/default,1,0",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch.object(module, "assert_remote_environment"),
            patch.object(
                module,
                "freeze_integrated_launch_preflight",
                return_value=preflight,
            ),
            patch.object(
                module.RealEightGPUAdapterConfig,
                "from_snapshot_reports",
                return_value=config,
            ) as config_factory,
            patch.object(
                module,
                "RealEightGPUIntegratedAdapters",
                return_value=adapter,
            ) as adapter_factory,
            patch.object(
                module,
                "ThreadedEightGPUMemoryRuntime",
                return_value=memory,
            ),
            patch.object(
                module,
                "EightGPUIntegratedStageMachine",
                return_value=stage_machine,
            ) as stage_factory,
            patch("builtins.print"),
        ):
            module.main(
                [
                    "--entry-mode",
                    "execute",
                    "--run-root",
                    str(run_root),
                    "--expected-project-commit",
                    "4" * 40,
                    "--model-report",
                    "/outside-project/model-report.json",
                    "--dataset-report",
                    "/outside-project/dataset-report.json",
                ]
            )

        config_factory.assert_called_once()
        self.assertEqual(
            config_factory.call_args.kwargs["model_report"],
            Path("/outside-project/model-report.json"),
        )
        self.assertEqual(
            config_factory.call_args.kwargs["dataset_report"],
            Path("/outside-project/dataset-report.json"),
        )
        adapter_factory.assert_called_once_with(config)
        self.assertEqual(stage_factory.call_args.kwargs["execution_mode"], "real-gpu")
        stage_machine.run.assert_called_once_with(adapters=adapter, memory=memory)


if __name__ == "__main__":
    unittest.main()
