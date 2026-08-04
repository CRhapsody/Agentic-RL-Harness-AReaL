from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jphrl.experiments.m0_joint_runner import M0ActivationAssets
from jphrl.experiments.m0_live_joint import (
    M0_GPU_LAUNCH_AUDIT_SCHEMA,
    M0_LIVE_SELECTION_SCHEMA,
    LiveM0GPULaunchGuard,
    M0LiveJointError,
    build_live_production_worker_factory,
    prepare_live_m0_selection,
)
from jphrl.joint_release import ReleaseManifest
from jphrl.trajectory.rlvr_workflow_admission import (
    prepare_rlvr_workflow_joint_admission,
)
from tests.test_rlvr_workflow_admission import _fake_areal_type_import, _source

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency-free interpreter
    torch = None


def _pinned_areal_available() -> bool:
    try:
        return importlib.util.find_spec("areal.api.cli_args") is not None
    except ModuleNotFoundError:
        return False


def _write_real_admissions(root: Path, count: int = 4) -> Path:
    admission_root = root / "runner-admissions"
    admission_root.mkdir(parents=True, mode=0o700)
    for index in range(count):
        bridge, interaction, joint_version, estimator = _source(
            reward=float(index % 2),
            task_id=7 + index,
            controller_kind="torch",
        )
        with _fake_areal_type_import():
            result = prepare_rlvr_workflow_joint_admission(
                bridge,
                pre_batch_interaction=interaction,
                estimator=estimator,
                active_joint_version=joint_version,
            )
        (admission_root / f"runner-admission-{index}.json").write_text(
            json.dumps(result.runner_admission, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
    return admission_root


@unittest.skipIf(torch is None, "torch is unavailable")
class LiveM0SelectionTests(unittest.TestCase):
    def test_exact_four_admissions_freeze_one_training_and_three_holdouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions = _write_real_admissions(root)
            selection = prepare_live_m0_selection(
                runner_admission_dir=admissions,
                selection_root=root / "selection",
            )

            self.assertEqual(
                selection.training_source.runner_admission["episode_trace"]["task_id"],
                "7",
            )
            self.assertEqual(
                [
                    source.runner_admission["episode_trace"]["task_id"]
                    for source in selection.holdout_sources
                ],
                ["8", "9", "10"],
            )
            checkpoint = json.loads(
                selection.harness_behavior_checkpoint.read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint["record_sha256"],
                selection.harness_behavior_record_sha256,
            )
            record = json.loads(
                selection.selection_record_path.read_text(encoding="utf-8")
            )
            self.assertEqual(record["schema_version"], M0_LIVE_SELECTION_SCHEMA)
            self.assertEqual(len(record["holdouts"]), 3)
            self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])
            self.assertFalse(record["evidence_scope"]["harness_optimizer_update"])
            self.assertEqual(
                selection.selection_record_path.stat().st_mode & 0o777,
                0o600,
            )

    def test_count_extra_entry_and_reuse_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions = _write_real_admissions(root, count=3)
            with self.assertRaisesRegex(M0LiveJointError, "exactly four"):
                prepare_live_m0_selection(
                    runner_admission_dir=admissions,
                    selection_root=root / "selection-count",
                )

            admissions = _write_real_admissions(root / "second")
            (admissions / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(M0LiveJointError, "exactly four"):
                prepare_live_m0_selection(
                    runner_admission_dir=admissions,
                    selection_root=root / "selection-extra",
                )

            admissions = _write_real_admissions(root / "third")
            selection_root = root / "selection-existing"
            selection_root.mkdir()
            with self.assertRaisesRegex(M0LiveJointError, "must be new"):
                prepare_live_m0_selection(
                    runner_admission_dir=admissions,
                    selection_root=selection_root,
                )


@unittest.skipIf(torch is None, "torch is unavailable")
class LiveM0ProductionFactoryTests(unittest.TestCase):
    def test_factory_uses_parent_hf_export_and_recorded_sglang_contract(self) -> None:
        if not _pinned_areal_available():
            self.skipTest("pinned AReaL is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admissions = _write_real_admissions(root)
            selection = prepare_live_m0_selection(
                runner_admission_dir=admissions,
                selection_root=root / "selection",
            )
            parent_export = root / "parent-hf"
            parent_export.mkdir()
            candidate_checkpoint = root / "candidate.pt"
            candidate_checkpoint.write_bytes(b"candidate-checkpoint")
            candidate_file_sha = hashlib.sha256(
                candidate_checkpoint.read_bytes()
            ).hexdigest()
            parent_release = ReleaseManifest(
                release_id="1" * 20,
                parent_release_id=None,
                joint_version=selection.training_source.active_joint_version,
                policy_object="objects/policy-parent.json",
                harness_object="objects/harness-parent.json",
            )
            candidate_release = ReleaseManifest(
                release_id="2" * 20,
                parent_release_id=parent_release.release_id,
                joint_version=selection.training_source.active_joint_version,
                policy_object="objects/policy-candidate.json",
                harness_object="objects/harness-candidate.json",
            )
            assets = M0ActivationAssets(
                parent_release=parent_release,
                candidate_release=candidate_release,
                serving_exports=SimpleNamespace(
                    parent=SimpleNamespace(serving_export_path=str(parent_export))
                ),
                parent_joint_version=selection.training_source.active_joint_version,
                candidate_joint_version=selection.training_source.active_joint_version,
                parent_policy_dcp=str(root / "parent-dcp"),
                candidate_policy_dcp=str(root / "candidate-dcp"),
                policy_parent_manifest_sha256="1" * 64,
                policy_candidate_manifest_sha256="2" * 64,
                parent_harness_rollout_checkpoint=str(
                    selection.harness_behavior_checkpoint
                ),
                parent_harness_checkpoint_sha256=(
                    selection.harness_behavior_record_sha256
                ),
                candidate_harness_checkpoint=str(candidate_checkpoint),
                candidate_harness_checkpoint_sha256=candidate_file_sha,
                candidate_harness_parameter_sha256="3" * 64,
                joint_safety_fixture=b"held-out",
                candidate_joint_safety_output_sha256="4" * 64,
                recorded_rollout_sglang_mem_fraction_static=0.29,
                max_new_gpu_memory_gib=None,
            )
            factory = build_live_production_worker_factory(
                selection=selection,
                artifact_root=root / "artifacts",
                physical_gpu_id=3,
                admin_api_key="x" * 48,
                experiment_name="m0-test",
                trial_name="trial-test",
            )
            worker = object()
            with patch(
                "jphrl.training.areal_production_worker."
                "launch_pinned_areal_sglang_activation_worker",
                return_value=worker,
            ) as launch:
                self.assertEqual(tuple(factory(assets)), (worker,))
            arguments = launch.call_args.kwargs
            config = arguments["controller_config"]
            scheduler = arguments["scheduler"]
            server_args = arguments["server_args"]
            self.assertEqual(config.model, str(parent_export))
            self.assertEqual(config.tokenizer_path, str(parent_export))
            self.assertEqual(config.backend, "sglang:d1")
            self.assertEqual(config._version, "v2")
            self.assertEqual(config.admin_api_key, "x" * 48)
            self.assertEqual(scheduler.gpu_devices, [3])
            self.assertEqual(server_args["model_path"], str(parent_export))
            self.assertEqual(server_args["tokenizer_path"], str(parent_export))
            self.assertEqual(server_args["mem_fraction_static"], 0.29)
            self.assertEqual(
                arguments["harness_checkpoints"][
                    parent_release.release_id
                ].checkpoint_sha256,
                selection.harness_behavior_file_sha256,
            )

    def test_short_admin_key_is_rejected_without_launch(self) -> None:
        with self.assertRaisesRegex(M0LiveJointError, "too short"):
            build_live_production_worker_factory(
                selection=SimpleNamespace(),
                artifact_root="/tmp/m0-live-test-artifacts",
                physical_gpu_id=0,
                admin_api_key="short",
                experiment_name="m0-test",
                trial_name="trial-test",
            )


class LiveM0GPULaunchGuardTests(unittest.TestCase):
    def test_training_and_production_snapshots_are_private_and_non_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_root = Path(directory) / "audits"
            guard = LiveM0GPULaunchGuard(physical_gpu_id=3, audit_root=audit_root)
            with (
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}),
                patch(
                    "jphrl.experiments.m0_live_joint._run_nvidia_smi",
                    side_effect=("512, 80000", ""),
                ),
            ):
                self.assertIsNone(guard("training-actor"))
            training = json.loads(
                (audit_root / "training-actor.json").read_text(encoding="utf-8")
            )
            self.assertEqual(training["schema_version"], M0_GPU_LAUNCH_AUDIT_SCHEMA)
            self.assertTrue(training["passed"])
            self.assertEqual(training["compute_processes"], [])
            self.assertFalse(training["evidence_scope"]["policy_optimizer_update"])
            self.assertEqual(
                (audit_root / "training-actor.json").stat().st_mode & 0o777,
                0o600,
            )

            current = os.getpid()
            with (
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}),
                patch(
                    "jphrl.experiments.m0_live_joint._run_nvidia_smi",
                    side_effect=(
                        "768, 79744",
                        f"{current}, python, 768",
                    ),
                ),
            ):
                self.assertIsNone(guard("production-sglang"))

    def test_foreign_process_is_observed_but_wrong_visibility_and_reuse_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_root = Path(directory) / "audits"
            guard = LiveM0GPULaunchGuard(physical_gpu_id=2, audit_root=audit_root)
            with (
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}),
                patch(
                    "jphrl.experiments.m0_live_joint._run_nvidia_smi",
                    side_effect=("1024, 79000", "999999, foreign, 256"),
                ),
            ):
                self.assertIsNone(guard("training-actor"))
            observed = json.loads(
                (audit_root / "training-actor.json").read_text(encoding="utf-8")
            )
            self.assertTrue(observed["passed"])
            self.assertFalse(observed["memory_limit_enforced"])
            self.assertTrue(observed["compute_processes_observation_only"])
            self.assertEqual(observed["observed_compute_process_count"], 1)

            with (
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}),
                self.assertRaisesRegex(M0LiveJointError, "CUDA_VISIBLE_DEVICES"),
            ):
                guard("production-sglang")

            with (
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}),
                patch(
                    "jphrl.experiments.m0_live_joint._run_nvidia_smi",
                    side_effect=("512, 80000", ""),
                ),
                self.assertRaises(FileExistsError),
            ):
                guard("training-actor")


if __name__ == "__main__":
    unittest.main()
