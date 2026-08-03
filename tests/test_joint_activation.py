from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from jphrl.joint_release import (
    CandidateArtifact,
    ConcurrentPublishError,
    JointReleaseStore,
    ReleaseManifest,
)
from jphrl.paths import repository_root
from jphrl.training.joint_activation import (
    _PRODUCTION_AUTHORIZATION_TOKEN,
    ACTIVATION_STAGES,
    InjectedActivationCrash,
    JointActivationCallbacks,
    JointActivationController,
    JointActivationError,
    JointActivationRecoveryRequired,
    JointActivationRollbackError,
    ProductionActivationAuthorization,
    ProductionActivationWorker,
    ProductionJointActivationController,
    ProductionProbeSpec,
    ProductionReleaseTarget,
    ProductionWorkerState,
    _ProductionWorkerBridge,
    accepted_report_receipt,
    callback_receipt,
    require_production_activation_authorization,
)
from jphrl.trajectory.schema import JointVersion


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _version(policy: str, harness: str) -> JointVersion:
    return JointVersion(
        policy=policy,
        harness_controller=harness,
        harness_artifact="harness-artifact-v1",
        tool_schema="tool-schema-v1",
        parser="parser-v1",
        environment="environment-v1",
        evaluator="evaluator-v1",
        tokenizer="tokenizer-v1",
        context_builder="context-builder-v1",
    )


def _artifact(component: str, version: str, marker: str) -> CandidateArtifact:
    return CandidateArtifact(
        component=component,
        version=version,
        payload={"marker": marker},
    )


class _CallbackHarness:
    def __init__(
        self,
        *,
        worker_ids: tuple[str, ...],
        fail_operations: set[str] | None = None,
        partial_operation: str | None = None,
    ) -> None:
        self.worker_ids = worker_ids
        self.fail_operations = fail_operations or set()
        self.partial_operation = partial_operation
        self.events: list[tuple[str, str]] = []
        self.observed_pairs: list[tuple[str, str]] = []

    def callback(
        self,
        operation: str,
        release: ReleaseManifest,
    ) -> dict[str, object]:
        self.events.append((operation, release.release_id))
        if operation in self.fail_operations:
            raise RuntimeError(f"injected callback failure: {operation}")
        if operation in {"quiesce", "rollback_quiesce"}:
            observations: dict[str, object] = {"quiesced": True}
        elif operation in {"sync_policy", "restore_policy"}:
            observations = {"policy_version": release.joint_version.policy}
        elif operation in {"stage_harness", "restore_harness"}:
            observations = {
                "harness_controller_version": (release.joint_version.harness_controller)
            }
        elif operation in {
            "verify_sync",
            "set_versions",
            "set_parent_versions",
            "post_publish_probe",
            "parent_probe",
        }:
            worker_versions = {
                worker_id: {
                    "policy": release.joint_version.policy,
                    "harness_controller": (release.joint_version.harness_controller),
                }
                for worker_id in self.worker_ids
            }
            if operation == self.partial_operation:
                worker_versions[self.worker_ids[-1]] = {
                    "policy": release.joint_version.policy,
                    "harness_controller": "stale-harness-version",
                }
            for pair in worker_versions.values():
                self.observed_pairs.append(
                    (str(pair["policy"]), str(pair["harness_controller"]))
                )
            observations = {"worker_versions": worker_versions}
            if operation in {"post_publish_probe", "parent_probe"}:
                observations["probe_passed"] = True
        elif operation in {"resume", "resume_parent"}:
            observations = {"resumed": True}
        else:  # pragma: no cover - test adapter contract guard
            raise AssertionError(f"unexpected callback operation {operation}")
        return callback_receipt(
            operation=operation,
            release=release,
            observations=observations,
            control_plane_only=True,
        )

    def callbacks(self) -> JointActivationCallbacks:
        return JointActivationCallbacks(
            quiesce=self.callback,
            sync_policy=self.callback,
            stage_harness=self.callback,
            verify_sync=self.callback,
            set_versions=self.callback,
            probe=self.callback,
            resume=self.callback,
            restore_policy=self.callback,
            restore_harness=self.callback,
        )


class _ProductionWorkerHarness(ProductionActivationWorker):
    def __init__(
        self,
        target: ProductionReleaseTarget,
        *,
        return_mapping_state: bool = False,
        self_report_install: bool = False,
    ) -> None:
        self._worker_id = "production-worker-0"
        self.target = target
        self.policy_engine_version = target.policy_engine_version
        self.policy_checkpoint_sha256 = target.policy_checkpoint_sha256
        self.harness_controller_version = target.joint_version.harness_controller
        self.harness_checkpoint_sha256 = target.harness_checkpoint_sha256
        self.harness_parameter_digest = target.harness_parameter_digest
        self.lifecycle = "serving"
        self.return_mapping_state = return_mapping_state
        self.self_report_install = self_report_install

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def quiesce(self) -> None:
        self.lifecycle = "quiesced"

    def install_policy(self, target: ProductionReleaseTarget) -> None:
        self.policy_engine_version = target.policy_engine_version
        self.policy_checkpoint_sha256 = target.policy_checkpoint_sha256
        if self.self_report_install:
            return {"success": True}  # type: ignore[return-value]
        return None

    def install_harness(self, target: ProductionReleaseTarget) -> None:
        self.harness_controller_version = target.joint_version.harness_controller
        self.harness_checkpoint_sha256 = target.harness_checkpoint_sha256
        self.harness_parameter_digest = target.harness_parameter_digest

    def bind_release(self, target: ProductionReleaseTarget) -> None:
        self.target = target

    def run_probe(self, fixture: bytes) -> bytes:
        return fixture

    def resume(self) -> None:
        self.lifecycle = "serving"

    def read_state(self) -> ProductionWorkerState:
        state = ProductionWorkerState(
            worker_id=self.worker_id,
            lifecycle_phase=self.lifecycle,
            active_release_id=self.target.release_id,
            joint_version=self.target.joint_version,
            policy_engine_version=self.policy_engine_version,
            policy_checkpoint_sha256=self.policy_checkpoint_sha256,
            harness_controller_version=self.harness_controller_version,
            harness_checkpoint_sha256=self.harness_checkpoint_sha256,
            harness_parameter_digest=self.harness_parameter_digest,
        )
        if self.return_mapping_state:
            return state.to_record()  # type: ignore[return-value]
        return state


class JointActivationTests(unittest.TestCase):
    worker_ids = ("worker-0", "worker-1")
    accepted_x_report: ClassVar[dict[str, object]] = {
        "decision": "accept",
        "gate": "X",
        "score": 1.0,
    }

    def _release_pair(
        self,
        root: Path,
        *,
        candidate_marker: str = "candidate",
    ) -> tuple[JointReleaseStore, ReleaseManifest, ReleaseManifest]:
        store = JointReleaseStore(root / "runtime" / "releases")
        parent_version = _version("policy-parent", "harness-parent")
        parent = store.publish(
            joint_version=parent_version,
            policy=_artifact("policy", parent_version.policy, "parent-policy"),
            harness=_artifact(
                "harness",
                parent_version.harness_controller,
                "parent-harness",
            ),
            expected_active_release_id=None,
        )
        candidate_version = _version(
            f"policy-{candidate_marker}",
            f"harness-{candidate_marker}",
        )
        candidate = store.stage(
            joint_version=candidate_version,
            policy=_artifact(
                "policy",
                candidate_version.policy,
                f"{candidate_marker}-policy",
            ),
            harness=_artifact(
                "harness",
                candidate_version.harness_controller,
                f"{candidate_marker}-harness",
            ),
            expected_active_release_id=parent.release_id,
        )
        return store, parent, candidate

    def _production_scenario(
        self,
        root: Path,
    ) -> tuple[
        JointReleaseStore,
        ReleaseManifest,
        ReleaseManifest,
        ProductionReleaseTarget,
        ProductionReleaseTarget,
        dict[str, ProductionProbeSpec],
        ProductionActivationAuthorization,
        _ProductionWorkerHarness,
    ]:
        store = JointReleaseStore(root / "runtime" / "production-releases")
        parent_version = _version("policy-parent", "harness-parent")
        parent = store.publish(
            joint_version=parent_version,
            policy=_artifact("policy", parent_version.policy, "bootstrap-policy"),
            harness=_artifact(
                "harness",
                parent_version.harness_controller,
                "bootstrap-harness",
            ),
            expected_active_release_id=None,
        )
        candidate_version = _version("policy-candidate", "harness-candidate")
        candidate_policy_sha = "4" * 64
        candidate_harness_sha = "5" * 64
        candidate_parameter_sha = "6" * 64
        candidate = store.stage(
            joint_version=candidate_version,
            policy=CandidateArtifact(
                component="policy",
                version=candidate_version.policy,
                payload={
                    "schema_version": "jph.production-policy-release-artifact.v2",
                    "candidate_joint_version_id": candidate_version.version_id,
                    "joint_candidate_bundle_sha256": "7" * 64,
                    "production_checkpoint_manifest_sha256": "8" * 64,
                    "policy_update_receipt_sha256": "9" * 64,
                    "policy_checkpoint_manifest_sha256": "b" * 64,
                    "policy_engine_version": 5,
                    "policy_serving_export_manifest_sha256": "c" * 64,
                    "policy_serving_parameter_sha256": candidate_policy_sha,
                    "policy_serving_lineage_record_sha256": "d" * 64,
                },
            ),
            harness=CandidateArtifact(
                component="harness",
                version=candidate_version.harness_controller,
                payload={
                    "schema_version": "jph.production-harness-release-artifact.v1",
                    "candidate_joint_version_id": candidate_version.version_id,
                    "joint_candidate_bundle_sha256": "7" * 64,
                    "production_checkpoint_manifest_sha256": "8" * 64,
                    "harness_update_receipt_sha256": "a" * 64,
                    "harness_checkpoint_sha256": candidate_harness_sha,
                    "harness_parameter_sha256": candidate_parameter_sha,
                },
            ),
            expected_active_release_id=parent.release_id,
        )
        parent_target = ProductionReleaseTarget(
            release_id=parent.release_id,
            joint_version=parent.joint_version,
            policy_engine_version=4,
            policy_checkpoint_sha256="1" * 64,
            harness_checkpoint_sha256="2" * 64,
            harness_parameter_digest="3" * 64,
        )
        candidate_target = ProductionReleaseTarget(
            release_id=candidate.release_id,
            joint_version=candidate.joint_version,
            policy_engine_version=5,
            policy_checkpoint_sha256=candidate_policy_sha,
            harness_checkpoint_sha256=candidate_harness_sha,
            harness_parameter_digest=candidate_parameter_sha,
        )
        parent_fixture = b"raw-parent-fixture-never-persist"
        candidate_fixture = b"raw-candidate-fixture-never-persist"
        probes = {
            parent.release_id: ProductionProbeSpec(
                probe_id="parent-production-probe",
                fixture=parent_fixture,
                fixture_sha256=hashlib.sha256(parent_fixture).hexdigest(),
                expected_output_sha256=hashlib.sha256(parent_fixture).hexdigest(),
            ),
            candidate.release_id: ProductionProbeSpec(
                probe_id="candidate-production-probe",
                fixture=candidate_fixture,
                fixture_sha256=hashlib.sha256(candidate_fixture).hexdigest(),
                expected_output_sha256=hashlib.sha256(candidate_fixture).hexdigest(),
            ),
        }
        probe_records = {
            release_id: probes[release_id].to_record() for release_id in sorted(probes)
        }
        authorization = ProductionActivationAuthorization._create(
            parent=parent_target,
            candidate=candidate_target,
            acceptance_record_sha256="b" * 64,
            exact_recovery_record_sha256="c" * 64,
            probe_specs_sha256=_canonical_sha256(probe_records),
            token=_PRODUCTION_AUTHORIZATION_TOKEN,
        )
        worker = _ProductionWorkerHarness(parent_target)
        return (
            store,
            parent,
            candidate,
            parent_target,
            candidate_target,
            probes,
            authorization,
            worker,
        )

    def _controller(
        self,
        root: Path,
        *,
        store: JointReleaseStore,
        parent: ReleaseManifest,
        candidate: ReleaseManifest,
        harness: _CallbackHarness,
    ) -> JointActivationController:
        project = repository_root()

        def validate(record: dict[str, object]) -> dict[str, object]:
            self.assertEqual(record, self.accepted_x_report)
            return accepted_report_receipt(
                accepted_x_report=record,
                parent_release_id=parent.release_id,
                candidate_release_id=candidate.release_id,
                control_plane_only=True,
            )

        controller = JointActivationController(
            store=store,
            journal_root=store.activation_journal_root,
            project_root=project,
            worker_ids=self.worker_ids,
            callbacks=harness.callbacks(),
            validate_accepted_report=validate,
            control_plane_only=True,
        )
        return controller

    def _activate(
        self,
        controller: JointActivationController,
        parent: ReleaseManifest,
        candidate: ReleaseManifest,
        **kwargs: object,
    ):
        return controller.activate(
            accepted_x_report=self.accepted_x_report,
            parent_release_id=parent.release_id,
            candidate_release_id=candidate.release_id,
            **kwargs,
        )

    def test_release_store_stage_activate_and_rollback_are_distinct_cas_steps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)

                self.assertEqual(store.read_active(), parent)
                self.assertEqual(store.read_manifest(candidate.release_id), candidate)
                activated = store.activate(
                    release_id=candidate.release_id,
                    expected_active_release_id=parent.release_id,
                )
                restored = store.rollback(
                    target_release_id=parent.release_id,
                    expected_active_release_id=candidate.release_id,
                )

                self.assertEqual(activated, candidate)
                self.assertEqual(restored, parent)
                self.assertEqual(store.read_active(), parent)
                with self.assertRaises(ConcurrentPublishError):
                    store.activate(
                        release_id=candidate.release_id,
                        expected_active_release_id=candidate.release_id,
                    )

    def test_concurrent_activate_cas_allows_exactly_one_staged_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, first = self._release_pair(
                    root, candidate_marker="first"
                )
                second_version = _version("policy-second", "harness-second")
                second = store.stage(
                    joint_version=second_version,
                    policy=_artifact("policy", second_version.policy, "second-policy"),
                    harness=_artifact(
                        "harness",
                        second_version.harness_controller,
                        "second-harness",
                    ),
                    expected_active_release_id=parent.release_id,
                )

                def attempt(candidate: ReleaseManifest) -> str:
                    try:
                        store.activate(
                            release_id=candidate.release_id,
                            expected_active_release_id=parent.release_id,
                        )
                    except ConcurrentPublishError:
                        return "lost"
                    return candidate.release_id

                with ThreadPoolExecutor(max_workers=2) as executor:
                    outcomes = list(executor.map(attempt, (first, second)))

                winners = [outcome for outcome in outcomes if outcome != "lost"]
                self.assertEqual(len(winners), 1)
                active = store.read_active()
                self.assertIsNotNone(active)
                self.assertEqual(active.release_id, winners[0])
                self.assertIn(winners[0], {first.release_id, second.release_id})

    def test_success_uses_unskipped_journal_and_claims_only_cpu_control_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                harness = _CallbackHarness(worker_ids=self.worker_ids)
                controller = self._controller(
                    root,
                    store=store,
                    parent=parent,
                    candidate=candidate,
                    harness=harness,
                )

                result = self._activate(controller, parent, candidate)
                journal = json.loads(result.journal_path.read_text(encoding="utf-8"))

                self.assertEqual(result.active_release_id, candidate.release_id)
                self.assertEqual(store.read_active(), candidate)
                self.assertEqual(journal["stage"], "RESUMED")
                self.assertEqual(journal["stage_index"], len(ACTIVATION_STAGES) - 1)
                self.assertEqual(set(journal["receipts"]), set(ACTIVATION_STAGES))
                self.assertEqual(journal["status"], "candidate_active")
                self.assertFalse(
                    result.evidence_scope["real_inference_weight_sync_verified"]
                )
                self.assertFalse(result.evidence_scope["real_harness_install_verified"])
                self.assertFalse(
                    result.evidence_scope["production_distributed_probe_verified"]
                )
                self.assertEqual(
                    os.stat(controller.journal_root).st_mode & 0o777, 0o700
                )
                self.assertEqual(os.stat(result.journal_path).st_mode & 0o777, 0o600)

    def test_fault_after_every_stage_restores_and_resumes_only_parent_pair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                allowed_pairs = {
                    (
                        parent.joint_version.policy,
                        parent.joint_version.harness_controller,
                    ),
                    (
                        candidate.joint_version.policy,
                        candidate.joint_version.harness_controller,
                    ),
                }
                for stage in ACTIVATION_STAGES:
                    with self.subTest(stage=stage):
                        harness = _CallbackHarness(worker_ids=self.worker_ids)
                        controller = self._controller(
                            root,
                            store=store,
                            parent=parent,
                            candidate=candidate,
                            harness=harness,
                        )

                        with self.assertRaisesRegex(
                            JointActivationError,
                            "parent pair was restored",
                        ):
                            self._activate(
                                controller,
                                parent,
                                candidate,
                                fault_after_stage=stage,
                            )

                        self.assertEqual(store.read_active(), parent)
                        resume_events = [
                            event
                            for event in harness.events
                            if event[0].startswith("resume")
                        ]
                        self.assertEqual(
                            resume_events[-1], ("resume_parent", parent.release_id)
                        )
                        self.assertTrue(
                            all(
                                pair in allowed_pairs for pair in harness.observed_pairs
                            )
                        )
                        journal = json.loads(
                            controller._journal_paths()[-1].read_text(encoding="utf-8")
                        )
                        self.assertEqual(journal["status"], "parent_restored")

    def test_partial_worker_version_receipt_rolls_back_without_candidate_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                harness = _CallbackHarness(
                    worker_ids=self.worker_ids,
                    partial_operation="set_versions",
                )
                controller = self._controller(
                    root,
                    store=store,
                    parent=parent,
                    candidate=candidate,
                    harness=harness,
                )

                with self.assertRaisesRegex(
                    JointActivationError,
                    "parent pair was restored",
                ):
                    self._activate(controller, parent, candidate)

                self.assertEqual(store.read_active(), parent)
                self.assertNotIn(("resume", candidate.release_id), harness.events)
                parent_probe_index = harness.events.index(
                    ("parent_probe", parent.release_id)
                )
                parent_resume_index = harness.events.index(
                    ("resume_parent", parent.release_id)
                )
                self.assertLess(parent_probe_index, parent_resume_index)

    def test_crash_journal_blocks_new_resume_until_parent_recovery_probe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                harness = _CallbackHarness(worker_ids=self.worker_ids)
                controller = self._controller(
                    root,
                    store=store,
                    parent=parent,
                    candidate=candidate,
                    harness=harness,
                )

                with self.assertRaises(InjectedActivationCrash):
                    self._activate(
                        controller,
                        parent,
                        candidate,
                        fault_after_stage="VERSIONS_SET",
                        fault_mode="crash",
                    )
                event_count = len(harness.events)
                with self.assertRaises(JointActivationRecoveryRequired):
                    self._activate(controller, parent, candidate)

                self.assertEqual(len(harness.events), event_count)
                self.assertNotIn(("resume", candidate.release_id), harness.events)
                recovered = controller.recover_pending()

                self.assertEqual(recovered.outcome, "parent_restored")
                self.assertEqual(store.read_active(), parent)
                self.assertLess(
                    harness.events.index(("parent_probe", parent.release_id)),
                    harness.events.index(("resume_parent", parent.release_id)),
                )
                self.assertEqual(controller.pending_journals(), ())

    def test_incomplete_paired_restore_fails_closed_and_never_resumes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                harness = _CallbackHarness(
                    worker_ids=self.worker_ids,
                    fail_operations={"sync_policy", "restore_policy"},
                )
                controller = self._controller(
                    root,
                    store=store,
                    parent=parent,
                    candidate=candidate,
                    harness=harness,
                )

                with self.assertRaises(JointActivationRollbackError):
                    self._activate(controller, parent, candidate)

                self.assertEqual(store.read_active(), parent)
                self.assertIn(("restore_policy", parent.release_id), harness.events)
                self.assertIn(("restore_harness", parent.release_id), harness.events)
                self.assertNotIn(("parent_probe", parent.release_id), harness.events)
                self.assertNotIn(("resume_parent", parent.release_id), harness.events)
                self.assertEqual(len(controller.pending_journals()), 1)
                journal = json.loads(
                    controller.pending_journals()[0].read_text(encoding="utf-8")
                )
                self.assertEqual(journal["status"], "fail_closed")
                serialized = json.dumps(journal, sort_keys=True)
                self.assertNotIn("admin_api_key", serialized)
                self.assertNotIn("injected callback failure", serialized)
                self.assertIn("builtins.RuntimeError", serialized)

    def test_tampered_skipped_stage_journal_blocks_recovery_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                harness = _CallbackHarness(worker_ids=self.worker_ids)
                controller = self._controller(
                    root,
                    store=store,
                    parent=parent,
                    candidate=candidate,
                    harness=harness,
                )
                with self.assertRaises(InjectedActivationCrash):
                    self._activate(
                        controller,
                        parent,
                        candidate,
                        fault_after_stage="PREPARED",
                        fault_mode="crash",
                    )
                path = controller._journal_paths()[0]
                journal = json.loads(path.read_text(encoding="utf-8"))
                journal["stage"] = "HARNESS_STAGED"
                journal["stage_index"] = ACTIVATION_STAGES.index("HARNESS_STAGED")
                unsigned = {
                    key: value
                    for key, value in journal.items()
                    if key != "record_sha256"
                }
                journal["record_sha256"] = _canonical_sha256(unsigned)
                path.write_text(
                    json.dumps(journal, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8",
                )
                event_count = len(harness.events)

                with self.assertRaisesRegex(
                    JointActivationError,
                    "skipped or out of order",
                ):
                    controller.recover_pending()

                self.assertEqual(len(harness.events), event_count)
                self.assertNotIn(("resume_parent", parent.release_id), harness.events)

    def test_journal_inside_git_project_and_non_strict_x_receipt_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                harness = _CallbackHarness(worker_ids=self.worker_ids)
                project = root / "not-the-real-project"
                with self.assertRaisesRegex(ValueError, "actual checkout"):
                    JointActivationController(
                        store=store,
                        journal_root=project / "journals",
                        project_root=project,
                        worker_ids=self.worker_ids,
                        callbacks=harness.callbacks(),
                        validate_accepted_report=lambda report: report,
                        control_plane_only=True,
                    )

                report = {"decision": "accept"}

                def invalid_validator(record: dict[str, object]) -> dict[str, object]:
                    receipt = accepted_report_receipt(
                        accepted_x_report=record,
                        parent_release_id=parent.release_id,
                        candidate_release_id=candidate.release_id,
                        control_plane_only=True,
                    )
                    receipt["unexpected"] = True
                    return receipt

                controller = JointActivationController(
                    store=store,
                    journal_root=store.activation_journal_root,
                    project_root=repository_root(),
                    worker_ids=self.worker_ids,
                    callbacks=harness.callbacks(),
                    validate_accepted_report=invalid_validator,
                    control_plane_only=True,
                )
                with self.assertRaisesRegex(
                    JointActivationError,
                    "field set differs",
                ):
                    controller.activate(
                        accepted_x_report=report,
                        parent_release_id=parent.release_id,
                        candidate_release_id=candidate.release_id,
                    )

                self.assertEqual(harness.events, [])
                self.assertEqual(controller._journal_paths(), ())

    def test_generic_receipts_and_controller_cannot_claim_production_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, candidate = self._release_pair(root)
                harness = _CallbackHarness(worker_ids=self.worker_ids)

                with self.assertRaisesRegex(
                    JointActivationError,
                    "cannot authorize production activation",
                ):
                    accepted_report_receipt(
                        accepted_x_report=self.accepted_x_report,
                        parent_release_id=parent.release_id,
                        candidate_release_id=candidate.release_id,
                        control_plane_only=False,
                    )
                with self.assertRaisesRegex(
                    JointActivationError,
                    "credential field",
                ):
                    callback_receipt(
                        operation="quiesce",
                        release=parent,
                        observations={
                            "quiesced": True,
                            "nested": {"deployment_access_token": "must-not-persist"},
                        },
                        control_plane_only=True,
                    )
                with self.assertRaisesRegex(
                    JointActivationError,
                    "cannot authorize production activation",
                ):
                    callback_receipt(
                        operation="quiesce",
                        release=parent,
                        observations={"quiesced": True},
                        control_plane_only=False,
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "cannot authorize production activation",
                ):
                    JointActivationController(
                        store=store,
                        journal_root=store.activation_journal_root,
                        project_root=repository_root(),
                        worker_ids=self.worker_ids,
                        callbacks=harness.callbacks(),
                        validate_accepted_report=lambda report: report,
                        control_plane_only=False,
                    )

                self.assertEqual(store.read_active(), parent)
                self.assertEqual(harness.events, [])

                with self.assertRaisesRegex(
                    JointActivationError,
                    "native live W/X authorization",
                ):
                    require_production_activation_authorization(
                        {
                            "accepted": True,
                            "parent_release_id": parent.release_id,
                            "candidate_release_id": candidate.release_id,
                        }
                    )
                with self.assertRaises(TypeError):
                    ProductionActivationAuthorization(  # type: ignore[call-arg]
                        parent={},
                        candidate={},
                        acceptance_record_sha256="0" * 64,
                        exact_recovery_record_sha256="1" * 64,
                        _token=object(),
                    )
                with self.assertRaisesRegex(
                    JointActivationError,
                    "typed adapter",
                ):
                    ProductionJointActivationController(
                        store=store,
                        workers=(harness,),  # type: ignore[arg-type]
                        probes={},
                        project_root=repository_root(),
                    )

    def test_static_control_plane_receipts_cannot_activate_production_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                store, parent, _ = self._release_pair(root)
                candidate_version = _version(
                    "policy-production",
                    "harness-production",
                )
                candidate = store.stage(
                    joint_version=candidate_version,
                    policy=CandidateArtifact(
                        component="policy",
                        version=candidate_version.policy,
                        payload={
                            "schema_version": (
                                "jph.production-policy-release-artifact.v2"
                            )
                        },
                    ),
                    harness=CandidateArtifact(
                        component="harness",
                        version=candidate_version.harness_controller,
                        payload={
                            "schema_version": (
                                "jph.production-harness-release-artifact.v1"
                            )
                        },
                    ),
                    expected_active_release_id=parent.release_id,
                )
                harness = _CallbackHarness(worker_ids=self.worker_ids)
                controller = self._controller(
                    root,
                    store=store,
                    parent=parent,
                    candidate=candidate,
                    harness=harness,
                )

                with self.assertRaisesRegex(
                    JointActivationError,
                    "cannot activate production artifacts",
                ):
                    self._activate(controller, parent, candidate)

                self.assertEqual(store.read_active(), parent)
                self.assertEqual(harness.events, [])
                self.assertEqual(controller._journal_paths(), ())

    def test_production_bridge_measures_state_and_rejects_self_reported_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                _, parent, candidate = self._release_pair(root)
                parent_target = ProductionReleaseTarget(
                    release_id=parent.release_id,
                    joint_version=parent.joint_version,
                    policy_engine_version=4,
                    policy_checkpoint_sha256="1" * 64,
                    harness_checkpoint_sha256="2" * 64,
                    harness_parameter_digest="3" * 64,
                )
                candidate_target = ProductionReleaseTarget(
                    release_id=candidate.release_id,
                    joint_version=candidate.joint_version,
                    policy_engine_version=5,
                    policy_checkpoint_sha256="4" * 64,
                    harness_checkpoint_sha256="5" * 64,
                    harness_parameter_digest="6" * 64,
                )
                parent_fixture = b"parent-probe-observations"
                candidate_fixture = b"candidate-probe-observations"
                probes = {
                    parent.release_id: ProductionProbeSpec(
                        probe_id="parent-probe",
                        fixture=parent_fixture,
                        fixture_sha256=hashlib.sha256(parent_fixture).hexdigest(),
                        expected_output_sha256=(
                            hashlib.sha256(parent_fixture).hexdigest()
                        ),
                    ),
                    candidate.release_id: ProductionProbeSpec(
                        probe_id="candidate-probe",
                        fixture=candidate_fixture,
                        fixture_sha256=hashlib.sha256(candidate_fixture).hexdigest(),
                        expected_output_sha256=(
                            hashlib.sha256(candidate_fixture).hexdigest()
                        ),
                    ),
                }
                worker = _ProductionWorkerHarness(parent_target)
                bridge = _ProductionWorkerBridge(
                    workers=(worker,),
                    parent=parent_target,
                    candidate=candidate_target,
                    probes=probes,
                )
                for operation, release in (
                    ("quiesce", parent),
                    ("sync_policy", candidate),
                    ("stage_harness", candidate),
                    ("verify_sync", candidate),
                    ("set_versions", candidate),
                    ("post_publish_probe", candidate),
                    ("resume", candidate),
                    ("rollback_quiesce", candidate),
                    ("restore_policy", parent),
                    ("restore_harness", parent),
                    ("set_parent_versions", parent),
                    ("parent_probe", parent),
                    ("resume_parent", parent),
                ):
                    bridge.callback(operation, release)

                final = worker.read_state()
                self.assertEqual(final.active_release_id, parent.release_id)
                self.assertEqual(final.lifecycle_phase, "serving")
                self.assertTrue(
                    any(record["probe_output_sha256"] for record in bridge.evidence)
                )

                mapping_worker = _ProductionWorkerHarness(
                    parent_target,
                    return_mapping_state=True,
                )
                mapping_bridge = _ProductionWorkerBridge(
                    workers=(mapping_worker,),
                    parent=parent_target,
                    candidate=candidate_target,
                    probes=probes,
                )
                with self.assertRaisesRegex(
                    JointActivationError,
                    "untyped state mapping",
                ):
                    mapping_bridge.callback("quiesce", parent)

                reporting_worker = _ProductionWorkerHarness(
                    parent_target,
                    self_report_install=True,
                )
                reporting_bridge = _ProductionWorkerBridge(
                    workers=(reporting_worker,),
                    parent=parent_target,
                    candidate=candidate_target,
                    probes=probes,
                )
                reporting_bridge.callback("quiesce", parent)
                with self.assertRaisesRegex(
                    JointActivationError,
                    "returned self-reported evidence",
                ):
                    reporting_bridge.callback("sync_policy", candidate)

    def test_fresh_process_rollback_only_recovers_every_crash_boundary(self) -> None:
        crash_stages = (
            "PREPARED",
            "QUIESCED",
            "POLICY_SYNCED",
            "HARNESS_STAGED",
            "ACTIVE_POINTER_SWITCHED",
            "VERSIONS_SET",
            "POST_PUBLISH_VERIFIED",
            "RESUMED",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                (
                    store,
                    parent,
                    candidate,
                    parent_target,
                    _,
                    probes,
                    authorization,
                    worker,
                ) = self._production_scenario(root)
                pre_journal_controller = ProductionJointActivationController(
                    store=store,
                    workers=(worker,),
                    probes=probes,
                    project_root=repository_root(),
                )
                pre_journal_record, sealed = (
                    pre_journal_controller._seal_rollback_only_record(
                        activation_id="d" * 32,
                        authorization=authorization,
                    )
                )
                self.assertEqual(sealed["mode"], "rollback_only")
                self.assertFalse(
                    sealed["evidence_scope"]["forward_activation_authorized"]
                )
                self.assertEqual(pre_journal_record.stat().st_mode & 0o777, 0o600)
                self.assertEqual(
                    pre_journal_record.parent.stat().st_mode & 0o777,
                    0o700,
                )
                sealed_text = pre_journal_record.read_text(encoding="utf-8")
                self.assertNotIn("session_api_key", sealed_text)
                self.assertNotIn("admin_api_key", sealed_text)
                self.assertNotIn("raw-parent-fixture-never-persist", sealed_text)
                pre_journal_recovered = pre_journal_controller.recover_pending(
                    pre_journal_record
                )
                self.assertEqual(
                    pre_journal_recovered.outcome,
                    "parent_restored_without_journal",
                )
                for stage in crash_stages:
                    with self.subTest(stage=stage):
                        before_records = set(
                            (store.root / "activation-rollback-only").glob("*.json")
                        )
                        controller = ProductionJointActivationController(
                            store=store,
                            workers=(worker,),
                            probes=probes,
                            project_root=repository_root(),
                        )
                        with self.assertRaises(InjectedActivationCrash):
                            controller.activate(
                                authorization,
                                fault_after_stage=stage,
                                fault_mode="crash",
                            )
                        after_records = set(
                            (store.root / "activation-rollback-only").glob("*.json")
                        )
                        new_records = after_records - before_records
                        self.assertEqual(len(new_records), 1)
                        rollback_record = new_records.pop()

                        fresh_controller = ProductionJointActivationController(
                            store=store,
                            workers=(worker,),
                            probes=probes,
                            project_root=repository_root(),
                        )
                        recovered = fresh_controller.recover_pending(rollback_record)

                        self.assertEqual(recovered.active_release_id, parent.release_id)
                        self.assertEqual(store.read_active(), parent)
                        final = worker.read_state()
                        self.assertEqual(final.active_release_id, parent.release_id)
                        self.assertEqual(final.joint_version, parent.joint_version)
                        self.assertEqual(
                            final.policy_checkpoint_sha256,
                            parent_target.policy_checkpoint_sha256,
                        )
                        self.assertEqual(
                            final.harness_checkpoint_sha256,
                            parent_target.harness_checkpoint_sha256,
                        )
                        self.assertEqual(final.lifecycle_phase, "serving")
                        journal = json.loads(
                            recovered.journal_path.read_text(encoding="utf-8")
                        )
                        self.assertEqual(journal["status"], "parent_restored")
                        self.assertEqual(
                            journal["candidate_release_id"], candidate.release_id
                        )

    def test_rollback_only_tamper_secret_stale_store_and_forward_forgery_fail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                (
                    store,
                    parent,
                    candidate,
                    _,
                    _,
                    probes,
                    authorization,
                    worker,
                ) = self._production_scenario(root)
                controller = ProductionJointActivationController(
                    store=store,
                    workers=(worker,),
                    probes=probes,
                    project_root=repository_root(),
                )
                with self.assertRaises(InjectedActivationCrash):
                    controller.activate(
                        authorization,
                        fault_after_stage="POLICY_SYNCED",
                        fault_mode="crash",
                    )
                rollback_record = next(
                    (store.root / "activation-rollback-only").glob("*.json")
                )
                symlink = rollback_record.parent / "rollback-symlink.json"
                symlink.symlink_to(rollback_record)
                with self.assertRaisesRegex(
                    JointActivationError,
                    "cannot be a symlink",
                ):
                    controller.recover_pending(symlink)
                record = json.loads(rollback_record.read_text(encoding="utf-8"))
                record["parent_target"]["policy_checkpoint_sha256"] = "d" * 64
                rollback_record.write_text(
                    json.dumps(record, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8",
                )
                state_before = worker.read_state()
                with self.assertRaisesRegex(
                    JointActivationError,
                    "hash mismatch",
                ):
                    controller.recover_pending(rollback_record)
                self.assertEqual(worker.read_state(), state_before)

                record["record_sha256"] = _canonical_sha256(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "record_sha256"
                    }
                )
                rollback_record.write_text(
                    json.dumps(record, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    JointActivationError,
                    "differs",
                ):
                    controller.recover_pending(rollback_record)
                record["nested"] = {
                    "deployment_refresh_token": "must-never-be-consumed"
                }
                record["record_sha256"] = _canonical_sha256(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "record_sha256"
                    }
                )
                rollback_record.write_text(
                    json.dumps(record, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    JointActivationError,
                    "credential field",
                ):
                    controller.recover_pending(rollback_record)

                with self.assertRaisesRegex(
                    JointActivationError,
                    "native live W/X authorization",
                ):
                    controller.activate(record)  # type: ignore[arg-type]
                self.assertNotEqual(store.read_active(), candidate)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                (
                    store,
                    parent,
                    _,
                    _,
                    _,
                    probes,
                    authorization,
                    worker,
                ) = self._production_scenario(root)
                controller = ProductionJointActivationController(
                    store=store,
                    workers=(worker,),
                    probes=probes,
                    project_root=repository_root(),
                )
                with self.assertRaises(InjectedActivationCrash):
                    controller.activate(
                        authorization,
                        fault_after_stage="QUIESCED",
                        fault_mode="crash",
                    )
                rollback_record = next(
                    (store.root / "activation-rollback-only").glob("*.json")
                )
                unrelated_version = _version("policy-unrelated", "harness-unrelated")
                unrelated = store.stage(
                    joint_version=unrelated_version,
                    policy=_artifact(
                        "policy",
                        unrelated_version.policy,
                        "unrelated-policy",
                    ),
                    harness=_artifact(
                        "harness",
                        unrelated_version.harness_controller,
                        "unrelated-harness",
                    ),
                    expected_active_release_id=parent.release_id,
                )
                store.activate(
                    release_id=unrelated.release_id,
                    expected_active_release_id=parent.release_id,
                )
                state_before = worker.read_state()

                with self.assertRaisesRegex(
                    JointActivationError,
                    "unrelated release",
                ):
                    controller.recover_pending(rollback_record)

                self.assertEqual(worker.read_state(), state_before)
                self.assertEqual(store.read_active(), unrelated)


if __name__ == "__main__":
    unittest.main()
