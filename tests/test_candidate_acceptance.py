from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jphrl.joint_release import CandidateArtifact, JointReleaseStore
from jphrl.training.candidate_acceptance import (
    PRODUCTION_HARNESS_ARTIFACT_SCHEMA,
    REQUIRED_SUITE_KINDS,
    CandidateAcceptanceError,
    CandidateAcceptanceSpec,
    CandidateAcceptanceSuite,
    CandidateProbeObservation,
    LiveCandidateAcceptance,
    build_production_candidate_artifacts,
    require_live_candidate_acceptance,
    run_joint_candidate_acceptance,
    validate_candidate_acceptance_report,
)
from jphrl.trajectory.schema import JointVersion


def _version() -> JointVersion:
    return JointVersion(
        policy="policy-parent",
        harness_controller="harness-parent",
        harness_artifact="artifact-v1",
        tool_schema="tool-v1",
        parser="parser-v1",
        environment="environment-v1",
        evaluator="evaluator-v1",
        tokenizer="tokenizer-v1",
        context_builder="context-v1",
    )


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resign(record: dict[str, object]) -> None:
    record["record_sha256"] = _sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


class CandidateAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = _version()
        self.candidate = replace(
            self.parent,
            policy="policy-candidate",
            harness_controller="harness-candidate",
        )
        self.spec = CandidateAcceptanceSpec(
            tuple(
                CandidateAcceptanceSuite(
                    kind=kind,
                    suite_id=f"{kind}-m0-v1",
                    fixture_sha256=hashlib.sha256(kind.encode()).hexdigest(),
                    metric_name="success_fraction",
                    minimum_score=0.75,
                    minimum_sample_count=1 if kind == "joint_safety" else 2,
                )
                for kind in REQUIRED_SUITE_KINDS
            )
        )

    def _scenario(self, root: Path, *, valid_candidate_artifacts: bool = True):
        store = JointReleaseStore(root / "release-store")
        parent_manifest = store.publish(
            joint_version=self.parent,
            policy=CandidateArtifact(
                component="policy",
                version=self.parent.policy,
                payload={"bootstrap": "policy"},
            ),
            harness=CandidateArtifact(
                component="harness",
                version=self.parent.harness_controller,
                payload={"bootstrap": "harness"},
            ),
            expected_active_release_id=None,
        )
        policy_checkpoint_sha256 = "e" * 64
        harness_checkpoint_sha256 = "f" * 64
        harness_parameter_sha256 = "1" * 64
        bundle_record = {
            "receipts": {
                "policy": {
                    "checkpoints": {
                        "parent_path": "/unused/parent",
                        "parent_manifest": {
                            "files": [
                                {"path": "parent", "size_bytes": 1, "sha256": "2" * 64}
                            ],
                            "manifest_sha256": "3" * 64,
                        },
                        "candidate_path": "/unused/candidate",
                        "candidate_manifest": {
                            "files": [
                                {
                                    "path": "candidate",
                                    "size_bytes": 1,
                                    "sha256": "4" * 64,
                                }
                            ],
                            "manifest_sha256": policy_checkpoint_sha256,
                        },
                    },
                    "record_sha256": "5" * 64,
                },
                "policy_sha256": "5" * 64,
                "harness": {
                    "checkpoint_sha256": harness_checkpoint_sha256,
                    "parameter_digest_after": harness_parameter_sha256,
                    "record_sha256": "6" * 64,
                },
                "harness_sha256": "6" * 64,
            },
            "record_sha256": "a" * 64,
        }
        bundle = SimpleNamespace(
            parent_release_id=parent_manifest.release_id,
            parent_joint_version=self.parent,
            candidate_joint_version=self.candidate,
            macro_step_id="macro-1",
            source_joint_credit_sha256="b" * 64,
            policy_engine_version=7,
            candidate_policy_engine_version=8,
            policy_receipt_sha256="5" * 64,
            harness_receipt_sha256="6" * 64,
            record_sha256="a" * 64,
        )
        topology = SimpleNamespace(world_size=1)
        checkpoint = SimpleNamespace(
            bundle_sha256="a" * 64,
            parent_joint_version=self.parent,
            candidate_joint_version=self.candidate,
            macro_step_id="macro-1",
            source_joint_credit_sha256="b" * 64,
            record_sha256="c" * 64,
            topology=topology,
        )
        if valid_candidate_artifacts:
            policy = CandidateArtifact(
                component="policy",
                version=self.candidate.policy,
                payload={
                    "schema_version": "jph.production-policy-release-artifact.v1",
                    "candidate_joint_version_id": self.candidate.version_id,
                    "joint_candidate_bundle_sha256": "a" * 64,
                    "production_checkpoint_manifest_sha256": "c" * 64,
                    "policy_update_receipt_sha256": "5" * 64,
                    "policy_checkpoint_manifest_sha256": policy_checkpoint_sha256,
                    "policy_engine_version": 8,
                },
            )
            harness = CandidateArtifact(
                component="harness",
                version=self.candidate.harness_controller,
                payload={
                    "schema_version": PRODUCTION_HARNESS_ARTIFACT_SCHEMA,
                    "candidate_joint_version_id": self.candidate.version_id,
                    "joint_candidate_bundle_sha256": "a" * 64,
                    "production_checkpoint_manifest_sha256": "c" * 64,
                    "harness_update_receipt_sha256": "6" * 64,
                    "harness_checkpoint_sha256": harness_checkpoint_sha256,
                    "harness_parameter_sha256": harness_parameter_sha256,
                },
            )
        else:
            policy = CandidateArtifact(
                component="policy",
                version=self.candidate.policy,
                payload={"marker": True},
            )
            harness = CandidateArtifact(
                component="harness",
                version=self.candidate.harness_controller,
                payload={"marker": True},
            )
        candidate_manifest = store.stage(
            joint_version=self.candidate,
            policy=policy,
            harness=harness,
            expected_active_release_id=parent_manifest.release_id,
        )
        recovery_record = {"record_sha256": "d" * 64}
        live_recovery = SimpleNamespace(
            record=recovery_record,
            record_sha256="d" * 64,
            checkpoint_manifest_sha256="c" * 64,
            candidate_joint_version=self.candidate,
            macro_step_id="macro-1",
        )
        return SimpleNamespace(
            store=store,
            bundle_record=bundle_record,
            bundle=bundle,
            checkpoint=checkpoint,
            candidate_release_id=candidate_manifest.release_id,
            live_recovery=live_recovery,
            recovery_record=recovery_record,
        )

    def _probes(self, *, value: float = 1.0):
        probes = {
            kind: (
                lambda version, suite, suite_kind=kind: [
                    CandidateProbeObservation(
                        sample_id=f"{suite_kind}-0",
                        metric_value=value,
                        output={"answer": "first", "version": version.version_id},
                    ),
                    CandidateProbeObservation(
                        sample_id=f"{suite_kind}-1",
                        metric_value=value,
                        output={"answer": "second", "fixture": suite.fixture_sha256},
                    ),
                ]
            )
            for kind in REQUIRED_SUITE_KINDS
            if kind != "joint_safety"
        }
        probes["joint_safety"] = lambda version, suite: [
            CandidateProbeObservation(
                sample_id="joint_safety-0",
                metric_value=value,
                output={"answer": "joint", "version": version.version_id},
                production_probe_output=(
                    b'{"schema_version":"test-production-probe.v1"}'
                ),
            )
        ]
        return probes

    def _patches(self, scenario):
        return (
            patch(
                "jphrl.training.candidate_acceptance.validate_joint_candidate_bundle",
                return_value=scenario.bundle,
            ),
            patch(
                "jphrl.training.candidate_acceptance.validate_production_joint_checkpoint",
                return_value=scenario.checkpoint,
            ),
            patch(
                "jphrl.training.candidate_acceptance.validate_exact_joint_recovery_evidence",
                return_value={
                    "integrity_valid": True,
                    "exact_joint_recovery": False,
                    "record_sha256": "d" * 64,
                },
            ),
            patch(
                "jphrl.training.production_checkpoint.require_live_exact_joint_recovery",
                return_value=scenario.live_recovery,
            ),
        )

    def _run(self, root: Path, scenario, **overrides):
        arguments = {
            "joint_candidate_bundle": scenario.bundle_record,
            "checkpoint_manifest": root / "manifest.json",
            "live_exact_recovery": scenario.live_recovery,
            "candidate_release_id": scenario.candidate_release_id,
            "expected_spec": self.spec,
            "probes": self._probes(),
            "release_store": scenario.store,
            "report_root": root / "runs" / "acceptance",
            "project_root": root / "src" / "repo",
            "require_component_files": False,
        }
        arguments.update(overrides)
        return run_joint_candidate_acceptance(**arguments)

    def test_live_acceptance_persists_integrity_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                patches = self._patches(scenario)
                with patches[0], patches[1], patches[2], patches[3]:
                    live = self._run(root, scenario)
                    audit = validate_candidate_acceptance_report(
                        live.report,
                        expected_spec=self.spec,
                        release_store=scenario.store,
                        joint_candidate_bundle=scenario.bundle_record,
                        checkpoint_manifest=root / "manifest.json",
                        exact_recovery_evidence=scenario.recovery_record,
                        require_component_files=False,
                    )
                    consumed = require_live_candidate_acceptance(
                        live,
                        expected_spec=self.spec,
                        release_store=scenario.store,
                        joint_candidate_bundle=scenario.bundle_record,
                        checkpoint_manifest=root / "manifest.json",
                        live_exact_recovery=scenario.live_recovery,
                        require_component_files=False,
                    )

            self.assertIs(type(live), LiveCandidateAcceptance)
            self.assertFalse(audit.live_acceptance)
            self.assertTrue(consumed.live_acceptance)
            self.assertEqual(audit.record_sha256, live.record_sha256)
            self.assertEqual(audit.spec_sha256, self.spec.digest)
            path = root / "runs" / "acceptance" / f"acceptance-{'a' * 64}.json"
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), live.report)
            self.assertTrue(live.report["decision"]["accepted"])
            self.assertFalse(
                live.report["evidence_scope"][
                    "persisted_report_regrants_live_acceptance"
                ]
            )

    def test_probe_callback_cannot_report_pass_score_or_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                forged = self._probes()
                forged["joint_safety"] = lambda version, suite: [
                    {
                        "sample_id": "fake",
                        "passed": True,
                        "score": 99,
                        "output_sha256": "0" * 64,
                    }
                ]
                patches = self._patches(scenario)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    self.assertRaisesRegex(CandidateAcceptanceError, "raw observation"),
                ):
                    self._run(root, scenario, probes=forged)

    def test_joint_safety_requires_one_framework_hashed_raw_production_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                missing = self._probes()
                missing["joint_safety"] = lambda version, suite: [
                    CandidateProbeObservation(
                        sample_id="missing",
                        metric_value=1.0,
                        output={"version": version.version_id},
                    )
                ]
                patches = self._patches(scenario)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    self.assertRaisesRegex(
                        CandidateAcceptanceError,
                        "exactly one raw production probe output",
                    ),
                ):
                    self._run(root, scenario, probes=missing)

            second = root / "second"
            second.mkdir()
            with patch.dict("os.environ", {"JPH_ROOT": str(second)}):
                scenario = self._scenario(second)
                crossed = self._probes()
                crossed["policy_heldout"] = lambda version, suite: [
                    CandidateProbeObservation(
                        sample_id="crossed",
                        metric_value=1.0,
                        output={"version": version.version_id},
                        production_probe_output=b"not-joint-safety",
                    )
                ]
                patches = self._patches(scenario)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    self.assertRaisesRegex(
                        CandidateAcceptanceError,
                        "only joint_safety",
                    ),
                ):
                    self._run(second, scenario, probes=crossed)

    def test_framework_computes_threshold_and_rejects_failed_or_unstable_probe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                patches = self._patches(scenario)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    self.assertRaisesRegex(CandidateAcceptanceError, "did not pass"),
                ):
                    self._run(root, scenario, probes=self._probes(value=0.5))

            root = root / "second"
            root.mkdir()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                calls = 0
                unstable = self._probes()

                def nondeterministic(version, suite):
                    nonlocal calls
                    calls += 1
                    return [
                        CandidateProbeObservation(
                            sample_id="one",
                            metric_value=1.0,
                            output={"call": calls},
                        ),
                        CandidateProbeObservation(
                            sample_id="two",
                            metric_value=1.0,
                            output={"version": version.version_id},
                        ),
                    ]

                unstable["policy_heldout"] = nondeterministic
                patches = self._patches(scenario)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    self.assertRaisesRegex(
                        CandidateAcceptanceError, "not deterministic"
                    ),
                ):
                    self._run(root, scenario, probes=unstable)

    def test_store_lineage_and_v_w_t_u_artifacts_are_directly_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root, valid_candidate_artifacts=False)
                patches = self._patches(scenario)
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    self.assertRaisesRegex(
                        CandidateAcceptanceError, "differ from V/W/T/U"
                    ),
                ):
                    self._run(root, scenario)

    def test_external_spec_is_required_again_when_report_is_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                patches = self._patches(scenario)
                with patches[0], patches[1], patches[2], patches[3]:
                    live = self._run(root, scenario)
                    changed = CandidateAcceptanceSpec(
                        tuple(
                            replace(suite, minimum_score=0.99)
                            if suite.kind == "joint_safety"
                            else suite
                            for suite in self.spec.suites
                        )
                    )
                    with self.assertRaisesRegex(
                        CandidateAcceptanceError, "external frozen spec"
                    ):
                        validate_candidate_acceptance_report(
                            live.report,
                            expected_spec=changed,
                            release_store=scenario.store,
                            joint_candidate_bundle=scenario.bundle_record,
                            checkpoint_manifest=root / "manifest.json",
                            exact_recovery_evidence=scenario.recovery_record,
                            require_component_files=False,
                        )

    def test_persisted_or_user_constructed_value_cannot_mint_live_acceptance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                patches = self._patches(scenario)
                with patches[0], patches[1], patches[2], patches[3]:
                    live = self._run(root, scenario)
                    with self.assertRaisesRegex(
                        CandidateAcceptanceError, "live candidate acceptance"
                    ):
                        require_live_candidate_acceptance(
                            deepcopy(live.report),
                            expected_spec=self.spec,
                            release_store=scenario.store,
                            joint_candidate_bundle=scenario.bundle_record,
                            checkpoint_manifest=root / "manifest.json",
                            live_exact_recovery=scenario.live_recovery,
                            require_component_files=False,
                        )

                    with self.assertRaises(TypeError):
                        LiveCandidateAcceptance()
                    with self.assertRaisesRegex(
                        CandidateAcceptanceError, "token is invalid"
                    ):
                        LiveCandidateAcceptance._create(
                            report=live.report,
                            audit=live._audit,
                            token=object(),
                        )

    def test_persisted_w_record_cannot_start_candidate_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                patches = self._patches(scenario)
                # Do not patch W's private-token checker for this negative case.
                with (
                    patches[0],
                    patches[1],
                    self.assertRaisesRegex(
                        CandidateAcceptanceError, "live exact joint recovery"
                    ),
                ):
                    self._run(
                        root,
                        scenario,
                        live_exact_recovery=scenario.recovery_record,
                    )

    def test_rehashed_framework_score_secret_and_skipped_gate_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                patches = self._patches(scenario)
                with patches[0], patches[1], patches[2], patches[3]:
                    live = self._run(root, scenario)
                    cases = []
                    score = deepcopy(live.report)
                    score["critical_suites"][0]["probe"]["score"] = 99.0
                    _resign(score)
                    cases.append((score, "framework-computed score"))
                    skipped = deepcopy(live.report)
                    skipped["decision"]["skipped_critical_gate_count"] = 1
                    _resign(skipped)
                    cases.append((skipped, "failed or skipped"))
                    secret = deepcopy(live.report)
                    secret["identity"]["session_api_key"] = "must-not-persist"
                    _resign(secret)
                    cases.append((secret, "credential field"))
                    for forged, message in cases:
                        with (
                            self.subTest(message=message),
                            self.assertRaisesRegex(CandidateAcceptanceError, message),
                        ):
                            validate_candidate_acceptance_report(
                                forged,
                                expected_spec=self.spec,
                                release_store=scenario.store,
                                joint_candidate_bundle=scenario.bundle_record,
                                checkpoint_manifest=root / "manifest.json",
                                exact_recovery_evidence=scenario.recovery_record,
                                require_component_files=False,
                            )

    def test_artifact_builder_emits_exact_production_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                scenario = self._scenario(root)
                patches = self._patches(scenario)
                with patches[0], patches[1]:
                    policy, harness = build_production_candidate_artifacts(
                        joint_candidate_bundle=scenario.bundle_record,
                        checkpoint_manifest=root / "manifest.json",
                        require_component_files=False,
                    )
            self.assertEqual(
                policy.payload["schema_version"],
                "jph.production-policy-release-artifact.v1",
            )
            self.assertEqual(policy.payload["policy_engine_version"], 8)
            self.assertEqual(policy.payload["policy_update_receipt_sha256"], "5" * 64)
            self.assertEqual(
                harness.payload["schema_version"], PRODUCTION_HARNESS_ARTIFACT_SCHEMA
            )
            self.assertEqual(harness.payload["harness_update_receipt_sha256"], "6" * 64)


if __name__ == "__main__":
    unittest.main()
