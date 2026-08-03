from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace

from jphrl.training.areal_production_worker import (
    ArealProductionWorkerError,
    HarnessServingCheckpoint,
    LiveArealServingExportPair,
    PinnedArealSGLangActivationWorker,
    build_production_probe_output,
    materialize_areal_serving_export_pair,
    require_live_areal_serving_export_pair,
)
from jphrl.trajectory.schema import JointVersion


def _version() -> JointVersion:
    return JointVersion(
        policy="policy-parent",
        harness_controller="harness-parent",
        harness_artifact="harness-artifact-v1",
        tool_schema="tools-v1",
        parser="parser-v1",
        environment="environment-v1",
        evaluator="evaluator-v1",
        tokenizer="tokenizer-v1",
        context_builder="context-v1",
    )


class ArealProductionWorkerTests(unittest.TestCase):
    def test_probe_output_keeps_dcp_and_live_serving_identity_distinct(self) -> None:
        parent = _version()
        candidate = replace(
            parent,
            policy="policy-candidate",
            harness_controller="harness-candidate",
        )
        arguments = {
            "fixture": b'{"prompt":"held-out"}',
            "target_release_id": "release-candidate",
            "target_joint_version": candidate,
            "policy_engine_version": 8,
            "policy_checkpoint_sha256": "1" * 64,
            "serving_parameter_sha256": "2" * 64,
            "harness_checkpoint_sha256": "3" * 64,
            "harness_parameter_sha256": "4" * 64,
        }
        first = build_production_probe_output(**arguments)
        second = build_production_probe_output(**arguments)
        record = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(record["policy_checkpoint_sha256"], "1" * 64)
        self.assertEqual(record["serving_parameter_sha256"], "2" * 64)
        self.assertNotEqual(
            record["policy_checkpoint_sha256"],
            record["serving_parameter_sha256"],
        )
        self.assertEqual(
            record["fixture_sha256"], hashlib.sha256(arguments["fixture"]).hexdigest()
        )

    def test_probe_output_rejects_empty_fixture_and_bad_digest(self) -> None:
        with self.assertRaisesRegex(ArealProductionWorkerError, "fixture"):
            build_production_probe_output(
                fixture=b"",
                target_release_id="release",
                target_joint_version=_version(),
                policy_engine_version=1,
                policy_checkpoint_sha256="1" * 64,
                serving_parameter_sha256="2" * 64,
                harness_checkpoint_sha256="3" * 64,
                harness_parameter_sha256="4" * 64,
            )
        with self.assertRaisesRegex(ArealProductionWorkerError, "digest"):
            build_production_probe_output(
                fixture=b"fixture",
                target_release_id="release",
                target_joint_version=_version(),
                policy_engine_version=1,
                policy_checkpoint_sha256="not-a-digest",
                serving_parameter_sha256="2" * 64,
                harness_checkpoint_sha256="3" * 64,
                harness_parameter_sha256="4" * 64,
            )

    def test_persisted_or_user_constructed_lineage_cannot_mint_live_pair(self) -> None:
        forged = LiveArealServingExportPair()
        with self.assertRaisesRegex(ArealProductionWorkerError, "native live"):
            require_live_areal_serving_export_pair(forged)
        with self.assertRaisesRegex(ArealProductionWorkerError, "real pinned"):
            materialize_areal_serving_export_pair(
                actor=object(),
                policy_candidate_record={},
                export_root="/tmp/unused-serving-export",
                parent_joint_version=_version(),
                candidate_joint_version=replace(
                    _version(), policy="candidate", harness_controller="candidate"
                ),
            )

    def test_worker_direct_constructor_is_blocked_for_cleanup_ownership(self) -> None:
        with self.assertRaisesRegex(ArealProductionWorkerError, r"\.create\(\)"):
            PinnedArealSGLangActivationWorker(
                controller=object(),
                serving_exports=LiveArealServingExportPair(),
                harness_checkpoints={},
                observation_root="/tmp/unused-worker-observations",
                parent_release_id="parent",
                candidate_release_id="candidate",
            )

    def test_harness_checkpoint_spec_is_exact(self) -> None:
        HarnessServingCheckpoint(
            path="/external/harness.json",
            checkpoint_sha256="a" * 64,
            kind="rollout_json",
        ).validate()
        with self.assertRaisesRegex(ArealProductionWorkerError, "kind"):
            HarnessServingCheckpoint(
                path="/external/harness.bin",
                checkpoint_sha256="a" * 64,
                kind="unknown",
            ).validate()


if __name__ == "__main__":
    unittest.main()
