from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_areal_joint_bridge import (
    SENSITIVE_KEY_PARTS,
    _audit_sensitive_artifacts,
    _score_sha256,
    _trajectory_binding_sha256,
    _verify_same_backend_scores,
)


class SameBackendScoreAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, rescored: list[float]):
        bridge_path = root / "bridge-records" / "bridge-one.json"
        bridge_path.parent.mkdir(parents=True)
        input_ids = [[10, 11, 12, 13]]
        loss_mask = [[0, 0, 1, 1]]
        stored = [[0.0, 0.0, -1.0, -2.0]]
        versions = [[0, 0, 0, 0]]
        request_id = "request-one"
        bridge_record_sha256 = "a" * 64
        record = {
            "request_id": request_id,
            "record_sha256": bridge_record_sha256,
            "origin": {"project_commit": "c" * 40},
            "joint_version": {"policy": "policy-release-v0"},
            "policy_binding": {"expected_inference_engine_version": 0},
            "areal_trace": {
                "origin": {
                    "behavior_revision": "d" * 40,
                    "areal_commit": "e" * 40,
                },
                "tensor_dict": {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                    "logprobs": stored,
                    "versions": versions,
                    "attention_mask": [[1, 1, 1, 1]],
                    "rewards": [1.0],
                }
            }
        }
        trajectory_binding_sha256 = _trajectory_binding_sha256(
            record["areal_trace"]["tensor_dict"]
        )
        score = {
            "schema_version": "jph.areal-same-backend-logprob.v2",
            "request_id": request_id,
            "bridge_record_sha256": bridge_record_sha256,
            "trajectory_binding_sha256": trajectory_binding_sha256,
            "scoring_origin": {
                "api": "RolloutController.compute_logp",
                "lifecycle": "same-controller-after-wait-before-destroy",
                "backend": "sglang:d1p1t1",
                "engine_version_before_score": 0,
                "engine_version_after_score": 0,
                "policy_release_id": "policy-release-v0",
                "behavior_revision": "d" * 40,
                "areal_commit": "e" * 40,
                "project_commit": "c" * 40,
            },
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "stored_logprobs": stored,
            "rescored_logprobs": [[0.0, 0.0, *rescored]],
            "versions": versions,
        }
        score["record_sha256"] = _score_sha256(score)
        score_dir = root / "same-backend-scores"
        score_dir.mkdir()
        (score_dir / "same-backend-score-one.json").write_text(
            json.dumps(score), encoding="utf-8"
        )
        return score_dir, [(bridge_path, record)]

    def test_accepts_pre_registered_same_backend_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            score_dir, records = self._fixture(
                Path(directory),
                [-0.995, -2.005],
            )
            report = _verify_same_backend_scores(score_dir, records)
            self.assertEqual(len(report), 1)
            self.assertTrue(report[0]["passed"])
            self.assertLess(report[0]["max_importance_ratio_error"], 0.10)

    def test_rejects_same_backend_tail_above_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            score_dir, records = self._fixture(
                Path(directory),
                [-0.8, -2.0],
            )
            with self.assertRaisesRegex(ValueError, "same-backend policy logprob"):
                _verify_same_backend_scores(score_dir, records)

    def test_rejects_unconsumed_score_when_inputs_are_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            score_dir, records = self._fixture(Path(directory), [-1.0, -2.0])
            duplicate = json.loads(json.dumps(records[0][1]))
            duplicate["request_id"] = "request-two"
            duplicate["record_sha256"] = "f" * 64
            records.append((Path(directory) / "bridge-two.json", duplicate))

            first_score_path = next(score_dir.glob("*.json"))
            unrelated = json.loads(first_score_path.read_text(encoding="utf-8"))
            unrelated["request_id"] = "unrelated-request"
            unrelated["record_sha256"] = "9" * 64
            unrelated["record_sha256"] = _score_sha256(unrelated)
            (score_dir / "same-backend-score-unrelated.json").write_text(
                json.dumps(unrelated), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "no matching same-backend score"):
                _verify_same_backend_scores(score_dir, records)


class SensitiveArtifactAuditTests(unittest.TestCase):
    def test_accepts_redacted_yaml_and_compact_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text(
                "admin_api_key: <redacted-runtime-admin-key>\n",
                encoding="utf-8",
            )
            (root / "config.json").write_text(
                '{"agent":{"admin_api_key":"<redacted-runtime-admin-key>"}}\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"JPH_AREAL_ADMIN_API_KEY": "jph-bridge-ephemeral"},
            ):
                report = _audit_sensitive_artifacts(root)
            self.assertEqual(report["unsafe_fields"], [])
            self.assertEqual(report["redacted_or_safe_sensitive_fields"], 2)

    def test_rejects_runtime_secret_anywhere_in_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = "jph-bridge-ephemeral"
            (root / "run.log").write_text(f"leak={secret}\n", encoding="utf-8")
            with patch.dict(os.environ, {"JPH_AREAL_ADMIN_API_KEY": secret}):
                with self.assertRaisesRegex(ValueError, "runtime admin key"):
                    _audit_sensitive_artifacts(root)

    def test_rejects_every_sensitive_field_in_extensionless_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "resolved-config"
            for key in SENSITIVE_KEY_PARTS:
                with self.subTest(key=key):
                    artifact.write_text(
                        f"{key}: should-not-be-here\n",
                        encoding="utf-8",
                    )
                    with patch.dict(
                        os.environ,
                        {"JPH_AREAL_ADMIN_API_KEY": "jph-bridge-ephemeral"},
                    ):
                        with self.assertRaisesRegex(ValueError, "sensitive fields"):
                            _audit_sensitive_artifacts(root)


if __name__ == "__main__":
    unittest.main()
