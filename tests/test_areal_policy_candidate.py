from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jphrl.paths import repository_root
from jphrl.training.areal_policy_candidate import (
    AREAL_POLICY_CANDIDATE_SCHEMA,
    PINNED_AREAL_COMMIT,
    ArealPolicyCandidateError,
    _prepare_run_root,
    checkpoint_manifest,
    run_areal_policy_candidate_update,
    validate_areal_policy_candidate,
)
from jphrl.trajectory.schema import JointVersion


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _manifest(file_digest: str) -> dict[str, object]:
    manifest: dict[str, object] = {
        "files": [
            {
                "path": ".metadata",
                "size_bytes": 19,
                "sha256": file_digest,
            }
        ]
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    return manifest


def _record() -> dict[str, object]:
    parent_manifest = _manifest("1" * 64)
    candidate_manifest = _manifest("2" * 64)
    record: dict[str, object] = {
        "schema_version": AREAL_POLICY_CANDIDATE_SCHEMA,
        "transaction": {
            "transaction_id": "txn-1",
            "episode_id": "episode-1",
            "source_admission_sha256": "a" * 64,
            "source_joint_credit_sha256": "b" * 64,
            "trainable_token_count": 3,
        },
        "parent": {
            "joint_version": _version().__dict__,
            "joint_version_id": _version().version_id,
            "policy_engine_version": 7,
            "policy_version": _version().policy,
        },
        "candidate": {
            "policy_version": (
                "areal-policy-" + candidate_manifest["manifest_sha256"][:20]
            ),
            "reserved_policy_engine_version": 8,
        },
        "optimizer": {
            "actor_type": "areal.engine.fsdp_engine.FSDPPPOActor",
            "optimizer_type": "torch.optim.adamw.AdamW",
            "config": {
                "adv_norm": None,
                "backend": "fsdp:d1",
                "c_clip": None,
                "disable_dropout": True,
                "eps_clip": 0.2,
                "eps_clip_higher": None,
                "importance_sampling_level": "token",
                "kl_ctl": 0.0,
                "lr": 1e-5,
                "lr_scheduler_type": "constant",
                "ppo_n_minibatches": 1,
                "recompute_logprob": False,
                "reward_norm": None,
                "temperature": 1.0,
                "use_decoupled_loss": False,
            },
            "stats": {
                "grad_norm": 0.5,
                "grad_norm_count": 1,
                "learning_rate": 1e-5,
                "learning_rate_count": 1,
                "optimizer_step_before": 0,
                "optimizer_step_after": 1,
                "update_successful": 1.0,
                "update_successful_count": 1,
            },
            "lr_scheduler": {
                "state_before_sha256": "6" * 64,
                "state_after_sha256": "7" * 64,
                "step_completed": True,
            },
        },
        "checkpoints": {
            "parent_path": "/tmp/run/policy-parent.dcp",
            "parent_manifest": parent_manifest,
            "candidate_path": "/tmp/run/policy-candidate.dcp",
            "candidate_manifest": candidate_manifest,
        },
        "probes": {
            "pre_sha256": "3" * 64,
            "post_sha256": "4" * 64,
            "max_abs_logprob_delta": 1e-3,
            "parent_reload_sha256": "3" * 64,
            "candidate_reload_sha256": "4" * 64,
        },
        "provenance": {
            "areal_commit": PINNED_AREAL_COMMIT,
            "project_commit": "5" * 40,
        },
        "evidence_scope": {
            "policy_optimizer_update": True,
            "policy_candidate_created": True,
            "policy_weights_published": False,
            "rollout_weights_synchronized": False,
            "active_joint_version_changed": False,
            "harness_optimizer_update": False,
            "joint_publish": False,
        },
    }
    _resign(record)
    return record


def _resign(record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = _sha256(unsigned)


class ArealPolicyCandidateTests(unittest.TestCase):
    def test_candidate_root_uses_actual_checkout_not_caller_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside Git checkout"):
            _prepare_run_root(
                repository_root() / "must-not-be-created" / "policy-candidate",
                "/tmp/caller-supplied-fake-project",
            )

    def test_valid_evidence_replays_candidate_identity_and_parent_version(self) -> None:
        audit = validate_areal_policy_candidate(
            json.loads(json.dumps(_record())),
            active_joint_version=_version(),
        )

        self.assertEqual(audit.parent_engine_version, 7)
        self.assertEqual(audit.reserved_candidate_engine_version, 8)
        self.assertTrue(audit.candidate_policy_version.startswith("areal-policy-"))

    def test_stale_parent_false_stats_and_credential_are_fail_closed(self) -> None:
        stale = replace(_version(), policy="another-parent")
        with self.assertRaisesRegex(ArealPolicyCandidateError, "lag-zero"):
            validate_areal_policy_candidate(
                _record(),
                active_joint_version=stale,
            )

        failed_update = _record()
        failed_update["optimizer"]["stats"]["update_successful"] = 0.0
        _resign(failed_update)
        with self.assertRaisesRegex(ArealPolicyCandidateError, "successful update"):
            validate_areal_policy_candidate(failed_update)

        credential = _record()
        credential["optimizer"]["config"]["admin_api_key"] = "secret"
        _resign(credential)
        with self.assertRaisesRegex(ArealPolicyCandidateError, "field set"):
            validate_areal_policy_candidate(credential)

        spoofed_actor = _record()
        spoofed_actor["optimizer"]["actor_type"] = "areal.fake.FSDPPPOActor"
        _resign(spoofed_actor)
        with self.assertRaisesRegex(ArealPolicyCandidateError, "identity"):
            validate_areal_policy_candidate(spoofed_actor)

        scheduler_claim = _record()
        scheduler_claim["optimizer"]["lr_scheduler"]["state_after_sha256"] = (
            scheduler_claim["optimizer"]["lr_scheduler"]["state_before_sha256"]
        )
        _resign(scheduler_claim)
        with self.assertRaisesRegex(ArealPolicyCandidateError, "scheduler evidence"):
            validate_areal_policy_candidate(scheduler_claim)

    def test_failure_restores_scheduler_state_not_covered_by_areal_dcp(self) -> None:
        class FakeScheduler:
            def __init__(self) -> None:
                self.state = {"last_epoch": 0, "_step_count": 1, "lr_lambdas": [None]}

            def state_dict(self):
                return dict(self.state)

            def load_state_dict(self, state):
                self.state = dict(state)

            def step(self) -> None:
                self.state["last_epoch"] += 1
                self.state["_step_count"] += 1

        scheduler = FakeScheduler()
        actor = SimpleNamespace(lr_scheduler=scheduler)

        def fail_after_scheduler_step(*args, **kwargs):
            del args, kwargs
            scheduler.step()
            raise RuntimeError("post-update candidate probe failed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with (
                patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                patch(
                    "jphrl.training.areal_policy_candidate."
                    "_run_areal_policy_candidate_update_unprotected",
                    side_effect=fail_after_scheduler_step,
                ),
                self.assertRaisesRegex(RuntimeError, "candidate probe failed"),
            ):
                run_areal_policy_candidate_update(
                    {},
                    source_joint_credit_record={},
                    actor=actor,
                    active_joint_version=_version(),
                    candidate_root=root / "candidate",
                    project_root=repository_root(),
                    transaction_id="txn-rollback",
                    project_commit="5" * 40,
                    areal_commit=PINNED_AREAL_COMMIT,
                    device="cpu",
                )

        self.assertEqual(
            scheduler.state,
            {"last_epoch": 0, "_step_count": 1, "lr_lambdas": [None]},
        )

    def test_checkpoint_manifest_hashes_files_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "rank0.distcp").write_bytes(b"real optimizer state")
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                manifest = checkpoint_manifest(checkpoint)
                self.assertEqual(manifest["files"][0]["size_bytes"], 20)

                (checkpoint / "unsafe-link").symlink_to(checkpoint / "rank0.distcp")
                with self.assertRaisesRegex(ArealPolicyCandidateError, "symbolic link"):
                    checkpoint_manifest(checkpoint)


if __name__ == "__main__":
    unittest.main()
