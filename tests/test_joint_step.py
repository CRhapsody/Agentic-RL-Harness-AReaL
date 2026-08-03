from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jphrl.paths import repository_root
from jphrl.training.joint_step import (
    JointStepError,
    JointStepRollbackError,
    build_joint_candidate_bundle,
    seal_joint_candidate_bundle,
    validate_joint_candidate_bundle,
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
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = _sha256(unsigned)


def _audits(*, source: str = "a" * 64):
    policy = SimpleNamespace(
        transaction_id="macro-1",
        source_joint_credit_sha256=source,
        trainable_token_count=3,
        parent_engine_version=7,
        reserved_candidate_engine_version=8,
        candidate_policy_version="areal-policy-candidate",
        record_sha256="b" * 64,
    )
    harness = SimpleNamespace(
        source_joint_credit_sha256=source,
        effective_batch_size=2,
        candidate_version="torch-harness-candidate",
    )
    return policy, harness


def _receipts() -> tuple[dict[str, object], dict[str, object]]:
    return {"record_sha256": "b" * 64}, {"record_sha256": "c" * 64}


class JointStepTests(unittest.TestCase):
    def _patch_audits(self, *, policy=None, harness=None):
        default_policy, default_harness = _audits()
        return (
            patch(
                "jphrl.training.joint_step.validate_areal_policy_candidate",
                return_value=policy or default_policy,
            ),
            patch(
                "jphrl.training.joint_step._validate_harness_receipt",
                return_value=harness or default_harness,
            ),
        )

    def test_contract_bundle_changes_only_policy_and_harness_versions(self) -> None:
        policy_record, harness_record = _receipts()
        policy_patch, harness_patch = self._patch_audits()
        with policy_patch, harness_patch:
            bundle = build_joint_candidate_bundle(
                policy_receipt=policy_record,
                harness_receipt=harness_record,
                active_joint_version=_version(),
                parent_release_id="release-parent",
                macro_step_id="macro-1",
                actor_public_version=7,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )
            audit = validate_joint_candidate_bundle(
                json.loads(json.dumps(bundle)),
                active_joint_version=_version(),
                actor_public_version=7,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )

        self.assertEqual(audit.candidate_joint_version.policy, "areal-policy-candidate")
        self.assertEqual(
            audit.candidate_joint_version.harness_controller,
            "torch-harness-candidate",
        )
        self.assertEqual(
            audit.candidate_joint_version.parser,
            audit.parent_joint_version.parser,
        )
        self.assertFalse(bundle["evidence_scope"]["joint_publish"])

    def test_caller_cannot_lie_about_git_root_for_barrier_journal(self) -> None:
        policy_record, harness_record = _receipts()
        with self.assertRaisesRegex(ValueError, "outside Git checkout"):
            seal_joint_candidate_bundle(
                seal_root=repository_root() / "must-not-be-created" / "macro-1",
                project_root="/tmp/caller-supplied-fake-project",
                policy_receipt=policy_record,
                harness_receipt=harness_record,
                active_joint_version=_version(),
                parent_release_id="release-parent",
                macro_step_id="macro-1",
                actor_public_version=7,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )

    def test_source_mismatch_and_early_activation_restore_both_parents(self) -> None:
        policy_record, harness_record = _receipts()
        policy_audit, _ = _audits(source="a" * 64)
        _, harness_audit = _audits(source="d" * 64)
        callbacks: list[str] = []
        policy_patch, harness_patch = self._patch_audits(
            policy=policy_audit,
            harness=harness_audit,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with (
                patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                policy_patch,
                harness_patch,
                self.assertRaisesRegex(JointStepError, "different S records"),
            ):
                seal_joint_candidate_bundle(
                    seal_root=root / "runs" / "macro-1",
                    project_root=root / "src" / "repo",
                    policy_receipt=policy_record,
                    harness_receipt=harness_record,
                    active_joint_version=_version(),
                    parent_release_id="release-parent",
                    macro_step_id="macro-1",
                    actor_public_version=7,
                    harness_public_version="harness-parent",
                    restore_policy_parent=lambda: callbacks.append("policy"),
                    restore_harness_parent=lambda: callbacks.append("harness"),
                    require_receipt_files=False,
                )
        self.assertEqual(callbacks, ["harness", "policy"])

        policy_patch, harness_patch = self._patch_audits()
        with (
            policy_patch,
            harness_patch,
            self.assertRaisesRegex(JointStepError, "published before"),
        ):
            build_joint_candidate_bundle(
                policy_receipt=policy_record,
                harness_receipt=harness_record,
                active_joint_version=_version(),
                parent_release_id="release-parent",
                macro_step_id="macro-1",
                actor_public_version=8,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )

    def test_exclusive_macro_seal_rejects_duplicate_and_rolls_back(self) -> None:
        policy_record, harness_record = _receipts()
        callbacks: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            project = root / "src" / "repo"
            policy_patch, harness_patch = self._patch_audits()
            with (
                patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                policy_patch,
                harness_patch,
            ):
                seal_joint_candidate_bundle(
                    seal_root=root / "runs" / "macro-1",
                    project_root=project,
                    policy_receipt=policy_record,
                    harness_receipt=harness_record,
                    active_joint_version=_version(),
                    parent_release_id="release-parent",
                    macro_step_id="macro-1",
                    actor_public_version=7,
                    harness_public_version="harness-parent",
                    require_receipt_files=False,
                )
            policy_patch, harness_patch = self._patch_audits()
            with (
                patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                policy_patch,
                harness_patch,
                self.assertRaises(FileExistsError),
            ):
                seal_joint_candidate_bundle(
                    seal_root=root / "runs" / "macro-1",
                    project_root=project,
                    policy_receipt=policy_record,
                    harness_receipt=harness_record,
                    active_joint_version=_version(),
                    parent_release_id="release-parent",
                    macro_step_id="macro-1",
                    actor_public_version=7,
                    harness_public_version="harness-parent",
                    restore_policy_parent=lambda: callbacks.append("policy"),
                    restore_harness_parent=lambda: callbacks.append("harness"),
                    require_receipt_files=False,
                )
        self.assertEqual(callbacks, ["harness", "policy"])

    def test_rollback_attempts_both_callbacks_and_reports_incomplete_restore(
        self,
    ) -> None:
        policy_record, harness_record = _receipts()
        callbacks: list[str] = []

        def fail_harness() -> None:
            callbacks.append("harness")
            raise RuntimeError("injected Harness restore failure")

        policy_audit, _ = _audits(source="a" * 64)
        _, harness_audit = _audits(source="d" * 64)
        policy_patch, harness_patch = self._patch_audits(
            policy=policy_audit,
            harness=harness_audit,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict("os.environ", {"JPH_ROOT": str(Path(temporary).resolve())}),
            policy_patch,
            harness_patch,
            self.assertRaisesRegex(JointStepRollbackError, "rollback was incomplete"),
        ):
            root = Path(temporary).resolve()
            seal_joint_candidate_bundle(
                seal_root=root / "runs" / "macro-1",
                project_root=root / "src" / "repo",
                policy_receipt=policy_record,
                harness_receipt=harness_record,
                active_joint_version=_version(),
                parent_release_id="release-parent",
                macro_step_id="macro-1",
                actor_public_version=7,
                harness_public_version="harness-parent",
                restore_policy_parent=lambda: callbacks.append("policy"),
                restore_harness_parent=fail_harness,
                require_receipt_files=False,
            )
        self.assertEqual(callbacks, ["harness", "policy"])

    def test_rehashed_post_barrier_publish_claim_is_rejected(self) -> None:
        policy_record, harness_record = _receipts()
        policy_patch, harness_patch = self._patch_audits()
        with policy_patch, harness_patch:
            bundle = build_joint_candidate_bundle(
                policy_receipt=policy_record,
                harness_receipt=harness_record,
                active_joint_version=_version(),
                parent_release_id="release-parent",
                macro_step_id="macro-1",
                actor_public_version=7,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )
            forged = deepcopy(bundle)
            forged["evidence_scope"]["joint_publish"] = True
            _resign(forged)
            with self.assertRaisesRegex(JointStepError, "evidence scope"):
                validate_joint_candidate_bundle(
                    forged,
                    require_receipt_files=False,
                )


if __name__ == "__main__":
    unittest.main()
