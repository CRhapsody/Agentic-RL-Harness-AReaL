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


def _audits(
    *,
    source: str = "a" * 64,
    policy_transaction: str = "macro-1",
    harness_transaction: str = "macro-1",
    source_binding: dict[str, object] | None = None,
):
    if source_binding is not None:
        source = str(source_binding["record_sha256"])
    policy = SimpleNamespace(
        transaction_id=policy_transaction,
        source_joint_credit_sha256=source,
        trainable_token_count=3,
        parent_engine_version=7,
        reserved_candidate_engine_version=8,
        candidate_policy_version="areal-policy-candidate",
        record_sha256="b" * 64,
        source_binding=source_binding,
    )
    harness = SimpleNamespace(
        transaction_id=harness_transaction,
        source_joint_credit_sha256=source,
        effective_batch_size=2,
        candidate_version="torch-harness-candidate",
        source_binding=source_binding,
    )
    return policy, harness


def _source_binding(*, first_s: str = "1" * 64) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "jph.multi-s-source-binding.v1",
        "joint_version_id": _version().version_id,
        "batch_record_sha256": "d" * 64,
        "batch_aggregate_sha256": "e" * 64,
        "member_claim_sha256s": [character * 64 for character in "4567"],
        "s_record_sha256s": [first_s, "2" * 64, "3" * 64, "4" * 64],
        "policy_sample_count": 4,
        "harness_action_count": 4,
    }
    record["record_sha256"] = _sha256(record)
    return record


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
        self.assertEqual(bundle["macro_step"]["transaction_id"], "macro-1")
        self.assertIsNone(bundle["source_binding"])
        self.assertEqual(bundle["macro_step"]["source_s_record_sha256s"], ["a" * 64])
        self.assertIsNone(audit.source_binding)
        self.assertEqual(audit.source_s_record_sha256s, ("a" * 64,))
        self.assertFalse(bundle["evidence_scope"]["joint_publish"])

    def test_policy_and_harness_transaction_mismatch_is_rejected(self) -> None:
        policy_record, harness_record = _receipts()
        policy_audit, harness_audit = _audits(harness_transaction="macro-2")
        policy_patch, harness_patch = self._patch_audits(
            policy=policy_audit,
            harness=harness_audit,
        )
        with (
            policy_patch,
            harness_patch,
            self.assertRaisesRegex(JointStepError, "transactions differ"),
        ):
            build_joint_candidate_bundle(
                policy_receipt=policy_record,
                harness_receipt=harness_record,
                active_joint_version=_version(),
                parent_release_id="release-parent",
                macro_step_id="macro-1",
                actor_public_version=7,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )

    def test_multi_s_bundle_preserves_complete_shared_source_binding(self) -> None:
        policy_record, harness_record = _receipts()
        binding = _source_binding()
        policy_audit, harness_audit = _audits(source_binding=binding)
        policy_patch, harness_patch = self._patch_audits(
            policy=policy_audit,
            harness=harness_audit,
        )
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
                bundle,
                active_joint_version=_version(),
                actor_public_version=7,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )
        self.assertEqual(bundle["source_binding"], binding)
        self.assertEqual(audit.source_binding, binding)
        self.assertEqual(
            audit.source_s_record_sha256s,
            tuple(binding["s_record_sha256s"]),
        )

    def test_multi_s_component_binding_mismatch_is_rejected(self) -> None:
        policy_record, harness_record = _receipts()
        policy_binding = _source_binding(first_s="1" * 64)
        harness_binding = _source_binding(first_s="9" * 64)
        policy_audit, _ = _audits(source_binding=policy_binding)
        _, harness_audit = _audits(source_binding=harness_binding)
        # Keep the common digest claim equal so the complete binding comparison,
        # rather than the legacy one-digest check, must reject the candidates.
        harness_audit.source_joint_credit_sha256 = (  # type: ignore[misc]
            policy_audit.source_joint_credit_sha256
        )
        policy_patch, harness_patch = self._patch_audits(
            policy=policy_audit,
            harness=harness_audit,
        )
        with (
            policy_patch,
            harness_patch,
            self.assertRaisesRegex(JointStepError, "bindings"),
        ):
            build_joint_candidate_bundle(
                policy_receipt=policy_record,
                harness_receipt=harness_record,
                active_joint_version=_version(),
                parent_release_id="release-parent",
                macro_step_id="macro-1",
                actor_public_version=7,
                harness_public_version="harness-parent",
                require_receipt_files=False,
            )

    def test_rehashed_multi_s_receipt_binding_tamper_is_rejected(self) -> None:
        policy_record, harness_record = _receipts()
        binding = _source_binding()
        policy_audit, harness_audit = _audits(source_binding=binding)
        policy_patch, harness_patch = self._patch_audits(
            policy=policy_audit,
            harness=harness_audit,
        )
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

        def policy_validator(record, **_kwargs):
            candidate_binding = record.get("source_binding", binding)
            candidate, _ = _audits(source_binding=candidate_binding)
            candidate.record_sha256 = record["record_sha256"]
            return candidate

        def harness_validator(record, **_kwargs):
            candidate_binding = record.get("source_binding", binding)
            _, candidate = _audits(source_binding=candidate_binding)
            return candidate

        for receipt_name, validator_name, validator in (
            (
                "policy",
                "validate_areal_policy_candidate",
                policy_validator,
            ),
            (
                "harness",
                "_validate_harness_receipt",
                harness_validator,
            ),
        ):
            forged = deepcopy(bundle)
            forged_receipt = forged["receipts"][receipt_name]
            forged_receipt["source_binding"] = _source_binding(first_s="9" * 64)
            _resign(forged_receipt)
            forged["receipts"][f"{receipt_name}_sha256"] = forged_receipt[
                "record_sha256"
            ]
            _resign(forged)
            patch_target = f"jphrl.training.joint_step.{validator_name}"
            other_policy, other_harness = self._patch_audits(
                policy=policy_audit,
                harness=harness_audit,
            )
            selected = patch(patch_target, side_effect=validator)
            other = other_harness if receipt_name == "policy" else other_policy
            with (
                self.subTest(receipt=receipt_name),
                selected,
                other,
                self.assertRaisesRegex(JointStepError, "bindings"),
            ):
                validate_joint_candidate_bundle(
                    forged,
                    active_joint_version=_version(),
                    actor_public_version=7,
                    harness_public_version="harness-parent",
                    require_receipt_files=False,
                )

    def test_caller_cannot_lie_about_git_root_for_barrier_journal(self) -> None:
        policy_record, harness_record = _receipts()
        with self.assertRaisesRegex(ValueError, "outside Git checkout"):
            seal_joint_candidate_bundle(
                seal_root=repository_root() / "must-not-be-created" / "macro-1",
                transaction_journal_root="/tmp/jph-joint-transactions",
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
                    transaction_journal_root=root / "transactions",
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
                    transaction_journal_root=root / "transactions",
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
                self.assertRaisesRegex(JointStepError, "already claimed"),
            ):
                seal_joint_candidate_bundle(
                    seal_root=root / "runs" / "macro-1",
                    transaction_journal_root=root / "transactions",
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

    def test_one_s_receipt_cannot_be_claimed_across_macro_roots(self) -> None:
        policy_record, harness_record = _receipts()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journal = root / "transactions"
            policy_patch, harness_patch = self._patch_audits()
            with (
                patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                policy_patch,
                harness_patch,
            ):
                seal_joint_candidate_bundle(
                    seal_root=root / "run-a" / "joint-candidate",
                    transaction_journal_root=journal,
                    project_root=root / "src" / "repo",
                    policy_receipt=policy_record,
                    harness_receipt=harness_record,
                    active_joint_version=_version(),
                    parent_release_id="release-parent",
                    macro_step_id="macro-1",
                    actor_public_version=7,
                    harness_public_version="harness-parent",
                    require_receipt_files=False,
                )

            policy_audit, harness_audit = _audits(
                policy_transaction="macro-2",
                harness_transaction="macro-2",
            )
            policy_patch, harness_patch = self._patch_audits(
                policy=policy_audit,
                harness=harness_audit,
            )
            with (
                patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                policy_patch,
                harness_patch,
                self.assertRaisesRegex(JointStepError, "already claimed"),
            ):
                seal_joint_candidate_bundle(
                    seal_root=root / "run-b" / "joint-candidate",
                    transaction_journal_root=journal,
                    project_root=root / "src" / "repo",
                    policy_receipt=policy_record,
                    harness_receipt=harness_record,
                    active_joint_version=_version(),
                    parent_release_id="release-parent",
                    macro_step_id="macro-2",
                    actor_public_version=7,
                    harness_public_version="harness-parent",
                    require_receipt_files=False,
                )

            claims = list((journal / "claimed-source-records").glob("*.json"))
            self.assertEqual(len(claims), 1)
            persisted = json.loads(claims[0].read_text(encoding="utf-8"))
            self.assertEqual(persisted["transaction_id"], "macro-1")
            self.assertEqual(persisted["source_joint_credit_sha256"], "a" * 64)

    def test_multi_s_claim_collision_removes_only_new_partial_claims(self) -> None:
        policy_record, harness_record = _receipts()
        binding = _source_binding()
        policy_audit, harness_audit = _audits(source_binding=binding)
        callbacks: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            journal = root / "transactions"
            claim_root = journal / "claimed-source-records"
            claim_root.mkdir(parents=True)
            preexisting = claim_root / f"{binding['s_record_sha256s'][2]}.json"
            preexisting.write_text("preexisting", encoding="utf-8")
            policy_patch, harness_patch = self._patch_audits(
                policy=policy_audit,
                harness=harness_audit,
            )
            with (
                patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                policy_patch,
                harness_patch,
                self.assertRaisesRegex(JointStepError, "already claimed"),
            ):
                seal_joint_candidate_bundle(
                    seal_root=root / "runs" / "macro-1",
                    transaction_journal_root=journal,
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
            self.assertEqual(list(claim_root.iterdir()), [preexisting])
            self.assertEqual(preexisting.read_text(encoding="utf-8"), "preexisting")
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
                transaction_journal_root=root / "transactions",
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
