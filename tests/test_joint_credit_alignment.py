from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from dataclasses import replace

from jphrl.envs.calculator import TASKS
from jphrl.harness.spec import HarnessSpec
from jphrl.runner import run_calculator_smoke
from jphrl.trajectory.areal_agent_service_adapter import (
    AgentServiceModelCallReceipt,
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    prepare_agent_service_training_record,
    validate_agent_service_training_trace,
)
from jphrl.trajectory.areal_policy_admission import (
    build_policy_training_admission,
)
from jphrl.trajectory.harness_action_admission import (
    admit_real_harness_action_samples,
    validate_harness_action_admission_record,
)
from jphrl.trajectory.joint_credit_alignment import (
    ESTIMATOR_VERSION,
    DualCreditEstimatorSpec,
    JointCreditAlignmentError,
    build_frozen_joint_credit_alignment,
    validate_frozen_joint_credit_alignment,
)
from tests.test_areal_policy_admission import _interactions
from tests.test_harness_action_admission import (
    LiveLearnableHarnessController,
    TokenBackedCalculatorModel,
)


def _resign(record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(payload).hexdigest()


class JointCreditAlignmentTests(unittest.TestCase):
    def _components(self, style: str = "individual"):
        result = run_calculator_smoke(
            model=TokenBackedCalculatorModel(),
            task=TASKS["add-17-25"],
            controller=LiveLearnableHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertTrue(result.success)
        trace = result.trace
        model_call_ids = tuple(
            validate_agent_service_training_trace(trace)["model_call_ids"]
        )
        session = AgentServiceSessionReceipt(
            group_id="group-s",
            session_id=f"session-s-{style}",
        )
        calls = [
            AgentServiceModelCallReceipt(
                model_call_id=model_call_ids[0],
                interaction_id="interaction-1",
                ordinal=0,
                parent_model_call_id=None,
            ),
            AgentServiceModelCallReceipt(
                model_call_id=model_call_ids[1],
                interaction_id="interaction-2",
                ordinal=1,
                parent_model_call_id=model_call_ids[0],
            ),
        ]
        trajectory = AgentServiceTrajectoryReceipt(
            session_id=session.session_id,
            trajectory_id=17,
            interaction_count=2,
            ready_transition=True,
        )
        training_record = prepare_agent_service_training_record(
            trace=trace,
            session=session,
            model_calls=calls,
            trajectory=trajectory,
            exported_interactions=_interactions(style),
            export_style=style,
            turn_discount=1.0,
        )
        policy_admission = build_policy_training_admission(
            training_record,
            active_joint_version=trace.joint_version,
        )
        harness_admission = admit_real_harness_action_samples(
            trace=trace,
            active_joint_version=trace.joint_version,
            pre_batch_training_record=training_record,
        )
        estimator = DualCreditEstimatorSpec(
            estimator_version=ESTIMATOR_VERSION,
            parent_joint_version_id=trace.joint_version.version_id,
            policy_source="policy-frozen-terminal-baseline-v1",
            harness_source="harness-frozen-terminal-baseline-v1",
            policy_baseline_snapshot_id="policy-baseline-snapshot-7",
            harness_baseline_snapshot_id="harness-baseline-snapshot-3",
            policy_baselines={
                model_call_id: 0.25 + 0.25 * index
                for index, model_call_id in enumerate(model_call_ids)
            },
            harness_baselines={
                action.decision_id: 0.1 + 0.1 * index
                for index, action in enumerate(harness_admission.actions)
            },
        )
        return trace, policy_admission, harness_admission, estimator

    def test_individual_and_concat_persist_exact_dual_credit(self) -> None:
        for style, expected_sample_count in (("individual", 2), ("concat", 1)):
            with self.subTest(style=style):
                trace, policy, harness, estimator = self._components(style)

                record = build_frozen_joint_credit_alignment(
                    policy_admission=policy,
                    harness_admission=harness.to_record(),
                    active_joint_version=trace.joint_version,
                    estimator=estimator,
                )
                persisted = json.loads(json.dumps(record))
                audit = validate_frozen_joint_credit_alignment(
                    persisted,
                    active_joint_version=trace.joint_version,
                )

                self.assertEqual(audit["policy_sample_count"], expected_sample_count)
                self.assertEqual(audit["policy_decision_span_count"], 2)
                self.assertEqual(audit["harness_action_count"], 2)
                self.assertEqual(
                    record["identity"]["source_training_record_sha256"],
                    harness.source_training_record_sha256,
                )
                self.assertEqual(
                    record["admissions"]["policy_admission_record"],
                    policy,
                )
                self.assertEqual(
                    record["admissions"]["harness_admission_sha256"],
                    harness.digest,
                )
                for sample in record["policy_samples"]:
                    self.assertEqual(
                        sample["credit_mask"][0],
                        sample["tensor_dict"]["loss_mask"][0],
                    )
                    for position, trainable in enumerate(sample["credit_mask"][0]):
                        if not trainable:
                            self.assertEqual(
                                sample["advantage_tensor"][0][position], 0.0
                            )
                for sample in record["harness_samples"]:
                    self.assertEqual(
                        sample["masked_advantage"],
                        sample["advantage"] * sample["action"]["harness_loss_mask"],
                    )
                self.assertEqual(
                    record["evidence_scope"],
                    {
                        "policy_samples_admitted": True,
                        "harness_action_samples_admitted": True,
                        "policy_advantages_aligned": True,
                        "harness_advantages_aligned": True,
                        "policy_optimizer_update": False,
                        "harness_optimizer_update": False,
                    },
                )

    def test_frozen_estimator_requires_real_distinct_complete_provenance(self) -> None:
        trace, policy, harness, estimator = self._components()
        cases = (
            (
                replace(
                    estimator,
                    policy_baselines={
                        next(iter(estimator.policy_baselines)): 0.25,
                    },
                ),
                "Policy baseline targets",
            ),
            (
                replace(estimator, harness_source=estimator.policy_source),
                "must remain distinct",
            ),
            (
                replace(
                    estimator,
                    harness_baseline_snapshot_id=(
                        estimator.policy_baseline_snapshot_id
                    ),
                ),
                "baseline snapshots must remain distinct",
            ),
            (
                replace(estimator, policy_source="synthetic-policy-credit"),
                "synthetic or placeholder",
            ),
            (
                replace(estimator, parent_joint_version_id="stale-version"),
                "parent differs",
            ),
        )
        for invalid_estimator, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    JointCreditAlignmentError,
                    message,
                ),
            ):
                build_frozen_joint_credit_alignment(
                    policy_admission=policy,
                    harness_admission=harness,
                    active_joint_version=trace.joint_version,
                    estimator=invalid_estimator,
                )

    def test_persisted_q_r_and_s_tampering_fail_closed(self) -> None:
        trace, policy, harness, estimator = self._components()
        record = build_frozen_joint_credit_alignment(
            policy_admission=policy,
            harness_admission=harness,
            active_joint_version=trace.joint_version,
            estimator=estimator,
        )

        q_tampered = deepcopy(record)
        q_record = q_tampered["admissions"]["policy_admission_record"]
        q_record["evidence_scope"]["policy_optimizer_update"] = True
        _resign(q_record)
        _resign(q_tampered)

        policy_tampered = deepcopy(record)
        policy_tampered["policy_samples"][0]["advantage_tensor"][0][0] = 1.0
        _resign(policy_tampered)

        harness_tampered = deepcopy(record)
        harness_tampered["harness_samples"][0]["masked_advantage"] = 99.0
        _resign(harness_tampered)

        optimizer_claim = deepcopy(record)
        optimizer_claim["evidence_scope"]["harness_optimizer_update"] = True
        _resign(optimizer_claim)

        for altered, message in (
            (q_tampered, "evidence"),
            (policy_tampered, "persisted Q admission"),
            (harness_tampered, "persisted R admission"),
            (optimizer_claim, "evidence"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    JointCreditAlignmentError,
                    message,
                ),
            ):
                validate_frozen_joint_credit_alignment(
                    altered,
                    active_joint_version=trace.joint_version,
                )

    def test_crossed_or_post_batch_admissions_never_align(self) -> None:
        trace, policy, harness, estimator = self._components()
        crossed_harness = replace(
            harness,
            source_training_record_sha256="0" * 64,
        )
        with self.assertRaisesRegex(
            JointCreditAlignmentError,
            "share one P training record",
        ):
            build_frozen_joint_credit_alignment(
                policy_admission=policy,
                harness_admission=crossed_harness,
                active_joint_version=trace.joint_version,
                estimator=estimator,
            )

        post_batch = {
            "input_ids": [[10, 20]],
            "loss_mask": [[0, 1]],
            "logprobs": [[0.0, -0.2]],
            "versions": [[-1, 7]],
            "attention_mask": [[True, True]],
            "rewards": [1.0],
        }
        with self.assertRaisesRegex(
            (JointCreditAlignmentError, ValueError),
            "schema|field set",
        ):
            build_frozen_joint_credit_alignment(
                policy_admission=post_batch,
                harness_admission=harness,
                active_joint_version=trace.joint_version,
                estimator=estimator,
            )

    def test_harness_loss_mask_controls_only_harness_credit(self) -> None:
        trace, policy, harness, estimator = self._components()
        harness_record = harness.to_record()
        harness_record["actions"][0]["harness_loss_mask"] = 0
        _resign(harness_record)
        masked_harness = validate_harness_action_admission_record(
            harness_record,
            active_joint_version=trace.joint_version,
        )

        record = build_frozen_joint_credit_alignment(
            policy_admission=policy,
            harness_admission=harness_record,
            active_joint_version=trace.joint_version,
            estimator=estimator,
        )

        self.assertEqual(record["harness_samples"][0]["masked_advantage"], 0.0)
        self.assertNotEqual(record["harness_samples"][0]["advantage"], 0.0)
        self.assertEqual(record["summary"]["harness_trainable_action_count"], 1)
        self.assertEqual(masked_harness.actions[0].harness_loss_mask, 0)


if __name__ == "__main__":
    unittest.main()
