from __future__ import annotations

import hashlib
import json
import math
import unittest
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import HarnessDecision
from jphrl.harness.spec import HarnessAction, HarnessSpec
from jphrl.models.base import ModelResponse
from jphrl.runner import run_calculator_smoke
from jphrl.training.areal_policy_optimizer import (
    ArealExternalAdvantageBatchError,
    build_areal_external_advantage_batch,
    validate_areal_external_advantage_batch,
    validate_areal_policy_optimizer_source,
)
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


class TokenBackedCalculatorModel:
    policy_version = "real-policy-v7"
    tokenizer_version = "real-tokenizer-v1"

    def __init__(self) -> None:
        self.call_index = 0

    def generate(self, messages, max_new_tokens):
        del messages, max_new_tokens
        if self.call_index == 0:
            response = ModelResponse(
                text='{"tool":"calculator","expression":"17 + 25"}',
                input_token_ids=[10, 11],
                output_token_ids=[20, 21],
                output_token_logprobs=[-0.2, -0.3],
                output_versions=[7, 7],
                completion_loss_mask=[1, 1],
                policy_version=self.policy_version,
                tokenizer_version=self.tokenizer_version,
                policy_kind="causal_lm",
                token_metadata_status="available",
            )
        else:
            response = ModelResponse(
                text='{"answer":"42"}',
                input_token_ids=[10, 11, 20, 21, 30],
                output_token_ids=[40],
                output_token_logprobs=[-0.4],
                output_versions=[7],
                completion_loss_mask=[1],
                policy_version=self.policy_version,
                tokenizer_version=self.tokenizer_version,
                policy_kind="causal_lm",
                token_metadata_status="available",
            )
        self.call_index += 1
        return response


class LiveLearnableHarnessController:
    version = "live-categorical-harness-v1"

    def __init__(self) -> None:
        self.ordinal = 0

    def choose(self, state):
        action_ids = tuple(action.value for action in HarnessAction)
        action = (
            HarnessAction.DIRECT
            if state.verifier_status == "not-run"
            else HarnessAction.VERIFY
        )
        action_index = action_ids.index(action.value)
        logits = tuple(
            1.25 if index == action_index else -0.25 for index in range(len(action_ids))
        )
        maximum = max(logits)
        normalizer = maximum + math.log(
            sum(math.exp(value - maximum) for value in logits)
        )
        decision = HarnessDecision(
            decision_id=f"live-decision-{self.ordinal}",
            action=action,
            old_harness_logprob=logits[action_index] - normalizer,
            controller_version=self.version,
            action_ids=action_ids,
            action_mask=(True,) * len(action_ids),
            pre_mask_logits=logits,
            harness_loss_mask=1,
        )
        self.ordinal += 1
        return decision


def _tensor_dict(
    *,
    input_ids: list[int],
    loss_mask: list[int],
    logprobs: list[float],
    versions: list[int],
) -> dict[str, object]:
    return {
        "input_ids": [input_ids],
        "loss_mask": [loss_mask],
        "logprobs": [logprobs],
        "versions": [versions],
        "attention_mask": [[True] * len(input_ids)],
        "rewards": [1.0],
    }


class FakeInteraction:
    def __init__(
        self,
        *,
        interaction_id: str,
        parent: FakeInteraction | None,
        chat_template_type: str,
        input_tokens: list[int],
        output_tokens: list[int],
        output_logprobs: list[float],
        output_versions: list[int],
        tensors: dict[str, object],
    ) -> None:
        self.interaction_id = interaction_id
        self.parent = parent
        self.chat_template_type = chat_template_type
        self.model_response = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
        )
        self._tensors = tensors

    def to_tensor_dict(self) -> dict[str, object]:
        return self._tensors


def _interactions(style: str) -> OrderedDict[str, FakeInteraction]:
    template_type = "concat" if style == "concat" else "hf"
    first = FakeInteraction(
        interaction_id="interaction-1",
        parent=None,
        chat_template_type=template_type,
        input_tokens=[10, 11],
        output_tokens=[20, 21],
        output_logprobs=[-0.2, -0.3],
        output_versions=[7, 7],
        tensors=_tensor_dict(
            input_ids=[10, 11, 20, 21],
            loss_mask=[0, 0, 1, 1],
            logprobs=[0.0, 0.0, -0.2, -0.3],
            versions=[-1, -1, 7, 7],
        ),
    )
    if style == "concat":
        second_mask = [0, 0, 1, 1, 0, 1]
        second_logprobs = [0.0, 0.0, -0.2, -0.3, 0.0, -0.4]
        second_versions = [-1, -1, 7, 7, -1, 7]
    else:
        second_mask = [0, 0, 0, 0, 0, 1]
        second_logprobs = [0.0, 0.0, 0.0, 0.0, 0.0, -0.4]
        second_versions = [-1, -1, -1, -1, -1, 7]
    second = FakeInteraction(
        interaction_id="interaction-2",
        parent=first,
        chat_template_type=template_type,
        input_tokens=[10, 11, 20, 21, 30],
        output_tokens=[40],
        output_logprobs=[-0.4],
        output_versions=[7],
        tensors=_tensor_dict(
            input_ids=[10, 11, 20, 21, 30, 40],
            loss_mask=second_mask,
            logprobs=second_logprobs,
            versions=second_versions,
        ),
    )
    if style == "concat":
        return OrderedDict(((second.interaction_id, second),))
    return OrderedDict(
        (interaction.interaction_id, interaction) for interaction in (first, second)
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
                policy_batch = build_areal_external_advantage_batch(
                    persisted,
                    active_joint_version=trace.joint_version,
                )
                policy_batch_audit = validate_areal_external_advantage_batch(
                    json.loads(json.dumps(policy_batch)),
                    active_joint_version=trace.joint_version,
                )
                optimizer_source_audit = validate_areal_policy_optimizer_source(
                    json.loads(json.dumps(policy_batch)),
                    source_joint_credit_record=persisted,
                    active_joint_version=trace.joint_version,
                )

                self.assertEqual(audit["policy_sample_count"], expected_sample_count)
                self.assertEqual(
                    len(policy_batch_audit.samples),
                    expected_sample_count,
                )
                self.assertEqual(policy_batch_audit.inference_engine_version, 7)
                self.assertEqual(
                    optimizer_source_audit.source_joint_credit_sha256,
                    persisted["record_sha256"],
                )
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

    def test_self_rehashed_optimizer_batch_cannot_replace_real_s_source(self) -> None:
        trace, policy, harness, estimator = self._components()
        source = build_frozen_joint_credit_alignment(
            policy_admission=policy,
            harness_admission=harness,
            active_joint_version=trace.joint_version,
            estimator=estimator,
        )
        batch = build_areal_external_advantage_batch(
            source,
            active_joint_version=trace.joint_version,
        )
        forged = deepcopy(batch)
        tensors = forged["samples"][0]["tensor_dict"]
        trainable_position = tensors["loss_mask"][0].index(1)
        tensors["advantages"][0][trainable_position] += 0.125
        tensors["returns"][0][trainable_position] += 0.125
        _resign(forged)

        validate_areal_external_advantage_batch(
            forged,
            active_joint_version=trace.joint_version,
        )
        with self.assertRaisesRegex(
            ArealExternalAdvantageBatchError,
            "does not exactly derive",
        ):
            validate_areal_policy_optimizer_source(
                forged,
                source_joint_credit_record=source,
                active_joint_version=trace.joint_version,
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
