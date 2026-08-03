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
from jphrl.trajectory.areal_agent_service_adapter import (
    AgentServiceModelCallReceipt,
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    prepare_agent_service_training_record,
    validate_agent_service_training_trace,
)
from jphrl.trajectory.harness_action_admission import (
    AdmittedHarnessActionBatch,
    HarnessActionAdmissionError,
    admit_real_harness_action_samples,
    validate_harness_action_admission_record,
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


class FailingCalculatorModel:
    policy_version = "real-policy-v7"
    tokenizer_version = "real-tokenizer-v1"

    def generate(self, messages, max_new_tokens):
        del messages, max_new_tokens
        raise ConnectionError("inference unavailable")


class WrongAnswerCalculatorModel(TokenBackedCalculatorModel):
    def generate(self, messages, max_new_tokens):
        response = super().generate(messages, max_new_tokens)
        if self.call_index == 2:
            return replace(response, text='{"answer":"41"}')
        return response


class ContractInvalidCalculatorModel(TokenBackedCalculatorModel):
    def generate(self, messages, max_new_tokens):
        response = super().generate(messages, max_new_tokens)
        if self.call_index == 1:
            return replace(response, policy_version="crossed-policy-version")
        return response


class LiveLearnableHarnessController:
    """A live categorical controller with non-degenerate, recorded probabilities."""

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
    reward: float,
) -> dict[str, object]:
    return {
        "input_ids": [input_ids],
        "loss_mask": [loss_mask],
        "logprobs": [logprobs],
        "versions": [versions],
        "attention_mask": [[True] * len(input_ids)],
        "rewards": [reward],
    }


class FakeInteraction:
    def __init__(
        self,
        *,
        interaction_id: str,
        parent: FakeInteraction | None,
        input_tokens: list[int],
        output_tokens: list[int],
        output_logprobs: list[float],
        output_versions: list[int],
        tensors: dict[str, object],
    ) -> None:
        self.interaction_id = interaction_id
        self.parent = parent
        self.chat_template_type = "hf"
        self.model_response = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
        )
        self._tensors = tensors

    def to_tensor_dict(self) -> dict[str, object]:
        return self._tensors


def _interactions(*, reward: float) -> OrderedDict[str, FakeInteraction]:
    first = FakeInteraction(
        interaction_id="interaction-1",
        parent=None,
        input_tokens=[10, 11],
        output_tokens=[20, 21],
        output_logprobs=[-0.2, -0.3],
        output_versions=[7, 7],
        tensors=_tensor_dict(
            input_ids=[10, 11, 20, 21],
            loss_mask=[0, 0, 1, 1],
            logprobs=[0.0, 0.0, -0.2, -0.3],
            versions=[-1, -1, 7, 7],
            reward=reward,
        ),
    )
    second = FakeInteraction(
        interaction_id="interaction-2",
        parent=first,
        input_tokens=[10, 11, 20, 21, 30],
        output_tokens=[40],
        output_logprobs=[-0.4],
        output_versions=[7],
        tensors=_tensor_dict(
            input_ids=[10, 11, 20, 21, 30, 40],
            loss_mask=[0, 0, 0, 0, 0, 1],
            logprobs=[0.0, 0.0, 0.0, 0.0, 0.0, -0.4],
            versions=[-1, -1, -1, -1, -1, 7],
            reward=reward,
        ),
    )
    return OrderedDict(
        (interaction.interaction_id, interaction) for interaction in (first, second)
    )


def _rehash(record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(payload).hexdigest()


class HarnessActionAdmissionTests(unittest.TestCase):
    def _trace(self):
        result = run_calculator_smoke(
            model=TokenBackedCalculatorModel(),
            task=TASKS["add-17-25"],
            controller=LiveLearnableHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertTrue(result.success)
        return result.trace

    def _record(self, trace):
        model_call_ids = validate_agent_service_training_trace(trace)["model_call_ids"]
        session = AgentServiceSessionReceipt(
            group_id="group-r",
            session_id="session-r",
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
            trajectory_id=11,
            interaction_count=2,
            ready_transition=True,
        )
        return prepare_agent_service_training_record(
            trace=trace,
            session=session,
            model_calls=calls,
            trajectory=trajectory,
            exported_interactions=_interactions(reward=float(trace.reward)),
            export_style="individual",
            turn_discount=1.0,
        )

    def _admit(self):
        trace = self._trace()
        record = self._record(trace)
        batch = admit_real_harness_action_samples(
            trace=trace,
            active_joint_version=trace.joint_version,
            pre_batch_training_record=record,
        )
        return trace, record, batch

    def test_admits_live_actions_and_emits_stable_credit_free_record(self) -> None:
        trace, record, batch = self._admit()

        self.assertIsInstance(batch, AdmittedHarnessActionBatch)
        self.assertEqual(batch.episode_id, trace.episode_id)
        self.assertEqual(batch.joint_version, trace.joint_version)
        self.assertEqual(batch.source_training_record_sha256, record["record_sha256"])
        self.assertEqual(len(batch.actions), 2)
        self.assertEqual(
            [sample.decision_id for sample in batch.actions],
            ["live-decision-0", "live-decision-1"],
        )
        self.assertEqual([sample.decision_ordinal for sample in batch.actions], [0, 1])
        self.assertTrue(all(sample.harness_loss_mask == 1 for sample in batch.actions))
        self.assertTrue(
            all(
                sample.action_ids == tuple(action.value for action in HarnessAction)
                for sample in batch.actions
            )
        )
        emitted = batch.to_record()
        self.assertEqual(emitted["record_sha256"], batch.digest)
        self.assertEqual(
            emitted["evidence_scope"],
            {
                "pre_batch_interaction_binding": True,
                "harness_action_samples_admitted": True,
                "harness_advantages_attached": False,
                "policy_optimizer_update": False,
                "harness_optimizer_update": False,
            },
        )
        audit = validate_harness_action_admission_record(
            json.loads(json.dumps(emitted)),
            active_joint_version=trace.joint_version,
        )
        self.assertEqual(
            [sample.decision_id for sample in audit.actions],
            ["live-decision-0", "live-decision-1"],
        )
        self.assertEqual(
            sum(sample.harness_loss_mask for sample in audit.actions),
            2,
        )

    def test_rejects_policy_tensor_batch_instead_of_p_record(self) -> None:
        trace = self._trace()
        post_batch_policy_tensors = {
            "input_ids": [[10, 20]],
            "loss_mask": [[0, 1]],
            "logprobs": [[0.0, -0.2]],
            "versions": [[-1, 7]],
            "attention_mask": [[True, True]],
            "rewards": [1.0],
        }
        with self.assertRaisesRegex(
            HarnessActionAdmissionError,
            "unknown schema|P pre-batch training record",
        ):
            admit_real_harness_action_samples(
                trace=trace,
                active_joint_version=trace.joint_version,
                pre_batch_training_record=post_batch_policy_tensors,
            )

    def test_rejects_stale_joint_version_and_crossed_p_record(self) -> None:
        first = self._trace()
        first_record = self._record(first)
        stale_active = replace(
            first.joint_version,
            harness_controller="next-harness-controller",
        )
        with self.assertRaisesRegex(
            HarnessActionAdmissionError,
            "lag-zero active version",
        ):
            admit_real_harness_action_samples(
                trace=first,
                active_joint_version=stale_active,
                pre_batch_training_record=first_record,
            )

        second = self._trace()
        with self.assertRaisesRegex(
            HarnessActionAdmissionError,
            "episode differs|trace hash differs",
        ):
            admit_real_harness_action_samples(
                trace=second,
                active_joint_version=second.joint_version,
                pre_batch_training_record=first_record,
            )

    def test_rejects_non_live_or_incomplete_harness_decision(self) -> None:
        def wrong_producer(trace, event_index):
            trace.events[event_index] = replace(
                trace.events[event_index], producer="policy"
            )

        def missing_state(trace, event_index):
            trace.events[event_index].payload.pop("state")

        def wrong_action_schema(trace, event_index):
            trace.events[event_index].payload["action_ids"] = (
                "DIRECT",
                "OTHER",
                "VERIFY",
                "REPLAN",
                "COMPRESS",
            )

        for mutation, message in (
            (wrong_producer, "Harness producer"),
            (missing_state, "payload field set"),
            (wrong_action_schema, "five-action schema"),
        ):
            with self.subTest(message=message):
                trace = self._trace()
                altered = deepcopy(trace)
                event_index = next(
                    index
                    for index, item in enumerate(altered.events)
                    if item.kind == "harness_decision"
                )
                mutation(altered, event_index)
                record = self._record(altered)
                with self.assertRaisesRegex(
                    HarnessActionAdmissionError,
                    message,
                ):
                    admit_real_harness_action_samples(
                        trace=altered,
                        active_joint_version=altered.joint_version,
                        pre_batch_training_record=record,
                    )

    def test_rejects_wrong_masked_logprob_and_invalid_episode(self) -> None:
        trace = self._trace()
        record = self._record(trace)
        altered = deepcopy(trace)
        decision = next(
            event for event in altered.events if event.kind == "harness_decision"
        )
        decision.payload["old_harness_logprob"] = -9.0
        with self.assertRaisesRegex(
            HarnessActionAdmissionError,
            "old Harness log-prob",
        ):
            admit_real_harness_action_samples(
                trace=altered,
                active_joint_version=altered.joint_version,
                pre_batch_training_record=record,
            )

        for model, expected_validity in (
            (FailingCalculatorModel(), "infrastructure_invalid"),
            (ContractInvalidCalculatorModel(), "trace_contract_invalid"),
        ):
            with self.subTest(validity_class=expected_validity):
                invalid_trace = run_calculator_smoke(
                    model=model,
                    task=TASKS["add-17-25"],
                    controller=LiveLearnableHarnessController(),
                    harness_spec=HarnessSpec(),
                ).trace
                self.assertEqual(invalid_trace.validity_class, expected_validity)
                with self.assertRaisesRegex(
                    HarnessActionAdmissionError,
                    "invalid infrastructure|trace-contract|trainable trace",
                ):
                    admit_real_harness_action_samples(
                        trace=invalid_trace,
                        active_joint_version=invalid_trace.joint_version,
                        pre_batch_training_record=record,
                    )

    def test_policy_failure_is_admitted_without_becoming_optimizer_evidence(
        self,
    ) -> None:
        result = run_calculator_smoke(
            model=WrongAnswerCalculatorModel(),
            task=TASKS["add-17-25"],
            controller=LiveLearnableHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.trace.validity_class, "policy_failure")
        self.assertEqual(result.trace.reward, 0.0)
        record = self._record(result.trace)

        batch = admit_real_harness_action_samples(
            trace=result.trace,
            active_joint_version=result.trace.joint_version,
            pre_batch_training_record=record,
        )

        self.assertEqual(batch.validity_class, "policy_failure")
        self.assertEqual(batch.terminal_reward, 0.0)
        self.assertFalse(
            batch.to_record()["evidence_scope"]["harness_optimizer_update"]
        )

    def test_record_validator_rejects_tampering_and_optimizer_claims(self) -> None:
        _, _, batch = self._admit()
        record = batch.to_record()
        tampered = deepcopy(record)
        tampered["actions"][0]["old_harness_logprob"] = -7.0
        _rehash(tampered)
        with self.assertRaisesRegex(
            HarnessActionAdmissionError,
            "old Harness log-prob",
        ):
            validate_harness_action_admission_record(tampered)

        false_claim = deepcopy(record)
        false_claim["evidence_scope"]["harness_optimizer_update"] = True
        _rehash(false_claim)
        with self.assertRaisesRegex(
            HarnessActionAdmissionError,
            "canonical|evidence",
        ):
            validate_harness_action_admission_record(false_claim)


if __name__ == "__main__":
    unittest.main()
