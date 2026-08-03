from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import SmokeHarnessController
from jphrl.harness.spec import HarnessSpec
from jphrl.models.base import ModelResponse
from jphrl.runner import run_calculator_smoke
from jphrl.trajectory.areal_agent_service_adapter import (
    AgentServiceModelCallReceipt,
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    prepare_agent_service_training_record,
    validate_agent_service_training_record,
    validate_agent_service_training_trace,
)
from jphrl.trajectory.areal_policy_admission import (
    ArealPolicyAdmissionError,
    areal_policy_tensor_batch,
    build_policy_training_admission,
    validate_policy_training_admission,
)


def _resign(record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(payload).hexdigest()


class TokenBackedCalculatorModel:
    policy_version = "real-policy-v7"
    tokenizer_version = "real-tokenizer-v1"

    def __init__(self) -> None:
        self._call_index = 0

    def generate(self, messages, max_new_tokens):
        del messages, max_new_tokens
        if self._call_index == 0:
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
        self._call_index += 1
        return response


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


class ArealPolicyAdmissionTests(unittest.TestCase):
    def _training_record(self, style: str = "individual"):
        result = run_calculator_smoke(
            model=TokenBackedCalculatorModel(),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        trace = result.trace
        call_ids = validate_agent_service_training_trace(trace)["model_call_ids"]
        session = AgentServiceSessionReceipt(
            group_id="group-1", session_id="add-17-25-0"
        )
        calls = [
            AgentServiceModelCallReceipt(
                model_call_id=call_ids[0],
                interaction_id="interaction-1",
                ordinal=0,
                parent_model_call_id=None,
            ),
            AgentServiceModelCallReceipt(
                model_call_id=call_ids[1],
                interaction_id="interaction-2",
                ordinal=1,
                parent_model_call_id=call_ids[0],
            ),
        ]
        trajectory = AgentServiceTrajectoryReceipt(
            session_id=session.session_id,
            trajectory_id=3,
            interaction_count=2,
            ready_transition=True,
        )
        record = prepare_agent_service_training_record(
            trace=trace,
            session=session,
            model_calls=calls,
            trajectory=trajectory,
            exported_interactions=_interactions(style),
            export_style=style,
            turn_discount=1.0,
        )
        return trace, record

    def test_individual_and_concat_retain_real_areal_batches_and_provenance(self):
        for style, expected_samples, expected_spans in (
            ("individual", 2, (1, 1)),
            ("concat", 1, (2,)),
        ):
            with self.subTest(style=style):
                trace, source = self._training_record(style)
                admission = build_policy_training_admission(
                    source, active_joint_version=trace.joint_version
                )
                audit = validate_policy_training_admission(
                    admission, active_joint_version=trace.joint_version
                )

                self.assertEqual(audit.joint_version, trace.joint_version)
                self.assertEqual(audit.episode_id, trace.episode_id)
                self.assertEqual(
                    audit.model_call_ids,
                    tuple(source["trace"]["model_call_ids"]),
                )
                self.assertEqual(len(audit.samples), expected_samples)
                self.assertEqual(
                    tuple(len(sample.decision_spans) for sample in audit.samples),
                    expected_spans,
                )
                self.assertTrue(
                    all(
                        span.inference_engine_version == 7
                        for sample in audit.samples
                        for span in sample.decision_spans
                    )
                )
                self.assertEqual(
                    areal_policy_tensor_batch(
                        admission, active_joint_version=trace.joint_version
                    ),
                    tuple(
                        sample["tensor_dict"]
                        for sample in source["training_archive"]["samples"]
                    ),
                )
                self.assertEqual(audit.digest, admission["record_sha256"])
                self.assertFalse(
                    admission["evidence_scope"]["policy_advantages_attached"]
                )
                self.assertFalse(admission["evidence_scope"]["policy_optimizer_update"])
                self.assertFalse(
                    admission["evidence_scope"]["harness_optimizer_update"]
                )
                self.assertTrue(
                    all(
                        "advantages" not in sample["tensor_dict"]
                        for sample in admission["samples"]
                    )
                )
                self.assertNotIn("session_api_key", json.dumps(admission))
                self.assertNotIn("admin_api_key", json.dumps(admission))

    def test_stale_joint_version_is_never_admitted(self):
        trace, source = self._training_record()
        stale = replace(trace.joint_version, policy="stale-policy-release")
        with self.assertRaisesRegex(
            ArealPolicyAdmissionError, "lag-zero active version"
        ):
            build_policy_training_admission(source, active_joint_version=stale)

        admission = build_policy_training_admission(
            source, active_joint_version=trace.joint_version
        )
        with self.assertRaisesRegex(
            ArealPolicyAdmissionError, "lag-zero active version"
        ):
            validate_policy_training_admission(admission, active_joint_version=stale)

    def test_p_record_must_remain_fail_closed_and_optimizer_free(self):
        trace, source = self._training_record()
        forged = copy.deepcopy(source)
        forged["trace"]["validity_class"] = "infrastructure_invalid"
        forged["evidence_scope"]["policy_optimizer_update"] = True
        _resign(forged)
        with self.assertRaises(ArealPolicyAdmissionError):
            build_policy_training_admission(
                forged, active_joint_version=trace.joint_version
            )

        post_batch = copy.deepcopy(source)
        post_batch["training_archive"] = {
            "input_ids": [[1]],
            "loss_mask": [[1]],
            "logprobs": [[-0.1]],
            "versions": [[7]],
            "attention_mask": [[True]],
            "rewards": [1.0],
        }
        with self.assertRaisesRegex(
            ArealPolicyAdmissionError, "training archive field set"
        ):
            build_policy_training_admission(
                post_batch, active_joint_version=trace.joint_version
            )

    def test_mixed_version_and_positive_logprob_fail_after_valid_p_hashes(self):
        trace, source = self._training_record()
        for mutation, message in (
            ("mixed-version", "one inference engine version"),
            ("positive-logprob", "cannot be positive"),
        ):
            with self.subTest(mutation=mutation):
                tampered = copy.deepcopy(source)
                tensors = tampered["training_archive"]["samples"][0]["tensor_dict"]
                if mutation == "mixed-version":
                    tensors["versions"][0][3] = 8
                else:
                    # P permits a tiny numerical tolerance; Q deliberately uses
                    # the mathematical old-logprob contract (<= 0).
                    tensors["logprobs"][0][2] = 0.000001
                _resign(tampered["training_archive"])
                _resign(tampered)
                validate_agent_service_training_record(tampered)
                with self.assertRaisesRegex(ArealPolicyAdmissionError, message):
                    build_policy_training_admission(
                        tampered, active_joint_version=trace.joint_version
                    )

    def test_admission_hash_summary_and_evidence_are_revalidated(self):
        trace, source = self._training_record("concat")
        admission = build_policy_training_admission(
            source, active_joint_version=trace.joint_version
        )
        for mutation, message in (
            ("summary", "summary differs"),
            ("evidence", "evidence scope differs"),
        ):
            with self.subTest(mutation=mutation):
                tampered = copy.deepcopy(admission)
                if mutation == "summary":
                    tampered["summary"]["trainable_token_count"] += 1
                else:
                    tampered["evidence_scope"]["policy_optimizer_update"] = True
                _resign(tampered)
                with self.assertRaisesRegex(ArealPolicyAdmissionError, message):
                    validate_policy_training_admission(
                        tampered, active_joint_version=trace.joint_version
                    )


if __name__ == "__main__":
    unittest.main()
