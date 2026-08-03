from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import SmokeHarnessController
from jphrl.harness.spec import HarnessSpec
from jphrl.models.base import ModelResponse
from jphrl.runner import run_calculator_smoke
from jphrl.trajectory.areal_agent_service_adapter import (
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    validate_agent_service_training_trace,
)
from jphrl.trajectory.areal_data_proxy_pre_batch import (
    ArealDataProxyPreBatchHookError,
    VerifiedDataProxyPreBatchHook,
)
from jphrl.trajectory.areal_online_binding import (
    ArealOnlineBindingError,
    PersistentAgentServicePreBatchBinder,
    stage_agent_service_training_binding,
)
from jphrl.trajectory.hermes_model_call_receipts import (
    HermesModelCallReceipt,
    receipts_from_public_dicts,
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


class ContractInvalidCalculatorModel:
    policy_version = "real-policy-v7"
    tokenizer_version = "real-tokenizer-v1"

    def generate(self, messages, max_new_tokens):
        del messages, max_new_tokens
        return ModelResponse(
            text='{"tool":"calculator","expression":"17 + 25"}',
            input_token_ids=[10, 11],
            output_token_ids=[20],
            output_token_logprobs=[-0.2],
            output_versions=[7],
            completion_loss_mask=[1],
            policy_version="crossed-policy-version",
            tokenizer_version=self.tokenizer_version,
            policy_kind="causal_lm",
            token_metadata_status="available",
        )


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


class ArealOnlineBindingTests(unittest.TestCase):
    def _trace(self):
        result = run_calculator_smoke(
            model=TokenBackedCalculatorModel(),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertTrue(result.success)
        return result.trace

    def _receipts(self, trace):
        session = AgentServiceSessionReceipt(
            group_id="group-1",
            session_id="add-17-25-0",
        )
        model_call_ids = validate_agent_service_training_trace(trace)[
            "model_call_ids"
        ]

        def call_receipt(
            model_call_id: str,
            interaction_id: str,
            ordinal: int,
            parent_model_call_id: str | None,
        ) -> HermesModelCallReceipt:
            return HermesModelCallReceipt(
                model_call_id=model_call_id,
                interaction_id=interaction_id,
                ordinal=ordinal,
                parent_model_call_id=parent_model_call_id,
                session_id=session.session_id,
            )

        emitted_calls = [
            call_receipt(model_call_ids[0], "interaction-1", 0, None),
            call_receipt(
                model_call_ids[1],
                "interaction-2",
                1,
                model_call_ids[0],
            ),
        ]
        calls = list(
            receipts_from_public_dicts(
                [
                    {
                        "model_call_id": call.model_call_id,
                        "interaction_id": call.interaction_id,
                        "ordinal": call.ordinal,
                        "parent_model_call_id": call.parent_model_call_id,
                        "session_id": call.session_id,
                    }
                    for call in emitted_calls
                ],
                expected_session_id=session.session_id,
            )
        )
        trajectory = AgentServiceTrajectoryReceipt(
            session_id=session.session_id,
            trajectory_id=3,
            interaction_count=2,
            ready_transition=True,
        )
        return session, calls, trajectory

    def test_persistent_binding_joins_individual_and_concat_exactly_once(self) -> None:
        for style, expected_samples in (("individual", 2), ("concat", 1)):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as directory:
                trace = self._trace()
                session, calls, trajectory = self._receipts(trace)
                stage_path = stage_agent_service_training_binding(
                    journal_root=directory,
                    trace=trace,
                    session=session,
                    model_calls=calls,
                    trajectory=trajectory,
                    export_style=style,
                    turn_discount=1.0,
                )
                staged_text = stage_path.read_text(encoding="utf-8")
                self.assertNotIn("session_api_key", staged_text)
                self.assertNotIn("admin_api_key", staged_text)
                self.assertEqual(stage_path.stat().st_mode & 0o777, 0o600)

                binder = PersistentAgentServicePreBatchBinder(directory)
                hook = VerifiedDataProxyPreBatchHook(binder)
                asyncio.run(
                    hook(
                        session_id=session.session_id,
                        trajectory_id=trajectory.trajectory_id,
                        interactions=_interactions(style),
                        discount=1.0,
                        style=style,
                    )
                )

                record_files = list((stage_path.parents[1] / "records").glob("*.json"))
                marker_files = list((stage_path.parents[1] / "finalized").glob("*.json"))
                self.assertEqual(len(record_files), 1)
                self.assertEqual(len(marker_files), 1)
                record = json.loads(record_files[0].read_text(encoding="utf-8"))
                self.assertEqual(record["training_archive"]["sample_count"], expected_samples)
                self.assertEqual(
                    record["identity"]["joint_version_id"],
                    trace.joint_version.version_id,
                )
                bindings = record["training_archive"]["interaction_sidecar"][
                    "bindings"
                ]
                self.assertEqual(
                    [binding["model_call_id"] for binding in bindings],
                    [call.model_call_id for call in calls],
                )
                self.assertEqual(
                    [binding["interaction_id"] for binding in bindings],
                    ["interaction-1", "interaction-2"],
                )
                self.assertEqual(
                    [binding["ordinal"] for binding in bindings],
                    [0, 1],
                )
                self.assertIsNone(bindings[0]["parent_interaction_id"])
                self.assertEqual(
                    bindings[1]["parent_interaction_id"],
                    "interaction-1",
                )
                self.assertEqual(
                    {binding["session_id"] for binding in bindings},
                    {session.session_id},
                )
                self.assertEqual(
                    {binding["trajectory_id"] for binding in bindings},
                    {trajectory.trajectory_id},
                )
                self.assertEqual(
                    {binding["joint_version_id"] for binding in bindings},
                    {trace.joint_version.version_id},
                )
                self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])
                self.assertFalse(record["evidence_scope"]["harness_optimizer_update"])

                with self.assertRaisesRegex(
                    ArealOnlineBindingError,
                    "already been finalized",
                ):
                    asyncio.run(
                        hook(
                            session_id=session.session_id,
                            trajectory_id=trajectory.trajectory_id,
                            interactions=_interactions(style),
                            discount=1.0,
                            style=style,
                        )
                    )

    def test_persistent_binding_rejects_post_batch_and_crossed_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = self._trace()
            session, calls, trajectory = self._receipts(trace)
            stage_agent_service_training_binding(
                journal_root=directory,
                trace=trace,
                session=session,
                model_calls=calls,
                trajectory=trajectory,
                export_style="individual",
                turn_discount=1.0,
            )
            hook = VerifiedDataProxyPreBatchHook(
                PersistentAgentServicePreBatchBinder(directory)
            )
            with self.assertRaisesRegex(
                ArealDataProxyPreBatchHookError,
                "post-batch trajectory data",
            ):
                asyncio.run(
                    hook(
                        session_id=session.session_id,
                        trajectory_id=trajectory.trajectory_id,
                        interactions={
                            "input_ids": [[1]],
                            "loss_mask": [[1]],
                            "logprobs": [[-0.1]],
                            "versions": [[7]],
                            "attention_mask": [[True]],
                            "rewards": [1.0],
                        },
                        discount=1.0,
                        style="individual",
                    )
                )
            with self.assertRaisesRegex(
                ArealOnlineBindingError,
                "no staged binding",
            ):
                asyncio.run(
                    hook(
                        session_id=session.session_id,
                        trajectory_id=trajectory.trajectory_id + 1,
                        interactions=_interactions("individual"),
                        discount=1.0,
                        style="individual",
                    )
                )

    def test_staging_rejects_hermes_receipt_from_another_session(self) -> None:
        trace = self._trace()
        session, calls, trajectory = self._receipts(trace)
        crossed_calls = [
            calls[0],
            replace(calls[1], session_id="another-session"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ArealOnlineBindingError,
                "cross inference sessions",
            ):
                stage_agent_service_training_binding(
                    journal_root=directory,
                    trace=trace,
                    session=session,
                    model_calls=crossed_calls,
                    trajectory=trajectory,
                    export_style="individual",
                    turn_discount=1.0,
                )
            self.assertFalse(list(Path(directory).rglob("*.json")))

    def test_invalid_episodes_are_never_staged_for_training(self) -> None:
        for model, expected_validity in (
            (FailingCalculatorModel(), "infrastructure_invalid"),
            (ContractInvalidCalculatorModel(), "trace_contract_invalid"),
        ):
            with self.subTest(validity_class=expected_validity):
                invalid_trace = run_calculator_smoke(
                    model=model,
                    task=TASKS["add-17-25"],
                    controller=SmokeHarnessController(),
                    harness_spec=HarnessSpec(),
                ).trace
                self.assertEqual(invalid_trace.validity_class, expected_validity)
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(
                        ArealOnlineBindingError,
                        "invalid infrastructure or trace-contract episode",
                    ):
                        stage_agent_service_training_binding(
                            journal_root=directory,
                            trace=invalid_trace,
                            session=AgentServiceSessionReceipt(
                                group_id="group-invalid",
                                session_id="invalid-0",
                            ),
                            model_calls=[],
                            trajectory=AgentServiceTrajectoryReceipt(
                                session_id="invalid-0",
                                trajectory_id=0,
                                interaction_count=1,
                                ready_transition=True,
                            ),
                            export_style="individual",
                            turn_discount=1.0,
                        )
                    self.assertFalse(list(Path(directory).rglob("*.json")))

    def test_secret_fields_cannot_enter_staged_or_final_records(self) -> None:
        for secret_field in ("session_api_key", "admin_api_key"):
            with self.subTest(secret_field=secret_field):
                trace = self._trace()
                trace.events[0].payload[secret_field] = "must-never-be-persisted"
                session, calls, trajectory = self._receipts(trace)
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(
                        ArealOnlineBindingError,
                        "credential field cannot enter online binding",
                    ):
                        stage_agent_service_training_binding(
                            journal_root=directory,
                            trace=trace,
                            session=session,
                            model_calls=calls,
                            trajectory=trajectory,
                            export_style="individual",
                            turn_discount=1.0,
                        )
                    self.assertFalse(list(Path(directory).rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
