from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import SmokeHarnessController
from jphrl.harness.spec import HarnessSpec
from jphrl.models.base import MockStructuredModel, ModelResponse
from jphrl.runner import run_calculator_smoke
from jphrl.trajectory.areal_agent_service_adapter import (
    ArealAgentServiceAdapterError,
    build_agent_service_interaction_sidecar,
    model_call_receipt_from_response,
    prepare_agent_service_training_record,
    session_receipt_from_start_response,
    trajectory_receipt_from_set_reward_response,
    validate_agent_service_training_record,
    validate_agent_service_training_trace,
    write_agent_service_training_record,
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


def tensor_dict(
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


class FakeInteractionCache(OrderedDict[str, FakeInteraction]):
    def export_interactions(
        self, *, style: str, reward_discount: float | None
    ) -> dict[str, FakeInteraction]:
        del reward_discount
        if style == "individual":
            return dict(self)
        if style == "concat":
            parent_ids = {
                interaction.parent.interaction_id
                for interaction in self.values()
                if interaction.parent is not None
            }
            return {
                interaction_id: interaction
                for interaction_id, interaction in self.items()
                if interaction_id not in parent_ids
            }
        raise ValueError(style)


class ArealAgentServiceAdapterTests(unittest.TestCase):
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
        session = session_receipt_from_start_response(
            {
                "group_id": "group-1",
                "sessions": [
                    {
                        "session_id": "add-17-25-0",
                        "session_api_key": "must-never-be-persisted",
                    }
                ],
            }
        )
        model_call_ids = validate_agent_service_training_trace(trace)[
            "model_call_ids"
        ]
        calls = [
            model_call_receipt_from_response(
                model_call_id=model_call_ids[0],
                response={"id": "interaction-1"},
                ordinal=0,
                parent_model_call_id=None,
            ),
            model_call_receipt_from_response(
                model_call_id=model_call_ids[1],
                response=SimpleNamespace(id="interaction-2"),
                ordinal=1,
                parent_model_call_id=model_call_ids[0],
            ),
        ]
        trajectory = trajectory_receipt_from_set_reward_response(
            {
                "message": "success",
                "interaction_count": 2,
                "session_id": "add-17-25-0",
                "trajectory_id": 3,
                "trajectory_ready": True,
                "ready_transition": True,
            }
        )
        return session, calls, trajectory

    def _cache(self, style: str) -> FakeInteractionCache:
        first = FakeInteraction(
            interaction_id="interaction-1",
            parent=None,
            chat_template_type=style,
            input_tokens=[10, 11],
            output_tokens=[20, 21],
            output_logprobs=[-0.2, -0.3],
            output_versions=[7, 7],
            tensors=tensor_dict(
                input_ids=[10, 11, 20, 21],
                loss_mask=[0, 0, 1, 1],
                logprobs=[0.0, 0.0, -0.2, -0.3],
                versions=[-1, -1, 7, 7],
            ),
        )
        second_mask = [0, 0, 1, 1, 0, 1] if style == "concat" else [0] * 5 + [1]
        second_logprobs = (
            [0.0, 0.0, -0.2, -0.3, 0.0, -0.4]
            if style == "concat"
            else [0.0] * 5 + [-0.4]
        )
        second_versions = (
            [-1, -1, 7, 7, -1, 7]
            if style == "concat"
            else [-1] * 5 + [7]
        )
        second = FakeInteraction(
            interaction_id="interaction-2",
            parent=first,
            chat_template_type=style,
            input_tokens=[10, 11, 20, 21, 30],
            output_tokens=[40],
            output_logprobs=[-0.4],
            output_versions=[7],
            tensors=tensor_dict(
                input_ids=[10, 11, 20, 21, 30, 40],
                loss_mask=second_mask,
                logprobs=second_logprobs,
                versions=second_versions,
            ),
        )
        return FakeInteractionCache(
            ((interaction.interaction_id, interaction) for interaction in (first, second))
        )

    def test_session_receipt_drops_api_credential(self) -> None:
        receipt = session_receipt_from_start_response(
            {
                "group_id": "group-1",
                "sessions": [
                    {
                        "session_id": "session-1",
                        "session_api_key": "secret-value",
                    }
                ],
            }
        )
        self.assertEqual(asdict(receipt), {"group_id": "group-1", "session_id": "session-1"})
        self.assertNotIn("secret", json.dumps(asdict(receipt)))

    def test_individual_record_binds_trace_session_and_two_rows(self) -> None:
        trace = self._trace()
        session, calls, trajectory = self._receipts(trace)
        record = prepare_agent_service_training_record(
            trace=trace,
            session=session,
            model_calls=calls,
            trajectory=trajectory,
            interaction_cache=self._cache("hf"),
            export_style="individual",
            turn_discount=1.0,
        )
        audit = validate_agent_service_training_record(record)
        self.assertEqual(audit["model_call_count"], 2)
        self.assertEqual(audit["sample_count"], 2)
        self.assertEqual(audit["export_style"], "individual")
        self.assertEqual(record["identity"]["session_id"], "add-17-25-0")
        self.assertEqual(record["identity"]["trajectory_id"], 3)
        self.assertNotIn("must-never-be-persisted", json.dumps(record))
        self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])

    def test_concat_record_runs_before_merge_and_preserves_both_calls(self) -> None:
        trace = self._trace()
        session, calls, trajectory = self._receipts(trace)
        cache = self._cache("concat")
        exported = cache.export_interactions(style="concat", reward_discount=1.0)
        record = prepare_agent_service_training_record(
            trace=trace,
            session=session,
            model_calls=calls,
            trajectory=trajectory,
            exported_interactions=exported,
            export_style="concat",
            turn_discount=1.0,
        )
        archive = record["training_archive"]
        self.assertEqual(archive["sample_count"], 1)
        self.assertEqual(
            archive["samples"][0]["included_interaction_ids"],
            ["interaction-1", "interaction-2"],
        )
        self.assertEqual(
            archive["samples"][0]["decision_spans"],
            [
                {
                    "model_call_id": calls[0].model_call_id,
                    "interaction_id": "interaction-1",
                    "start": 2,
                    "end": 4,
                },
                {
                    "model_call_id": calls[1].model_call_id,
                    "interaction_id": "interaction-2",
                    "start": 5,
                    "end": 6,
                },
            ],
        )

    def test_adapter_rejects_post_merge_batch_and_ineligible_trace(self) -> None:
        trace = self._trace()
        session, calls, trajectory = self._receipts(trace)
        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError, "exported interaction set"
        ):
            prepare_agent_service_training_record(
                trace=trace,
                session=session,
                model_calls=calls,
                trajectory=trajectory,
                exported_interactions={"input_ids": [[1, 2, 3]]},
                export_style="individual",
                turn_discount=1.0,
            )

        mock_trace = run_calculator_smoke(
            model=MockStructuredModel(),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        ).trace
        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError, "real token metadata"
        ):
            validate_agent_service_training_trace(mock_trace)

    def test_adapter_requires_exactly_one_premerge_source(self) -> None:
        trace = self._trace()
        session, calls, trajectory = self._receipts(trace)
        cache = self._cache("hf")
        arguments = {
            "trace": trace,
            "session": session,
            "model_calls": calls,
            "trajectory": trajectory,
            "export_style": "individual",
            "turn_discount": 1.0,
        }
        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError,
            "exactly one pre-merge",
        ):
            prepare_agent_service_training_record(**arguments)
        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError,
            "exactly one pre-merge",
        ):
            prepare_agent_service_training_record(
                **arguments,
                interaction_cache=cache,
                exported_interactions=cache.export_interactions(
                    style="individual",
                    reward_discount=1.0,
                ),
            )

    def test_crossed_session_parent_and_count_fail_closed(self) -> None:
        trace = self._trace()
        session, calls, trajectory = self._receipts(trace)
        crossed_trajectory = type(trajectory)(
            session_id="another-session",
            trajectory_id=trajectory.trajectory_id,
            interaction_count=trajectory.interaction_count,
            ready_transition=trajectory.ready_transition,
        )
        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError, "session differs"
        ):
            build_agent_service_interaction_sidecar(
                trace=trace,
                session=session,
                model_calls=calls,
                trajectory=crossed_trajectory,
            )

        bad_parent = [
            type(calls[0])(
                model_call_id=calls[0].model_call_id,
                interaction_id=calls[0].interaction_id,
                ordinal=0,
                parent_model_call_id=calls[1].model_call_id,
            ),
            calls[1],
        ]
        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError, "parent model call must precede"
        ):
            build_agent_service_interaction_sidecar(
                trace=trace,
                session=session,
                model_calls=bad_parent,
                trajectory=trajectory,
            )

        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError, "interaction count differs"
        ):
            build_agent_service_interaction_sidecar(
                trace=trace,
                session=session,
                model_calls=calls[:1],
                trajectory=trajectory,
            )

    def test_reward_receipt_requires_ready_trajectory(self) -> None:
        with self.assertRaisesRegex(
            ArealAgentServiceAdapterError, "no ready trajectory ID"
        ):
            trajectory_receipt_from_set_reward_response(
                {
                    "interaction_count": 2,
                    "session_id": "session-1",
                    "trajectory_id": None,
                    "trajectory_ready": False,
                    "ready_transition": False,
                }
            )

    def test_private_writer_does_not_overwrite_or_escape(self) -> None:
        trace = self._trace()
        session, calls, trajectory = self._receipts(trace)
        record = prepare_agent_service_training_record(
            trace=trace,
            session=session,
            model_calls=calls,
            trajectory=trajectory,
            interaction_cache=self._cache("hf"),
            export_style="individual",
            turn_discount=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "agent-service" / "episode.json"
            path = write_agent_service_training_record(
                record,
                destination=destination,
                allowed_root=root,
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                json.loads(path.read_text())["record_sha256"],
                record["record_sha256"],
            )
            with self.assertRaises(FileExistsError):
                write_agent_service_training_record(
                    record,
                    destination=destination,
                    allowed_root=root,
                )
            with self.assertRaisesRegex(
                ArealAgentServiceAdapterError, "escapes configured root"
            ):
                write_agent_service_training_record(
                    record,
                    destination=root.parent / "outside.json",
                    allowed_root=root,
                )


if __name__ == "__main__":
    unittest.main()
