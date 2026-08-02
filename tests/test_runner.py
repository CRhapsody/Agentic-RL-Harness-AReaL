from pathlib import Path
import tempfile
import unittest

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import FixedHarnessController, SmokeHarnessController
from jphrl.harness.spec import HarnessAction, HarnessSpec
from jphrl.models.base import MockStructuredModel
from jphrl.models.base import ModelResponse
from jphrl.runner import run_calculator_smoke


class FailingModel:
    policy_version = "failing-policy-v1"
    tokenizer_version = "failing-tokenizer-v1"

    def generate(self, messages, max_new_tokens):
        del messages, max_new_tokens
        raise RuntimeError("simulated model service failure")


class FailingController:
    version = "failing-controller-v1"

    def choose(self, state):
        del state
        raise RuntimeError("simulated controller service failure")


class ScriptedModel:
    policy_version = "scripted-policy-v1"
    tokenizer_version = "scripted-tokenizer-v1"

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, messages, max_new_tokens):
        del messages, max_new_tokens
        text = next(self.responses)
        return ModelResponse(
            text=text,
            input_token_ids=[],
            output_token_ids=[],
            output_token_logprobs=[],
            output_versions=[],
            completion_loss_mask=[],
            policy_version=self.policy_version,
            tokenizer_version=self.tokenizer_version,
            policy_kind="scripted",
            token_metadata_status="not_applicable",
        )


class RunnerTests(unittest.TestCase):
    def test_mock_episode_passes_with_exact_trace(self) -> None:
        result = run_calculator_smoke(
            model=MockStructuredModel(),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.tool_result, "42")
        self.assertEqual(result.final_answer, "42")
        self.assertEqual(
            {event.joint_version_id for event in result.trace.events},
            {result.trace.joint_version.version_id},
        )
        kinds = [event.kind for event in result.trace.events]
        self.assertEqual(
            kinds,
            [
                "episode_started",
                "harness_decision",
                "model_request",
                "model_response",
                "parse_result",
                "tool_result",
                "harness_decision",
                "verifier_result",
                "model_request",
                "model_response",
                "parse_result",
                "reward_assigned",
                "episode_ended",
            ],
        )
        responses = [event for event in result.trace.events if event.kind == "model_response"]
        self.assertTrue(all(event.payload["policy_kind"] == "scripted" for event in responses))
        self.assertTrue(
            all(event.payload["token_metadata_status"] == "not_applicable" for event in responses)
        )
        self.assertTrue(all(event.payload["output_token_ids"] == [] for event in responses))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            result.trace.write_json(path)
            self.assertTrue(path.is_file())

    def test_non_direct_initial_action_fails_without_tool_use(self) -> None:
        result = run_calculator_smoke(
            model=MockStructuredModel(),
            task=TASKS["add-17-25"],
            controller=FixedHarnessController(HarnessAction.VERIFY),
            harness_spec=HarnessSpec(),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.trace.failure_category, "harness")
        self.assertNotIn("tool_result", [event.kind for event in result.trace.events])
        self.assertEqual(result.trace.validity_class, "policy_failure")

    def test_model_service_failure_is_invalid_not_zero_reward(self) -> None:
        result = run_calculator_smoke(
            model=FailingModel(),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertFalse(result.success)
        self.assertFalse(result.trace.valid)
        self.assertIsNone(result.reward)
        self.assertEqual(result.trace.failure_category, "infra")
        self.assertEqual(result.trace.validity_class, "infrastructure_invalid")
        self.assertIn("model_error", [event.kind for event in result.trace.events])

    def test_wrong_final_answer_is_evaluator_failure(self) -> None:
        result = run_calculator_smoke(
            model=ScriptedModel(
                [
                    '{"tool":"calculator","expression":"17 + 25"}',
                    '{"answer":"41"}',
                ]
            ),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertFalse(result.success)
        self.assertTrue(result.trace.valid)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.trace.failure_category, "evaluator")

    def test_forbidden_tool_expression_is_policy_failure(self) -> None:
        result = run_calculator_smoke(
            model=ScriptedModel(
                ['{"tool":"calculator","expression":"__import__(\'os\').system(\'id\')"}']
            ),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertFalse(result.success)
        self.assertTrue(result.trace.valid)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.trace.failure_category, "tool")

    def test_available_token_metadata_fails_closed_when_empty(self) -> None:
        class BrokenMetadataModel(ScriptedModel):
            def generate(self, messages, max_new_tokens):
                response = super().generate(messages, max_new_tokens)
                return ModelResponse(
                    text=response.text,
                    input_token_ids=[1],
                    output_token_ids=[],
                    output_token_logprobs=[],
                    output_versions=[],
                    completion_loss_mask=[],
                    policy_version=self.policy_version,
                    tokenizer_version=self.tokenizer_version,
                    policy_kind="causal_lm",
                    token_metadata_status="available",
                )

        result = run_calculator_smoke(
            model=BrokenMetadataModel(['{"tool":"calculator","expression":"17 + 25"}']),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        self.assertFalse(result.success)
        self.assertFalse(result.trace.valid)
        self.assertIsNone(result.reward)
        self.assertEqual(result.trace.validity_class, "trace_contract_invalid")
        self.assertIn("model_contract_error", [event.kind for event in result.trace.events])

    def test_controller_failure_is_invalid_and_trace_is_closed(self) -> None:
        result = run_calculator_smoke(
            model=MockStructuredModel(),
            task=TASKS["add-17-25"],
            controller=FailingController(),
            harness_spec=HarnessSpec(),
        )
        self.assertFalse(result.success)
        self.assertFalse(result.trace.valid)
        self.assertIsNone(result.reward)
        self.assertEqual(result.trace.validity_class, "infrastructure_invalid")
        self.assertEqual(result.trace.failure_category, "controller_infra")
        self.assertEqual(result.trace.events[-2].kind, "reward_assigned")
        self.assertEqual(result.trace.events[-1].kind, "episode_ended")

    def test_strict_parser_rejects_markdown_or_extra_text(self) -> None:
        result = run_calculator_smoke(
            model=ScriptedModel(
                ['说明：```json\n{"tool":"calculator","expression":"17 + 25"}\n```']
            ),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(max_model_retries=0),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.trace.validity_class, "policy_failure")
        self.assertEqual(result.trace.failure_category, "parser")

    def test_retry_cannot_silently_exceed_frozen_model_call_budget(self) -> None:
        result = run_calculator_smoke(
            model=ScriptedModel(
                [
                    '{"answer":"42"}',
                    '{"tool":"calculator","expression":"17 + 25"}',
                    '{"answer":"42"}',
                ]
            ),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(max_model_retries=1),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.trace.validity_class, "policy_failure")
        self.assertEqual(result.trace.failure_category, "budget")
        self.assertEqual(
            sum(event.kind == "model_request" for event in result.trace.events),
            3,
        )


if __name__ == "__main__":
    unittest.main()
