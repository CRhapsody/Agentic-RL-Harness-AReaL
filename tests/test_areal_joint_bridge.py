from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from jphrl.harness.controller import HarnessState
from jphrl.harness.learning import TabularHarnessController
from jphrl.harness.spec import HarnessAction
from jphrl.trajectory.areal_joint_bridge import (
    ArealJointBridgeError,
    build_areal_joint_bridge_record,
    build_joint_version,
    inject_harness_instruction,
    prompt_context_chars,
    validate_areal_joint_bridge_record,
    write_areal_joint_bridge_record,
)


class ArealJointBridgeTests(unittest.TestCase):
    def _record(self) -> dict[str, object]:
        state = HarnessState(
            turn=0,
            remaining_tool_calls=0,
            remaining_model_retries=0,
            context_chars=prompt_context_chars(
                [{"role": "user", "content": "What is 20 + 22?"}]
            ),
            last_error=None,
            retrieval_hit=False,
            verifier_status="not-run",
            task_domain="gsm8k",
        )
        controller = TabularHarnessController(seed=3)
        controller_checkpoint = controller.checkpoint()
        decision = replace(
            controller.choose(state),
            decision_id="request-1:harness:0",
        )
        base_messages = [{"role": "user", "content": "What is 20 + 22?"}]
        effective_messages, _ = inject_harness_instruction(
            base_messages, decision.action
        )
        response = SimpleNamespace(
            input_tokens=[9, 1, 2],
            output_tokens=[3, 4],
            output_logprobs=[-0.1, -0.2],
            output_versions=[0, 0],
            stop_reason="stop",
        )
        interaction = SimpleNamespace(reward=1.0, chat_template_type=None)
        tensor_dict = {
            "input_ids": [[9, 1, 2, 3, 4]],
            "loss_mask": [[0, 0, 0, 1, 1]],
            "logprobs": [[0.0, 0.0, 0.0, -0.1, -0.2]],
            "versions": [[-1, -1, -1, 0, 0]],
            "attention_mask": [[True, True, True, True, True]],
            "rewards": [1.0],
        }
        joint_version = build_joint_version(
            policy_release_id="areal-sglang@model-commit:engine-v0",
            harness_controller_version=controller.version,
            areal_commit="a" * 40,
            behavior_revision="b" * 40,
            dataset_revision="c" * 40,
        )
        return build_areal_joint_bridge_record(
            task_id=7,
            request_id="request-1",
            joint_version=joint_version,
            expected_policy_version=0,
            harness_state=state,
            harness_decision=decision,
            harness_controller_checkpoint=controller_checkpoint,
            base_messages=base_messages,
            effective_messages=effective_messages,
            base_input_tokens=[1, 2],
            effective_input_tokens=[9, 1, 2],
            model_response=response,
            interaction=interaction,
            tensor_dict=tensor_dict,
            project_commit="d" * 40,
            areal_commit="a" * 40,
            behavior_snapshot_path="/allowed/model",
            behavior_revision="b" * 40,
        )

    def test_real_interaction_and_harness_prompt_are_bound(self) -> None:
        record = self._record()
        audit = validate_areal_joint_bridge_record(
            record, expected_policy_version=0
        )
        self.assertTrue(audit["ok"])
        self.assertTrue(audit["prompt_tokens_changed"])
        self.assertEqual(audit["harness_loss_mask"], 1)
        self.assertEqual(audit["inference_engine_versions"], [0])
        self.assertEqual(
            record["areal_trace"]["model_response"]["input_tokens"],
            record["prompt_binding"]["effective_input_tokens"],
        )
        self.assertIsNone(record["credit_binding"]["policy_advantage"])
        self.assertIsNone(record["credit_binding"]["harness_advantage"])
        self.assertEqual(
            record["harness"]["controller_checkpoint_before_decision"]["sample_count"],
            0,
        )

    def test_every_harness_action_changes_the_prompt(self) -> None:
        base = [{"role": "user", "content": "Compute 1 + 1."}]
        for action in HarnessAction:
            with self.subTest(action=action.value):
                effective, instruction = inject_harness_instruction(base, action)
                self.assertNotEqual(base, effective)
                self.assertEqual(effective[0]["role"], "system")
                self.assertIn(action.value, effective[0]["content"])
                self.assertIn(instruction, effective[0]["content"])

    def test_bridge_record_fails_closed_after_mutation(self) -> None:
        record = self._record()
        record["prompt_binding"]["effective_input_tokens"][0] = 999
        with self.assertRaisesRegex(ArealJointBridgeError, "hash"):
            validate_areal_joint_bridge_record(record, expected_policy_version=0)

    def test_bridge_writer_is_unique_private_and_root_bounded(self) -> None:
        record = self._record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_areal_joint_bridge_record(
                record,
                trace_dir=root / "bridge",
                allowed_root=root,
            )
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["record_sha256"],
                record["record_sha256"],
            )
            with self.assertRaises(FileExistsError):
                write_areal_joint_bridge_record(
                    record,
                    trace_dir=root / "bridge",
                    allowed_root=root,
                )
            with self.assertRaisesRegex(ArealJointBridgeError, "escapes"):
                write_areal_joint_bridge_record(
                    record,
                    trace_dir=root.parent / "outside",
                    allowed_root=root,
                )


if __name__ == "__main__":
    unittest.main()
