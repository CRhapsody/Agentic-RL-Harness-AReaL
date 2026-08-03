from dataclasses import replace
import hashlib
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
    deterministic_bridge_request_id,
    inference_runtime_contract_sha256,
    inject_harness_instruction,
    prompt_context_chars,
    validate_areal_joint_bridge_record,
    write_areal_joint_bridge_record,
)
from jphrl.trajectory.areal_interaction_sidecar import (
    InteractionBinding,
    build_interaction_adapter_sidecar,
)


class ArealJointBridgeTests(unittest.TestCase):
    def _runtime_contract(self) -> dict[str, object]:
        return {
            "schema_version": "jph.sglang-inference-runtime.v1",
            "identity": {"run_id": "test-run", "screen_pair_id": None},
            "fixed": {
                "areal_commit": "a" * 40,
                "areal_version": "2.0.0",
                "behavior_revision": "b" * 40,
                "clean_environment_policy": "filtered-inherited-v1",
                "cuda_runtime_version": "12.6",
                "cuda_visible_devices": "0",
                "dataset_revision": "c" * 40,
                "dataset_selection": "sequential-offset0-count4-v1",
                "driver_version": "test-driver",
                "generation": {"temperature": 1.0},
                "gpu_name": "test-gpu",
                "gpu_uuid": "GPU-test",
                "physical_gpu_id": 0,
                "python_version": "3.11.0",
                "project_commit": "d" * 40,
                "rollout": {
                    "backend": "sglang:d1p1t1",
                    "max_concurrent_rollouts": 1,
                },
                "seed": 1,
                "server_args": {
                    "base_gpu_id": 0,
                    "disable_cuda_graph": False,
                    "model_path": "/allowed/model",
                    "tokenizer_path": "/allowed/model",
                    "tp_size": 1,
                },
                "sglang_environment": {"SGLANG_CACHE_DIR": "/allowed/cache"},
                "sglang_version": "0.5.10.post1",
                "torch_version": "2.8.0",
                "transformers_version": "4.57.1",
            },
            "treatment": {
                "generation_logprob_mode": "standard-log-of-softmax-v1",
                "sglang_return_original_logprob": False,
            },
        }

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
        request_id = deterministic_bridge_request_id(
            task_id=7,
            dataset_selection="sequential-offset0-count4-v1",
            base_messages=base_messages,
        )
        decision = replace(decision, decision_id=f"{request_id}:harness:0")
        response = SimpleNamespace(
            input_tokens=[9, 1, 2],
            output_tokens=[3, 4],
            output_logprobs=[-0.1, -0.2],
            output_versions=[0, 0],
            stop_reason="stop",
        )
        interaction = SimpleNamespace(
            interaction_id=request_id,
            reward=1.0,
            chat_template_type=None,
        )
        tensor_dict = {
            "input_ids": [[9, 1, 2, 3, 4]],
            "loss_mask": [[0, 0, 0, 1, 1]],
            "logprobs": [[0.0, 0.0, 0.0, -0.1, -0.2]],
            "versions": [[-1, -1, -1, 0, 0]],
            "attention_mask": [[True, True, True, True, True]],
            "rewards": [1.0],
        }
        runtime_contract = self._runtime_contract()
        joint_version = build_joint_version(
            policy_release_id="areal-sglang@model-commit:engine-v0",
            harness_controller_version=controller.version,
            areal_commit="a" * 40,
            behavior_revision="b" * 40,
            dataset_revision="c" * 40,
            dataset_selection="sequential-offset0-count4-v1",
            sglang_version="0.5.10.post1",
            generation_logprob_mode="standard-log-of-softmax-v1",
            inference_runtime_contract_sha256=(
                inference_runtime_contract_sha256(runtime_contract)
            ),
        )
        episode_id = f"test-run:{request_id}"
        model_call_id = f"{episode_id}:model:0"
        interaction_sidecar = build_interaction_adapter_sidecar(
            [
                InteractionBinding(
                    episode_id=episode_id,
                    model_call_id=model_call_id,
                    session_id=None,
                    trajectory_id=None,
                    interaction_id=request_id,
                    parent_interaction_id=None,
                    ordinal=0,
                    joint_version_id=joint_version.version_id,
                    route_kind="rlvr-workflow",
                )
            ]
        )
        return build_areal_joint_bridge_record(
            task_id=7,
            request_id=request_id,
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
            dataset_selection="sequential-offset0-count4-v1",
            sglang_version="0.5.10.post1",
            generation_logprob_mode="standard-log-of-softmax-v1",
            inference_runtime_contract=runtime_contract,
            interaction_adapter_sidecar=interaction_sidecar,
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
            record["policy_binding"]["generation_logprob_mode"],
            "standard-log-of-softmax-v1",
        )
        self.assertIn(
            "selection=sequential-offset0-count4-v1",
            record["joint_version"]["environment"],
        )
        self.assertEqual(record["policy_binding"]["sglang_version"], "0.5.10.post1")
        self.assertEqual(
            record["policy_binding"]["inference_runtime_contract_sha256"],
            inference_runtime_contract_sha256(self._runtime_contract()),
        )
        self.assertEqual(
            record["harness"]["controller_checkpoint_before_decision"]["sample_count"],
            0,
        )
        binding = record["interaction_adapter_sidecar"]["bindings"][0]
        self.assertEqual(audit["model_call_id"], binding["model_call_id"])
        self.assertEqual(audit["interaction_id"], binding["interaction_id"])
        self.assertNotEqual(binding["model_call_id"], binding["interaction_id"])
        self.assertEqual(
            record["credit_binding"]["policy_target_model_call_id"],
            binding["model_call_id"],
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

    def test_bridge_rejects_rehashed_interaction_sidecar_mismatch(self) -> None:
        record = self._record()
        sidecar = record["interaction_adapter_sidecar"]
        sidecar["bindings"][0]["interaction_id"] = "crossed-interaction"
        sidecar_unsigned = {
            key: item for key, item in sidecar.items() if key != "sidecar_sha256"
        }
        sidecar["sidecar_sha256"] = hashlib.sha256(
            json.dumps(
                sidecar_unsigned,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        unsigned = {
            key: item for key, item in record.items() if key != "record_sha256"
        }
        record["record_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ArealJointBridgeError, "trace interaction"):
            validate_areal_joint_bridge_record(record, expected_policy_version=0)

    def test_bridge_rejects_unbound_logprob_mode_or_dataset_selection(self) -> None:
        for field, value, error in (
            ("generation_logprob_mode", "unknown-mode", "unknown generation"),
            ("dataset_selection", "other-selection", "fixed identity"),
        ):
            with self.subTest(field=field):
                record = self._record()
                record["policy_binding"][field] = value
                unsigned = {
                    key: item
                    for key, item in record.items()
                    if key != "record_sha256"
                }
                record["record_sha256"] = hashlib.sha256(
                    json.dumps(
                        unsigned,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                with self.assertRaisesRegex(ArealJointBridgeError, error):
                    validate_areal_joint_bridge_record(
                        record, expected_policy_version=0
                    )

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
