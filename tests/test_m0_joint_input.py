from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from jphrl.experiments.m0_joint_input import (
    M0JointInputError,
    prepare_m0_joint_training_input,
)
from jphrl.harness.controller import HarnessState
from jphrl.trajectory.areal_interaction_sidecar import (
    InteractionBinding,
    build_interaction_adapter_sidecar,
)
from jphrl.trajectory.areal_joint_bridge import (
    build_areal_joint_bridge_record,
    build_joint_version,
    deterministic_bridge_request_id,
    inference_runtime_contract_sha256,
    inject_harness_instruction,
    prompt_context_chars,
)
from jphrl.trajectory.joint_credit_alignment import (
    ESTIMATOR_VERSION,
    DualCreditEstimatorSpec,
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.schema import JointVersion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AREAL_ROOT = PROJECT_ROOT.parent / "AReaL"


def _load_pinned_areal():
    areal_root = Path(
        os.environ.get("JPH_AREAL_SOURCE", str(DEFAULT_AREAL_ROOT))
    ).resolve()
    if not (areal_root / "areal").is_dir():
        raise unittest.SkipTest(f"pinned AReaL source is unavailable: {areal_root}")
    if str(areal_root) not in sys.path:
        sys.path.insert(0, str(areal_root))
    try:
        import torch
        from areal.api import ModelResponse
        from areal.experimental.openai.types import InteractionWithTokenLogpReward
        from areal.utils.data import concat_padded_tensors
        from areal.v2.inference_service.data_proxy.session import SessionData

        from jphrl.harness.torch_learning import (
            TorchHarnessPolicy,
            build_torch_harness_rollout_checkpoint,
        )
    except ImportError as exc:
        raise unittest.SkipTest(
            f"pinned AReaL/Torch CPU dependencies are unavailable: {exc}"
        ) from exc
    return {
        "torch": torch,
        "ModelResponse": ModelResponse,
        "Interaction": InteractionWithTokenLogpReward,
        "concat_padded_tensors": concat_padded_tensors,
        "SessionData": SessionData,
        "TorchHarnessPolicy": TorchHarnessPolicy,
        "build_rollout_checkpoint": build_torch_harness_rollout_checkpoint,
    }


def _canonical_sha(record: dict[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_contract(*, style: str) -> dict[str, object]:
    return {
        "schema_version": "jph.sglang-inference-runtime.v1",
        "identity": {"run_id": f"m0-{style}", "screen_pair_id": None},
        "fixed": {
            "areal_commit": "a" * 40,
            "areal_version": "2.0.0",
            "behavior_revision": "b" * 40,
            "clean_environment_policy": "filtered-inherited-v1",
            "cuda_runtime_version": "12.6",
            "cuda_visible_devices": "0",
            "dataset_revision": "c" * 40,
            "dataset_selection": f"aa-m0-{style}-one-real-session-v1",
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
            "seed": 19,
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


class M0Source:
    def __init__(
        self,
        *,
        bridge,
        exported,
        start_response,
        reward_response,
        estimator,
    ) -> None:
        self.bridge = bridge
        self.exported = exported
        self.start_response = start_response
        self.reward_response = reward_response
        self.estimator = estimator


def _source(
    *,
    style: str,
    reward: float = 0.0,
    route_kind: str = "agent-service-session",
) -> M0Source:
    areal = _load_pinned_areal()
    base_messages = [{"role": "user", "content": "What is 20 + 22?"}]
    state = HarnessState(
        turn=0,
        remaining_tool_calls=0,
        remaining_model_retries=0,
        context_chars=prompt_context_chars(base_messages),
        last_error=None,
        retrieval_hit=False,
        verifier_status="not-run",
        task_domain="gsm8k",
    )
    controller = areal["TorchHarnessPolicy"](seed=19, hidden_size=16)
    controller_checkpoint = areal["build_rollout_checkpoint"](controller)
    decision = controller.choose(state)
    runtime_contract = _runtime_contract(style=style)
    dataset_selection = runtime_contract["fixed"]["dataset_selection"]
    request_id = deterministic_bridge_request_id(
        task_id=7,
        dataset_selection=dataset_selection,
        base_messages=base_messages,
    )
    decision = replace(decision, decision_id=f"{request_id}:harness:0")
    effective_messages, _ = inject_harness_instruction(base_messages, decision.action)
    joint_version = build_joint_version(
        policy_release_id="areal-sglang@model-commit:engine-v0",
        harness_controller_version=controller.version,
        areal_commit="a" * 40,
        behavior_revision="b" * 40,
        dataset_revision="c" * 40,
        dataset_selection=dataset_selection,
        sglang_version="0.5.10.post1",
        generation_logprob_mode="standard-log-of-softmax-v1",
        inference_runtime_contract_sha256=(
            inference_runtime_contract_sha256(runtime_contract)
        ),
    )
    session_id = f"m0-session-{style}"
    session = areal["SessionData"](session_id=session_id)
    interaction = areal["Interaction"](
        messages=effective_messages,
        output_message_list=[{"role": "assistant", "content": "The answer is 41."}],
        model_response=areal["ModelResponse"](
            input_tokens=[9, 1, 2],
            output_tokens=[3, 4],
            output_logprobs=[-0.1, -0.2],
            output_versions=[0, 0],
            stop_reason="stop",
        ),
        chat_template_type="concat" if style == "concat" else "hf",
    )
    interaction._interaction_id = request_id
    session.active_completions[interaction.interaction_id] = interaction
    reward_result = session.set_reward(interaction_id=None, reward=reward)
    if reward_result.trajectory_id is None:
        raise AssertionError("test SessionData did not finalize its trajectory")
    trajectory_id, exported = session.export_trajectory(
        discount=1.0,
        style=style,
        trajectory_id=reward_result.trajectory_id,
    )
    model_call_id = f"m0-{style}-episode:model:0"
    route_session_id = session_id if route_kind == "agent-service-session" else None
    route_trajectory_id = trajectory_id if route_kind == "agent-service-session" else None
    sidecar = build_interaction_adapter_sidecar(
        [
            InteractionBinding(
                episode_id=f"m0-{style}-episode",
                model_call_id=model_call_id,
                session_id=route_session_id,
                trajectory_id=route_trajectory_id,
                interaction_id=request_id,
                parent_interaction_id=None,
                ordinal=0,
                joint_version_id=joint_version.version_id,
                route_kind=route_kind,
            )
        ]
    )
    bridge = build_areal_joint_bridge_record(
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
        model_response=interaction.model_response,
        interaction=interaction,
        tensor_dict=interaction.to_tensor_dict(),
        project_commit="d" * 40,
        areal_commit="a" * 40,
        behavior_snapshot_path="/allowed/model",
        behavior_revision="b" * 40,
        dataset_selection=dataset_selection,
        sglang_version="0.5.10.post1",
        generation_logprob_mode="standard-log-of-softmax-v1",
        inference_runtime_contract=runtime_contract,
        interaction_adapter_sidecar=sidecar,
    )
    estimator = DualCreditEstimatorSpec(
        estimator_version=ESTIMATOR_VERSION,
        parent_joint_version_id=joint_version.version_id,
        policy_source="aa-m0-policy-frozen-terminal-baseline-v1",
        harness_source="aa-m0-harness-frozen-terminal-baseline-v1",
        policy_baseline_snapshot_id="aa-m0-policy-baseline-run-19",
        harness_baseline_snapshot_id="aa-m0-harness-baseline-run-19",
        policy_baselines={model_call_id: 0.25},
        harness_baselines={decision.decision_id: 0.1},
    )
    return M0Source(
        bridge=bridge,
        exported=exported,
        start_response={
            "group_id": "m0-group-19",
            "sessions": [
                {
                    "session_id": session_id,
                    "session_api_key": "test-routing-secret-never-persist",
                }
            ],
        },
        reward_response={
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "interaction_count": 1,
            "trajectory_ready": True,
            "ready_transition": reward_result.ready_transition,
        },
        estimator=estimator,
    )


class M0JointInputTests(unittest.TestCase):
    def _prepare(self, source: M0Source, *, style: str):
        return prepare_m0_joint_training_input(
            source.bridge,
            start_session_response=source.start_response,
            set_reward_response=source.reward_response,
            pre_batch_exported_interactions=source.exported,
            estimator=source.estimator,
            export_style=style,
            turn_discount=1.0,
        )

    def test_real_sessiondata_individual_and_concat_reach_p_q_r_s(self) -> None:
        for style in ("individual", "concat"):
            with self.subTest(style=style):
                source = _source(style=style)
                result = self._prepare(source, style=style)
                s_audit = validate_frozen_joint_credit_alignment(
                    result.s_joint_credit,
                    active_joint_version=JointVersion(
                        **source.bridge["joint_version"]
                    ),
                )

                self.assertEqual(result.trace.validity_class, "policy_failure")
                self.assertEqual(result.trace.reward, 0.0)
                self.assertEqual(result.session_receipt.session_id, f"m0-session-{style}")
                self.assertEqual(
                    result.model_call_receipt.interaction_id,
                    next(iter(source.exported)),
                )
                self.assertEqual(result.model_call_receipt.ordinal, 0)
                self.assertIsNone(result.model_call_receipt.parent_model_call_id)
                self.assertEqual(result.trajectory_receipt.interaction_count, 1)
                self.assertIs(
                    result.exported_interaction.source,
                    next(iter(source.exported.values())),
                )
                self.assertEqual(
                    result.p_training_record["training_archive"]["export_style"],
                    style,
                )
                self.assertEqual(s_audit["policy_sample_count"], 1)
                self.assertEqual(s_audit["harness_action_count"], 1)
                self.assertEqual(
                    result.s_joint_credit["policy_samples"][0]["decision_credits"][0][
                        "advantage"
                    ],
                    -0.25,
                )
                self.assertEqual(
                    result.s_joint_credit["harness_samples"][0]["advantage"],
                    -0.1,
                )
                serialized = json.dumps(
                    {
                        "p": result.p_training_record,
                        "q": result.q_policy_admission,
                        "r": result.r_harness_admission,
                        "s": result.s_joint_credit,
                    },
                    sort_keys=True,
                )
                self.assertNotIn("session_api_key", serialized)
                self.assertNotIn("test-routing-secret-never-persist", serialized)
                for record in (
                    result.p_training_record,
                    result.q_policy_admission,
                    result.r_harness_admission,
                    result.s_joint_credit,
                ):
                    self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])
                    self.assertFalse(record["evidence_scope"]["harness_optimizer_update"])

    def test_rlvr_bridge_cannot_mint_agent_service_receipts(self) -> None:
        source = _source(style="individual", route_kind="rlvr-workflow")
        with self.assertRaisesRegex(M0JointInputError, "RLVR IDs cannot mint receipts"):
            self._prepare(source, style="individual")

    def test_positive_reward_bridge_is_outside_the_m0_policy_failure_contract(self) -> None:
        source = _source(style="individual", reward=1.0)
        with self.assertRaisesRegex(M0JointInputError, "only a real zero-reward"):
            self._prepare(source, style="individual")

    def test_post_batch_tensor_mapping_is_rejected(self) -> None:
        source = _source(style="individual")
        areal = _load_pinned_areal()
        padded = areal["concat_padded_tensors"](
            [item.to_tensor_dict() for item in source.exported.values()]
        )
        with self.assertRaisesRegex(M0JointInputError, "post-batch"):
            prepare_m0_joint_training_input(
                source.bridge,
                start_session_response=source.start_response,
                set_reward_response=source.reward_response,
                pre_batch_exported_interactions=padded,
                estimator=source.estimator,
                export_style="individual",
                turn_discount=1.0,
            )

    def test_version_session_and_trajectory_crosses_fail_closed(self) -> None:
        source = _source(style="individual")

        wrong_version = deepcopy(source.bridge)
        wrong_version["policy_binding"]["expected_inference_engine_version"] = 1
        wrong_version["record_sha256"] = _canonical_sha(wrong_version)
        with self.assertRaisesRegex(M0JointInputError, "expected policy version"):
            prepare_m0_joint_training_input(
                wrong_version,
                start_session_response=source.start_response,
                set_reward_response=source.reward_response,
                pre_batch_exported_interactions=source.exported,
                estimator=source.estimator,
            )

        wrong_session = deepcopy(source.start_response)
        wrong_session["sessions"][0]["session_id"] = "crossed-session"
        with self.assertRaisesRegex(M0JointInputError, "start-session receipt differs"):
            prepare_m0_joint_training_input(
                source.bridge,
                start_session_response=wrong_session,
                set_reward_response=source.reward_response,
                pre_batch_exported_interactions=source.exported,
                estimator=source.estimator,
            )

        wrong_trajectory = deepcopy(source.reward_response)
        wrong_trajectory["trajectory_id"] += 1
        with self.assertRaisesRegex(M0JointInputError, "trajectory receipt differs"):
            prepare_m0_joint_training_input(
                source.bridge,
                start_session_response=source.start_response,
                set_reward_response=wrong_trajectory,
                pre_batch_exported_interactions=source.exported,
                estimator=source.estimator,
            )

    def test_trace_contract_tensor_and_reward_tampering_cannot_be_reclassified(self) -> None:
        source = _source(style="individual")
        invalid = deepcopy(source.bridge)
        invalid["areal_trace"]["tensor_dict"]["loss_mask"][0][-1] = 0
        invalid["areal_trace"]["record_sha256"] = _canonical_sha(
            invalid["areal_trace"]
        )
        invalid["record_sha256"] = _canonical_sha(invalid)
        with self.assertRaisesRegex(M0JointInputError, "loss_mask"):
            prepare_m0_joint_training_input(
                invalid,
                start_session_response=source.start_response,
                set_reward_response=source.reward_response,
                pre_batch_exported_interactions=source.exported,
                estimator=source.estimator,
            )

        crossed_reward = deepcopy(source.bridge)
        crossed_reward["credit_binding"]["raw_terminal_reward"] = 1.0
        crossed_reward["record_sha256"] = _canonical_sha(crossed_reward)
        with self.assertRaisesRegex(M0JointInputError, "terminal reward differs"):
            prepare_m0_joint_training_input(
                crossed_reward,
                start_session_response=source.start_response,
                set_reward_response=source.reward_response,
                pre_batch_exported_interactions=source.exported,
                estimator=source.estimator,
            )


if __name__ == "__main__":
    unittest.main()
