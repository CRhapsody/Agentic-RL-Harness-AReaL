from __future__ import annotations

from dataclasses import replace
import json
import os
from typing import Any

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.workflow.rlvr import RLVRWorkflow

from jphrl.harness.controller import HarnessState
from jphrl.harness.learning import TabularHarnessController
from jphrl.trajectory.areal_joint_bridge import (
    build_areal_joint_bridge_record,
    build_joint_version,
    deterministic_bridge_request_id,
    inject_harness_instruction,
    prompt_context_chars,
    write_areal_joint_bridge_record,
)
from jphrl.trajectory.areal_interaction_sidecar import (
    InteractionBinding,
    build_interaction_adapter_sidecar,
)


class ArealJointBridgeWorkflow(RLVRWorkflow):
    """Make one Harness decision affect a real AReaL request and save both streams."""

    def __init__(self, *args: Any, harness_seed: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.harness_controller = TabularHarnessController(seed=harness_seed)

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward]:
        if isinstance(self.reward_fn, str):
            from areal.api import AsyncRewardWrapper
            from areal.utils.dynamic_import import import_from_string

            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        base_messages = self.data_extract_prompt_fn(data)
        context_chars = prompt_context_chars(base_messages)
        state = HarnessState(
            turn=0,
            remaining_tool_calls=0,
            remaining_model_retries=0,
            context_chars=context_chars,
            last_error=None,
            retrieval_hit=False,
            verifier_status="not-run",
            task_domain="gsm8k",
        )
        task_id = workflow_context.get().task_id
        request_id = deterministic_bridge_request_id(
            task_id=task_id,
            dataset_selection=os.environ["JPH_DATASET_SELECTION"],
            base_messages=base_messages,
        )
        controller_checkpoint = self.harness_controller.checkpoint()
        decision = replace(
            self.harness_controller.choose(state),
            decision_id=f"{request_id}:harness:0",
        )
        effective_messages, _ = inject_harness_instruction(
            base_messages, decision.action
        )
        base_input_ids = self.get_input_ids_fn(
            base_messages,
            self.tokenizer,
            self.enable_thinking,
        )
        effective_input_ids = self.get_input_ids_fn(
            effective_messages,
            self.tokenizer,
            self.enable_thinking,
        )
        request = ModelRequest(
            rid=request_id,
            input_ids=effective_input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        prompt_text = self.tokenizer.decode(effective_input_ids)
        response, reward = await self._collect_samples(
            engine, request, prompt_text, data
        )
        interaction = InteractionWithTokenLogpReward(
            model_response=response,
            reward=float(reward),
        )
        interaction.interaction_id = request_id
        tensor_dict = interaction.to_tensor_dict()

        expected_policy_version = int(os.environ["JPH_EXPECTED_POLICY_VERSION"])
        behavior_revision = os.environ["JPH_BEHAVIOR_REVISION"]
        generation_logprob_mode = os.environ["JPH_SGLANG_LOGPROB_MODE"]
        inference_runtime_contract = json.loads(
            os.environ["JPH_INFERENCE_RUNTIME_CONTRACT"]
        )
        inference_runtime_contract_sha256 = os.environ[
            "JPH_INFERENCE_RUNTIME_CONTRACT_SHA256"
        ]
        joint_version = build_joint_version(
            policy_release_id=(
                f"areal-sglang@{behavior_revision}:engine-v{expected_policy_version}"
            ),
            harness_controller_version=self.harness_controller.version,
            areal_commit=os.environ["JPH_AREAL_COMMIT"],
            behavior_revision=behavior_revision,
            dataset_revision=os.environ["JPH_DATASET_REVISION"],
            dataset_selection=os.environ["JPH_DATASET_SELECTION"],
            sglang_version=os.environ["JPH_SGLANG_VERSION"],
            generation_logprob_mode=generation_logprob_mode,
            inference_runtime_contract_sha256=(
                inference_runtime_contract_sha256
            ),
        )
        run_id = str(inference_runtime_contract["identity"]["run_id"])
        episode_id = f"{run_id}:{request_id}"
        model_call_id = f"{episode_id}:model:0"
        interaction_sidecar = build_interaction_adapter_sidecar(
            [
                InteractionBinding(
                    episode_id=episode_id,
                    model_call_id=model_call_id,
                    session_id=None,
                    trajectory_id=None,
                    interaction_id=interaction.interaction_id,
                    parent_interaction_id=None,
                    ordinal=0,
                    joint_version_id=joint_version.version_id,
                    route_kind="rlvr-workflow",
                )
            ]
        )
        record = build_areal_joint_bridge_record(
            task_id=task_id,
            request_id=request_id,
            joint_version=joint_version,
            expected_policy_version=expected_policy_version,
            harness_state=state,
            harness_decision=decision,
            harness_controller_checkpoint=controller_checkpoint,
            base_messages=base_messages,
            effective_messages=effective_messages,
            base_input_tokens=base_input_ids,
            effective_input_tokens=effective_input_ids,
            model_response=response,
            interaction=interaction,
            tensor_dict=tensor_dict,
            project_commit=os.environ["JPH_PROJECT_COMMIT"],
            areal_commit=os.environ["JPH_AREAL_COMMIT"],
            behavior_snapshot_path=os.environ["JPH_BEHAVIOR_SNAPSHOT"],
            behavior_revision=behavior_revision,
            dataset_selection=os.environ["JPH_DATASET_SELECTION"],
            sglang_version=os.environ["JPH_SGLANG_VERSION"],
            generation_logprob_mode=generation_logprob_mode,
            inference_runtime_contract=inference_runtime_contract,
            interaction_adapter_sidecar=interaction_sidecar,
        )
        write_areal_joint_bridge_record(
            record,
            trace_dir=os.environ["JPH_AREAL_JOINT_BRIDGE_DIR"],
            allowed_root=os.environ["JPH_ROOT"],
        )
        return {request_id: interaction}
