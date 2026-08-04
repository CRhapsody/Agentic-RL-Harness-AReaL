from __future__ import annotations

"""AReaL V2 agent that stages one real RLVR interaction for pre-batch binding."""

import asyncio
import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from jphrl.harness.controller import HarnessState
from jphrl.trajectory.areal_joint_bridge import (
    build_joint_version,
    deterministic_bridge_request_id,
    inference_runtime_contract_sha256,
    inject_harness_instruction,
    prompt_context_chars,
)
from jphrl.trajectory.rlvr_online_binding import stage_rlvr_v2_agent_response


async def _evaluate_strict_reward(
    reward_fn: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    **kwargs: Any,
) -> float:
    try:
        reward_value = await asyncio.wait_for(
            asyncio.to_thread(reward_fn, *args, **kwargs),
            timeout=timeout_seconds,
        )
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise RuntimeError("strict GSM8K reward infrastructure timed out") from exc
    reward = float(reward_value)
    if not math.isfinite(reward):
        raise ValueError("strict GSM8K reward is not finite")
    return reward


class ArealJointBridgeAgent:
    """Use the V2 Gateway, while leaving tensor finalization to DataProxy."""

    def __init__(
        self,
        *,
        reward_fn: Callable[..., Any] | str,
        gconfig: Any,
        tokenizer: str | Any,
        enable_thinking: bool = False,
        harness_seed: int = 0,
        harness_kind: str = "torch",
        harness_hidden_size: int = 32,
        reward_timeout_seconds: float = 15.0,
    ) -> None:
        from areal.utils.dynamic_import import import_from_string
        from areal.utils.hf_utils import load_hf_tokenizer

        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        reward_path = reward_fn if isinstance(reward_fn, str) else (
            f"{getattr(reward_fn, '__module__', '')}."
            f"{getattr(reward_fn, '__name__', '')}"
        )
        if reward_path != "jphrl.rewards.strict_gsm8k_reward_fn":
            raise ValueError("formal V2 RLVR agent requires strict GSM8K reward")
        if isinstance(reward_fn, str):
            reward_fn = import_from_string(reward_fn)
        if not callable(reward_fn):
            raise TypeError("formal V2 RLVR reward is not callable")
        if not math.isfinite(float(reward_timeout_seconds)) or reward_timeout_seconds <= 0:
            raise ValueError("formal V2 RLVR reward timeout must be finite and positive")
        if harness_kind != "torch":
            raise ValueError("formal V2 RLVR agent requires the Torch Harness")
        from jphrl.harness.torch_learning import TorchHarnessPolicy

        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.reward_timeout_seconds = float(reward_timeout_seconds)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.enable_thinking = bool(enable_thinking)
        self.harness_controller = TorchHarnessPolicy(
            seed=harness_seed,
            hidden_size=harness_hidden_size,
        )

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        from openai import AsyncOpenAI

        from areal import workflow_context
        from areal.workflow.rlvr import default_get_input_ids_fn
        from jphrl.harness.torch_learning import (
            build_torch_harness_rollout_checkpoint,
        )

        base_url = extra_kwargs.get("base_url")
        http_client = extra_kwargs.get("http_client")
        api_key = extra_kwargs.get("api_key")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("AReaL V2 agent base URL is missing")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("AReaL V2 agent session route is missing")
        base_messages = data.get("messages")
        if not isinstance(base_messages, list) or not base_messages:
            raise ValueError("RLVR task has no messages")
        base_messages = [dict(message) for message in base_messages]
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
        task_id = int(workflow_context.get().task_id)
        dataset_selection = os.environ["JPH_DATASET_SELECTION"]
        request_id = deterministic_bridge_request_id(
            task_id=task_id,
            dataset_selection=dataset_selection,
            base_messages=base_messages,
        )
        checkpoint = build_torch_harness_rollout_checkpoint(
            self.harness_controller
        )
        decision = replace(
            self.harness_controller.choose(state),
            decision_id=f"{request_id}:harness:0",
        )
        effective_messages, _ = inject_harness_instruction(
            base_messages, decision.action
        )
        base_input_ids = default_get_input_ids_fn(
            base_messages,
            self.tokenizer,
            self.enable_thinking,
        )
        effective_input_ids = default_get_input_ids_fn(
            effective_messages,
            self.tokenizer,
            self.enable_thinking,
        )
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        response = await client.chat.completions.create(
            messages=effective_messages,
            model="default",
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            },
            **self.gconfig.to_openai_args_dict(),
        )
        interaction_id = response.id
        content = response.choices[0].message.content
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ValueError("AReaL V2 response has no public interaction ID")
        if not isinstance(content, str):
            raise ValueError("AReaL V2 response has no text completion")
        prompt_text = self.tokenizer.decode(effective_input_ids)
        completion_ids = self.tokenizer.encode(content, add_special_tokens=False)
        reward = await _evaluate_strict_reward(
            self.reward_fn,
            prompt_text,
            content,
            effective_input_ids,
            completion_ids,
            **data,
            timeout_seconds=self.reward_timeout_seconds,
        )

        runtime_contract = json.loads(os.environ["JPH_INFERENCE_RUNTIME_CONTRACT"])
        runtime_hash = inference_runtime_contract_sha256(runtime_contract)
        if runtime_hash != os.environ["JPH_INFERENCE_RUNTIME_CONTRACT_SHA256"]:
            raise ValueError("inference runtime contract changed inside V2 agent")
        expected_policy_version = int(os.environ["JPH_EXPECTED_POLICY_VERSION"])
        behavior_revision = os.environ["JPH_BEHAVIOR_REVISION"]
        generation_logprob_mode = os.environ["JPH_SGLANG_LOGPROB_MODE"]
        joint_version = build_joint_version(
            policy_release_id=(
                f"areal-sglang@{behavior_revision}:engine-v{expected_policy_version}"
            ),
            harness_controller_version=self.harness_controller.version,
            areal_commit=os.environ["JPH_AREAL_COMMIT"],
            behavior_revision=behavior_revision,
            dataset_revision=os.environ["JPH_DATASET_REVISION"],
            dataset_selection=dataset_selection,
            sglang_version=os.environ["JPH_SGLANG_VERSION"],
            generation_logprob_mode=generation_logprob_mode,
            inference_runtime_contract_sha256=runtime_hash,
        )
        run_id = str(runtime_contract["identity"]["run_id"])
        episode_id = f"{run_id}:{request_id}"
        model_call_id = f"{episode_id}:model:0"
        stage_rlvr_v2_agent_response(
            journal_root=os.environ["JPH_RLVR_V2_AGENT_JOURNAL_ROOT"],
            task_id=task_id,
            interaction_id=interaction_id,
            episode_id=episode_id,
            model_call_id=model_call_id,
            joint_version=joint_version,
            harness_state=state,
            harness_decision=decision,
            harness_sampling_before_checkpoint=checkpoint,
            base_messages=base_messages,
            effective_messages=effective_messages,
            base_input_tokens=base_input_ids,
            effective_input_tokens=effective_input_ids,
            expected_policy_version=expected_policy_version,
            project_commit=os.environ["JPH_PROJECT_COMMIT"],
            areal_commit=os.environ["JPH_AREAL_COMMIT"],
            behavior_snapshot_path=os.environ["JPH_BEHAVIOR_SNAPSHOT"],
            behavior_revision=behavior_revision,
            dataset_selection=dataset_selection,
            sglang_version=os.environ["JPH_SGLANG_VERSION"],
            generation_logprob_mode=generation_logprob_mode,
            inference_runtime_contract=runtime_contract,
        )
        return reward


__all__ = ["ArealJointBridgeAgent"]
