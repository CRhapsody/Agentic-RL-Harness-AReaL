from __future__ import annotations

import os
import uuid
from typing import Any

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.workflow.rlvr import RLVRWorkflow

from jphrl.trajectory.areal_trace_contract import (
    build_areal_trace_record,
    write_areal_trace_record,
)


class ArealTraceRLVRWorkflow(RLVRWorkflow):
    """Run AReaL generation and persist the official interaction tensor roundtrip."""

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward]:
        if isinstance(self.reward_fn, str):
            from areal.api import AsyncRewardWrapper
            from areal.utils.dynamic_import import import_from_string

            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        request = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        prompt_text = self.tokenizer.decode(input_ids)
        response, reward = await self._collect_samples(
            engine, request, prompt_text, data
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=response,
            reward=float(reward),
            original_reward=float(reward),
        )
        interaction.interaction_id = request.rid
        tensor_dict = interaction.to_tensor_dict()

        trace_dir = os.environ["JPH_AREAL_TRACE_DIR"]
        allowed_root = os.environ["JPH_ROOT"]
        record = build_areal_trace_record(
            task_id=workflow_context.get().task_id,
            request_id=request.rid,
            model_response=response,
            interaction=interaction,
            tensor_dict=tensor_dict,
            areal_commit=os.environ["JPH_AREAL_COMMIT"],
            behavior_snapshot_path=os.environ["JPH_BEHAVIOR_SNAPSHOT"],
            behavior_revision=os.environ["JPH_BEHAVIOR_REVISION"],
        )
        write_areal_trace_record(
            record,
            trace_dir=trace_dir,
            allowed_root=allowed_root,
        )
        return {request.rid: interaction}
