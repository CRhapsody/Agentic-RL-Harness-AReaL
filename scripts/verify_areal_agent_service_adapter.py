#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from areal.api import ModelResponse as ArealModelResponse
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.v2.inference_service.data_proxy.session import SessionData

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import SmokeHarnessController
from jphrl.harness.spec import HarnessSpec
from jphrl.models.base import ModelResponse
from jphrl.runner import run_calculator_smoke
from jphrl.trajectory.areal_agent_service_adapter import (
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    validate_agent_service_training_record,
    validate_agent_service_training_trace,
)
from jphrl.trajectory.areal_data_proxy_pre_batch import (
    VerifiedDataProxyPreBatchHook,
    export_session_trajectory_with_pre_batch_hook,
)
from jphrl.trajectory.areal_online_binding import (
    PersistentAgentServicePreBatchBinder,
    stage_agent_service_training_binding,
)
from jphrl.trajectory.hermes_model_call_receipts import HermesModelCallReceipt


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


def _trace():
    result = run_calculator_smoke(
        model=TokenBackedCalculatorModel(),
        task=TASKS["add-17-25"],
        controller=SmokeHarnessController(),
        harness_spec=HarnessSpec(),
    )
    if not result.success:
        raise RuntimeError("token-backed calculator trace did not succeed")
    return result.trace


def _add_interactions(session: SessionData, *, style: str) -> None:
    first = InteractionWithTokenLogpReward(
        messages=[{"role": "user", "content": "calculate 17 + 25"}],
        output_message_list=[
            {
                "role": "assistant",
                "content": '{"tool":"calculator","expression":"17 + 25"}',
            }
        ],
        model_response=ArealModelResponse(
            input_tokens=[10, 11],
            output_tokens=[20, 21],
            output_logprobs=[-0.2, -0.3],
            output_versions=[7, 7],
            stop_reason="tool_calls",
        ),
        chat_template_type=style,
    )
    first._interaction_id = "interaction-1"
    session.active_completions[first.interaction_id] = first

    second = InteractionWithTokenLogpReward(
        messages=[
            {"role": "user", "content": "calculate 17 + 25"},
            {
                "role": "assistant",
                "content": '{"tool":"calculator","expression":"17 + 25"}',
            },
            {"role": "tool", "content": "42"},
        ],
        output_message_list=[{"role": "assistant", "content": '{"answer":"42"}'}],
        model_response=ArealModelResponse(
            input_tokens=[10, 11, 20, 21, 30],
            output_tokens=[40],
            output_logprobs=[-0.4],
            output_versions=[7],
            stop_reason="stop",
        ),
        chat_template_type=style,
    )
    second._interaction_id = "interaction-2"
    session.active_completions[second.interaction_id] = second
    if second.parent is not first:
        raise RuntimeError("AReaL did not construct the expected parent relation")


def _run_style(style: str) -> dict[str, object]:
    trace = _trace()
    trace_audit = validate_agent_service_training_trace(trace)
    model_call_ids = list(trace_audit["model_call_ids"])
    session_id = f"adapter-{style}-0"
    session_data = SessionData(session_id=session_id)
    _add_interactions(
        session_data,
        style="concat" if style == "concat" else "hf",
    )
    reward_result = session_data.set_reward(None, 1.0)
    if reward_result.trajectory_id is None:
        raise RuntimeError("AReaL did not close the rewarded trajectory")
    session_receipt = AgentServiceSessionReceipt(
        group_id=f"group-{style}",
        session_id=session_id,
    )
    trajectory_receipt = AgentServiceTrajectoryReceipt(
        session_id=session_id,
        trajectory_id=reward_result.trajectory_id,
        interaction_count=reward_result.interaction_count,
        ready_transition=reward_result.ready_transition,
    )
    hermes_receipts = [
        HermesModelCallReceipt(
            model_call_id=model_call_ids[0],
            interaction_id="interaction-1",
            ordinal=0,
            parent_model_call_id=None,
            session_id=session_id,
        ),
        HermesModelCallReceipt(
            model_call_id=model_call_ids[1],
            interaction_id="interaction-2",
            ordinal=1,
            parent_model_call_id=model_call_ids[0],
            session_id=session_id,
        ),
    ]

    with tempfile.TemporaryDirectory(prefix=f"jph-{style}-") as directory:
        journal_root = Path(directory)
        stage_agent_service_training_binding(
            journal_root=journal_root,
            trace=trace,
            session=session_receipt,
            model_calls=hermes_receipts,
            trajectory=trajectory_receipt,
            export_style=style,
            turn_discount=1.0,
        )
        hook = VerifiedDataProxyPreBatchHook(
            PersistentAgentServicePreBatchBinder(journal_root)
        )
        trajectory_id, exported = asyncio.run(
            export_session_trajectory_with_pre_batch_hook(
                session_data,
                discount=1.0,
                style=style,
                trajectory_id=reward_result.trajectory_id,
                hook=hook,
            )
        )
        if trajectory_id != trajectory_receipt.trajectory_id:
            raise RuntimeError("pre-batch hook observed the wrong trajectory")
        if style == "individual" and list(exported) != [
            "interaction-1",
            "interaction-2",
        ]:
            raise RuntimeError("individual export lost interaction identity")
        if style == "concat" and list(exported) != ["interaction-2"]:
            raise RuntimeError("concat export did not expose its real leaf")

        record_files = list((journal_root / "records").glob("*.json"))
        marker_files = list((journal_root / "finalized").glob("*.json"))
        if len(record_files) != 1 or len(marker_files) != 1:
            raise RuntimeError("pre-batch binding was not finalized exactly once")
        record_text = record_files[0].read_text(encoding="utf-8")
        marker_text = marker_files[0].read_text(encoding="utf-8")
        if "session_api_key" in record_text or "admin_api_key" in record_text:
            raise RuntimeError("credential field entered the training record")
        marker = json.loads(marker_text)
        if marker["evidence_scope"]["policy_optimizer_update"]:
            raise RuntimeError("pre-batch binding fabricated optimizer evidence")
        if marker["evidence_scope"]["harness_optimizer_update"]:
            raise RuntimeError("pre-batch binding fabricated harness update evidence")
        record = json.loads(record_text)
        return validate_agent_service_training_record(record)


def main() -> None:
    audits = {
        "individual": _run_style("individual"),
        "concat": _run_style("concat"),
    }
    if audits["individual"]["sample_count"] != 2:
        raise RuntimeError("individual export did not preserve two samples")
    if audits["concat"]["sample_count"] != 1:
        raise RuntimeError("concat export did not preserve one leaf sample")
    print(
        json.dumps(
            {
                "ok": True,
                "areal_hook": "after-export-trajectory-before-concat-padded-tensors",
                "gpu_used": False,
                "audits": audits,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
