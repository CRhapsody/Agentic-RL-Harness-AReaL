from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_token_ids: list[int]
    output_token_ids: list[int]
    output_token_logprobs: list[float]
    output_versions: list[int]
    completion_loss_mask: list[int]
    policy_version: str
    tokenizer_version: str
    policy_kind: str
    token_metadata_status: str


class StructuredChatModel(Protocol):
    policy_version: str
    tokenizer_version: str

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> ModelResponse:
        ...


class MockStructuredModel:
    """A deterministic model used only for local data-plane tests."""

    policy_version = "mock-policy-v1"
    tokenizer_version = "mock-tokenizer-v1"

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> ModelResponse:
        del max_new_tokens
        conversation = "\n".join(message["content"] for message in messages)
        if "calculator 返回" in conversation:
            tool_result = conversation.rsplit("calculator 返回：", maxsplit=1)[-1].splitlines()[0].strip()
            text = '{"answer":"' + tool_result + '"}'
        elif "17 与 25" in conversation:
            text = '{"tool":"calculator","expression":"17 + 25"}'
        elif "13 与 7" in conversation:
            text = '{"tool":"calculator","expression":"13 * 7"}'
        else:
            text = '{"error":"unknown task"}'
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
