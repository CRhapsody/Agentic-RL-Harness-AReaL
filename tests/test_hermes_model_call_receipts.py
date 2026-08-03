from __future__ import annotations

import asyncio
import base64
import json
import unittest
import uuid
from dataclasses import asdict
from types import SimpleNamespace

from jphrl.hermes_agent_service import build_areal_hermes_agent_class
from jphrl.trajectory.hermes_model_call_receipts import (
    INFERENCE_SESSION_ID_METADATA_KEY,
    RECEIPT_METADATA_KEY,
    RECEIPT_STREAM_HEADER,
    SUPPORTED_HERMES_AGENT_COMMIT,
    SUPPORTED_HERMES_AGENT_VERSION,
    HermesModelCallReceiptCollector,
    HermesModelCallReceiptError,
    build_receipt_capturing_ai_agent_class,
    inference_session_id_from_metadata,
    receipts_from_public_dicts,
    receipts_to_public_dicts,
    require_supported_hermes_agent,
)


class _UUIDFactory:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> uuid.UUID:
        self._value += 1
        return uuid.UUID(int=self._value)


class FakeAIAgent:
    def __init__(self, **kwargs) -> None:
        self.api_key = kwargs["api_key"]
        self.session_api_key = kwargs["api_key"]
        self._calls = 0
        self._disable_streaming = False

    def _interruptible_api_call(self, api_kwargs):
        del api_kwargs
        self._calls += 1
        message = SimpleNamespace(
            role="assistant",
            content=f"output-{self._calls}",
        )
        return SimpleNamespace(
            id=f"interaction-{self._calls}",
            choices=[SimpleNamespace(message=message)],
            api_key="response-secret",
            session_api_key="response-session-secret",
        )


class FakeArealHermesAgent:
    def __init__(self, *, stream_result: bool = False) -> None:
        self._max_turns = 10
        self._enabled_toolsets = None
        self._disabled_toolsets = None
        self._stream_result = stream_result
        self._state = None

    async def _resolve_state(self, request):
        if self._state is None:
            inference = request.metadata["areal_inference"]
            upstream = SimpleNamespace(
                base_url=inference["base_url"],
                api_key=inference["api_key"],
                model=inference["model"],
            )
            self._state = SimpleNamespace(agent=self._build_agent(upstream))
        return self._state

    async def run(self, request, *, emitter):
        del emitter
        state = await self._resolve_state(request)
        first_messages = [{"role": "user", "content": "question"}]
        first = state.agent._interruptible_api_call({"messages": first_messages})
        second_messages = [
            *first_messages,
            {
                "role": first.choices[0].message.role,
                "content": first.choices[0].message.content,
            },
            {"role": "tool", "content": "observation"},
        ]
        state.agent._interruptible_api_call({"messages": second_messages})
        if self._stream_result:
            return SimpleNamespace(headers={"content-type": "application/json"})
        return SimpleNamespace(summary="done", metadata={"completed": True})


class InterleavingFakeArealHermesAgent(FakeArealHermesAgent):
    """Base agent that would mix receipt cursors without the outer turn lock."""

    async def run(self, request, *, emitter):
        del emitter
        state = await self._resolve_state(request)
        first_messages = [{"role": "user", "content": request.message}]
        first = state.agent._interruptible_api_call({"messages": first_messages})
        await asyncio.sleep(0)
        second_messages = [
            *first_messages,
            {
                "role": first.choices[0].message.role,
                "content": first.choices[0].message.content,
            },
            {"role": "tool", "content": "observation"},
        ]
        state.agent._interruptible_api_call({"messages": second_messages})
        return SimpleNamespace(summary="done", metadata={"completed": True})


def _request(
    *,
    session_id: str | None = "calculator-0",
    session_api_key: str = "must-never-be-exposed",
):
    metadata = {
        "areal_inference": {
            "base_url": "http://inference.example/v1",
            "api_key": session_api_key,
            "model": "policy",
        }
    }
    if session_id is not None:
        metadata[INFERENCE_SESSION_ID_METADATA_KEY] = session_id
    return SimpleNamespace(metadata=metadata, message="question")


class HermesModelCallReceiptTests(unittest.TestCase):
    def test_collector_exposes_exact_fields_parent_ordinal_and_session(self) -> None:
        collector = HermesModelCallReceiptCollector(uuid_factory=_UUIDFactory())
        collector.activate_session("calculator-0")
        cursor = collector.begin_turn()
        first_call_id = collector.new_model_call_id()
        collector.capture_response(
            model_call_id=first_call_id,
            response={
                "id": "interaction-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "tool call",
                        }
                    }
                ],
                "session_api_key": "must-never-be-recorded",
            },
            request_messages=[{"role": "user", "content": "question"}],
        )
        second_call_id = collector.new_model_call_id()
        collector.capture_response(
            model_call_id=second_call_id,
            response=SimpleNamespace(
                id="interaction-2",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content="answer",
                        )
                    )
                ],
                api_key="also-must-never-be-recorded",
            ),
            request_messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "tool call"},
                {"role": "tool", "content": "observation"},
            ],
        )

        receipts = collector.receipts_since(cursor)
        payload = receipts_to_public_dicts(
            receipts, expected_session_id="calculator-0"
        )
        self.assertEqual(
            set(payload[0]),
            {
                "model_call_id",
                "interaction_id",
                "ordinal",
                "parent_model_call_id",
                "session_id",
            },
        )
        self.assertEqual(payload[0]["interaction_id"], "interaction-1")
        self.assertEqual(payload[0]["ordinal"], 0)
        self.assertIsNone(payload[0]["parent_model_call_id"])
        self.assertEqual(payload[1]["ordinal"], 1)
        self.assertEqual(payload[1]["parent_model_call_id"], first_call_id)
        self.assertEqual(payload[1]["session_id"], "calculator-0")
        serialized = json.dumps(payload)
        self.assertNotIn("must-never-be-recorded", serialized)
        self.assertNotIn("also-must-never-be-recorded", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("authorization", serialized)

        reparsed = receipts_from_public_dicts(
            payload,
            expected_session_id="calculator-0",
        )
        self.assertEqual(reparsed, receipts)
        with self.assertRaisesRegex(
            HermesModelCallReceiptError,
            "field set differs",
        ):
            receipts_from_public_dicts(
                [{**payload[0], "session_api_key": "must-never-be-recorded"}],
                expected_session_id="calculator-0",
            )

    def test_turn_and_session_boundaries_reset_parent_without_crossing_sessions(
        self,
    ) -> None:
        collector = HermesModelCallReceiptCollector(uuid_factory=_UUIDFactory())
        collector.activate_session("session-a")
        first_cursor = collector.begin_turn()
        first_id = collector.new_model_call_id()
        collector.capture_response(
            model_call_id=first_id,
            response={
                "id": "interaction-a-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "answer-a"}}
                ],
            },
            request_messages=[{"role": "user", "content": "question-a"}],
        )
        self.assertEqual(len(collector.receipts_since(first_cursor)), 1)

        collector.begin_turn()
        second_id = collector.new_model_call_id()
        second = collector.capture_response(
            model_call_id=second_id,
            response={
                "id": "interaction-a-2",
                "choices": [
                    {"message": {"role": "assistant", "content": "answer-b"}}
                ],
            },
            request_messages=[{"role": "user", "content": "question-b"}],
        )
        self.assertEqual(second.ordinal, 1)
        self.assertIsNone(second.parent_model_call_id)

        collector.activate_session("session-b")
        third_cursor = collector.begin_turn()
        third = collector.capture_response(
            model_call_id=collector.new_model_call_id(),
            response={
                "id": "interaction-b-1",
                "choices": [
                    {"message": {"role": "assistant", "content": "answer-c"}}
                ],
            },
            request_messages=[{"role": "user", "content": "question-c"}],
        )
        self.assertEqual(third.ordinal, 0)
        self.assertIsNone(third.parent_model_call_id)
        self.assertEqual(third.session_id, "session-b")
        self.assertEqual(len(collector.receipts_since(third_cursor)), 1)

    def test_collector_rejects_missing_or_secret_shaped_public_identity(self) -> None:
        collector = HermesModelCallReceiptCollector(uuid_factory=_UUIDFactory())
        with self.assertRaisesRegex(
            HermesModelCallReceiptError, "looks like a credential"
        ):
            collector.activate_session("sk-session-secret")

        collector.activate_session("public-session")
        collector.begin_turn()
        with self.assertRaisesRegex(
            HermesModelCallReceiptError, "response ID is missing"
        ):
            collector.capture_response(
                model_call_id=collector.new_model_call_id(),
                response={"session_api_key": "secret"},
                request_messages=[],
            )
        with self.assertRaisesRegex(
            HermesModelCallReceiptError, "looks like a credential"
        ):
            collector.capture_response(
                model_call_id=collector.new_model_call_id(),
                response={"id": "sk-secret-response"},
                request_messages=[],
            )

    def test_inference_metadata_requires_public_session_and_rejects_conflict(
        self,
    ) -> None:
        self.assertEqual(
            inference_session_id_from_metadata(_request().metadata),
            "calculator-0",
        )
        with self.assertRaisesRegex(
            HermesModelCallReceiptError, "requires the non-secret"
        ):
            inference_session_id_from_metadata(_request(session_id=None).metadata)

        metadata = _request().metadata
        metadata["areal_inference"]["session_id"] = "another-session"
        with self.assertRaisesRegex(HermesModelCallReceiptError, "IDs differ"):
            inference_session_id_from_metadata(metadata)

    def test_explicit_ai_agent_subclass_captures_real_response_id(self) -> None:
        receipt_class = build_receipt_capturing_ai_agent_class(FakeAIAgent)
        self.assertTrue(issubclass(receipt_class, FakeAIAgent))
        collector = HermesModelCallReceiptCollector(uuid_factory=_UUIDFactory())
        collector.activate_session("calculator-0")
        cursor = collector.begin_turn()
        agent = receipt_class(
            base_url="http://inference.example/v1",
            api_key="must-never-be-exposed",
            model="policy",
            jphrl_receipt_collector=collector,
        )
        response = agent._interruptible_api_call({"messages": []})
        self.assertEqual(response.id, "interaction-1")
        self.assertTrue(agent._disable_streaming)
        receipt = collector.receipts_since(cursor)[0]
        self.assertEqual(receipt.interaction_id, response.id)
        self.assertNotIn("must-never-be-exposed", json.dumps(asdict(receipt)))

    def test_dependency_version_is_locked_to_audited_release(self) -> None:
        self.assertEqual(SUPPORTED_HERMES_AGENT_VERSION, "0.19.0")
        self.assertEqual(
            SUPPORTED_HERMES_AGENT_COMMIT,
            "3ef6bbd201263d354fd83ec55b3c306ded2eb72a",
        )
        self.assertEqual(
            require_supported_hermes_agent(distribution_version="0.19.0"),
            "0.19.0",
        )
        with self.assertRaisesRegex(
            HermesModelCallReceiptError, "unsupported hermes-agent version"
        ):
            require_supported_hermes_agent(distribution_version="0.20.0")

    def test_areal_entrypoint_exposes_each_receipt_and_no_credentials(self) -> None:
        receipt_agent_class = build_areal_hermes_agent_class(
            FakeArealHermesAgent,
            FakeAIAgent,
        )
        agent = receipt_agent_class()
        response = asyncio.run(agent.run(_request(), emitter=object()))
        receipts = response.metadata[RECEIPT_METADATA_KEY]
        self.assertEqual(len(receipts), 2)
        self.assertEqual(
            [receipt["interaction_id"] for receipt in receipts],
            ["interaction-1", "interaction-2"],
        )
        self.assertEqual([receipt["ordinal"] for receipt in receipts], [0, 1])
        self.assertIsNone(receipts[0]["parent_model_call_id"])
        self.assertEqual(
            receipts[1]["parent_model_call_id"], receipts[0]["model_call_id"]
        )
        self.assertEqual(
            {receipt["session_id"] for receipt in receipts}, {"calculator-0"}
        )
        serialized = json.dumps(response.metadata)
        self.assertNotIn("must-never-be-exposed", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("session_api_key", serialized)

    def test_areal_entrypoint_serializes_receipt_cursor_per_session(self) -> None:
        receipt_agent_class = build_areal_hermes_agent_class(
            InterleavingFakeArealHermesAgent,
            FakeAIAgent,
        )
        agent = receipt_agent_class()

        async def run_concurrently():
            return await asyncio.gather(
                agent.run(_request(), emitter=object()),
                agent.run(_request(), emitter=object()),
            )

        first, second = asyncio.run(run_concurrently())
        first_receipts = first.metadata[RECEIPT_METADATA_KEY]
        second_receipts = second.metadata[RECEIPT_METADATA_KEY]
        self.assertEqual(len(first_receipts), 2)
        self.assertEqual(len(second_receipts), 2)
        self.assertEqual(
            [receipt["ordinal"] for receipt in first_receipts],
            [0, 1],
        )
        self.assertEqual(
            [receipt["ordinal"] for receipt in second_receipts],
            [2, 3],
        )
        self.assertTrue(
            {receipt["model_call_id"] for receipt in first_receipts}.isdisjoint(
                receipt["model_call_id"] for receipt in second_receipts
            )
        )

    def test_areal_entrypoint_fails_closed_without_inference_session_id(self) -> None:
        receipt_agent_class = build_areal_hermes_agent_class(
            FakeArealHermesAgent,
            FakeAIAgent,
        )
        agent = receipt_agent_class()
        with self.assertRaisesRegex(
            HermesModelCallReceiptError, "requires the non-secret"
        ):
            asyncio.run(
                agent.run(_request(session_id=None), emitter=object())
            )

    def test_chat_passthrough_exposes_non_secret_base64_header(self) -> None:
        receipt_agent_class = build_areal_hermes_agent_class(
            FakeArealHermesAgent,
            FakeAIAgent,
        )
        agent = receipt_agent_class(stream_result=True)
        response = asyncio.run(agent.run(_request(), emitter=object()))
        encoded = response.headers[RECEIPT_STREAM_HEADER]
        receipts = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
        self.assertEqual(len(receipts), 2)
        self.assertEqual(receipts[0]["interaction_id"], "interaction-1")
        self.assertNotIn("must-never-be-exposed", json.dumps(response.headers))


if __name__ == "__main__":
    unittest.main()
