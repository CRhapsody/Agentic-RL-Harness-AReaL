"""Receipt-aware replacement entry point for AReaL's Hermes example.

Deploy it from the project checkout instead of modifying fixed AReaL::

    areal agent run --agent jphrl.hermes_agent_service.HermesAgent ...

For a self-evolution ``/v1/responses`` request, the caller must send the
non-secret ID returned by ``rl/start_session`` as::

    {
      "session_api_key": "<routing credential>",
      "metadata": {"jphrl_inference_session_id": "<public session_id>"}
    }

The AReaL OpenResponses bridge already forwards arbitrary request metadata.
The credential remains in routing metadata used by upstream Hermes, while only
the public ``session_id`` and four model-call identity fields are returned in
``AgentResponse.metadata['jphrl_model_call_receipts']``. The chat-completions
channel exposes the same JSON as URL-safe base64 in
``x-jphrl-model-call-receipts-b64`` because its body must remain byte-exact.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from typing import Any

from .trajectory.hermes_model_call_receipts import (
    RECEIPT_METADATA_KEY,
    RECEIPT_STREAM_HEADER,
    HermesModelCallReceiptCollector,
    HermesModelCallReceiptError,
    build_receipt_capturing_ai_agent_class,
    inference_session_id_from_metadata,
    receipts_to_public_dicts,
    require_supported_hermes_agent,
)


def _compact_base64_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def build_areal_hermes_agent_class(
    base_hermes_agent_class: type,
    base_ai_agent_class: type,
) -> type:
    """Build the pluggable AReaL agent without patching either dependency."""

    if not isinstance(base_hermes_agent_class, type):
        raise HermesModelCallReceiptError("AReaL Hermes agent base must be a class")
    receipt_ai_agent_class = build_receipt_capturing_ai_agent_class(
        base_ai_agent_class
    )

    class ReceiptAwareArealHermesAgent(  # type: ignore[misc, valid-type]
        base_hermes_agent_class
    ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # The fixed Hermes base serializes the blocking conversation, but
            # its lock is acquired after this subclass takes a receipt cursor.
            # Keep the complete cursor -> model calls -> response publication
            # window single-flight per public inference session as well.
            self._jphrl_receipt_turn_locks: dict[str, asyncio.Lock] = {}

        def _build_agent(self, upstream: Any) -> Any:
            collector = HermesModelCallReceiptCollector()
            return receipt_ai_agent_class(
                base_url=upstream.base_url,
                api_key=upstream.api_key,
                model=upstream.model,
                max_iterations=self._max_turns,
                enabled_toolsets=self._enabled_toolsets,
                disabled_toolsets=self._disabled_toolsets,
                save_trajectories=False,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=None,
                jphrl_receipt_collector=collector,
            )

        async def _resolve_state(self, request: Any) -> Any:
            metadata = request.metadata or {}
            session_id = inference_session_id_from_metadata(metadata)
            state = await super()._resolve_state(request)
            if session_id is not None:
                collector = getattr(state.agent, "jphrl_receipt_collector", None)
                if not isinstance(collector, HermesModelCallReceiptCollector):
                    raise HermesModelCallReceiptError(
                        "self-evolution Hermes agent has no receipt collector"
                    )
                collector.activate_session(session_id)
            return state

        async def run(self, request: Any, *, emitter: Any) -> Any:
            metadata: Mapping[str, object] = request.metadata or {}
            session_id = inference_session_id_from_metadata(metadata)
            if session_id is None:
                return await super().run(request, emitter=emitter)

            lock = self._jphrl_receipt_turn_locks.setdefault(
                session_id,
                asyncio.Lock(),
            )
            async with lock:
                return await self._run_with_receipts(
                    request,
                    emitter=emitter,
                    session_id=session_id,
                )

        async def _run_with_receipts(
            self,
            request: Any,
            *,
            emitter: Any,
            session_id: str,
        ) -> Any:
            # Resolve once before the base class resolves the same cached state;
            # this establishes the per-turn cursor before Hermes can call its LLM.
            state = await self._resolve_state(request)
            collector = state.agent.jphrl_receipt_collector
            cursor = collector.begin_turn()
            result = await super().run(request, emitter=emitter)
            receipts = collector.receipts_since(cursor)
            if not receipts:
                raise HermesModelCallReceiptError(
                    "self-evolution Hermes turn produced no upstream response receipt"
                )
            public_receipts = receipts_to_public_dicts(
                receipts,
                expected_session_id=session_id,
                allowed_parent_model_call_ids=collector.model_call_ids_before(
                    cursor
                ),
            )

            result_metadata = getattr(result, "metadata", None)
            if isinstance(result_metadata, dict):
                result.metadata = {
                    **result_metadata,
                    RECEIPT_METADATA_KEY: public_receipts,
                }
                return result

            result_headers = getattr(result, "headers", None)
            if isinstance(result_headers, dict):
                result.headers = {
                    **result_headers,
                    RECEIPT_STREAM_HEADER: _compact_base64_json(public_receipts),
                }
                return result

            raise HermesModelCallReceiptError(
                "AReaL Hermes result has no metadata or passthrough headers"
            )

    ReceiptAwareArealHermesAgent.__name__ = "ReceiptAwareArealHermesAgent"
    ReceiptAwareArealHermesAgent.__qualname__ = "ReceiptAwareArealHermesAgent"
    return ReceiptAwareArealHermesAgent


_RUNTIME_CLASS: type | None = None


def _runtime_class() -> type:
    global _RUNTIME_CLASS
    if _RUNTIME_CLASS is None:
        require_supported_hermes_agent()
        try:
            from examples.hermes.hermes import HermesAgent as ArealHermesAgent
            from run_agent import AIAgent
        except ImportError as exc:
            raise HermesModelCallReceiptError(
                "receipt-aware Hermes requires fixed AReaL and hermes-agent==0.19.0"
            ) from exc
        _RUNTIME_CLASS = build_areal_hermes_agent_class(
            ArealHermesAgent,
            AIAgent,
        )
    return _RUNTIME_CLASS


class HermesAgent:
    """Dynamic-import entry point returning the receipt-aware AReaL subclass."""

    def __new__(cls, **kwargs: Any) -> Any:
        return _runtime_class()(**kwargs)
