from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from typing import Any

# AReaL's fixed Hermes example was audited against this released Hermes build.
# In particular, its quiet, headless path calls ``_interruptible_api_call`` and
# returns the real OpenAI completion object.  Later Hermes builds are not
# accepted implicitly because their streaming adapter may synthesize an ID.
SUPPORTED_HERMES_AGENT_VERSION = "0.19.0"
SUPPORTED_HERMES_AGENT_COMMIT = "3ef6bbd201263d354fd83ec55b3c306ded2eb72a"

RECEIPT_METADATA_KEY = "jphrl_model_call_receipts"
RECEIPT_STREAM_HEADER = "x-jphrl-model-call-receipts-b64"
INFERENCE_SESSION_ID_METADATA_KEY = "jphrl_inference_session_id"

_SECRET_FIELD_NAMES = frozenset(
    {
        "admin_api_key",
        "api_key",
        "authorization",
        "session_api_key",
    }
)


class HermesModelCallReceiptError(ValueError):
    """Raised when Hermes cannot expose a trustworthy non-secret receipt."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HermesModelCallReceiptError(message)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _looks_like_credential(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized.startswith(("sk-", "bearer "))


def _require_public_id(value: object, field_name: str) -> str:
    _require(_is_non_empty_string(value), f"{field_name} is missing")
    public_id = str(value).strip()
    _require(
        not _looks_like_credential(public_id),
        f"{field_name} looks like a credential, not a public identity",
    )
    return public_id


def _response_id(response: Mapping[str, object] | object) -> str:
    value = (
        response.get("id")
        if isinstance(response, Mapping)
        else getattr(response, "id", None)
    )
    return _require_public_id(value, "Hermes upstream response ID")


def _plain_message(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items() if item is not None}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        _require(isinstance(dumped, Mapping), "Hermes response message is invalid")
        return {str(key): item for key, item in dumped.items()}
    attributes = getattr(value, "__dict__", None)
    _require(isinstance(attributes, Mapping), "Hermes response message is invalid")
    return {
        str(key): item
        for key, item in attributes.items()
        if not str(key).startswith("_") and item is not None
    }


def _message_fingerprint(value: object) -> str:
    try:
        payload = json.dumps(
            _plain_message(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HermesModelCallReceiptError(
            "Hermes message cannot be fingerprinted for parent identity"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _request_message_fingerprints(
    request_messages: object,
) -> tuple[str, ...]:
    _require(
        isinstance(request_messages, (list, tuple)),
        "Hermes upstream request messages are missing",
    )
    return tuple(_message_fingerprint(message) for message in request_messages)


def _response_message_fingerprints(
    response: Mapping[str, object] | object,
) -> tuple[str, ...]:
    choices = (
        response.get("choices")
        if isinstance(response, Mapping)
        else getattr(response, "choices", None)
    )
    _require(
        isinstance(choices, (list, tuple)) and len(choices) == 1,
        "Hermes upstream response must contain exactly one choice",
    )
    choice = choices[0]
    message = (
        choice.get("message")
        if isinstance(choice, Mapping)
        else getattr(choice, "message", None)
    )
    _require(message is not None, "Hermes upstream response message is missing")
    return (_message_fingerprint(message),)


@dataclass(frozen=True)
class HermesModelCallReceipt:
    """The complete public identity of one successful Hermes model call.

    The field set is intentionally closed.  In particular, request metadata,
    HTTP headers, base URLs, prompts, and API credentials can never enter the
    serialized receipt through this type.
    """

    model_call_id: str
    interaction_id: str
    ordinal: int
    parent_model_call_id: str | None
    session_id: str


def validate_hermes_model_call_receipts(
    receipts: Sequence[HermesModelCallReceipt],
    *,
    expected_session_id: str | None = None,
    allowed_parent_model_call_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Validate a contiguous, parent-before-child receipt sequence."""

    _require(bool(receipts), "Hermes receipt sequence is empty")
    expected_session = (
        _require_public_id(expected_session_id, "inference session ID")
        if expected_session_id is not None
        else None
    )
    model_call_ids: list[str] = []
    interaction_ids: list[str] = []
    session_ids: set[str] = set()
    by_model_call: dict[str, HermesModelCallReceipt] = {}
    external_parent_ids = {
        _require_public_id(value, "external parent model call ID")
        for value in allowed_parent_model_call_ids
    }
    first_ordinal = receipts[0].ordinal
    _require(
        isinstance(first_ordinal, int)
        and not isinstance(first_ordinal, bool)
        and first_ordinal >= 0,
        "Hermes receipt ordinal is invalid",
    )

    for offset, receipt in enumerate(receipts):
        _require(
            set(asdict(receipt)) == set(HermesModelCallReceipt.__dataclass_fields__),
            "Hermes receipt field set differs from the public schema",
        )
        model_call_id = _require_public_id(receipt.model_call_id, "model call ID")
        interaction_id = _require_public_id(
            receipt.interaction_id, "interaction ID"
        )
        session_id = _require_public_id(receipt.session_id, "inference session ID")
        _require(
            receipt.ordinal == first_ordinal + offset,
            "Hermes receipt ordinals must be contiguous and ordered",
        )
        parent_id = receipt.parent_model_call_id
        _require(
            parent_id is None or _is_non_empty_string(parent_id),
            "parent model call ID must be null or non-empty",
        )
        if parent_id is not None:
            _require(
                parent_id in by_model_call or parent_id in external_parent_ids,
                "parent model call must precede its child receipt or turn slice",
            )
        _require(
            model_call_id not in by_model_call,
            "Hermes model call ID appears more than once",
        )
        _require(
            interaction_id not in interaction_ids,
            "Hermes interaction ID appears more than once",
        )
        by_model_call[model_call_id] = receipt
        model_call_ids.append(model_call_id)
        interaction_ids.append(interaction_id)
        session_ids.add(session_id)

    _require(len(session_ids) == 1, "Hermes receipts cross inference sessions")
    session_id = receipts[0].session_id
    if expected_session is not None:
        _require(
            session_id == expected_session,
            "Hermes receipt session differs from the routed inference session",
        )
    return {
        "ok": True,
        "session_id": session_id,
        "first_ordinal": first_ordinal,
        "receipt_count": len(receipts),
        "model_call_ids": model_call_ids,
        "interaction_ids": interaction_ids,
    }


def receipts_to_public_dicts(
    receipts: Sequence[HermesModelCallReceipt],
    *,
    expected_session_id: str | None = None,
    allowed_parent_model_call_ids: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Serialize only the five approved public receipt fields."""

    validate_hermes_model_call_receipts(
        receipts,
        expected_session_id=expected_session_id,
        allowed_parent_model_call_ids=allowed_parent_model_call_ids,
    )
    payload = [asdict(receipt) for receipt in receipts]
    for item in payload:
        _require(
            not (set(item) & _SECRET_FIELD_NAMES),
            "credential field cannot enter a Hermes model-call receipt",
        )
    return payload


def receipts_from_public_dicts(
    payload: object,
    *,
    expected_session_id: str | None = None,
    allowed_parent_model_call_ids: Sequence[str] = (),
) -> tuple[HermesModelCallReceipt, ...]:
    """Parse the exact five-field payload emitted by the Hermes entrypoint.

    This is the process boundary used by the persistent online binder. Extra
    metadata (including a credential accidentally copied by a caller) is
    rejected instead of silently discarded.
    """

    _require(isinstance(payload, list), "Hermes receipt payload must be a list")
    expected_fields = set(HermesModelCallReceipt.__dataclass_fields__)
    receipts: list[HermesModelCallReceipt] = []
    for raw in payload:
        _require(isinstance(raw, Mapping), "Hermes receipt payload item is invalid")
        _require(
            set(raw) == expected_fields,
            "Hermes receipt payload field set differs from the public schema",
        )
        try:
            receipts.append(HermesModelCallReceipt(**dict(raw)))
        except TypeError as exc:
            raise HermesModelCallReceiptError(
                "Hermes receipt payload item is invalid"
            ) from exc
    validate_hermes_model_call_receipts(
        receipts,
        expected_session_id=expected_session_id,
        allowed_parent_model_call_ids=allowed_parent_model_call_ids,
    )
    return tuple(receipts)


class HermesModelCallReceiptCollector:
    """Session-scoped collector used by a receipt-aware Hermes ``AIAgent``.

    One collector belongs to one in-process Hermes agent.  Calls are serialized
    by AReaL's per-session lock, while the collector's own lock protects the
    short boundary between the Agent Service event loop and Hermes' worker
    thread.  A new inference ``session_id`` resets ordinal and parent state.
    Parent identity follows AReaL's longest strict message-prefix rule.  Only
    SHA-256 message fingerprints are retained for that comparison; prompt and
    response content never enter the receipt collector's stored identity data.
    """

    def __init__(self, *, uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4) -> None:
        self._uuid_factory = uuid_factory
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._receipts: list[HermesModelCallReceipt] = []
        self._next_ordinal = 0
        self._message_trees: list[
            tuple[str, tuple[str, ...], tuple[str, ...]]
        ] = []

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def activate_session(self, session_id: str) -> None:
        public_session_id = _require_public_id(session_id, "inference session ID")
        with self._lock:
            if self._session_id == public_session_id:
                return
            self._session_id = public_session_id
            self._receipts.clear()
            self._next_ordinal = 0
            self._message_trees.clear()

    def begin_turn(self) -> int:
        """Return a cursor before one Hermes turn starts."""

        with self._lock:
            _require(
                self._session_id is not None,
                "inference session ID must be activated before a Hermes turn",
            )
            return len(self._receipts)

    def new_model_call_id(self) -> str:
        with self._lock:
            _require(
                self._session_id is not None,
                "inference session ID must be activated before a model call",
            )
        generated = self._uuid_factory()
        value = getattr(generated, "hex", None)
        _require(_is_non_empty_string(value), "model call UUID factory is invalid")
        return f"jph-model-call-{value}"

    def capture_response(
        self,
        *,
        model_call_id: str,
        response: Mapping[str, object] | object,
        request_messages: object,
    ) -> HermesModelCallReceipt:
        public_model_call_id = _require_public_id(model_call_id, "model call ID")
        interaction_id = _response_id(response)
        request_fingerprints = _request_message_fingerprints(request_messages)
        output_fingerprints = _response_message_fingerprints(response)
        with self._lock:
            _require(
                self._session_id is not None,
                "inference session ID must be activated before a model call",
            )
            _require(
                all(
                    receipt.model_call_id != public_model_call_id
                    for receipt in self._receipts
                ),
                "Hermes model call ID appears more than once",
            )
            _require(
                all(
                    receipt.interaction_id != interaction_id
                    for receipt in self._receipts
                ),
                "Hermes interaction ID appears more than once",
            )
            parent_model_call_id: str | None = None
            for candidate_id, candidate_request, candidate_output in sorted(
                self._message_trees,
                key=lambda item: len(item[1]),
                reverse=True,
            ):
                candidate_prefix = candidate_request + candidate_output
                if (
                    len(candidate_prefix) <= len(request_fingerprints)
                    and request_fingerprints[: len(candidate_prefix)]
                    == candidate_prefix
                ):
                    parent_model_call_id = candidate_id
                    break
            receipt = HermesModelCallReceipt(
                model_call_id=public_model_call_id,
                interaction_id=interaction_id,
                ordinal=self._next_ordinal,
                parent_model_call_id=parent_model_call_id,
                session_id=self._session_id,
            )
            self._receipts.append(receipt)
            self._message_trees.append(
                (
                    public_model_call_id,
                    request_fingerprints,
                    output_fingerprints,
                )
            )
            self._next_ordinal += 1
            return receipt

    def receipts_since(self, cursor: int) -> tuple[HermesModelCallReceipt, ...]:
        _require(
            isinstance(cursor, int) and not isinstance(cursor, bool) and cursor >= 0,
            "Hermes receipt cursor is invalid",
        )
        with self._lock:
            _require(cursor <= len(self._receipts), "Hermes receipt cursor is stale")
            all_receipts = tuple(self._receipts)
            session_id = self._session_id
        if all_receipts:
            validate_hermes_model_call_receipts(
                all_receipts, expected_session_id=session_id
            )
        return all_receipts[cursor:]

    def model_call_ids_before(self, cursor: int) -> tuple[str, ...]:
        _require(
            isinstance(cursor, int) and not isinstance(cursor, bool) and cursor >= 0,
            "Hermes receipt cursor is invalid",
        )
        with self._lock:
            _require(cursor <= len(self._receipts), "Hermes receipt cursor is stale")
            return tuple(receipt.model_call_id for receipt in self._receipts[:cursor])


class _ReceiptCapturingCompletionsProxy:
    def __init__(self, resource: Any, collector: HermesModelCallReceiptCollector) -> None:
        self._resource = resource
        self._collector = collector

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resource, name)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        _require(not args, "Hermes direct completion call must use keyword arguments")
        model_call_id = self._collector.new_model_call_id()
        response = self._resource.create(**kwargs)
        self._collector.capture_response(
            model_call_id=model_call_id,
            response=response,
            request_messages=kwargs.get("messages"),
        )
        return response


class _ReceiptCapturingChatProxy:
    def __init__(self, resource: Any, collector: HermesModelCallReceiptCollector) -> None:
        self._resource = resource
        self.completions = _ReceiptCapturingCompletionsProxy(
            resource.completions,
            collector,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resource, name)


class _ReceiptCapturingClientProxy:
    def __init__(self, client: Any, collector: HermesModelCallReceiptCollector) -> None:
        self._client = client
        self.chat = _ReceiptCapturingChatProxy(client.chat, collector)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def inference_session_id_from_metadata(
    metadata: Mapping[str, object] | None,
) -> str | None:
    """Read the caller-supplied public session ID for self-evolution.

    AReaL's OpenResponses bridge already forwards arbitrary request metadata,
    so callers send ``metadata[jphrl_inference_session_id]`` alongside the
    top-level ``session_api_key``.  A future DataProxy may place the same value
    directly in ``metadata['areal_inference']['session_id']``; both forms are
    accepted only when they agree.  The secret key is deliberately ignored.
    """

    if metadata is None:
        return None
    _require(isinstance(metadata, Mapping), "Agent request metadata is invalid")
    inference = metadata.get("areal_inference")
    if inference is None:
        return None
    _require(isinstance(inference, Mapping), "AReaL inference metadata is invalid")

    routed_value = inference.get("session_id")
    caller_value = metadata.get(INFERENCE_SESSION_ID_METADATA_KEY)
    _require(
        routed_value is not None or caller_value is not None,
        (
            "self-evolution requires the non-secret inference session ID in "
            f"metadata['{INFERENCE_SESSION_ID_METADATA_KEY}']"
        ),
    )
    if routed_value is not None and caller_value is not None:
        _require(
            routed_value == caller_value,
            "caller and DataProxy inference session IDs differ",
        )
    return _require_public_id(
        routed_value if routed_value is not None else caller_value,
        "inference session ID",
    )


def require_supported_hermes_agent(
    *, distribution_version: str | None = None
) -> str:
    """Fail closed unless the audited Hermes distribution is installed."""

    if distribution_version is None:
        try:
            distribution_version = importlib_metadata.version("hermes-agent")
        except importlib_metadata.PackageNotFoundError as exc:
            raise HermesModelCallReceiptError(
                "hermes-agent is not installed; require hermes-agent==0.19.0"
            ) from exc
    _require(
        distribution_version == SUPPORTED_HERMES_AGENT_VERSION,
        (
            "unsupported hermes-agent version: "
            f"{distribution_version!r}; require {SUPPORTED_HERMES_AGENT_VERSION} "
            f"at {SUPPORTED_HERMES_AGENT_COMMIT}"
        ),
    )
    return distribution_version


def build_receipt_capturing_ai_agent_class(base_class: type) -> type:
    """Create an explicit ``AIAgent`` subclass that captures real response IDs.

    This is intentionally a subclass hook, not a module-level monkeypatch.  It
    leaves other Hermes agents and concurrent AReaL sessions untouched.
    """

    _require(isinstance(base_class, type), "Hermes AIAgent base must be a class")

    class ReceiptCapturingAIAgent(base_class):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args: Any,
            jphrl_receipt_collector: HermesModelCallReceiptCollector,
            **kwargs: Any,
        ) -> None:
            _require(
                isinstance(
                    jphrl_receipt_collector, HermesModelCallReceiptCollector
                ),
                "Hermes receipt collector is invalid",
            )
            self.jphrl_receipt_collector = jphrl_receipt_collector
            super().__init__(*args, **kwargs)
            _require(
                getattr(self, "api_mode", "chat_completions")
                == "chat_completions",
                "AReaL Hermes receipts require the chat-completions API mode",
            )
            # Hermes 0.19.0's stream adapter emits a synthetic ``stream-*`` ID.
            # AReaL's headless path does not need token display, so force the
            # complete-response path that preserves the upstream interaction ID.
            self._disable_streaming = True

        def _interruptible_api_call(self, api_kwargs: dict[str, Any]) -> Any:
            model_call_id = self.jphrl_receipt_collector.new_model_call_id()
            response = super()._interruptible_api_call(api_kwargs)
            self.jphrl_receipt_collector.capture_response(
                model_call_id=model_call_id,
                response=response,
                request_messages=api_kwargs.get("messages"),
            )
            return response

        def _ensure_primary_openai_client(self, *, reason: str) -> Any:
            client = super()._ensure_primary_openai_client(reason=reason)
            if reason.startswith("iteration_limit_summary"):
                # Hermes 0.19.0 performs this one fallback call directly on
                # its shared client instead of using _interruptible_api_call.
                # Wrap only the returned view for this call; never monkeypatch
                # or replace the shared client used by other sessions.
                return _ReceiptCapturingClientProxy(
                    client,
                    self.jphrl_receipt_collector,
                )
            return client

        def _interruptible_streaming_api_call(
            self,
            api_kwargs: dict[str, Any],
            *,
            on_first_delta: Callable[[], None] | None = None,
        ) -> Any:
            # Defensive fallback if upstream routing ignores
            # ``_disable_streaming``.  Do not accept Hermes' synthetic stream ID.
            del on_first_delta
            return self._interruptible_api_call(api_kwargs)

    ReceiptCapturingAIAgent.__name__ = "ReceiptCapturingAIAgent"
    ReceiptCapturingAIAgent.__qualname__ = "ReceiptCapturingAIAgent"
    return ReceiptCapturingAIAgent
