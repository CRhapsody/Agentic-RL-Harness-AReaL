from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class SGLangScoreTailBinding:
    """Bind one synchronous SGLang score response to its requested token tail."""

    def __init__(self) -> None:
        self._expected_token_ids: tuple[int, ...] | None = None

    def begin(self, input_ids: Sequence[int], target_len: int) -> None:
        if target_len <= 0 or target_len > len(input_ids):
            raise ValueError("target_len must select a non-empty input token tail")
        # AReaL v2.0.0 performs build -> synchronous HTTP -> parse serially per
        # engine. Overwrite an unconsumed binding so a transport/JSON failure
        # cannot permanently poison the backend before the next synchronous call.
        self._expected_token_ids = tuple(input_ids[-target_len:])

    def consume(
        self,
        response: Mapping[str, Any],
        target_len: int,
    ) -> list[float]:
        expected = self._expected_token_ids
        if expected is None:
            raise RuntimeError("SGLang score response has no matching request")
        try:
            return parse_sglang_score_tail(
                response,
                target_len,
                expected_token_ids=expected,
            )
        finally:
            self._expected_token_ids = None


def parse_sglang_score_tail(
    response: Mapping[str, Any],
    target_len: int,
    *,
    expected_token_ids: Sequence[int] | None = None,
) -> list[float]:
    """Parse only SGLang's requested score tail before converting log-probs.

    Some SGLang versions put a ``None`` log-prob on an earlier context entry.
    AReaL v2.0.0 converts every entry before taking the requested tail, so that
    harmless prefix sentinel aborts scoring. This helper preserves AReaL's
    intended ``entries[-target_len:]`` semantics and still rejects ``None`` or
    malformed values inside the requested tail.
    """

    if (
        not isinstance(target_len, int)
        or isinstance(target_len, bool)
        or target_len <= 0
    ):
        raise ValueError("target_len must be a positive integer")
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, Mapping):
        raise ValueError("SGLang response missing meta_info for score request")
    entries = meta_info.get("input_token_logprobs")
    if not isinstance(entries, list):
        raise ValueError("SGLang input_token_logprobs must be a list")
    if len(entries) < target_len:
        raise ValueError(
            "SGLang returned insufficient input_token_logprobs: "
            f"{len(entries)} < {target_len}"
        )
    if expected_token_ids is not None and len(expected_token_ids) != target_len:
        raise ValueError("expected score token ID count differs from target_len")

    scores: list[float] = []
    actual_token_ids: list[int] = []
    for entry in entries[-target_len:]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError("SGLang score entry must contain log-prob and token ID")
        raw_logprob = entry[0]
        if raw_logprob is None:
            raise ValueError("SGLang requested score tail contains a None log-prob")
        try:
            logprob = float(raw_logprob)
        except (TypeError, ValueError) as exc:
            raise ValueError("SGLang score tail contains a non-numeric log-prob") from exc
        if not math.isfinite(logprob) or logprob > 1e-5:
            raise ValueError("SGLang score tail contains an invalid log-prob")
        token_id = entry[1]
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("SGLang score tail contains an invalid token ID")
        scores.append(logprob)
        actual_token_ids.append(token_id)
    if expected_token_ids is not None and actual_token_ids != list(expected_token_ids):
        raise ValueError("SGLang score tail token IDs do not match requested tokens")
    return scores
