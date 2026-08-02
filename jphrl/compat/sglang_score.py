from __future__ import annotations

import math
from typing import Any, Mapping


def parse_sglang_score_tail(
    response: Mapping[str, Any],
    target_len: int,
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

    scores: list[float] = []
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
        scores.append(logprob)
    return scores
