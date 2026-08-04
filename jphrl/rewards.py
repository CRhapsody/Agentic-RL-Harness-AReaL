from __future__ import annotations

"""Fail-closed reward functions used by formal joint-training runs."""

from typing import Any


def strict_gsm8k_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: object,
    completion_ids: object,
    answer: object,
    **kwargs: Any,
) -> float:
    """Score GSM8K without converting verifier failures into wrong answers.

    The pinned AReaL convenience reward intentionally catches every verifier
    exception and returns zero.  That is useful for broad evaluation, but it
    cannot distinguish an incorrect answer from invalid reward infrastructure.
    Formal joint training needs the distinction, so exceptions escape and the
    V2 workflow rejects the episode.  The caller owns the wall-clock timeout.

    GSM8K scoring is text-only.  Token arguments are accepted solely to retain
    the standard AReaL reward signature; they do not influence the score.
    """

    del prompt, prompt_ids, completion_ids, kwargs
    from math_verify.grader import verify
    from math_verify.parser import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        parse,
    )

    extraction = (
        ExprExtractionConfig(try_extract_without_anchor=True),
        LatexExtractionConfig(),
    )
    gold = parse(
        str(answer),
        extraction_config=extraction,
        parsing_timeout=None,
    )
    prediction = parse(
        str(completions),
        extraction_config=extraction,
        parsing_timeout=None,
    )
    if not gold or not prediction:
        return 0.0
    return float(
        bool(
            verify(
                gold,
                prediction,
                float_rounding=6,
                timeout_seconds=None,
            )
        )
    )


__all__ = ["strict_gsm8k_reward_fn"]
