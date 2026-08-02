from __future__ import annotations

import unittest

from jphrl.compat.sglang_score import parse_sglang_score_tail


class SGLangScoreCompatibilityTests(unittest.TestCase):
    def test_ignores_none_only_before_requested_tail(self) -> None:
        response = {
            "meta_info": {
                "input_token_logprobs": [
                    [None, 100],
                    [-0.5, 101],
                    [-1.25, 102],
                ]
            }
        }
        self.assertEqual(parse_sglang_score_tail(response, 2), [-0.5, -1.25])

    def test_rejects_none_inside_requested_tail(self) -> None:
        response = {
            "meta_info": {
                "input_token_logprobs": [
                    [-0.5, 100],
                    [None, 101],
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, "tail contains a None"):
            parse_sglang_score_tail(response, 2)

    def test_rejects_short_or_non_finite_score_tail(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient"):
            parse_sglang_score_tail(
                {"meta_info": {"input_token_logprobs": [[-0.5, 100]]}},
                2,
            )
        with self.assertRaisesRegex(ValueError, "invalid log-prob"):
            parse_sglang_score_tail(
                {"meta_info": {"input_token_logprobs": [[float("nan"), 100]]}},
                1,
            )
