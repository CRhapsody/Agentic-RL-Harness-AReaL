from __future__ import annotations

import unittest

from jphrl.compat.sglang_score import (
    SGLangScoreTailBinding,
    parse_sglang_score_tail,
)


class SGLangScoreCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _response(*entries: tuple[float | None, int]) -> dict[str, object]:
        return {"meta_info": {"input_token_logprobs": list(entries)}}

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
        self.assertEqual(
            parse_sglang_score_tail(
                response,
                2,
                expected_token_ids=[101, 102],
            ),
            [-0.5, -1.25],
        )

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

    def test_rejects_misaligned_requested_token_ids(self) -> None:
        response = {
            "meta_info": {
                "input_token_logprobs": [
                    [None, 100],
                    [-0.5, 101],
                    [-1.25, 999],
                ]
            }
        }
        with self.assertRaisesRegex(ValueError, "do not match requested"):
            parse_sglang_score_tail(
                response,
                2,
                expected_token_ids=[101, 102],
            )

    def test_binding_accepts_consecutive_request_response_pairs(self) -> None:
        binding = SGLangScoreTailBinding()
        binding.begin([10, 11], 1)
        self.assertEqual(binding.consume(self._response((-0.5, 11)), 1), [-0.5])
        binding.begin([20, 21], 1)
        self.assertEqual(binding.consume(self._response((-0.6, 21)), 1), [-0.6])

    def test_binding_rejects_response_without_request(self) -> None:
        binding = SGLangScoreTailBinding()
        with self.assertRaisesRegex(RuntimeError, "no matching request"):
            binding.consume(self._response((-0.5, 11)), 1)

    def test_binding_recovers_after_token_mismatch(self) -> None:
        binding = SGLangScoreTailBinding()
        binding.begin([10, 11], 1)
        with self.assertRaisesRegex(ValueError, "do not match requested"):
            binding.consume(self._response((-0.5, 99)), 1)
        binding.begin([20, 21], 1)
        self.assertEqual(binding.consume(self._response((-0.6, 21)), 1), [-0.6])

    def test_binding_recovers_when_transport_failed_before_parse(self) -> None:
        binding = SGLangScoreTailBinding()
        binding.begin([10, 11], 1)
        binding.begin([20, 21], 1)
        self.assertEqual(binding.consume(self._response((-0.6, 21)), 1), [-0.6])

    def test_binding_rejects_changed_target_length_then_recovers(self) -> None:
        binding = SGLangScoreTailBinding()
        binding.begin([10, 11], 2)
        with self.assertRaisesRegex(ValueError, "count differs from target_len"):
            binding.consume(self._response((-0.5, 11)), 1)
        binding.begin([20, 21], 1)
        self.assertEqual(binding.consume(self._response((-0.6, 21)), 1), [-0.6])

    def test_rejects_invalid_actual_token_ids(self) -> None:
        for invalid_token_id in (None, -1, True, 1.5):
            with self.subTest(token_id=invalid_token_id):
                with self.assertRaisesRegex(ValueError, "invalid token ID"):
                    parse_sglang_score_tail(
                        self._response((-0.5, invalid_token_id)),
                        1,
                        expected_token_ids=[11],
                    )

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
