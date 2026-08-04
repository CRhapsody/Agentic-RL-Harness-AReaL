from __future__ import annotations

import math
import time
import unittest

from jphrl.areal_joint_bridge_agent import _evaluate_strict_reward


class StrictRewardTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_zero_is_distinct_from_infrastructure_failure(self) -> None:
        self.assertEqual(
            await _evaluate_strict_reward(
                lambda: 0.0,
                timeout_seconds=1.0,
            ),
            0.0,
        )

        def broken() -> float:
            raise RuntimeError("verifier unavailable")

        with self.assertRaisesRegex(RuntimeError, "verifier unavailable"):
            await _evaluate_strict_reward(broken, timeout_seconds=1.0)

    async def test_timeout_and_non_finite_reward_fail_closed(self) -> None:
        def too_slow() -> float:
            time.sleep(0.05)
            return 1.0

        with self.assertRaisesRegex(RuntimeError, "timed out"):
            await _evaluate_strict_reward(too_slow, timeout_seconds=0.001)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "not finite",
            ):
                await _evaluate_strict_reward(
                    lambda: value,
                    timeout_seconds=1.0,
                )


if __name__ == "__main__":
    unittest.main()
