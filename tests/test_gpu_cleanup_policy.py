from __future__ import annotations

import unittest

from jphrl.training.gpu_cleanup_policy import (
    CONFIRMED_OOM_FAILURE_CLASS,
    GpuCleanupCandidate,
    plan_lyy_oom_cleanup_candidates,
)


def _process(
    pid: int,
    *,
    owner_user: str = "lyy",
    owner_uid: int = 2345,
    process_start_time: int = 987654,
) -> dict[str, object]:
    return {
        "pid": pid,
        "process_start_time": process_start_time,
        "owner_user": owner_user,
        "owner_uid": owner_uid,
    }


class GpuCleanupPolicyTests(unittest.TestCase):
    def test_confirmed_oom_returns_only_exact_lyy_user_and_uid(self) -> None:
        candidates = plan_lyy_oom_cleanup_candidates(
            failure_class=CONFIRMED_OOM_FAILURE_CLASS,
            expected_lyy_uid=2345,
            observed_processes=(
                _process(101),
                _process(102, owner_user="other", owner_uid=3456),
                _process(103, owner_user="lyy", owner_uid=3456),
                _process(104, owner_user="other", owner_uid=2345),
                _process(105, process_start_time=123456),
            ),
        )

        self.assertEqual(
            candidates,
            (
                GpuCleanupCandidate(101, 987654, "lyy", 2345),
                GpuCleanupCandidate(105, 123456, "lyy", 2345),
            ),
        )

    def test_non_oom_and_every_other_owner_produce_no_candidate(self) -> None:
        observations = (
            _process(201, owner_user="root", owner_uid=0),
            _process(202, owner_user="ljw", owner_uid=1000),
        )
        for failure_class in (None, "", "insufficient-free-memory", "timeout"):
            with self.subTest(failure_class=failure_class):
                self.assertEqual(
                    plan_lyy_oom_cleanup_candidates(
                        failure_class=failure_class,
                        expected_lyy_uid=2345,
                        observed_processes=observations,
                    ),
                    (),
                )
        self.assertEqual(
            plan_lyy_oom_cleanup_candidates(
                failure_class=CONFIRMED_OOM_FAILURE_CLASS,
                expected_lyy_uid=2345,
                observed_processes=observations,
            ),
            (),
        )

    def test_malformed_duplicate_or_unbound_identity_fails_closed(self) -> None:
        cases = (
            (-1, (_process(301),)),
            (2345, ({"pid": 301},)),
            (2345, (_process(301), _process(301))),
            (2345, (_process(1),)),
            (2345, (_process(301, process_start_time=0),)),
            (2345, "not-an-observation-sequence"),
        )
        for expected_uid, observations in cases:
            with self.subTest(expected_uid=expected_uid, observations=observations):
                self.assertEqual(
                    plan_lyy_oom_cleanup_candidates(
                        failure_class=CONFIRMED_OOM_FAILURE_CLASS,
                        expected_lyy_uid=expected_uid,
                        observed_processes=observations,
                    ),
                    (),
                )


if __name__ == "__main__":
    unittest.main()
