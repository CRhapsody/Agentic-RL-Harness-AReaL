from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import patch

from jphrl.training.online_admission import (
    LagAdmissionDecision,
    OnlineAdmissionError,
    ReleaseLease,
    admit_lag_zero_training_record,
    decide_lag_zero_admission,
    revalidate_before_training,
    validate_resample_request,
    validate_resampled_identity,
)
from jphrl.trajectory.schema import EpisodeTrace, JointVersion


def _version(policy: str = "policy-1", harness: str = "harness-1") -> JointVersion:
    return JointVersion(
        policy=policy,
        harness_controller=harness,
        harness_artifact="harness-artifact-1",
        tool_schema="tools-1",
        parser="parser-1",
        environment="env-1",
        evaluator="evaluator-1",
        tokenizer="tokenizer-1",
        context_builder="context-1",
    )


def _lease(
    release: str = "release-1",
    *,
    policy_generation: int = 1,
    harness_generation: int = 1,
    macro_step: int = 1,
    version: JointVersion | None = None,
) -> ReleaseLease:
    return ReleaseLease(
        release_id=release,
        joint_version=version or _version(),
        policy_generation=policy_generation,
        harness_generation=harness_generation,
        macro_step=macro_step,
    )


def _trace(version: JointVersion | None = None) -> EpisodeTrace:
    return EpisodeTrace(
        episode_id="episode-old",
        task_id="task-17-plus-25",
        seed=73,
        joint_version=version or _version(),
        harness_spec_hash="harness-spec-hash",
    )


def _s_record(version: JointVersion | None = None) -> dict[str, object]:
    joint_version = version or _version()
    return {
        "record_sha256": "a" * 64,
        "joint_version": asdict(joint_version),
        "identity": {
            "episode_id": "episode-old",
            "trace_sha256": "b" * 64,
            "model_call_ids": ["call-old-0", "call-old-1"],
        },
        "admissions": {
            "policy_admission_record": {
                "identity": {
                    "task_id": "task-17-plus-25",
                    "session_id": "session-old",
                }
            }
        },
    }


def _s_audit(version: JointVersion | None = None) -> dict[str, object]:
    joint_version = version or _version()
    return {
        "episode_id": "episode-old",
        "joint_version_id": joint_version.version_id,
    }


def _trace_audit(version: JointVersion | None = None) -> dict[str, object]:
    joint_version = version or _version()
    return {
        "episode_id": "episode-old",
        "task_id": "task-17-plus-25",
        "joint_version_id": joint_version.version_id,
        "trace_sha256": "b" * 64,
        "model_call_ids": ["call-old-0", "call-old-1"],
    }


class ReleaseLeaseAdmissionTests(unittest.TestCase):
    def test_accepts_only_the_exact_active_lease(self) -> None:
        lease = _lease()
        decision = decide_lag_zero_admission(lease, lease)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "accepted")
        self.assertEqual((decision.policy_lag, decision.harness_lag), (0, 0))
        round_trip = LagAdmissionDecision.from_record(decision.to_record())
        self.assertEqual(round_trip, decision)

    def test_distinguishes_policy_harness_and_both_stale(self) -> None:
        cases = (
            (
                _lease(),
                _lease(
                    "release-2",
                    policy_generation=2,
                    macro_step=2,
                    version=_version(policy="policy-2"),
                ),
                "policy_stale",
                (1, 0),
            ),
            (
                _lease(),
                _lease(
                    "release-2",
                    harness_generation=2,
                    macro_step=2,
                    version=_version(harness="harness-2"),
                ),
                "harness_stale",
                (0, 1),
            ),
            (
                _lease(),
                _lease(
                    "release-2",
                    policy_generation=3,
                    harness_generation=2,
                    macro_step=2,
                    version=_version(policy="policy-3", harness="harness-2"),
                ),
                "both_stale",
                (2, 1),
            ),
        )
        for behavior, active, reason, lag in cases:
            with self.subTest(reason=reason):
                decision = decide_lag_zero_admission(behavior, active)
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, reason)
                self.assertEqual((decision.policy_lag, decision.harness_lag), lag)

    def test_distinguishes_non_axis_contract_mismatch(self) -> None:
        changed_versions = (
            replace(_version(), parser="parser-2"),
            replace(_version(), harness_artifact="harness-contract-2"),
        )
        for changed in changed_versions:
            with self.subTest(changed=changed):
                decision = decide_lag_zero_admission(
                    _lease(),
                    _lease("release-2", macro_step=2, version=changed),
                )
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, "contract_mismatch")
                self.assertEqual(
                    (decision.policy_lag, decision.harness_lag), (None, None)
                )

    def test_detects_publish_straddles(self) -> None:
        cases = (
            (
                _lease("release-1"),
                _lease("release-2", macro_step=1),
            ),
            (
                _lease("release-1"),
                _lease(
                    "release-1",
                    policy_generation=2,
                    macro_step=2,
                    version=_version(policy="policy-2"),
                ),
            ),
            (
                _lease(policy_generation=2),
                _lease(policy_generation=1),
            ),
            (
                _lease(),
                _lease(policy_generation=2, macro_step=2),
            ),
        )
        for behavior, active in cases:
            with self.subTest(behavior=behavior, active=active):
                decision = decide_lag_zero_admission(behavior, active)
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, "publish_straddle")

    def test_lease_and_decision_schema_hash_fail_closed(self) -> None:
        lease_record = _lease().to_record()
        lease_record["macro_step"] = 9
        with self.assertRaisesRegex(OnlineAdmissionError, "hash mismatch"):
            ReleaseLease.from_record(lease_record)

        decision_record = decide_lag_zero_admission(_lease(), _lease()).to_record()
        decision_record["evidence_scope"]["policy_optimizer_update"] = True
        payload = {
            key: value
            for key, value in decision_record.items()
            if key != "record_sha256"
        }
        decision_record["record_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(OnlineAdmissionError, "evidence scope"):
            LagAdmissionDecision.from_record(decision_record)

        inconsistent = decide_lag_zero_admission(_lease(), _lease()).to_record()
        inconsistent["active_lease"] = _lease(
            "release-2",
            policy_generation=2,
            macro_step=2,
            version=_version(policy="policy-2"),
        ).to_record()
        unsigned = {
            key: value for key, value in inconsistent.items() if key != "record_sha256"
        }
        inconsistent["record_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(OnlineAdmissionError, "differs from its release"):
            LagAdmissionDecision.from_record(inconsistent)


class TrainingAdmissionAndResampleTests(unittest.TestCase):
    def _admit(
        self,
        behavior: ReleaseLease,
        active: ReleaseLease,
        *,
        record: dict[str, object] | None = None,
        trace: EpisodeTrace | None = None,
        source_input: object = None,
    ) -> LagAdmissionDecision:
        source_input = (
            {"question": "What is 17 + 25?"} if source_input is None else source_input
        )
        with (
            patch(
                "jphrl.training.online_admission.validate_frozen_joint_credit_alignment",
                return_value=_s_audit(behavior.joint_version),
            ),
            patch(
                "jphrl.training.online_admission.validate_agent_service_training_trace",
                return_value=_trace_audit(behavior.joint_version),
            ),
        ):
            return admit_lag_zero_training_record(
                s_record=record or _s_record(behavior.joint_version),
                source_episode=trace or _trace(behavior.joint_version),
                source_input=source_input,
                behavior_lease=behavior,
                active_lease=active,
            )

    def test_valid_stale_episode_is_rejected_with_semantic_resample(self) -> None:
        behavior = _lease()
        active = _lease(
            "release-2",
            policy_generation=2,
            macro_step=2,
            version=_version(policy="policy-2"),
        )
        decision = self._admit(behavior, active)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "policy_stale")
        request = decision.resample_request
        self.assertIsNotNone(request)
        audit = validate_resample_request(request)
        self.assertEqual(audit["task_id"], "task-17-plus-25")
        self.assertEqual(audit["seed"], 73)
        self.assertEqual(request["task"]["input"], {"question": "What is 17 + 25?"})
        self.assertEqual(request["target_lease"], active.to_record())
        self.assertFalse(request["evidence_scope"]["policy_optimizer_update"])
        self.assertFalse(request["evidence_scope"]["harness_optimizer_update"])

    def test_resample_requires_all_fresh_mutually_distinct_ids(self) -> None:
        active = _lease(
            "release-2",
            harness_generation=2,
            macro_step=2,
            version=_version(harness="harness-2"),
        )
        request = self._admit(_lease(), active).resample_request
        validate_resampled_identity(
            request,
            episode_id="episode-new",
            session_id="session-new",
            model_call_ids=("call-new-0", "call-new-1"),
        )
        bad_cases = (
            ("episode-old", "session-new", ("call-new",)),
            ("episode-new", "session-old", ("call-new",)),
            ("episode-new", "session-new", ("call-old-0",)),
            ("episode-new", "session-new", ("call-new", "call-new")),
            ("same", "same", ("call-new",)),
        )
        for episode_id, session_id, call_ids in bad_cases:
            with (
                self.subTest(
                    episode_id=episode_id, session_id=session_id, call_ids=call_ids
                ),
                self.assertRaises(OnlineAdmissionError),
            ):
                validate_resampled_identity(
                    request,
                    episode_id=episode_id,
                    session_id=session_id,
                    model_call_ids=call_ids,
                )

    def test_second_gate_revalidates_and_rejects_release_advance(self) -> None:
        behavior = _lease()
        initial = self._admit(behavior, behavior)
        self.assertTrue(initial.accepted)
        advanced = _lease(
            "release-2",
            policy_generation=2,
            harness_generation=2,
            macro_step=2,
            version=_version(policy="policy-2", harness="harness-2"),
        )
        record = _s_record()
        trace = _trace()
        with (
            patch(
                "jphrl.training.online_admission.validate_frozen_joint_credit_alignment",
                return_value=_s_audit(),
            ) as s_validator,
            patch(
                "jphrl.training.online_admission.validate_agent_service_training_trace",
                return_value=_trace_audit(),
            ) as trace_validator,
        ):
            second = revalidate_before_training(
                initial,
                s_record=record,
                source_episode=trace,
                source_input={"question": "What is 17 + 25?"},
                active_lease=advanced,
            )
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, "both_stale")
        self.assertIsNotNone(second.resample_request)
        self.assertGreaterEqual(s_validator.call_count, 1)
        self.assertGreaterEqual(trace_validator.call_count, 1)

    def test_full_joint_version_source_episode_and_input_fail_closed(self) -> None:
        lease = _lease()
        bad_record = _s_record()
        bad_record["joint_version"]["tokenizer"] = "other-tokenizer"
        with self.assertRaisesRegex(OnlineAdmissionError, "full JointVersion"):
            self._admit(lease, lease, record=bad_record)

        bad_episode = _trace(replace(_version(), tokenizer="other-tokenizer"))
        with self.assertRaisesRegex(OnlineAdmissionError, "full JointVersion"):
            self._admit(lease, lease, trace=bad_episode)

        with self.assertRaisesRegex(OnlineAdmissionError, "credential field"):
            self._admit(
                lease,
                lease,
                source_input={"session_api_key": "must-not-enter"},
            )

        secret_episode = _trace()
        secret_episode.append("diagnostic", "test", {"admin_api_key": "secret"})
        with self.assertRaisesRegex(OnlineAdmissionError, "credential field"):
            self._admit(lease, lease, trace=secret_episode)

    def test_source_episode_hash_task_and_model_calls_are_rechecked(self) -> None:
        behavior = _lease()
        record = _s_record()
        with (
            patch(
                "jphrl.training.online_admission.validate_frozen_joint_credit_alignment",
                return_value=_s_audit(),
            ),
            patch(
                "jphrl.training.online_admission.validate_agent_service_training_trace",
                return_value={**_trace_audit(), "trace_sha256": "c" * 64},
            ),
            self.assertRaisesRegex(OnlineAdmissionError, "episode hash"),
        ):
            admit_lag_zero_training_record(
                s_record=record,
                source_episode=_trace(),
                source_input={"question": "What is 17 + 25?"},
                behavior_lease=behavior,
                active_lease=behavior,
            )

    def test_resample_hash_and_credentials_fail_closed(self) -> None:
        active = _lease(
            "release-2",
            policy_generation=2,
            macro_step=2,
            version=_version(policy="policy-2"),
        )
        request = deepcopy(self._admit(_lease(), active).resample_request)
        request["task"]["input"]["question"] = "changed"
        with self.assertRaisesRegex(OnlineAdmissionError, "hash mismatch"):
            validate_resample_request(request)

        request = deepcopy(self._admit(_lease(), active).resample_request)
        request["task"]["input"]["admin_api_key"] = "secret"
        unsigned = {
            key: value for key, value in request.items() if key != "record_sha256"
        }
        request["record_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(OnlineAdmissionError, "credential field"):
            validate_resample_request(request)


if __name__ == "__main__":
    unittest.main()
