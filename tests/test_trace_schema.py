import unittest

from jphrl.trajectory.schema import EpisodeTrace, JointVersion, TraceEvent


def version() -> JointVersion:
    return JointVersion(
        policy="p0",
        harness_controller="h0",
        harness_artifact="a0",
        tool_schema="t0",
        parser="parser0",
        environment="env0",
        evaluator="eval0",
        tokenizer="tok0",
        context_builder="ctx0",
    )


class TraceSchemaTests(unittest.TestCase):
    def test_mixed_version_is_rejected(self) -> None:
        trace = EpisodeTrace("episode", "task", 0, version(), "hash")
        trace.events.append(
            TraceEvent(
                index=0,
                event_id="episode:0000",
                parent_event_id=None,
                kind="tool_result",
                producer="tool",
                joint_version_id="wrong",
                payload={},
            )
        )
        with self.assertRaisesRegex(ValueError, "mixed-version"):
            trace.validate()

    def test_token_logprob_length_must_match(self) -> None:
        trace = EpisodeTrace("episode", "task", 0, version(), "hash")
        trace.append(
            "model_response",
            "policy",
            {
                "input_token_ids": [10],
                "output_token_ids": [1, 2],
                "output_token_logprobs": [-0.1],
                "completion_loss_mask": [1, 1],
                "policy_kind": "causal_lm",
                "token_metadata_status": "available",
            },
        )
        with self.assertRaisesRegex(ValueError, "different lengths"):
            trace.validate()

    def test_scripted_policy_cannot_fabricate_token_metadata(self) -> None:
        trace = EpisodeTrace("episode", "task", 0, version(), "hash")
        trace.append(
            "model_response",
            "policy",
            {
                "output_token_ids": [1],
                "output_token_logprobs": [],
                "completion_loss_mask": [],
                "policy_kind": "scripted",
                "token_metadata_status": "not_applicable",
            },
        )
        with self.assertRaisesRegex(ValueError, "must not fabricate"):
            trace.validate()

    def test_non_finite_logprob_and_non_binary_mask_are_rejected(self) -> None:
        for logprob, mask, expected_error in (
            (float("nan"), 1, "finite"),
            (-0.1, 2, "only integer 0 or 1"),
        ):
            with self.subTest(logprob=logprob, mask=mask):
                trace = EpisodeTrace("episode", "task", 0, version(), "hash")
                trace.append(
                    "model_response",
                    "policy",
                    {
                        "input_token_ids": [10],
                        "output_token_ids": [1],
                        "output_token_logprobs": [logprob],
                        "completion_loss_mask": [mask],
                        "policy_kind": "causal_lm",
                        "token_metadata_status": "available",
                    },
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    trace.validate()

    def test_harness_logprob_must_match_masked_logits(self) -> None:
        trace = EpisodeTrace("episode", "task", 0, version(), "hash")
        trace.append(
            "harness_decision",
            "harness",
            {
                "decision_id": "d0",
                "action": "DIRECT",
                "old_harness_logprob": 0.0,
                "controller_version": "h0",
                "action_ids": ["DIRECT", "VERIFY"],
                "action_mask": [True, True],
                "pre_mask_logits": [0.0, 0.0],
            },
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            trace.validate()


if __name__ == "__main__":
    unittest.main()
