import math
from dataclasses import replace
import unittest

from jphrl.harness.controller import HarnessState
from jphrl.harness.learning import HarnessExperience, TabularHarnessController
from jphrl.harness.spec import HarnessAction


def state_for(domain: str) -> HarnessState:
    return HarnessState(
        turn=0,
        remaining_tool_calls=1,
        remaining_model_retries=1,
        context_chars=128,
        last_error=None,
        retrieval_hit=False,
        verifier_status="decision",
        task_domain=domain,
    )


class HarnessLearningTests(unittest.TestCase):
    def test_contextual_policy_gradient_learns_distinct_actions(self) -> None:
        controller = TabularHarnessController(seed=7)
        optimal_actions = {
            "cheap-direct": HarnessAction.DIRECT,
            "risky-verify": HarnessAction.VERIFY,
        }
        total_parameter_delta = 0.0

        for _ in range(400):
            experiences: list[HarnessExperience] = []
            for domain, optimal_action in optimal_actions.items():
                state = state_for(domain)
                decision = controller.choose(state)
                reward = 1.0 if decision.action is optimal_action else 0.0
                experiences.append(HarnessExperience(state, decision, reward))
            controller, stats = controller.updated(
                experiences,
                learning_rate=0.4,
            )
            total_parameter_delta += stats.parameter_delta_l2

        for domain, optimal_action in optimal_actions.items():
            probability = controller.probabilities(state_for(domain))[optimal_action]
            self.assertGreater(probability, 0.8)
        self.assertGreater(total_parameter_delta, 0.0)
        self.assertEqual(controller.step, 400)

    def test_old_logprob_is_auditable_from_behavior_logits(self) -> None:
        controller = TabularHarnessController(seed=11)
        state = state_for("audit")
        decision = controller.choose(state)
        allowed_logits = [
            logit
            for logit, allowed in zip(
                decision.pre_mask_logits, decision.action_mask
            )
            if allowed
        ]
        maximum = max(allowed_logits)
        log_normalizer = maximum + math.log(
            sum(math.exp(logit - maximum) for logit in allowed_logits)
        )
        action_index = decision.action_ids.index(decision.action.value)
        expected = decision.pre_mask_logits[action_index] - log_normalizer
        self.assertAlmostEqual(decision.old_harness_logprob, expected, places=12)

    def test_update_rejects_experience_from_an_old_controller_version(self) -> None:
        behavior = TabularHarnessController(seed=3)
        state = state_for("stale")
        decision = behavior.choose(state)
        experience = HarnessExperience(state, decision, advantage=1.0)
        candidate, _ = behavior.updated([experience], learning_rate=0.1)

        with self.assertRaisesRegex(ValueError, "behavior version"):
            candidate.updated([experience], learning_rate=0.1)

    def test_zero_or_invalid_update_configuration_fails_closed(self) -> None:
        controller = TabularHarnessController(seed=5)
        with self.assertRaisesRegex(ValueError, "at least one"):
            controller.updated([], learning_rate=0.1)
        state = state_for("invalid")
        decision = controller.choose(state)
        for learning_rate in (0.0, float("nan")):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaisesRegex(ValueError, "learning_rate"):
                    controller.updated(
                        [HarnessExperience(state, decision, advantage=1.0)],
                        learning_rate=learning_rate,
                    )

    def test_zero_harness_loss_mask_has_no_update_effect(self) -> None:
        controller = TabularHarnessController(seed=23)
        state = state_for("masked")
        decision = replace(controller.choose(state), harness_loss_mask=0)
        candidate, stats = controller.updated(
            [HarnessExperience(state, decision, advantage=10.0)],
            learning_rate=0.4,
        )
        self.assertIs(candidate, controller)
        self.assertEqual(stats.behavior_version, stats.candidate_version)
        self.assertEqual(stats.effective_batch_size, 0)
        self.assertEqual(stats.parameter_delta_l2, 0.0)


if __name__ == "__main__":
    unittest.main()
