from __future__ import annotations

import argparse
import json
from typing import Sequence

from ..harness.controller import HarnessState
from ..harness.learning import HarnessExperience, TabularHarnessController
from ..harness.spec import HarnessAction
from ..paths import require_within_configured_root


OPTIMAL_ACTIONS = {
    "cheap-direct": HarnessAction.DIRECT,
    "risky-verify": HarnessAction.VERIFY,
}


def _state(domain: str) -> HarnessState:
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


def run_seed(seed: int, steps: int, learning_rate: float) -> dict[str, object]:
    controller = TabularHarnessController(seed=seed)
    initial_success = sum(
        controller.probabilities(_state(domain))[optimal_action]
        for domain, optimal_action in OPTIMAL_ACTIONS.items()
    ) / len(OPTIMAL_ACTIONS)
    total_parameter_delta = 0.0
    rewards = 0.0
    decisions = 0

    for _ in range(steps):
        experiences: list[HarnessExperience] = []
        for domain, optimal_action in OPTIMAL_ACTIONS.items():
            state = _state(domain)
            decision = controller.choose(state)
            reward = 1.0 if decision.action is optimal_action else 0.0
            rewards += reward
            decisions += 1
            experiences.append(
                HarnessExperience(
                    state=state,
                    decision=decision,
                    advantage=reward,
                )
            )
        controller, stats = controller.updated(
            experiences,
            learning_rate=learning_rate,
        )
        total_parameter_delta += stats.parameter_delta_l2

    final_probabilities = {
        domain: {
            action.value: probability
            for action, probability in controller.probabilities(_state(domain)).items()
        }
        for domain in OPTIMAL_ACTIONS
    }
    final_success = sum(
        final_probabilities[domain][optimal_action.value]
        for domain, optimal_action in OPTIMAL_ACTIONS.items()
    ) / len(OPTIMAL_ACTIONS)
    return {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "behavior_version": TabularHarnessController(seed=seed).version,
        "candidate_version": controller.version,
        "random_policy_expected_success": 1.0 / len(HarnessAction),
        "initial_expected_success": initial_success,
        "training_sample_success": rewards / decisions,
        "final_expected_success": final_success,
        "optimal_action_probabilities": {
            domain: final_probabilities[domain][optimal_action.value]
            for domain, optimal_action in OPTIMAL_ACTIONS.items()
        },
        "total_parameter_delta_l2": total_parameter_delta,
        "controller_snapshot": controller.snapshot(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen-policy Harness contextual-bandit sanity"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    destination = require_within_configured_root(args.output)
    results = [
        run_seed(seed, steps=args.steps, learning_rate=args.learning_rate)
        for seed in args.seeds
    ]
    passed = all(
        min(result["optimal_action_probabilities"].values()) > 0.8
        for result in results
    )
    payload = {
        "experiment": "harness-only-contextual-bandit-v1",
        "policy_status": "frozen-not-invoked",
        "credit": "Harness action reward only",
        "success_threshold": "every optimal action probability > 0.8",
        "passed": passed,
        "results": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
