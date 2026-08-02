from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from .spec import HarnessAction


@dataclass(frozen=True)
class HarnessState:
    turn: int
    remaining_tool_calls: int
    remaining_model_retries: int
    context_chars: int
    last_error: str | None
    retrieval_hit: bool
    verifier_status: str
    task_domain: str


@dataclass(frozen=True)
class HarnessDecision:
    decision_id: str
    action: HarnessAction
    old_harness_logprob: float
    controller_version: str
    action_ids: tuple[str, ...]
    action_mask: tuple[bool, ...]
    pre_mask_logits: tuple[float, ...]


def _deterministic_decision(
    action: HarnessAction,
    controller_version: str,
    decision_id: str,
) -> HarnessDecision:
    action_ids = tuple(candidate.value for candidate in HarnessAction)
    action_mask = tuple(candidate is action for candidate in HarnessAction)
    return HarnessDecision(
        decision_id=decision_id,
        action=action,
        old_harness_logprob=math.log(1.0),
        controller_version=controller_version,
        action_ids=action_ids,
        action_mask=action_mask,
        pre_mask_logits=tuple(0.0 for _ in action_ids),
    )


class HarnessController(Protocol):
    version: str

    def choose(self, state: HarnessState) -> HarnessDecision:
        ...


class FixedHarnessController:
    """A deterministic B0 controller with an auditable behavior probability."""

    def __init__(
        self,
        action: HarnessAction = HarnessAction.VERIFY,
        version: str = "fixed-controller-v1",
    ) -> None:
        self.action = action
        self.version = version

    def choose(self, state: HarnessState) -> HarnessDecision:
        return _deterministic_decision(
            action=self.action,
            controller_version=self.version,
            decision_id=f"fixed-turn-{state.turn}",
        )


class SmokeHarnessController:
    """The frozen two-phase controller for the calculator contract smoke."""

    version = "smoke-controller-v1"

    def choose(self, state: HarnessState) -> HarnessDecision:
        if state.verifier_status == "not-run":
            action = HarnessAction.DIRECT
        elif state.verifier_status == "tool-ready":
            action = HarnessAction.VERIFY
        else:
            action = HarnessAction.REPLAN
        return _deterministic_decision(
            action=action,
            controller_version=self.version,
            decision_id=f"smoke-turn-{state.turn}-{action.value.lower()}",
        )
