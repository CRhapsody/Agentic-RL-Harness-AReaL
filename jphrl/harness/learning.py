from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Mapping, Sequence

from .controller import HarnessDecision, HarnessState
from .spec import HarnessAction


_ACTION_IDS = tuple(action.value for action in HarnessAction)


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _state_key(state: HarnessState) -> str:
    """Encode only observable Harness state, never task answers or rewards."""

    payload = {
        "context_chars": state.context_chars,
        "last_error": state.last_error,
        "remaining_model_retries": state.remaining_model_retries,
        "remaining_tool_calls": state.remaining_tool_calls,
        "retrieval_hit": state.retrieval_hit,
        "task_domain": state.task_domain,
        "turn": state.turn,
        "verifier_status": state.verifier_status,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _log_softmax(logits: Sequence[float], mask: Sequence[bool]) -> tuple[float, ...]:
    if len(logits) != len(mask) or not any(mask):
        raise ValueError("logits and a non-empty action mask must have equal length")
    allowed_logits = [float(logit) for logit, allowed in zip(logits, mask) if allowed]
    if not all(math.isfinite(logit) for logit in allowed_logits):
        raise ValueError("allowed Harness logits must be finite")
    maximum = max(allowed_logits)
    log_normalizer = maximum + math.log(
        sum(math.exp(logit - maximum) for logit in allowed_logits)
    )
    return tuple(
        float(logit) - log_normalizer if allowed else -math.inf
        for logit, allowed in zip(logits, mask)
    )


@dataclass(frozen=True)
class HarnessExperience:
    """One Harness action with its own behavior probability and credit."""

    state: HarnessState
    decision: HarnessDecision
    advantage: float


@dataclass(frozen=True)
class HarnessUpdateStats:
    behavior_version: str
    candidate_version: str
    batch_size: int
    mean_advantage: float
    parameter_delta_l2: float


class TabularHarnessController:
    """Small categorical policy used to validate Harness-side policy gradients.

    The production controller will be a batched torch module.  This dependency-free
    implementation deliberately comes first: it proves that Harness actions have
    independent behavior probabilities, credit, versioning, and a non-zero update
    before those semantics are connected to the distributed AReaL path.

    ``updated`` never mutates the behavior snapshot.  It returns a new controller
    version, so trajectories collected by the old object remain reproducible.
    """

    schema_version = "tabular-categorical-v1"

    def __init__(
        self,
        *,
        seed: int = 0,
        step: int = 0,
        logits_by_state: Mapping[str, Sequence[float]] | None = None,
        rng_state: object | None = None,
        sample_count: int = 0,
    ) -> None:
        if step < 0 or sample_count < 0:
            raise ValueError("step and sample_count must be non-negative")
        self.seed = seed
        self.step = step
        self._sample_count = sample_count
        self._rng = random.Random(seed)
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self._logits_by_state: dict[str, tuple[float, ...]] = {}
        for key, values in (logits_by_state or {}).items():
            logits = tuple(float(value) for value in values)
            if len(logits) != len(_ACTION_IDS):
                raise ValueError("every state must define one logit per Harness action")
            if not all(math.isfinite(logit) for logit in logits):
                raise ValueError("Harness logits must be finite")
            self._logits_by_state[str(key)] = logits

    @property
    def version(self) -> str:
        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "seed": self.seed,
                "step": self.step,
                "logits_by_state": self._logits_by_state,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        parameter_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return (
            f"{self.schema_version}-s{self.seed}-step{self.step:06d}"
            f"-{parameter_hash}"
        )

    def _logits(self, state: HarnessState) -> tuple[float, ...]:
        return self._logits_by_state.get(_state_key(state), (0.0,) * len(_ACTION_IDS))

    def probabilities(self, state: HarnessState) -> dict[HarnessAction, float]:
        logprobs = _log_softmax(self._logits(state), (True,) * len(_ACTION_IDS))
        return {
            action: math.exp(logprob)
            for action, logprob in zip(HarnessAction, logprobs)
        }

    def choose(self, state: HarnessState) -> HarnessDecision:
        logits = self._logits(state)
        action_mask = (True,) * len(_ACTION_IDS)
        logprobs = _log_softmax(logits, action_mask)
        draw = self._rng.random()
        cumulative = 0.0
        chosen_index = len(_ACTION_IDS) - 1
        for index, logprob in enumerate(logprobs):
            cumulative += math.exp(logprob)
            if draw <= cumulative:
                chosen_index = index
                break
        state_digest = hashlib.sha256(_state_key(state).encode("utf-8")).hexdigest()[:10]
        decision = HarnessDecision(
            decision_id=(
                f"tabular-step-{self.step}-sample-{self._sample_count}-{state_digest}"
            ),
            action=HarnessAction(_ACTION_IDS[chosen_index]),
            old_harness_logprob=logprobs[chosen_index],
            controller_version=self.version,
            action_ids=_ACTION_IDS,
            action_mask=action_mask,
            pre_mask_logits=logits,
            harness_loss_mask=1,
        )
        self._sample_count += 1
        return decision

    def updated(
        self,
        experiences: Sequence[HarnessExperience],
        *,
        learning_rate: float,
    ) -> tuple[TabularHarnessController, HarnessUpdateStats]:
        """Apply one on-behavior REINFORCE step and return an immutable candidate."""

        if not experiences:
            raise ValueError("Harness update requires at least one experience")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")

        gradients: dict[str, list[float]] = {}
        advantage_sum = 0.0
        for experience in experiences:
            decision = experience.decision
            if decision.controller_version != self.version:
                raise ValueError("experience was not produced by this behavior version")
            if not math.isfinite(experience.advantage):
                raise ValueError("Harness advantage must be finite")
            if decision.action_ids != _ACTION_IDS:
                raise ValueError("experience action IDs differ from controller schema")
            if decision.action_mask != (True,) * len(_ACTION_IDS):
                raise ValueError("experience action mask differs from controller schema")

            state_key = _state_key(experience.state)
            behavior_logits = self._logits(experience.state)
            if any(
                not math.isclose(actual, recorded, rel_tol=0.0, abs_tol=1e-12)
                for actual, recorded in zip(
                    behavior_logits, decision.pre_mask_logits
                )
            ):
                raise ValueError("recorded logits differ from behavior snapshot")
            logprobs = _log_softmax(behavior_logits, decision.action_mask)
            action_index = _ACTION_IDS.index(decision.action.value)
            if not math.isclose(
                logprobs[action_index],
                decision.old_harness_logprob,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("recorded old log-prob differs from behavior snapshot")

            state_gradient = gradients.setdefault(
                state_key, [0.0] * len(_ACTION_IDS)
            )
            for index, (logprob, allowed) in enumerate(
                zip(logprobs, decision.action_mask)
            ):
                if allowed:
                    probability = math.exp(logprob)
                    indicator = 1.0 if index == action_index else 0.0
                    state_gradient[index] += experience.advantage * (
                        indicator - probability
                    )
            advantage_sum += experience.advantage

        scale = learning_rate / len(experiences)
        new_logits_by_state = dict(self._logits_by_state)
        squared_delta_sum = 0.0
        for state_key, gradient in gradients.items():
            old_logits = new_logits_by_state.get(
                state_key, (0.0,) * len(_ACTION_IDS)
            )
            new_logits = tuple(
                old_logit + scale * gradient_value
                for old_logit, gradient_value in zip(old_logits, gradient)
            )
            squared_delta_sum += sum(
                (new_logit - old_logit) ** 2
                for old_logit, new_logit in zip(old_logits, new_logits)
            )
            new_logits_by_state[state_key] = new_logits

        candidate = TabularHarnessController(
            seed=self.seed,
            step=self.step + 1,
            logits_by_state=new_logits_by_state,
            rng_state=self._rng.getstate(),
            sample_count=self._sample_count,
        )
        stats = HarnessUpdateStats(
            behavior_version=self.version,
            candidate_version=candidate.version,
            batch_size=len(experiences),
            mean_advantage=advantage_sum / len(experiences),
            parameter_delta_l2=math.sqrt(squared_delta_sum),
        )
        return candidate, stats

    def snapshot(self) -> dict[str, object]:
        """Return a small parameter-only snapshot for reporting, not restoration."""

        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "step": self.step,
            "version": self.version,
            "logits_by_state": {
                key: list(values) for key, values in sorted(self._logits_by_state.items())
            },
        }

    def checkpoint(self) -> dict[str, object]:
        """Return all state required to reproduce the next sampled decision."""

        payload = self.snapshot()
        payload.update(
            {
                "rng_state": self._rng.getstate(),
                "sample_count": self._sample_count,
            }
        )
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> TabularHarnessController:
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError("Harness checkpoint schema version differs")
        logits_payload = payload.get("logits_by_state")
        if not isinstance(logits_payload, Mapping):
            raise ValueError("Harness checkpoint logits must be an object")
        controller = cls(
            seed=int(payload["seed"]),
            step=int(payload["step"]),
            logits_by_state={
                str(key): tuple(float(value) for value in values)
                for key, values in logits_payload.items()
            },
            rng_state=_nested_tuple(payload["rng_state"]),
            sample_count=int(payload["sample_count"]),
        )
        if payload.get("version") != controller.version:
            raise ValueError("Harness checkpoint version differs from restored parameters")
        return controller
