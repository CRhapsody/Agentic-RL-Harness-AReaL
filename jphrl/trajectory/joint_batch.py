from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from .schema import EpisodeTrace, JointVersion


class StaleJointVersionError(ValueError):
    pass


def require_lag_zero_admission(
    trace: EpisodeTrace,
    active_joint_version: JointVersion,
) -> None:
    """Reject a valid old trace when the synchronous updater requires lag=(0, 0)."""

    trace.validate()
    if trace.joint_version != active_joint_version:
        raise StaleJointVersionError(
            "trace behavior JointVersion differs from the lag-zero active version"
        )


@dataclass(frozen=True)
class DecisionCredit:
    """An explicit advantage and provenance for one trainable decision."""

    advantage: float
    source: str

    def validate(self) -> None:
        if type(self.advantage) not in (int, float) or not math.isfinite(
            float(self.advantage)
        ):
            raise ValueError("decision advantage must be finite")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("decision credit source must be a non-empty string")


@dataclass(frozen=True)
class EpisodeCredit:
    """Separate credit maps prevent call IDs and Harness IDs from being conflated."""

    policy_calls: Mapping[str, DecisionCredit]
    harness_decisions: Mapping[str, DecisionCredit]


@dataclass(frozen=True)
class PolicyTokenSample:
    episode_id: str
    model_call_id: str
    output_position: int
    token_id: int
    old_policy_logprob: float
    policy_loss_mask: int
    policy_release_id: str
    inference_engine_version: int
    advantage: float
    credit_source: str


@dataclass(frozen=True)
class HarnessActionSample:
    episode_id: str
    decision_id: str
    action: str
    action_ids: tuple[str, ...]
    action_mask: tuple[bool, ...]
    pre_mask_logits: tuple[float, ...]
    old_harness_logprob: float
    harness_loss_mask: int
    harness_behavior_version: str
    advantage: float
    credit_source: str


def _harness_logprob(sample: HarnessActionSample) -> float:
    valid_logits = [
        logit for logit, allowed in zip(sample.pre_mask_logits, sample.action_mask) if allowed
    ]
    maximum = max(valid_logits)
    log_normalizer = maximum + math.log(
        sum(math.exp(logit - maximum) for logit in valid_logits)
    )
    return sample.pre_mask_logits[sample.action_ids.index(sample.action)] - log_normalizer


@dataclass(frozen=True)
class JointDecisionBatch:
    """A decision/credit sidecar with distinct policy-token and Harness streams."""

    joint_version: JointVersion
    episode_ids: tuple[str, ...]
    policy_tokens: tuple[PolicyTokenSample, ...]
    harness_actions: tuple[HarnessActionSample, ...]

    def validate(self) -> None:
        if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("batch episode IDs must be unique and non-empty")
        episode_ids = set(self.episode_ids)
        policy_keys: set[tuple[str, str, int]] = set()
        policy_call_ids: set[tuple[str, str]] = set()
        for sample in self.policy_tokens:
            if sample.episode_id not in episode_ids:
                raise ValueError("policy token refers to an episode outside the batch")
            key = (sample.episode_id, sample.model_call_id, sample.output_position)
            if key in policy_keys:
                raise ValueError("duplicate policy token sample")
            policy_keys.add(key)
            policy_call_ids.add((sample.episode_id, sample.model_call_id))
            if type(sample.token_id) is not int or sample.token_id < 0:
                raise ValueError("policy token ID must be a non-negative integer")
            if type(sample.policy_loss_mask) is not int or sample.policy_loss_mask not in (0, 1):
                raise ValueError("policy loss mask must be integer 0 or 1")
            if not math.isfinite(sample.old_policy_logprob) or sample.old_policy_logprob > 0.0:
                raise ValueError("old policy log-prob must be finite and non-positive")
            if sample.policy_release_id != self.joint_version.policy:
                raise ValueError("policy sample release ID differs from the batch")
            if (
                type(sample.inference_engine_version) is not int
                or sample.inference_engine_version < 0
            ):
                raise ValueError("inference engine version must be a non-negative integer")
            DecisionCredit(sample.advantage, sample.credit_source).validate()

        harness_keys: set[tuple[str, str]] = set()
        for sample in self.harness_actions:
            if sample.episode_id not in episode_ids:
                raise ValueError("Harness action refers to an episode outside the batch")
            key = (sample.episode_id, sample.decision_id)
            if key in harness_keys:
                raise ValueError("duplicate Harness action sample")
            harness_keys.add(key)
            if key in policy_call_ids:
                raise ValueError("policy call ID and Harness decision ID must be disjoint")
            if type(sample.harness_loss_mask) is not int or sample.harness_loss_mask not in (0, 1):
                raise ValueError("Harness loss mask must be integer 0 or 1")
            if not sample.action_ids or not (
                len(sample.action_ids)
                == len(sample.action_mask)
                == len(sample.pre_mask_logits)
            ):
                raise ValueError("Harness action schema lengths differ")
            if len(set(sample.action_ids)) != len(sample.action_ids) or not all(
                isinstance(action_id, str) and action_id
                for action_id in sample.action_ids
            ):
                raise ValueError("Harness action IDs must be unique non-empty strings")
            if not all(type(value) is bool for value in sample.action_mask) or not any(
                sample.action_mask
            ):
                raise ValueError("Harness action mask must contain an allowed action")
            if sample.action not in sample.action_ids or not sample.action_mask[
                sample.action_ids.index(sample.action)
            ]:
                raise ValueError("chosen Harness action is absent or masked out")
            if not all(math.isfinite(value) for value in sample.pre_mask_logits):
                raise ValueError("Harness logits must be finite")
            if not math.isclose(
                sample.old_harness_logprob,
                _harness_logprob(sample),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("old Harness log-prob does not match masked logits")
            if sample.harness_behavior_version != self.joint_version.harness_controller:
                raise ValueError("Harness sample behavior version differs from the batch")
            DecisionCredit(sample.advantage, sample.credit_source).validate()

        if not self.policy_tokens:
            raise ValueError("joint batch must contain policy token samples")
        if not self.harness_actions:
            raise ValueError("joint batch must contain Harness action samples")

    @property
    def digest(self) -> str:
        self.validate()
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, object]:
        self.validate()
        return {
            "synthetic_traces": len(self.episode_ids),
            "policy_tokens": len(self.policy_tokens),
            "trainable_policy_tokens": sum(
                sample.policy_loss_mask for sample in self.policy_tokens
            ),
            "harness_actions": len(self.harness_actions),
            "trainable_harness_actions": sum(
                sample.harness_loss_mask for sample in self.harness_actions
            ),
            "policy_release_id": self.joint_version.policy,
            "inference_engine_versions": sorted(
                {sample.inference_engine_version for sample in self.policy_tokens}
            ),
            "harness_behavior_version": self.joint_version.harness_controller,
            "policy_credit_sources": sorted(
                {sample.credit_source for sample in self.policy_tokens}
            ),
            "harness_credit_sources": sorted(
                {sample.credit_source for sample in self.harness_actions}
            ),
            "digest": self.digest,
        }


def build_joint_decision_batch(
    traces: Sequence[EpisodeTrace],
    credits: Mapping[str, EpisodeCredit],
    *,
    allow_open_fixtures: bool = False,
) -> JointDecisionBatch:
    if not traces:
        raise ValueError("joint batch requires at least one trace")
    joint_version = traces[0].joint_version
    episode_ids: list[str] = []
    policy_samples: list[PolicyTokenSample] = []
    harness_samples: list[HarnessActionSample] = []

    for trace in traces:
        trace.validate()
        if not allow_open_fixtures and (
            trace.reward is None
            or trace.success is None
            or not trace.valid
            or not any(event.kind == "episode_ended" for event in trace.events)
        ):
            raise ValueError("training decision batch requires a closed valid episode")
        if trace.joint_version != joint_version:
            raise ValueError("batch contains more than one joint behavior version")
        if trace.episode_id in episode_ids:
            raise ValueError("duplicate episode ID in joint batch")
        episode_ids.append(trace.episode_id)
        if trace.episode_id not in credits:
            raise ValueError("episode has no explicit credit assignment")
        episode_credit = credits[trace.episode_id]
        seen_policy_calls: set[str] = set()
        seen_harness_decisions: set[str] = set()

        for event in trace.events:
            if event.kind == "model_response":
                if event.payload.get("token_metadata_status") == "not_applicable":
                    continue
                call_id = event.payload.get("model_call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("model response requires a non-empty model call ID")
                if call_id in seen_policy_calls:
                    raise ValueError("duplicate model call ID in episode")
                seen_policy_calls.add(call_id)
                if call_id not in episode_credit.policy_calls:
                    raise ValueError("trainable model call has no policy credit")
                credit = episode_credit.policy_calls[call_id]
                credit.validate()
                policy_release_id = event.payload.get("policy_release_id")
                output_versions = event.payload.get("output_versions")
                if policy_release_id != joint_version.policy:
                    raise ValueError(
                        "model response policy release differs from the pinned episode"
                    )
                if not isinstance(output_versions, list) or len(output_versions) != len(
                    event.payload["output_token_ids"]
                ):
                    raise ValueError(
                        "model response inference versions do not align with output tokens"
                    )
                for position, (token_id, old_logprob, loss_mask) in enumerate(
                    zip(
                        event.payload["output_token_ids"],
                        event.payload["output_token_logprobs"],
                        event.payload["completion_loss_mask"],
                    )
                ):
                    policy_samples.append(
                        PolicyTokenSample(
                            episode_id=trace.episode_id,
                            model_call_id=call_id,
                            output_position=position,
                            token_id=token_id,
                            old_policy_logprob=float(old_logprob),
                            policy_loss_mask=loss_mask,
                            policy_release_id=policy_release_id,
                            inference_engine_version=output_versions[position],
                            advantage=float(credit.advantage),
                            credit_source=credit.source,
                        )
                    )
            elif event.kind == "harness_decision":
                decision_id = event.payload["decision_id"]
                if decision_id in seen_harness_decisions:
                    raise ValueError("duplicate Harness decision ID in episode")
                seen_harness_decisions.add(decision_id)
                if decision_id not in episode_credit.harness_decisions:
                    raise ValueError("Harness decision has no Harness credit")
                credit = episode_credit.harness_decisions[decision_id]
                credit.validate()
                harness_samples.append(
                    HarnessActionSample(
                        episode_id=trace.episode_id,
                        decision_id=decision_id,
                        action=event.payload["action"],
                        action_ids=tuple(event.payload["action_ids"]),
                        action_mask=tuple(event.payload["action_mask"]),
                        pre_mask_logits=tuple(
                            float(value) for value in event.payload["pre_mask_logits"]
                        ),
                        old_harness_logprob=float(event.payload["old_harness_logprob"]),
                        harness_loss_mask=event.payload["harness_loss_mask"],
                        harness_behavior_version=event.payload["controller_version"],
                        advantage=float(credit.advantage),
                        credit_source=credit.source,
                    )
                )

        unknown_policy = set(episode_credit.policy_calls) - seen_policy_calls
        unknown_harness = set(episode_credit.harness_decisions) - seen_harness_decisions
        if unknown_policy or unknown_harness:
            raise ValueError("credit assignment refers to an absent decision")

    batch = JointDecisionBatch(
        joint_version=joint_version,
        episode_ids=tuple(episode_ids),
        policy_tokens=tuple(policy_samples),
        harness_actions=tuple(harness_samples),
    )
    batch.validate()
    return batch


JointTrainingBatch = JointDecisionBatch


def build_joint_training_batch(
    traces: Sequence[EpisodeTrace],
    credits: Mapping[str, EpisodeCredit],
    *,
    allow_open_fixtures: bool = False,
) -> JointDecisionBatch:
    """Compatibility alias for callers that have not adopted the sidecar name."""

    return build_joint_decision_batch(
        traces,
        credits,
        allow_open_fixtures=allow_open_fixtures,
    )
