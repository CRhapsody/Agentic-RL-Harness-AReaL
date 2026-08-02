from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .token_contract import validate_token_metadata


SUCCESS_EVENT_KINDS = (
    "episode_started",
    "harness_decision",
    "model_request",
    "model_response",
    "parse_result",
    "tool_result",
    "harness_decision",
    "verifier_result",
    "model_request",
    "model_response",
    "parse_result",
    "reward_assigned",
    "episode_ended",
)


@dataclass(frozen=True)
class JointVersion:
    policy: str
    harness_controller: str
    harness_artifact: str
    tool_schema: str
    parser: str
    environment: str
    evaluator: str
    tokenizer: str
    context_builder: str

    @property
    def version_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TraceEvent:
    index: int
    event_id: str
    parent_event_id: str | None
    kind: str
    producer: str
    joint_version_id: str
    payload: dict[str, Any]


@dataclass
class EpisodeTrace:
    episode_id: str
    task_id: str
    seed: int
    joint_version: JointVersion
    harness_spec_hash: str
    events: list[TraceEvent] = field(default_factory=list)
    reward: float | None = None
    success: bool | None = None
    validity_class: str = "valid"
    failure_category: str | None = None

    @property
    def valid(self) -> bool:
        return self.validity_class in {"valid", "policy_failure"}

    def append(self, kind: str, producer: str, payload: dict[str, Any]) -> None:
        event_id = f"{self.episode_id}:{len(self.events):04d}"
        self.events.append(
            TraceEvent(
                index=len(self.events),
                event_id=event_id,
                parent_event_id=self.events[-1].event_id if self.events else None,
                kind=kind,
                producer=producer,
                joint_version_id=self.joint_version.version_id,
                payload=payload,
            )
        )

    def validate(self) -> None:
        event_ids: set[str] = set()
        decision_ids: set[str] = set()
        for expected_index, event in enumerate(self.events):
            if event.index != expected_index:
                raise ValueError(
                    f"non-contiguous event index: expected {expected_index}, got {event.index}"
                )
            if event.joint_version_id != self.joint_version.version_id:
                raise ValueError("mixed-version episode detected")
            if event.parent_event_id != (
                self.events[expected_index - 1].event_id if expected_index > 0 else None
            ):
                raise ValueError("event parent does not match the preceding event")
            if event.event_id in event_ids:
                raise ValueError("duplicate event ID")
            event_ids.add(event.event_id)
            if event.kind == "harness_decision":
                decision_id = event.payload.get("decision_id")
                action = event.payload.get("action")
                action_ids = event.payload.get("action_ids")
                action_mask = event.payload.get("action_mask")
                logits = event.payload.get("pre_mask_logits")
                old_logprob = event.payload.get("old_harness_logprob")
                controller_version = event.payload.get("controller_version")
                if not isinstance(decision_id, str) or not decision_id:
                    raise ValueError("harness decision ID must be a non-empty string")
                if decision_id in decision_ids:
                    raise ValueError("duplicate harness decision ID")
                decision_ids.add(decision_id)
                if not all(isinstance(values, (list, tuple)) for values in (action_ids, action_mask, logits)):
                    raise ValueError("harness action IDs, mask, and logits must be sequences")
                if not action_ids or not (len(action_ids) == len(action_mask) == len(logits)):
                    raise ValueError("harness action IDs, mask, and logits have different lengths")
                if len(set(action_ids)) != len(action_ids) or not all(
                    isinstance(action_id, str) and action_id for action_id in action_ids
                ):
                    raise ValueError("harness action IDs must be unique non-empty strings")
                if not all(type(mask_value) is bool for mask_value in action_mask):
                    raise ValueError("harness action mask must contain booleans")
                if not any(action_mask):
                    raise ValueError("harness action mask cannot mask every action")
                if action not in action_ids or not action_mask[action_ids.index(action)]:
                    raise ValueError("chosen Harness action is absent or masked out")
                if not all(
                    type(logit) in (int, float) and math.isfinite(float(logit))
                    for logit in logits
                ):
                    raise ValueError("Harness logits must be finite numbers")
                if type(old_logprob) not in (int, float) or not math.isfinite(float(old_logprob)):
                    raise ValueError("old Harness log-prob must be finite")
                if float(old_logprob) > 0.0:
                    raise ValueError("old Harness log-prob cannot be positive")
                if controller_version != self.joint_version.harness_controller:
                    raise ValueError("Harness controller version differs from pinned episode version")
                valid_logits = [float(logit) for logit, allowed in zip(logits, action_mask) if allowed]
                maximum = max(valid_logits)
                log_normalizer = maximum + math.log(
                    sum(math.exp(logit - maximum) for logit in valid_logits)
                )
                expected_logprob = float(logits[action_ids.index(action)]) - log_normalizer
                if not math.isclose(float(old_logprob), expected_logprob, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("old Harness log-prob does not match masked logits")
            if event.kind == "model_response":
                validate_token_metadata(
                    input_token_ids=event.payload.get("input_token_ids", []),
                    output_token_ids=event.payload.get("output_token_ids", []),
                    output_token_logprobs=event.payload.get("output_token_logprobs", []),
                    completion_loss_mask=event.payload.get("completion_loss_mask", []),
                    policy_kind=event.payload.get("policy_kind"),
                    token_metadata_status=event.payload.get("token_metadata_status"),
                )
        if self.validity_class not in {
            "valid",
            "policy_failure",
            "infrastructure_invalid",
            "trace_contract_invalid",
        }:
            raise ValueError(f"unknown validity class: {self.validity_class}")
        if self.success is True and self.reward != 1.0:
            raise ValueError("successful smoke episode must have reward 1.0")
        if self.success is False and self.reward not in (0.0, None):
            raise ValueError("failed smoke episode cannot have a positive reward")
        if self.success is True and self.validity_class != "valid":
            raise ValueError("successful episode must have validity_class=valid")
        if self.success is False and self.validity_class == "valid":
            raise ValueError("failed episode cannot have validity_class=valid")
        if self.validity_class == "policy_failure" and (
            self.success is not False or self.reward != 0.0
        ):
            raise ValueError("policy_failure must be a failed zero-reward episode")
        if self.validity_class in {"infrastructure_invalid", "trace_contract_invalid"} and (
            self.success is not False or self.reward is not None
        ):
            raise ValueError("invalid episode must fail with reward=None")
        if self.success is True:
            kinds = tuple(event.kind for event in self.events)
            if kinds != SUCCESS_EVENT_KINDS:
                raise ValueError("successful smoke episode does not match the frozen 13-event contract")
        if any(event.kind == "episode_ended" for event in self.events) and self.success is None:
            raise ValueError("ended episode must define success and reward semantics")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
