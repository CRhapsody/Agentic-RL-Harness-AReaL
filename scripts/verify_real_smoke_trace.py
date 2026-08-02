from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jphrl.trajectory.schema import EpisodeTrace, JointVersion, SUCCESS_EVENT_KINDS, TraceEvent
from jphrl.trajectory.token_contract import validate_token_metadata


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_trace(path: Path) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    _require(trace.get("success") is True, "real-model smoke did not succeed")
    _require(trace.get("reward") == 1.0, "real-model smoke reward is not 1.0")
    _require(trace.get("validity_class") == "valid", "real-model trace is not valid")

    events = trace.get("events")
    _require(isinstance(events, list), "trace events must be a list")
    _require(
        tuple(event.get("kind") for event in events) == SUCCESS_EVENT_KINDS,
        "real-model trace does not match the frozen 13-event contract",
    )
    joint_version = JointVersion(**trace["joint_version"])
    joint_version_id = joint_version.version_id
    _require(isinstance(joint_version_id, str) and joint_version_id, "missing joint version ID")

    reconstructed = EpisodeTrace(
        episode_id=trace["episode_id"],
        task_id=trace["task_id"],
        seed=trace["seed"],
        joint_version=joint_version,
        harness_spec_hash=trace["harness_spec_hash"],
        events=[TraceEvent(**event) for event in events],
        reward=trace["reward"],
        success=trace["success"],
        validity_class=trace["validity_class"],
        failure_category=trace["failure_category"],
    )
    reconstructed.validate()

    for index, event in enumerate(events):
        _require(event.get("index") == index, "event indexes are not contiguous")
        _require(event.get("joint_version_id") == joint_version_id, "mixed joint version IDs")
        expected_parent = None if index == 0 else events[index - 1].get("event_id")
        _require(event.get("parent_event_id") == expected_parent, "broken event parent chain")

    decisions = [event["payload"] for event in events if event["kind"] == "harness_decision"]
    _require([decision.get("action") for decision in decisions] == ["DIRECT", "VERIFY"], "wrong Harness action sequence")
    _require(
        all(decision.get("old_harness_logprob") == 0.0 for decision in decisions),
        "fixture Harness old log-prob is not zero",
    )

    responses = [event["payload"] for event in events if event["kind"] == "model_response"]
    _require(len(responses) == 2, "successful real-model smoke must have two responses")
    token_counts: list[int] = []
    for response in responses:
        validate_token_metadata(
            input_token_ids=response.get("input_token_ids"),
            output_token_ids=response.get("output_token_ids"),
            output_token_logprobs=response.get("output_token_logprobs"),
            completion_loss_mask=response.get("completion_loss_mask"),
            policy_kind=response.get("policy_kind"),
            token_metadata_status=response.get("token_metadata_status"),
        )
        _require(response.get("policy_kind") != "scripted", "scripted response in real smoke")
        _require(response.get("token_metadata_status") == "available", "missing real token metadata")
        token_counts.append(len(response["output_token_ids"]))

    _require(joint_version.policy.startswith("hf:"), "real HF smoke has a non-HF policy version")
    _require(joint_version.tokenizer.startswith("hf:"), "real HF smoke has a non-HF tokenizer version")
    for label, version in (
        ("policy", joint_version.policy),
        ("tokenizer", joint_version.tokenizer),
    ):
        _require("@" in version, f"HF {label} version is not pinned to a snapshot commit")
        commit = version.split("@", maxsplit=1)[1].split(":", maxsplit=1)[0]
        _require(
            len(commit) == 40 and all(character in "0123456789abcdef" for character in commit),
            f"HF {label} version has an invalid snapshot commit",
        )

    reward_payload = next(event["payload"] for event in events if event["kind"] == "reward_assigned")
    _require(len(reward_payload.get("target_model_call_ids", [])) == 2, "reward does not target two model calls")
    _require(len(reward_payload.get("target_harness_decision_ids", [])) == 2, "reward does not target two Harness decisions")

    return {
        "ok": True,
        "trace": str(path),
        "event_count": len(events),
        "joint_version_id": joint_version_id,
        "policy_version": joint_version.policy,
        "tokenizer_version": joint_version.tokenizer,
        "completion_token_counts": token_counts,
        "reward": trace["reward"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a real-model JPH smoke trace")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_trace(args.trace.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
