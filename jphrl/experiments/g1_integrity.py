from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from ..harness.controller import HarnessState
from ..harness.learning import TabularHarnessController
from ..harness.spec import HarnessAction
from ..joint_release import (
    CandidateArtifact,
    ComponentCheckpoint,
    JointCheckpoint,
    JointReleaseStore,
    ReleaseManifest,
    read_joint_checkpoint,
    write_joint_checkpoint,
)
from ..paths import require_within_configured_root
from ..trajectory.joint_batch import (
    DecisionCredit,
    EpisodeCredit,
    build_joint_training_batch,
)
from ..trajectory.schema import EpisodeTrace, JointVersion, TraceEvent


EXPERIMENT_NAME = "g1-joint-integrity-v1"
POLICY_CREDIT_SOURCE = "policy-token-verifier-advantage-v1"
HARNESS_CREDIT_SOURCE = "harness-action-counterfactual-advantage-v1"


def _joint_version() -> JointVersion:
    return JointVersion(
        policy="toy-policy-step-000000",
        harness_controller="toy-harness-step-000000",
        harness_artifact="calculator-harness-v1",
        tool_schema="calculator-fraction-v1",
        parser="strict-json-v1",
        environment="g1-contract-environment-v1",
        evaluator="g1-frozen-evaluator-v1",
        tokenizer="g1-toy-tokenizer-v1",
        context_builder="g1-context-builder-v1",
    )


def _masked_logprob(logits: tuple[float, ...], index: int) -> float:
    maximum = max(logits)
    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    return logits[index] - normalizer


def build_contract_episode(
    index: int,
    joint_version: JointVersion,
) -> tuple[EpisodeTrace, EpisodeCredit]:
    episode_id = f"g1-episode-{index:04d}"
    decision_id = f"{episode_id}:harness:0"
    model_call_id = f"{episode_id}:model:0"
    action_ids = tuple(action.value for action in HarnessAction)
    action_index = index % len(action_ids)
    logits = (-0.4, -0.2, 0.0, 0.2, 0.4)
    harness_loss_mask = 0 if index % 4 == 0 else 1
    trace = EpisodeTrace(
        episode_id=episode_id,
        task_id=f"contract-task-{index % 8}",
        seed=index,
        joint_version=joint_version,
        harness_spec_hash="g1-frozen-harness-spec",
    )
    trace.append(
        "harness_decision",
        "harness",
        {
            "decision_id": decision_id,
            "action": action_ids[action_index],
            "old_harness_logprob": _masked_logprob(logits, action_index),
            "controller_version": joint_version.harness_controller,
            "action_ids": action_ids,
            "action_mask": (True,) * len(action_ids),
            "pre_mask_logits": logits,
            "harness_loss_mask": harness_loss_mask,
        },
    )
    trace.append(
        "model_response",
        "policy",
        {
            "model_call_id": model_call_id,
            "input_token_ids": [100 + index % 31],
            "output_token_ids": [200 + index % 37, 300 + index % 41],
            "output_token_logprobs": [-0.25 - index % 5 * 0.01, -0.5],
            "completion_loss_mask": [1, index % 2],
            "policy_kind": "causal_lm-contract-fixture",
            "token_metadata_status": "available",
        },
    )
    trace.validate()
    policy_advantage = 1.0 if index % 2 == 0 else -0.25
    harness_advantage = -0.5 if index % 3 == 0 else 0.75
    return trace, EpisodeCredit(
        policy_calls={
            model_call_id: DecisionCredit(policy_advantage, POLICY_CREDIT_SOURCE)
        },
        harness_decisions={
            decision_id: DecisionCredit(harness_advantage, HARNESS_CREDIT_SOURCE)
        },
    )


def _component_version(
    component: str,
    step: int,
    parameters: tuple[float, ...],
    momentum: tuple[float, ...],
) -> str:
    payload = json.dumps(
        {
            "component": component,
            "step": step,
            "parameters": parameters,
            "momentum": momentum,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"toy-{component}-step-{step:06d}-{digest}"


def initial_checkpoint() -> JointCheckpoint:
    version = _joint_version()
    policy = ComponentCheckpoint(
        component="policy",
        version=version.policy,
        parameters=(0.125, -0.25),
        optimizer_momentum=(0.0, 0.0),
        optimizer_step=0,
        rng_state=101,
        sample_count=0,
    )
    harness = ComponentCheckpoint(
        component="harness",
        version=version.harness_controller,
        parameters=(0.375, -0.5),
        optimizer_momentum=(0.0, 0.0),
        optimizer_step=0,
        rng_state=202,
        sample_count=0,
    )
    active_release_id = _predicted_release_id(
        version, policy, harness, parent_release_id=None
    )
    return JointCheckpoint(
        joint_version=version,
        active_release_id=active_release_id,
        macro_step=0,
        policy=policy,
        harness=harness,
        rng_state=1729,
        rollout_cursor=0,
    )


def deterministic_joint_step(
    checkpoint: JointCheckpoint,
    *,
    batch_digest: str,
    episode_count: int,
) -> JointCheckpoint:
    checkpoint.validate()
    digest_term = int(batch_digest[:8], 16)
    rng_state = (
        1103515245 * checkpoint.rng_state + 12345 + digest_term
    ) % (2**31)
    policy_rng_state = (
        1103515245 * checkpoint.policy.rng_state + 12345 + digest_term
    ) % (2**31)
    harness_rng_state = (
        1103515245 * checkpoint.harness.rng_state + 12345 + digest_term
    ) % (2**31)
    jitter = (
        ((rng_state ^ policy_rng_state ^ harness_rng_state) % 2001) - 1000
    ) / 1_000_000.0
    policy_gradient = (0.2 + jitter, -0.1 - jitter)
    harness_gradient = (-0.15 + jitter, 0.3 - jitter)

    def update(
        component: ComponentCheckpoint,
        gradient: tuple[float, ...],
        next_rng_state: int,
    ) -> ComponentCheckpoint:
        momentum = tuple(
            0.9 * old_momentum + gradient_value
            for old_momentum, gradient_value in zip(
                component.optimizer_momentum, gradient
            )
        )
        parameters = tuple(
            value + 0.05 * momentum_value
            for value, momentum_value in zip(component.parameters, momentum)
        )
        step = component.optimizer_step + 1
        return ComponentCheckpoint(
            component=component.component,
            version=_component_version(component.component, step, parameters, momentum),
            parameters=parameters,
            optimizer_momentum=momentum,
            optimizer_step=step,
            rng_state=next_rng_state,
            sample_count=component.sample_count + 1,
        )

    policy = update(checkpoint.policy, policy_gradient, policy_rng_state)
    harness = update(checkpoint.harness, harness_gradient, harness_rng_state)
    joint_version = replace(
        checkpoint.joint_version,
        policy=policy.version,
        harness_controller=harness.version,
    )
    active_release_id = _predicted_release_id(
        joint_version,
        policy,
        harness,
        parent_release_id=checkpoint.active_release_id,
    )
    candidate = JointCheckpoint(
        joint_version=joint_version,
        active_release_id=active_release_id,
        macro_step=checkpoint.macro_step + 1,
        policy=policy,
        harness=harness,
        rng_state=rng_state,
        rollout_cursor=checkpoint.rollout_cursor + episode_count,
    )
    candidate.validate()
    return candidate


def next_step_evidence(
    checkpoint: JointCheckpoint,
    *,
    batch_digest: str,
) -> dict[str, object]:
    digest_term = int(batch_digest[:8], 16)
    policy_draw = (
        1103515245 * checkpoint.policy.rng_state + 12345 + digest_term
    ) % (2**31)
    harness_draw = (
        1103515245 * checkpoint.harness.rng_state + 12345 + digest_term
    ) % (2**31)
    action_ids = tuple(action.value for action in HarnessAction)
    evidence = {
        "policy_token_id": 1000 + policy_draw % 97,
        "policy_old_logprob": -0.1 - (policy_draw % 1000) / 1000.0,
        "policy_loss_mask": 1,
        "harness_action": action_ids[harness_draw % len(action_ids)],
        "harness_old_logprob": -math.log(len(action_ids)),
        "harness_action_mask": [True] * len(action_ids),
        "harness_loss_mask": 1,
        "policy_sample_count": checkpoint.policy.sample_count,
        "harness_sample_count": checkpoint.harness.sample_count,
        "rollout_cursor": checkpoint.rollout_cursor,
    }
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    evidence["digest"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return evidence


def _artifact(component: ComponentCheckpoint) -> CandidateArtifact:
    return CandidateArtifact(
        component=component.component,
        version=component.version,
        payload=component.to_dict(),
    )


def _predicted_release_id(
    joint_version: JointVersion,
    policy: ComponentCheckpoint,
    harness: ComponentCheckpoint,
    *,
    parent_release_id: str | None,
) -> str:
    policy_artifact = _artifact(policy)
    harness_artifact = _artifact(harness)
    manifest = ReleaseManifest.create(
        parent_release_id=parent_release_id,
        joint_version=joint_version,
        policy_object=(
            f"objects/policy-{policy_artifact.digest}.json"
        ),
        harness_object=(
            f"objects/harness-{harness_artifact.digest}.json"
        ),
    )
    return manifest.release_id


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    path = require_within_configured_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_worker(config_path: str | Path) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    candidate = read_joint_checkpoint(config["candidate_checkpoint"])
    store = JointReleaseStore(config["store_root"])
    store.publish(
        joint_version=candidate.joint_version,
        policy=_artifact(candidate.policy),
        harness=_artifact(candidate.harness),
        expected_active_release_id=config["expected_active_release_id"],
        fault_at=config["fault_at"],
        fault_mode="exit",
    )
    return 0


def _run_publish_fault_matrix(
    root: Path,
    initial: JointCheckpoint,
    candidate: JointCheckpoint,
) -> dict[str, object]:
    phases: tuple[str | None, ...] = (
        "after_policy_object",
        "after_harness_object",
        "before_active_switch",
        "after_active_switch",
        None,
    )
    rows: list[dict[str, object]] = []
    old_pair = (initial.policy.version, initial.harness.version)
    new_pair = (candidate.policy.version, candidate.harness.version)
    for phase in phases:
        case_name = phase or "success"
        store = JointReleaseStore(root / f"publish-{case_name}")
        first = store.publish(
            joint_version=initial.joint_version,
            policy=_artifact(initial.policy),
            harness=_artifact(initial.harness),
            expected_active_release_id=None,
        )
        if first.release_id != initial.active_release_id:
            raise AssertionError("initial checkpoint does not identify its active release")
        candidate_checkpoint = root / f"publish-{case_name}-candidate.json"
        write_joint_checkpoint(candidate_checkpoint, candidate)
        config_path = root / f"publish-{case_name}-worker.json"
        _write_private_json(
            config_path,
            {
                "store_root": str(store.root),
                "candidate_checkpoint": str(candidate_checkpoint),
                "expected_active_release_id": first.release_id,
                "fault_at": phase,
            },
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "jphrl.experiments.g1_integrity",
                "--publish-worker-config",
                str(config_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        injected = completed.returncode == 91
        active = store.read_active()
        if active is None:
            raise AssertionError("publish store lost its active manifest")
        active_pair = (
            active.joint_version.policy,
            active.joint_version.harness_controller,
        )
        expected_pair = (
            new_pair if phase in {"after_active_switch", None} else old_pair
        )
        no_half_version = active_pair in {old_pair, new_pair}
        passed = (
            active_pair == expected_pair
            and no_half_version
            and injected == (phase is not None)
            and (
                active.release_id
                == (
                    candidate.active_release_id
                    if expected_pair == new_pair
                    else initial.active_release_id
                )
            )
        )
        rows.append(
            {
                "fault_at": phase,
                "failure_observed": injected,
                "worker_exit_code": completed.returncode,
                "active_pair": list(active_pair),
                "expected_pair": list(expected_pair),
                "no_half_version": no_half_version,
                "passed": passed,
            }
        )
    rejection_store = JointReleaseStore(root / "publish-release-gate-reject")
    initial_release = rejection_store.publish(
        joint_version=initial.joint_version,
        policy=_artifact(initial.policy),
        harness=_artifact(initial.harness),
        expected_active_release_id=None,
    )
    gate_rejected = False
    try:
        rejection_store.publish(
            joint_version=candidate.joint_version,
            policy=_artifact(candidate.policy),
            harness=_artifact(initial.harness),
            expected_active_release_id=initial_release.release_id,
        )
    except ValueError:
        gate_rejected = True
    active_after_rejection = rejection_store.read_active()
    rejection_pair = (
        active_after_rejection.joint_version.policy,
        active_after_rejection.joint_version.harness_controller,
    )
    rejection_passed = gate_rejected and rejection_pair == old_pair
    rows.append(
        {
            "fault_at": "release_gate_reject",
            "failure_observed": gate_rejected,
            "worker_exit_code": None,
            "active_pair": list(rejection_pair),
            "expected_pair": list(old_pair),
            "no_half_version": rejection_pair == old_pair,
            "passed": rejection_passed,
        }
    )
    return {
        "cases": rows,
        "passed": all(row["passed"] for row in rows),
    }


def _scheduled_version(base: JointVersion, generation: int) -> JointVersion:
    return replace(
        base,
        policy=f"scheduled-policy-generation-{generation:02d}",
        harness_controller=f"scheduled-harness-generation-{generation:02d}",
    )


def _scheduled_artifact(component: str, version: str, generation: int) -> CandidateArtifact:
    return CandidateArtifact(
        component=component,
        version=version,
        payload={"generation": generation, "version": version},
    )


def _run_version_schedule(root: Path, episodes: int) -> dict[str, object]:
    publish_count = 10
    if episodes < 1000 or episodes % publish_count != 0:
        raise ValueError("version schedule requires at least 1000 episodes divisible by 10")
    episodes_per_publish = episodes // publish_count
    straddled_per_publish = 10
    base = _joint_version()
    initial_version = _scheduled_version(base, 0)
    store = JointReleaseStore(root / "version-schedule-store")
    active = store.publish(
        joint_version=initial_version,
        policy=_scheduled_artifact("policy", initial_version.policy, 0),
        harness=_scheduled_artifact(
            "harness", initial_version.harness_controller, 0
        ),
        expected_active_release_id=None,
    )

    started = 0
    ended = 0
    internally_valid = 0
    straddled = 0
    mixed_version_episodes = 0
    half_version_observations = 0
    stale_accepted_at_lag_0 = 0
    stale_discarded_at_lag_0 = 0
    unresolved_manifest_reads = 0

    for generation in range(1, publish_count + 1):
        pinned_before_publish = store.read_active()
        if pinned_before_publish is None:
            unresolved_manifest_reads += 1
            raise AssertionError("active manifest disappeared before publish")
        started += straddled_per_publish
        next_version = _scheduled_version(base, generation)
        active = store.publish(
            joint_version=next_version,
            policy=_scheduled_artifact("policy", next_version.policy, generation),
            harness=_scheduled_artifact(
                "harness", next_version.harness_controller, generation
            ),
            expected_active_release_id=active.release_id,
        )
        current = store.read_active()
        if current is None:
            unresolved_manifest_reads += 1
            raise AssertionError("active manifest disappeared after publish")
        expected_pair = (next_version.policy, next_version.harness_controller)
        observed_pair = (
            current.joint_version.policy,
            current.joint_version.harness_controller,
        )
        if observed_pair != expected_pair:
            half_version_observations += 1

        for local_index in range(episodes_per_publish):
            index = (generation - 1) * episodes_per_publish + local_index
            is_straddled = local_index < straddled_per_publish
            episode_version = (
                pinned_before_publish.joint_version
                if is_straddled
                else current.joint_version
            )
            if not is_straddled:
                started += 1
            trace, _ = build_contract_episode(index, episode_version)
            try:
                trace.validate()
                internally_valid += 1
            except ValueError as exc:
                if "version" in str(exc):
                    mixed_version_episodes += 1
                else:
                    raise
            ended += 1
            if is_straddled:
                straddled += 1
                if trace.joint_version.version_id == current.joint_version.version_id:
                    stale_accepted_at_lag_0 += 1
                else:
                    stale_discarded_at_lag_0 += 1
            observed = store.read_active()
            if observed is None:
                unresolved_manifest_reads += 1
            elif (
                observed.joint_version.policy,
                observed.joint_version.harness_controller,
            ) != expected_pair:
                half_version_observations += 1

    passed = all(
        (
            started == episodes,
            ended == episodes,
            internally_valid == episodes,
            straddled >= 100,
            mixed_version_episodes == 0,
            half_version_observations == 0,
            stale_accepted_at_lag_0 == 0,
            stale_discarded_at_lag_0 == straddled,
            unresolved_manifest_reads == 0,
        )
    )
    return {
        "episodes_started": started,
        "episodes_ended": ended,
        "internally_valid": internally_valid,
        "joint_publishes": publish_count,
        "straddled_publish": straddled,
        "mixed_version_episodes": mixed_version_episodes,
        "half_version_observations": half_version_observations,
        "stale_accepted_at_lag_0": stale_accepted_at_lag_0,
        "stale_discarded_at_lag_0": stale_discarded_at_lag_0,
        "unresolved_manifest_reads": unresolved_manifest_reads,
        "passed": passed,
    }


def _run_tabular_harness_restore() -> dict[str, object]:
    state = HarnessState(
        turn=2,
        remaining_tool_calls=1,
        remaining_model_retries=1,
        context_chars=256,
        last_error=None,
        retrieval_hit=True,
        verifier_status="decision",
        task_domain="g1-restore",
    )
    controller = TabularHarnessController(seed=29)
    for _ in range(3):
        controller.choose(state)
    serialized = json.loads(json.dumps(controller.checkpoint()))
    restored = TabularHarnessController.from_checkpoint(serialized)
    continuous_decision = controller.choose(state)
    restored_decision = restored.choose(state)
    exact_next_decision = continuous_decision == restored_decision
    return {
        "checkpoint_version": serialized["version"],
        "sample_count_before_next": serialized["sample_count"],
        "decision_id": continuous_decision.decision_id,
        "action": continuous_decision.action.value,
        "old_harness_logprob": continuous_decision.old_harness_logprob,
        "exact_next_decision": exact_next_decision,
        "passed": exact_next_decision,
    }


def run_experiment(
    *,
    episodes: int,
    work_dir: str | Path,
    output: str | Path,
    project_commit: str = "unrecorded-local",
) -> dict[str, object]:
    if episodes < 1000 or episodes % 10 != 0:
        raise ValueError(
            "G1 integrity experiment requires at least 1000 episodes divisible by 10"
        )
    work = require_within_configured_root(work_dir)
    destination = require_within_configured_root(output)
    work.mkdir(parents=True, exist_ok=False)
    os.chmod(work, 0o700)
    destination.parent.mkdir(parents=True, exist_ok=True)

    version = _joint_version()
    separation_episodes = 32
    traces: list[EpisodeTrace] = []
    credits: dict[str, EpisodeCredit] = {}
    for index in range(separation_episodes):
        trace, episode_credit = build_contract_episode(index, version)
        traces.append(trace)
        credits[trace.episode_id] = episode_credit
    batch = build_joint_training_batch(traces, credits)
    summary = batch.summary()

    injected_events = list(traces[0].events)
    injected_events[0] = replace(
        injected_events[0], joint_version_id="injected-half-version"
    )
    injected_trace = EpisodeTrace(
        episode_id="g1-injected-mixed-version",
        task_id="negative-control",
        seed=0,
        joint_version=version,
        harness_spec_hash="g1-frozen-harness-spec",
        events=injected_events,
    )
    injected_mixed_version_rejected = False
    try:
        injected_trace.validate()
    except ValueError as exc:
        injected_mixed_version_rejected = "mixed-version" in str(exc)

    initial = initial_checkpoint()
    checkpoint_path = work / "checkpoint-step-000000.json"
    write_joint_checkpoint(checkpoint_path, initial)
    continuous_next = deterministic_joint_step(
        initial, batch_digest=batch.digest, episode_count=episodes
    )
    restored = read_joint_checkpoint(checkpoint_path)
    continuous_evidence = next_step_evidence(initial, batch_digest=batch.digest)
    restored_evidence = next_step_evidence(restored, batch_digest=batch.digest)
    restored_next = deterministic_joint_step(
        restored, batch_digest=batch.digest, episode_count=episodes
    )
    checkpoint_replay_equal = continuous_next.to_dict() == restored_next.to_dict()
    next_action_equal = continuous_evidence == restored_evidence
    write_joint_checkpoint(work / "checkpoint-step-000001.json", continuous_next)

    publish_matrix = _run_publish_fault_matrix(work, initial, continuous_next)
    version_schedule = _run_version_schedule(work, episodes)
    tabular_harness_restore = _run_tabular_harness_restore()
    credit_sources_disjoint = set(summary["policy_credit_sources"]).isdisjoint(
        summary["harness_credit_sources"]
    )
    all_credits_distinct = all(
        credits[trace.episode_id].policy_calls[
            f"{trace.episode_id}:model:0"
        ].advantage
        != credits[trace.episode_id].harness_decisions[
            f"{trace.episode_id}:harness:0"
        ].advantage
        for trace in traces
    )
    passed = all(
        (
            injected_mixed_version_rejected,
            credit_sources_disjoint,
            all_credits_distinct,
            checkpoint_replay_equal,
            next_action_equal,
            publish_matrix["passed"],
            version_schedule["passed"],
            tabular_harness_restore["passed"],
            summary["trainable_policy_tokens"] > 0,
            summary["trainable_harness_actions"] > 0,
        )
    )
    payload: dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "project_commit": project_commit,
        "python_version": sys.version.split()[0],
        "claim_boundary": (
            "CPU control-plane integrity gate; it does not perform an AReaL policy "
            "update or claim policy/Harness joint learning"
        ),
        "updates": {
            "areal_policy_update": False,
            "production_harness_update": False,
            "toy_candidate_transition_only": True,
        },
        "passed": passed,
        "batch": summary,
        "credit_separation": {
            "policy_source": POLICY_CREDIT_SOURCE,
            "harness_source": HARNESS_CREDIT_SOURCE,
            "sources_disjoint": credit_sources_disjoint,
            "per_episode_advantages_distinct": all_credits_distinct,
        },
        "mixed_version": {
            **version_schedule,
            "negative_control_rejected": injected_mixed_version_rejected,
        },
        "checkpoint_replay": {
            "initial_digest": initial.digest,
            "initial_active_release_id": initial.active_release_id,
            "candidate_active_release_id": continuous_next.active_release_id,
            "continuous_next_digest": continuous_next.digest,
            "restored_next_digest": restored_next.digest,
            "next_step_equal": checkpoint_replay_equal,
            "next_action_equal": next_action_equal,
            "continuous_action_evidence": continuous_evidence,
            "restored_action_evidence": restored_evidence,
            "restored_rng_state": restored.rng_state,
            "restored_rollout_cursor": restored.rollout_cursor,
            "restored_policy_rng_state": restored.policy.rng_state,
            "restored_harness_rng_state": restored.harness.rng_state,
            "restored_policy_sample_count": restored.policy.sample_count,
            "restored_harness_sample_count": restored.harness.sample_count,
            "tabular_harness": tabular_harness_restore,
        },
        "atomic_publish": publish_matrix,
        "artifacts": {
            "work_dir": str(work),
            "checkpoint_step_0": str(checkpoint_path),
            "checkpoint_step_1": str(work / "checkpoint-step-000001.json"),
        },
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the G1 joint trajectory, publish, and recovery integrity gate"
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--work-dir")
    parser.add_argument("--output")
    parser.add_argument("--publish-worker-config")
    parser.add_argument("--project-commit", default="unrecorded-local")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.publish_worker_config:
        return _publish_worker(args.publish_worker_config)
    if not args.work_dir or not args.output:
        raise ValueError("--work-dir and --output are required for the main experiment")
    payload = run_experiment(
        episodes=args.episodes,
        work_dir=args.work_dir,
        output=args.output,
        project_commit=args.project_commit,
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
