from __future__ import annotations

import argparse
import copy
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
    JointDecisionBatch,
    StaleJointVersionError,
    build_joint_decision_batch,
    require_lag_zero_admission,
)
from ..trajectory.schema import EpisodeTrace, JointVersion, TraceEvent


EXPERIMENT_NAME = "g1-joint-integrity-v1"
POLICY_CREDIT_SOURCE = "synthetic-policy-credit-fixture-v1"
HARNESS_CREDIT_SOURCE = "synthetic-harness-credit-fixture-v1"


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
            "policy_release_id": joint_version.policy,
            "output_versions": [0, 0],
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
    policy = ComponentCheckpoint.create(
        component="policy",
        version=version.policy,
        parameters=(0.125, -0.25),
        optimizer_momentum=(0.0, 0.0),
        optimizer_step=0,
        rng_state=101,
        sample_count=0,
    )
    harness = ComponentCheckpoint.create(
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
        schema_version="jph.joint-checkpoint.v1",
        joint_version=version,
        active_release_id=active_release_id,
        macro_step=0,
        policy=policy,
        harness=harness,
        rng_state=1729,
        rollout_cursor=0,
    )


def update_checkpoint_from_batch(
    checkpoint: JointCheckpoint,
    *,
    batch: JointDecisionBatch,
) -> JointCheckpoint:
    checkpoint.validate()
    batch.validate()
    if batch.joint_version != checkpoint.joint_version:
        raise ValueError("decision batch behavior version differs from checkpoint")

    policy_gradient_sum = [0.0, 0.0]
    active_policy_tokens = 0
    for sample in batch.policy_tokens:
        if sample.policy_loss_mask == 0:
            continue
        features = (
            ((sample.token_id % 17) - 8) / 8.0,
            ((sample.output_position % 3) - 1) / 2.0,
        )
        for index, feature in enumerate(features):
            policy_gradient_sum[index] += sample.advantage * feature
        active_policy_tokens += 1

    harness_gradient_sum = [0.0, 0.0]
    active_harness_actions = 0
    for sample in batch.harness_actions:
        if sample.harness_loss_mask == 0:
            continue
        action_index = sample.action_ids.index(sample.action)
        midpoint = (len(sample.action_ids) - 1) / 2.0
        features = (
            (action_index - midpoint) / len(sample.action_ids),
            1.0 if action_index % 2 == 0 else -1.0,
        )
        for index, feature in enumerate(features):
            harness_gradient_sum[index] += sample.advantage * feature
        active_harness_actions += 1

    def update(
        component: ComponentCheckpoint,
        gradient_sum: list[float],
        active_count: int,
    ) -> ComponentCheckpoint:
        if active_count == 0:
            return component
        gradient = tuple(value / active_count for value in gradient_sum)
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
        return ComponentCheckpoint.create(
            component=component.component,
            version=_component_version(component.component, step, parameters, momentum),
            parameters=parameters,
            optimizer_momentum=momentum,
            optimizer_step=step,
            rng_state=component.rng_state,
            sample_count=component.sample_count,
        )

    policy = update(
        checkpoint.policy,
        policy_gradient_sum,
        active_policy_tokens,
    )
    harness = update(
        checkpoint.harness,
        harness_gradient_sum,
        active_harness_actions,
    )
    if policy is checkpoint.policy and harness is checkpoint.harness:
        return checkpoint
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
        schema_version=checkpoint.schema_version,
        joint_version=joint_version,
        active_release_id=active_release_id,
        macro_step=checkpoint.macro_step + 1,
        policy=policy,
        harness=harness,
        rng_state=checkpoint.rng_state,
        rollout_cursor=checkpoint.rollout_cursor + len(batch.episode_ids),
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


def _run_update_interventions(
    initial: JointCheckpoint,
    batch: JointDecisionBatch,
) -> dict[str, object]:
    baseline = update_checkpoint_from_batch(initial, batch=batch)
    policy_credit_batch = replace(
        batch,
        policy_tokens=tuple(
            replace(sample, advantage=sample.advantage + 0.75)
            if sample.policy_loss_mask == 1
            else sample
            for sample in batch.policy_tokens
        ),
    )
    policy_credit_candidate = update_checkpoint_from_batch(
        initial, batch=policy_credit_batch
    )
    harness_credit_batch = replace(
        batch,
        harness_actions=tuple(
            replace(sample, advantage=sample.advantage - 0.625)
            if sample.harness_loss_mask == 1
            else sample
            for sample in batch.harness_actions
        ),
    )
    harness_credit_candidate = update_checkpoint_from_batch(
        initial, batch=harness_credit_batch
    )
    masked_policy_batch = replace(
        batch,
        policy_tokens=tuple(
            replace(
                sample,
                token_id=sample.token_id + 5000,
                advantage=sample.advantage + 100.0,
            )
            if sample.policy_loss_mask == 0
            else sample
            for sample in batch.policy_tokens
        ),
    )
    masked_policy_candidate = update_checkpoint_from_batch(
        initial, batch=masked_policy_batch
    )
    masked_harness_batch = replace(
        batch,
        harness_actions=tuple(
            replace(sample, advantage=sample.advantage - 100.0)
            if sample.harness_loss_mask == 0
            else sample
            for sample in batch.harness_actions
        ),
    )
    masked_harness_candidate = update_checkpoint_from_batch(
        initial, batch=masked_harness_batch
    )

    checks = {
        "policy_credit_changes_policy": (
            policy_credit_candidate.policy != baseline.policy
        ),
        "policy_credit_leaves_harness_unchanged": (
            policy_credit_candidate.harness == baseline.harness
        ),
        "harness_credit_changes_harness": (
            harness_credit_candidate.harness != baseline.harness
        ),
        "harness_credit_leaves_policy_unchanged": (
            harness_credit_candidate.policy == baseline.policy
        ),
        "masked_policy_perturbation_leaves_policy_unchanged": (
            masked_policy_candidate.policy == baseline.policy
        ),
        "masked_policy_perturbation_leaves_harness_unchanged": (
            masked_policy_candidate.harness == baseline.harness
        ),
        "masked_harness_perturbation_leaves_harness_unchanged": (
            masked_harness_candidate.harness == baseline.harness
        ),
        "masked_harness_perturbation_leaves_policy_unchanged": (
            masked_harness_candidate.policy == baseline.policy
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "baseline_policy_version": baseline.policy.version,
        "baseline_harness_version": baseline.harness.version,
    }


def _run_negative_mutations(
    trace: EpisodeTrace,
    credit: EpisodeCredit,
) -> dict[str, object]:
    def rejected(name: str, operation: object) -> tuple[str, bool]:
        try:
            operation()
        except (KeyError, TypeError, ValueError):
            return name, True
        return name, False

    def mutate_event(kind: str, field: str, value: object) -> EpisodeTrace:
        candidate = copy.deepcopy(trace)
        event = next(event for event in candidate.events if event.kind == kind)
        event.payload[field] = value
        return candidate

    model_event = next(event for event in trace.events if event.kind == "model_response")
    harness_event = next(
        event for event in trace.events if event.kind == "harness_decision"
    )
    crossed_credit = EpisodeCredit(
        policy_calls={
            harness_event.payload["decision_id"]: DecisionCredit(
                1.0, "crossed-policy-target"
            )
        },
        harness_decisions={
            model_event.payload["model_call_id"]: DecisionCredit(
                1.0, "crossed-harness-target"
            )
        },
    )
    mixed = copy.deepcopy(trace)
    mixed.events[0] = replace(
        mixed.events[0], joint_version_id="injected-half-version"
    )

    cases = dict(
        (
            rejected(
                "policy_logprob_length",
                lambda: mutate_event(
                    "model_response", "output_token_logprobs", [-0.1]
                ).validate(),
            ),
            rejected(
                "policy_non_binary_mask",
                lambda: mutate_event(
                    "model_response", "completion_loss_mask", [1, 2]
                ).validate(),
            ),
            rejected(
                "policy_release_mismatch",
                lambda: build_joint_decision_batch(
                    [mutate_event("model_response", "policy_release_id", "wrong")],
                    {trace.episode_id: credit},
                    allow_open_fixtures=True,
                ),
            ),
            rejected(
                "inference_version_length",
                lambda: build_joint_decision_batch(
                    [mutate_event("model_response", "output_versions", [0])],
                    {trace.episode_id: credit},
                    allow_open_fixtures=True,
                ),
            ),
            rejected(
                "harness_chosen_action_masked",
                lambda: mutate_event(
                    "harness_decision", "action_mask", [False] * len(HarnessAction)
                ).validate(),
            ),
            rejected(
                "harness_logprob_mismatch",
                lambda: mutate_event(
                    "harness_decision", "old_harness_logprob", 0.0
                ).validate(),
            ),
            rejected(
                "harness_version_mismatch",
                lambda: mutate_event(
                    "harness_decision", "controller_version", "wrong"
                ).validate(),
            ),
            rejected(
                "crossed_credit_targets",
                lambda: build_joint_decision_batch(
                    [copy.deepcopy(trace)],
                    {trace.episode_id: crossed_credit},
                    allow_open_fixtures=True,
                ),
            ),
            rejected("mixed_event_version", mixed.validate),
            rejected(
                "open_trace_in_production_builder",
                lambda: build_joint_decision_batch(
                    [copy.deepcopy(trace)], {trace.episode_id: credit}
                ),
            ),
        )
    )
    return {
        "cases": cases,
        "rejected": sum(cases.values()),
        "total": len(cases),
        "passed": len(cases) == 10 and all(cases.values()),
    }


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
                try:
                    require_lag_zero_admission(trace, current.joint_version)
                    stale_accepted_at_lag_0 += 1
                except StaleJointVersionError:
                    stale_discarded_at_lag_0 += 1
            else:
                require_lag_zero_admission(trace, current.joint_version)
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
        "fixture_kind": "deterministic-synthetic-version-trace",
        "synthetic_fixtures_started": started,
        "synthetic_fixtures_ended": ended,
        "synthetic_fixtures_internally_valid": internally_valid,
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
    version_fixtures: int,
    work_dir: str | Path,
    output: str | Path,
    project_commit: str = "unrecorded-local",
) -> dict[str, object]:
    if version_fixtures < 1000 or version_fixtures % 10 != 0:
        raise ValueError(
            "G1 integrity experiment requires at least 1000 version fixtures divisible by 10"
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
    batch = build_joint_decision_batch(
        traces,
        credits,
        allow_open_fixtures=True,
    )
    summary = batch.summary()
    negative_mutations = _run_negative_mutations(
        traces[0], credits[traces[0].episode_id]
    )

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
    checkpoint_store = JointReleaseStore(work / "checkpoint-release-store")
    initial_release = checkpoint_store.publish(
        joint_version=initial.joint_version,
        policy=_artifact(initial.policy),
        harness=_artifact(initial.harness),
        expected_active_release_id=None,
    )
    if initial_release.release_id != initial.active_release_id:
        raise AssertionError("initial checkpoint release prediction differs from the store")
    checkpoint_store.validate_checkpoint(initial)
    checkpoint_path = work / "checkpoint-step-000000.json"
    write_joint_checkpoint(checkpoint_path, initial)
    restored = read_joint_checkpoint(checkpoint_path)
    checkpoint_store.validate_checkpoint(restored)
    continuous_next = update_checkpoint_from_batch(initial, batch=batch)
    continuous_evidence = next_step_evidence(initial, batch_digest=batch.digest)
    restored_evidence = next_step_evidence(restored, batch_digest=batch.digest)
    restored_next = update_checkpoint_from_batch(restored, batch=batch)
    checkpoint_replay_equal = continuous_next.to_dict() == restored_next.to_dict()
    next_action_equal = continuous_evidence == restored_evidence
    candidate_release = checkpoint_store.publish(
        joint_version=continuous_next.joint_version,
        policy=_artifact(continuous_next.policy),
        harness=_artifact(continuous_next.harness),
        expected_active_release_id=initial_release.release_id,
    )
    if candidate_release.release_id != continuous_next.active_release_id:
        raise AssertionError("candidate checkpoint release prediction differs from the store")
    checkpoint_store.validate_checkpoint(continuous_next)
    write_joint_checkpoint(work / "checkpoint-step-000001.json", continuous_next)

    publish_matrix = _run_publish_fault_matrix(work, initial, continuous_next)
    version_schedule = _run_version_schedule(work, version_fixtures)
    tabular_harness_restore = _run_tabular_harness_restore()
    update_interventions = _run_update_interventions(initial, batch)
    credit_sources_disjoint = set(summary["policy_credit_sources"]).isdisjoint(
        summary["harness_credit_sources"]
    )
    fixture_advantages_distinct = all(
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
            update_interventions["passed"],
            negative_mutations["passed"],
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
            "Deterministic synthetic CPU control-plane fixtures, a local POSIX "
            "commit-point crash matrix, and toy one-step replay. This run does not "
            "perform an AReaL policy update or claim policy/Harness joint learning."
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
            "fixture_advantages_distinct": fixture_advantages_distinct,
            "update_interventions": update_interventions,
            "negative_mutations": negative_mutations,
        },
        "mixed_version": {
            **version_schedule,
            "negative_control_rejected": injected_mixed_version_rejected,
        },
        "checkpoint_replay": {
            "initial_digest": initial.digest,
            "initial_active_release_id": initial.active_release_id,
            "candidate_active_release_id": continuous_next.active_release_id,
            "active_release_store_validation": True,
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
    parser.add_argument(
        "--version-fixtures",
        "--episodes",
        dest="version_fixtures",
        type=int,
        default=1000,
    )
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
        version_fixtures=args.version_fixtures,
        work_dir=args.work_dir,
        output=args.output,
        project_commit=args.project_commit,
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
