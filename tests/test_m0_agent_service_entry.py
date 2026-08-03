from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from jphrl.experiments.m0_agent_service_entry import (
    M0AgentServiceEntryError,
    PersistentM0AgentServicePreBatchEntry,
    stage_m0_agent_service_rollout,
    validate_finalized_m0_agent_service_envelope,
    validate_staged_m0_agent_service_envelope,
)
from jphrl.harness.controller import HarnessDecision, HarnessState
from jphrl.harness.spec import HarnessAction
from jphrl.trajectory.areal_data_proxy_pre_batch import PreBatchTrajectoryExport
from jphrl.trajectory.areal_joint_bridge import (
    build_joint_version,
    inference_runtime_contract_sha256,
    inject_harness_instruction,
    prompt_context_chars,
)
from jphrl.trajectory.joint_credit_alignment import (
    ESTIMATOR_VERSION,
    DualCreditEstimatorSpec,
)
from jphrl.trajectory.schema import JointVersion
from tests.test_m0_joint_input import (
    M0Source,
    _load_pinned_areal,
    _runtime_contract,
    _source,
)


def _decision(raw: dict[str, object]) -> HarnessDecision:
    return HarnessDecision(
        decision_id=raw["decision_id"],
        action=HarnessAction(raw["action"]),
        old_harness_logprob=raw["old_harness_logprob"],
        controller_version=raw["controller_version"],
        action_ids=tuple(raw["action_ids"]),
        action_mask=tuple(raw["action_mask"]),
        pre_mask_logits=tuple(raw["pre_mask_logits"]),
        harness_loss_mask=raw["harness_loss_mask"],
    )


def _hermes_payload(source: M0Source) -> list[dict[str, object]]:
    binding = source.bridge["interaction_adapter_sidecar"]["bindings"][0]
    return [
        {
            "model_call_id": binding["model_call_id"],
            "interaction_id": binding["interaction_id"],
            "ordinal": binding["ordinal"],
            "parent_model_call_id": None,
            "session_id": binding["session_id"],
        }
    ]


def _stage(
    root: str | Path,
    source: M0Source,
    *,
    start_response: dict[str, object] | None = None,
    reward_response: dict[str, object] | None = None,
    hermes_payload: object | None = None,
):
    bridge = source.bridge
    harness = bridge["harness"]
    prompt = bridge["prompt_binding"]
    runtime = bridge["policy_binding"]
    return stage_m0_agent_service_rollout(
        journal_root=root,
        start_session_response=start_response or source.start_response,
        set_reward_response=reward_response or source.reward_response,
        hermes_receipt_payload=(
            _hermes_payload(source) if hermes_payload is None else hermes_payload
        ),
        episode_id=bridge["episode_id"],
        task_id=bridge["task_id"],
        joint_version=JointVersion(**bridge["joint_version"]),
        harness_state=HarnessState(**harness["state"]),
        harness_decision=_decision(harness["decision"]),
        harness_sampling_before_checkpoint=harness[
            "controller_checkpoint_before_decision"
        ],
        base_messages=prompt["base_messages"],
        effective_messages=prompt["effective_messages"],
        base_input_tokens=prompt["base_input_tokens"],
        effective_input_tokens=prompt["effective_input_tokens"],
        expected_policy_version=runtime["expected_inference_engine_version"],
        project_commit=bridge["origin"]["project_commit"],
        areal_commit=bridge["areal_trace"]["origin"]["areal_commit"],
        behavior_snapshot_path=bridge["areal_trace"]["origin"][
            "behavior_snapshot_path"
        ],
        behavior_revision=bridge["areal_trace"]["origin"]["behavior_revision"],
        dataset_selection=runtime["dataset_selection"],
        sglang_version=runtime["sglang_version"],
        generation_logprob_mode=runtime["generation_logprob_mode"],
        inference_runtime_contract=runtime["inference_runtime_contract"],
        estimator=source.estimator,
        export_style=source.bridge["areal_trace"]["interaction"][
            "chat_template_type"
        ]
        == "concat"
        and "concat"
        or "individual",
        turn_discount=1.0,
    )


def _event(source: M0Source, style: str) -> PreBatchTrajectoryExport:
    return PreBatchTrajectoryExport(
        session_id=source.reward_response["session_id"],
        trajectory_id=source.reward_response["trajectory_id"],
        exported_interactions=source.exported,
        export_style=style,
        turn_discount=1.0,
    )


def _staging_only_arguments() -> dict[str, object]:
    try:
        from jphrl.harness.torch_learning import (
            TorchHarnessPolicy,
            build_torch_harness_rollout_checkpoint,
        )
    except ImportError as exc:
        raise unittest.SkipTest(f"cached Torch is unavailable: {exc}") from exc
    base_messages = [{"role": "user", "content": "What is 20 + 22?"}]
    state = HarnessState(
        turn=0,
        remaining_tool_calls=0,
        remaining_model_retries=0,
        context_chars=prompt_context_chars(base_messages),
        last_error=None,
        retrieval_hit=False,
        verifier_status="not-run",
        task_domain="gsm8k",
    )
    policy = TorchHarnessPolicy(seed=29, hidden_size=16)
    checkpoint = build_torch_harness_rollout_checkpoint(policy)
    decision = policy.choose(state)
    runtime = _runtime_contract(style="individual")
    dataset_selection = runtime["fixed"]["dataset_selection"]
    effective_messages, _ = inject_harness_instruction(base_messages, decision.action)
    joint_version = build_joint_version(
        policy_release_id="areal-sglang@model-commit:engine-v0",
        harness_controller_version=policy.version,
        areal_commit="a" * 40,
        behavior_revision="b" * 40,
        dataset_revision="c" * 40,
        dataset_selection=dataset_selection,
        sglang_version="0.5.10.post1",
        generation_logprob_mode="standard-log-of-softmax-v1",
        inference_runtime_contract_sha256=inference_runtime_contract_sha256(runtime),
    )
    model_call_id = "m0-staging-only:model:0"
    interaction_id = "real-upstream-response-id-29"
    session_id = "m0-staging-only-session"
    estimator = DualCreditEstimatorSpec(
        estimator_version=ESTIMATOR_VERSION,
        parent_joint_version_id=joint_version.version_id,
        policy_source="aa-m0-policy-frozen-terminal-baseline-v1",
        harness_source="aa-m0-harness-frozen-terminal-baseline-v1",
        policy_baseline_snapshot_id="aa-m0-policy-baseline-run-29",
        harness_baseline_snapshot_id="aa-m0-harness-baseline-run-29",
        policy_baselines={model_call_id: 0.25},
        harness_baselines={decision.decision_id: 0.1},
    )
    return {
        "start_session_response": {
            "group_id": "m0-staging-only-group",
            "sessions": [
                {
                    "session_id": session_id,
                    "session_api_key": "must-be-stripped-before-persistence",
                }
            ],
        },
        "set_reward_response": {
            "session_id": session_id,
            "trajectory_id": 0,
            "interaction_count": 1,
            "trajectory_ready": True,
            "ready_transition": True,
        },
        "hermes_receipt_payload": [
            {
                "model_call_id": model_call_id,
                "interaction_id": interaction_id,
                "ordinal": 0,
                "parent_model_call_id": None,
                "session_id": session_id,
            }
        ],
        "episode_id": "m0-staging-only-episode",
        "task_id": 7,
        "joint_version": joint_version,
        "harness_state": state,
        "harness_decision": decision,
        "harness_sampling_before_checkpoint": checkpoint,
        "base_messages": base_messages,
        "effective_messages": effective_messages,
        "base_input_tokens": [1, 2],
        "effective_input_tokens": [9, 1, 2],
        "expected_policy_version": 0,
        "project_commit": "d" * 40,
        "areal_commit": "a" * 40,
        "behavior_snapshot_path": "/allowed/model",
        "behavior_revision": "b" * 40,
        "dataset_selection": dataset_selection,
        "sglang_version": "0.5.10.post1",
        "generation_logprob_mode": "standard-log-of-softmax-v1",
        "inference_runtime_contract": runtime,
        "estimator": estimator,
        "export_style": "individual",
        "turn_discount": 1.0,
    }


class M0AgentServiceEntryTests(unittest.TestCase):
    def test_staging_contract_runs_with_cached_torch_without_areal_runtime(self) -> None:
        arguments = _staging_only_arguments()
        with tempfile.TemporaryDirectory() as directory:
            path = stage_m0_agent_service_rollout(
                journal_root=directory,
                **arguments,
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            audit = validate_staged_m0_agent_service_envelope(record)
            serialized = json.dumps(record, sort_keys=True)

            self.assertEqual(audit["hermes"].interaction_id, "real-upstream-response-id-29")
            self.assertEqual(audit["trajectory"].trajectory_id, 0)
            self.assertNotIn("session_api_key", serialized)
            self.assertNotIn("must-be-stripped-before-persistence", serialized)
            self.assertEqual(path.stat().st_mode & 0o077, 0)
            with self.assertRaisesRegex(M0AgentServiceEntryError, "already staged"):
                stage_m0_agent_service_rollout(
                    journal_root=directory,
                    **arguments,
                )

    def test_staging_raw_http_and_receipt_contracts_fail_without_areal(self) -> None:
        arguments = _staging_only_arguments()
        missing_start = deepcopy(arguments)
        del missing_start["start_session_response"]["sessions"][0]["session_api_key"]
        missing_reward = deepcopy(arguments)
        del missing_reward["set_reward_response"]["trajectory_ready"]
        secret_receipt = deepcopy(arguments)
        secret_receipt["hermes_receipt_payload"][0]["admin_api_key"] = "must-not-enter"
        crossed_receipt = deepcopy(arguments)
        crossed_receipt["hermes_receipt_payload"][0]["session_id"] = "crossed-session"
        nested_runtime_secret = deepcopy(arguments)
        nested_runtime_secret["inference_runtime_contract"]["fixed"]["server_args"][
            "github_token"
        ] = "must-not-enter"
        for broken, message in (
            (missing_start, "routing credential"),
            (missing_reward, "not ready"),
            (secret_receipt, "field set differs"),
            (crossed_receipt, "routed inference session"),
            (nested_runtime_secret, "credential field cannot enter"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    stage_m0_agent_service_rollout(
                        journal_root=directory,
                        **broken,
                    )
                self.assertFalse(list(Path(directory).rglob("*.json")))

    def test_real_sessiondata_individual_and_concat_finalize_exactly_once(self) -> None:
        for style in ("individual", "concat"):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as directory:
                source = _source(style=style)
                staged_path = _stage(directory, source)
                staged = json.loads(staged_path.read_text(encoding="utf-8"))
                staged_audit = validate_staged_m0_agent_service_envelope(staged)
                serialized_stage = json.dumps(staged, sort_keys=True)

                finalizer = PersistentM0AgentServicePreBatchEntry(directory)
                final = finalizer(_event(source, style))
                final_audit = validate_finalized_m0_agent_service_envelope(final)
                serialized_final = json.dumps(final, sort_keys=True)

                self.assertEqual(staged_audit["hermes"].ordinal, 0)
                self.assertEqual(staged_audit["hermes"].interaction_id, next(iter(source.exported)))
                self.assertNotIn("session_api_key", serialized_stage)
                self.assertNotIn("test-routing-secret-never-persist", serialized_stage)
                self.assertNotIn("session_api_key", serialized_final)
                self.assertEqual(
                    final["bridge_record"]["interaction_adapter_sidecar"]["bindings"][0][
                        "route_kind"
                    ],
                    "agent-service-session",
                )
                self.assertEqual(final_audit["p"]["export_style"], style)
                self.assertFalse(final["evidence_scope"]["policy_optimizer_update"])
                self.assertFalse(final["evidence_scope"]["harness_optimizer_update"])
                finalized_files = list(Path(directory, "finalized").glob("*.json"))
                self.assertEqual(len(finalized_files), 1)

                with self.assertRaisesRegex(
                    M0AgentServiceEntryError,
                    "already finalized",
                ):
                    finalizer(_event(source, style))
                self.assertEqual(len(list(Path(directory, "finalized").glob("*.json"))), 1)

    def test_post_batch_mapping_never_reaches_bridge_or_p_q_r_s(self) -> None:
        source = _source(style="individual")
        areal = _load_pinned_areal()
        padded = areal["concat_padded_tensors"](
            [item.to_tensor_dict() for item in source.exported.values()]
        )
        with tempfile.TemporaryDirectory() as directory:
            _stage(directory, source)
            event = PreBatchTrajectoryExport(
                session_id=source.reward_response["session_id"],
                trajectory_id=source.reward_response["trajectory_id"],
                exported_interactions=padded,
                export_style="individual",
                turn_discount=1.0,
            )
            with self.assertRaisesRegex(M0AgentServiceEntryError, "post-batch"):
                PersistentM0AgentServicePreBatchEntry(directory)(event)
            self.assertFalse(Path(directory, "finalized").exists())

    def test_raw_http_missing_fields_and_crossed_hermes_fail_before_staging(self) -> None:
        source = _source(style="individual")
        missing_routing_key = deepcopy(source.start_response)
        del missing_routing_key["sessions"][0]["session_api_key"]
        missing_ready = deepcopy(source.reward_response)
        del missing_ready["trajectory_ready"]
        crossed = _hermes_payload(source)
        crossed[0]["session_id"] = "another-agent-service-session"
        for start, reward, payload, message in (
            (
                missing_routing_key,
                source.reward_response,
                _hermes_payload(source),
                "routing credential",
            ),
            (
                source.start_response,
                missing_ready,
                _hermes_payload(source),
                "not ready",
            ),
            (
                source.start_response,
                source.reward_response,
                crossed,
                "routed inference session",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    _stage(
                        directory,
                        source,
                        start_response=start,
                        reward_response=reward,
                        hermes_payload=payload,
                    )
                self.assertFalse(list(Path(directory).rglob("*.json")))

    def test_secret_extra_fields_and_staged_tampering_fail_closed(self) -> None:
        source = _source(style="individual")
        secret_receipt = _hermes_payload(source)
        secret_receipt[0]["admin_api_key"] = "must-not-enter"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "field set differs"):
                _stage(directory, source, hermes_payload=secret_receipt)
            self.assertFalse(list(Path(directory).rglob("*.json")))

        with tempfile.TemporaryDirectory() as directory:
            path = _stage(directory, source)
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["identity"]["interaction_id"] = "crossed-interaction"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(M0AgentServiceEntryError, "hash mismatch"):
                PersistentM0AgentServicePreBatchEntry(directory)(
                    _event(source, "individual")
                )
            self.assertFalse(Path(directory, "finalized").exists())

    def test_crossed_prebatch_route_and_positive_reward_fail_closed(self) -> None:
        source = _source(style="individual")
        with tempfile.TemporaryDirectory() as directory:
            _stage(directory, source)
            crossed_event = PreBatchTrajectoryExport(
                session_id=source.reward_response["session_id"],
                trajectory_id=source.reward_response["trajectory_id"] + 1,
                exported_interactions=source.exported,
                export_style="individual",
                turn_discount=1.0,
            )
            with self.assertRaisesRegex(M0AgentServiceEntryError, "no staged"):
                PersistentM0AgentServicePreBatchEntry(directory)(crossed_event)

        positive = _source(style="individual", reward=1.0)
        with tempfile.TemporaryDirectory() as directory:
            _stage(directory, positive)
            with self.assertRaisesRegex(
                M0AgentServiceEntryError,
                "policy_failure debug contract",
            ):
                PersistentM0AgentServicePreBatchEntry(directory)(
                    _event(positive, "individual")
                )
            self.assertFalse(Path(directory, "finalized").exists())

    def test_runtime_journal_inside_git_checkout_is_rejected(self) -> None:
        arguments = _staging_only_arguments()
        forbidden = Path(__file__).resolve().parents[1] / "forbidden-m0-runtime"
        with self.assertRaisesRegex(M0AgentServiceEntryError, "outside Git checkout"):
            stage_m0_agent_service_rollout(
                journal_root=forbidden,
                **arguments,
            )
        self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
