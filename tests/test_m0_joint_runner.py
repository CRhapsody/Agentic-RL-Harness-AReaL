from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jphrl.experiments.m0_joint_runner import (
    AgentServiceM0JointUpdateRunner,
    AgentServiceM0SourceRecords,
    M0ArealActorSpec,
    M0CandidateEvaluator,
    M0JointRunConfig,
    M0JointRunnerError,
    RLVRM0JointUpdateRunner,
    RLVRM0SourceRecords,
    _attach_joint_safety_production_probe,
    _cleanup_production_workers,
    _destroy_actor,
    _initialize_actor_after_gpu_guard,
    _joint_safety_production_probe_sha256,
    _require_areal_update_batch,
    _source_summary,
    _StageLedger,
    _start_production_workers_after_gpu_guard,
    _validate_imported_actor_source,
    build_pinned_areal_actor_config,
    initialize_pinned_areal_actor,
    load_m0_agent_service_source_records,
    load_m0_rlvr_source_records,
)
from jphrl.training.candidate_acceptance import CandidateProbeObservation
from jphrl.trajectory.schema import JointVersion


def _pinned_areal_config_available() -> bool:
    try:
        return importlib.util.find_spec("areal.api.cli_args") is not None
    except ModuleNotFoundError:
        return False


def _version() -> JointVersion:
    return JointVersion(
        policy="policy-v0",
        harness_controller="harness-v0",
        harness_artifact="artifact-v0",
        tool_schema="tool-v0",
        parser="parser-v0",
        environment="environment-v0",
        evaluator="evaluator-v0",
        tokenizer="tokenizer-v0",
        context_builder="context-v0",
    )


def _p_record(*, route_kind: str = "agent-service-session") -> dict[str, object]:
    version = _version()
    return {
        "schema_version": "p",
        "identity": {
            "episode_id": "episode-1",
            "task_id": "task-1",
            "group_id": "group-1",
            "session_id": "session-1",
            "trajectory_id": 0,
            "joint_version_id": version.version_id,
        },
        "trace": {},
        "ready_transition": True,
        "training_archive": {
            "interaction_sidecar": {
                "bindings": [
                    {
                        "episode_id": "episode-1",
                        "model_call_id": "model-call-1",
                        "route_kind": route_kind,
                        "session_id": "session-1",
                        "trajectory_id": 0,
                        "interaction_id": "interaction-1",
                        "parent_interaction_id": None,
                        "ordinal": 0,
                        "joint_version_id": version.version_id,
                    }
                ]
            }
        },
        "evidence_scope": {
            "pre_batch_interaction_binding": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "record_sha256": "a" * 64,
    }


def _s_record(*, p_sha256: str = "a" * 64) -> dict[str, object]:
    return {
        "schema_version": "s",
        "identity": {
            "episode_id": "episode-1",
            "source_training_record_sha256": p_sha256,
        },
        "joint_version": _version().__dict__,
        "admissions": {
            "policy_admission_record": {
                "source": {"agent_service_training_record_sha256": p_sha256}
            }
        },
        "estimator": {},
        "policy_samples": [],
        "harness_samples": [],
        "summary": {},
        "evidence_scope": {
            "policy_samples_admitted": True,
            "harness_action_samples_admitted": True,
            "policy_advantages_aligned": True,
            "harness_advantages_aligned": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "record_sha256": "b" * 64,
    }


def _bridge_record(*, route_kind: str = "agent-service-session") -> dict[str, object]:
    return {
        "record_sha256": "c" * 64,
        "episode_id": "episode-1",
        "joint_version": _version().__dict__,
        "interaction_adapter_sidecar": {
            "bindings": [
                {
                    "episode_id": "episode-1",
                    "model_call_id": "model-call-1",
                    "route_kind": route_kind,
                    "session_id": "session-1"
                    if route_kind == "agent-service-session"
                    else None,
                    "trajectory_id": 0
                    if route_kind == "agent-service-session"
                    else None,
                    "interaction_id": "interaction-1",
                    "parent_interaction_id": None,
                    "ordinal": 0,
                    "joint_version_id": _version().version_id,
                }
            ]
        },
        "policy_binding": {
            "expected_inference_engine_version": 0,
            "inference_runtime_contract": {
                "fixed": {"server_args": {"mem_fraction_static": 0.29}}
            },
        },
        "harness": {"decision": {"decision_id": "harness-decision-1"}},
    }


class _Evaluator(M0CandidateEvaluator):
    def observe(self, **_kwargs):
        return ()


class M0JointRunnerScaffoldTests(unittest.TestCase):
    def _write_source_pair(
        self,
        root: Path,
        *,
        route_kind: str = "agent-service-session",
        p_sha256: str = "a" * 64,
    ) -> tuple[Path, Path, Path]:
        p_path = root / "p.json"
        s_path = root / "s.json"
        bridge_path = root / "bridge.json"
        p_path.write_text(json.dumps(_p_record(route_kind=route_kind)))
        s_path.write_text(json.dumps(_s_record(p_sha256=p_sha256)))
        bridge_path.write_text(json.dumps(_bridge_record(route_kind=route_kind)))
        return p_path, s_path, bridge_path

    def test_source_requires_separate_p_and_proves_agent_service_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            p_path, s_path, bridge_path = self._write_source_pair(root)
            with (
                patch(
                    "jphrl.experiments.m0_joint_runner."
                    "validate_agent_service_training_record",
                    return_value={
                        "episode_id": "episode-1",
                        "export_style": "individual",
                    },
                ),
                patch(
                    "jphrl.experiments.m0_joint_runner."
                    "validate_frozen_joint_credit_alignment",
                    return_value={
                        "episode_id": "episode-1",
                        "policy_model_call_ids": ["model-call-1"],
                        "harness_decision_ids": ["harness-decision-1"],
                    },
                ),
                patch(
                    "jphrl.trajectory.areal_joint_bridge."
                    "validate_areal_joint_bridge_record",
                    return_value={"ok": True},
                ),
            ):
                source = load_m0_agent_service_source_records(
                    p_training_record_path=p_path,
                    s_joint_credit_path=s_path,
                    rollout_bridge_record_path=bridge_path,
                )
            self.assertEqual(source.session_id, "session-1")
            self.assertEqual(source.trajectory_id, 0)
            self.assertEqual(source.p_record_sha256, "a" * 64)
            self.assertEqual(source.s_record_sha256, "b" * 64)
            self.assertEqual(source.rollout_bridge_sha256, "c" * 64)
            self.assertEqual(source.rollout_sglang_mem_fraction_static, 0.29)
            with self.assertRaises(TypeError):
                load_m0_agent_service_source_records(  # type: ignore[call-arg]
                    s_joint_credit_path=s_path
                )

    def test_rlvr_crossed_p_and_optimizer_claim_fail_before_training(self) -> None:
        for case in ("rlvr", "crossed", "p_claim", "s_claim"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                route = "rlvr-workflow" if case == "rlvr" else "agent-service-session"
                source_sha = "c" * 64 if case == "crossed" else "a" * 64
                p_path, s_path, bridge_path = self._write_source_pair(
                    root,
                    route_kind=route,
                    p_sha256=source_sha,
                )
                if case == "p_claim":
                    value = json.loads(p_path.read_text())
                    value["evidence_scope"]["policy_optimizer_update"] = True
                    p_path.write_text(json.dumps(value))
                if case == "s_claim":
                    value = json.loads(s_path.read_text())
                    value["evidence_scope"]["policy_optimizer_update"] = True
                    s_path.write_text(json.dumps(value))
                with (
                    patch(
                        "jphrl.experiments.m0_joint_runner."
                        "validate_agent_service_training_record",
                        return_value={
                            "episode_id": "episode-1",
                            "export_style": "concat",
                        },
                    ),
                    patch(
                        "jphrl.experiments.m0_joint_runner."
                        "validate_frozen_joint_credit_alignment",
                        return_value={
                            "episode_id": "episode-1",
                            "policy_model_call_ids": ["model-call-1"],
                            "harness_decision_ids": ["harness-decision-1"],
                        },
                    ),
                    patch(
                        "jphrl.trajectory.areal_joint_bridge."
                        "validate_areal_joint_bridge_record",
                        return_value={"ok": True},
                    ),
                    self.assertRaises(M0JointRunnerError),
                ):
                    load_m0_agent_service_source_records(
                        p_training_record_path=p_path,
                        s_joint_credit_path=s_path,
                        rollout_bridge_record_path=bridge_path,
                    )

    def test_broad_credential_name_is_rejected_before_contract_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            p_path, s_path, bridge_path = self._write_source_pair(root)
            p_record = json.loads(p_path.read_text())
            p_record["github_token"] = "must-not-persist"
            p_path.write_text(json.dumps(p_record))
            with self.assertRaisesRegex(M0JointRunnerError, "credential field"):
                load_m0_agent_service_source_records(
                    p_training_record_path=p_path,
                    s_joint_credit_path=s_path,
                    rollout_bridge_record_path=bridge_path,
                )

    def test_resource_envelope_and_stage_order_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model"
            model.mkdir()
            areal = root / "areal"
            (areal / ".git").mkdir(parents=True)
            valid = M0JointRunConfig(
                artifact_root=str(root / "run"),
                project_commit="1" * 40,
                areal_root=str(areal),
                transaction_id="m0-step-0",
                macro_step=0,
            )
            self.assertEqual(valid.validate(), root / "run")
            for fraction in (0.27, 0.31):
                with self.assertRaisesRegex(M0JointRunnerError, "mem_fraction_static"):
                    M0JointRunConfig(
                        artifact_root=str(root / f"run-{fraction}"),
                        project_commit="1" * 40,
                        areal_root=str(areal),
                        transaction_id="m0-step-0",
                        macro_step=0,
                        rollout_sglang_mem_fraction_static=fraction,
                    ).validate()
            with self.assertRaisesRegex(M0JointRunnerError, "at most 26 GiB"):
                M0JointRunConfig(
                    artifact_root=str(root / "run-too-large"),
                    project_commit="1" * 40,
                    areal_root=str(areal),
                    transaction_id="m0-step-0",
                    macro_step=0,
                    max_new_gpu_memory_gib=26.1,
                ).validate()

            ledger = _StageLedger()
            with self.assertRaisesRegex(M0JointRunnerError, "requires T"):
                ledger.complete("U")
            for stage in ("T", "U", "V", "W", "X", "Y"):
                ledger.complete(stage)
            self.assertEqual(ledger.completed, ("T", "U", "V", "W", "X", "Y"))

    def test_missing_real_y_worker_adapter_fails_before_git_or_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model"
            model.mkdir()
            areal = root / "areal"
            (areal / ".git").mkdir(parents=True)
            source = AgentServiceM0SourceRecords(
                p_training_record={},
                s_joint_credit={},
                active_joint_version=_version(),
                p_record_sha256="a" * 64,
                s_record_sha256="b" * 64,
                session_id="session-1",
                trajectory_id=0,
                export_style="individual",
                rollout_bridge_sha256="c" * 64,
                rollout_sglang_mem_fraction_static=0.29,
                expected_inference_engine_version=0,
            )
            runner = AgentServiceM0JointUpdateRunner(
                source=source,
                actor_spec=M0ArealActorSpec(
                    model_path=str(model),
                    experiment_name="m0",
                    trial_name="cpu-preflight",
                    dtype="float32",
                    attention_implementation="eager",
                    gradient_checkpointing=False,
                ),
                run_config=M0JointRunConfig(
                    artifact_root=str(root / "run"),
                    project_commit="1" * 40,
                    areal_root=str(areal),
                    transaction_id="m0-step-0",
                    macro_step=0,
                ),
                harness_behavior_checkpoint=root / "harness.json",
                acceptance_gates=(),
                evaluator=_Evaluator(),
                production_worker_factory=None,
            )
            with (
                patch(
                    "jphrl.experiments.m0_joint_runner._validate_source_checkouts"
                ) as git_check,
                self.assertRaisesRegex(
                    M0JointRunnerError,
                    "no built-in AReaL inference-service adapter",
                ),
            ):
                runner._preflight()
            git_check.assert_not_called()
            self.assertFalse((root / "run").exists())

            mismatched_source = AgentServiceM0SourceRecords(
                **{
                    **source.__dict__,
                    "rollout_sglang_mem_fraction_static": 0.28,
                }
            )
            mismatched = AgentServiceM0JointUpdateRunner(
                source=mismatched_source,
                actor_spec=runner.actor_spec,
                run_config=runner.run_config,
                harness_behavior_checkpoint=root / "harness.json",
                acceptance_gates=(),
                evaluator=_Evaluator(),
                production_worker_factory=lambda _assets: (),
            )
            with self.assertRaisesRegex(
                M0JointRunnerError,
                "differs from the recorded rollout launch contract",
            ):
                mismatched._preflight()

    def test_actor_factory_requires_explicit_single_worker_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model"
            model.mkdir()
            spec = M0ArealActorSpec(
                model_path=str(model),
                experiment_name="m0",
                trial_name="cpu-preflight",
                dtype="float32",
                attention_implementation="eager",
                gradient_checkpointing=False,
            )
            with (
                patch.dict(
                    "os.environ",
                    {
                        "WORLD_SIZE": "2",
                        "RANK": "0",
                        "LOCAL_RANK": "0",
                        "MASTER_ADDR": "127.0.0.1",
                        "MASTER_PORT": "29500",
                    },
                    clear=True,
                ),
                self.assertRaisesRegex(M0JointRunnerError, "single-worker"),
            ):
                initialize_pinned_areal_actor(spec, inference_engine_version=0)

    def test_imported_actor_source_must_be_inside_configured_areal_root(self) -> None:
        class FakeActor:
            pass

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                "os.environ",
                {"JPH_AREAL_ROOT": str(Path(directory).resolve())},
                clear=False,
            ),
            self.assertRaisesRegex(
                M0JointRunnerError,
                "outside JPH_AREAL_ROOT",
            ),
        ):
            _validate_imported_actor_source(FakeActor)

    def test_rlvr_has_separate_fail_closed_entry_and_never_mints_p(self) -> None:
        with self.assertRaisesRegex(
            M0JointRunnerError,
            "cannot mint an Agent Service P/session record",
        ):
            RLVRM0JointUpdateRunner.from_rlvr_bridge({"route_kind": "rlvr-workflow"})

    def test_rlvr_loader_consumes_dedicated_envelope_without_agent_identity(self) -> None:
        version = _version()
        pre_batch = {
            "record_sha256": "d" * 64,
            "source": {"episode_trace_sha256": "e" * 64},
        }
        s_record = {"record_sha256": "f" * 64}
        loaded = SimpleNamespace(
            route_kind="rlvr-workflow",
            joint_version=version,
            bridge_record={
                "policy_binding": {"expected_inference_engine_version": 3}
            },
            bridge_record_sha256="c" * 64,
            rlvr_pre_batch_record=pre_batch,
            episode_trace=object(),
            s_joint_credit=s_record,
            rollout_sglang_mem_fraction_static=0.29,
            mem_fraction_static_source_path=(
                "policy_binding.inference_runtime_contract.fixed."
                "server_args.mem_fraction_static"
            ),
            record_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "rlvr-runner.json"
            path.write_text(json.dumps({"route_kind": "rlvr-workflow"}))
            with patch(
                "jphrl.trajectory.rlvr_workflow_admission."
                "load_rlvr_workflow_runner_admission",
                return_value=loaded,
            ):
                source = load_m0_rlvr_source_records(
                    runner_admission_path=path,
                    active_joint_version=version,
                )
        self.assertIs(type(source), RLVRM0SourceRecords)
        self.assertEqual(source.expected_inference_engine_version, 3)
        self.assertFalse(hasattr(source, "p_training_record"))
        self.assertFalse(hasattr(source, "session_id"))
        self.assertFalse(hasattr(source, "trajectory_id"))
        summary = _source_summary(source)
        self.assertEqual(summary["route_kind"], "rlvr-workflow")
        self.assertIsNone(summary["session_id"])
        self.assertIsNone(summary["trajectory_id"])
        self.assertNotIn("p_training_record_sha256", summary)

    def test_ppo_update_batch_must_be_a_nonempty_list_of_dicts(self) -> None:
        sample = {"input_ids": object()}
        self.assertIs(_require_areal_update_batch([sample])[0], sample)
        for invalid in ({"input_ids": object()}, [], [object()]):
            with (
                self.subTest(invalid=type(invalid).__name__),
                self.assertRaisesRegex(
                    M0JointRunnerError,
                    "list of tensor dictionaries",
                ),
            ):
                _require_areal_update_batch(invalid)

    def test_y_uses_raw_production_probe_digest_not_x_aggregate(self) -> None:
        aggregate = "d" * 64
        production = "e" * 64
        report = {
            "critical_suites": [
                {
                    "spec": {"kind": "joint_safety"},
                    "probe": {
                        "output_sha256": aggregate,
                        "production_probe_output_sha256": production,
                        "production_probe_output_size_bytes": 17,
                        "observations": [
                            {
                                "production_probe_output_sha256": production,
                                "production_probe_output_size_bytes": 17,
                            }
                        ],
                    },
                }
            ]
        }
        self.assertEqual(
            _joint_safety_production_probe_sha256(report), production
        )
        del report["critical_suites"][0]["probe"][
            "production_probe_output_sha256"
        ]
        with self.assertRaisesRegex(
            M0JointRunnerError,
            "raw production-probe identity",
        ):
            _joint_safety_production_probe_sha256(report)

    def test_runner_attaches_only_framework_owned_joint_safety_probe(self) -> None:
        observation = CandidateProbeObservation(
            sample_id="joint-1",
            metric_value=1.0,
            output={"measured": True},
        )
        raw = b"live-serving-and-harness-identity"
        attached = _attach_joint_safety_production_probe(
            (observation,),
            gate_kind="joint_safety",
            production_probe_output=raw,
        )
        self.assertEqual(attached[0].production_probe_output, raw)
        crossed = CandidateProbeObservation(
            sample_id="joint-1",
            metric_value=1.0,
            output={"measured": True},
            production_probe_output=b"crossed",
        )
        with self.assertRaisesRegex(M0JointRunnerError, "crossed production"):
            _attach_joint_safety_production_probe(
                (crossed,),
                gate_kind="joint_safety",
                production_probe_output=raw,
            )

    def test_production_worker_cleanup_attempts_every_worker_and_is_separate(self) -> None:
        events: list[str] = []

        class Closable:
            def __init__(self, name: str, *, fails: bool = False) -> None:
                self.name = name
                self.fails = fails

            def close(self) -> None:
                events.append(self.name)
                if self.fails:
                    raise RuntimeError("closed with failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            success_path = root / "success.json"
            returned = _cleanup_production_workers(
                (Closable("one"), Closable("two")),
                record_path=success_path,
                summary_record_sha256="a" * 64,
                production_attestation_sha256="b" * 64,
            )
            self.assertEqual(returned, success_path)
            record = json.loads(success_path.read_text())
            self.assertTrue(record["factory_completed"])
            self.assertTrue(record["all_cleanup_calls_returned"])
            self.assertFalse(record["evidence_scope"]["gpu_process_absence_verified"])
            self.assertIsNone(record["active_release_unchanged"])
            self.assertEqual(events, ["one", "two"])

            class Store:
                def read_active(self):
                    return type("Active", (), {"release_id": "candidate-release"})()

            checked_path = root / "checked.json"
            _cleanup_production_workers(
                (Closable("checked"),),
                record_path=checked_path,
                summary_record_sha256="a" * 64,
                production_attestation_sha256="b" * 64,
                release_store=Store(),  # type: ignore[arg-type]
                expected_active_release_id="candidate-release",
            )
            checked = json.loads(checked_path.read_text())
            self.assertTrue(checked["active_release_unchanged"])
            self.assertTrue(checked["evidence_scope"]["active_release_rechecked"])

            events.clear()
            failed_path = root / "failed.json"
            with self.assertRaisesRegex(
                M0JointRunnerError,
                "cleanup failed",
            ):
                _cleanup_production_workers(
                    (Closable("bad", fails=True), Closable("still-attempted")),
                    record_path=failed_path,
                    summary_record_sha256=None,
                    production_attestation_sha256=None,
                )
            failed = json.loads(failed_path.read_text())
            self.assertFalse(failed["all_cleanup_calls_returned"])
            self.assertEqual(events, ["bad", "still-attempted"])

    def test_gpu_launch_guards_run_in_two_phase_launch_order(self) -> None:
        events: list[str] = []

        class Actor:
            def destroy(self) -> None:
                events.append("destroy:training-actor")

        actor = Actor()

        def guard(phase: str) -> None:
            events.append(f"guard:{phase}")

        def initialize(_spec, *, inference_engine_version: int):
            self.assertEqual(inference_engine_version, 7)
            events.append("launch:training-actor")
            return actor

        def factory(_assets):
            events.append("launch:production-sglang")
            return (object(),)

        with patch(
            "jphrl.experiments.m0_joint_runner.initialize_pinned_areal_actor",
            side_effect=initialize,
        ):
            initialized = _initialize_actor_after_gpu_guard(
                object(),  # type: ignore[arg-type]
                inference_engine_version=7,
                gpu_launch_guard=guard,
            )
        self.assertIs(initialized, actor)
        _destroy_actor(initialized)
        created = _start_production_workers_after_gpu_guard(
            factory,
            object(),  # type: ignore[arg-type]
            gpu_launch_guard=guard,
        )
        self.assertEqual(len(created), 1)
        self.assertEqual(
            events,
            [
                "guard:training-actor",
                "launch:training-actor",
                "destroy:training-actor",
                "guard:production-sglang",
                "launch:production-sglang",
            ],
        )

    def test_gpu_launch_guard_failure_prevents_actor_and_factory_launch(self) -> None:
        def reject(phase: str) -> None:
            raise RuntimeError(f"rejected {phase}")

        with (
            patch(
                "jphrl.experiments.m0_joint_runner."
                "initialize_pinned_areal_actor"
            ) as initialize,
            self.assertRaisesRegex(
                M0JointRunnerError,
                "rejected phase training-actor",
            ),
        ):
            _initialize_actor_after_gpu_guard(
                object(),  # type: ignore[arg-type]
                inference_engine_version=0,
                gpu_launch_guard=reject,
            )
        initialize.assert_not_called()

        factory = Mock(return_value=())
        with self.assertRaisesRegex(
            M0JointRunnerError,
            "rejected phase production-sglang",
        ):
            _start_production_workers_after_gpu_guard(
                factory,
                object(),  # type: ignore[arg-type]
                gpu_launch_guard=reject,
            )
        factory.assert_not_called()

    @unittest.skipUnless(
        _pinned_areal_config_available(),
        "pinned AReaL is unavailable in this Python environment",
    )
    def test_actor_config_is_the_pinned_dataclass_and_matches_m0_contract(self) -> None:
        from areal.api.cli_args import PPOActorConfig

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory).resolve() / "model"
            model.mkdir()
            config = build_pinned_areal_actor_config(
                M0ArealActorSpec(
                    model_path=str(model),
                    experiment_name="m0",
                    trial_name="config-contract",
                    dtype="float32",
                    attention_implementation="eager",
                    gradient_checkpointing=False,
                )
            )
        self.assertIs(type(config), PPOActorConfig)
        self.assertEqual(config.ppo_n_minibatches, 1)
        self.assertEqual(config.kl_ctl, 0.0)
        self.assertEqual(config.optimizer.lr_scheduler_type, "constant")


if __name__ == "__main__":
    unittest.main()
