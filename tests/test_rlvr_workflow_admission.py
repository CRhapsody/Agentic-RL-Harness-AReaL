from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from jphrl.harness.controller import HarnessState
from jphrl.harness.learning import TabularHarnessController
from jphrl.trajectory.areal_agent_service_adapter import (
    ArealAgentServiceAdapterError,
    validate_agent_service_training_record,
)
from jphrl.trajectory.areal_interaction_sidecar import (
    InteractionBinding,
    build_interaction_adapter_sidecar,
)
from jphrl.trajectory.areal_joint_bridge import (
    build_areal_joint_bridge_record,
    build_joint_version,
    deterministic_bridge_request_id,
    inference_runtime_contract_sha256,
    inject_harness_instruction,
    prompt_context_chars,
)
from jphrl.trajectory.areal_policy_admission import (
    RLVR_WORKFLOW_POLICY_ADMISSION_SCHEMA_VERSION,
    validate_policy_training_admission,
)
from jphrl.trajectory.joint_credit_alignment import (
    ESTIMATOR_VERSION,
    DualCreditEstimatorSpec,
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.rlvr_workflow_admission import (
    RLVR_FROZEN_ESTIMATOR_TEMPLATE_SCHEMA_VERSION,
    RLVR_PRE_BATCH_SCHEMA_VERSION,
    RLVR_RUNNER_ADMISSION_SCHEMA_VERSION,
    ROUTE_KIND,
    RlvrWorkflowAdmissionError,
    build_frozen_dual_credit_estimator_template,
    frozen_dual_credit_estimator_template_from_record,
    load_rlvr_workflow_runner_admission,
    load_rlvr_workflow_runner_admission_file,
    materialize_dual_credit_estimator_from_template,
    prepare_rlvr_workflow_joint_admission,
    rlvr_runner_admission_path_for_request,
    validate_rlvr_workflow_pre_batch_record,
    validate_rlvr_workflow_runner_admission,
    write_rlvr_workflow_runner_admission,
)
from jphrl.trajectory.schema import JointVersion


def _canonical_sha(record: dict[str, object], hash_field: str) -> str:
    unsigned = {key: value for key, value in record.items() if key != hash_field}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resign(record: dict[str, object], hash_field: str = "record_sha256") -> None:
    record[hash_field] = _canonical_sha(record, hash_field)


class InteractionWithTokenLogpReward:
    def __init__(
        self,
        *,
        interaction_id: str,
        model_response: object,
        reward: float,
        chat_template_type: str = "hf",
    ) -> None:
        self._interaction_id = interaction_id
        self.model_response = model_response
        self.reward = reward
        self.parent = None
        self.chat_template_type = chat_template_type
        self._tensor_dict = {
            "input_ids": [
                list(model_response.input_tokens) + list(model_response.output_tokens)
            ],
            "loss_mask": [
                [0] * len(model_response.input_tokens)
                + [1] * len(model_response.output_tokens)
            ],
            "logprobs": [
                [0.0] * len(model_response.input_tokens)
                + list(model_response.output_logprobs)
            ],
            "versions": [
                [-1] * len(model_response.input_tokens)
                + list(model_response.output_versions)
            ],
            "attention_mask": [
                [True]
                * (len(model_response.input_tokens) + len(model_response.output_tokens))
            ],
            "rewards": [reward],
        }

    @property
    def interaction_id(self) -> str:
        return self._interaction_id

    def to_tensor_dict(self) -> dict[str, object]:
        return deepcopy(self._tensor_dict)


InteractionWithTokenLogpReward.__module__ = "areal.experimental.openai.types"


@contextmanager
def _fake_areal_type_import():
    areal = ModuleType("areal")
    experimental = ModuleType("areal.experimental")
    openai = ModuleType("areal.experimental.openai")
    types = ModuleType("areal.experimental.openai.types")
    areal.__path__ = []
    experimental.__path__ = []
    openai.__path__ = []
    types.InteractionWithTokenLogpReward = InteractionWithTokenLogpReward
    with patch.dict(
        sys.modules,
        {
            "areal": areal,
            "areal.experimental": experimental,
            "areal.experimental.openai": openai,
            "areal.experimental.openai.types": types,
        },
    ):
        yield


def _runtime_contract() -> dict[str, object]:
    return {
        "schema_version": "jph.sglang-inference-runtime.v1",
        "identity": {"run_id": "rlvr-m0", "screen_pair_id": None},
        "fixed": {
            "areal_commit": "a" * 40,
            "areal_version": "2.0.0",
            "behavior_revision": "b" * 40,
            "clean_environment_policy": "filtered-inherited-v1",
            "cuda_runtime_version": "12.6",
            "cuda_visible_devices": "0",
            "dataset_revision": "c" * 40,
            "dataset_selection": "rlvr-m0-one-root-v1",
            "driver_version": "test-driver",
            "generation": {"temperature": 1.0},
            "gpu_name": "test-gpu",
            "gpu_uuid": "GPU-test",
            "physical_gpu_id": 0,
            "python_version": "3.11.0",
            "project_commit": "d" * 40,
            "rollout": {
                "backend": "sglang:d1p1t1",
                "max_concurrent_rollouts": 1,
            },
            "seed": 19,
            "server_args": {
                "base_gpu_id": 0,
                "disable_cuda_graph": False,
                "mem_fraction_static": 0.29,
                "model_path": "/allowed/model",
                "tokenizer_path": "/allowed/model",
                "tp_size": 1,
            },
            "sglang_environment": {"SGLANG_CACHE_DIR": "/allowed/cache"},
            "sglang_version": "0.5.10.post1",
            "torch_version": "2.8.0",
            "transformers_version": "4.57.1",
        },
        "treatment": {
            "generation_logprob_mode": "standard-log-of-softmax-v1",
            "sglang_return_original_logprob": False,
        },
    }


def _source(
    *,
    reward: float,
    route_kind: str = ROUTE_KIND,
    task_id: int = 7,
    controller_kind: str = "tabular",
) -> tuple[
    dict[str, object],
    InteractionWithTokenLogpReward,
    JointVersion,
    DualCreditEstimatorSpec,
]:
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
    if controller_kind == "tabular":
        controller = TabularHarnessController(seed=19)
        controller_checkpoint = controller.checkpoint()
    elif controller_kind == "torch":
        from jphrl.harness.torch_learning import (
            TorchHarnessPolicy,
            build_torch_harness_rollout_checkpoint,
        )

        controller = TorchHarnessPolicy(seed=19, hidden_size=16)
        controller_checkpoint = build_torch_harness_rollout_checkpoint(controller)
    else:
        raise ValueError(f"unknown test controller kind: {controller_kind}")
    runtime = _runtime_contract()
    request_id = deterministic_bridge_request_id(
        task_id=task_id,
        dataset_selection=runtime["fixed"]["dataset_selection"],
        base_messages=base_messages,
    )
    decision = replace(controller.choose(state), decision_id=f"{request_id}:harness:0")
    effective_messages, _ = inject_harness_instruction(base_messages, decision.action)
    joint_version = build_joint_version(
        policy_release_id="areal-sglang@model-commit:engine-v0",
        harness_controller_version=controller.version,
        areal_commit="a" * 40,
        behavior_revision="b" * 40,
        dataset_revision="c" * 40,
        dataset_selection=runtime["fixed"]["dataset_selection"],
        sglang_version="0.5.10.post1",
        generation_logprob_mode="standard-log-of-softmax-v1",
        inference_runtime_contract_sha256=inference_runtime_contract_sha256(runtime),
    )
    response = SimpleNamespace(
        input_tokens=[9, 1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
        stop_reason="stop",
    )
    interaction = InteractionWithTokenLogpReward(
        interaction_id=request_id,
        model_response=response,
        reward=reward,
    )
    episode_id = f"rlvr-m0:{request_id}"
    model_call_id = f"{episode_id}:model:0"
    sidecar = build_interaction_adapter_sidecar(
        [
            InteractionBinding(
                episode_id=episode_id,
                model_call_id=model_call_id,
                session_id=("misrouted-session" if route_kind != ROUTE_KIND else None),
                trajectory_id=(3 if route_kind != ROUTE_KIND else None),
                interaction_id=request_id,
                parent_interaction_id=None,
                ordinal=0,
                joint_version_id=joint_version.version_id,
                route_kind=route_kind,
            )
        ]
    )
    bridge = build_areal_joint_bridge_record(
        task_id=task_id,
        request_id=request_id,
        joint_version=joint_version,
        expected_policy_version=0,
        harness_state=state,
        harness_decision=decision,
        harness_controller_checkpoint=controller_checkpoint,
        base_messages=base_messages,
        effective_messages=effective_messages,
        base_input_tokens=[1, 2],
        effective_input_tokens=[9, 1, 2],
        model_response=response,
        interaction=interaction,
        tensor_dict=interaction.to_tensor_dict(),
        project_commit="d" * 40,
        areal_commit="a" * 40,
        behavior_snapshot_path="/allowed/model",
        behavior_revision="b" * 40,
        dataset_selection=runtime["fixed"]["dataset_selection"],
        sglang_version="0.5.10.post1",
        generation_logprob_mode="standard-log-of-softmax-v1",
        inference_runtime_contract=runtime,
        interaction_adapter_sidecar=sidecar,
    )
    estimator = DualCreditEstimatorSpec(
        estimator_version=ESTIMATOR_VERSION,
        parent_joint_version_id=joint_version.version_id,
        policy_source="rlvr-m0-policy-frozen-terminal-baseline-v1",
        harness_source="rlvr-m0-harness-frozen-terminal-baseline-v1",
        policy_baseline_snapshot_id="rlvr-m0-policy-baseline-run-19",
        harness_baseline_snapshot_id="rlvr-m0-harness-baseline-run-19",
        policy_baselines={model_call_id: 0.25},
        harness_baselines={decision.decision_id: 0.1},
    )
    return bridge, interaction, joint_version, estimator


class RlvrWorkflowAdmissionTests(unittest.TestCase):
    def _prepare(self, *, reward: float):
        bridge, interaction, joint_version, estimator = _source(reward=reward)
        with _fake_areal_type_import():
            result = prepare_rlvr_workflow_joint_admission(
                bridge,
                pre_batch_interaction=interaction,
                estimator=estimator,
                active_joint_version=joint_version,
            )
        return bridge, interaction, joint_version, estimator, result

    def test_positive_and_zero_reward_reach_dedicated_p_q_r_s_and_loader(self) -> None:
        for reward, validity in ((1.0, "valid"), (0.0, "policy_failure")):
            with self.subTest(reward=reward):
                bridge, _, joint_version, _, result = self._prepare(reward=reward)
                p_audit = validate_rlvr_workflow_pre_batch_record(
                    result.rlvr_pre_batch_record,
                    active_joint_version=joint_version,
                )
                q_audit = validate_policy_training_admission(
                    result.q_policy_admission,
                    active_joint_version=joint_version,
                )
                s_audit = validate_frozen_joint_credit_alignment(
                    result.s_joint_credit,
                    active_joint_version=joint_version,
                )
                runner_audit = validate_rlvr_workflow_runner_admission(
                    result.runner_admission,
                    active_joint_version=joint_version,
                )
                loaded = load_rlvr_workflow_runner_admission(
                    result.runner_admission,
                    active_joint_version=joint_version,
                )

                self.assertEqual(result.route_kind, ROUTE_KIND)
                self.assertEqual(result.episode_trace.validity_class, validity)
                self.assertEqual(result.episode_trace.reward, reward)
                self.assertEqual(
                    result.rlvr_pre_batch_record["schema_version"],
                    RLVR_PRE_BATCH_SCHEMA_VERSION,
                )
                self.assertEqual(
                    result.q_policy_admission["schema_version"],
                    RLVR_WORKFLOW_POLICY_ADMISSION_SCHEMA_VERSION,
                )
                self.assertEqual(
                    result.runner_admission["schema_version"],
                    RLVR_RUNNER_ADMISSION_SCHEMA_VERSION,
                )
                self.assertIsNone(p_audit["session_id"])
                self.assertIsNone(p_audit["trajectory_id"])
                self.assertEqual(q_audit.route_kind, ROUTE_KIND)
                self.assertEqual(q_audit.samples[0].versions[-2:], (0, 0))
                self.assertEqual(q_audit.samples[0].logprobs[-2:], (-0.1, -0.2))
                self.assertEqual(s_audit["policy_sample_count"], 1)
                self.assertEqual(s_audit["harness_action_count"], 1)
                self.assertTrue(
                    math.isclose(
                        result.s_joint_credit["policy_samples"][0]["decision_credits"][
                            0
                        ]["advantage"],
                        reward - 0.25,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        result.s_joint_credit["harness_samples"][0]["advantage"],
                        reward - 0.1,
                    )
                )
                self.assertEqual(runner_audit["route_kind"], ROUTE_KIND)
                self.assertEqual(loaded.bridge_record, bridge)
                self.assertEqual(loaded.s_joint_credit, result.s_joint_credit)
                self.assertEqual(loaded.rollout_sglang_mem_fraction_static, 0.29)
                self.assertEqual(
                    loaded.mem_fraction_static_source_path,
                    "policy_binding.inference_runtime_contract.fixed."
                    "server_args.mem_fraction_static",
                )
                for record in (
                    result.rlvr_pre_batch_record,
                    result.q_policy_admission,
                    result.r_harness_admission,
                    result.s_joint_credit,
                    result.runner_admission,
                ):
                    serialized = json.dumps(record, sort_keys=True)
                    self.assertNotIn("session_api_key", serialized)
                    self.assertNotIn("admin_api_key", serialized)
                    self.assertNotIn('"policy_optimizer_update": true', serialized)
                    self.assertNotIn('"harness_optimizer_update": true', serialized)

    def test_agent_service_route_and_receipt_schema_are_not_accepted(self) -> None:
        bridge, interaction, joint_version, estimator = _source(
            reward=0.0,
            route_kind="agent-service-session",
        )
        with (
            _fake_areal_type_import(),
            self.assertRaisesRegex(
                RlvrWorkflowAdmissionError, "rejects an Agent Service"
            ),
        ):
            prepare_rlvr_workflow_joint_admission(
                bridge,
                pre_batch_interaction=interaction,
                estimator=estimator,
                active_joint_version=joint_version,
            )

        _, _, _, _, result = self._prepare(reward=0.0)
        with self.assertRaises(ArealAgentServiceAdapterError):
            validate_agent_service_training_record(result.rlvr_pre_batch_record)

    def test_post_batch_or_wrong_live_object_fails_closed(self) -> None:
        bridge, interaction, joint_version, estimator = _source(reward=0.0)
        post_batch = interaction.to_tensor_dict()
        with (
            _fake_areal_type_import(),
            self.assertRaisesRegex(RlvrWorkflowAdmissionError, "real pre-batch"),
        ):
            prepare_rlvr_workflow_joint_admission(
                bridge,
                pre_batch_interaction=post_batch,
                estimator=estimator,
                active_joint_version=joint_version,
            )
        with (
            _fake_areal_type_import(),
            self.assertRaisesRegex(RlvrWorkflowAdmissionError, "post-batch"),
        ):
            prepare_rlvr_workflow_joint_admission(
                bridge,
                pre_batch_interaction=interaction,
                estimator=estimator,
                active_joint_version=joint_version,
                pre_batch_stage="after-concat-padded-tensors",
            )

    def test_secret_and_live_tensor_crosses_fail_closed(self) -> None:
        bridge, interaction, joint_version, estimator = _source(reward=0.0)
        secret = deepcopy(bridge)
        secret["session_api_key"] = "must-never-persist"
        _resign(secret)
        with (
            _fake_areal_type_import(),
            self.assertRaisesRegex(RlvrWorkflowAdmissionError, "credential field"),
        ):
            prepare_rlvr_workflow_joint_admission(
                secret,
                pre_batch_interaction=interaction,
                estimator=estimator,
                active_joint_version=joint_version,
            )

        interaction._tensor_dict["versions"][0][-1] = 1
        with (
            _fake_areal_type_import(),
            self.assertRaisesRegex(RlvrWorkflowAdmissionError, "tensors differ"),
        ):
            prepare_rlvr_workflow_joint_admission(
                bridge,
                pre_batch_interaction=interaction,
                estimator=estimator,
                active_joint_version=joint_version,
            )

    def test_rehashed_invalid_trace_and_runner_tampering_fail_closed(self) -> None:
        _, _, joint_version, _, result = self._prepare(reward=0.0)
        invalid = deepcopy(result.rlvr_pre_batch_record)
        invalid["episode_trace"]["validity_class"] = "infrastructure_invalid"
        invalid["episode_trace"]["reward"] = None
        invalid["source"]["episode_trace_sha256"] = _canonical_sha(
            invalid["episode_trace"], "absent"
        )
        _resign(invalid)
        with self.assertRaisesRegex(
            RlvrWorkflowAdmissionError,
            "terminal outcome|invalid episode|policy_failure",
        ):
            validate_rlvr_workflow_pre_batch_record(
                invalid, active_joint_version=joint_version
            )

        tampered = deepcopy(result.runner_admission)
        tampered["mem_fraction_static_provenance"]["value"] = 0.30
        _resign(tampered)
        with self.assertRaisesRegex(RlvrWorkflowAdmissionError, "provenance differs"):
            validate_rlvr_workflow_runner_admission(
                tampered, active_joint_version=joint_version
            )

    def test_private_o_excl_runner_envelope_and_live_workflow_ordering(self) -> None:
        _, _, joint_version, _, result = self._prepare(reward=0.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "runner-admissions"
            output_dir.mkdir(mode=0o700)
            destination = rlvr_runner_admission_path_for_request(
                output_dir=output_dir,
                request_id="request-one",
                allowed_root=root,
            )
            second_destination = rlvr_runner_admission_path_for_request(
                output_dir=output_dir,
                request_id="../../request-two",
                allowed_root=root,
            )
            self.assertNotEqual(destination, second_destination)
            self.assertEqual(destination.parent, output_dir.resolve())
            self.assertEqual(second_destination.parent, output_dir.resolve())
            self.assertNotIn("request-one", destination.name)
            self.assertNotIn("request-two", second_destination.name)
            written = write_rlvr_workflow_runner_admission(
                result.runner_admission,
                output_path=destination,
                allowed_root=root,
                active_joint_version=joint_version,
            )
            loaded = load_rlvr_workflow_runner_admission_file(
                written,
                allowed_root=root,
                active_joint_version=joint_version,
            )
            self.assertEqual(written, destination.resolve())
            self.assertEqual(written.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                loaded.s_joint_credit["record_sha256"],
                result.s_joint_credit["record_sha256"],
            )
            with self.assertRaisesRegex(RlvrWorkflowAdmissionError, "already exists"):
                write_rlvr_workflow_runner_admission(
                    result.runner_admission,
                    output_path=destination,
                    allowed_root=root,
                    active_joint_version=joint_version,
                )

        workflow = (
            Path(__file__).resolve().parents[1]
            / "jphrl"
            / "areal_joint_bridge_workflow.py"
        ).read_text(encoding="utf-8")
        prepare_call = workflow.rindex("prepare_rlvr_workflow_joint_admission(")
        write_call = workflow.rindex("write_rlvr_workflow_runner_admission(")
        return_mapping = workflow.rindex("return {request_id: interaction}")
        self.assertLess(prepare_call, write_call)
        self.assertLess(write_call, return_mapping)
        self.assertIn('admission_mode != "m0-torch-joint-v1"', workflow)
        self.assertIn("JPH_RLVR_RUNNER_ADMISSION_DIR", workflow)

    def test_one_frozen_template_materializes_two_exact_request_estimators(
        self,
    ) -> None:
        template_record = build_frozen_dual_credit_estimator_template(
            policy_source="rlvr-m0-policy-production-baseline-v1",
            harness_source="rlvr-m0-harness-production-baseline-v1",
            policy_baseline_snapshot_id="rlvr-policy-baseline-screen-19",
            harness_baseline_snapshot_id="rlvr-harness-baseline-screen-19",
            policy_baseline=0.25,
            harness_baseline=0.1,
        )
        self.assertEqual(
            template_record["schema_version"],
            RLVR_FROZEN_ESTIMATOR_TEMPLATE_SCHEMA_VERSION,
        )
        template = frozen_dual_credit_estimator_template_from_record(template_record)
        materialized = []
        for task_id in (7, 8):
            bridge, _, joint_version, _ = _source(reward=0.0, task_id=task_id)
            binding = bridge["interaction_adapter_sidecar"]["bindings"][0]
            decision_id = bridge["harness"]["decision"]["decision_id"]
            materialized.append(
                materialize_dual_credit_estimator_from_template(
                    template,
                    joint_version=joint_version,
                    model_call_id=binding["model_call_id"],
                    harness_decision_id=decision_id,
                )
            )
        self.assertNotEqual(
            set(materialized[0].policy_baselines),
            set(materialized[1].policy_baselines),
        )
        self.assertNotEqual(
            set(materialized[0].harness_baselines),
            set(materialized[1].harness_baselines),
        )
        self.assertEqual(tuple(materialized[0].policy_baselines.values()), (0.25,))
        self.assertEqual(tuple(materialized[1].harness_baselines.values()), (0.1,))

        invalid_cases = []
        wrong_schema = deepcopy(template_record)
        wrong_schema["schema_version"] = "jph.wrong-template.v1"
        _resign(wrong_schema)
        invalid_cases.append((wrong_schema, "unknown.*schema"))
        nonfinite = deepcopy(template_record)
        nonfinite["policy_baseline"] = float("nan")
        invalid_cases.append((nonfinite, "hash mismatch|finite"))
        synthetic = deepcopy(template_record)
        synthetic["policy_source"] = "synthetic-policy-fixture"
        _resign(synthetic)
        invalid_cases.append((synthetic, "synthetic or placeholder"))
        rehashed_cross = deepcopy(template_record)
        rehashed_cross["harness_baseline_snapshot_id"] = rehashed_cross[
            "policy_baseline_snapshot_id"
        ]
        _resign(rehashed_cross)
        invalid_cases.append((rehashed_cross, "snapshots must remain distinct"))
        for invalid, expected in invalid_cases:
            with (
                self.subTest(expected=expected),
                self.assertRaisesRegex(RlvrWorkflowAdmissionError, expected),
            ):
                frozen_dual_credit_estimator_template_from_record(invalid)


if __name__ == "__main__":
    unittest.main()
