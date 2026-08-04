from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from jphrl.experiments.m0_joint_runner import load_m0_rlvr_source_records
from jphrl.experiments.m0_live_evaluator import (
    M0LiveEvaluatorError,
    RealRlvrM0CandidateEvaluator,
    _summarize_policy_outputs,
)
from jphrl.training.areal_distributed_policy import JPHPPOActorController
from jphrl.trajectory.rlvr_workflow_admission import (
    prepare_rlvr_workflow_joint_admission,
)
from tests.test_rlvr_workflow_admission import _fake_areal_type_import, _source

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency-free interpreter
    torch = None


class FSDPPPOActor:
    def __init__(self) -> None:
        self.device = "cpu"

    def get_version(self) -> int:
        return 0

    def compute_logp(self, inputs):
        assert torch is not None
        return [
            torch.full(
                (1, int(item["input_ids"].shape[-1]) - 1),
                -0.25,
                dtype=torch.float32,
            )
            for item in inputs
        ]


FSDPPPOActor.__module__ = "areal.engine.fsdp_engine"


class TorchHarnessPolicy:
    def __init__(self, version: str) -> None:
        self.version = version
        self.parameter_digest = "e" * 64

    def logits_for(self, states):
        assert torch is not None
        return torch.zeros((len(states), 5), dtype=torch.float32)


TorchHarnessPolicy.__module__ = "jphrl.harness.torch_learning"


@unittest.skipIf(torch is None, "torch is unavailable")
class RealRlvrM0CandidateEvaluatorTests(unittest.TestCase):
    def _sources(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        loaded = []
        joint_version = None
        for index, task_id in enumerate((7, 8, 9)):
            bridge, interaction, current, estimator = _source(
                reward=float(index % 2), task_id=task_id
            )
            if joint_version is None:
                joint_version = current
            else:
                self.assertEqual(current, joint_version)
            with _fake_areal_type_import():
                admission = prepare_rlvr_workflow_joint_admission(
                    bridge,
                    pre_batch_interaction=interaction,
                    estimator=estimator,
                    active_joint_version=current,
                )
            path = root / f"runner-{index}.json"
            path.write_text(
                json.dumps(admission.runner_admission, sort_keys=True),
                encoding="utf-8",
            )
            loaded.append(
                load_m0_rlvr_source_records(
                    runner_admission_path=path,
                    active_joint_version=current,
                )
            )
        assert joint_version is not None
        return temporary, joint_version, loaded

    def test_real_policy_harness_joint_and_restart_observations(self) -> None:
        temporary, joint_version, sources = self._sources()
        self.addCleanup(temporary.cleanup)
        evaluator = RealRlvrM0CandidateEvaluator(
            training_source=sources[0], holdout_sources=sources[1:]
        )
        actor = FSDPPPOActor()
        harness = TorchHarnessPolicy(joint_version.harness_controller)
        gates = evaluator.acceptance_gates
        self.assertEqual(
            tuple(gate.kind for gate in gates),
            (
                "policy_heldout",
                "harness_offpolicy",
                "joint_safety",
                "restart_recovery",
            ),
        )
        for gate in gates:
            with self.subTest(kind=gate.kind):
                observations = evaluator.observe(
                    joint_version=joint_version,
                    gate=gate,
                    actor=actor,
                    harness_policy=harness,
                )
                expected = 2 if gate.kind in {
                    "policy_heldout",
                    "harness_offpolicy",
                } else 1
                self.assertEqual(len(observations), expected)
                self.assertTrue(
                    all(item.metric_value == 1.0 for item in observations)
                )
                self.assertTrue(
                    all(item.production_probe_output is None for item in observations)
                )

    def test_fixture_tamper_and_training_holdout_overlap_fail_closed(self) -> None:
        temporary, joint_version, sources = self._sources()
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(M0LiveEvaluatorError, "not disjoint"):
            RealRlvrM0CandidateEvaluator(
                training_source=sources[0], holdout_sources=(sources[0],)
            )
        evaluator = RealRlvrM0CandidateEvaluator(
            training_source=sources[0], holdout_sources=sources[1:]
        )
        gate = evaluator.acceptance_gates[0]
        crossed = type(gate)(
            kind=gate.kind,
            suite_id=gate.suite_id,
            fixture=gate.fixture + b"crossed",
            metric_name=gate.metric_name,
            minimum_score=gate.minimum_score,
            minimum_sample_count=gate.minimum_sample_count,
        )
        with self.assertRaisesRegex(M0LiveEvaluatorError, "fixture differs"):
            evaluator.observe(
                joint_version=joint_version,
                gate=crossed,
                actor=FSDPPPOActor(),
                harness_policy=TorchHarnessPolicy(
                    joint_version.harness_controller
                ),
            )

    def test_distributed_actor_rejects_missing_current_state_before_compute(self) -> None:
        temporary, joint_version, sources = self._sources()
        self.addCleanup(temporary.cleanup)
        evaluator = RealRlvrM0CandidateEvaluator(
            training_source=sources[0], holdout_sources=sources[1:]
        )
        actor = object.__new__(JPHPPOActorController)
        actor.compute_logp = Mock(return_value=[])
        actor.get_version = Mock(return_value=7)

        with self.assertRaisesRegex(
            M0LiveEvaluatorError, "current-state evidence is missing"
        ):
            evaluator.observe(
                joint_version=joint_version,
                gate=evaluator.acceptance_gates[0],
                actor=actor,
                harness_policy=TorchHarnessPolicy(
                    joint_version.harness_controller
                ),
            )
        actor.compute_logp.assert_not_called()

    def test_distributed_rtensor_outputs_are_localized_scored_and_cleared(self) -> None:
        from areal.infra.rpc.rtensor import RTensor, TensorShardInfo

        actor = Mock()
        inputs = [
            {"input_ids": torch.zeros((1, length), dtype=torch.long)}
            for length in (2, 3)
        ]
        outputs = [
            RTensor(
                shard=TensorShardInfo(
                    shard_id=f"heldout-{index}",
                    node_addr="unused-for-local-test",
                ),
                data=torch.empty(
                    tuple(item["input_ids"].shape),
                    dtype=torch.float32,
                    device="meta",
                ),
            )
            for index, item in enumerate(inputs)
        ]
        localized = [
            torch.full(
                tuple(item["input_ids"].shape),
                -0.25,
                dtype=torch.float32,
            )
            for item in inputs
        ]

        with patch.object(RTensor, "localize", return_value=localized) as localize:
            summaries = _summarize_policy_outputs(
                actor,
                inputs,
                outputs,
                distributed=True,
            )

        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(item["finite_fraction"] == 1.0 for item in summaries))
        localize.assert_called_once_with(outputs)
        actor.clear_batches.assert_called_once()
        cleared = actor.clear_batches.call_args.args
        self.assertEqual(len(cleared), 2)
        self.assertTrue(
            all(
                item["m0_x_candidate_logprobs"] is output
                for item, output in zip(cleared, outputs, strict=True)
            )
        )

    def test_distributed_tensor_or_mixed_transport_fails_closed(self) -> None:
        from areal.infra.rpc.rtensor import RTensor, TensorShardInfo

        actor = Mock()
        inputs = [{"input_ids": torch.zeros((1, 1), dtype=torch.long)}]
        remote = RTensor(
            shard=TensorShardInfo(
                shard_id="heldout-remote",
                node_addr="unused-for-local-test",
            ),
            data=torch.empty((1, 1), dtype=torch.float32, device="meta"),
        )
        for outputs in (
            [torch.tensor([[-0.25]], dtype=torch.float32)],
            [remote, torch.tensor([[-0.25]], dtype=torch.float32)],
        ):
            with self.subTest(output_types=tuple(type(item) for item in outputs)):
                with self.assertRaisesRegex(
                    M0LiveEvaluatorError,
                    "not an exact AReaL RTensor list",
                ):
                    _summarize_policy_outputs(
                        actor,
                        inputs * len(outputs),
                        outputs,
                        distributed=True,
                    )
        actor.clear_batches.assert_not_called()

    def test_distributed_rtensor_metadata_and_localized_tensor_are_strict(self) -> None:
        from areal.infra.rpc.rtensor import RTensor, TensorShardInfo

        actor = Mock()
        inputs = [
            {"input_ids": torch.zeros((1, length), dtype=torch.long)}
            for length in (2, 3)
        ]

        def remote(shard_id, shape=(1, 2), dtype=torch.float32):
            return RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id,
                    node_addr="unused-for-local-test",
                ),
                data=torch.empty(shape, dtype=dtype, device="meta"),
            )

        invalid_raw_cases = (
            ("duplicate", [remote("same"), remote("same", (1, 3))], "not unique"),
            (
                "wrong-shape",
                [remote("a", (1, 3)), remote("b", (1, 3))],
                "metadata differs",
            ),
            (
                "wrong-dtype",
                [remote("a", dtype=torch.float64), remote("b", (1, 3))],
                "metadata differs",
            ),
        )
        for label, outputs, message in invalid_raw_cases:
            with self.subTest(raw=label):
                with self.assertRaisesRegex(M0LiveEvaluatorError, message):
                    _summarize_policy_outputs(
                        actor,
                        inputs,
                        outputs,
                        distributed=True,
                    )

        valid_raw = [remote("a"), remote("b", (1, 3))]
        invalid_localized_cases = (
            (
                "crossed-shape",
                [torch.zeros((1, 3)), torch.zeros((1, 2))],
                "localized tensor differs",
            ),
            (
                "wrong-dtype",
                [torch.zeros((1, 2), dtype=torch.float64), torch.zeros((1, 3))],
                "localized tensor differs",
            ),
            (
                "non-finite",
                [torch.tensor([[float("nan"), -0.5]]), torch.zeros((1, 3))],
                "log-probability is invalid",
            ),
            (
                "positive-logprob",
                [torch.tensor([[0.1, -0.5]]), torch.zeros((1, 3))],
                "log-probability is invalid",
            ),
        )
        for label, localized, message in invalid_localized_cases:
            actor.reset_mock()
            with self.subTest(localized=label):
                with patch.object(RTensor, "localize", return_value=localized):
                    with self.assertRaisesRegex(M0LiveEvaluatorError, message):
                        _summarize_policy_outputs(
                            actor,
                            inputs,
                            valid_raw,
                            distributed=True,
                        )
                actor.clear_batches.assert_called_once()

    def test_distributed_rtensor_cleanup_failure_fails_closed(self) -> None:
        from areal.infra.rpc.rtensor import RTensor, TensorShardInfo

        actor = Mock()
        actor.clear_batches.side_effect = RuntimeError("cleanup failed")
        inputs = [{"input_ids": torch.zeros((1, 1), dtype=torch.long)}]
        output = RTensor(
            shard=TensorShardInfo(
                shard_id="heldout-cleanup",
                node_addr="unused-for-local-test",
            ),
            data=torch.empty((1, 1), dtype=torch.float32, device="meta"),
        )

        with patch.object(
            RTensor,
            "localize",
            return_value=[torch.tensor([[-0.25]], dtype=torch.float32)],
        ):
            with self.assertRaisesRegex(
                M0LiveEvaluatorError,
                "RTensor shard cleanup failed",
            ):
                _summarize_policy_outputs(
                    actor,
                    inputs,
                    [output],
                    distributed=True,
                )


if __name__ == "__main__":
    unittest.main()
