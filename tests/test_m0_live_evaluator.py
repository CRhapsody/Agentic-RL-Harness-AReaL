from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from jphrl.experiments.m0_joint_runner import load_m0_rlvr_source_records
from jphrl.experiments.m0_live_evaluator import (
    M0LiveEvaluatorError,
    RealRlvrM0CandidateEvaluator,
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


if __name__ == "__main__":
    unittest.main()
