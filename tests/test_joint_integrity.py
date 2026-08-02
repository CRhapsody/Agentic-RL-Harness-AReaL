from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from jphrl.experiments.g1_integrity import (
    build_contract_episode,
    initial_checkpoint,
    run_experiment,
)
from jphrl.harness.controller import HarnessState, SmokeHarnessController
from jphrl.harness.learning import TabularHarnessController
from jphrl.joint_release import (
    CandidateArtifact,
    ConcurrentPublishError,
    JointReleaseStore,
    read_joint_checkpoint,
    write_joint_checkpoint,
)
from jphrl.trajectory.joint_batch import (
    DecisionCredit,
    EpisodeCredit,
    build_joint_decision_batch,
)
from scripts.verify_g1_integrity import verify


def harness_state() -> HarnessState:
    return HarnessState(
        turn=0,
        remaining_tool_calls=1,
        remaining_model_retries=1,
        context_chars=64,
        last_error=None,
        retrieval_hit=False,
        verifier_status="decision",
        task_domain="test",
    )


class JointIntegrityTests(unittest.TestCase):
    def test_policy_and_harness_streams_are_separate(self) -> None:
        checkpoint = initial_checkpoint()
        trace, credits = build_contract_episode(1, checkpoint.joint_version)
        batch = build_joint_decision_batch(
            [trace],
            {trace.episode_id: credits},
            allow_open_fixtures=True,
        )

        self.assertEqual(len(batch.policy_tokens), 2)
        self.assertEqual(len(batch.harness_actions), 1)
        self.assertEqual(batch.policy_tokens[0].policy_loss_mask, 1)
        self.assertEqual(batch.harness_actions[0].harness_loss_mask, 1)
        self.assertEqual(
            batch.policy_tokens[0].policy_release_id,
            checkpoint.joint_version.policy,
        )
        self.assertEqual(
            batch.harness_actions[0].harness_behavior_version,
            checkpoint.joint_version.harness_controller,
        )
        self.assertNotEqual(
            batch.policy_tokens[0].credit_source,
            batch.harness_actions[0].credit_source,
        )
        self.assertEqual(batch.policy_tokens[0].inference_engine_version, 0)

    def test_open_fixture_is_rejected_by_production_batch_builder(self) -> None:
        checkpoint = initial_checkpoint()
        trace, credits = build_contract_episode(1, checkpoint.joint_version)
        with self.assertRaisesRegex(ValueError, "closed valid episode"):
            build_joint_decision_batch(
                [trace],
                {trace.episode_id: credits},
            )

    def test_crossed_credit_targets_fail_closed(self) -> None:
        checkpoint = initial_checkpoint()
        trace, _ = build_contract_episode(2, checkpoint.joint_version)
        episode_id = trace.episode_id
        crossed = EpisodeCredit(
            policy_calls={
                f"{episode_id}:harness:0": DecisionCredit(1.0, "wrong-policy-target")
            },
            harness_decisions={
                f"{episode_id}:model:0": DecisionCredit(1.0, "wrong-harness-target")
            },
        )
        with self.assertRaisesRegex(ValueError, "no Harness credit"):
            build_joint_decision_batch(
                [trace],
                {episode_id: crossed},
                allow_open_fixtures=True,
            )

    def test_batch_rejects_multiple_behavior_versions(self) -> None:
        checkpoint = initial_checkpoint()
        first, first_credit = build_contract_episode(3, checkpoint.joint_version)
        second_version = replace(
            checkpoint.joint_version,
            policy="other-policy",
            harness_controller="other-harness",
        )
        second, second_credit = build_contract_episode(4, second_version)
        with self.assertRaisesRegex(ValueError, "more than one joint behavior version"):
            build_joint_decision_batch(
                [first, second],
                {
                    first.episode_id: first_credit,
                    second.episode_id: second_credit,
                },
                allow_open_fixtures=True,
            )

    def test_frozen_and_trainable_harness_masks_differ(self) -> None:
        state = harness_state()
        self.assertEqual(SmokeHarnessController().choose(state).harness_loss_mask, 0)
        self.assertEqual(
            TabularHarnessController(seed=1).choose(state).harness_loss_mask,
            1,
        )

    def test_tabular_harness_checkpoint_restores_next_decision(self) -> None:
        state = harness_state()
        controller = TabularHarnessController(seed=17)
        controller.choose(state)
        serialized = json.loads(json.dumps(controller.checkpoint()))
        restored = TabularHarnessController.from_checkpoint(serialized)
        self.assertEqual(controller.choose(state), restored.choose(state))

    def test_joint_checkpoint_roundtrip_and_publish_cas(self) -> None:
        checkpoint = initial_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.json"
            write_joint_checkpoint(checkpoint_path, checkpoint)
            self.assertEqual(read_joint_checkpoint(checkpoint_path), checkpoint)
            corrupted_path = root / "corrupted-checkpoint.json"
            corrupted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            corrupted["checkpoint"]["policy"]["parameters"][0] = 999.0
            corrupted_path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "state hash|digest"):
                read_joint_checkpoint(corrupted_path)

            store = JointReleaseStore(root / "release")
            policy = CandidateArtifact(
                "policy", checkpoint.policy.version, checkpoint.policy.to_dict()
            )
            harness = CandidateArtifact(
                "harness", checkpoint.harness.version, checkpoint.harness.to_dict()
            )
            release = store.publish(
                joint_version=checkpoint.joint_version,
                policy=policy,
                harness=harness,
                expected_active_release_id=None,
            )
            with self.assertRaisesRegex(ConcurrentPublishError, "active release changed"):
                store.publish(
                    joint_version=checkpoint.joint_version,
                    policy=policy,
                    harness=harness,
                    expected_active_release_id="stale-release",
                )
            self.assertEqual(store.read_active(), release)

    def test_full_g1_integrity_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_experiment(
                version_fixtures=1000,
                work_dir=root / "work",
                output=root / "result.json",
            )
            audit = verify(root / "result.json", root / "audit.json")
        self.assertTrue(result["passed"])
        self.assertTrue(audit["passed"])
        self.assertEqual(result["mixed_version"]["mixed_version_episodes"], 0)
        self.assertEqual(
            result["mixed_version"]["synthetic_fixtures_ended"], 1000
        )
        self.assertEqual(result["mixed_version"]["straddled_publish"], 100)
        self.assertEqual(result["mixed_version"]["stale_accepted_at_lag_0"], 0)
        self.assertEqual(len(result["atomic_publish"]["cases"]), 6)
        self.assertTrue(result["checkpoint_replay"]["next_step_equal"])
        self.assertTrue(
            result["checkpoint_replay"]["tabular_harness"]["exact_next_decision"]
        )
        self.assertTrue(
            result["credit_separation"]["update_interventions"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
