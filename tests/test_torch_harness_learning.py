from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jphrl.harness.torch_learning import (
        ACTION_IDS,
        TorchHarnessLearningError,
        TorchHarnessOptimizer,
        TorchHarnessPolicy,
        load_torch_harness_checkpoint,
        validate_torch_harness_update_evidence,
    )

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import HarnessState
from jphrl.harness.spec import HarnessAction, HarnessSpec
from jphrl.models.base import ModelResponse
from jphrl.runner import run_calculator_smoke
from jphrl.trajectory.areal_agent_service_adapter import (
    AgentServiceModelCallReceipt,
    AgentServiceSessionReceipt,
    AgentServiceTrajectoryReceipt,
    prepare_agent_service_training_record,
    validate_agent_service_training_trace,
)
from jphrl.trajectory.areal_policy_admission import build_policy_training_admission
from jphrl.trajectory.harness_action_admission import (
    admit_real_harness_action_samples,
)
from jphrl.trajectory.joint_credit_alignment import (
    ESTIMATOR_VERSION,
    DualCreditEstimatorSpec,
    build_frozen_joint_credit_alignment,
)


class _TokenBackedModel:
    policy_version = "real-policy-v7"
    tokenizer_version = "real-tokenizer-v1"

    def __init__(self) -> None:
        self.call_index = 0

    def generate(self, messages, max_new_tokens):
        del messages, max_new_tokens
        if self.call_index == 0:
            response = ModelResponse(
                text='{"tool":"calculator","expression":"17 + 25"}',
                input_token_ids=[10, 11],
                output_token_ids=[20, 21],
                output_token_logprobs=[-0.2, -0.3],
                output_versions=[7, 7],
                completion_loss_mask=[1, 1],
                policy_version=self.policy_version,
                tokenizer_version=self.tokenizer_version,
                policy_kind="causal_lm",
                token_metadata_status="available",
            )
        else:
            response = ModelResponse(
                text='{"answer":"42"}',
                input_token_ids=[10, 11, 20, 21, 30],
                output_token_ids=[40],
                output_token_logprobs=[-0.4],
                output_versions=[7],
                completion_loss_mask=[1],
                policy_version=self.policy_version,
                tokenizer_version=self.tokenizer_version,
                policy_kind="causal_lm",
                token_metadata_status="available",
            )
        self.call_index += 1
        return response


def _tensor_dict(
    *,
    input_ids: list[int],
    loss_mask: list[int],
    logprobs: list[float],
    versions: list[int],
) -> dict[str, object]:
    return {
        "input_ids": [input_ids],
        "loss_mask": [loss_mask],
        "logprobs": [logprobs],
        "versions": [versions],
        "attention_mask": [[True] * len(input_ids)],
        "rewards": [1.0],
    }


class _Interaction:
    def __init__(
        self,
        *,
        interaction_id: str,
        parent: _Interaction | None,
        input_tokens: list[int],
        output_tokens: list[int],
        output_logprobs: list[float],
        output_versions: list[int],
        tensors: dict[str, object],
    ) -> None:
        self.interaction_id = interaction_id
        self.parent = parent
        self.chat_template_type = "hf"
        self.model_response = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
        )
        self._tensors = tensors

    def to_tensor_dict(self) -> dict[str, object]:
        return self._tensors


def _interactions() -> OrderedDict[str, _Interaction]:
    first = _Interaction(
        interaction_id="interaction-1",
        parent=None,
        input_tokens=[10, 11],
        output_tokens=[20, 21],
        output_logprobs=[-0.2, -0.3],
        output_versions=[7, 7],
        tensors=_tensor_dict(
            input_ids=[10, 11, 20, 21],
            loss_mask=[0, 0, 1, 1],
            logprobs=[0.0, 0.0, -0.2, -0.3],
            versions=[-1, -1, 7, 7],
        ),
    )
    second = _Interaction(
        interaction_id="interaction-2",
        parent=first,
        input_tokens=[10, 11, 20, 21, 30],
        output_tokens=[40],
        output_logprobs=[-0.4],
        output_versions=[7],
        tensors=_tensor_dict(
            input_ids=[10, 11, 20, 21, 30, 40],
            loss_mask=[0, 0, 0, 0, 0, 1],
            logprobs=[0.0, 0.0, 0.0, 0.0, 0.0, -0.4],
            versions=[-1, -1, -1, -1, -1, 7],
        ),
    )
    return OrderedDict(
        (interaction.interaction_id, interaction) for interaction in (first, second)
    )


class _RequiredActionController:
    """Select runner-required actions through the real torch sampling path."""

    def __init__(self, policy, *, loss_mask: int = 1) -> None:
        self.policy = policy
        self.version = policy.version
        self.loss_mask = loss_mask

    def choose(self, state):
        required = (
            HarnessAction.DIRECT
            if state.verifier_status == "not-run"
            else HarnessAction.VERIFY
        )
        for _ in range(1000):
            decision = self.policy.choose(
                state,
                harness_loss_mask=self.loss_mask,
            )
            if decision.action is required:
                return decision
        raise RuntimeError("test policy did not sample the required action")


def _resign(record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(payload).hexdigest()


def _real_s_record(policy, *, baseline: float = 0.2, loss_mask: int = 1):
    controller = _RequiredActionController(policy, loss_mask=loss_mask)
    result = run_calculator_smoke(
        model=_TokenBackedModel(),
        task=TASKS["add-17-25"],
        controller=controller,
        harness_spec=HarnessSpec(),
    )
    if not result.success:
        raise AssertionError("real torch Harness trace did not complete")
    trace = result.trace
    model_call_ids = tuple(
        validate_agent_service_training_trace(trace)["model_call_ids"]
    )
    session = AgentServiceSessionReceipt(
        group_id="group-torch-u",
        session_id="session-torch-u",
    )
    calls = [
        AgentServiceModelCallReceipt(
            model_call_id=model_call_ids[0],
            interaction_id="interaction-1",
            ordinal=0,
            parent_model_call_id=None,
        ),
        AgentServiceModelCallReceipt(
            model_call_id=model_call_ids[1],
            interaction_id="interaction-2",
            ordinal=1,
            parent_model_call_id=model_call_ids[0],
        ),
    ]
    training_record = prepare_agent_service_training_record(
        trace=trace,
        session=session,
        model_calls=calls,
        trajectory=AgentServiceTrajectoryReceipt(
            session_id=session.session_id,
            trajectory_id=31,
            interaction_count=2,
            ready_transition=True,
        ),
        exported_interactions=_interactions(),
        export_style="individual",
        turn_discount=1.0,
    )
    policy_admission = build_policy_training_admission(
        training_record,
        active_joint_version=trace.joint_version,
    )
    harness_admission = admit_real_harness_action_samples(
        trace=trace,
        active_joint_version=trace.joint_version,
        pre_batch_training_record=training_record,
    )
    estimator = DualCreditEstimatorSpec(
        estimator_version=ESTIMATOR_VERSION,
        parent_joint_version_id=trace.joint_version.version_id,
        policy_source="policy-frozen-terminal-baseline-v1",
        harness_source="harness-frozen-terminal-baseline-v1",
        policy_baseline_snapshot_id="policy-baseline-snapshot-torch-u",
        harness_baseline_snapshot_id="harness-baseline-snapshot-torch-u",
        policy_baselines={model_call_id: 0.25 for model_call_id in model_call_ids},
        harness_baselines={
            action.decision_id: baseline for action in harness_admission.actions
        },
    )
    return (
        build_frozen_joint_credit_alignment(
            policy_admission=policy_admission,
            harness_admission=harness_admission,
            active_joint_version=trace.joint_version,
            estimator=estimator,
        ),
        trace.joint_version,
    )


@unittest.skipUnless(
    torch is not None,
    "torch is required for production Harness tests",
)
class TorchHarnessLearningTests(unittest.TestCase):
    def test_choose_emits_complete_stable_five_action_decision(self) -> None:
        policy = TorchHarnessPolicy(seed=17, hidden_size=16)
        state = HarnessState(
            turn=2,
            remaining_tool_calls=1,
            remaining_model_retries=0,
            context_chars=512,
            last_error=None,
            retrieval_hit=True,
            verifier_status="decision",
            task_domain="calculator",
        )
        mask = (True, False, True, True, False)
        decision = policy.choose(state, action_mask=mask)

        self.assertEqual(decision.action_ids, ACTION_IDS)
        self.assertEqual(decision.action_mask, mask)
        self.assertTrue(mask[ACTION_IDS.index(decision.action.value)])
        self.assertEqual(len(decision.pre_mask_logits), 5)
        self.assertTrue(all(math.isfinite(value) for value in decision.pre_mask_logits))
        self.assertLessEqual(decision.old_harness_logprob, 0.0)
        self.assertEqual(decision.controller_version, policy.version)
        self.assertEqual(decision.harness_loss_mask, 1)

    def test_real_s_record_runs_one_candidate_adam_update_and_round_trip(self) -> None:
        policy = TorchHarnessPolicy(seed=29, hidden_size=16)
        record, version = _real_s_record(policy)
        digest_before = policy.parameter_digest
        generator_before = policy._generator.get_state().clone()
        sample_count_before = policy.sample_count

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "harness-candidate.pt"
            result = TorchHarnessOptimizer(
                policy,
                learning_rate=1e-2,
                clip_ratio=0.2,
            ).update_from_frozen_joint_credit(
                record,
                active_joint_version=version,
                checkpoint_path=path,
            )
            restored, optimizer, _ = load_torch_harness_checkpoint(path)

            self.assertTrue(path.is_file())
            self.assertEqual(policy.parameter_digest, digest_before)
            self.assertEqual(policy.sample_count, sample_count_before)
            self.assertTrue(
                torch.equal(policy._generator.get_state(), generator_before)
            )
            self.assertNotEqual(result.candidate_policy.parameter_digest, digest_before)
            self.assertEqual(
                restored.parameter_digest,
                result.candidate_policy.parameter_digest,
            )
            self.assertTrue(optimizer.state)
            self.assertTrue(math.isfinite(result.evidence.gradient_norm))
            self.assertGreater(result.evidence.gradient_norm, 0.0)
            self.assertEqual(result.evidence.effective_batch_size, 2)
            evidence = result.evidence.to_record()
            audit = validate_torch_harness_update_evidence(
                json.loads(json.dumps(evidence)),
                active_joint_version=version,
            )
            self.assertTrue(evidence["evidence_scope"]["harness_optimizer_update"])
            self.assertFalse(evidence["evidence_scope"]["policy_optimizer_update"])
            self.assertEqual(audit.optimizer_step_after, 1)

            false_claim = deepcopy(evidence)
            false_claim["evidence_scope"]["policy_optimizer_update"] = True
            _resign(false_claim)
            with self.assertRaisesRegex(
                TorchHarnessLearningError,
                "evidence scope",
            ):
                validate_torch_harness_update_evidence(false_claim)

            probe = HarnessState(
                turn=3,
                remaining_tool_calls=0,
                remaining_model_retries=0,
                context_chars=64,
                last_error=None,
                retrieval_hit=False,
                verifier_status="probe",
                task_domain="calculator",
            )
            expected = result.candidate_policy.choose(probe)
            actual = restored.choose(probe)
            self.assertEqual(expected.action, actual.action)
            self.assertEqual(expected.old_harness_logprob, actual.old_harness_logprob)

    def test_invalid_stale_logprob_mask_zero_credit_and_credential_fail(self) -> None:
        policy = TorchHarnessPolicy(seed=41, hidden_size=16)
        record, version = _real_s_record(policy)
        optimizer = TorchHarnessOptimizer(policy, learning_rate=1e-2)

        invalid = deepcopy(record)
        invalid["record_sha256"] = "0" * 64

        invalid_episode = deepcopy(record)
        invalid_harness = invalid_episode["admissions"]["harness_admission_record"]
        invalid_harness["terminal_outcome"]["validity_class"] = "infrastructure_invalid"
        _resign(invalid_harness)
        _resign(invalid_episode)

        logprob = deepcopy(record)
        logprob["harness_samples"][0]["action"]["old_harness_logprob"] += 0.1
        _resign(logprob)

        masked = deepcopy(record)
        action = masked["harness_samples"][0]["action"]
        selected = action["action_ids"].index(action["action"])
        tampered_mask = list(action["action_mask"])
        tampered_mask[selected] = False
        action["action_mask"] = tampered_mask
        _resign(masked)

        credential = deepcopy(record)
        credential["session_api_key"] = "must-not-enter"

        stale = replace(version, environment="stale-environment")
        cases = (
            (invalid, version, "invalid.pt", "hash"),
            (invalid_episode, version, "invalid-episode.pt", "not trainable"),
            (record, stale, "stale.pt", "JointVersion"),
            (logprob, version, "logprob.pt", "Harness"),
            (masked, version, "mask.pt", "Harness"),
            (credential, version, "credential.pt", "credential"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for value, active, name, message in cases:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(TorchHarnessLearningError, message),
                ):
                    optimizer.update_from_frozen_joint_credit(
                        value,
                        active_joint_version=active,
                        checkpoint_path=Path(directory) / name,
                    )

        zero_policy = TorchHarnessPolicy(seed=43, hidden_size=16)
        zero_record, zero_version = _real_s_record(zero_policy, baseline=1.0)
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(TorchHarnessLearningError, "zero effective credit"),
        ):
            TorchHarnessOptimizer(zero_policy).update_from_frozen_joint_credit(
                zero_record,
                active_joint_version=zero_version,
                checkpoint_path=Path(directory) / "zero.pt",
            )

    def test_zero_loss_mask_and_behavior_policy_mismatch_fail_closed(self) -> None:
        masked_policy = TorchHarnessPolicy(seed=47, hidden_size=16)
        masked_record, masked_version = _real_s_record(masked_policy, loss_mask=0)
        other_policy = TorchHarnessPolicy(seed=53, hidden_size=16)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TorchHarnessLearningError, "zero trainable"):
                TorchHarnessOptimizer(masked_policy).update_from_frozen_joint_credit(
                    masked_record,
                    active_joint_version=masked_version,
                    checkpoint_path=Path(directory) / "masked.pt",
                )
            with self.assertRaisesRegex(TorchHarnessLearningError, "behavior version"):
                TorchHarnessOptimizer(other_policy).update_from_frozen_joint_credit(
                    masked_record,
                    active_joint_version=masked_version,
                    checkpoint_path=Path(directory) / "behavior.pt",
                )


if __name__ == "__main__":
    unittest.main()
