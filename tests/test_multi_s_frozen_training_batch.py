from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jphrl.runner as runner_module
from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import HarnessDecision
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
from jphrl.trajectory.multi_s_frozen_training_batch import (
    MultiSFrozenTrainingBatchError,
    iter_member_s_records,
    load_multi_s_frozen_training_batch,
    multi_s_source_binding,
    persist_multi_s_frozen_training_batch,
    prepare_multi_s_frozen_training_batch,
    required_v_member_claims,
    validate_multi_s_frozen_training_batch,
    validate_multi_s_source_binding,
    validate_v_member_claim_coverage,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _resign(record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()


class TokenBackedCalculatorModel:
    tokenizer_version = "real-tokenizer-v1"

    def __init__(self, policy_version: str = "real-policy-v7") -> None:
        self.policy_version = policy_version
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


class UniqueLearnableHarnessController:
    version = "live-categorical-harness-v1"

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.ordinal = 0

    def choose(self, state):
        action_ids = tuple(action.value for action in HarnessAction)
        action = (
            HarnessAction.DIRECT
            if state.verifier_status == "not-run"
            else HarnessAction.VERIFY
        )
        action_index = action_ids.index(action.value)
        logits = tuple(
            1.25 if index == action_index else -0.25
            for index in range(len(action_ids))
        )
        maximum = max(logits)
        normalizer = maximum + math.log(
            sum(math.exp(value - maximum) for value in logits)
        )
        decision = HarnessDecision(
            decision_id=f"{self.prefix}-decision-{self.ordinal}",
            action=action,
            old_harness_logprob=logits[action_index] - normalizer,
            controller_version=self.version,
            action_ids=action_ids,
            action_mask=(True,) * len(action_ids),
            pre_mask_logits=logits,
            harness_loss_mask=1,
        )
        self.ordinal += 1
        return decision


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


class FakeInteraction:
    def __init__(
        self,
        *,
        interaction_id: str,
        parent: FakeInteraction | None,
        chat_template_type: str,
        input_tokens: list[int],
        output_tokens: list[int],
        output_logprobs: list[float],
        output_versions: list[int],
        tensors: dict[str, object],
    ) -> None:
        self.interaction_id = interaction_id
        self.parent = parent
        self.chat_template_type = chat_template_type
        self.model_response = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
        )
        self._tensors = tensors

    def to_tensor_dict(self) -> dict[str, object]:
        return self._tensors


def _interactions(prefix: str, style: str) -> OrderedDict[str, FakeInteraction]:
    template_type = "concat" if style == "concat" else "hf"
    first = FakeInteraction(
        interaction_id=f"{prefix}-interaction-1",
        parent=None,
        chat_template_type=template_type,
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
    if style == "concat":
        second_mask = [0, 0, 1, 1, 0, 1]
        second_logprobs = [0.0, 0.0, -0.2, -0.3, 0.0, -0.4]
        second_versions = [-1, -1, 7, 7, -1, 7]
    else:
        second_mask = [0, 0, 0, 0, 0, 1]
        second_logprobs = [0.0, 0.0, 0.0, 0.0, 0.0, -0.4]
        second_versions = [-1, -1, -1, -1, -1, 7]
    second = FakeInteraction(
        interaction_id=f"{prefix}-interaction-2",
        parent=first,
        chat_template_type=template_type,
        input_tokens=[10, 11, 20, 21, 30],
        output_tokens=[40],
        output_logprobs=[-0.4],
        output_versions=[7],
        tensors=_tensor_dict(
            input_ids=[10, 11, 20, 21, 30, 40],
            loss_mask=second_mask,
            logprobs=second_logprobs,
            versions=second_versions,
        ),
    )
    if style == "concat":
        return OrderedDict(((second.interaction_id, second),))
    return OrderedDict(
        (interaction.interaction_id, interaction) for interaction in (first, second)
    )


def _build_s_record(
    index: int,
    *,
    style: str = "individual",
    policy_version: str = "real-policy-v7",
    decision_prefix: str | None = None,
    shared_model_calls: bool = False,
):
    prefix = f"episode-{index}"
    original_request_model = runner_module._request_model

    def _request_with_shared_call_ids(
        trace,
        model,
        phase,
        attempt,
        messages,
        max_new_tokens,
    ):
        old_call_id, response, failure_class, error = original_request_model(
            trace,
            model,
            phase,
            attempt,
            messages,
            max_new_tokens,
        )
        shared_call_id = f"shared-model-call:{phase}:{attempt}"
        for event in trace.events:
            if event.payload.get("model_call_id") == old_call_id:
                event.payload["model_call_id"] = shared_call_id
        return shared_call_id, response, failure_class, error

    context = (
        patch.object(runner_module, "_request_model", _request_with_shared_call_ids)
        if shared_model_calls
        else patch.object(runner_module, "_request_model", original_request_model)
    )
    with context:
        result = run_calculator_smoke(
            model=TokenBackedCalculatorModel(policy_version),
            task=TASKS["add-17-25"],
            controller=UniqueLearnableHarnessController(decision_prefix or prefix),
            harness_spec=HarnessSpec(),
            seed=index,
        )
    trace = result.trace
    model_call_ids = tuple(
        validate_agent_service_training_trace(trace)["model_call_ids"]
    )
    interactions = _interactions(prefix, style)
    interaction_ids = [f"{prefix}-interaction-1", f"{prefix}-interaction-2"]
    session = AgentServiceSessionReceipt(
        group_id="group-multi-s",
        session_id=f"session-multi-s-{index}",
    )
    calls = [
        AgentServiceModelCallReceipt(
            model_call_id=model_call_ids[0],
            interaction_id=interaction_ids[0],
            ordinal=0,
            parent_model_call_id=None,
        ),
        AgentServiceModelCallReceipt(
            model_call_id=model_call_ids[1],
            interaction_id=interaction_ids[1],
            ordinal=1,
            parent_model_call_id=model_call_ids[0],
        ),
    ]
    trajectory = AgentServiceTrajectoryReceipt(
        session_id=session.session_id,
        trajectory_id=index + 1,
        interaction_count=2,
        ready_transition=True,
    )
    training_record = prepare_agent_service_training_record(
        trace=trace,
        session=session,
        model_calls=calls,
        trajectory=trajectory,
        exported_interactions=interactions,
        export_style=style,
        turn_discount=1.0,
    )
    policy = build_policy_training_admission(
        training_record,
        active_joint_version=trace.joint_version,
    )
    harness = admit_real_harness_action_samples(
        trace=trace,
        active_joint_version=trace.joint_version,
        pre_batch_training_record=training_record,
    )
    estimator = DualCreditEstimatorSpec(
        estimator_version=ESTIMATOR_VERSION,
        parent_joint_version_id=trace.joint_version.version_id,
        policy_source="policy-frozen-terminal-baseline-v1",
        harness_source="harness-frozen-terminal-baseline-v1",
        policy_baseline_snapshot_id="policy-baseline-snapshot-7",
        harness_baseline_snapshot_id="harness-baseline-snapshot-3",
        policy_baselines={
            model_call_id: 0.25 + 0.25 * ordinal
            for ordinal, model_call_id in enumerate(model_call_ids)
        },
        harness_baselines={
            action.decision_id: 0.1 + 0.1 * ordinal
            for ordinal, action in enumerate(harness.actions)
        },
    )
    return (
        build_frozen_joint_credit_alignment(
            policy_admission=policy,
            harness_admission=harness.to_record(),
            active_joint_version=trace.joint_version,
            estimator=estimator,
        ),
        trace.joint_version,
    )


def _persist_s_records(
    directory: Path,
    *,
    styles: tuple[str, ...] = ("individual",) * 4,
    policy_versions: tuple[str, ...] = ("real-policy-v7",) * 4,
) -> tuple[list[Path], object]:
    paths: list[Path] = []
    active_joint_version = None
    for index, (style, policy_version) in enumerate(zip(styles, policy_versions)):
        record, joint_version = _build_s_record(
            index,
            style=style,
            policy_version=policy_version,
        )
        if active_joint_version is None:
            active_joint_version = joint_version
        path = directory / f"s-{index}.json"
        path.write_bytes(_canonical_json(record) + b"\n")
        paths.append(path)
    return paths, active_joint_version


class MultiSFrozenTrainingBatchTests(unittest.TestCase):
    def test_four_individual_records_persist_load_and_expose_t_u_v_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            output = directory / "runtime" / "canonical-multi-s.json"

            batch = persist_multi_s_frozen_training_batch(
                reversed(paths),
                output,
                active_joint_version=joint_version,
            )
            loaded = load_multi_s_frozen_training_batch(
                output,
                active_joint_version=joint_version,
            )
            consumed = tuple(iter_member_s_records(loaded))
            claims = required_v_member_claims(loaded)

            self.assertEqual(batch, loaded)
            self.assertEqual(len(loaded.members), 4)
            self.assertEqual(loaded.policy_sample_count, 8)
            self.assertEqual(loaded.harness_action_count, 8)
            self.assertEqual(
                tuple(member.member_claim_sha256 for member in loaded.members),
                tuple(claim for claim, _ in consumed),
            )
            self.assertEqual(
                tuple(member.s_record_sha256 for member in loaded.members),
                claims,
            )
            self.assertEqual(validate_v_member_claim_coverage(loaded, claims), claims)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(persisted["evidence_scope"]["policy_optimizer_update"])
            self.assertFalse(persisted["evidence_scope"]["harness_optimizer_update"])
            self.assertFalse(persisted["evidence_scope"]["joint_optimizer_barrier"])
            for member, (_, exact_s) in zip(persisted["members"], consumed):
                self.assertEqual(member["s_record"], exact_s)

    def test_input_order_does_not_change_canonical_batch_or_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)

            forward = prepare_multi_s_frozen_training_batch(
                paths,
                active_joint_version=joint_version,
            )
            reverse = prepare_multi_s_frozen_training_batch(
                list(reversed(paths)),
                active_joint_version=joint_version,
            )

            self.assertEqual(forward, reverse)
            self.assertEqual(
                forward["aggregate_sha256"],
                reverse["aggregate_sha256"],
            )

    def test_o_excl_refuses_to_replace_existing_batch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            output = directory / "canonical-multi-s.json"
            persist_multi_s_frozen_training_batch(
                paths,
                output,
                active_joint_version=joint_version,
            )

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "exclusively persist",
            ):
                persist_multi_s_frozen_training_batch(
                    paths,
                    output,
                    active_joint_version=joint_version,
                )

    def test_less_than_four_records_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "at least 4",
            ):
                prepare_multi_s_frozen_training_batch(
                    paths[:3],
                    active_joint_version=joint_version,
                )

    def test_same_persisted_member_twice_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "same persisted S path",
            ):
                prepare_multi_s_frozen_training_batch(
                    [paths[0], paths[1], paths[2], paths[2]],
                    active_joint_version=joint_version,
                )

    def test_copy_of_same_s_at_different_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            copied = directory / "copied-s.json"
            copied.write_bytes(paths[2].read_bytes())

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "duplicate episode|same persisted S member/source training record|same persisted S member digest",
            ):
                prepare_multi_s_frozen_training_batch(
                    [paths[0], paths[1], paths[2], copied],
                    active_joint_version=joint_version,
                )

    def test_cross_joint_version_member_fails_lag_zero_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(
                directory,
                policy_versions=(
                    "real-policy-v7",
                    "real-policy-v7",
                    "real-policy-v7",
                    "real-policy-v8",
                ),
            )

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "JointVersion differs from lag-zero",
            ):
                prepare_multi_s_frozen_training_batch(
                    paths,
                    active_joint_version=joint_version,
                )

    def test_different_episodes_reusing_model_call_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths: list[Path] = []
            joint_version = None
            for index in range(4):
                record, current_joint_version = _build_s_record(
                    index,
                    shared_model_calls=index >= 2,
                )
                joint_version = joint_version or current_joint_version
                path = directory / f"s-{index}.json"
                path.write_bytes(_canonical_json(record) + b"\n")
                paths.append(path)

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "duplicate model-call ID across",
            ):
                prepare_multi_s_frozen_training_batch(
                    paths,
                    active_joint_version=joint_version,
                )

    def test_invalid_s_member_fails_its_existing_validator(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            invalid = json.loads(paths[3].read_text(encoding="utf-8"))
            invalid["summary"]["policy_sample_count"] = 999
            _resign(invalid)
            paths[3].write_bytes(_canonical_json(invalid) + b"\n")

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "failed frozen-credit validation",
            ):
                prepare_multi_s_frozen_training_batch(
                    paths,
                    active_joint_version=joint_version,
                )

    def test_secret_field_fails_before_it_can_enter_batch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            secret = json.loads(paths[3].read_text(encoding="utf-8"))
            secret["session_api_key"] = "must-not-persist"
            paths[3].write_bytes(_canonical_json(secret) + b"\n")

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "credential field",
            ):
                prepare_multi_s_frozen_training_batch(
                    paths,
                    active_joint_version=joint_version,
                )

    def test_concat_member_is_rejected_even_when_s_is_otherwise_valid(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(
                directory,
                styles=("individual", "individual", "individual", "concat"),
            )

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "rejects concat/post-batch",
            ):
                prepare_multi_s_frozen_training_batch(
                    paths,
                    active_joint_version=joint_version,
                )

    def test_post_batch_object_cannot_be_used_to_guess_member_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            merged_post_batch = {
                "input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "attention_mask": [[True, True]] * 4,
            }

            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "persisted S paths only",
            ):
                prepare_multi_s_frozen_training_batch(
                    [merged_post_batch, paths[1], paths[2], paths[3]],
                    active_joint_version=joint_version,
                )

    def test_loader_detects_changed_source_and_v_requires_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            output = directory / "canonical-multi-s.json"
            batch = persist_multi_s_frozen_training_batch(
                paths,
                output,
                active_joint_version=joint_version,
            )
            claims = required_v_member_claims(batch)

            for bad_claims in (claims[:-1], claims + (claims[-1],), tuple(reversed(claims))):
                with self.subTest(claim_count=len(bad_claims)):
                    with self.assertRaisesRegex(
                        MultiSFrozenTrainingBatchError,
                        "cover the ordered",
                    ):
                        validate_v_member_claim_coverage(batch, bad_claims)

            original = paths[0].read_bytes()
            paths[0].write_bytes(original + b" ")
            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "source file hash mismatch",
            ):
                load_multi_s_frozen_training_batch(
                    output,
                    active_joint_version=joint_version,
                )

    def test_v_stable_s_digest_ledger_rejects_cross_batch_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(
                directory,
                styles=("individual",) * 7,
                policy_versions=("real-policy-v7",) * 7,
            )
            copied = directory / "same-s-new-path.json"
            copied.write_bytes(paths[0].read_bytes())
            first = validate_multi_s_frozen_training_batch(
                prepare_multi_s_frozen_training_batch(
                    paths[:4],
                    active_joint_version=joint_version,
                ),
                active_joint_version=joint_version,
            )
            second = validate_multi_s_frozen_training_batch(
                prepare_multi_s_frozen_training_batch(
                    [copied, paths[4], paths[5], paths[6]],
                    active_joint_version=joint_version,
                ),
                active_joint_version=joint_version,
            )
            first_claims = required_v_member_claims(first)
            second_claims = required_v_member_claims(second)

            self.assertEqual(paths[0].read_bytes(), copied.read_bytes())
            self.assertTrue(set(first_claims) & set(second_claims))
            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "prior batch",
            ):
                validate_v_member_claim_coverage(
                    second,
                    second_claims,
                    already_claimed_s_record_sha256s=first_claims,
                )

    def test_source_binding_covers_ordered_envelope_and_stable_claims(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            batch = validate_multi_s_frozen_training_batch(
                prepare_multi_s_frozen_training_batch(
                    paths,
                    active_joint_version=joint_version,
                ),
                active_joint_version=joint_version,
            )
            binding = multi_s_source_binding(batch)

            self.assertEqual(binding["joint_version_id"], joint_version.version_id)
            self.assertEqual(binding["batch_record_sha256"], batch.record_sha256)
            self.assertEqual(
                binding["member_claim_sha256s"],
                [member.member_claim_sha256 for member in batch.members],
            )
            self.assertEqual(
                binding["s_record_sha256s"], list(required_v_member_claims(batch))
            )
            self.assertEqual(
                validate_multi_s_source_binding(binding, batch=batch), binding
            )

            reordered = deepcopy(binding)
            reordered["s_record_sha256s"] = list(
                reversed(reordered["s_record_sha256s"])
            )
            _resign(reordered)
            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "differs from batch",
            ):
                validate_multi_s_source_binding(reordered, batch=batch)

            duplicate = deepcopy(binding)
            duplicate["member_claim_sha256s"][1] = duplicate[
                "member_claim_sha256s"
            ][0]
            _resign(duplicate)
            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "member claims are invalid",
            ):
                validate_multi_s_source_binding(duplicate)

    def test_in_memory_validator_rejects_tampered_aggregate_and_secret(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            paths, joint_version = _persist_s_records(directory)
            record = prepare_multi_s_frozen_training_batch(
                paths,
                active_joint_version=joint_version,
            )

            aggregate_tamper = deepcopy(record)
            aggregate_tamper["aggregate_sha256"] = "0" * 64
            _resign(aggregate_tamper)
            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "aggregate hash mismatch",
            ):
                validate_multi_s_frozen_training_batch(
                    aggregate_tamper,
                    active_joint_version=joint_version,
                )

            secret_tamper = deepcopy(record)
            secret_tamper["members"][0]["s_record"]["admin_api_key"] = "hidden"
            _resign(secret_tamper)
            with self.assertRaisesRegex(
                MultiSFrozenTrainingBatchError,
                "credential field",
            ):
                validate_multi_s_frozen_training_batch(
                    secret_tamper,
                    active_joint_version=joint_version,
                    verify_source_files=False,
                )


if __name__ == "__main__":
    unittest.main()
