from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import unittest
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from jphrl.training.areal_policy_optimizer import (
    ArealExternalAdvantageBatchError,
    build_areal_external_advantage_batch,
    materialize_areal_ppo_update_tensors,
    validate_areal_external_advantage_batch,
    validate_m0_areal_actor_config,
)
from jphrl.trajectory.schema import JointVersion


def _version() -> JointVersion:
    return JointVersion(
        policy="policy-v7",
        harness_controller="harness-v3",
        harness_artifact="artifact-v1",
        tool_schema="tool-v1",
        parser="parser-v1",
        environment="environment-v1",
        evaluator="evaluator-v1",
        tokenizer="tokenizer-v1",
        context_builder="context-v1",
    )


def _source(style: str = "individual") -> dict[str, object]:
    return {
        "record_sha256": "a" * 64,
        "admissions": {"policy_export_style": style},
        "evidence_scope": {
            "policy_advantages_aligned": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "policy_samples": [
            {
                "sample_id": "sample-1",
                "decision_credits": [{"inference_engine_version": 7}],
                "tensor_dict": {
                    "input_ids": [[10, 11, 20, 21]],
                    "loss_mask": [[0, 0, 1, 1]],
                    "logprobs": [[0.0, 0.0, -0.2, -0.3]],
                    "versions": [[-1, -1, 7, 7]],
                    "attention_mask": [[True, True, True, True]],
                    "rewards": [1.0],
                },
                "advantage_tensor": [[0.0, 0.0, 0.75, 0.75]],
            }
        ],
    }


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


class ArealPolicyOptimizerAdapterTests(unittest.TestCase):
    def _build(self, style: str = "individual") -> dict[str, object]:
        with patch(
            "jphrl.training.areal_policy_optimizer."
            "validate_frozen_joint_credit_alignment",
            return_value={"episode_id": "episode-1"},
        ):
            return build_areal_external_advantage_batch(
                _source(style),
                active_joint_version=_version(),
            )

    def test_build_causal_shifts_external_advantage_for_pinned_areal(self) -> None:
        record = self._build()

        validated = validate_areal_external_advantage_batch(
            json.loads(json.dumps(record)),
            active_joint_version=_version(),
        )
        tensors = record["samples"][0]["tensor_dict"]

        self.assertEqual(tensors["input_ids"], [[10, 11, 20, 21]])
        self.assertEqual(tensors["loss_mask"], [[0, 1, 1, 0]])
        self.assertEqual(tensors["logprobs"], [[0.0, -0.2, -0.3, 0.0]])
        self.assertEqual(tensors["prox_logp"], tensors["logprobs"])
        self.assertEqual(tensors["advantages"], [[0.0, 0.75, 0.75, 0.0]])
        self.assertEqual(tensors["returns"], tensors["advantages"])
        self.assertEqual(tensors["versions"], [[-1, -1, 7, 7]])
        self.assertEqual(validated.trainable_token_count, 2)
        self.assertEqual(validated.inference_engine_version, 7)
        self.assertFalse(record["evidence_scope"]["areal_ppo_update_invoked"])
        self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])

    def test_concat_and_stale_version_are_fail_closed(self) -> None:
        record = self._build("concat")
        self.assertEqual(record["summary"]["sample_count"], 1)
        stale = replace(_version(), policy="policy-v8")
        with self.assertRaisesRegex(
            ArealExternalAdvantageBatchError,
            "lag-zero active JointVersion",
        ):
            validate_areal_external_advantage_batch(
                record,
                active_joint_version=stale,
            )

    def test_rehashed_advantage_mask_or_optimizer_claim_is_rejected(self) -> None:
        record = self._build()
        wrong_mask = deepcopy(record)
        wrong_mask["samples"][0]["tensor_dict"]["loss_mask"][0][-1] = 1
        wrong_mask["summary"]["trainable_token_count"] += 1
        _resign(wrong_mask)
        with self.assertRaisesRegex(
            ArealExternalAdvantageBatchError,
            "causal-shifted binary mask",
        ):
            validate_areal_external_advantage_batch(wrong_mask)

        false_claim = deepcopy(record)
        false_claim["evidence_scope"]["policy_optimizer_update"] = True
        _resign(false_claim)
        with self.assertRaisesRegex(
            ArealExternalAdvantageBatchError,
            "evidence",
        ):
            validate_areal_external_advantage_batch(false_claim)

    def test_mixed_rollout_version_and_unsafe_actor_config_are_rejected(self) -> None:
        record = self._build()
        mixed = deepcopy(record)
        mixed["samples"][0]["rollout_tensor_dict"]["versions"][0][2] = 8
        mixed["samples"][0]["tensor_dict"]["versions"][0][2] = 8
        _resign(mixed)
        with self.assertRaisesRegex(
            ArealExternalAdvantageBatchError,
            "inference engine version",
        ):
            validate_areal_external_advantage_batch(mixed)

        unsafe_actor = SimpleNamespace(
            config=SimpleNamespace(
                adv_norm=None,
                c_clip=None,
                disable_dropout=True,
                eps_clip_higher=None,
                importance_sampling_level="token",
                is_critic=False,
                kl_ctl=0.0,
                log_agent_stats=False,
                m2_threshold=None,
                mask_no_eos_with_zero=False,
                overlong_reward_penalty=False,
                ppo_n_minibatches=2,
                recompute_logprob=False,
                rejection_sampling=None,
                reward_bias=0.0,
                reward_norm=None,
                reward_scaling=1.0,
                optimizer=SimpleNamespace(lr=1e-5, lr_scheduler_type="constant"),
                use_cispo_loss=False,
                use_decoupled_loss=False,
                use_sapo_loss=False,
            )
        )
        with self.assertRaisesRegex(
            ArealExternalAdvantageBatchError,
            "ppo_n_minibatches",
        ):
            validate_m0_areal_actor_config(unsafe_actor)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_materialization_is_typed_and_fresh_for_mutating_ppo_update(self) -> None:
        import torch

        class FakeNativeActor:
            config = SimpleNamespace(
                adv_norm=None,
                c_clip=None,
                disable_dropout=True,
                eps_clip_higher=None,
                importance_sampling_level="token",
                is_critic=False,
                kl_ctl=0.0,
                log_agent_stats=False,
                m2_threshold=None,
                mask_no_eos_with_zero=False,
                overlong_reward_penalty=False,
                ppo_n_minibatches=1,
                recompute_logprob=False,
                rejection_sampling=None,
                reward_bias=0.0,
                reward_norm=None,
                reward_scaling=1.0,
                optimizer=SimpleNamespace(lr=1e-5, lr_scheduler_type="constant"),
                use_cispo_loss=False,
                use_decoupled_loss=False,
                use_sapo_loss=False,
            )

            def get_version(self) -> int:
                return 7

            def compute_advantages(self, batch):
                for tensors in batch:
                    tensors["loss_mask"] = torch.roll(
                        tensors["loss_mask"].float(), -1, dims=-1
                    )
                    tensors["logprobs"] = (
                        torch.roll(tensors["logprobs"], -1, dims=-1)
                        * tensors["loss_mask"]
                    )
                    tensors["prox_logp"] = tensors["logprobs"]
                    tensors["advantages"] = torch.zeros_like(tensors["logprobs"])
                    tensors["returns"] = torch.zeros_like(tensors["logprobs"])
                    tensors["kl_rewards"] = torch.zeros_like(tensors["logprobs"])
                    tensors["tot_rewards"] = torch.zeros_like(tensors["logprobs"])
                return batch

        record = self._build()
        actor = FakeNativeActor()
        first = materialize_areal_ppo_update_tensors(
            record,
            actor=actor,
            active_joint_version=_version(),
            device="cpu",
        )
        first[0].pop("rewards")
        second = materialize_areal_ppo_update_tensors(
            record,
            actor=actor,
            active_joint_version=_version(),
            device="cpu",
        )

        self.assertIn("rewards", second[0])
        self.assertEqual(second[0]["input_ids"].dtype, torch.long)
        self.assertEqual(second[0]["attention_mask"].dtype, torch.bool)
        self.assertEqual(second[0]["advantages"].dtype, torch.float32)
        self.assertEqual(
            second[0]["advantages"].tolist(),
            [[0.0, 0.75, 0.75, 0.0]],
        )

        actor.config.ppo_n_minibatches = 2
        with self.assertRaisesRegex(
            ArealExternalAdvantageBatchError,
            "ppo_n_minibatches",
        ):
            materialize_areal_ppo_update_tensors(
                record,
                actor=actor,
                active_joint_version=_version(),
                device="cpu",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("areal"),
        "pinned AReaL CPU dependencies are unavailable",
    )
    def test_pinned_areal_ppo_actor_consumes_injected_external_advantage(self) -> None:
        import torch
        from areal.api.cli_args import OptimizerConfig, PPOActorConfig
        from areal.trainer.ppo.actor import PPOActor

        class RecordingEngine:
            def __init__(self) -> None:
                self.version = 7
                self.seen_advantages = None
                self.loss = None

            def get_version(self) -> int:
                return self.version

            def train(self) -> None:
                return None

            def train_batch(self, input_, loss_fn, loss_weight_fn):
                self.seen_advantages = input_["advantages"].detach().clone()
                self.asserted_loss_weight = int(loss_weight_fn(input_).item())
                current_logprobs = input_["logprobs"].detach().clone()
                current_logprobs.requires_grad_(True)
                entropy = torch.zeros_like(current_logprobs)
                loss = loss_fn(current_logprobs, entropy, input_)
                loss.backward()
                self.loss = float(loss.detach().item())
                return {
                    "update_successful": 1.0,
                    "grad_norm": float(current_logprobs.grad.norm().item()),
                    "lr": 1e-5,
                }

        config = PPOActorConfig(
            experiment_name="external-advantage-contract",
            trial_name="cpu",
            backend="fsdp:d1",
            optimizer=OptimizerConfig(lr=1e-5, lr_scheduler_type="constant"),
            disable_dropout=True,
            ppo_n_minibatches=1,
            kl_ctl=0.0,
            adv_norm=None,
            reward_norm=None,
            use_decoupled_loss=False,
            recompute_logprob=False,
            log_agent_stats=False,
        )
        engine = RecordingEngine()
        native_actor = PPOActor(config, engine)
        facade = SimpleNamespace(
            config=config,
            compute_advantages=native_actor.compute_advantages,
            get_version=engine.get_version,
        )
        prepared = materialize_areal_ppo_update_tensors(
            self._build(),
            actor=facade,
            active_joint_version=_version(),
            device="cpu",
        )

        native_actor.ppo_update(prepared)

        self.assertEqual(engine.asserted_loss_weight, 2)
        self.assertEqual(
            engine.seen_advantages.tolist(),
            [[0.0, 0.75, 0.75, 0.0]],
        )
        self.assertTrue(math.isfinite(engine.loss))
        self.assertNotEqual(engine.loss, 0.0)


if __name__ == "__main__":
    unittest.main()
