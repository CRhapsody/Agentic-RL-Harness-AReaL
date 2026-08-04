from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.verify_areal_official_b0 import (
    MEMORY_AUDIT_SCHEMA,
    MEMORY_RUN_KIND,
    PINNED_AREAL_COMMIT,
    verify_areal_official_b0,
)


PROJECT_COMMIT = "a" * 40
STAMP = "20260804T010203Z"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _with_record_sha(record: dict[str, object]) -> dict[str, object]:
    record = dict(record)
    record["record_sha256"] = hashlib.sha256(_canonical_json(record)).hexdigest()
    return record


class OfficialB0Fixture:
    def __init__(self, directory: str):
        self.base = Path(directory)
        self.root = self.base / STAMP
        self.log_dir = self.root / "logs" / "tester" / "jph-b0" / STAMP
        self.log_dir.mkdir(parents=True)
        self.config = self.log_dir / "config.yaml"
        self.main = self.log_dir / "main.log"
        self.actor = self.log_dir / "actor.log"
        self.run_log = self.base / f"areal-b0-{STAMP}.log"
        self.memory = self.root / "gpu-memory-audit.json"
        self._write_config()
        self._write_main()
        self._write_actor()
        self._write_run_log()
        self._write_memory()
        self._write_checkpoint()

    def _write_config(self) -> None:
        self.config.write_text(
            f"""experiment_name: jph-b0
trial_name: {STAMP}
cluster:
  fileroot: {self.root.resolve()}
  n_nodes: 1
  n_gpus_per_node: 8
scheduler:
  type: local
total_train_steps: 1
train_dataset:
  batch_size: 8
valid_dataset:
  batch_size: 8
gconfig:
  n_samples: 2
  max_new_tokens: 256
  max_tokens: 512
sglang:
  mem_fraction_static: 0.29
  context_length: 1024
  max_running_requests: 2
saver:
  mode: sync
  freq_epochs: null
  freq_steps: 1
  freq_secs: null
rollout:
  backend: sglang:d4p1t1
  max_concurrent_rollouts: 8
  max_head_offpolicyness: 0
  agent:
    admin_api_key: <redacted-runtime-admin-key>
actor:
  backend: fsdp:d4p1t1
  kl_ctl: 0.0
  optimizer:
    lr: 1.7e-5
    warmup_steps_proportion: 0.0
ref:
  backend: fsdp:d4p1t1
  scheduling_strategy:
    type: colocation
    target: actor
  admin_api_key: <redacted-default-admin-key>
""",
            encoding="utf-8",
        )

    @staticmethod
    def _stats_table(**overrides: float) -> str:
        values = {
            "ppo_actor/update/update_successful": 1.0,
            "ppo_actor/update/grad_norm": 2.5704,
            "ppo_actor/update/lr": 1.7e-5,
            "ppo_actor/update/n_valid_tokens": 2540.0,
            "ppo_actor/update/actor_loss/avg": -2.5289e-4,
            "ppo_actor/advantages/max": 1.204,
            "ppo_actor/advantages/min": -1.3413,
            "timeperf/train_step": 4.1772,
            "timeperf/update_weights": 1.7268,
        }
        values.update(overrides)
        lines = []
        items = list(values.items())
        for index in range(0, len(items), 2):
            left_name, left_value = items[index]
            if index + 1 < len(items):
                right_name, right_value = items[index + 1]
                lines.append(
                    f"│ {left_name} │ {left_value:.8e} │ {right_name} │ {right_value:.8e} │"
                )
            else:
                lines.append(f"│ {left_name} │ {left_value:.8e} │")
        return "\n".join(lines)

    def _write_main(self, **metric_overrides: float) -> None:
        lines = [
            "LocalScheduler INFO: LocalScheduler initialized with GPU devices: [0, 1, 2, 3, 4, 5, 6, 7], log directory: test",
            "LocalScheduler INFO: Creating 4 workers for role 'actor' (strategy: SchedulingStrategyType.separation, colocate_with: None)",
        ]
        for rank in range(4):
            lines.append(
                f"LocalScheduler INFO: Worker actor/{rank} started (PID: {100 + rank}, GPUs: [{rank}], ports: [1, 2])"
            )
        lines.extend(
            [
                "TrainController INFO: TrainController initialization complete",
                "LocalScheduler INFO: Creating 4 workers for role 'rollout' (strategy: SchedulingStrategyType.separation, colocate_with: None)",
            ]
        )
        for rank in range(4):
            lines.append(
                f"LocalScheduler INFO: Worker rollout/{rank} started (PID: {200 + rank}, GPUs: [{rank + 4}], ports: [1, 2])"
            )
        lines.extend(
            [
                "StatsLogger INFO: Epoch 1/10 Step 1/934 Train step 1/9340 done.",
                "StatsLogger INFO: Stats (1/1):",
                self._stats_table(**metric_overrides),
                "StatsLogger INFO: Training completes! Total time elapsed 10.00.",
                "TrainController INFO: TrainController destroyed",
            ]
        )
        self.main.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_actor(self) -> None:
        lines = []
        for rank in range(4):
            lines.extend(
                [
                    f"EngineBP INFO: Engine 'actor/{rank}' (class: areal.engine.fsdp_engine.FSDPPPOActor) instantiated successfully",
                    f"[FSDPEngine Rank {rank}] INFO: Initializing device mesh with parallel dims (dp=4, sp=1, tp=1, ep=1, etp=1, world_size=4).",
                    f"[FSDPEngine Rank {rank}] INFO: Create optimizer time: 0.001",
                    f"[FSDPEngine Rank {rank}] INFO: Microbatch #tokens (rank {rank}): [100], padded to: [128], padding lengths: [28]",
                ]
            )
        self.actor.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_run_log(self) -> None:
        self.run_log.write_text(
            f"project={PROJECT_COMMIT} AReaL={PINNED_AREAL_COMMIT} "
            f"GPUs=0,1,2,3,4,5,6,7 run_root={self.root.resolve()}\n",
            encoding="utf-8",
        )

    def _write_memory(self, **gpu_zero_overrides: object) -> None:
        gpus = []
        for gpu_id in range(8):
            entry: dict[str, object] = {
                "physical_gpu_id": gpu_id,
                "baseline_used_mib": 1000 + gpu_id,
                "peak_used_mib": 25000 + gpu_id,
                "peak_delta_mib": 24000,
                "peak_free_mib": 54000,
                "sample_count": 20,
                "memory_limit_enforced": False,
                "max_new_memory_mib": None,
                "passed": True,
            }
            if gpu_id == 0:
                entry.update(gpu_zero_overrides)
            gpus.append(entry)
        record = _with_record_sha(
            {
                "schema_version": MEMORY_AUDIT_SCHEMA,
                "run_kind": MEMORY_RUN_KIND,
                "project_commit": PROJECT_COMMIT,
                "areal_commit": PINNED_AREAL_COMMIT,
                "memory_limit_enforced": False,
                "max_new_memory_mib": None,
                "gpus": gpus,
                "passed": True,
            }
        )
        self.memory.write_bytes(_canonical_json(record))
        os.chmod(self.memory, 0o600)

    def _write_checkpoint(self) -> None:
        checkpoint = (
            self.root
            / "checkpoints"
            / "tester"
            / "jph-b0"
            / STAMP
            / "default"
            / "epoch0epochstep0globalstep0"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(
            json.dumps({"model_type": "qwen2"}), encoding="utf-8"
        )
        (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
        (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"real-model-bytes")

    def verify(self) -> dict[str, object]:
        return verify_areal_official_b0(
            run_root=self.root,
            run_log=self.run_log,
            expected_project_commit=PROJECT_COMMIT,
        )


class VerifyArealOfficialB0Tests(unittest.TestCase):
    def test_accepts_policy_only_one_step_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            report = fixture.verify()
        self.assertTrue(report["passed"])
        self.assertTrue(report["claims"]["policy_optimizer_update"])
        self.assertFalse(report["claims"]["harness_optimizer_update"])
        self.assertFalse(report["claims"]["harness_optimizer_evidence"])
        self.assertFalse(report["claims"]["ref_worker_instantiation_evidence"])
        self.assertEqual(report["topology"]["scheduler_gpu_ids"], list(range(8)))
        self.assertEqual(report["training"]["completed_train_steps"], 1)

    def test_rejects_old_dmon_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            fixture.memory.unlink()
            (fixture.root / "gpu-dmon.log").write_text("raw dmon is not an audit\n")
            with self.assertRaisesRegex(ValueError, "gpu-memory-audit.json"):
                fixture.verify()

    def test_accepts_arbitrary_memory_and_rejects_observation_integrity_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            fixture._write_memory(peak_used_mib=79000, peak_delta_mib=78000)
            fixture.verify()

        cases = (
            ({"memory_limit_enforced": True}, "observation"),
            ({"sample_count": 0}, "no samples"),
            ({"peak_delta_mib": 23999}, "inconsistent"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = OfficialB0Fixture(directory)
                    fixture._write_memory(**overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            record = json.loads(fixture.memory.read_text(encoding="utf-8"))
            record["gpus"][0]["sample_count"] = 99
            fixture.memory.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            record = json.loads(fixture.memory.read_text(encoding="utf-8"))
            record["gpus"][0]["unrecognized"] = 1
            record.pop("record_sha256")
            record = _with_record_sha(record)
            fixture.memory.write_bytes(_canonical_json(record))
            with self.assertRaisesRegex(ValueError, "field set"):
                fixture.verify()

    def test_rejects_commit_topology_and_ref_contract_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            fixture.run_log.write_text(
                fixture.run_log.read_text().replace(PROJECT_COMMIT, "b" * 40),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "commit provenance"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            fixture.main.write_text(
                fixture.main.read_text().replace("GPUs: [7]", "GPUs: [6]"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not unique|disjoint"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            fixture.config.write_text(
                fixture.config.read_text().replace("target: actor", "target: rollout"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ref.scheduling_strategy.target"):
                fixture.verify()

    def test_rejects_noncanonical_bounded_b0_config(self) -> None:
        mutations = (
            (
                "train_dataset:\n  batch_size: 8",
                "train_dataset:\n  batch_size: 16",
                "train_dataset.batch_size",
            ),
            (
                "valid_dataset:\n  batch_size: 8",
                "valid_dataset:\n  batch_size: 16",
                "valid_dataset.batch_size",
            ),
            ("n_samples: 2", "n_samples: 4", "gconfig.n_samples"),
            ("max_new_tokens: 256", "max_new_tokens: 512", "gconfig.max_new_tokens"),
            ("max_tokens: 512", "max_tokens: 1024", "gconfig.max_tokens"),
            (
                "mem_fraction_static: 0.29",
                "mem_fraction_static: 0.8",
                "sglang.mem_fraction_static",
            ),
            ("context_length: 1024", "context_length: 32768", "sglang.context_length"),
            (
                "max_running_requests: 2",
                "max_running_requests: null",
                "sglang.max_running_requests",
            ),
            ("mode: sync", "mode: auto", "saver.mode"),
            ("freq_epochs: null", "freq_epochs: 1", "saver.freq_epochs"),
            ("freq_steps: 1", "freq_steps: null", "saver.freq_steps"),
            ("freq_secs: null", "freq_secs: 3600", "saver.freq_secs"),
            ("kl_ctl: 0.0", "kl_ctl: 0.1", "actor.kl_ctl"),
            (
                "max_concurrent_rollouts: 8",
                "max_concurrent_rollouts: 16",
                "rollout.max_concurrent_rollouts",
            ),
            (
                "max_head_offpolicyness: 0",
                "max_head_offpolicyness: 2",
                "rollout.max_head_offpolicyness",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(field=message):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = OfficialB0Fixture(directory)
                    text = fixture.config.read_text(encoding="utf-8")
                    text = text.replace(old, new)
                    fixture.config.write_text(text, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message.replace(".", r"\.")):
                        fixture.verify()

    def test_wrapper_self_report_cannot_substitute_for_areal_optimizer_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            fixture.main.write_text(
                fixture.main.read_text().replace(
                    "ppo_actor/update/update_successful", "wrapper/other_metric"
                ),
                encoding="utf-8",
            )
            with fixture.run_log.open("a", encoding="utf-8") as stream:
                stream.write("policy_optimizer_update=true update_successful=1\n")
            with self.assertRaisesRegex(ValueError, "update_successful"):
                fixture.verify()

    def test_rejects_noop_or_invalid_training_stats(self) -> None:
        cases = (
            ({"ppo_actor/update/update_successful": 0.0}, "update_successful"),
            ({"ppo_actor/update/grad_norm": 0.0}, "grad_norm"),
            ({"ppo_actor/update/lr": 0.0}, "learning rate"),
            ({"ppo_actor/update/n_valid_tokens": 0.0}, "valid training tokens"),
            ({"ppo_actor/advantages/max": 0.0}, "advantages/max"),
            ({"ppo_actor/advantages/min": 0.0}, "advantages/min"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = OfficialB0Fixture(directory)
                    fixture._write_main(**overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        fixture.verify()

    def test_rejects_warmup_second_step_and_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            fixture.config.write_text(
                fixture.config.read_text().replace(
                    "warmup_steps_proportion: 0.0", "warmup_steps_proportion: 0.001"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "warmup_steps_proportion"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            with fixture.main.open("a", encoding="utf-8") as stream:
                stream.write(
                    "StatsLogger INFO: Epoch 1/10 Step 2/934 Train step 2/9340 done.\n"
                )
            with self.assertRaisesRegex(ValueError, "exactly one completed train step"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = OfficialB0Fixture(directory)
            model = next(fixture.root.rglob("model.safetensors"))
            model.unlink()
            with self.assertRaisesRegex(ValueError, "safetensors"):
                fixture.verify()

    def test_rejects_unredacted_runtime_credentials(self) -> None:
        for secret_line in (
            "admin_api_key: areal-admin-key\n",
            "session_api_key: live-session-value\n",
            "note=jph-b0-super-secret-token\n",
        ):
            with self.subTest(secret_line=secret_line):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = OfficialB0Fixture(directory)
                    leaked = fixture.root / "leaked.txt"
                    leaked.write_text(secret_line, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "credential|unredacted"):
                        fixture.verify()


if __name__ == "__main__":
    unittest.main()
