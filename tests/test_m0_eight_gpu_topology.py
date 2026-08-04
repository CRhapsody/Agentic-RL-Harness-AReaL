from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace

from jphrl.experiments.m0_eight_gpu_topology import (
    M0_EIGHT_GPU_TOPOLOGY_SCHEMA,
    M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA,
    M0_REMOTE_OPTIMIZER_RECEIPT_SCHEMA,
    M0EightGPUOwnershipLedger,
    M0EightGPUTopology,
    M0EightGPUTopologyError,
    M0WorkerPlacement,
    admit_individual_training_samples,
    assert_controller_has_no_local_optimizer,
    observe_local_scheduler_placements,
    validate_per_gpu_memory_envelope,
    validate_remote_optimizer_receipt,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _seal(record: dict[str, object]) -> dict[str, object]:
    record = deepcopy(record)
    record.pop("record_sha256", None)
    record["record_sha256"] = _digest(record)
    return record


def _receipt() -> dict[str, object]:
    ranks = []
    scheduler_state_after = {
        "last_epoch": 3,
        "_step_count": 4,
        "_last_lr": [1e-6],
    }
    scheduler_state_after_sha256 = _digest(scheduler_state_after)
    stats = {
        "update_successful": 1.0,
        "update_successful_count": 1,
        "grad_norm": 2.5,
        "grad_norm_count": 1,
        "learning_rate": 1e-6,
        "learning_rate_count": 1,
    }
    for rank in range(4):
        ranks.append(
            _seal(
                {
                    "schema_version": M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA,
                    "transaction_id": "m0-transaction",
                    "joint_version_id": "joint-version",
                    "source_admission_sha256": "a" * 64,
                    "worker_rank": rank,
                    "world_size": 4,
                    "engine_name": f"actor/{rank}",
                    "physical_gpu_id": rank,
                    "engine_class": (
                        "jphrl.training.areal_distributed_policy.JPHFSDPPPOActor"
                    ),
                    "optimizer_class": "torch.optim.adamw.AdamW",
                    "inference_engine_version": 7,
                    "global_sample_count": 4,
                    "global_sample_sha256": "b" * 64,
                    "local_sample_indices": [rank],
                    "optimizer_step_before": 2,
                    "optimizer_step_after": 3,
                    "lr_scheduler_state_before_sha256": "c" * 64,
                    "lr_scheduler_state_after_sha256": scheduler_state_after_sha256,
                    "lr_scheduler_state_after": (
                        scheduler_state_after if rank == 0 else None
                    ),
                    "parent_dcp_manifest_sha256": "e" * 64,
                    "candidate_dcp_manifest_sha256": "f" * 64,
                    "runtime_state": {
                        "hostname": f"actor-host-{rank}",
                        "local_rank": 0,
                        "device": "cuda:0",
                        "cuda_visible_devices": str(rank),
                        "python_rng_state_sha256": f"{rank + 1:x}" * 64,
                        "torch_cpu_rng_state_sha256": f"{rank + 5:x}" * 64,
                        "torch_cuda_rng_state_sha256": f"{rank + 9:x}" * 64,
                    },
                    "update_stats": stats,
                    "evidence_scope": {
                        "remote_worker_optimizer_observed": True,
                        "controller_read_worker_optimizer": False,
                        "policy_optimizer_update": True,
                        "harness_optimizer_update": False,
                    },
                }
            )
        )
    return _seal(
        {
            "schema_version": M0_REMOTE_OPTIMIZER_RECEIPT_SCHEMA,
            "transaction_id": "m0-transaction",
            "joint_version_id": "joint-version",
            "source_admission_sha256": "a" * 64,
            "actor_backend": "fsdp:d4",
            "controller_class": (
                "jphrl.training.areal_distributed_policy.JPHPPOActorController"
            ),
            "global_sample_count": 4,
            "global_sample_sha256": "b" * 64,
            "rank0_lr_scheduler_state_after": scheduler_state_after,
            "rank_receipts": ranks,
            "evidence_scope": {
                "all_actor_ranks_attested": True,
                "remote_worker_optimizer_observed": True,
                "controller_read_worker_optimizer": False,
                "policy_optimizer_update": True,
                "harness_optimizer_update": False,
            },
        }
    )


class M0EightGPUTopologyTests(unittest.TestCase):
    def test_frozen_topology_is_one_scheduler_disjoint_d4_and_observation_only(self) -> None:
        topology = M0EightGPUTopology()
        topology.validate()
        record = topology.record()
        self.assertEqual(record["schema_version"], M0_EIGHT_GPU_TOPOLOGY_SCHEMA)
        self.assertEqual(record["scheduler_gpu_ids"], list(range(8)))
        self.assertEqual(record["actor"]["backend"], "fsdp:d4")
        self.assertEqual(record["actor"]["gpu_ids"], [0, 1, 2, 3])
        self.assertEqual(record["rollout"]["backend"], "sglang:d4")
        self.assertEqual(record["rollout"]["gpu_ids"], [4, 5, 6, 7])
        self.assertEqual(record["training_batch"]["export_style"], "individual")
        self.assertEqual(record["training_batch"]["minimum_sample_count"], 4)
        self.assertEqual(record["memory"]["policy"], "observation-only")
        self.assertEqual(record["memory"]["prelaunch_snapshot_count"], 2)
        self.assertEqual(record["memory"]["runtime_sample_interval_seconds"], 1)
        self.assertIsNone(record["memory"]["fixed_limit_mib"])
        self.assertFalse(record["ownership"]["controller_reads_remote_optimizer"])
        self.assertFalse(record["evidence_scope"]["policy_optimizer_update"])

    def test_topology_rejects_concat_overlap_and_relaxed_memory(self) -> None:
        cases = (
            M0EightGPUTopology(export_style="concat"),
            M0EightGPUTopology(actor_gpu_ids=(0, 1, 2, 4)),
            M0EightGPUTopology(memory_policy="fixed-limit"),
            M0EightGPUTopology(actor_backend="fsdp:d1"),
        )
        for topology in cases:
            with self.subTest(topology=topology), self.assertRaises(
                M0EightGPUTopologyError
            ):
                topology.validate()

    def test_actor_must_remain_alive_through_y_and_cleanup_is_reverse_order(self) -> None:
        actor = [
            M0WorkerPlacement("actor", rank, rank, f"actor/{rank}")
            for rank in range(4)
        ]
        rollout = [
            M0WorkerPlacement("rollout", rank, rank + 4, f"rollout-inf/{rank}")
            for rank in range(4)
        ]
        ledger = M0EightGPUOwnershipLedger()
        ledger.start_actor(actor)
        ledger.start_rollout(rollout)
        with self.assertRaisesRegex(M0EightGPUTopologyError, "actor cannot stop"):
            ledger.stop_actor()
        ledger.complete_y()
        with self.assertRaisesRegex(M0EightGPUTopologyError, "before rollout cleanup"):
            ledger.stop_actor()
        ledger.stop_rollout()
        ledger.stop_actor()
        self.assertFalse(ledger.actor_alive)
        self.assertFalse(ledger.rollout_alive)

    def test_scheduler_placement_observation_never_reads_optimizer(self) -> None:
        actor_workers = [
            SimpleNamespace(
                gpu_devices=[rank], worker=SimpleNamespace(id=f"actor/{rank}")
            )
            for rank in range(4)
        ]
        rollout_workers = [
            SimpleNamespace(
                gpu_devices=[rank + 4],
                worker=SimpleNamespace(id=f"rollout-inf/{rank}"),
            )
            for rank in range(4)
        ]
        scheduler = SimpleNamespace(
            _workers={"actor": actor_workers, "rollout-inf": rollout_workers}
        )
        self.assertEqual(
            tuple(
                item.physical_gpu_id
                for item in observe_local_scheduler_placements(
                    scheduler, role="actor"
                )
            ),
            (0, 1, 2, 3),
        )
        self.assertEqual(
            tuple(
                item.physical_gpu_id
                for item in observe_local_scheduler_placements(
                    scheduler, role="rollout"
                )
            ),
            (4, 5, 6, 7),
        )
        assert_controller_has_no_local_optimizer(SimpleNamespace(config=object()))
        with self.assertRaisesRegex(M0EightGPUTopologyError, "must not hold"):
            assert_controller_has_no_local_optimizer(
                SimpleNamespace(optimizer=object())
            )

    def test_individual_batch_requires_four_unique_samples(self) -> None:
        samples = [{"sample_id": f"sample-{index}"} for index in range(4)]
        self.assertEqual(
            len(admit_individual_training_samples(samples, export_style="individual")),
            4,
        )
        with self.assertRaisesRegex(M0EightGPUTopologyError, "at least four"):
            admit_individual_training_samples(samples[:3], export_style="individual")
        with self.assertRaisesRegex(M0EightGPUTopologyError, "rejects concat"):
            admit_individual_training_samples(samples, export_style="concat")
        duplicated = deepcopy(samples)
        duplicated[-1]["sample_id"] = duplicated[0]["sample_id"]
        with self.assertRaisesRegex(M0EightGPUTopologyError, "duplicated"):
            admit_individual_training_samples(duplicated, export_style="individual")

    def test_all_eight_gpu_memory_deltas_are_observed_without_fixed_cap(self) -> None:
        baselines = {gpu: 2000 for gpu in range(8)}
        peaks = {gpu: 2000 + 25000 for gpu in range(8)}
        self.assertEqual(
            validate_per_gpu_memory_envelope(
                baseline_used_mib=baselines, peak_used_mib=peaks
            ),
            {gpu: 25000 for gpu in range(8)},
        )
        peaks[7] = 2000 + 70000
        self.assertEqual(
            validate_per_gpu_memory_envelope(
                baseline_used_mib=baselines, peak_used_mib=peaks
            )[7],
            70000,
        )


class M0RemoteOptimizerReceiptTests(unittest.TestCase):
    def test_accepts_four_consistent_worker_created_receipts(self) -> None:
        validated = validate_remote_optimizer_receipt(_receipt())
        self.assertEqual(validated.global_sample_count, 4)
        self.assertEqual(validated.optimizer_step_before, 2)
        self.assertEqual(validated.optimizer_step_after, 3)
        self.assertEqual(validated.rank0_lr_scheduler_state_after["last_epoch"], 3)
        self.assertEqual(
            tuple(state["local_rank"] for state in validated.runtime_states),
            (0, 0, 0, 0),
        )
        self.assertEqual(
            tuple(state["cuda_visible_devices"] for state in validated.runtime_states),
            ("0", "1", "2", "3"),
        )

    def test_controller_claim_missing_rank_and_crossed_step_fail_closed(self) -> None:
        missing = _receipt()
        missing["rank_receipts"] = missing["rank_receipts"][:3]
        missing = _seal(missing)
        with self.assertRaisesRegex(M0EightGPUTopologyError, "all four"):
            validate_remote_optimizer_receipt(missing)

        local_claim = _receipt()
        local_claim["evidence_scope"]["controller_read_worker_optimizer"] = True
        local_claim = _seal(local_claim)
        with self.assertRaisesRegex(M0EightGPUTopologyError, "evidence scope"):
            validate_remote_optimizer_receipt(local_claim)

        crossed = _receipt()
        crossed["rank_receipts"][2]["optimizer_step_after"] = 4
        crossed["rank_receipts"][2] = _seal(crossed["rank_receipts"][2])
        crossed = _seal(crossed)
        with self.assertRaisesRegex(M0EightGPUTopologyError, "exactly one"):
            validate_remote_optimizer_receipt(crossed)

    def test_rank_partitions_must_be_disjoint_and_exhaustive(self) -> None:
        receipt = _receipt()
        receipt["rank_receipts"][3]["local_sample_indices"] = [2]
        receipt["rank_receipts"][3] = _seal(receipt["rank_receipts"][3])
        receipt = _seal(receipt)
        with self.assertRaisesRegex(M0EightGPUTopologyError, "disjoint and exhaustive"):
            validate_remote_optimizer_receipt(receipt)

    def test_scheduler_state_and_runtime_identity_are_worker_bound(self) -> None:
        scheduler_tamper = _receipt()
        scheduler_tamper["rank0_lr_scheduler_state_after"]["last_epoch"] = 99
        scheduler_tamper = _seal(scheduler_tamper)
        with self.assertRaisesRegex(M0EightGPUTopologyError, "rank 0 scheduler"):
            validate_remote_optimizer_receipt(scheduler_tamper)

        runtime_tamper = _receipt()
        runtime_tamper["rank_receipts"][2]["runtime_state"]["local_rank"] = 2
        runtime_tamper["rank_receipts"][2] = _seal(
            runtime_tamper["rank_receipts"][2]
        )
        runtime_tamper = _seal(runtime_tamper)
        with self.assertRaisesRegex(M0EightGPUTopologyError, "runtime state"):
            validate_remote_optimizer_receipt(runtime_tamper)


if __name__ == "__main__":
    unittest.main()
