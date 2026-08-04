from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import unittest
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import jphrl.training.areal_distributed_policy as distributed
from jphrl.experiments.m0_eight_gpu_topology import (
    M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA,
    validate_remote_optimizer_receipt,
)
from jphrl.training.areal_distributed_policy import (
    AREAL_DISTRIBUTED_POLICY_CANDIDATE_SCHEMA,
    ArealDistributedPolicyError,
    JPH_AREAL_DISTRIBUTED_ACTOR_CLASS,
    JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS,
    JPHPPOActorController,
    LiveArealDistributedPolicyCandidate,
    M0_DISTRIBUTED_POLICY_CONTINUATION_RANK_SCHEMA,
    M0_DISTRIBUTED_POLICY_CURRENT_STATE_RANK_SCHEMA,
    M0_DISTRIBUTED_POLICY_CURRENT_STATE_SCHEMA,
    M0_DISTRIBUTED_POLICY_RESTORE_RANK_SCHEMA,
    M0_DISTRIBUTED_SERVING_EXPORT_RANK_SCHEMA,
    M0_DISTRIBUTED_SERVING_EXPORT_SCHEMA,
    build_distributed_serving_export_receipt,
    build_distributed_policy_current_state_receipt,
    build_remote_optimizer_aggregate,
    require_live_remote_policy_candidate,
    validate_distributed_policy_candidate,
    validate_distributed_policy_current_state_receipt,
)
from jphrl.training.areal_policy_candidate import (
    checkpoint_manifest,
    validate_areal_policy_candidate,
)
from jphrl.training.production_checkpoint import capture_rank_runtime_state
from jphrl.trajectory.schema import JointVersion


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


class ExactRecoveryCudaRuntimeTests(unittest.TestCase):
    def test_runtime_requires_launcher_environment_before_torch_setup(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": ":16:8", "PYTHONHASHSEED": "0"},
            ),
            self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            ),
        ):
            distributed._configure_exact_recovery_cuda_runtime()

    def test_runtime_enables_strict_deterministic_backends(self) -> None:
        torch = ModuleType("torch")
        torch.use_deterministic_algorithms = Mock()  # type: ignore[attr-defined]
        torch.are_deterministic_algorithms_enabled = Mock(  # type: ignore[attr-defined]
            return_value=True
        )
        cudnn = SimpleNamespace(
            benchmark=True,
            deterministic=False,
            allow_tf32=True,
        )
        cuda = SimpleNamespace(
            matmul=SimpleNamespace(allow_tf32=True),
            enable_flash_sdp=Mock(),
            enable_mem_efficient_sdp=Mock(),
            enable_math_sdp=Mock(),
        )
        torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
            cudnn=cudnn,
            cuda=cuda,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "PYTHONHASHSEED": "0",
                },
            ),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            distributed._configure_exact_recovery_cuda_runtime()

        torch.use_deterministic_algorithms.assert_called_once_with(  # type: ignore[attr-defined]
            True,
            warn_only=False,
        )
        self.assertFalse(cudnn.benchmark)
        self.assertTrue(cudnn.deterministic)
        self.assertFalse(cudnn.allow_tf32)
        self.assertFalse(cuda.matmul.allow_tf32)
        cuda.enable_flash_sdp.assert_called_once_with(False)
        cuda.enable_mem_efficient_sdp.assert_called_once_with(False)
        cuda.enable_math_sdp.assert_called_once_with(True)


def _seal(record: dict[str, object]) -> dict[str, object]:
    value = deepcopy(record)
    value.pop("record_sha256", None)
    value["record_sha256"] = _digest(value)
    return value


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


def _fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    tuple[dict[str, object], ...],
]:
    version = _version()
    source_binding = _seal(
        {
            "schema_version": "jph.multi-s-source-binding.v1",
            "joint_version_id": version.version_id,
            "batch_record_sha256": "1" * 64,
            "batch_aggregate_sha256": "2" * 64,
            "member_claim_sha256s": [str(rank + 3) * 64 for rank in range(4)],
            "s_record_sha256s": [str(rank + 7) * 64 for rank in range(3)]
            + ["b" * 64],
            "policy_sample_count": 4,
            "harness_action_count": 4,
        }
    )
    scheduler_state = {"last_epoch": 3, "_step_count": 4, "base_lrs": [1e-6]}
    scheduler_sha256 = _digest(scheduler_state)
    parent_manifest = {
        "files": [{"path": "state", "size_bytes": 1, "sha256": "e" * 64}]
    }
    parent_manifest["manifest_sha256"] = _digest(parent_manifest)
    candidate_manifest = {
        "files": [{"path": "state", "size_bytes": 2, "sha256": "f" * 64}]
    }
    candidate_manifest["manifest_sha256"] = _digest(candidate_manifest)
    rank_receipts: list[dict[str, object]] = []
    live_states: list[dict[str, object]] = []
    for rank in range(4):
        runtime = capture_rank_runtime_state(
            rank=rank,
            local_rank=0,
            device="cuda:0",
        ).to_record()
        torch_rng = runtime["torch_rng"]
        compact_runtime = {
            "hostname": runtime["hostname"],
            "local_rank": 0,
            "device": "cuda:0",
            "cuda_visible_devices": str(rank),
            "python_rng_state_sha256": _digest(runtime["python_random_state"]),
            "torch_cpu_rng_state_sha256": _digest(torch_rng["cpu_state"]),
            "torch_cuda_rng_state_sha256": _digest(torch_rng["cuda_states"]),
        }
        rank_receipt = _seal(
            {
                "schema_version": M0_REMOTE_OPTIMIZER_RANK_RECEIPT_SCHEMA,
                "transaction_id": "m0-transaction",
                "joint_version_id": version.version_id,
                "source_admission_sha256": source_binding["record_sha256"],
                "worker_rank": rank,
                "world_size": 4,
                "engine_name": f"actor/{rank}",
                "physical_gpu_id": rank,
                "engine_class": JPH_AREAL_DISTRIBUTED_ACTOR_CLASS,
                "optimizer_class": "torch.optim.adamw.AdamW",
                "inference_engine_version": 7,
                "global_sample_count": 4,
                "global_sample_sha256": "b" * 64,
                "local_sample_indices": [rank],
                "optimizer_step_before": 2,
                "optimizer_step_after": 3,
                "lr_scheduler_state_before_sha256": "c" * 64,
                "lr_scheduler_state_after_sha256": scheduler_sha256,
                "lr_scheduler_state_after": (
                    scheduler_state if rank == 0 else None
                ),
                "parent_dcp_manifest_sha256": parent_manifest[
                    "manifest_sha256"
                ],
                "candidate_dcp_manifest_sha256": candidate_manifest[
                    "manifest_sha256"
                ],
                "runtime_state": compact_runtime,
                "update_stats": {
                    "update_successful": 1.0,
                    "update_successful_count": 1,
                    "grad_norm": 2.5,
                    "grad_norm_count": 1,
                    "learning_rate": 1e-6,
                    "learning_rate_count": 1,
                },
                "evidence_scope": {
                    "remote_worker_optimizer_observed": True,
                    "controller_read_worker_optimizer": False,
                    "policy_optimizer_update": True,
                    "harness_optimizer_update": False,
                },
            }
        )
        rank_receipts.append(rank_receipt)
        live_states.append(
            _seal(
                {
                    "schema_version": "jph.m0-live-policy-worker-state.v1",
                    "transaction_id": "m0-transaction",
                    "worker_rank": rank,
                    "aggregate_sha256": "",
                    "rank_receipt_sha256": rank_receipt["record_sha256"],
                    "rank_runtime_state": runtime,
                    "rank0_lr_scheduler_class": (
                        "torch.optim.lr_scheduler.LambdaLR"
                        if rank == 0
                        else None
                    ),
                    "rank0_lr_scheduler_state_after": (
                        scheduler_state if rank == 0 else None
                    ),
                }
            )
        )
    aggregate = build_remote_optimizer_aggregate(rank_receipts)
    members = [
        {
            "member_index": rank,
            "member_claim_sha256": source_binding["member_claim_sha256s"][rank],
            "episode_id": f"episode-{rank}",
            "admission_record_sha256": str(rank + 1) * 64,
            "source_joint_credit_sha256": source_binding["s_record_sha256s"][rank],
            "sample_ids": [f"sample-{rank}"],
            "trainable_token_count": 2,
        }
        for rank in range(4)
    ]
    source_batch_identity_sha256 = _digest(
        {
            "schema_version": "jph.m0-ordered-policy-source-batch.v1",
            "members": members,
        }
    )
    candidate = _seal(
        {
            "schema_version": AREAL_DISTRIBUTED_POLICY_CANDIDATE_SCHEMA,
            "source_binding": source_binding,
            "transaction": {
                "transaction_id": "m0-transaction",
                "source_admission_sha256": source_binding["record_sha256"],
                "source_batch_identity_sha256": source_batch_identity_sha256,
                "members": members,
                "global_sample_count": 4,
                "global_sample_sha256": "b" * 64,
                "trainable_token_count": 8,
            },
            "parent": {
                "joint_version": {
                    field: getattr(version, field)
                    for field in version.__dataclass_fields__
                },
                "joint_version_id": version.version_id,
                "policy_engine_version": 7,
                "policy_version": version.policy,
            },
            "candidate": {
                "policy_version": "areal-distributed-policy-"
                + candidate_manifest["manifest_sha256"][:20],
                "reserved_policy_engine_version": 8,
            },
            "optimizer": {"remote_optimizer_receipt": aggregate},
            "checkpoints": {
                "parent_path": "/external/policy-parent.dcp",
                "parent_manifest": parent_manifest,
                "candidate_path": "/external/policy-candidate.dcp",
                "candidate_manifest": candidate_manifest,
            },
            "provenance": {
                "areal_commit": "fee938eada49208a5aabdbc1095730a13076a349",
                "project_commit": "1" * 40,
                "actor_class": JPH_AREAL_DISTRIBUTED_ACTOR_CLASS,
                "controller_class": JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS,
            },
            "evidence_scope": {
                "all_actor_ranks_attested": True,
                "policy_optimizer_update": True,
                "policy_candidate_created": True,
                "policy_weights_published": False,
                "active_joint_version_changed": False,
                "harness_optimizer_update": False,
                "joint_publish": False,
            },
        }
    )
    for live_state in live_states:
        live_state["aggregate_sha256"] = aggregate["record_sha256"]
        live_state["policy_candidate_sha256"] = candidate["record_sha256"]
        live_state.update(_seal(live_state))
    return aggregate, candidate, tuple(live_states)


def _live_candidate() -> LiveArealDistributedPolicyCandidate:
    _aggregate, receipt, worker_states = _fixture()
    return LiveArealDistributedPolicyCandidate._create(
        receipt=receipt,
        worker_states=worker_states,
        token=distributed._LIVE_CANDIDATE_TOKEN,
    )


def _serving_export_rank_receipts(
    live: LiveArealDistributedPolicyCandidate,
) -> list[dict[str, object]]:
    checkpoints = live.receipt["checkpoints"]
    branches: dict[str, dict[str, object]] = {}
    for kind in ("parent", "candidate"):
        manifest = {
            "files": [
                {
                    "path": "model.safetensors",
                    "size_bytes": 128 if kind == "parent" else 129,
                    "sha256": "4" * 64 if kind == "parent" else "5" * 64,
                }
            ]
        }
        manifest["manifest_sha256"] = _digest(manifest)
        branches[kind] = {
            "dcp_path": checkpoints[f"{kind}_path"],
            "dcp_manifest_sha256": checkpoints[f"{kind}_manifest"][
                "manifest_sha256"
            ],
            "export_path": f"/external/serving/policy-{kind}-hf",
            "export_manifest": manifest,
            "parameter_sha256": "6" * 64 if kind == "parent" else "7" * 64,
        }
    return [
        _seal(
            {
                "schema_version": M0_DISTRIBUTED_SERVING_EXPORT_RANK_SCHEMA,
                "transaction_id": "m0-transaction",
                "policy_candidate_sha256": live.receipt["record_sha256"],
                "worker_rank": rank,
                "world_size": 4,
                "parent": deepcopy(branches["parent"]),
                "candidate": deepcopy(branches["candidate"]),
                "evidence_scope": {
                    "collective_hf_save_executed": True,
                    "candidate_dcp_restored": True,
                    "policy_optimizer_update": False,
                    "harness_optimizer_update": False,
                },
            }
        )
        for rank in range(4)
    ]


def _current_state_rank_receipts(
    live: LiveArealDistributedPolicyCandidate,
    export: Mapping[str, object],
    *,
    lineage_sha256: str = "8" * 64,
) -> list[dict[str, object]]:
    remote = live.receipt["optimizer"]["remote_optimizer_receipt"]
    scheduler_sha256 = remote["rank_receipts"][0][
        "lr_scheduler_state_after_sha256"
    ]
    return [
        _seal(
            {
                "schema_version": (
                    M0_DISTRIBUTED_POLICY_CURRENT_STATE_RANK_SCHEMA
                ),
                "transaction_id": "m0-transaction",
                "policy_candidate_sha256": live.receipt["record_sha256"],
                "distributed_serving_export_receipt_sha256": export[
                    "record_sha256"
                ],
                "collective_export_rank_receipt_sha256": export[
                    "rank_receipts"
                ][rank]["record_sha256"],
                "candidate_serving_export_lineage_sha256": lineage_sha256,
                "candidate_serving_parameter_sha256": export["candidate"][
                    "parameter_sha256"
                ],
                "worker_rank": rank,
                "world_size": 4,
                "engine_name": f"actor/{rank}",
                "physical_gpu_id": rank,
                "candidate_dcp_manifest_sha256": live.receipt["checkpoints"][
                    "candidate_manifest"
                ]["manifest_sha256"],
                "optimizer_step": remote["rank_receipts"][rank][
                    "optimizer_step_after"
                ],
                "lr_scheduler_state_sha256": scheduler_sha256,
                "runtime_rng_state_sha256": _digest(
                    live.worker_states[rank]["rank_runtime_state"]
                ),
                "actor_public_version": 7,
                "evidence_scope": {
                    "candidate_dcp_restored_after_collective_export": True,
                    "optimizer_state_attested": True,
                    "lr_scheduler_state_attested": True,
                    "rank_rng_state_restored_and_attested": True,
                    "candidate_serving_parameter_digest_bound": True,
                    "live_weight_parameter_digest_observed": False,
                    "policy_optimizer_update": False,
                    "harness_optimizer_update": False,
                },
            }
        )
        for rank in range(4)
    ]


def _live_candidate_with_real_checkpoints(
    root: Path,
) -> LiveArealDistributedPolicyCandidate:
    aggregate, candidate, worker_states = _fixture()
    manifests: dict[str, dict[str, object]] = {}
    for kind, payload in (("parent", b"parent"), ("candidate", b"candidate")):
        checkpoint = root / f"policy-{kind}.dcp"
        checkpoint.mkdir(parents=True)
        (checkpoint / "state").write_bytes(payload)
        manifests[kind] = checkpoint_manifest(checkpoint)
        candidate["checkpoints"][f"{kind}_path"] = str(checkpoint)
        candidate["checkpoints"][f"{kind}_manifest"] = manifests[kind]
    rank_receipts = deepcopy(aggregate["rank_receipts"])
    for receipt in rank_receipts:
        receipt["parent_dcp_manifest_sha256"] = manifests["parent"][
            "manifest_sha256"
        ]
        receipt["candidate_dcp_manifest_sha256"] = manifests["candidate"][
            "manifest_sha256"
        ]
        receipt.update(_seal(receipt))
    aggregate = build_remote_optimizer_aggregate(rank_receipts)
    candidate["optimizer"]["remote_optimizer_receipt"] = aggregate
    candidate["candidate"]["policy_version"] = (
        "areal-distributed-policy-"
        + str(manifests["candidate"]["manifest_sha256"])[:20]
    )
    candidate = _seal(candidate)
    updated_states: list[dict[str, object]] = []
    for rank, state in enumerate(worker_states):
        value = deepcopy(state)
        value["aggregate_sha256"] = aggregate["record_sha256"]
        value["policy_candidate_sha256"] = candidate["record_sha256"]
        value["rank_receipt_sha256"] = rank_receipts[rank]["record_sha256"]
        updated_states.append(_seal(value))
    return LiveArealDistributedPolicyCandidate._create(
        receipt=candidate,
        worker_states=updated_states,
        token=distributed._LIVE_CANDIDATE_TOKEN,
    )


def _controller() -> JPHPPOActorController:
    controller = object.__new__(JPHPPOActorController)
    controller.config = SimpleNamespace(backend="fsdp:d4")
    controller._worker_role = "actor"
    controller.workers = [
        SimpleNamespace(id=f"actor/{rank}") for rank in range(4)
    ]
    controller.workers_is_dp_head = [True, True, True, True]
    return controller


def _restore_rank_receipts(
    live: LiveArealDistributedPolicyCandidate,
    branch_id: str,
) -> list[dict[str, object]]:
    candidate = live.receipt
    load_observed = branch_id != "uninterrupted"
    remote = candidate["optimizer"]["remote_optimizer_receipt"]
    records = []
    for rank, (worker_state, rank_receipt) in enumerate(
        zip(live.worker_states, remote["rank_receipts"])
    ):
        records.append(
            _seal(
                {
                    "schema_version": M0_DISTRIBUTED_POLICY_RESTORE_RANK_SCHEMA,
                    "branch_id": branch_id,
                    "transaction_id": "m0-transaction",
                    "policy_candidate_sha256": candidate["record_sha256"],
                    "worker_rank": rank,
                    "world_size": 4,
                    "engine_name": f"actor/{rank}",
                    "physical_gpu_id": rank,
                    "candidate_dcp_manifest_sha256": candidate["checkpoints"][
                        "candidate_manifest"
                    ]["manifest_sha256"],
                    "optimizer_step": rank_receipt["optimizer_step_after"],
                    "lr_scheduler_state_sha256": rank_receipt[
                        "lr_scheduler_state_after_sha256"
                    ],
                    "runtime_rng_state_sha256": _digest(
                        worker_state["rank_runtime_state"]
                    ),
                    "actor_public_version": 7,
                    "evidence_scope": {
                        "live_candidate_state_attested": not load_observed,
                        "candidate_dcp_loaded": load_observed,
                        "optimizer_state_loaded": load_observed,
                        "lr_scheduler_state_loaded": load_observed,
                        "rank_rng_state_loaded": load_observed,
                        "rank_scheduler_rng_state_attested": True,
                        "continuation_executed": False,
                        "exact_joint_recovery": False,
                    },
                }
            )
        )
    return records


def _continuation_rank_receipts(
    live: LiveArealDistributedPolicyCandidate,
    restore: Mapping[str, object],
    branch_id: str,
) -> list[dict[str, object]]:
    candidate = live.receipt
    remote = candidate["optimizer"]["remote_optimizer_receipt"]
    candidate_manifest = candidate["checkpoints"]["candidate_manifest"][
        "manifest_sha256"
    ]
    records = []
    continuation_manifest_sha256 = (
        "d" * 64 if branch_id == "uninterrupted" else "e" * 64
    )
    for rank, rank_receipt in enumerate(remote["rank_receipts"]):
        records.append(
            _seal(
                {
                    "schema_version": (
                        M0_DISTRIBUTED_POLICY_CONTINUATION_RANK_SCHEMA
                    ),
                    "branch_id": branch_id,
                    "transaction_id": "m0-transaction",
                    "policy_candidate_sha256": candidate["record_sha256"],
                    "restore_receipt_sha256": restore["record_sha256"],
                    "worker_rank": rank,
                    "world_size": 4,
                    "engine_name": f"actor/{rank}",
                    "physical_gpu_id": rank,
                    "source_binding_sha256": candidate["source_binding"][
                        "record_sha256"
                    ],
                    "candidate_dcp_manifest_sha256": candidate_manifest,
                    "continuation_dcp_manifest_sha256": (
                        continuation_manifest_sha256
                    ),
                    "continuation_dcp_payload_sha256": "6" * 64,
                    "local_sample_indices": [rank],
                    "optimizer_step_before": 3,
                    "optimizer_step_after": 4,
                    "lr_scheduler_state_before_sha256": rank_receipt[
                        "lr_scheduler_state_after_sha256"
                    ],
                    "lr_scheduler_state_after_sha256": "9" * 64,
                    "runtime_rng_state_after_sha256": str(rank + 1) * 64,
                    "actor_public_version": 7,
                    "evidence_scope": {
                        "bound_pre_batch_sources_reused": True,
                        "diagnostic_policy_optimizer_step_observed": True,
                        "continuation_dcp_with_optimizer_observed": True,
                        "exact_joint_recovery": False,
                    },
                }
            )
        )
    return records


class DistributedPolicyReceiptTests(unittest.TestCase):
    def test_dcp_payload_digest_ignores_only_metadata(self) -> None:
        def manifest(metadata_sha256: str, shard_sha256: str):
            return {
                "files": [
                    {
                        "path": ".metadata",
                        "size_bytes": 10,
                        "sha256": metadata_sha256,
                    },
                    {
                        "path": "__0_0.distcp",
                        "size_bytes": 20,
                        "sha256": shard_sha256,
                    },
                ],
                "manifest_sha256": "f" * 64,
            }

        first = distributed._checkpoint_dcp_payload_sha256(
            manifest("a" * 64, "b" * 64)
        )
        self.assertEqual(
            first,
            distributed._checkpoint_dcp_payload_sha256(
                manifest("c" * 64, "b" * 64)
            ),
        )
        self.assertNotEqual(
            first,
            distributed._checkpoint_dcp_payload_sha256(
                manifest("a" * 64, "d" * 64)
            ),
        )

    def test_collective_checkpoint_path_preflight_barriers_before_save(
        self,
    ) -> None:
        actor = object.__new__(distributed.JPHFSDPPPOActor)
        actor._require_group_common = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "continuation.dcp"
            self.assertEqual(
                actor._collectively_require_new_checkpoint_path(
                    name="test continuation",
                    path=checkpoint,
                ),
                checkpoint,
            )
            actor._require_group_common.assert_called_once_with(
                "test continuation path preflight",
                {
                    "path": str(checkpoint),
                    "exists": False,
                    "is_symlink": False,
                },
            )

            checkpoint.mkdir()
            actor._require_group_common.reset_mock()
            with self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "collectively new",
            ):
                actor._collectively_require_new_checkpoint_path(
                    name="test continuation",
                    path=checkpoint,
                )
            actor._require_group_common.assert_called_once()

    def test_pending_candidate_state_is_complete_and_rank_receipt_bound(
        self,
    ) -> None:
        scheduler = {"last_epoch": 1, "_step_count": 2}
        manifest_sha256 = "a" * 64
        rank_receipt = {
            "candidate_dcp_manifest_sha256": manifest_sha256,
            "parent_dcp_manifest_sha256": "b" * 64,
            "optimizer_step_after": 2,
            "lr_scheduler_state_after_sha256": _digest(scheduler),
            "inference_engine_version": 7,
            "record_sha256": "c" * 64,
        }
        pending = {
            "parent_path": "/run/policy-parent.dcp",
            "candidate_path": "/run/policy-candidate.dcp",
            "candidate_manifest_sha256": manifest_sha256,
            "optimizer_step_after": 2,
            "scheduler_state_after": scheduler,
            "rank_runtime_state": {"rank": 0},
            "rank_receipt": rank_receipt,
        }
        self.assertIs(
            distributed._validated_pending_m0_candidate_state(pending),
            rank_receipt,
        )

        for field in (
            "parent_path",
            "candidate_path",
            "candidate_manifest_sha256",
            "optimizer_step_after",
            "scheduler_state_after",
            "rank_runtime_state",
            "rank_receipt",
        ):
            changed = deepcopy(pending)
            changed.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "incomplete",
            ):
                distributed._validated_pending_m0_candidate_state(changed)

        for field, value in (
            ("candidate_manifest_sha256", "d" * 64),
            ("optimizer_step_after", 3),
            ("scheduler_state_after", {"last_epoch": 2, "_step_count": 3}),
        ):
            changed = deepcopy(pending)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "differs from its rank receipt",
            ):
                distributed._validated_pending_m0_candidate_state(changed)

    def test_w_candidate_checkpoint_requires_exact_pending_path_and_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            candidate = root / "policy-candidate.dcp"
            candidate.mkdir()
            (candidate / "state").write_bytes(b"candidate")
            manifest_sha256 = checkpoint_manifest(candidate)["manifest_sha256"]
            pending = {
                "candidate_path": str(candidate),
                "candidate_manifest_sha256": manifest_sha256,
            }
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                self.assertEqual(
                    distributed._validated_pending_w_candidate_path(
                        pending=pending,
                        candidate_path=str(candidate),
                        candidate_dcp_manifest_sha256=manifest_sha256,
                    ),
                    candidate,
                )

                for changed_pending, supplied_manifest, message in (
                    (
                        {"candidate_manifest_sha256": manifest_sha256},
                        manifest_sha256,
                        "path differs from T",
                    ),
                    (
                        pending,
                        "f" * 64,
                        "contents differ from T",
                    ),
                    (
                        {
                            "candidate_path": str(candidate),
                            "candidate_manifest_sha256": "e" * 64,
                        },
                        manifest_sha256,
                        "manifest binding differs from T",
                    ),
                ):
                    with self.subTest(message=message), self.assertRaisesRegex(
                        ArealDistributedPolicyError,
                        message,
                    ):
                        distributed._validated_pending_w_candidate_path(
                            pending=changed_pending,
                            candidate_path=str(candidate),
                            candidate_dcp_manifest_sha256=supplied_manifest,
                        )

    def test_parent_dcp_baseline_allows_only_optional_lazy_adamw_init(self) -> None:
        distributed._require_post_parent_optimizer_baseline(
            pre_parent_dcp_step=0,
            post_parent_dcp_step=1,
        )
        distributed._require_post_parent_optimizer_baseline(
            pre_parent_dcp_step=7,
            post_parent_dcp_step=7,
        )
        for observed in (-1, 2):
            with self.subTest(observed=observed), self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "optional one-step lazy initialization",
            ):
                distributed._require_post_parent_optimizer_baseline(
                    pre_parent_dcp_step=0,
                    post_parent_dcp_step=observed,
                )

    def test_worker_update_stats_fail_closed_before_receipt_use(self) -> None:
        self.assertEqual(
            distributed._normalized_update_stats(
                {"update_successful": 1.0, "grad_norm": 0.25, "lr": 1e-6}
            )["update_successful_count"],
            1,
        )
        for value in (
            {"update_successful": 0.0, "grad_norm": 0.25, "lr": 1e-6},
            {"update_successful": 1.0, "grad_norm": float("nan"), "lr": 1e-6},
            {"update_successful": 1.0, "grad_norm": 0.25, "lr": 0.0},
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "does not prove one successful update",
            ):
                distributed._normalized_update_stats(value)

    def test_plural_sources_are_revalidated_memberwise_then_flattened_in_order(self) -> None:
        version = JointVersion(
            policy="policy",
            harness_controller="harness",
            harness_artifact="artifact",
            tool_schema="tool",
            parser="parser",
            environment="environment",
            evaluator="evaluator",
            tokenizer="tokenizer",
            context_builder="context",
        )
        stable_s = [str(member + 1) * 64 for member in range(4)]
        binding = _seal(
            {
                "schema_version": "jph.multi-s-source-binding.v1",
                "joint_version_id": version.version_id,
                "batch_record_sha256": "5" * 64,
                "batch_aggregate_sha256": "6" * 64,
                "member_claim_sha256s": [
                    str(member + 7) * 64 for member in range(3)
                ]
                + ["b" * 64],
                "s_record_sha256s": stable_s,
                "policy_sample_count": 4,
                "harness_action_count": 4,
            }
        )
        admissions = [
            SimpleNamespace(
                record_sha256=str(member + 2) * 64,
                inference_engine_version=7,
                export_style="individual",
                samples=({"sample_id": f"{member}-0"},),
            )
            for member in range(4)
        ]
        admission_records = tuple(
            {"admission": member} for member in range(4)
        )
        sources = tuple(
            {"s": member, "record_sha256": stable_s[member]}
            for member in range(4)
        )
        validator = Mock(side_effect=admissions)
        with patch.object(
            distributed,
            "validate_areal_policy_optimizer_source",
            validator,
        ):
            validated, flattened = distributed._validate_ordered_sources(
                admission_records,
                sources,
                active_joint_version=version,
                source_binding=binding,
            )

        self.assertEqual(validator.call_count, 4)
        self.assertEqual(validated, tuple(admissions))
        self.assertEqual(
            [
                (member, sample, value["sample_id"])
                for member, sample, value in flattened
            ],
            [(member, 0, f"{member}-0") for member in range(4)],
        )
        self.assertEqual(
            distributed._partition_indices(len(flattened)),
            ((0,), (1,), (2,), (3,)),
        )
        with self.assertRaisesRegex(ArealDistributedPolicyError, "one-to-one"):
            distributed._validate_ordered_sources(
                admission_records,
                sources[:3],
                active_joint_version=version,
                source_binding=binding,
            )

    def test_aggregate_binds_exact_project_classes_scheduler_and_all_ranks(self) -> None:
        receipt, candidate, _worker_states = _fixture()

        audit = validate_remote_optimizer_receipt(receipt)
        candidate_audit = validate_distributed_policy_candidate(
            candidate,
            active_joint_version=_version(),
        )
        routed_audit = validate_areal_policy_candidate(
            candidate,
            active_joint_version=_version(),
        )

        self.assertEqual(
            receipt["controller_class"], JPH_AREAL_DISTRIBUTED_CONTROLLER_CLASS
        )
        self.assertEqual(audit.global_sample_count, 4)
        self.assertEqual(candidate_audit.transaction_id, "m0-transaction")
        self.assertEqual(routed_audit.record_sha256, candidate_audit.record_sha256)
        self.assertEqual(candidate_audit.trainable_token_count, 8)
        self.assertEqual(
            candidate_audit.source_joint_credit_sha256,
            candidate["source_binding"]["record_sha256"],
        )
        self.assertEqual(len(audit.runtime_states), 4)
        self.assertEqual(
            audit.rank0_lr_scheduler_state_after,
            receipt["rank0_lr_scheduler_state_after"],
        )
        self.assertTrue(
            all(
                item["engine_class"] == JPH_AREAL_DISTRIBUTED_ACTOR_CLASS
                for item in receipt["rank_receipts"]
            )
        )

        tampered = deepcopy(candidate)
        tampered["source_binding"]["s_record_sha256s"][0] = "c" * 64
        tampered["source_binding"] = _seal(tampered["source_binding"])
        tampered = _seal(tampered)
        with self.assertRaisesRegex(
            ArealDistributedPolicyError, "canonical multi-S binding"
        ):
            validate_distributed_policy_candidate(tampered)

    def test_first_distributed_step_requires_exactly_four_samples(self) -> None:
        self.assertEqual(distributed._partition_indices(4), ((0,), (1,), (2,), (3,)))
        for count in (0, 3, 5, 6, 7, 8):
            with self.subTest(count=count), self.assertRaisesRegex(
                ArealDistributedPolicyError, "exactly four"
            ):
                distributed._partition_indices(count)

    def test_candidate_rejects_remote_joint_version_not_bound_to_parent(self) -> None:
        _aggregate, candidate, _worker_states = _fixture()
        tampered = deepcopy(candidate)
        remote = tampered["optimizer"]["remote_optimizer_receipt"]
        remote["rank_receipts"] = [
            _seal({**rank_receipt, "joint_version_id": "wrong-joint-version"})
            for rank_receipt in remote["rank_receipts"]
        ]
        remote["joint_version_id"] = "wrong-joint-version"
        tampered["optimizer"]["remote_optimizer_receipt"] = _seal(remote)
        tampered = _seal(tampered)

        with self.assertRaisesRegex(
            ArealDistributedPolicyError, "optimizer differs from source or parent"
        ):
            validate_distributed_policy_candidate(tampered)

    def test_candidate_requires_exactly_four_unique_global_samples(self) -> None:
        _aggregate, candidate, _worker_states = _fixture()
        tampered = deepcopy(candidate)
        tampered["transaction"]["members"][1]["sample_ids"] = ["sample-0"]
        tampered["transaction"]["source_batch_identity_sha256"] = _digest(
            {
                "schema_version": "jph.m0-ordered-policy-source-batch.v1",
                "members": tampered["transaction"]["members"],
            }
        )
        tampered = _seal(tampered)

        with self.assertRaisesRegex(
            ArealDistributedPolicyError, "canonical multi-S binding"
        ):
            validate_distributed_policy_candidate(tampered)

    def test_live_capability_binds_full_rng_state_to_public_digests(self) -> None:
        live = _live_candidate()
        self.assertIs(require_live_remote_policy_candidate(live), live)

        forged_states = [dict(item) for item in live.worker_states]
        forged_states[2] = deepcopy(forged_states[2])
        forged_states[2]["rank_runtime_state"]["python_random_state"][1][0] += 1
        forged_states[2] = _seal(forged_states[2])
        forged = LiveArealDistributedPolicyCandidate._create(
            receipt=live.receipt,
            worker_states=forged_states,
            token=distributed._LIVE_CANDIDATE_TOKEN,
        )
        with self.assertRaisesRegex(
            ArealDistributedPolicyError, "RNG state differs"
        ):
            require_live_remote_policy_candidate(forged)

    def test_pending_transaction_rejects_same_or_next_macro_before_mutation(self) -> None:
        distributed._require_new_transaction(None)
        for pending in (
            {"transaction_id": "same-transaction"},
            {"transaction_id": "next-transaction"},
        ):
            with self.assertRaisesRegex(
                ArealDistributedPolicyError, "earlier M0 candidate"
            ):
                distributed._require_new_transaction(pending)


class DistributedPolicyControllerTests(unittest.TestCase):
    def test_x_current_state_binds_all_four_ranks_to_t_and_collective_export(
        self,
    ) -> None:
        controller = _controller()
        with tempfile.TemporaryDirectory() as temporary:
            live = _live_candidate_with_real_checkpoints(Path(temporary))
            export = build_distributed_serving_export_receipt(
                _serving_export_rank_receipts(live),
                candidate=live,
            )
            rank_receipts = _current_state_rank_receipts(live, export)
            controller._dispatch_rank_specific = Mock(return_value=rank_receipts)
            current = controller.attest_m0_policy_candidate_current_state(
                live,
                distributed_serving_export_receipt=export,
                candidate_serving_export_lineage_sha256="8" * 64,
            )
            self.assertEqual(
                current["schema_version"],
                M0_DISTRIBUTED_POLICY_CURRENT_STATE_SCHEMA,
            )
            self.assertEqual(
                [item["worker_rank"] for item in current["rank_receipts"]],
                list(range(4)),
            )
            self.assertEqual(
                current["candidate_serving_parameter_sha256"],
                export["candidate"]["parameter_sha256"],
            )
            self.assertFalse(
                current["evidence_scope"][
                    "live_weight_parameter_digest_observed"
                ]
            )
            call = controller._dispatch_rank_specific.call_args
            self.assertEqual(
                call.args[0], "attest_m0_policy_candidate_current_state"
            )
            self.assertEqual(
                [item["rank_runtime_state"]["rank"] for item in call.args[1]],
                list(range(4)),
            )
            validate_distributed_policy_current_state_receipt(
                current,
                candidate=live,
                distributed_serving_export_receipt=export,
                candidate_serving_export_lineage_sha256="8" * 64,
            )

    def test_x_current_state_missing_crossed_and_tampered_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            live = _live_candidate_with_real_checkpoints(Path(temporary))
            export = build_distributed_serving_export_receipt(
                _serving_export_rank_receipts(live),
                candidate=live,
            )
            receipts = _current_state_rank_receipts(live, export)
            with self.assertRaisesRegex(
                ArealDistributedPolicyError, "exactly four rank"
            ):
                build_distributed_policy_current_state_receipt(
                    receipts[:-1],
                    candidate=live,
                    distributed_serving_export_receipt=export,
                    candidate_serving_export_lineage_sha256="8" * 64,
                )

            tampered = deepcopy(receipts)
            tampered[2]["runtime_rng_state_sha256"] = "9" * 64
            tampered[2] = _seal(tampered[2])
            with self.assertRaisesRegex(ArealDistributedPolicyError, "rank 2"):
                build_distributed_policy_current_state_receipt(
                    tampered,
                    candidate=live,
                    distributed_serving_export_receipt=export,
                    candidate_serving_export_lineage_sha256="8" * 64,
                )

            with self.assertRaisesRegex(ArealDistributedPolicyError, "rank 0"):
                build_distributed_policy_current_state_receipt(
                    receipts,
                    candidate=live,
                    distributed_serving_export_receipt=export,
                    candidate_serving_export_lineage_sha256="a" * 64,
                )

            aggregate = build_distributed_policy_current_state_receipt(
                receipts,
                candidate=live,
                distributed_serving_export_receipt=export,
                candidate_serving_export_lineage_sha256="8" * 64,
            )
            forged = deepcopy(aggregate)
            forged["candidate_serving_parameter_sha256"] = "b" * 64
            forged = _seal(forged)
            with self.assertRaisesRegex(
                ArealDistributedPolicyError, "differs from four rank"
            ):
                validate_distributed_policy_current_state_receipt(
                    forged,
                    candidate=live,
                    distributed_serving_export_receipt=export,
                    candidate_serving_export_lineage_sha256="8" * 64,
                )

    def test_collective_serving_export_aggregates_and_persists_all_four_ranks(
        self,
    ) -> None:
        controller = _controller()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = _live_candidate_with_real_checkpoints(root)
            rank_receipts = _serving_export_rank_receipts(live)
            controller._dispatch_rank_specific = Mock(return_value=rank_receipts)
            export_root = root / "collective-serving-export"
            aggregate = controller.materialize_m0_serving_export_pair(
                live,
                export_root=export_root,
            )
            persisted = json.loads(
                (export_root / "distributed-serving-export.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            aggregate["schema_version"], M0_DISTRIBUTED_SERVING_EXPORT_SCHEMA
        )
        self.assertEqual(persisted, aggregate)
        self.assertEqual(
            [item["worker_rank"] for item in aggregate["rank_receipts"]],
            list(range(4)),
        )
        self.assertTrue(
            aggregate["evidence_scope"]["all_four_actor_ranks_attested"]
        )
        self.assertFalse(aggregate["evidence_scope"]["policy_optimizer_update"])
        call = controller._dispatch_rank_specific.call_args
        self.assertEqual(call.args[0], "materialize_m0_serving_export_pair")
        self.assertEqual(len(call.args[1]), 4)
        self.assertTrue(all(kwargs == call.args[1][0] for kwargs in call.args[1]))

    def test_collective_serving_export_rank_disagreement_and_rpc_error_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = _live_candidate_with_real_checkpoints(root)
            receipts = _serving_export_rank_receipts(live)
            crossed = deepcopy(receipts)
            crossed[2]["candidate"]["parameter_sha256"] = "8" * 64
            crossed[2] = _seal(crossed[2])
            with self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "ranks disagree",
            ):
                build_distributed_serving_export_receipt(crossed, candidate=live)

            identical = deepcopy(receipts)
            for rank, receipt in enumerate(identical):
                receipt["candidate"]["parameter_sha256"] = receipt["parent"][
                    "parameter_sha256"
                ]
                identical[rank] = _seal(receipt)
            with self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "parent and candidate parameters are identical",
            ):
                build_distributed_serving_export_receipt(
                    identical,
                    candidate=live,
                )

            controller = _controller()
            controller._dispatch_rank_specific = Mock(
                return_value=[
                    receipts[0],
                    receipts[1],
                    RuntimeError("rank failed"),
                    receipts[3],
                ]
            )
            export_root = root / "failed-export"
            with self.assertRaisesRegex(
                ArealDistributedPolicyError,
                "failed collective serving export",
            ):
                controller.materialize_m0_serving_export_pair(
                    live,
                    export_root=export_root,
                )
            self.assertFalse(
                (export_root / "distributed-serving-export.json").exists()
            )

    def test_optimizer_rpc_is_single_attempt_and_preserves_all_four_results(self) -> None:
        calls: list[dict[str, object]] = []

        class Scheduler:
            async def async_call_engine(self, **kwargs):
                calls.append(kwargs)
                return kwargs["worker_id"]

        controller = _controller()
        controller.scheduler = Scheduler()
        controller._engine_name = lambda rank: f"actor/{rank}"
        results = asyncio.run(
            controller._call_rank_specific(
                "run_m0_policy_candidate_step",
                [{"local_sample_indices": [rank]} for rank in range(4)],
            )
        )

        self.assertEqual(results, [f"actor/{rank}" for rank in range(4)])
        self.assertEqual([call["max_retries"] for call in calls], [1] * 4)
        self.assertTrue(
            all(call["rpc_meta"] == {"broadcast": False} for call in calls)
        )

        calls.clear()
        asyncio.run(
            controller._call_rank_specific(
                "rollback_m0_policy_candidate",
                [{"transaction_id": "m0-transaction"} for _ in range(4)],
            )
        )
        self.assertEqual([call["max_retries"] for call in calls], [1] * 4)

    def test_w_collective_restore_and_continuation_cover_exactly_four_ranks(self) -> None:
        controller = _controller()
        live = _live_candidate()
        restore_results = _restore_rank_receipts(live, "uninterrupted")
        controller._dispatch_rank_specific = Mock(return_value=restore_results)

        restore = controller.attest_m0_live_policy_candidate_for_w(live)
        self.assertFalse(restore["evidence_scope"]["exact_joint_recovery"])
        self.assertEqual(
            [item["worker_rank"] for item in restore["rank_receipts"]],
            list(range(4)),
        )
        restore_call = controller._dispatch_rank_specific.call_args
        self.assertEqual(
            restore_call.args[0],
            "attest_m0_live_policy_candidate_for_w",
        )
        self.assertEqual(len(restore_call.args[1]), 4)
        self.assertEqual(
            [item["rank_runtime_state"]["rank"] for item in restore_call.args[1]],
            list(range(4)),
        )
        self.assertTrue(
            restore["evidence_scope"]["live_candidate_state_attested"]
        )
        self.assertFalse(restore["evidence_scope"]["candidate_dcp_loaded"])
        self.assertTrue(
            all("branch_id" not in kwargs for kwargs in restore_call.args[1])
        )

        continuation_results = _continuation_rank_receipts(
            live,
            restore,
            "uninterrupted",
        )
        controller._dispatch_rank_specific = Mock(return_value=continuation_results)
        continuation = controller.run_m0_policy_recovery_continuation(
            live,
            restore_receipt=restore,
            branch_id="uninterrupted",
        )
        self.assertFalse(continuation["evidence_scope"]["exact_joint_recovery"])
        self.assertTrue(
            continuation["evidence_scope"]["bound_pre_batch_sources_reused"]
        )
        self.assertEqual(
            controller._dispatch_rank_specific.call_args.args[0],
            "run_m0_policy_recovery_continuation",
        )
        self.assertTrue(
            all(
                set(item)
                == {
                    "branch_id",
                    "transaction_id",
                    "policy_candidate_sha256",
                    "restore_receipt",
                }
                for item in controller._dispatch_rank_specific.call_args.args[1]
            )
        )

        recovered_results = _restore_rank_receipts(live, "recovered")
        controller._dispatch_rank_specific = Mock(return_value=recovered_results)
        recovered = controller.restore_m0_policy_candidate_for_w(
            live,
            branch_id="recovered",
        )
        recovered_call = controller._dispatch_rank_specific.call_args
        self.assertEqual(
            recovered_call.args[0],
            "restore_m0_policy_candidate_for_w",
        )
        self.assertTrue(
            all(
                kwargs["branch_id"] == "recovered"
                for kwargs in recovered_call.args[1]
            )
        )
        self.assertFalse(
            recovered["evidence_scope"]["live_candidate_state_attested"]
        )
        self.assertTrue(recovered["evidence_scope"]["candidate_dcp_loaded"])
        self.assertTrue(recovered["evidence_scope"]["optimizer_state_loaded"])

    def test_w_uninterrupted_and_recovered_state_digests_match_only_if_rankwise_equal(self) -> None:
        controller = _controller()
        live = _live_candidate()
        continuations = []
        for branch in ("uninterrupted", "recovered"):
            restore_results = _restore_rank_receipts(live, branch)
            controller._dispatch_rank_specific = Mock(return_value=restore_results)
            restore = (
                controller.attest_m0_live_policy_candidate_for_w(live)
                if branch == "uninterrupted"
                else controller.restore_m0_policy_candidate_for_w(
                    live,
                    branch_id=branch,
                )
            )
            continuation_results = _continuation_rank_receipts(
                live,
                restore,
                branch,
            )
            controller._dispatch_rank_specific = Mock(
                return_value=continuation_results
            )
            continuations.append(
                controller.run_m0_policy_recovery_continuation(
                    live,
                    restore_receipt=restore,
                    branch_id=branch,
                )
            )
        self.assertEqual(
            continuations[0]["continuation_state_sha256"],
            continuations[1]["continuation_state_sha256"],
        )
        self.assertNotEqual(
            continuations[0]["rank_receipts"][0][
                "continuation_dcp_manifest_sha256"
            ],
            continuations[1]["rank_receipts"][0][
                "continuation_dcp_manifest_sha256"
            ],
        )

        tampered = deepcopy(continuations[1]["rank_receipts"])
        tampered[2]["runtime_rng_state_after_sha256"] = "e" * 64
        tampered[2] = _seal(tampered[2])
        controller._dispatch_rank_specific = Mock(
            return_value=_restore_rank_receipts(live, "recovered")
        )
        recovered_restore = controller.restore_m0_policy_candidate_for_w(
            live,
            branch_id="recovered",
        )
        for item in tampered:
            item["restore_receipt_sha256"] = recovered_restore["record_sha256"]
            item.update(_seal(item))
        controller._dispatch_rank_specific = Mock(return_value=tampered)
        changed = controller.run_m0_policy_recovery_continuation(
            live,
            restore_receipt=recovered_restore,
            branch_id="recovered",
        )
        self.assertNotEqual(
            continuations[0]["continuation_state_sha256"],
            changed["continuation_state_sha256"],
        )

    def test_w_missing_rank_tampered_rank_and_fake_post_batch_fail_closed(self) -> None:
        controller = _controller()
        live = _live_candidate()
        missing = _restore_rank_receipts(live, "uninterrupted")[:-1]
        controller._dispatch_rank_specific = Mock(return_value=missing)
        with self.assertRaisesRegex(ArealDistributedPolicyError, "four rank"):
            controller.attest_m0_live_policy_candidate_for_w(live)

        tampered = _restore_rank_receipts(live, "uninterrupted")
        tampered[2]["worker_rank"] = 1
        tampered[2] = _seal(tampered[2])
        controller._dispatch_rank_specific = Mock(return_value=tampered)
        with self.assertRaisesRegex(ArealDistributedPolicyError, "rank 2"):
            controller.attest_m0_live_policy_candidate_for_w(live)

        wrong_action = _restore_rank_receipts(live, "uninterrupted")
        wrong_action[0]["evidence_scope"]["live_candidate_state_attested"] = False
        wrong_action[0]["evidence_scope"]["candidate_dcp_loaded"] = True
        wrong_action[0] = _seal(wrong_action[0])
        controller._dispatch_rank_specific = Mock(return_value=wrong_action)
        with self.assertRaisesRegex(ArealDistributedPolicyError, "rank 0"):
            controller.attest_m0_live_policy_candidate_for_w(live)

        with self.assertRaisesRegex(
            ArealDistributedPolicyError,
            "attest the live candidate",
        ):
            controller.restore_m0_policy_candidate_for_w(
                live,
                branch_id="uninterrupted",
            )

        with self.assertRaises(TypeError):
            controller.run_m0_policy_recovery_continuation(
                live,
                restore_receipt={},
                branch_id="uninterrupted",
                post_batch={"guessed_interaction_ids": ["forged"]},
            )

    def test_commit_rejects_early_or_wrong_live_transaction(self) -> None:
        controller = _controller()
        live = _live_candidate()
        with self.assertRaisesRegex(
            ArealDistributedPolicyError, "completed native Y"
        ):
            controller.commit_m0_policy_candidate(
                live,
                production_activation=object(),
            )

        forged_states = [dict(item) for item in live.worker_states]
        forged_states[0] = deepcopy(forged_states[0])
        forged_states[0]["transaction_id"] = "wrong-transaction"
        forged_states[0] = _seal(forged_states[0])
        forged = LiveArealDistributedPolicyCandidate._create(
            receipt=live.receipt,
            worker_states=forged_states,
            token=distributed._LIVE_CANDIDATE_TOKEN,
        )
        with self.assertRaisesRegex(
            ArealDistributedPolicyError, "differs from T receipt"
        ):
            controller.commit_m0_policy_candidate(
                forged,
                production_activation=object(),
            )

    def test_one_rank_commit_failure_is_not_reported_as_terminal(self) -> None:
        from jphrl.training.joint_activation import ProductionJointActivationResult

        controller = _controller()
        live = _live_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attestation = root / "attestation.json"
            attestation.write_text("{}", encoding="utf-8")
            activation = ProductionJointActivationResult(
                activation_id="activation",
                parent_release_id="parent",
                candidate_release_id="candidate",
                active_release_id="candidate",
                outcome="candidate_active",
                journal_path=root / "journal.json",
                attestation_path=attestation,
                attestation_sha256="a" * 64,
                rollback_record_path=root / "rollback.json",
                rollback_record_sha256="b" * 64,
                evidence_scope={},
            )
            dispatch = Mock(
                return_value=[{}, {}, RuntimeError("rank 2 failed"), {}]
            )
            controller._dispatch_rank_specific = dispatch

            with self.assertRaisesRegex(
                ArealDistributedPolicyError, "one or more workers"
            ):
                controller.commit_m0_policy_candidate(
                    live,
                    production_activation=activation,
                )

        self.assertEqual(
            dispatch.call_args.args[0], "commit_m0_policy_candidate"
        )


if __name__ == "__main__":
    unittest.main()
