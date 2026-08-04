from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jphrl.training.production_checkpoint import (
    LiveExactDistributedJointRecovery,
    LiveExactJointRecovery,
    PolicyRankCheckpointState,
    PolicySchedulerCheckpointState,
    ProductionCheckpointError,
    RuntimeCursorState,
    RuntimeTopology,
    capture_rank_runtime_state,
    distributed_policy_checkpoint_inputs,
    dry_load_production_joint_checkpoint,
    require_live_exact_joint_recovery,
    save_production_joint_checkpoint,
    validate_exact_joint_recovery_evidence,
    validate_production_joint_checkpoint,
    verify_exact_distributed_joint_recovery,
    verify_exact_joint_recovery,
)
from jphrl.training.production_checkpoint import (
    _execute_live_continuation as execute_live_continuation,
)
from jphrl.trajectory.schema import JointVersion


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _version(policy: str, harness: str) -> JointVersion:
    return JointVersion(
        policy=policy,
        harness_controller=harness,
        harness_artifact="artifact-v1",
        tool_schema="tools-v1",
        parser="parser-v1",
        environment="env-v1",
        evaluator="evaluator-v1",
        tokenizer="tokenizer-v1",
        context_builder="context-v1",
    )


class FakeScheduler:
    def __init__(self) -> None:
        self.state = {"last_epoch": 1, "_step_count": 2, "base_lrs": [0.001]}

    def state_dict(self):
        return deepcopy(self.state)


class FakeActor:
    def __init__(self) -> None:
        self.lr_scheduler = FakeScheduler()

    def get_version(self) -> int:
        return 7


class FakeDistributedController:
    """Deliberately has no local scheduler or optimizer state."""

    def get_version(self) -> int:
        return 7


def _dcp_manifest(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_bytes(path.read_bytes()),
                }
            )
    return {"files": files, "manifest_sha256": _sha256_json({"files": files})}


def _bundle(
    root: Path,
    *,
    world_size: int = 1,
) -> tuple[dict[str, object], SimpleNamespace]:
    parent_version = _version("policy-parent", "harness-parent")
    candidate_version = _version("areal-policy-candidate", "harness-candidate")
    parent = root / "policy-parent.dcp"
    candidate = root / "policy-candidate.dcp"
    parent.mkdir()
    candidate.mkdir()
    for rank in range(world_size):
        (parent / f"rank{rank}.distcp").write_bytes(
            f"parent-model-and-optim-rank-{rank}".encode()
        )
        (candidate / f"rank{rank}.distcp").write_bytes(
            f"candidate-model-and-optim-rank-{rank}".encode()
        )
    harness = root / "harness-candidate.pt"
    harness.write_bytes(b"harness-model-adam-generator")
    source = "a" * 64
    policy_receipt = {
        "schema_version": "jph.areal-policy-candidate.v2",
        "transaction": {
            "transaction_id": "macro-7",
            "episode_id": "episode-7",
            "source_admission_sha256": "e" * 64,
            "source_joint_credit_sha256": source,
            "trainable_token_count": 3,
        },
        "optimizer": {
            "actor_type": "areal.engine.fsdp_engine.FSDPPPOActor",
            "optimizer_type": "torch.optim.adamw.AdamW",
            "config": {},
            "stats": {
                "grad_norm": 1.0,
                "grad_norm_count": 1,
                "learning_rate": 0.001,
                "learning_rate_count": 1,
                "optimizer_step_before": 7,
                "optimizer_step_after": 8,
                "update_successful": 1.0,
                "update_successful_count": 1,
            },
            "lr_scheduler": {
                "state_before_sha256": "f" * 64,
                "state_after_sha256": _sha256_json(FakeScheduler().state),
                "step_completed": True,
            },
        },
        "checkpoints": {
            "parent_path": str(parent),
            "parent_manifest": _dcp_manifest(parent),
            "candidate_path": str(candidate),
            "candidate_manifest": _dcp_manifest(candidate),
        },
        "record_sha256": "b" * 64,
    }
    harness_receipt = {
        "schema_version": "jph.torch-harness-candidate.v2",
        "transaction_id": "macro-7",
        "checkpoint_path": str(harness),
        "checkpoint_sha256": _sha256_bytes(harness.read_bytes()),
    }
    bundle = {
        "parent": {"harness_controller_version": parent_version.harness_controller},
        "receipts": {
            "policy": policy_receipt,
            "policy_sha256": "b" * 64,
            "harness": harness_receipt,
            "harness_sha256": "c" * 64,
        },
        "record_sha256": "d" * 64,
    }
    audit = SimpleNamespace(
        macro_step_id="macro-7",
        parent_joint_version=parent_version,
        candidate_joint_version=candidate_version,
        source_joint_credit_sha256=source,
        policy_engine_version=7,
        candidate_policy_engine_version=8,
        record_sha256="d" * 64,
    )
    return bundle, audit


def _topology() -> RuntimeTopology:
    return RuntimeTopology(
        world_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        rank_to_device=("cpu",),
    )


def _topology4() -> RuntimeTopology:
    return RuntimeTopology(
        world_size=4,
        data_parallel_size=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        rank_to_device=("cpu",) * 4,
    )


def _four_rank_inputs(
    bundle: dict[str, object],
    audit: SimpleNamespace,
) -> tuple[
    tuple[object, ...],
    tuple[PolicyRankCheckpointState, ...],
    PolicySchedulerCheckpointState,
]:
    runtime_states = tuple(
        capture_rank_runtime_state(
            rank=rank,
            local_rank=0,
            device="cpu",
        )
        for rank in range(4)
    )
    policy_receipt = bundle["receipts"]["policy"]
    checkpoints = policy_receipt["checkpoints"]
    stats = policy_receipt["optimizer"]["stats"]
    scheduler = policy_receipt["optimizer"]["lr_scheduler"]
    policy_states = tuple(
        PolicyRankCheckpointState(
            rank=runtime.rank,
            local_rank=runtime.local_rank,
            world_size=4,
            device=runtime.device,
            hostname=runtime.hostname,
            physical_gpu_id=str(runtime.rank),
            cuda_visible_devices=str(runtime.rank),
            transaction_id=audit.macro_step_id,
            source_joint_credit_sha256=audit.source_joint_credit_sha256,
            policy_candidate_receipt_sha256=policy_receipt["record_sha256"],
            parent_dcp_manifest_sha256=checkpoints["parent_manifest"][
                "manifest_sha256"
            ],
            candidate_dcp_manifest_sha256=checkpoints["candidate_manifest"][
                "manifest_sha256"
            ],
            actor_public_version=audit.policy_engine_version,
            actor_reserved_version=audit.candidate_policy_engine_version,
            optimizer_step_before=stats["optimizer_step_before"],
            optimizer_step_after=stats["optimizer_step_after"],
            lr_scheduler_state_after_sha256=scheduler["state_after_sha256"],
        )
        for runtime in runtime_states
    )
    scheduler_state = PolicySchedulerCheckpointState(
        owner_rank=0,
        scheduler_class=(f"{FakeScheduler.__module__}.{FakeScheduler.__qualname__}"),
        state=FakeScheduler().state,
    )
    return runtime_states, policy_states, scheduler_state


def _resign(record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = _sha256_json(unsigned)


class ProductionCheckpointTests(unittest.TestCase):
    def _saved(self, root: Path) -> tuple[Path, SimpleNamespace]:
        project = root / "src" / "repo"
        project.mkdir(parents=True)
        components = root / "runs" / "components"
        components.mkdir(parents=True)
        bundle, audit = _bundle(components)
        state = capture_rank_runtime_state(
            rank=0,
            local_rank=0,
            device="cpu",
        )
        with patch(
            "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
            return_value=audit,
        ):
            manifest = save_production_joint_checkpoint(
                checkpoint_root=root / "runs" / "checkpoint-7",
                project_root=project,
                joint_candidate_bundle=bundle,
                actor=FakeActor(),
                harness_policy=None,
                topology=_topology(),
                rank_states=(state,),
                macro_step=7,
                rollout_cursor=11,
                dataloader_cursor=13,
            )
        return manifest, audit

    def _saved4(self, root: Path) -> tuple[Path, SimpleNamespace]:
        project = root / "src" / "repo"
        project.mkdir(parents=True)
        components = root / "runs" / "components"
        components.mkdir(parents=True)
        bundle, audit = _bundle(components, world_size=4)
        runtime_states, policy_states, scheduler_state = _four_rank_inputs(
            bundle, audit
        )
        with patch(
            "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
            return_value=audit,
        ):
            manifest = save_production_joint_checkpoint(
                checkpoint_root=root / "runs" / "checkpoint-7",
                project_root=project,
                joint_candidate_bundle=bundle,
                actor=FakeDistributedController(),
                harness_policy=None,
                topology=_topology4(),
                rank_states=runtime_states,
                policy_rank_states=policy_states,
                policy_scheduler_state=scheduler_state,
                macro_step=7,
                rollout_cursor=11,
                dataloader_cursor=13,
            )
        return manifest, audit

    def test_contract_checkpoint_is_complete_but_never_exact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                manifest, audit = self._saved(root)
                with patch(
                    "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                    return_value=audit,
                ):
                    result = validate_production_joint_checkpoint(
                        manifest,
                        current_topology=_topology(),
                    )
                    dry = dry_load_production_joint_checkpoint(
                        manifest,
                        current_topology=_topology(),
                    )
            self.assertEqual(result.macro_step, 7)
            self.assertEqual(
                (result.rollout_cursor, result.dataloader_cursor), (11, 13)
            )
            self.assertFalse(result.real_areal_checkpoint)
            self.assertFalse(result.real_harness_checkpoint)
            self.assertFalse(result.exact_joint_recovery)
            self.assertFalse(dry["real_component_dry_load"])
            self.assertFalse(dry["evidence_scope"]["exact_joint_recovery"])
            self.assertEqual(
                dry["schema_version"], "jph.production-joint-dry-load-probe.v1"
            )

    def test_manifest_contains_full_lineage_topology_scheduler_rng_and_refs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                manifest, _ = self._saved(root)
                record = json.loads(manifest.read_text())
            self.assertEqual(
                record["identity"]["parent_joint_version_id"],
                _version("policy-parent", "harness-parent").version_id,
            )
            self.assertEqual(
                record["identity"]["candidate_joint_version_id"],
                _version("areal-policy-candidate", "harness-candidate").version_id,
            )
            self.assertEqual(
                record["identity"]["joint_candidate_bundle_sha256"], "d" * 64
            )
            self.assertTrue(record["policy"]["dcp_with_optimizer"])
            self.assertTrue(record["harness"]["checkpoint_contains_optimizer"])
            self.assertIn("lr_scheduler", record["policy"])
            self.assertEqual(len(record["rng"]["rank_states"]), 1)
            self.assertEqual(record["topology"]["rank_to_device"], ["cpu"])
            self.assertFalse(record["evidence_scope"]["exact_joint_recovery"])

    def test_four_rank_checkpoint_binds_every_actor_rank_without_controller_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                manifest, audit = self._saved4(root)
                with patch(
                    "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                    return_value=audit,
                ):
                    result = validate_production_joint_checkpoint(
                        manifest,
                        current_topology=_topology4(),
                    )
                record = json.loads(manifest.read_text())

            self.assertEqual(result.policy_rank_state_count, 4)
            self.assertEqual(result.topology.data_parallel_size, 4)
            self.assertFalse(result.real_areal_checkpoint)
            self.assertFalse(result.exact_joint_recovery)
            self.assertEqual(record["policy"]["fsdp_world_size"], 4)
            self.assertEqual(len(record["policy"]["rank_checkpoint_states"]), 4)
            self.assertEqual(len(record["rng"]["rank_states"]), 4)
            scheduler_path = manifest.parent / record["policy"]["lr_scheduler"]["path"]
            self.assertEqual(json.loads(scheduler_path.read_text())["owner_rank"], 0)
            self.assertTrue(
                record["evidence_scope"]["all_actor_ranks_bound_to_policy_candidate"]
            )
            self.assertEqual(record["harness"]["transaction_id"], "macro-7")

    def test_live_distributed_candidate_adapts_all_worker_states_without_controller_optimizer(self) -> None:
        from tests.test_areal_distributed_policy import _live_candidate

        live = _live_candidate()
        checkpoints = live.receipt["checkpoints"]

        def observed_manifest(source):
            return (
                checkpoints["parent_manifest"]
                if str(source) == checkpoints["parent_path"]
                else checkpoints["candidate_manifest"]
            )

        with patch(
            "jphrl.training.areal_distributed_policy.checkpoint_manifest",
            side_effect=observed_manifest,
        ):
            runtime, policy, scheduler = distributed_policy_checkpoint_inputs(
                live,
                harness_policy=None,
            )
        self.assertEqual([state.rank for state in runtime], list(range(4)))
        self.assertEqual([state.rank for state in policy], list(range(4)))
        self.assertEqual(
            [state.physical_gpu_id for state in policy],
            [str(rank) for rank in range(4)],
        )
        self.assertTrue(
            all(state.world_size == 4 for state in policy)
        )
        self.assertEqual(scheduler.owner_rank, 0)
        self.assertNotIn("optimizer", vars(FakeDistributedController()))

    def test_cpu_fake_controller_cannot_mint_distributed_live_exact(self) -> None:
        from tests.test_areal_distributed_policy import _live_candidate

        with self.assertRaisesRegex(
            ProductionCheckpointError,
            "real project fsdp:d4 controller",
        ):
            verify_exact_distributed_joint_recovery(
                "/unused/manifest.json",
                controller=FakeDistributedController(),
                live_policy_candidate=_live_candidate(),
                harness_policy=object(),
                harness_optimizer=object(),
                current_topology=_topology4(),
                run_harness_optimizer_step=lambda _policy, _optimizer: None,
            )

    def test_four_rank_missing_crossed_or_colliding_worker_state_fails_closed(
        self,
    ) -> None:
        mutations = {
            "missing_policy_rank": lambda runtime, policy: (runtime, policy[:-1]),
            "crossed_source": lambda runtime, policy: (
                runtime,
                (replace(policy[0], source_joint_credit_sha256="9" * 64),) + policy[1:],
            ),
            "duplicate_physical_gpu": lambda runtime, policy: (
                runtime,
                (
                    policy[0],
                    replace(
                        policy[1],
                        physical_gpu_id=policy[0].physical_gpu_id,
                        cuda_visible_devices=policy[0].cuda_visible_devices,
                    ),
                )
                + policy[2:],
            ),
            "nonzero_harness_generator": lambda runtime, policy: (
                (runtime[0], replace(runtime[1], harness_generator_state=[1]))
                + runtime[2:],
                policy,
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                project = root / "src" / "repo"
                project.mkdir(parents=True)
                components = root / "runs" / "components"
                components.mkdir(parents=True)
                bundle, audit = _bundle(components, world_size=4)
                runtime, policy, scheduler = _four_rank_inputs(bundle, audit)
                runtime, policy = mutate(runtime, policy)
                with (
                    patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                    patch(
                        "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                        return_value=audit,
                    ),
                    self.assertRaises(ProductionCheckpointError),
                ):
                    save_production_joint_checkpoint(
                        checkpoint_root=root / "runs" / "checkpoint-7",
                        project_root=project,
                        joint_candidate_bundle=bundle,
                        actor=FakeDistributedController(),
                        harness_policy=None,
                        topology=_topology4(),
                        rank_states=runtime,
                        policy_rank_states=policy,
                        policy_scheduler_state=scheduler,
                        macro_step=7,
                        rollout_cursor=11,
                        dataloader_cursor=13,
                    )

    def test_four_rank_requires_aggregated_scheduler_and_policy_worker_receipts(
        self,
    ) -> None:
        for omitted in ("policy", "scheduler"):
            with (
                self.subTest(omitted=omitted),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                project = root / "src" / "repo"
                project.mkdir(parents=True)
                components = root / "runs" / "components"
                components.mkdir(parents=True)
                bundle, audit = _bundle(components, world_size=4)
                runtime, policy, scheduler = _four_rank_inputs(bundle, audit)
                with (
                    patch.dict("os.environ", {"JPH_ROOT": str(root)}),
                    patch(
                        "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                        return_value=audit,
                    ),
                    self.assertRaisesRegex(
                        ProductionCheckpointError,
                        "multi-rank W requires",
                    ),
                ):
                    save_production_joint_checkpoint(
                        checkpoint_root=root / "runs" / "checkpoint-7",
                        project_root=project,
                        joint_candidate_bundle=bundle,
                        actor=FakeDistributedController(),
                        harness_policy=None,
                        topology=_topology4(),
                        rank_states=runtime,
                        policy_rank_states=None if omitted == "policy" else policy,
                        policy_scheduler_state=(
                            None if omitted == "scheduler" else scheduler
                        ),
                        macro_step=7,
                        rollout_cursor=11,
                        dataloader_cursor=13,
                    )

    def test_missing_scheduler_rng_cursor_joint_version_or_optimizer_ref_fails(
        self,
    ) -> None:
        mutations = (
            lambda record: record["policy"].pop("lr_scheduler"),
            lambda record: record.pop("rng"),
            lambda record: record.pop("cursors"),
            lambda record: record["identity"].pop("candidate_joint_version"),
            lambda record: record["policy"].pop("candidate_dcp_manifest"),
            lambda record: record["harness"].pop("checkpoint_contains_optimizer"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                    manifest, audit = self._saved(root)
                    record = json.loads(manifest.read_text())
                    mutate(record)
                    _resign(record)
                    manifest.write_text(json.dumps(record), encoding="utf-8")
                    with (
                        patch(
                            "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                            return_value=audit,
                        ),
                        self.assertRaises(ProductionCheckpointError),
                    ):
                        validate_production_joint_checkpoint(manifest)

    def test_file_tamper_topology_mismatch_and_credential_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                manifest, audit = self._saved(root)
                record = json.loads(manifest.read_text())
                scheduler = manifest.parent / record["policy"]["lr_scheduler"]["path"]
                scheduler.write_text("{}", encoding="utf-8")
                with (
                    patch(
                        "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                        return_value=audit,
                    ),
                    self.assertRaisesRegex(ProductionCheckpointError, "hash or size"),
                ):
                    validate_production_joint_checkpoint(manifest)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                manifest, audit = self._saved(root)
                mismatch = RuntimeTopology(1, 1, 1, 1, ("cuda:0",))
                with (
                    patch(
                        "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                        return_value=audit,
                    ),
                    self.assertRaisesRegex(
                        ProductionCheckpointError, "topology mismatch"
                    ),
                ):
                    validate_production_joint_checkpoint(
                        manifest,
                        current_topology=mismatch,
                    )

                record = json.loads(manifest.read_text())
                record["joint_candidate_bundle"]["admin_api_key"] = "secret"
                _resign(record)
                manifest.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaisesRegex(ProductionCheckpointError, "credential"):
                    validate_production_joint_checkpoint(manifest)

    def test_exact_recovery_cannot_be_claimed_by_cpu_fake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict("os.environ", {"JPH_ROOT": str(root)}):
                manifest, audit = self._saved(root)
                next_step = {
                    "policy_state_sha256": "1" * 64,
                    "harness_state_sha256": "2" * 64,
                    "harness_optimizer_state_sha256": "3" * 64,
                    "harness_sample_count": 4,
                    "harness_training": True,
                    "policy_optimizer_step": 8,
                    "harness_optimizer_step": 8,
                    "lr_scheduler_state_sha256": "4" * 64,
                    "runtime_rng_state_sha256": "5" * 64,
                    "actor_public_version": 7,
                    "rollout_cursor": {
                        "name": "rollout",
                        "source_sha256": "6" * 64,
                        "consumed_item_sha256": "7" * 64,
                        "position": 12,
                    },
                    "dataloader_cursor": {
                        "name": "dataloader",
                        "source_sha256": "8" * 64,
                        "consumed_item_sha256": "9" * 64,
                        "position": 14,
                    },
                }
                with (
                    patch(
                        "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                        return_value=audit,
                    ),
                    self.assertRaisesRegex(ProductionCheckpointError, "real AReaL"),
                ):
                    verify_exact_joint_recovery(
                        manifest,
                        actor=FakeActor(),
                        current_topology=_topology(),
                        rank=0,
                        admission_record={"record_sha256": "9" * 64},
                        device="cpu",
                        run_policy_optimizer_step=lambda _: None,
                        run_harness_optimizer_step=lambda _policy, _optimizer: None,
                    )

                forged = {
                    "schema_version": "jph.production-joint-recovery-evidence.v2",
                    "checkpoint_manifest_sha256": json.loads(manifest.read_text())[
                        "record_sha256"
                    ],
                    "actor_class": "areal.engine.fsdp_engine.FSDPPPOActor",
                    "harness_policy_class": (
                        "jphrl.harness.torch_learning.TorchHarnessPolicy"
                    ),
                    "harness_optimizer_class": "torch.optim.adam.Adam",
                    "uninterrupted_next_step": next_step,
                    "recovered_next_step": next_step,
                    "evidence_scope": {
                        "framework_owned_state_measurement": True,
                        "real_areal_fsdp_actor_loaded": True,
                        "real_harness_checkpoint_loaded": True,
                        "scheduler_rng_cursor_restored": True,
                        "continuous_next_step_verified": True,
                        "persisted_record_regrants_live_exact": False,
                        "exact_joint_recovery": True,
                    },
                }
                forged["record_sha256"] = _sha256_json(forged)
                with (
                    patch(
                        "jphrl.training.production_checkpoint.validate_joint_candidate_bundle",
                        return_value=audit,
                    ),
                    self.assertRaisesRegex(ProductionCheckpointError, "real AReaL"),
                ):
                    validate_exact_joint_recovery_evidence(
                        forged,
                        manifest=manifest,
                        current_topology=_topology(),
                    )

    def test_persisted_exact_claim_never_recreates_live_capability(self) -> None:
        rollout = RuntimeCursorState(
            name="rollout",
            position=11,
            source_sha256="6" * 64,
            pending_item_sha256="7" * 64,
        )
        dataloader = RuntimeCursorState(
            name="dataloader",
            position=13,
            source_sha256="8" * 64,
            pending_item_sha256="9" * 64,
        )
        next_step = {
            "policy_state_sha256": "1" * 64,
            "harness_state_sha256": "2" * 64,
            "harness_optimizer_state_sha256": "3" * 64,
            "harness_sample_count": 4,
            "harness_training": True,
            "policy_optimizer_step": 8,
            "harness_optimizer_step": 8,
            "lr_scheduler_state_sha256": "4" * 64,
            "runtime_rng_state_sha256": "5" * 64,
            "actor_public_version": 7,
            "rollout_cursor": {
                "name": "rollout",
                "source_sha256": "6" * 64,
                "consumed_item_sha256": "7" * 64,
                "position": 12,
            },
            "dataloader_cursor": {
                "name": "dataloader",
                "source_sha256": "8" * 64,
                "consumed_item_sha256": "9" * 64,
                "position": 14,
            },
        }
        record = {
            "schema_version": "jph.production-joint-recovery-evidence.v2",
            "checkpoint_manifest_sha256": "a" * 64,
            "actor_class": "areal.engine.fsdp_engine.FSDPPPOActor",
            "harness_policy_class": ("jphrl.harness.torch_learning.TorchHarnessPolicy"),
            "harness_optimizer_class": "torch.optim.adam.Adam",
            "uninterrupted_next_step": next_step,
            "recovered_next_step": deepcopy(next_step),
            "evidence_scope": {
                "framework_owned_state_measurement": True,
                "real_areal_fsdp_actor_loaded": True,
                "real_harness_checkpoint_loaded": True,
                "scheduler_rng_cursor_restored": True,
                "continuous_next_step_verified": True,
                "persisted_record_regrants_live_exact": False,
                "exact_joint_recovery": True,
            },
        }
        _resign(record)
        checkpoint = SimpleNamespace(
            manifest_path="/unused/manifest.json",
            record_sha256="a" * 64,
            real_areal_checkpoint=True,
            real_harness_checkpoint=True,
            topology=_topology(),
            rollout_cursor_state=rollout,
            dataloader_cursor_state=dataloader,
            actor_public_version=7,
            parent_joint_version=_version("policy-parent", "harness-parent"),
            candidate_joint_version=_version(
                "areal-policy-candidate", "harness-candidate"
            ),
            macro_step_id="macro-7",
            macro_step=7,
        )
        with (
            patch(
                "jphrl.training.production_checkpoint.validate_production_joint_checkpoint",
                return_value=checkpoint,
            ),
            patch(
                "jphrl.training.production_checkpoint._saved_optimizer_steps",
                return_value=(7, 7),
            ),
        ):
            audit = validate_exact_joint_recovery_evidence(
                record,
                manifest="/unused/manifest.json",
            )
            self.assertTrue(audit["integrity_valid"])
            self.assertTrue(audit["persisted_exact_claim"])
            self.assertFalse(audit["live_exact_joint_recovery"])
            self.assertFalse(audit["exact_joint_recovery"])

            tampered = deepcopy(record)
            tampered["evidence_scope"]["exact_joint_recovery"] = False
            _resign(tampered)
            with self.assertRaisesRegex(ProductionCheckpointError, "evidence scope"):
                validate_exact_joint_recovery_evidence(
                    tampered,
                    manifest="/unused/manifest.json",
                )

    def test_noop_and_self_reported_callbacks_cannot_mint_live_exact(self) -> None:
        rollout = RuntimeCursorState("rollout", 11, "6" * 64, "7" * 64)
        dataloader = RuntimeCursorState("dataloader", 13, "8" * 64, "9" * 64)
        audit = SimpleNamespace(
            rollout_cursor_state=rollout,
            dataloader_cursor_state=dataloader,
            source_joint_credit_sha256="7" * 64,
            actor_public_version=7,
            parent_joint_version=_version("policy-parent", "harness-parent"),
            topology=_topology(),
        )
        policy = SimpleNamespace(parameter_digest="2" * 64, update_step=7)
        restored = SimpleNamespace(
            audit=audit,
            harness_policy=policy,
            harness_optimizer=SimpleNamespace(),
        )
        actor = FakeActor()
        actor.optimizer = SimpleNamespace(state={"parameter": {"step": 7}})
        admission = {"record_sha256": "9" * 64}
        with (
            patch(
                "jphrl.training.production_checkpoint._real_areal_actor",
                return_value=True,
            ),
            patch(
                "jphrl.training.production_checkpoint._require_real_harness_optimizer"
            ),
            patch(
                "jphrl.training.production_checkpoint._harness_optimizer_step",
                return_value=7,
            ),
            patch(
                "jphrl.training.production_checkpoint._policy_probe_sha256",
                return_value="1" * 64,
            ),
            self.assertRaisesRegex(
                ProductionCheckpointError, "did not advance exactly once"
            ),
        ):
            execute_live_continuation(
                restored,
                actor=actor,
                admission_record=admission,
                device="cpu",
                rank=0,
                saved_policy_step=7,
                saved_harness_step=7,
                run_policy_optimizer_step=lambda _actor: None,
                run_harness_optimizer_step=lambda _policy, _optimizer: None,
            )

        with (
            patch(
                "jphrl.training.production_checkpoint._real_areal_actor",
                return_value=True,
            ),
            patch(
                "jphrl.training.production_checkpoint._require_real_harness_optimizer"
            ),
            patch(
                "jphrl.training.production_checkpoint._harness_optimizer_step",
                return_value=7,
            ),
            patch(
                "jphrl.training.production_checkpoint._policy_probe_sha256",
                return_value="1" * 64,
            ),
            self.assertRaisesRegex(ProductionCheckpointError, "self-report"),
        ):
            execute_live_continuation(
                restored,
                actor=actor,
                admission_record=admission,
                device="cpu",
                rank=0,
                saved_policy_step=7,
                saved_harness_step=7,
                run_policy_optimizer_step=lambda _actor: {"passed": True},
                run_harness_optimizer_step=lambda _policy, _optimizer: None,
            )

    def test_forged_live_capability_type_fails_closed(self) -> None:
        for forged in (
            {"exact_joint_recovery": True},
            LiveExactJointRecovery(),
            LiveExactDistributedJointRecovery(),
        ):
            with (
                self.subTest(type=type(forged).__qualname__),
                self.assertRaisesRegex(
                    ProductionCheckpointError, "capability is required"
                ),
            ):
                require_live_exact_joint_recovery(forged)


if __name__ == "__main__":
    unittest.main()
