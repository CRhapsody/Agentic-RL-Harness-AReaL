from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.dataset import get_custom_dataset
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler
from areal.utils import logging, seeding
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats


logger = logging.getLogger("JPHAReaLJointBridgeEval")


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _canonical(value: Mapping[str, object]) -> bytes:
    unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
    return json.dumps(
        unsigned,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _trajectory_binding(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: _plain(value[key])
        for key in (
            "input_ids",
            "loss_mask",
            "logprobs",
            "versions",
            "attention_mask",
            "rewards",
        )
    }


def _binding_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(_trajectory_binding(value))).hexdigest()


def _write_same_backend_scores(
    trajectories: list[dict[str, Any]],
    rescored_logprobs: list[Any],
    *,
    bridge_dir: Path,
    scoring_backend: str,
    engine_version_before_score: int,
    engine_version_after_score: int,
) -> None:
    if len(trajectories) != len(rescored_logprobs):
        raise RuntimeError("same-backend score count differs from trajectory count")
    root = Path(os.environ["JPH_ROOT"]).resolve()
    score_dir = Path(os.environ["JPH_AREAL_SAME_BACKEND_SCORE_DIR"]).resolve()
    if not score_dir.is_relative_to(root):
        raise ValueError(f"same-backend score directory escapes JPH_ROOT: {score_dir}")
    score_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    bridge_paths = sorted(bridge_dir.glob("bridge-*.json"))
    if len(bridge_paths) != len(trajectories):
        raise RuntimeError("bridge record count differs from scored trajectory count")
    bridges_by_binding: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for bridge_path in bridge_paths:
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        binding_id = _binding_sha256(bridge["areal_trace"]["tensor_dict"])
        bridges_by_binding.setdefault(binding_id, []).append((bridge_path, bridge))

    consumed_bridge_paths: set[Path] = set()
    for trajectory, rescored in zip(trajectories, rescored_logprobs):
        binding_id = _binding_sha256(trajectory)
        candidates = bridges_by_binding.get(binding_id, [])
        if not candidates:
            raise RuntimeError("scored trajectory has no matching bridge record")
        bridge_path, bridge = candidates.pop(0)
        consumed_bridge_paths.add(bridge_path)
        request_id = str(bridge["request_id"])
        record: dict[str, object] = {
            "schema_version": "jph.areal-same-backend-logprob.v2",
            "request_id": request_id,
            "bridge_record_sha256": bridge["record_sha256"],
            "trajectory_binding_sha256": binding_id,
            "scoring_origin": {
                "api": "RolloutController.compute_logp",
                "lifecycle": "same-controller-after-wait-before-destroy",
                "backend": scoring_backend,
                "engine_version_before_score": engine_version_before_score,
                "engine_version_after_score": engine_version_after_score,
                "policy_release_id": bridge["joint_version"]["policy"],
                "behavior_revision": os.environ["JPH_BEHAVIOR_REVISION"],
                "areal_commit": os.environ["JPH_AREAL_COMMIT"],
                "project_commit": os.environ["JPH_PROJECT_COMMIT"],
            },
            "input_ids": _plain(trajectory["input_ids"]),
            "loss_mask": _plain(trajectory["loss_mask"]),
            "stored_logprobs": _plain(trajectory["logprobs"]),
            "rescored_logprobs": _plain(rescored),
            "versions": _plain(trajectory["versions"]),
        }
        record["record_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
        identity = hashlib.sha256(
            request_id.encode("utf-8")
        ).hexdigest()[:20]
        path = score_dir / f"same-backend-score-{identity}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                json.dump(
                    record,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
    if consumed_bridge_paths != set(bridge_paths):
        raise RuntimeError("not every bridge record was consumed by same-backend scoring")


def _task_limit() -> int:
    raw = os.environ.get("JPH_AREAL_JOINT_BRIDGE_TASKS", "4")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"JPH_AREAL_JOINT_BRIDGE_TASKS must be an integer: {raw}"
        ) from exc
    if value < 1 or value > 8:
        raise ValueError(
            f"JPH_AREAL_JOINT_BRIDGE_TASKS must be between 1 and 8: {value}"
        )
    return value


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    bridge_dir = Path(os.environ["JPH_AREAL_JOINT_BRIDGE_DIR"]).resolve()
    logging.setup_file_logging(str(bridge_dir.parent / "areal-bridge-eval.log"))

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key="jph-areal-joint-bridge-eval")
    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if config.scheduler.type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif config.scheduler.type == "ray":
        scheduler = RayScheduler(exp_config=config)
    elif config.scheduler.type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise ValueError(f"unsupported scheduler type: {config.scheduler.type}")

    dataset = get_custom_dataset(
        split="test",
        path=config.valid_dataset.path,
        type=config.valid_dataset.type,
        max_length=config.valid_dataset.max_length,
        tokenizer=tokenizer,
    )
    dataloader = create_dataloader(
        dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )
    config.rollout.max_head_offpolicyness = int(1e12)

    if rollout_alloc.backend == "sglang":
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=rollout_alloc.parallel.tp_size,
            base_gpu_id=0,
        )
    elif rollout_alloc.backend == "vllm":
        engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=rollout_alloc.parallel.tp_size,
            pp_size=rollout_alloc.parallel.pp_size,
        )
    else:
        raise ValueError(f"unsupported rollout backend: {rollout_alloc.backend}")

    controller = engine_cls.as_controller(config.rollout, scheduler)
    submitted = 0
    results = []
    try:
        controller.initialize(role="jph-joint-bridge-rollout", server_args=server_args)
        workflow_kwargs = {
            "reward_fn": "areal.reward.gsm8k.gsm8k_reward_fn",
            "gconfig": config.gconfig,
            "tokenizer": config.tokenizer_path,
            "enable_thinking": False,
            "harness_seed": config.seed,
        }
        limit = _task_limit()
        for batch in dataloader:
            for item in batch:
                controller.submit(
                    item,
                    workflow=(
                        "jphrl.areal_joint_bridge_workflow."
                        "ArealJointBridgeWorkflow"
                    ),
                    workflow_kwargs=workflow_kwargs,
                    group_size=1,
                )
                submitted += 1
                if submitted >= limit:
                    break
            if submitted >= limit:
                break
        if submitted != limit:
            raise RuntimeError(f"requested {limit} tasks but dataset yielded {submitted}")
        results = controller.wait(submitted, timeout=900.0)
        if len(results) != submitted or any(result is None for result in results):
            raise RuntimeError(
                f"AReaL returned {len(results)} results for {submitted} submitted tasks"
            )
        engine_version_before_score = controller.get_version()
        rescored_logprobs = controller.compute_logp(results)
        engine_version_after_score = controller.get_version()
        expected_policy_version = int(os.environ["JPH_EXPECTED_POLICY_VERSION"])
        if not (
            engine_version_before_score
            == engine_version_after_score
            == expected_policy_version
        ):
            raise RuntimeError(
                "inference engine version changed across same-backend scoring"
            )
        _write_same_backend_scores(
            results,
            rescored_logprobs,
            bridge_dir=bridge_dir,
            scoring_backend=str(config.rollout.backend),
            engine_version_before_score=engine_version_before_score,
            engine_version_after_score=engine_version_after_score,
        )
        logger.info(
            "AReaL joint bridge evaluation: %s",
            tabulate_stats(controller.export_stats()),
        )
    finally:
        controller.destroy()

    bridge_files = sorted(bridge_dir.glob("bridge-*.json"))
    if len(bridge_files) != submitted:
        raise RuntimeError(
            f"expected {submitted} bridge files, found {len(bridge_files)} in {bridge_dir}"
        )
    print(
        f"AReaL joint bridge complete: submitted={submitted} "
        f"accepted={len(results)} bridge_records={len(bridge_files)}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
