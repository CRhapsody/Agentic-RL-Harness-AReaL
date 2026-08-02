from __future__ import annotations

import os
import sys
from pathlib import Path

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
