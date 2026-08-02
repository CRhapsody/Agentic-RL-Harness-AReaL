from __future__ import annotations

import argparse
import json
import sys

from .envs.calculator import TASKS
from .harness.controller import SmokeHarnessController
from .harness.spec import HarnessSpec
from .models.base import MockStructuredModel
from .paths import require_within_configured_root
from .runner import run_calculator_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one deterministic JPH-RL smoke task")
    parser.add_argument("--backend", choices=("mock", "hf"), default="mock")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=sorted(TASKS), default="add-17-25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = require_within_configured_root(args.output)
    if args.backend == "mock":
        model = MockStructuredModel()
    else:
        from .models.hf_chat import HuggingFaceChatModel

        model = HuggingFaceChatModel(args.model, device=args.device, revision=args.revision)
    controller = SmokeHarnessController()
    result = run_calculator_smoke(
        model=model,
        task=TASKS[args.task],
        controller=controller,
        harness_spec=HarnessSpec(),
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    result.trace.write_json(destination)
    summary = {
        "success": result.success,
        "reward": result.reward,
        "task_id": args.task,
        "policy_version": result.trace.joint_version.policy,
        "harness_version": result.trace.joint_version.harness_artifact,
        "joint_version_id": result.trace.joint_version.version_id,
        "trace": str(destination),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.success:
        return 0
    return 2 if not result.trace.valid else 1


if __name__ == "__main__":
    sys.exit(main())
