from __future__ import annotations

"""Formal eight-GPU M0 entry with the registered production AReaL adapter."""

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from jphrl.experiments.m0_eight_gpu_integrated import (
    EightGPUIntegratedStageMachine,
    freeze_integrated_launch_preflight,
    prepare_eight_gpu_admission_selection,
)
from jphrl.experiments.m0_eight_gpu_real_adapter import (
    RealEightGPUAdapterConfig,
    RealEightGPUIntegratedAdapters,
    ThreadedEightGPUMemoryRuntime,
)
from jphrl.paths import assert_remote_environment


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry-mode",
        choices=("execute", "freeze-existing-admissions"),
        default="execute",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-project-commit", required=True)
    parser.add_argument("--runner-admission-dir")
    parser.add_argument("--model-report")
    parser.add_argument("--dataset-report")
    parser.add_argument("--transaction-id")
    parser.add_argument("--trial-name")
    args = parser.parse_args(argv)

    assert_remote_environment()
    jph_root = Path(os.environ["JPH_ROOT"])
    project = Path(os.environ["JPH_PROJECT_DIR"])
    areal = Path(os.environ["JPH_AREAL_ROOT"])
    preflight = freeze_integrated_launch_preflight(
        jph_root=jph_root,
        project_repository=project,
        areal_repository=areal,
        run_root=args.run_root,
        expected_project_commit=args.expected_project_commit,
        tmux_connection=os.environ.get("TMUX", ""),
    )
    if args.entry_mode == "execute":
        model_report = Path(args.model_report) if args.model_report else (
            jph_root / "artifacts" / "bootstrap" / "qwen2.5-1.5b-snapshot.json"
        )
        dataset_report = Path(args.dataset_report) if args.dataset_report else (
            jph_root / "artifacts" / "bootstrap" / "gsm8k-snapshot.json"
        )
        transaction_id = args.transaction_id or preflight.run_root.name
        trial_name = args.trial_name or transaction_id
        config = RealEightGPUAdapterConfig.from_snapshot_reports(
            jph_root=jph_root,
            run_root=preflight.run_root,
            project_commit=preflight.project_commit,
            model_report=model_report,
            dataset_report=dataset_report,
            admin_api_key=os.environ.get("JPH_AREAL_ADMIN_API_KEY", ""),
            transaction_id=transaction_id,
            trial_name=trial_name,
        )
        stage_machine = EightGPUIntegratedStageMachine(
            run_root=preflight.run_root,
            exact_session_id=os.getsid(0),
            execution_mode="real-gpu",
        )
        result = stage_machine.run(
            adapters=RealEightGPUIntegratedAdapters(config),
            memory=ThreadedEightGPUMemoryRuntime(
                audit_root=preflight.run_root / "gpu-memory-runtime"
            ),
        )
        print(
            json.dumps(
                {
                    "mode": "execute",
                    "gpu_execution": True,
                    "preflight_record": str(preflight.record_path),
                    "stage_machine_record": str(result.state_record_path),
                    "selection_record": str(result.selection.selection_record_path),
                    "tuvw_record": str(result.tuvw.record_path),
                    "x_record": str(result.x.record_path),
                    "y_record": str(result.y.record_path),
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return
    if not args.runner_admission_dir:
        parser.error("--runner-admission-dir is required for freeze-existing-admissions")
    selection = prepare_eight_gpu_admission_selection(
        runner_admission_dir=args.runner_admission_dir,
        selection_root=preflight.run_root / "admission-selection",
    )
    result = {
        "mode": "freeze-existing-admissions",
        "gpu_execution": False,
        "preflight_record": str(preflight.record_path),
        "selection_record": str(selection.selection_record_path),
        "multi_s_batch": str(selection.multi_s_batch_path),
    }
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
