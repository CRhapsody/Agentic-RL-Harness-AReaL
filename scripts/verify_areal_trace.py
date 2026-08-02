from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from jphrl.trajectory.areal_trace_contract import validate_areal_trace_record


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_within(path: Path, root: Path) -> None:
    common = Path(os.path.commonpath((path.resolve(), root.resolve())))
    _require(common == root.resolve(), f"path escapes configured root: {path}")


def _trace_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    _require(path.is_dir(), f"trace path does not exist: {path}")
    paths = sorted(path.glob("trace-*.json"))
    _require(bool(paths), f"no trace JSON files found in {path}")
    return paths


def _evenly_spaced(indices: list[int], limit: int) -> list[int]:
    if len(indices) <= limit:
        return indices
    if limit == 1:
        return [indices[0]]
    selected = {
        indices[round(offset * (len(indices) - 1) / (limit - 1))]
        for offset in range(limit)
    }
    return sorted(selected)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _load_contract_records(
    trace_path: Path,
    *,
    expected_policy_version: int,
    expected_areal_commit: str,
    snapshot_path: Path,
    behavior_revision: str,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    records = []
    audits = []
    for path in _trace_paths(trace_path):
        record = json.loads(path.read_text(encoding="utf-8"))
        audit = validate_areal_trace_record(
            record, expected_policy_version=expected_policy_version
        )
        origin = record["origin"]
        _require(
            origin["areal_commit"] == expected_areal_commit,
            f"{path}: AReaL commit mismatch",
        )
        _require(
            Path(origin["behavior_snapshot_path"]).resolve() == snapshot_path,
            f"{path}: behavior snapshot path mismatch",
        )
        _require(
            origin["behavior_revision"] == behavior_revision,
            f"{path}: behavior revision mismatch",
        )
        records.append((path, record))
        audits.append({"trace": str(path), **audit})
    return records, audits


def recompute_behavior_logprobs(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    snapshot_path: Path,
    device: str,
    max_tokens_per_trace: int,
    max_traces: int,
    max_abs_error: float,
    max_mean_abs_error: float,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM

    _require(torch.cuda.is_available(), "CUDA is required for behavior logprob recomputation")
    _require(device.startswith("cuda"), f"recompute device must be CUDA: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    reports: list[dict[str, Any]] = []
    try:
        for path, record in records[:max_traces]:
            tensors = record["tensor_dict"]
            input_ids = tensors["input_ids"][0]
            loss_mask = tensors["loss_mask"][0]
            stored = tensors["logprobs"][0]
            action_positions = [
                index for index, active in enumerate(loss_mask) if active == 1
            ]
            _require(
                action_positions and action_positions[0] > 0,
                f"{path}: action tokens need a preceding causal context token",
            )
            selected = _evenly_spaced(action_positions, max_tokens_per_trace)
            token_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=token_tensor, use_cache=False).logits[0]
                predictor_positions = torch.tensor(
                    [position - 1 for position in selected],
                    dtype=torch.long,
                    device=device,
                )
                targets = torch.tensor(
                    [input_ids[position] for position in selected],
                    dtype=torch.long,
                    device=device,
                )
                selected_logits = logits.index_select(0, predictor_positions).float()
                recomputed = (
                    selected_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                    - torch.logsumexp(selected_logits, dim=1)
                ).cpu()
            recomputed_values = [float(value) for value in recomputed.tolist()]
            stored_values = [float(stored[position]) for position in selected]
            errors = [
                abs(actual - expected)
                for actual, expected in zip(recomputed_values, stored_values)
            ]
            mean_abs_error = sum(errors) / len(errors)
            observed_max_abs_error = max(errors)
            p95_abs_error = _percentile(errors, 0.95)
            importance_ratio_errors = [
                abs(math.exp(actual - expected) - 1.0)
                for actual, expected in zip(recomputed_values, stored_values)
            ]
            passed = (
                observed_max_abs_error <= max_abs_error
                and mean_abs_error <= max_mean_abs_error
            )
            reports.append(
                {
                    "trace": str(path),
                    "checked_tokens": len(selected),
                    "selected_action_positions": selected,
                    "mean_abs_error": mean_abs_error,
                    "p95_abs_error": p95_abs_error,
                    "max_abs_error": observed_max_abs_error,
                    "max_importance_ratio_error": max(importance_ratio_errors),
                    "tolerance": {
                        "max_abs_error": max_abs_error,
                        "max_mean_abs_error": max_mean_abs_error,
                    },
                    "passed": passed,
                }
            )
            del token_tensor, logits, selected_logits, recomputed
            torch.cuda.empty_cache()
    finally:
        del model
        torch.cuda.empty_cache()
    _require(
        reports and all(report["passed"] for report in reports),
        "logprob recomputation failed tolerance",
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an AReaL ModelResponse interaction roundtrip and behavior logprobs"
    )
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--model-report", required=True, type=Path)
    parser.add_argument("--expected-areal-commit", required=True)
    parser.add_argument("--expected-policy-version", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens-per-trace", type=int, default=64)
    parser.add_argument("--max-traces", type=int, default=1)
    parser.add_argument("--max-abs-error", type=float, default=0.25)
    parser.add_argument("--max-mean-abs-error", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _require(args.max_tokens_per_trace > 0, "max tokens must be positive")
    _require(args.max_traces > 0, "max traces must be positive")
    root = Path(os.environ["JPH_ROOT"]).resolve()
    _require(root.is_dir(), f"JPH_ROOT does not exist: {root}")
    _require_within(args.trace_path, root)
    _require_within(args.model_report, root)
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    snapshot_path = Path(model_report["snapshot_path"]).resolve()
    behavior_revision = model_report["resolved_commit"]
    _require_within(snapshot_path, root)
    _require(snapshot_path.is_dir(), f"model snapshot does not exist: {snapshot_path}")

    records, contract_audits = _load_contract_records(
        args.trace_path,
        expected_policy_version=args.expected_policy_version,
        expected_areal_commit=args.expected_areal_commit,
        snapshot_path=snapshot_path,
        behavior_revision=behavior_revision,
    )
    recompute_audits = recompute_behavior_logprobs(
        records,
        snapshot_path=snapshot_path,
        device=args.device,
        max_tokens_per_trace=args.max_tokens_per_trace,
        max_traces=args.max_traces,
        max_abs_error=args.max_abs_error,
        max_mean_abs_error=args.max_mean_abs_error,
    )
    report = {
        "ok": True,
        "evidence_scope": {
            "areal_generation": True,
            "model_response_interaction_roundtrip": True,
            "frozen_behavior_snapshot_recompute": True,
            "policy_update": False,
            "harness_update": False,
        },
        "areal_commit": args.expected_areal_commit,
        "behavior_snapshot_path": str(snapshot_path),
        "behavior_revision": behavior_revision,
        "contract_audits": contract_audits,
        "recompute_audits": recompute_audits,
    }
    rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output is not None:
        _require_within(args.output, root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        os.chmod(args.output, 0o600)
    print(rendered)


if __name__ == "__main__":
    main()
