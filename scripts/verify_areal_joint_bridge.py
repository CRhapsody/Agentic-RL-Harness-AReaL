from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from jphrl.trajectory.areal_joint_bridge import validate_areal_joint_bridge_record
from scripts.verify_areal_trace import recompute_behavior_logprobs


SENSITIVE_KEY_PARTS = (
    "admin_api_key",
    "authorization",
    "bearer",
    "github_token",
    "refresh_token",
    "secret_key",
    "session_api_key",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_within(path: Path, root: Path) -> None:
    common = Path(os.path.commonpath((path.resolve(), root.resolve())))
    _require(common == root.resolve(), f"path escapes configured root: {path}")


def _sensitive_key_paths(value: object, prefix: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            path = f"{prefix}.{key}"
            if any(part in key_text for part in SENSITIVE_KEY_PARTS):
                matches.append(path)
            matches.extend(_sensitive_key_paths(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            matches.extend(_sensitive_key_paths(nested, f"{prefix}[{index}]"))
    return matches


def _audit_modes(run_root: Path) -> dict[str, object]:
    file_count = 0
    directory_count = 0
    violations: list[dict[str, object]] = []
    symlinks: list[str] = []
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            symlinks.append(str(path))
            continue
        mode = path.stat().st_mode & 0o777
        if path.is_dir():
            directory_count += 1
            if mode != 0o700:
                violations.append({"path": str(path), "mode": oct(mode)})
        elif path.is_file():
            file_count += 1
            if mode != 0o600:
                violations.append({"path": str(path), "mode": oct(mode)})
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "mode_violations": violations,
        "symlinks": symlinks,
    }


def _recompute_prompt_tokens(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    tokenizer_path: Path,
) -> list[dict[str, object]]:
    from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer

    tokenizer = load_hf_tokenizer(str(tokenizer_path))
    audits: list[dict[str, object]] = []
    for path, record in records:
        prompt = record["prompt_binding"]
        base_ids = apply_chat_template(
            tokenizer,
            prompt["base_messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        effective_ids = apply_chat_template(
            tokenizer,
            prompt["effective_messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        _require(base_ids == prompt["base_input_tokens"], f"{path}: base prompt token mismatch")
        _require(
            effective_ids == prompt["effective_input_tokens"],
            f"{path}: effective prompt token mismatch",
        )
        _require(base_ids != effective_ids, f"{path}: Harness prompt has no token effect")
        _require(
            effective_ids == record["areal_trace"]["model_response"]["input_tokens"],
            f"{path}: AReaL did not consume independently recomputed effective tokens",
        )
        audits.append(
            {
                "path": str(path),
                "base_prompt_tokens": len(base_ids),
                "effective_prompt_tokens": len(effective_ids),
                "added_prompt_tokens": len(effective_ids) - len(base_ids),
                "passed": True,
            }
        )
    return audits


def verify(
    bridge_dir: Path,
    *,
    model_report_path: Path,
    dataset_report_path: Path,
    expected_areal_commit: str,
    expected_project_commit: str,
    expected_policy_version: int,
    expected_count: int,
    device: str,
    max_tokens_per_trace: int,
    max_abs_error: float,
    max_mean_abs_error: float,
    output: Path,
) -> dict[str, object]:
    root = Path(os.environ["JPH_ROOT"]).resolve()
    _require(root.is_dir(), f"JPH_ROOT does not exist: {root}")
    _require(expected_count == 4, "v1 bridge contract requires exactly four records")
    for path in (bridge_dir, model_report_path, dataset_report_path, output):
        _require_within(path, root)
    _require(bridge_dir.is_dir(), f"bridge directory does not exist: {bridge_dir}")
    paths = sorted(bridge_dir.glob("bridge-*.json"))
    _require(len(paths) == expected_count, f"expected {expected_count} bridge records")

    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    snapshot_path = Path(model_report["snapshot_path"]).resolve()
    behavior_revision = str(model_report["resolved_commit"])
    _require_within(snapshot_path, root)
    _require(snapshot_path.is_dir(), f"model snapshot does not exist: {snapshot_path}")
    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    dataset_snapshot_path = Path(dataset_report["snapshot_path"]).resolve()
    dataset_revision = str(dataset_report["resolved_commit"])
    _require_within(dataset_snapshot_path, root)
    _require(
        dataset_snapshot_path.is_dir(),
        f"dataset snapshot does not exist: {dataset_snapshot_path}",
    )

    audits: list[dict[str, object]] = []
    nested_records: list[tuple[Path, dict[str, Any]]] = []
    joint_version_ids: set[str] = set()
    request_ids: set[str] = set()
    decision_ids: set[str] = set()
    sensitive_keys: list[str] = []
    absolute_paths: list[str] = []
    raw_records: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        audit = validate_areal_joint_bridge_record(
            record,
            expected_policy_version=expected_policy_version,
        )
        _require(
            audit["project_commit"] == expected_project_commit,
            f"{path}: project commit mismatch",
        )
        _require(
            record["joint_version"]["environment"]
            == f"gsm8k-test@{dataset_revision}",
            f"{path}: dataset revision differs from JointVersion environment",
        )
        origin = record["areal_trace"]["origin"]
        _require(
            origin["areal_commit"] == expected_areal_commit,
            f"{path}: AReaL commit mismatch",
        )
        _require(
            Path(origin["behavior_snapshot_path"]).resolve() == snapshot_path,
            f"{path}: behavior snapshot mismatch",
        )
        _require(
            origin["behavior_revision"] == behavior_revision,
            f"{path}: behavior revision mismatch",
        )
        joint_version_ids.add(str(audit["joint_version_id"]))
        _require(audit["request_id"] not in request_ids, "duplicate request ID")
        request_ids.add(str(audit["request_id"]))
        decision_id = record["harness"]["decision"]["decision_id"]
        _require(decision_id not in decision_ids, "duplicate Harness decision ID")
        decision_ids.add(decision_id)
        sensitive_keys.extend(_sensitive_key_paths(record, prefix=str(path)))
        for candidate in (
            origin["behavior_snapshot_path"],
            str(path),
        ):
            if os.path.isabs(candidate):
                _require_within(Path(candidate), root)
                absolute_paths.append(candidate)
        audits.append({"path": str(path), **audit})
        nested_records.append((path, record["areal_trace"]))
        raw_records.append((path, record))

    _require(len(joint_version_ids) == 1, "bridge records use multiple JointVersions")
    _require(not sensitive_keys, "sensitive-looking keys found in bridge records")
    recompute_audits = recompute_behavior_logprobs(
        nested_records,
        snapshot_path=snapshot_path,
        device=device,
        max_tokens_per_trace=max_tokens_per_trace,
        max_traces=expected_count,
        max_abs_error=max_abs_error,
        max_mean_abs_error=max_mean_abs_error,
    )
    prompt_token_audits = _recompute_prompt_tokens(
        raw_records,
        tokenizer_path=snapshot_path,
    )
    mode_audit = _audit_modes(bridge_dir.parent)
    _require(not mode_audit["mode_violations"], "artifact permissions are not private")
    _require(not mode_audit["symlinks"], "artifact tree contains symlinks")

    report: dict[str, object] = {
        "ok": True,
        "experiment": "jph-areal-joint-interaction-bridge-v1",
        "claim_boundary": (
            "Four real AReaL rollout interactions with prompt-effective Harness "
            "decision sidecars under one pinned JointVersion. No policy or Harness "
            "optimizer update is performed or claimed."
        ),
        "evidence_scope": {
            "real_areal_rollout": True,
            "harness_decision_changed_prompt": True,
            "joint_interaction_sidecar": True,
            "frozen_behavior_snapshot_recompute": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
            "joint_learning_claim": False,
        },
        "bridge_record_count": len(audits),
        "project_commit": expected_project_commit,
        "areal_commit": expected_areal_commit,
        "dataset_revision": dataset_revision,
        "dataset_snapshot_path": str(dataset_snapshot_path),
        "joint_version_ids": sorted(joint_version_ids),
        "unique_request_ids": len(request_ids),
        "unique_harness_decision_ids": len(decision_ids),
        "bridge_audits": audits,
        "behavior_logprob_recompute": recompute_audits,
        "prompt_token_recompute": prompt_token_audits,
        "all_paths_within_configured_root": True,
        "absolute_paths_checked": sorted(set(absolute_paths)),
        "sensitive_key_matches": sensitive_keys,
        "mode_audit": mode_audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(output, 0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify real AReaL rollout and Harness decision bridge records"
    )
    parser.add_argument("bridge_dir", type=Path)
    parser.add_argument("--model-report", required=True, type=Path)
    parser.add_argument("--dataset-report", required=True, type=Path)
    parser.add_argument("--expected-areal-commit", required=True)
    parser.add_argument("--expected-project-commit", required=True)
    parser.add_argument("--expected-policy-version", type=int, default=0)
    parser.add_argument("--expected-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens-per-trace", type=int, default=64)
    parser.add_argument("--max-abs-error", type=float, default=0.25)
    parser.add_argument("--max-mean-abs-error", type=float, default=0.05)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(
        args.bridge_dir,
        model_report_path=args.model_report,
        dataset_report_path=args.dataset_report,
        expected_areal_commit=args.expected_areal_commit,
        expected_project_commit=args.expected_project_commit,
        expected_policy_version=args.expected_policy_version,
        expected_count=args.expected_count,
        device=args.device,
        max_tokens_per_trace=args.max_tokens_per_trace,
        max_abs_error=args.max_abs_error,
        max_mean_abs_error=args.max_mean_abs_error,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
