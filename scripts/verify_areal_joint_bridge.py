from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
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
SAME_BACKEND_SCHEMA_VERSION = "jph.areal-same-backend-logprob.v3"
MAX_SAME_BACKEND_MEAN_IMPORTANCE_RATIO_ERROR = 0.02
MAX_SAME_BACKEND_IMPORTANCE_RATIO_ERROR = 0.10
_SENSITIVE_FIELD_PATTERN = "|".join(re.escape(key) for key in SENSITIVE_KEY_PARTS)
_SENSITIVE_VALUE_PATTERN = r'''(\$\{[^}\r\n]+\}|"[^"]*"|'[^']*'|[^,}\s]+)'''
_SENSITIVE_ASSIGNMENT = re.compile(
    rf'''(?ix)
    (?<![A-Za-z0-9_])
    ["']?(?:[A-Za-z0-9_-]+\.)*({_SENSITIVE_FIELD_PATTERN})["']?
    \s*(?::|=)\s*{_SENSITIVE_VALUE_PATTERN}
    '''
)
_SENSITIVE_CLI_ARGUMENT = re.compile(
    rf'''(?ix)
    (?<!\S)--({_SENSITIVE_FIELD_PATTERN})
    (?:\s+|=){_SENSITIVE_VALUE_PATTERN}
    '''
)
_SAFE_LITERAL_VALUES = {"", "null", "none"}
_ENV_REFERENCE = re.compile(r"^\$\{(?:oc\.env:)?[A-Za-z_][A-Za-z0-9_]*\}$")
_REDACTED_VALUE = re.compile(r"^<redacted(?:-[a-z0-9]+)*>$", re.IGNORECASE)
_TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}


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


def _score_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _trajectory_binding_sha256(value: Mapping[str, object]) -> str:
    payload = {
        key: value[key]
        for key in (
            "input_ids",
            "loss_mask",
            "logprobs",
            "versions",
            "attention_mask",
            "rewards",
        )
    }
    return _score_sha256(payload)


def _single_row(value: object, name: str) -> list[Any]:
    _require(
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list),
        f"{name} must have batch size one",
    )
    return value[0]


def _verify_same_backend_scores(
    score_dir: Path,
    records: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, object]]:
    _require(score_dir.is_dir(), f"same-backend score directory is missing: {score_dir}")
    paths = sorted(score_dir.glob("same-backend-score-*.json"))
    _require(len(paths) == len(records), "same-backend score count mismatch")
    scores_by_request: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        score = json.loads(path.read_text(encoding="utf-8"))
        _require(
            score.get("schema_version") == SAME_BACKEND_SCHEMA_VERSION,
            f"{path}: unknown score schema",
        )
        _require(
            score.get("record_sha256") == _score_sha256(score),
            f"{path}: score hash mismatch",
        )
        request_id = score.get("request_id")
        _require(
            isinstance(request_id, str) and bool(request_id),
            f"{path}: score request ID is missing",
        )
        _require(
            request_id not in scores_by_request,
            "duplicate same-backend score request ID",
        )
        scores_by_request[request_id] = (path, score)

    reports: list[dict[str, object]] = []
    consumed_paths: set[Path] = set()
    for bridge_path, record in records:
        tensors = record["areal_trace"]["tensor_dict"]
        expected_ids = tensors["input_ids"][0]
        request_id = record["request_id"]
        entry = scores_by_request.get(request_id)
        _require(entry is not None, f"{bridge_path}: no matching same-backend score")
        score_path, score = entry
        consumed_paths.add(score_path)
        _require(
            score.get("bridge_record_sha256") == record["record_sha256"],
            f"{score_path}: bridge record hash mismatch",
        )
        _require(
            score.get("trajectory_binding_sha256")
            == _trajectory_binding_sha256(tensors),
            f"{score_path}: trajectory binding hash mismatch",
        )
        scoring_origin = score.get("scoring_origin")
        _require(isinstance(scoring_origin, dict), f"{score_path}: scoring origin missing")
        expected_origin = {
            "api": "RolloutController.compute_logp",
            "controller_api_version": "v1",
            "lifecycle": "same-controller-after-wait-before-destroy",
            "score_parser": "jph-tail-before-conversion-v1",
            "backend": "sglang:d1p1t1",
            "engine_version_before_score": record["policy_binding"][
                "expected_inference_engine_version"
            ],
            "engine_version_after_score": record["policy_binding"][
                "expected_inference_engine_version"
            ],
            "policy_release_id": record["joint_version"]["policy"],
            "behavior_revision": record["areal_trace"]["origin"][
                "behavior_revision"
            ],
            "areal_commit": record["areal_trace"]["origin"]["areal_commit"],
            "project_commit": record["origin"]["project_commit"],
        }
        _require(
            scoring_origin == expected_origin,
            f"{score_path}: scoring origin differs from bridge identity",
        )
        input_ids = _single_row(score["input_ids"], "score.input_ids")
        _require(input_ids == expected_ids, f"{score_path}: input IDs mismatch")
        loss_mask = _single_row(score["loss_mask"], "score.loss_mask")
        stored = _single_row(score["stored_logprobs"], "score.stored_logprobs")
        rescored = _single_row(score["rescored_logprobs"], "score.rescored_logprobs")
        versions = _single_row(score["versions"], "score.versions")
        length = len(expected_ids)
        _require(
            len(loss_mask) == len(stored) == len(rescored) == len(versions) == length,
            f"{score_path}: score vector lengths differ",
        )
        _require(loss_mask == tensors["loss_mask"][0], f"{score_path}: loss mask mismatch")
        _require(versions == tensors["versions"][0], f"{score_path}: versions mismatch")
        _require(
            all(
                math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
                for left, right in zip(stored, tensors["logprobs"][0])
            ),
            f"{score_path}: stored logprobs differ from bridge tensor",
        )
        active = [index for index, mask in enumerate(loss_mask) if mask == 1]
        _require(bool(active), f"{score_path}: no trainable policy tokens")
        logprob_errors = [
            abs(float(rescored[index]) - float(stored[index])) for index in active
        ]
        ratio_errors = [
            abs(math.exp(float(rescored[index]) - float(stored[index])) - 1.0)
            for index in active
        ]
        mean_ratio_error = sum(ratio_errors) / len(ratio_errors)
        max_ratio_error = max(ratio_errors)
        passed = (
            mean_ratio_error <= MAX_SAME_BACKEND_MEAN_IMPORTANCE_RATIO_ERROR
            and max_ratio_error <= MAX_SAME_BACKEND_IMPORTANCE_RATIO_ERROR
        )
        reports.append(
            {
                "bridge_path": str(bridge_path),
                "score_path": str(score_path),
                "checked_tokens": len(active),
                "mean_abs_logprob_error": sum(logprob_errors) / len(logprob_errors),
                "max_abs_logprob_error": max(logprob_errors),
                "mean_importance_ratio_error": mean_ratio_error,
                "max_importance_ratio_error": max_ratio_error,
                "thresholds": {
                    "max_mean_importance_ratio_error": (
                        MAX_SAME_BACKEND_MEAN_IMPORTANCE_RATIO_ERROR
                    ),
                    "max_importance_ratio_error": (
                        MAX_SAME_BACKEND_IMPORTANCE_RATIO_ERROR
                    ),
                },
                "passed": passed,
            }
        )
    _require(
        len(reports) == len(records) and all(report["passed"] for report in reports),
        "same-backend policy logprob audit failed",
    )
    _require(
        consumed_paths == set(paths),
        "not every same-backend score file was consumed exactly once",
    )
    return reports


def _audit_sensitive_artifacts(run_root: Path) -> dict[str, object]:
    runtime_secret = os.environ.pop("JPH_AREAL_ADMIN_API_KEY", None)
    unsafe_fields: list[dict[str, object]] = []
    redacted_fields = 0
    scanned_files = 0
    scanned_text_files = 0
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        scanned_files += 1
        data = path.read_bytes()
        _require(
            not runtime_secret or runtime_secret.encode("utf-8") not in data,
            f"runtime admin key remains in artifact: {path}",
        )
        _require(
            b"areal-admin-key" not in data,
            f"default AReaL admin key remains in artifact: {path}",
        )
        text_like = path.suffix.lower() in _TEXT_SUFFIXES or b"\x00" not in data[:8192]
        if not text_like:
            continue
        scanned_text_files += 1
        text = data.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            matched_fields = {
                (item.group(1).lower(), item.group(2))
                for pattern in (_SENSITIVE_ASSIGNMENT, _SENSITIVE_CLI_ARGUMENT)
                for item in pattern.finditer(line)
            }
            fields = sorted(matched_fields)
            for key, raw_value in fields:
                normalized = raw_value.strip().removesuffix(",").strip()
                if (
                    len(normalized) >= 2
                    and normalized[0] == normalized[-1]
                    and normalized[0] in {"'", '"'}
                ):
                    normalized = normalized[1:-1].strip()
                safe = (
                    normalized.lower() in _SAFE_LITERAL_VALUES
                    or _REDACTED_VALUE.fullmatch(normalized) is not None
                    or _ENV_REFERENCE.fullmatch(normalized) is not None
                )
                if safe:
                    redacted_fields += 1
                else:
                    unsafe_fields.append(
                        {"path": str(path), "line": line_number, "key": key.lower()}
                    )
    _require(not unsafe_fields, "unredacted sensitive fields remain in artifacts")
    return {
        "scanned_files": scanned_files,
        "scanned_text_files": scanned_text_files,
        "redacted_or_safe_sensitive_fields": redacted_fields,
        "unsafe_fields": unsafe_fields,
        "runtime_secret_matches": 0,
        "default_key_matches": 0,
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
    same_backend_score_dir: Path,
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
    for path in (
        bridge_dir,
        same_backend_score_dir,
        model_report_path,
        dataset_report_path,
        output,
    ):
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
    same_backend_audits = _verify_same_backend_scores(
        same_backend_score_dir,
        raw_records,
    )
    cross_backend_audits = recompute_behavior_logprobs(
        nested_records,
        snapshot_path=snapshot_path,
        device=device,
        max_tokens_per_trace=max_tokens_per_trace,
        max_traces=expected_count,
        max_abs_error=max_abs_error,
        max_mean_abs_error=max_mean_abs_error,
        require_all_passed=False,
    )
    prompt_token_audits = _recompute_prompt_tokens(
        raw_records,
        tokenizer_path=snapshot_path,
    )
    mode_audit = _audit_modes(bridge_dir.parent)
    _require(not mode_audit["mode_violations"], "artifact permissions are not private")
    _require(not mode_audit["symlinks"], "artifact tree contains symlinks")
    sensitive_artifact_audit = _audit_sensitive_artifacts(bridge_dir.parent)

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
            "same_backend_policy_logprob_recompute": True,
            "frozen_hf_cross_backend_diagnostic": True,
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
        "same_backend_policy_logprob_recompute": same_backend_audits,
        "frozen_hf_cross_backend_diagnostic": cross_backend_audits,
        "prompt_token_recompute": prompt_token_audits,
        "all_paths_within_configured_root": True,
        "absolute_paths_checked": sorted(set(absolute_paths)),
        "sensitive_key_matches": sensitive_keys,
        "sensitive_artifact_audit": sensitive_artifact_audit,
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
    parser.add_argument("--same-backend-score-dir", required=True, type=Path)
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
        same_backend_score_dir=args.same_backend_score_dir,
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
