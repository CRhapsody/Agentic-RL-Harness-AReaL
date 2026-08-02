from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jphrl.paths import require_within_configured_root


_SENSITIVE_KEY = re.compile(
    r"(?:admin|api[_-]?key|authorization|bearer|password|secret|token[_-]?key)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_json(value: Any, *, sensitive_keys: list[str], absolute_paths: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                sensitive_keys.append(str(key))
            _walk_json(
                child,
                sensitive_keys=sensitive_keys,
                absolute_paths=absolute_paths,
            )
    elif isinstance(value, list):
        for child in value:
            _walk_json(
                child,
                sensitive_keys=sensitive_keys,
                absolute_paths=absolute_paths,
            )
    elif isinstance(value, str) and value.startswith("/"):
        absolute_paths.append(value)


def verify(result_path: str | Path, audit_path: str | Path) -> dict[str, object]:
    result_source = require_within_configured_root(result_path)
    audit_destination = require_within_configured_root(audit_path)
    payload = json.loads(result_source.read_text(encoding="utf-8"))
    if result_source.stat().st_mode & 0o777 != 0o600:
        raise ValueError("G1 result permissions are broader than 0600")
    if payload.get("experiment") != "g1-joint-integrity-v1" or payload.get("passed") is not True:
        raise ValueError("G1 result is absent or did not pass")
    if payload.get("updates") != {
        "areal_policy_update": False,
        "production_harness_update": False,
        "toy_candidate_transition_only": True,
    }:
        raise ValueError("G1 result overstates its update evidence")

    batch = payload["batch"]
    if not (
        batch["synthetic_traces"] == 32
        and batch["policy_tokens"] == 64
        and batch["trainable_policy_tokens"] == 48
        and batch["harness_actions"] == 32
        and batch["trainable_harness_actions"] == 24
    ):
        raise ValueError("G1 separation batch metrics differ from the frozen contract")

    version = payload["mixed_version"]
    expected_version_metrics = {
        "synthetic_fixtures_started": 1000,
        "synthetic_fixtures_ended": 1000,
        "synthetic_fixtures_internally_valid": 1000,
        "joint_publishes": 10,
        "straddled_publish": 100,
        "mixed_version_episodes": 0,
        "half_version_observations": 0,
        "stale_accepted_at_lag_0": 0,
        "stale_discarded_at_lag_0": 100,
        "unresolved_manifest_reads": 0,
    }
    if any(version.get(key) != value for key, value in expected_version_metrics.items()):
        raise ValueError("G1 version schedule metrics differ from the frozen contract")
    if version.get("negative_control_rejected") is not True:
        raise ValueError("mixed-version negative control was not rejected")
    if version.get("fixture_kind") != "deterministic-synthetic-version-trace":
        raise ValueError("G1 version evidence is not labeled as synthetic fixtures")

    separation = payload["credit_separation"]
    if not (
        separation["policy_source"] == "synthetic-policy-credit-fixture-v1"
        and separation["harness_source"] == "synthetic-harness-credit-fixture-v1"
        and separation["update_interventions"]["passed"] is True
        and separation["negative_mutations"]["passed"] is True
        and separation["negative_mutations"]["rejected"] == 10
    ):
        raise ValueError("two-stream update intervention checks did not pass")

    cases = payload["atomic_publish"]["cases"]
    if len(cases) != 6 or not all(case.get("passed") is True for case in cases):
        raise ValueError("atomic publish failure matrix is incomplete")
    if {
        case["fault_at"] for case in cases
    } != {
        "after_policy_object",
        "after_harness_object",
        "before_active_switch",
        "after_active_switch",
        "release_gate_reject",
        None,
    }:
        raise ValueError("atomic publish failure phases differ from the frozen contract")

    checkpoint = payload["checkpoint_replay"]
    if not (
        checkpoint["continuous_next_digest"] == checkpoint["restored_next_digest"]
        and checkpoint["next_step_equal"] is True
        and checkpoint["next_action_equal"] is True
        and checkpoint["tabular_harness"]["exact_next_decision"] is True
        and checkpoint["active_release_store_validation"] is True
    ):
        raise ValueError("checkpoint restore did not reproduce the next joint step")

    work_dir = require_within_configured_root(payload["artifacts"]["work_dir"])
    files = sorted(path for path in work_dir.rglob("*") if path.is_file())
    directories = sorted(path for path in work_dir.rglob("*") if path.is_dir())
    if not files:
        raise ValueError("G1 work directory contains no persistent evidence")
    bad_file_modes = [str(path) for path in files if path.stat().st_mode & 0o777 != 0o600]
    bad_directory_modes = [
        str(path) for path in [work_dir, *directories] if path.stat().st_mode & 0o777 != 0o700
    ]
    if bad_file_modes or bad_directory_modes:
        raise ValueError("G1 artifact permissions are broader than 0600/0700")

    sensitive_keys: list[str] = []
    absolute_paths: list[str] = []
    for path in [result_source, *files]:
        if path.suffix == ".json":
            _walk_json(
                json.loads(path.read_text(encoding="utf-8")),
                sensitive_keys=sensitive_keys,
                absolute_paths=absolute_paths,
            )
    escaped_paths = []
    for value in absolute_paths:
        try:
            require_within_configured_root(value)
        except ValueError:
            escaped_paths.append(value)
    if sensitive_keys or escaped_paths:
        raise ValueError("G1 artifact contains a sensitive key or escaped absolute path")

    audit: dict[str, object] = {
        "experiment": payload["experiment"],
        "passed": True,
        "result_sha256": _sha256(result_source),
        "files": [
            {
                "path": str(path),
                "mode": oct(path.stat().st_mode & 0o777),
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "file_count": len(files),
        "directory_count": len(directories) + 1,
        "absolute_path_count": len(absolute_paths),
        "all_paths_within_configured_root": not escaped_paths,
        "sensitive_key_matches": sensitive_keys,
        "modes": {"files": "0600", "directories": "0700"},
    }
    descriptor = os.open(
        audit_destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "experiment": audit["experiment"],
                "passed": audit["passed"],
                "result_sha256": audit["result_sha256"],
                "file_count": audit["file_count"],
                "directory_count": audit["directory_count"],
                "all_paths_within_configured_root": audit[
                    "all_paths_within_configured_root"
                ],
                "sensitive_key_matches": audit["sensitive_key_matches"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the frozen G1 integrity gate")
    parser.add_argument("--result", required=True)
    parser.add_argument("--audit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify(args.result, args.audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
