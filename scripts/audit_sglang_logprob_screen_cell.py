from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from jphrl.experiments.sglang_logprob_screen import artifact_tree_sha256
from scripts.verify_areal_joint_bridge import (
    _audit_modes,
    _audit_sensitive_artifacts,
    _require_within,
)


def _record_sha256(value: dict[str, object]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
    payload = json.dumps(
        unsigned,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one SGLang log-prob screen cell before pairing"
    )
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    configured_root = Path(os.environ["JPH_ROOT"]).resolve()
    run_root = args.run_root.resolve()
    _require_within(run_root, configured_root)
    if not run_root.is_dir() or run_root.is_symlink():
        raise ValueError(f"invalid screen run root: {run_root}")
    mode_audit = _audit_modes(run_root)
    if mode_audit["mode_violations"] or mode_audit["symlinks"]:
        raise ValueError("screen artifact permissions or symlink audit failed")
    sensitive_audit = _audit_sensitive_artifacts(run_root)
    report: dict[str, object] = {
        "schema_version": "jph.sglang-logprob-screen-cell-audit.v1",
        "run_root": str(run_root),
        "audited_tree_sha256": artifact_tree_sha256(run_root),
        "mode_audit_before_report": mode_audit,
        "sensitive_artifact_audit": sensitive_audit,
        "passed": True,
    }
    report["record_sha256"] = _record_sha256(report)
    output = run_root / "cell-audit.json"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(
                report,
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
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
