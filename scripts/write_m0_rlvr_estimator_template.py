#!/usr/bin/env python3

from __future__ import annotations

"""Write the immutable preregistered M0 dual-credit baseline template."""

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from jphrl.paths import require_outside_repository, require_within_configured_root
from jphrl.trajectory.rlvr_workflow_admission import (
    build_frozen_dual_credit_estimator_template,
    frozen_dual_credit_estimator_template_from_record,
)


def write_m0_rlvr_estimator_template(
    *,
    output_path: str | Path,
    allowed_root: str | Path,
) -> dict[str, object]:
    root = Path(allowed_root).expanduser().resolve(strict=True)
    destination = require_outside_repository(output_path)
    require_within_configured_root(destination)
    if destination == root or root not in destination.parents:
        raise ValueError("M0 estimator template escapes its allowed root")
    if destination.exists():
        raise ValueError("M0 estimator template output already exists")
    record = build_frozen_dual_credit_estimator_template(
        policy_source="m0-preregistered-constant-policy-return-baseline-v1",
        harness_source="m0-preregistered-constant-harness-return-baseline-v1",
        policy_baseline_snapshot_id="m0-policy-half-return-baseline-20260803",
        harness_baseline_snapshot_id="m0-harness-half-return-baseline-20260803",
        policy_baseline=0.5,
        harness_baseline=0.5,
    )
    frozen_dual_credit_estimator_template_from_record(record)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                record,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(destination, 0o600)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-root", required=True)
    args = parser.parse_args(argv)
    record = write_m0_rlvr_estimator_template(
        output_path=args.output,
        allowed_root=args.allowed_root,
    )
    print(json.dumps(record, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
