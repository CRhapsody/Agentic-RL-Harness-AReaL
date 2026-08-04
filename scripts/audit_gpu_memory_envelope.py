from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from jphrl.paths import require_outside_repository

SCHEMA_VERSION = "jph.gpu-memory-observation.v2"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _parse_samples(path: Path) -> list[tuple[int, int, int]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("GPU memory sample file is missing or unsafe")
    samples: list[tuple[int, int, int]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        fields = raw_line.split(",")
        if len(fields) != 3:
            raise ValueError(f"invalid GPU memory sample at line {line_number}")
        try:
            timestamp, used_mib, free_mib = (int(field) for field in fields)
        except ValueError as exc:
            raise ValueError(
                f"non-integer GPU memory sample at line {line_number}"
            ) from exc
        if timestamp <= 0 or used_mib < 0 or free_mib < 0:
            raise ValueError(f"negative GPU memory sample at line {line_number}")
        samples.append((timestamp, used_mib, free_mib))
    if not samples:
        raise ValueError("GPU memory sample file is empty")
    if any(
        samples[index][0] < samples[index - 1][0] for index in range(1, len(samples))
    ):
        raise ValueError("GPU memory sample timestamps are not monotonic")
    return samples


def audit_gpu_memory_envelope(
    *,
    samples_path: str | Path,
    output_path: str | Path,
    physical_gpu_id: int,
    baseline_used_mib: int,
    max_new_memory_mib: int | None = None,
    run_kind: str,
    project_commit: str,
) -> dict[str, object]:
    if physical_gpu_id < 0:
        raise ValueError("physical GPU ID must be non-negative")
    if baseline_used_mib < 0:
        raise ValueError("GPU memory baseline must be non-negative")
    if max_new_memory_mib is not None and max_new_memory_mib <= 0:
        raise ValueError("optional GPU memory limit must be positive")
    if not run_kind:
        raise ValueError("run kind is missing")
    if len(project_commit) != 40 or any(
        character not in "0123456789abcdef" for character in project_commit
    ):
        raise ValueError("project commit is not a full lowercase Git SHA-1")
    source = require_outside_repository(samples_path)
    destination = require_outside_repository(output_path)
    if destination.exists():
        raise ValueError("GPU memory audit output already exists")
    if source.parent != destination.parent:
        raise ValueError("GPU memory audit must remain beside its samples")
    samples = _parse_samples(source)
    peak_timestamp, peak_used_mib, peak_free_mib = max(
        samples,
        key=lambda item: item[1],
    )
    measured_new_mib = max(0, peak_used_mib - baseline_used_mib)
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "physical_gpu_id": physical_gpu_id,
        "run_kind": run_kind,
        "project_commit": project_commit,
        "sample_count": len(samples),
        "first_timestamp_unix": samples[0][0],
        "last_timestamp_unix": samples[-1][0],
        "baseline_used_mib": baseline_used_mib,
        "peak": {
            "timestamp_unix": peak_timestamp,
            "memory_used_mib": peak_used_mib,
            "memory_free_mib": peak_free_mib,
        },
        "measured_new_memory_mib": measured_new_mib,
        "max_new_memory_mib": max_new_memory_mib,
        "memory_limit_enforced": max_new_memory_mib is not None,
        "passed": (
            True
            if max_new_memory_mib is None
            else measured_new_mib <= max_new_memory_mib
        ),
        "evidence_scope": {
            "gpu_memory_observed": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
    }
    record["record_sha256"] = hashlib.sha256(_canonical_json(record)).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_json(record))
        stream.flush()
        os.fsync(stream.fileno())
    if record["passed"] is not True:
        raise RuntimeError(
            "GPU memory envelope exceeded: "
            f"measured_new={measured_new_mib}MiB "
            f"limit={max_new_memory_mib}MiB"
        )
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-gpu-id", required=True, type=int)
    parser.add_argument("--baseline-used-mib", required=True, type=int)
    parser.add_argument("--max-new-memory-mib", type=int)
    parser.add_argument("--run-kind", required=True)
    parser.add_argument("--project-commit", required=True)
    args = parser.parse_args(argv)
    record = audit_gpu_memory_envelope(
        samples_path=args.samples,
        output_path=args.output,
        physical_gpu_id=args.physical_gpu_id,
        baseline_used_mib=args.baseline_used_mib,
        max_new_memory_mib=args.max_new_memory_mib,
        run_kind=args.run_kind,
        project_commit=args.project_commit,
    )
    print(json.dumps(record, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
