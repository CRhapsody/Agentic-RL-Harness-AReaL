#!/usr/bin/env python3
"""Create and consume fail-closed control records for the 8-GPU holder."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import stat
import tempfile
import time
from pathlib import Path
from typing import Any


CONTROL_SCHEMA = "jph.areal-gpu-holder-control.v1"
STOP_SCHEMA = "jph.areal-gpu-holder-stop-request.v1"
DEFAULT_JPH_ROOT = Path("/mnt/sdb/ljw/chizm")
RUN_ID_RE = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z-gpu-holder-[0-9a-f]{16}$"
)
CONTROL_FIELDS = {
    "schema_version",
    "run_id",
    "run_root",
    "launcher_pid",
    "launcher_start_time",
    "coordinator_pid",
    "coordinator_start_time",
    "coordinator_session_id",
    "project_commit",
    "areal_commit",
    "created_unix",
}
STOP_FIELDS = {
    "schema_version",
    "run_id",
    "launcher_pid",
    "launcher_start_time",
    "reason",
    "requested_by_uid",
    "requested_unix",
}


def _require_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _process_start_time(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError(f"cannot parse /proc/{pid}/stat")
    fields = raw[closing + 2 :].split()
    if len(fields) < 20:
        raise ValueError(f"short /proc/{pid}/stat")
    return int(fields[19])


def _process_session_id(pid: int) -> int:
    return os.getsid(pid)


def _validate_process(pid: int, start_time: int, *, session_id: int | None = None) -> None:
    proc = Path(f"/proc/{pid}")
    info = proc.stat()
    if info.st_uid != os.getuid():
        raise ValueError(f"PID {pid} is not owned by the caller")
    if _process_start_time(pid) != start_time:
        raise ValueError(f"PID {pid} start time no longer matches")
    if session_id is not None and _process_session_id(pid) != session_id:
        raise ValueError(f"PID {pid} session ID no longer matches")


def _root_and_run(jph_root_raw: str, run_id: str) -> tuple[Path, Path]:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("invalid holder run ID")
    root = Path(jph_root_raw)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("JPH root must be an absolute non-symlink directory")
    root = root.resolve(strict=True)
    expected = root / "artifacts" / "areal-gpu-holder" / run_id
    return root, expected


def _read_private_json(path: Path, expected_fields: set[str]) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError(f"unsafe record: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError(f"record is not private: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError(f"unexpected record fields: {path}")
    return value


def _write_new_private_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_control(jph_root_raw: str, run_id: str) -> tuple[Path, dict[str, Any]]:
    _, expected_run_root = _root_and_run(jph_root_raw, run_id)
    run_root = expected_run_root.resolve(strict=True)
    if run_root != expected_run_root:
        raise ValueError("holder run root escapes its expected location")
    control = _read_private_json(run_root / "holder-control.json", CONTROL_FIELDS)
    if control["schema_version"] != CONTROL_SCHEMA:
        raise ValueError("unsupported holder control schema")
    if control["run_id"] != run_id or control["run_root"] != str(run_root):
        raise ValueError("holder control identity mismatch")
    launcher_pid = _require_int(control["launcher_pid"], "launcher_pid")
    launcher_start = _require_int(
        control["launcher_start_time"], "launcher_start_time"
    )
    coordinator_pid = _require_int(control["coordinator_pid"], "coordinator_pid")
    coordinator_start = _require_int(
        control["coordinator_start_time"], "coordinator_start_time"
    )
    coordinator_sid = _require_int(
        control["coordinator_session_id"], "coordinator_session_id"
    )
    if coordinator_pid != coordinator_sid:
        raise ValueError("holder coordinator must lead its own session")
    _validate_process(launcher_pid, launcher_start)
    _validate_process(coordinator_pid, coordinator_start, session_id=coordinator_sid)
    return run_root, control


def write_control(args: argparse.Namespace) -> None:
    _, expected_run_root = _root_and_run(args.jph_root, args.run_id)
    run_root = Path(args.run_root).resolve(strict=True)
    if run_root != expected_run_root:
        raise ValueError("run root does not match holder run ID")
    launcher_pid = _require_int(args.launcher_pid, "launcher_pid")
    launcher_start = _require_int(args.launcher_start_time, "launcher_start_time")
    coordinator_pid = _require_int(args.coordinator_pid, "coordinator_pid")
    coordinator_start = _require_int(
        args.coordinator_start_time, "coordinator_start_time"
    )
    coordinator_sid = _require_int(
        args.coordinator_session_id, "coordinator_session_id"
    )
    if coordinator_pid != coordinator_sid:
        raise ValueError("coordinator must lead its own session")
    _validate_process(launcher_pid, launcher_start)
    _validate_process(coordinator_pid, coordinator_start, session_id=coordinator_sid)
    for name, value in (
        ("project_commit", args.project_commit),
        ("areal_commit", args.areal_commit),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"{name} must be a lowercase 40-hex commit")
    payload = {
        "schema_version": CONTROL_SCHEMA,
        "run_id": args.run_id,
        "run_root": str(run_root),
        "launcher_pid": launcher_pid,
        "launcher_start_time": launcher_start,
        "coordinator_pid": coordinator_pid,
        "coordinator_start_time": coordinator_start,
        "coordinator_session_id": coordinator_sid,
        "project_commit": args.project_commit,
        "areal_commit": args.areal_commit,
        "created_unix": int(time.time()),
    }
    _write_new_private_json(run_root / "holder-control.json", payload)


def request_stop(args: argparse.Namespace) -> None:
    run_root, control = _load_control(args.jph_root, args.run_id)
    launcher_pid = int(control["launcher_pid"])
    launcher_start = int(control["launcher_start_time"])
    request_path = run_root / "stop.requested.json"
    payload = {
        "schema_version": STOP_SCHEMA,
        "run_id": args.run_id,
        "launcher_pid": launcher_pid,
        "launcher_start_time": launcher_start,
        "reason": args.reason,
        "requested_by_uid": os.getuid(),
        "requested_unix": int(time.time()),
    }
    created_request = False
    try:
        _write_new_private_json(request_path, payload)
        created_request = True
    except FileExistsError:
        existing = _read_private_json(request_path, STOP_FIELDS)
        stable_fields = (
            "schema_version",
            "run_id",
            "launcher_pid",
            "launcher_start_time",
        )
        if any(existing[name] != payload[name] for name in stable_fields):
            raise ValueError("existing stop request has a different holder identity")
    try:
        # Close the identity race between loading the control record and
        # signaling.  A newly-created request is removed if this final check
        # fails, so it cannot later legitimize an unrelated launcher failure.
        _validate_process(launcher_pid, launcher_start)
    except Exception:
        if created_request:
            request_path.unlink(missing_ok=True)
        raise
    os.kill(launcher_pid, signal.SIGTERM)


def watch_runtime(args: argparse.Namespace) -> None:
    runtime_seconds = _require_int(args.runtime_seconds, "runtime_seconds", minimum=300)
    deadline = time.monotonic() + runtime_seconds
    while True:
        _, control = _load_control(args.jph_root, args.run_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(30.0, remaining))
    stop_args = argparse.Namespace(
        jph_root=args.jph_root,
        run_id=args.run_id,
        reason="runtime-limit",
    )
    request_stop(stop_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write-control")
    write.add_argument("--jph-root", default=str(DEFAULT_JPH_ROOT))
    write.add_argument("--run-id", required=True)
    write.add_argument("--run-root", required=True)
    write.add_argument("--launcher-pid", required=True, type=int)
    write.add_argument("--launcher-start-time", required=True, type=int)
    write.add_argument("--coordinator-pid", required=True, type=int)
    write.add_argument("--coordinator-start-time", required=True, type=int)
    write.add_argument("--coordinator-session-id", required=True, type=int)
    write.add_argument("--project-commit", required=True)
    write.add_argument("--areal-commit", required=True)
    write.set_defaults(func=write_control)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--jph-root", default=str(DEFAULT_JPH_ROOT))
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--reason", choices=("manual", "runtime-limit"), default="manual")
    stop.set_defaults(func=request_stop)

    watch = subparsers.add_parser("watch")
    watch.add_argument("--jph-root", default=str(DEFAULT_JPH_ROOT))
    watch.add_argument("--run-id", required=True)
    watch.add_argument("--runtime-seconds", required=True, type=int)
    watch.set_defaults(func=watch_runtime)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
