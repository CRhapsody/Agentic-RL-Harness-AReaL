#!/usr/bin/env python3
"""Remove the ephemeral AReaL admin key from persisted text artifacts."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

SECRET_ENV = "JPH_AREAL_ADMIN_API_KEY"
REPLACEMENT = b"<redacted-runtime-admin-key>"
DEFAULT_AREAL_ADMIN_KEY = b"areal-admin-key"
DEFAULT_REPLACEMENT = b"<redacted-default-admin-key>"
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"refusing path outside JPH_ROOT: {resolved}")
    return resolved


def _candidates(target: Path):
    if target.is_file():
        yield target
    elif target.is_dir():
        yield from target.rglob("*")


def _redact(targets: list[Path], secret: bytes) -> int:
    changed = 0
    for target in targets:
        for candidate in _candidates(target):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            data = candidate.read_bytes()
            if (
                candidate.suffix.lower() not in TEXT_SUFFIXES
                and b"\x00" in data[:8192]
            ):
                continue
            updated = data.replace(secret, REPLACEMENT).replace(
                DEFAULT_AREAL_ADMIN_KEY,
                DEFAULT_REPLACEMENT,
            )
            if updated == data:
                continue
            mode = candidate.stat().st_mode
            candidate.write_bytes(updated)
            candidate.chmod(mode)
            changed += 1
    return changed


def _verify_absent(targets: list[Path], secret: bytes) -> None:
    matches: list[str] = []
    for target in targets:
        for candidate in _candidates(target):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            data = candidate.read_bytes()
            if (
                candidate.suffix.lower() not in TEXT_SUFFIXES
                and b"\x00" in data[:8192]
            ):
                continue
            if secret in data or DEFAULT_AREAL_ADMIN_KEY in data:
                matches.append(str(candidate))
    if matches:
        raise RuntimeError(
            "runtime admin key remains in persisted text artifacts: "
            + ", ".join(matches)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    parser.add_argument("--verify-absent", action="store_true")
    args = parser.parse_args()

    if args.watch_seconds < 0:
        parser.error("--watch-seconds must be non-negative")
    root_value = os.environ.get("JPH_ROOT")
    if not root_value:
        parser.error("JPH_ROOT is required")
    secret_value = os.environ.pop(SECRET_ENV, None)
    if not secret_value:
        parser.error(f"{SECRET_ENV} is required")

    root = Path(root_value).resolve(strict=True)
    targets = [_within(target, root) for target in args.targets]
    secret = secret_value.encode("utf-8")
    deadline = time.monotonic() + args.watch_seconds
    while True:
        _redact(targets, secret)
        if args.verify_absent:
            _verify_absent(targets, secret)
        if args.watch_seconds == 0:
            return 0
        if time.monotonic() >= deadline:
            return 0
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
