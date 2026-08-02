from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


_SAFE_PAIR_ID = re.compile(r"^[A-Za-z0-9._-]{16,160}$")


def write_pointer(
    *,
    configured_root: Path,
    pair_id: str,
    cell: str,
    pointer: Path,
    run_root: Path,
) -> None:
    if configured_root.is_symlink():
        raise ValueError(f"configured root is a symlink: {configured_root}")
    root = configured_root.resolve()
    if not root.is_dir():
        raise ValueError(f"invalid configured root: {root}")
    if _SAFE_PAIR_ID.fullmatch(pair_id) is None:
        raise ValueError("unsafe screen pair ID")
    if cell not in {"c0", "c1"}:
        raise ValueError(f"unknown screen cell: {cell}")
    declared_pair_root = (
        root / "artifacts" / "sglang-logprob-screen" / "pairs" / pair_id
    )
    if declared_pair_root.is_symlink():
        raise ValueError(f"screen pair root is a symlink: {declared_pair_root}")
    expected_pair_root = declared_pair_root.resolve()
    if (
        not expected_pair_root.is_relative_to(root)
        or not expected_pair_root.is_dir()
    ):
        raise ValueError(f"invalid screen pair root: {expected_pair_root}")
    if expected_pair_root.stat().st_mode & 0o777 != 0o700:
        raise ValueError("screen pair root is not private")

    resolved_pointer = pointer.resolve()
    if (
        resolved_pointer.parent != expected_pair_root
        or resolved_pointer.name != f"{cell}-run-root.txt"
    ):
        raise ValueError(f"screen pointer escapes pair root: {resolved_pointer}")
    if run_root.is_symlink():
        raise ValueError(f"screen cell run root is a symlink: {run_root}")
    resolved_run_root = run_root.resolve()
    if (
        not resolved_run_root.is_relative_to(root)
        or not resolved_run_root.is_dir()
    ):
        raise ValueError(f"invalid screen cell run root: {resolved_run_root}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(expected_pair_root, directory_flags)
    fd = -1
    try:
        fd = os.open(
            resolved_pointer.name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(str(resolved_run_root))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write one root-bounded SGLang screen pair pointer"
    )
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--cell", choices=("c0", "c1"), required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    write_pointer(
        configured_root=Path(os.environ["JPH_ROOT"]),
        pair_id=args.pair_id,
        cell=args.cell,
        pointer=args.pointer,
        run_root=args.run_root,
    )


if __name__ == "__main__":
    main()
