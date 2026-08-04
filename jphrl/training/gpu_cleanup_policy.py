"""Pure fail-closed planning for the exceptional lyy/OOM cleanup boundary.

This module deliberately has no signaling or process-control operation.  A
candidate carries the observed PID start time so a separately authorized
executor would still have to re-read and match process identity immediately
before acting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CONFIRMED_OOM_FAILURE_CLASS = "cuda-out-of-memory"
_OBSERVATION_FIELDS = {
    "pid",
    "process_start_time",
    "owner_user",
    "owner_uid",
}


@dataclass(frozen=True)
class GpuCleanupCandidate:
    pid: int
    process_start_time: int
    owner_user: str
    owner_uid: int


def plan_lyy_oom_cleanup_candidates(
    *,
    failure_class: object,
    expected_lyy_uid: object,
    observed_processes: Sequence[object],
) -> tuple[GpuCleanupCandidate, ...]:
    """Return identity-bound candidates only for an explicit OOM and exact lyy UID.

    Malformed or ambiguous observation sets return no candidates.  Other users
    are observed but never returned, even when their UID or username collides
    with one half of the required identity pair.
    """

    if failure_class != CONFIRMED_OOM_FAILURE_CLASS:
        return ()
    if type(expected_lyy_uid) is not int or expected_lyy_uid < 0:
        return ()
    if isinstance(observed_processes, (str, bytes)) or not isinstance(
        observed_processes, Sequence
    ):
        return ()

    candidates: list[GpuCleanupCandidate] = []
    seen_pids: set[int] = set()
    for raw in observed_processes:
        if not isinstance(raw, Mapping) or set(raw) != _OBSERVATION_FIELDS:
            return ()
        pid = raw.get("pid")
        process_start_time = raw.get("process_start_time")
        owner_user = raw.get("owner_user")
        owner_uid = raw.get("owner_uid")
        if (
            type(pid) is not int
            or pid <= 1
            or pid in seen_pids
            or type(process_start_time) is not int
            or process_start_time <= 0
            or not isinstance(owner_user, str)
            or not owner_user
            or type(owner_uid) is not int
            or owner_uid < 0
        ):
            return ()
        seen_pids.add(pid)
        if owner_user == "lyy" and owner_uid == expected_lyy_uid:
            candidates.append(
                GpuCleanupCandidate(
                    pid=pid,
                    process_start_time=process_start_time,
                    owner_user=owner_user,
                    owner_uid=owner_uid,
                )
            )
    return tuple(candidates)


__all__ = [
    "CONFIRMED_OOM_FAILURE_CLASS",
    "GpuCleanupCandidate",
    "plan_lyy_oom_cleanup_candidates",
]
