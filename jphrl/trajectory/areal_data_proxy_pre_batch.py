from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

EXPORT_STYLES = frozenset({"individual", "concat"})
HOOK_STAGE = "after-export-trajectory-before-concat-padded-tensors"
_TENSOR_BATCH_FIELDS = frozenset(
    {
        "input_ids",
        "loss_mask",
        "logprobs",
        "versions",
        "attention_mask",
        "rewards",
    }
)


class ArealDataProxyPreBatchHookError(ValueError):
    """Raised when a DataProxy callback no longer has trajectory identity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealDataProxyPreBatchHookError(message)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class PreBatchTrajectoryExport:
    """One real AReaL trajectory before padded tensor concatenation.

    ``exported_interactions`` is a read-only view of the mapping returned by
    ``SessionData.export_trajectory()``. In ``concat`` mode its top-level values
    are leaves; every ancestor remains available through the real ``parent``
    pointers on those interaction objects.
    """

    session_id: str
    trajectory_id: int
    exported_interactions: Mapping[str, Any]
    export_style: str
    turn_discount: float
    hook_stage: str = HOOK_STAGE

    @property
    def interactions(self) -> Mapping[str, Any]:
        """Compatibility spelling used by the existing training-record adapter."""

        return self.exported_interactions


@dataclass(frozen=True)
class PreBatchTrajectoryAudit:
    """Identity summary produced without inspecting a merged batch row."""

    session_id: str
    trajectory_id: int
    export_style: str
    exported_interaction_ids: tuple[str, ...]
    all_interaction_ids: tuple[str, ...]


class DataProxyPreBatchExportHook(Protocol):
    """Signature added to the pinned AReaL DataProxy by the project patch."""

    def __call__(
        self,
        *,
        session_id: str,
        trajectory_id: int,
        interactions: Mapping[str, Any],
        discount: float,
        style: str,
    ) -> Awaitable[None] | None: ...


PreBatchTrajectoryConsumer = Callable[[PreBatchTrajectoryExport], object]


def validate_pre_batch_trajectory_export(
    export: PreBatchTrajectoryExport,
) -> PreBatchTrajectoryAudit:
    """Require real interaction objects and recover their pre-merge identity tree."""

    _require(_is_non_empty_string(export.session_id), "session ID is missing")
    _require(export.hook_stage == HOOK_STAGE, "DataProxy hook stage is not pre-batch")
    _require(
        _is_int(export.trajectory_id) and export.trajectory_id >= 0,
        "trajectory ID must be a non-negative integer",
    )
    _require(export.export_style in EXPORT_STYLES, "unknown AReaL export style")
    _require(
        _is_finite_number(export.turn_discount)
        and 0.0 <= float(export.turn_discount) <= 1.0,
        "turn discount must be a finite value in [0, 1]",
    )
    _require(
        isinstance(export.exported_interactions, Mapping),
        "pre-batch interactions must be a mapping",
    )

    exported_ids = tuple(str(key) for key in export.exported_interactions)
    raw_keys = set(export.exported_interactions)
    if _TENSOR_BATCH_FIELDS.issubset(raw_keys) or raw_keys == {"interactions"}:
        raise ArealDataProxyPreBatchHookError(
            "post-batch trajectory data cannot be bound to interaction IDs"
        )
    _require(bool(exported_ids), "pre-batch trajectory has no interactions")
    _require(
        all(_is_non_empty_string(key) for key in export.exported_interactions),
        "exported interaction key must be a non-empty string",
    )

    discovered: dict[str, Any] = {}
    parent_ids: set[str] = set()
    for exported_id, leaf in export.exported_interactions.items():
        actual_exported_id = getattr(leaf, "interaction_id", None)
        _require(
            actual_exported_id == exported_id,
            "export key differs from the real interaction identity",
        )

        current = leaf
        chain_ids: set[str] = set()
        while current is not None:
            interaction_id = getattr(current, "interaction_id", None)
            _require(
                _is_non_empty_string(interaction_id),
                "pre-batch interaction object has no interaction ID",
            )
            interaction_id = str(interaction_id)
            _require(
                interaction_id not in chain_ids,
                "pre-batch interaction parent chain contains a cycle",
            )
            chain_ids.add(interaction_id)

            previous = discovered.get(interaction_id)
            _require(
                previous is None or previous is current,
                "one interaction ID refers to different pre-batch objects",
            )
            if previous is None:
                discovered[interaction_id] = current

            parent = getattr(current, "parent", None)
            if parent is not None:
                parent_id = getattr(parent, "interaction_id", None)
                _require(
                    _is_non_empty_string(parent_id),
                    "pre-batch interaction parent has no interaction ID",
                )
                parent_ids.add(str(parent_id))
            current = parent

    if export.export_style == "individual":
        _require(
            set(exported_ids) == set(discovered),
            "individual export does not expose every interaction identity",
        )
        exported_positions = {
            interaction_id: index
            for index, interaction_id in enumerate(exported_ids)
        }
        for interaction_id, interaction in discovered.items():
            parent = getattr(interaction, "parent", None)
            if parent is None:
                continue
            parent_id = str(parent.interaction_id)
            _require(
                exported_positions[parent_id] < exported_positions[interaction_id],
                "individual export places a child before its parent",
            )
    else:
        _require(
            all(interaction_id not in parent_ids for interaction_id in exported_ids),
            "concat export contains a non-leaf top-level interaction",
        )
        _require(
            all(
                getattr(interaction, "chat_template_type", None) == "concat"
                for interaction in discovered.values()
            ),
            "concat export contains a non-concat interaction",
        )

    ordered_all_ids = tuple(
        interaction_id
        for interaction_id, _ in sorted(
            discovered.items(),
            key=lambda item: _interaction_depth(item[1]),
        )
    )
    return PreBatchTrajectoryAudit(
        session_id=export.session_id,
        trajectory_id=export.trajectory_id,
        export_style=export.export_style,
        exported_interaction_ids=exported_ids,
        all_interaction_ids=ordered_all_ids,
    )


def _interaction_depth(interaction: Any) -> int:
    depth = 0
    current = getattr(interaction, "parent", None)
    seen: set[int] = set()
    while current is not None:
        identity = id(current)
        _require(
            identity not in seen,
            "pre-batch interaction parent chain contains a cycle",
        )
        seen.add(identity)
        depth += 1
        current = getattr(current, "parent", None)
    return depth


class VerifiedDataProxyPreBatchHook:
    """Validate AReaL's unmerged trajectory, then invoke a project consumer.

    Sync and async consumers are supported. Their return value is deliberately
    ignored because this observational hook cannot replace the exported
    trajectory. Exceptions are not caught and therefore abort DataProxy export.
    """

    def __init__(self, consumer: PreBatchTrajectoryConsumer):
        _require(callable(consumer), "pre-batch trajectory consumer is not callable")
        self._consumer = consumer

    async def __call__(
        self,
        *,
        session_id: str,
        trajectory_id: int,
        interactions: Mapping[str, Any],
        discount: float,
        style: str,
    ) -> None:
        export = PreBatchTrajectoryExport(
            session_id=session_id,
            trajectory_id=trajectory_id,
            exported_interactions=MappingProxyType(dict(interactions)),
            export_style=style,
            turn_discount=discount,
        )
        validate_pre_batch_trajectory_export(export)
        result = self._consumer(export)
        if inspect.isawaitable(result):
            await result


def require_data_proxy_pre_batch_export(
    *,
    session_id: str,
    trajectory_id: int,
    interactions: Mapping[str, Any],
    discount: float,
    style: str,
) -> None:
    """Importable validation-only hook for ``AREAL_PRE_BATCH_HOOK``.

    Production record persistence should wrap :class:`VerifiedDataProxyPreBatchHook`
    with the staged-record consumer. This callable is useful for deployment smoke
    because it proves the patched ``__main__`` injection path without inventing a
    post-batch binding.
    """

    export = PreBatchTrajectoryExport(
        session_id=session_id,
        trajectory_id=trajectory_id,
        exported_interactions=MappingProxyType(dict(interactions)),
        export_style=style,
        turn_discount=discount,
    )
    validate_pre_batch_trajectory_export(export)


async def export_session_trajectory_with_pre_batch_hook(
    session: Any,
    *,
    discount: float,
    style: str,
    trajectory_id: int | None,
    hook: DataProxyPreBatchExportHook,
) -> tuple[int, dict[str, Any]]:
    """Model the exact DataProxy seam against a real ``SessionData`` object.

    The project patch performs this same ordering inside ``/export_trajectories``:
    export one trajectory, await its callback, and only then merge/tensorize it.
    Hook failure is intentionally fail-closed; the trajectory must not continue
    into a training batch without a verified identity binding.
    """

    _require(
        callable(getattr(session, "export_trajectory", None)),
        "session does not expose export_trajectory",
    )
    _require(callable(hook), "pre-batch export hook is not callable")
    exported_trajectory_id, interactions = session.export_trajectory(
        discount=discount,
        style=style,
        trajectory_id=trajectory_id,
    )
    result = hook(
        session_id=session.session_id,
        trajectory_id=exported_trajectory_id,
        interactions=interactions,
        discount=discount,
        style=style,
    )
    if inspect.isawaitable(result):
        await result
    return exported_trajectory_id, interactions
