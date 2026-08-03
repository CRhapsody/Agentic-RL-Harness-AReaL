from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SIDECAR_SCHEMA_VERSION = "jph.areal-interaction-adapter-sidecar.v1"
ARCHIVE_SCHEMA_VERSION = "jph.areal-bound-training-samples.v1"
ROUTE_KINDS = frozenset({"agent-service-session", "rlvr-workflow"})
EXPORT_STYLES = frozenset({"individual", "concat"})
REQUIRED_TENSOR_FIELDS = (
    "input_ids",
    "loss_mask",
    "logprobs",
    "versions",
    "attention_mask",
    "rewards",
)


class ArealInteractionAdapterError(ValueError):
    """Raised when an Agent decision cannot be bound to one AReaL interaction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealInteractionAdapterError(message)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArealInteractionAdapterError(
            "record is not finite canonical JSON"
        ) from exc


def _record_sha256(record: Mapping[str, object], hash_field: str) -> str:
    unsigned = {key: value for key, value in record.items() if key != hash_field}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


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
class InteractionBinding:
    """One-to-one identity binding for a model call and an AReaL interaction."""

    episode_id: str
    model_call_id: str
    session_id: str | None
    trajectory_id: int | None
    interaction_id: str
    parent_interaction_id: str | None
    ordinal: int
    joint_version_id: str
    route_kind: str


def build_interaction_adapter_sidecar(
    bindings: Sequence[InteractionBinding],
) -> dict[str, object]:
    """Build a canonical, hashed identity sidecar for one episode."""

    record: dict[str, object] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "bindings": [asdict(binding) for binding in bindings],
    }
    record["sidecar_sha256"] = _record_sha256(record, "sidecar_sha256")
    validate_interaction_adapter_sidecar(record)
    return record


def validate_interaction_adapter_sidecar(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed unless model-call and interaction identities are one-to-one."""

    _require(
        record.get("schema_version") == SIDECAR_SCHEMA_VERSION,
        "unknown interaction sidecar schema",
    )
    _require(
        record.get("sidecar_sha256") == _record_sha256(record, "sidecar_sha256"),
        "interaction sidecar hash mismatch",
    )
    raw_bindings = record.get("bindings")
    _require(
        isinstance(raw_bindings, list) and bool(raw_bindings),
        "interaction sidecar requires at least one binding",
    )

    required_fields = set(InteractionBinding.__dataclass_fields__)
    bindings: list[InteractionBinding] = []
    for raw in raw_bindings:
        _require(isinstance(raw, Mapping), "interaction binding must be an object")
        _require(
            set(raw) == required_fields,
            "interaction binding field set differs from schema",
        )
        try:
            binding = InteractionBinding(**dict(raw))
        except TypeError as exc:
            raise ArealInteractionAdapterError(
                "invalid interaction binding fields"
            ) from exc
        for name in (
            "episode_id",
            "model_call_id",
            "interaction_id",
            "joint_version_id",
            "route_kind",
        ):
            _require(
                _is_non_empty_string(getattr(binding, name)),
                f"interaction binding {name} cannot be empty",
            )
        _require(
            binding.route_kind in ROUTE_KINDS,
            "unknown AReaL interaction route kind",
        )
        _require(
            _is_int(binding.ordinal) and binding.ordinal >= 0,
            "interaction ordinal must be a non-negative integer",
        )
        _require(
            binding.parent_interaction_id is None
            or _is_non_empty_string(binding.parent_interaction_id),
            "parent interaction ID must be null or non-empty",
        )
        _require(
            binding.trajectory_id is None
            or (_is_int(binding.trajectory_id) and binding.trajectory_id >= 0),
            "trajectory ID must be null or a non-negative integer",
        )
        if binding.route_kind == "agent-service-session":
            _require(
                _is_non_empty_string(binding.session_id),
                "Agent Service binding requires a session ID",
            )
        else:
            _require(
                binding.session_id is None and binding.trajectory_id is None,
                "RLVR binding cannot fabricate session or trajectory IDs",
            )
        bindings.append(binding)

    _require(
        [binding.ordinal for binding in bindings] == list(range(len(bindings))),
        "interaction ordinals must be contiguous and ordered",
    )
    episode_ids = {binding.episode_id for binding in bindings}
    session_ids = {binding.session_id for binding in bindings}
    trajectory_ids = {binding.trajectory_id for binding in bindings}
    version_ids = {binding.joint_version_id for binding in bindings}
    route_kinds = {binding.route_kind for binding in bindings}
    _require(len(episode_ids) == 1, "sidecar contains more than one episode")
    _require(len(session_ids) == 1, "sidecar contains more than one session")
    _require(len(trajectory_ids) == 1, "sidecar contains more than one trajectory")
    _require(len(version_ids) == 1, "sidecar contains more than one JointVersion")
    _require(len(route_kinds) == 1, "sidecar contains more than one route kind")

    model_call_ids = [binding.model_call_id for binding in bindings]
    interaction_ids = [binding.interaction_id for binding in bindings]
    _require(
        len(set(model_call_ids)) == len(model_call_ids),
        "model call ID is bound more than once",
    )
    _require(
        len(set(interaction_ids)) == len(interaction_ids),
        "AReaL interaction ID is bound more than once",
    )

    by_interaction = {binding.interaction_id: binding for binding in bindings}
    for binding in bindings:
        parent_id = binding.parent_interaction_id
        if parent_id is None:
            continue
        _require(parent_id != binding.interaction_id, "interaction cannot parent itself")
        _require(
            parent_id in by_interaction,
            "parent interaction is absent from the sidecar",
        )
        _require(
            by_interaction[parent_id].ordinal < binding.ordinal,
            "parent interaction must precede its child",
        )

    return {
        "ok": True,
        "episode_id": bindings[0].episode_id,
        "session_id": bindings[0].session_id,
        "trajectory_id": bindings[0].trajectory_id,
        "joint_version_id": bindings[0].joint_version_id,
        "route_kind": bindings[0].route_kind,
        "model_call_ids": model_call_ids,
        "interaction_ids": interaction_ids,
        "binding_count": len(bindings),
        "sidecar_sha256": record["sidecar_sha256"],
    }


def model_call_id_for_interaction(
    record: Mapping[str, object], interaction_id: str
) -> str:
    validate_interaction_adapter_sidecar(record)
    for raw in record["bindings"]:
        if raw["interaction_id"] == interaction_id:
            return str(raw["model_call_id"])
    raise ArealInteractionAdapterError(
        f"interaction ID is absent from sidecar: {interaction_id}"
    )


def interaction_id_for_model_call(
    record: Mapping[str, object], model_call_id: str
) -> str:
    validate_interaction_adapter_sidecar(record)
    for raw in record["bindings"]:
        if raw["model_call_id"] == model_call_id:
            return str(raw["interaction_id"])
    raise ArealInteractionAdapterError(
        f"model call ID is absent from sidecar: {model_call_id}"
    )


def _plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _single_row(tensors: Mapping[str, object], field: str) -> list[object]:
    value = tensors.get(field)
    _require(isinstance(value, list), f"tensor {field} must be a list")
    _require(len(value) == 1, f"tensor {field} must have batch size one")
    row = value[0]
    _require(isinstance(row, list), f"tensor {field} row must be a list")
    return row


def _validate_tensor_dict(tensors: Mapping[str, object]) -> int:
    _require(
        set(tensors) == set(REQUIRED_TENSOR_FIELDS),
        "training sample tensor field set differs from AReaL contract",
    )
    input_ids = _single_row(tensors, "input_ids")
    loss_mask = _single_row(tensors, "loss_mask")
    logprobs = _single_row(tensors, "logprobs")
    versions = _single_row(tensors, "versions")
    attention_mask = _single_row(tensors, "attention_mask")
    rewards = tensors.get("rewards")
    _require(
        isinstance(rewards, list)
        and len(rewards) == 1
        and _is_finite_number(rewards[0]),
        "training sample requires one finite reward",
    )
    sequence_length = len(input_ids)
    _require(sequence_length > 0, "training sample sequence cannot be empty")
    for field, row in (
        ("loss_mask", loss_mask),
        ("logprobs", logprobs),
        ("versions", versions),
        ("attention_mask", attention_mask),
    ):
        _require(
            len(row) == sequence_length,
            f"training sample {field} length differs from input_ids",
        )
    _require(
        all(_is_int(token) and token >= 0 for token in input_ids),
        "training sample token IDs must be non-negative integers",
    )
    _require(
        all(_is_int(mask) and mask in (0, 1) for mask in loss_mask),
        "training sample loss mask must contain integer zero or one",
    )
    _require(
        all(_is_finite_number(logprob) for logprob in logprobs),
        "training sample logprobs must be finite",
    )
    _require(
        all(_is_int(version) and version >= -1 for version in versions),
        "training sample versions must be integers greater than or equal to -1",
    )
    _require(
        all(value is True or value == 1 for value in attention_mask),
        "training sample attention mask must activate every unpadded token",
    )
    for index, mask in enumerate(loss_mask):
        if mask == 0:
            _require(
                float(logprobs[index]) == 0.0 and versions[index] == -1,
                "masked prompt positions require logprob=0 and version=-1",
            )
        else:
            _require(
                float(logprobs[index]) <= 1e-5 and versions[index] >= 0,
                "trainable positions require a non-positive logprob and policy version",
            )
    _require(any(mask == 1 for mask in loss_mask), "training sample has no actions")
    return sequence_length


def _bindings_by_interaction(
    sidecar: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    validate_interaction_adapter_sidecar(sidecar)
    return {binding["interaction_id"]: binding for binding in sidecar["bindings"]}


def _binding_chain(
    leaf_interaction_id: str,
    by_interaction: Mapping[str, Mapping[str, object]],
) -> list[Mapping[str, object]]:
    chain: list[Mapping[str, object]] = []
    current_id: str | None = leaf_interaction_id
    while current_id is not None:
        _require(current_id in by_interaction, "binding chain references an unknown parent")
        binding = by_interaction[current_id]
        chain.append(binding)
        parent_id = binding["parent_interaction_id"]
        current_id = str(parent_id) if parent_id is not None else None
    chain.reverse()
    return chain


def _actual_parent_interaction_id(interaction: Any) -> str | None:
    parent = getattr(interaction, "parent", None)
    if parent is None:
        return None
    parent_id = getattr(parent, "interaction_id", None)
    _require(
        _is_non_empty_string(parent_id),
        "AReaL interaction parent has no interaction ID",
    )
    return str(parent_id)


def _decision_spans(
    *,
    chain: Sequence[Mapping[str, object]],
    interactions: Mapping[str, Any],
    tensors: Mapping[str, object],
) -> list[dict[str, object]]:
    input_ids = _single_row(tensors, "input_ids")
    loss_mask = _single_row(tensors, "loss_mask")
    logprobs = _single_row(tensors, "logprobs")
    versions = _single_row(tensors, "versions")
    spans: list[dict[str, object]] = []
    previous_end = -1
    for binding in chain:
        interaction_id = str(binding["interaction_id"])
        interaction = interactions[interaction_id]
        response = getattr(interaction, "model_response", None)
        _require(response is not None, "AReaL interaction has no ModelResponse")
        input_tokens = list(getattr(response, "input_tokens", ()))
        output_tokens = list(getattr(response, "output_tokens", ()))
        output_logprobs = list(getattr(response, "output_logprobs", ()))
        output_versions = list(getattr(response, "output_versions", ()))
        _require(input_tokens and output_tokens, "interaction token data cannot be empty")
        _require(
            len(output_tokens) == len(output_logprobs) == len(output_versions),
            "interaction output token metadata lengths differ",
        )
        start = len(input_tokens)
        end = start + len(output_tokens)
        _require(
            start >= previous_end and end <= len(input_ids),
            "interaction decision spans overlap or exceed the exported tensor",
        )
        _require(
            input_ids[: len(input_tokens)] == input_tokens,
            "interaction input tokens are not a prefix of the exported concat sample",
        )
        _require(
            input_ids[start:end] == output_tokens,
            "interaction output tokens differ from the exported tensor span",
        )
        _require(
            loss_mask[start:end] == [1] * len(output_tokens),
            "interaction output span is not fully trainable",
        )
        _require(
            all(
                math.isclose(
                    float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6
                )
                for actual, expected in zip(logprobs[start:end], output_logprobs)
            ),
            "interaction output logprobs differ from the exported tensor span",
        )
        _require(
            versions[start:end] == output_versions,
            "interaction output versions differ from the exported tensor span",
        )
        spans.append(
            {
                "model_call_id": binding["model_call_id"],
                "interaction_id": interaction_id,
                "start": start,
                "end": end,
            }
        )
        previous_end = end

    expected_mask = [0] * len(input_ids)
    for span in spans:
        expected_mask[span["start"] : span["end"]] = [1] * (
            span["end"] - span["start"]
        )
    _require(
        loss_mask == expected_mask,
        "exported loss mask contains actions not represented by sidecar spans",
    )
    return spans


def _expected_export_ids(
    sidecar: Mapping[str, object], export_style: str
) -> list[str]:
    bindings = sidecar["bindings"]
    if export_style == "individual":
        return [str(binding["interaction_id"]) for binding in bindings]
    parent_ids = {
        str(binding["parent_interaction_id"])
        for binding in bindings
        if binding["parent_interaction_id"] is not None
    }
    return [
        str(binding["interaction_id"])
        for binding in bindings
        if binding["interaction_id"] not in parent_ids
    ]


def _sample_id(
    *, episode_id: str, export_style: str, leaf_interaction_id: str
) -> str:
    payload = {
        "schema_version": "jph.areal-bound-training-sample-id.v1",
        "episode_id": episode_id,
        "export_style": export_style,
        "leaf_interaction_id": leaf_interaction_id,
    }
    return f"jph-sample-{hashlib.sha256(_canonical_json(payload)).hexdigest()[:32]}"


def _validate_export_arguments(
    *,
    interaction_sidecar: Mapping[str, object],
    export_style: str,
    turn_discount: float | None,
) -> dict[str, object]:
    sidecar_audit = validate_interaction_adapter_sidecar(interaction_sidecar)
    _require(export_style in EXPORT_STYLES, "unknown AReaL export style")
    _require(
        turn_discount is None
        or (
            _is_finite_number(turn_discount)
            and 0.0 <= float(turn_discount) <= 1.0
        ),
        "turn discount must be null or a finite value in [0, 1]",
    )
    return sidecar_audit


def _collect_interaction_tree(
    *,
    exported_interactions: Mapping[str, Any],
    interaction_sidecar: Mapping[str, object],
    export_style: str,
) -> dict[str, Any]:
    expected_export_ids = _expected_export_ids(interaction_sidecar, export_style)
    _require(
        list(exported_interactions) == expected_export_ids,
        "AReaL exported interaction set differs from sidecar tree",
    )
    discovered: dict[str, Any] = {}
    for exported_id, leaf in exported_interactions.items():
        _require(
            getattr(leaf, "interaction_id", None) == exported_id,
            "AReaL export key differs from interaction identity",
        )
        current = leaf
        while current is not None:
            current_id = getattr(current, "interaction_id", None)
            _require(
                _is_non_empty_string(current_id),
                "AReaL interaction tree contains an object without an ID",
            )
            if current_id in discovered:
                _require(
                    discovered[current_id] is current,
                    "AReaL interaction tree reuses an ID for different objects",
                )
                break
            discovered[str(current_id)] = current
            current = getattr(current, "parent", None)

    expected_all_ids = [
        str(binding["interaction_id"])
        for binding in interaction_sidecar["bindings"]
    ]
    _require(
        set(discovered) == set(expected_all_ids),
        "pre-merge interaction tree differs from sidecar membership",
    )
    interactions = {
        interaction_id: discovered[interaction_id]
        for interaction_id in expected_all_ids
    }
    by_interaction = _bindings_by_interaction(interaction_sidecar)
    for interaction_id, binding in by_interaction.items():
        interaction = interactions[interaction_id]
        _require(
            _actual_parent_interaction_id(interaction)
            == binding["parent_interaction_id"],
            "AReaL parent relation differs from sidecar",
        )
        if export_style == "concat":
            _require(
                getattr(interaction, "chat_template_type", None) == "concat",
                "concat export requires chat_template_type=concat",
            )
    return interactions


def _build_archive_from_exported_interactions(
    *,
    exported_interactions: Mapping[str, Any],
    interaction_sidecar: Mapping[str, object],
    export_style: str,
    turn_discount: float | None,
) -> dict[str, object]:
    sidecar_audit = _validate_export_arguments(
        interaction_sidecar=interaction_sidecar,
        export_style=export_style,
        turn_discount=turn_discount,
    )
    interactions = _collect_interaction_tree(
        exported_interactions=exported_interactions,
        interaction_sidecar=interaction_sidecar,
        export_style=export_style,
    )
    by_interaction = _bindings_by_interaction(interaction_sidecar)

    samples: list[dict[str, object]] = []
    for leaf_interaction_id, interaction in exported_interactions.items():
        chain = (
            [by_interaction[leaf_interaction_id]]
            if export_style == "individual"
            else _binding_chain(leaf_interaction_id, by_interaction)
        )
        raw_tensors = interaction.to_tensor_dict()
        _require(
            isinstance(raw_tensors, Mapping),
            "to_tensor_dict must return a mapping",
        )
        _require(
            all(field in raw_tensors for field in REQUIRED_TENSOR_FIELDS),
            "to_tensor_dict is missing an AReaL tensor field",
        )
        tensors = {
            field: _plain(raw_tensors[field]) for field in REQUIRED_TENSOR_FIELDS
        }
        sequence_length = _validate_tensor_dict(tensors)
        spans = _decision_spans(
            chain=chain,
            interactions=interactions,
            tensors=tensors,
        )
        samples.append(
            {
                "sample_id": _sample_id(
                    episode_id=str(sidecar_audit["episode_id"]),
                    export_style=export_style,
                    leaf_interaction_id=leaf_interaction_id,
                ),
                "leaf_interaction_id": leaf_interaction_id,
                "leaf_model_call_id": chain[-1]["model_call_id"],
                "included_interaction_ids": [
                    binding["interaction_id"] for binding in chain
                ],
                "included_model_call_ids": [
                    binding["model_call_id"] for binding in chain
                ],
                "decision_spans": spans,
                "sequence_length": sequence_length,
                "tensor_dict": tensors,
            }
        )

    record: dict[str, object] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "interaction_sidecar": dict(interaction_sidecar),
        "source_sidecar_sha256": interaction_sidecar["sidecar_sha256"],
        "export_style": export_style,
        "turn_discount": turn_discount,
        "sample_count": len(samples),
        "samples": samples,
        "evidence_scope": {
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
            "training_sample_archive_only": True,
        },
    }
    record["record_sha256"] = _record_sha256(record, "record_sha256")
    validate_bound_training_sample_archive(record)
    return record


def archive_premerged_exported_interactions(
    *,
    exported_interactions: Mapping[str, Any],
    interaction_sidecar: Mapping[str, object],
    export_style: str,
    turn_discount: float | None,
) -> dict[str, object]:
    """Archive AReaL's styled interactions before padded batch concatenation.

    The supported hook point is immediately after
    ``SessionData.export_trajectory()`` and before ``concat_padded_tensors()``.
    In concat mode, ancestor objects are recovered from each leaf's ``parent``
    chain and checked against the full sidecar.
    """

    _require(
        isinstance(exported_interactions, Mapping),
        "pre-merge AReaL interactions must be a mapping",
    )
    return _build_archive_from_exported_interactions(
        exported_interactions=exported_interactions,
        interaction_sidecar=interaction_sidecar,
        export_style=export_style,
        turn_discount=turn_discount,
    )


def export_bound_training_sample_archive(
    *,
    interaction_cache: Any,
    interaction_sidecar: Mapping[str, object],
    export_style: str,
    turn_discount: float | None,
) -> dict[str, object]:
    """Export AReaL samples while preserving every model-call identity.

    The AReaL cache remains the source of truth for reward discounting, tree
    construction, and tensorization. This wrapper validates its output and records
    which Agent model calls occupy each trainable token span.
    """

    sidecar_audit = _validate_export_arguments(
        interaction_sidecar=interaction_sidecar,
        export_style=export_style,
        turn_discount=turn_discount,
    )
    _require(
        hasattr(interaction_cache, "export_interactions"),
        "interaction cache has no export_interactions method",
    )

    by_interaction = _bindings_by_interaction(interaction_sidecar)
    cache_ids = list(interaction_cache.keys())
    _require(
        cache_ids == list(sidecar_audit["interaction_ids"]),
        "interaction cache order or membership differs from sidecar",
    )
    for interaction_id, binding in by_interaction.items():
        interaction = interaction_cache[interaction_id]
        _require(
            getattr(interaction, "interaction_id", None) == interaction_id,
            "AReaL cache key differs from interaction identity",
        )
        _require(
            _actual_parent_interaction_id(interaction)
            == binding["parent_interaction_id"],
            "AReaL parent relation differs from sidecar",
        )
        if export_style == "concat":
            _require(
                getattr(interaction, "chat_template_type", None) == "concat",
                "concat export requires chat_template_type=concat",
            )

    exported = interaction_cache.export_interactions(
        style=export_style,
        reward_discount=turn_discount,
    )
    _require(
        isinstance(exported, Mapping),
        "AReaL interaction export must return a mapping",
    )
    _require(
        sidecar_audit["interaction_ids"] == cache_ids,
        "interaction cache identity changed during export",
    )
    return _build_archive_from_exported_interactions(
        exported_interactions=exported,
        interaction_sidecar=interaction_sidecar,
        export_style=export_style,
        turn_discount=turn_discount,
    )


def validate_bound_training_sample_archive(
    record: Mapping[str, object],
) -> dict[str, object]:
    _require(
        record.get("schema_version") == ARCHIVE_SCHEMA_VERSION,
        "unknown training sample archive schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record, "record_sha256"),
        "training sample archive hash mismatch",
    )
    sidecar = record.get("interaction_sidecar")
    _require(isinstance(sidecar, Mapping), "archive interaction sidecar is missing")
    sidecar_audit = validate_interaction_adapter_sidecar(sidecar)
    _require(
        record.get("source_sidecar_sha256") == sidecar_audit["sidecar_sha256"],
        "archive source sidecar hash mismatch",
    )
    export_style = record.get("export_style")
    _require(export_style in EXPORT_STYLES, "unknown archive export style")
    turn_discount = record.get("turn_discount")
    _require(
        turn_discount is None
        or (
            _is_finite_number(turn_discount)
            and 0.0 <= float(turn_discount) <= 1.0
        ),
        "archive turn discount must be null or in [0, 1]",
    )
    _require(
        record.get("evidence_scope")
        == {
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
            "training_sample_archive_only": True,
        },
        "training sample evidence scope differs from contract",
    )
    samples = record.get("samples")
    _require(isinstance(samples, list), "archive samples must be a list")
    expected_ids = _expected_export_ids(sidecar, str(export_style))
    _require(
        record.get("sample_count") == len(samples) == len(expected_ids),
        "archive sample count differs from sidecar tree",
    )
    by_interaction = _bindings_by_interaction(sidecar)
    seen_sample_ids: set[str] = set()
    for sample, expected_leaf_id in zip(samples, expected_ids):
        _require(isinstance(sample, Mapping), "archive sample must be an object")
        leaf_id = sample.get("leaf_interaction_id")
        _require(leaf_id == expected_leaf_id, "archive leaf order differs from sidecar")
        expected_chain = (
            [by_interaction[expected_leaf_id]]
            if export_style == "individual"
            else _binding_chain(expected_leaf_id, by_interaction)
        )
        _require(
            sample.get("included_interaction_ids")
            == [binding["interaction_id"] for binding in expected_chain]
            and sample.get("included_model_call_ids")
            == [binding["model_call_id"] for binding in expected_chain],
            "archive sample decision chain differs from sidecar",
        )
        _require(
            sample.get("leaf_model_call_id") == expected_chain[-1]["model_call_id"],
            "archive leaf model call differs from sidecar",
        )
        expected_sample_id = _sample_id(
            episode_id=str(sidecar_audit["episode_id"]),
            export_style=str(export_style),
            leaf_interaction_id=expected_leaf_id,
        )
        _require(
            sample.get("sample_id") == expected_sample_id,
            "archive sample ID differs from its identity",
        )
        _require(expected_sample_id not in seen_sample_ids, "duplicate archive sample ID")
        seen_sample_ids.add(expected_sample_id)
        tensors = sample.get("tensor_dict")
        _require(isinstance(tensors, Mapping), "archive tensor_dict is missing")
        sequence_length = _validate_tensor_dict(tensors)
        _require(
            sample.get("sequence_length") == sequence_length,
            "archive sequence length summary differs from tensors",
        )
        spans = sample.get("decision_spans")
        _require(
            isinstance(spans, list) and len(spans) == len(expected_chain),
            "archive decision span count differs from sidecar chain",
        )
        expected_mask = [0] * sequence_length
        previous_end = -1
        for span, binding in zip(spans, expected_chain):
            _require(isinstance(span, Mapping), "archive decision span must be an object")
            _require(
                span.get("model_call_id") == binding["model_call_id"]
                and span.get("interaction_id") == binding["interaction_id"],
                "archive decision span identity differs from sidecar",
            )
            start = span.get("start")
            end = span.get("end")
            _require(
                _is_int(start)
                and _is_int(end)
                and previous_end <= start < end <= sequence_length,
                "archive decision span is invalid or overlaps",
            )
            expected_mask[start:end] = [1] * (end - start)
            previous_end = end
        _require(
            _single_row(tensors, "loss_mask") == expected_mask,
            "archive loss mask differs from recorded decision spans",
        )

    return {
        "ok": True,
        "episode_id": sidecar_audit["episode_id"],
        "export_style": export_style,
        "sample_count": len(samples),
        "sidecar_sha256": sidecar_audit["sidecar_sha256"],
        "record_sha256": record["record_sha256"],
    }


def _write_private_record(
    record: Mapping[str, object],
    *,
    destination: str | Path,
    allowed_root: str | Path,
) -> Path:
    root = Path(allowed_root).expanduser().resolve()
    path = Path(destination).expanduser().resolve()
    _require(root.is_dir(), f"configured root does not exist: {root}")
    try:
        common = Path(os.path.commonpath((path, root)))
    except ValueError as exc:
        raise ArealInteractionAdapterError(
            "cannot compare output path with configured root"
        ) from exc
    _require(common == root, f"path escapes configured root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def write_interaction_adapter_sidecar(
    record: Mapping[str, object],
    *,
    destination: str | Path,
    allowed_root: str | Path,
) -> Path:
    validate_interaction_adapter_sidecar(record)
    return _write_private_record(
        record, destination=destination, allowed_root=allowed_root
    )


def write_bound_training_sample_archive(
    record: Mapping[str, object],
    *,
    destination: str | Path,
    allowed_root: str | Path,
) -> Path:
    validate_bound_training_sample_archive(record)
    return _write_private_record(
        record, destination=destination, allowed_root=allowed_root
    )
