from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from jphrl.trajectory.joint_credit_alignment import (
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.schema import JointVersion

AREAL_EXTERNAL_ADVANTAGE_BATCH_SCHEMA = "jph.areal-external-advantage-ppo-batch.v1"
_EVIDENCE_SCOPE = {
    "source_joint_credit_validated": True,
    "causal_shift_applied": True,
    "external_policy_advantage_preserved": True,
    "areal_ppo_update_invoked": False,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}
_TENSOR_FIELDS = {
    "advantages",
    "attention_mask",
    "input_ids",
    "logprobs",
    "loss_mask",
    "prox_logp",
    "returns",
    "rewards",
    "versions",
}
_ROLLOUT_TENSOR_FIELDS = {
    "attention_mask",
    "input_ids",
    "logprobs",
    "loss_mask",
    "rewards",
    "versions",
}


class ArealExternalAdvantageBatchError(ValueError):
    """Raised when S credit cannot safely enter AReaL's actor update."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealExternalAdvantageBatchError(message)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArealExternalAdvantageBatchError(
            "AReaL external-advantage batch is not finite canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _record_sha256(record: Mapping[str, object]) -> str:
    return _sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _exact_mapping(
    value: object,
    expected_fields: set[str],
    label: str,
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == expected_fields, f"{label} field set differs from schema")
    return value


def _single_row(tensors: Mapping[str, object], field: str) -> list[object]:
    raw = tensors.get(field)
    _require(isinstance(raw, list) and len(raw) == 1, f"{field} must have one row")
    row = raw[0]
    _require(isinstance(row, list), f"{field} row must be a list")
    return row


def _shift_left(row: list[object], sentinel: object) -> list[object]:
    _require(bool(row), "cannot causal-shift an empty sequence")
    return [*row[1:], sentinel]


def _joint_version(raw: object) -> JointVersion:
    fields = set(JointVersion.__dataclass_fields__)
    value = _exact_mapping(raw, fields, "Policy optimizer JointVersion")
    try:
        version = JointVersion(**dict(value))
    except TypeError as exc:
        raise ArealExternalAdvantageBatchError("invalid JointVersion") from exc
    _require(
        all(
            isinstance(getattr(version, field), str) and getattr(version, field)
            for field in fields
        ),
        "JointVersion fields must be non-empty strings",
    )
    return version


@dataclass(frozen=True)
class ValidatedArealExternalAdvantageBatch:
    joint_version: JointVersion
    episode_id: str
    source_joint_credit_sha256: str
    export_style: str
    inference_engine_version: int
    samples: tuple[Mapping[str, object], ...]
    trainable_token_count: int
    record_sha256: str

    @property
    def digest(self) -> str:
        return self.record_sha256


def _build_update_tensor_dict(policy_sample: Mapping[str, object]) -> dict[str, object]:
    base = _exact_mapping(
        policy_sample.get("tensor_dict"),
        {
            "attention_mask",
            "input_ids",
            "logprobs",
            "loss_mask",
            "rewards",
            "versions",
        },
        "S Policy tensor_dict",
    )
    input_ids = _single_row(base, "input_ids")
    attention_mask = _single_row(base, "attention_mask")
    loss_mask = _single_row(base, "loss_mask")
    logprobs = _single_row(base, "logprobs")
    versions = _single_row(base, "versions")
    advantages = _single_row(
        {"advantages": policy_sample.get("advantage_tensor")},
        "advantages",
    )
    rewards = base.get("rewards")
    length = len(input_ids)
    _require(
        length > 1
        and all(
            len(row) == length
            for row in (
                attention_mask,
                loss_mask,
                logprobs,
                versions,
                advantages,
            )
        ),
        "S Policy tensor lengths differ",
    )
    _require(
        isinstance(rewards, list)
        and len(rewards) == 1
        and _is_finite_number(rewards[0]),
        "S Policy reward must be one finite scalar",
    )

    shifted_mask = [int(value) for value in _shift_left(loss_mask, 0)]
    shifted_logprobs = [
        float(value) * mask
        for value, mask in zip(_shift_left(logprobs, 0.0), shifted_mask)
    ]
    shifted_advantages = [
        float(value) * mask
        for value, mask in zip(_shift_left(advantages, 0.0), shifted_mask)
    ]
    _require(
        any(shifted_mask),
        "AReaL PPO sample has no trainable token after causal shift",
    )
    _require(
        all(value in (0, 1) for value in shifted_mask),
        "causal-shifted loss mask must remain binary",
    )
    _require(
        all(
            _is_finite_number(value) and value <= 0.0
            for value, mask in zip(shifted_logprobs, shifted_mask)
            if mask
        ),
        "causal-shifted old Policy log-probabilities are invalid",
    )
    _require(
        all(_is_finite_number(value) for value in shifted_advantages),
        "causal-shifted Policy advantages must be finite",
    )
    return {
        "input_ids": [[int(value) for value in input_ids]],
        "attention_mask": [[bool(value) for value in attention_mask]],
        "loss_mask": [shifted_mask],
        "logprobs": [shifted_logprobs],
        "prox_logp": [list(shifted_logprobs)],
        # Pinned AReaL v2.0.0 keeps versions in their exported token positions
        # after compute_advantages(); preserve that exact contract here.
        "versions": [[int(value) for value in versions]],
        "advantages": [shifted_advantages],
        "returns": [list(shifted_advantages)],
        "rewards": [float(rewards[0])],
    }


def _build_rollout_tensor_dict(
    policy_sample: Mapping[str, object],
) -> dict[str, object]:
    base = _exact_mapping(
        policy_sample.get("tensor_dict"),
        _ROLLOUT_TENSOR_FIELDS,
        "S Policy rollout tensor_dict",
    )
    rewards = base.get("rewards")
    _require(
        isinstance(rewards, list)
        and len(rewards) == 1
        and _is_finite_number(rewards[0]),
        "S Policy rollout reward must be one finite scalar",
    )
    return {
        "input_ids": [[int(value) for value in _single_row(base, "input_ids")]],
        "attention_mask": [
            [bool(value) for value in _single_row(base, "attention_mask")]
        ],
        "loss_mask": [[int(value) for value in _single_row(base, "loss_mask")]],
        "logprobs": [[float(value) for value in _single_row(base, "logprobs")]],
        "versions": [[int(value) for value in _single_row(base, "versions")]],
        "rewards": [float(value) for value in rewards],
    }


def build_areal_external_advantage_batch(
    joint_credit_record: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
) -> dict[str, object]:
    """Convert S credit to the post-compute_advantages AReaL actor contract."""

    try:
        audit = validate_frozen_joint_credit_alignment(
            joint_credit_record,
            active_joint_version=active_joint_version,
        )
    except ValueError as exc:
        raise ArealExternalAdvantageBatchError(str(exc)) from exc
    evidence = joint_credit_record.get("evidence_scope")
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("policy_advantages_aligned") is True
        and evidence.get("policy_optimizer_update") is False
        and evidence.get("harness_optimizer_update") is False,
        "S evidence does not permit a fresh Policy candidate update",
    )
    admissions = joint_credit_record.get("admissions")
    _require(isinstance(admissions, Mapping), "S admissions are missing")
    export_style = admissions.get("policy_export_style")
    _require(
        export_style in {"individual", "concat"},
        "S Policy export style is invalid",
    )
    policy_samples = joint_credit_record.get("policy_samples")
    _require(
        isinstance(policy_samples, list) and bool(policy_samples),
        "S Policy samples are missing",
    )
    inference_versions = {
        credit.get("inference_engine_version")
        for sample in policy_samples
        if isinstance(sample, Mapping)
        for credit in sample.get("decision_credits", ())
        if isinstance(credit, Mapping)
    }
    _require(
        len(inference_versions) == 1
        and all(type(value) is int and value >= 0 for value in inference_versions),
        "S Policy decisions must share one non-negative inference engine version",
    )
    inference_engine_version = next(iter(inference_versions))
    samples: list[dict[str, object]] = []
    for policy_sample in policy_samples:
        _require(
            isinstance(policy_sample, Mapping), "S Policy sample must be an object"
        )
        sample_id = policy_sample.get("sample_id")
        _require(
            isinstance(sample_id, str) and sample_id,
            "S Policy sample ID is missing",
        )
        samples.append(
            {
                "sample_id": sample_id,
                "rollout_tensor_dict": _build_rollout_tensor_dict(policy_sample),
                "tensor_dict": _build_update_tensor_dict(policy_sample),
            }
        )
    record: dict[str, object] = {
        "schema_version": AREAL_EXTERNAL_ADVANTAGE_BATCH_SCHEMA,
        "identity": {
            "episode_id": audit["episode_id"],
            "joint_version_id": active_joint_version.version_id,
            "source_joint_credit_sha256": joint_credit_record["record_sha256"],
            "policy_export_style": export_style,
            "inference_engine_version": inference_engine_version,
        },
        "joint_version": asdict(active_joint_version),
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "trainable_token_count": sum(
                sum(sample["tensor_dict"]["loss_mask"][0]) for sample in samples
            ),
        },
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }
    record["record_sha256"] = _record_sha256(record)
    validate_areal_external_advantage_batch(
        record,
        active_joint_version=active_joint_version,
    )
    return record


def validate_areal_external_advantage_batch(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
) -> ValidatedArealExternalAdvantageBatch:
    expected_fields = {
        "schema_version",
        "identity",
        "joint_version",
        "samples",
        "summary",
        "evidence_scope",
        "record_sha256",
    }
    _require(set(record) == expected_fields, "Policy PPO batch field set differs")
    _require(
        record.get("schema_version") == AREAL_EXTERNAL_ADVANTAGE_BATCH_SCHEMA,
        "unknown Policy PPO batch schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "Policy PPO batch hash mismatch",
    )
    joint_version = _joint_version(record.get("joint_version"))
    if active_joint_version is not None:
        _require(
            joint_version == active_joint_version,
            "Policy PPO batch differs from lag-zero active JointVersion",
        )
    identity = _exact_mapping(
        record.get("identity"),
        {
            "episode_id",
            "joint_version_id",
            "source_joint_credit_sha256",
            "policy_export_style",
            "inference_engine_version",
        },
        "Policy PPO batch identity",
    )
    episode_id = identity.get("episode_id")
    _require(isinstance(episode_id, str) and episode_id, "episode ID is missing")
    _require(
        identity.get("joint_version_id") == joint_version.version_id,
        "Policy PPO batch identity differs from JointVersion",
    )
    source_sha256 = identity.get("source_joint_credit_sha256")
    _require(_is_sha256(source_sha256), "source S record SHA-256 is invalid")
    export_style = identity.get("policy_export_style")
    _require(
        export_style in {"individual", "concat"},
        "Policy PPO export style is invalid",
    )
    inference_engine_version = identity.get("inference_engine_version")
    _require(
        type(inference_engine_version) is int and inference_engine_version >= 0,
        "Policy PPO inference engine version is invalid",
    )
    samples = record.get("samples")
    _require(isinstance(samples, list) and bool(samples), "Policy PPO samples missing")
    trainable_token_count = 0
    sample_ids: set[str] = set()
    for sample in samples:
        sample = _exact_mapping(
            sample,
            {"sample_id", "rollout_tensor_dict", "tensor_dict"},
            "Policy PPO sample",
        )
        sample_id = sample.get("sample_id")
        _require(
            isinstance(sample_id, str) and sample_id and sample_id not in sample_ids,
            "Policy PPO sample IDs must be unique non-empty strings",
        )
        sample_ids.add(sample_id)
        rollout_tensors = _exact_mapping(
            sample.get("rollout_tensor_dict"),
            _ROLLOUT_TENSOR_FIELDS,
            "Policy PPO rollout tensor_dict",
        )
        tensors = _exact_mapping(
            sample.get("tensor_dict"),
            _TENSOR_FIELDS,
            "Policy PPO tensor_dict",
        )
        input_ids = _single_row(tensors, "input_ids")
        attention_mask = _single_row(tensors, "attention_mask")
        loss_mask = _single_row(tensors, "loss_mask")
        logprobs = _single_row(tensors, "logprobs")
        prox_logp = _single_row(tensors, "prox_logp")
        versions = _single_row(tensors, "versions")
        advantages = _single_row(tensors, "advantages")
        returns = _single_row(tensors, "returns")
        rewards = tensors.get("rewards")
        rollout_input_ids = _single_row(rollout_tensors, "input_ids")
        rollout_attention_mask = _single_row(rollout_tensors, "attention_mask")
        rollout_loss_mask = _single_row(rollout_tensors, "loss_mask")
        rollout_logprobs = _single_row(rollout_tensors, "logprobs")
        rollout_versions = _single_row(rollout_tensors, "versions")
        rollout_rewards = rollout_tensors.get("rewards")
        length = len(input_ids)
        _require(
            length > 1
            and all(
                len(row) == length
                for row in (
                    attention_mask,
                    loss_mask,
                    logprobs,
                    prox_logp,
                    versions,
                    advantages,
                    returns,
                )
            ),
            "Policy PPO tensor lengths differ",
        )
        _require(
            all(type(value) is int and value >= 0 for value in input_ids),
            "Policy PPO input IDs must be non-negative integers",
        )
        _require(
            all(type(value) is bool and value for value in attention_mask),
            "unpadded Policy PPO attention mask must be true",
        )
        _require(
            all(type(value) is int and value in (0, 1) for value in loss_mask)
            and bool(sum(loss_mask))
            and loss_mask[-1] == 0,
            "Policy PPO loss mask is not a causal-shifted binary mask",
        )
        _require(
            all(_is_finite_number(value) for value in logprobs)
            and logprobs == prox_logp
            and all(
                float(value) <= 0.0 for value, mask in zip(logprobs, loss_mask) if mask
            ),
            "Policy PPO old/proximal log-probabilities are invalid",
        )
        _require(
            all(type(value) is int and value >= -1 for value in versions),
            "Policy PPO versions are invalid",
        )
        _require(
            all(
                version == inference_engine_version
                for version, mask in zip(rollout_versions, rollout_loss_mask)
                if mask
            ),
            "Policy PPO trainable tokens differ from S inference engine version",
        )
        _require(
            all(_is_finite_number(value) for value in advantages)
            and advantages == returns
            and all(
                float(value) == 0.0
                for value, mask in zip(advantages, loss_mask)
                if not mask
            ),
            "Policy PPO advantages differ from the causal loss mask",
        )
        _require(
            isinstance(rewards, list)
            and len(rewards) == 1
            and _is_finite_number(rewards[0]),
            "Policy PPO reward is invalid",
        )
        expected_shifted_mask = [
            int(value) for value in _shift_left(rollout_loss_mask, 0)
        ]
        expected_shifted_logprobs = [
            float(value) * mask
            for value, mask in zip(
                _shift_left(rollout_logprobs, 0.0),
                expected_shifted_mask,
            )
        ]
        _require(
            rollout_input_ids == input_ids
            and rollout_attention_mask == attention_mask
            and rollout_versions == versions
            and rollout_rewards == rewards
            and expected_shifted_mask == loss_mask
            and expected_shifted_logprobs == logprobs,
            "Policy PPO tensors differ from pinned AReaL causal shift",
        )
        trainable_token_count += sum(loss_mask)
    if export_style == "concat":
        _require(len(samples) == 1, "concat Policy PPO batch requires one sample")
    summary = _exact_mapping(
        record.get("summary"),
        {"sample_count", "trainable_token_count"},
        "Policy PPO batch summary",
    )
    _require(
        summary
        == {
            "sample_count": len(samples),
            "trainable_token_count": trainable_token_count,
        },
        "Policy PPO batch summary differs from tensors",
    )
    _require(
        record.get("evidence_scope") == _EVIDENCE_SCOPE,
        "Policy PPO adapter evidence differs from contract",
    )
    return ValidatedArealExternalAdvantageBatch(
        joint_version=joint_version,
        episode_id=episode_id,
        source_joint_credit_sha256=str(source_sha256),
        export_style=str(export_style),
        inference_engine_version=inference_engine_version,
        samples=tuple(samples),
        trainable_token_count=trainable_token_count,
        record_sha256=str(record["record_sha256"]),
    )


def validate_areal_policy_optimizer_source(
    record: Mapping[str, object],
    *,
    source_joint_credit_record: Mapping[str, object],
    active_joint_version: JointVersion,
) -> ValidatedArealExternalAdvantageBatch:
    """Prove that an optimizer batch was derived from the supplied S record.

    A digest carried by a self-contained optimizer batch is only an identity
    claim.  The production update boundary therefore revalidates the original
    Q/R/S record, deterministically rebuilds the adapter output, and requires
    exact equality before any actor state may change.
    """

    validated = validate_areal_external_advantage_batch(
        record,
        active_joint_version=active_joint_version,
    )
    expected = build_areal_external_advantage_batch(
        source_joint_credit_record,
        active_joint_version=active_joint_version,
    )
    _require(
        validated.source_joint_credit_sha256
        == source_joint_credit_record.get("record_sha256"),
        "Policy PPO batch source differs from the supplied S record",
    )
    _require(
        _canonical_json(record) == _canonical_json(expected),
        "Policy PPO batch does not exactly derive from the supplied S record",
    )
    return validated


def validate_m0_areal_actor_config(actor: object) -> None:
    """Fail closed unless the actor uses the audited external-advantage mode."""

    config = getattr(actor, "config", None)
    _require(config is not None, "AReaL actor config is unavailable")
    exact_requirements = {
        "adv_norm": None,
        "c_clip": None,
        "disable_dropout": True,
        "eps_clip_higher": None,
        "importance_sampling_level": "token",
        "is_critic": False,
        "log_agent_stats": False,
        "m2_threshold": None,
        "mask_no_eos_with_zero": False,
        "overlong_reward_penalty": False,
        "ppo_n_minibatches": 1,
        "recompute_logprob": False,
        "rejection_sampling": None,
        "reward_bias": 0.0,
        "reward_norm": None,
        "reward_scaling": 1.0,
        "use_cispo_loss": False,
        "use_decoupled_loss": False,
        "use_sapo_loss": False,
    }
    for field, expected in exact_requirements.items():
        _require(
            hasattr(config, field) and getattr(config, field) == expected,
            f"AReaL actor {field} must be {expected!r} for M0",
        )
    kl_ctl = getattr(config, "kl_ctl", None)
    _require(
        _is_finite_number(kl_ctl) and float(kl_ctl) == 0.0,
        "AReaL actor kl_ctl must be 0 for external advantages",
    )
    optimizer = getattr(config, "optimizer", None)
    _require(optimizer is not None, "AReaL actor optimizer config is unavailable")
    _require(
        getattr(optimizer, "lr_scheduler_type", None) == "constant",
        "AReaL actor lr_scheduler_type must be 'constant' for M0 recovery",
    )
    learning_rate = getattr(optimizer, "lr", None)
    _require(
        _is_finite_number(learning_rate) and float(learning_rate) > 0.0,
        "AReaL actor learning rate must be finite and positive",
    )


def _actor_version(actor: object) -> int:
    get_version = getattr(actor, "get_version", None)
    _require(callable(get_version), "AReaL actor get_version is unavailable")
    version = get_version()
    _require(
        type(version) is int and version >= 0,
        "AReaL actor version must be a non-negative integer",
    )
    return version


def materialize_areal_ppo_update_tensors(
    record: Mapping[str, object],
    *,
    actor: object,
    active_joint_version: JointVersion,
    device: str | Any,
    sample_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Run native advantage preparation, then inject audited S advantages.

    This function deliberately stops before ``actor.ppo_update()``. Its input
    record therefore cannot claim that an optimizer update occurred.
    """

    validated = validate_areal_external_advantage_batch(
        record,
        active_joint_version=active_joint_version,
    )
    if sample_indices is None:
        selected_indices = tuple(range(len(validated.samples)))
    else:
        _require(
            isinstance(sample_indices, Sequence)
            and not isinstance(sample_indices, (str, bytes, bytearray)),
            "AReaL PPO sample indices must be a sequence",
        )
        selected_indices = tuple(sample_indices)
        _require(
            bool(selected_indices)
            and all(type(index) is int for index in selected_indices)
            and len(set(selected_indices)) == len(selected_indices)
            and all(0 <= index < len(validated.samples) for index in selected_indices),
            "AReaL PPO sample indices are empty, duplicated, or out of range",
        )
    selected_samples = tuple(validated.samples[index] for index in selected_indices)
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - local dependency gate
        raise ArealExternalAdvantageBatchError(
            "torch is required to materialize AReaL PPO tensors"
        ) from exc
    validate_m0_areal_actor_config(actor)
    _require(
        _actor_version(actor) == validated.inference_engine_version,
        "AReaL actor version differs from S inference engine version",
    )
    compute_advantages = getattr(actor, "compute_advantages", None)
    _require(
        callable(compute_advantages),
        "AReaL actor compute_advantages is unavailable",
    )
    rollout_dtypes = {
        "input_ids": torch.long,
        "attention_mask": torch.bool,
        "loss_mask": torch.long,
        "logprobs": torch.float32,
        "versions": torch.long,
        "rewards": torch.float32,
    }
    expected_dtypes = {
        **rollout_dtypes,
        "prox_logp": torch.float32,
        "advantages": torch.float32,
        "returns": torch.float32,
    }
    rollout_batch: list[dict[str, Any]] = []
    for sample in selected_samples:
        rollout_tensors = sample["rollout_tensor_dict"]
        rollout_batch.append(
            {
                field: torch.tensor(
                    value,
                    dtype=rollout_dtypes[field],
                    device=device,
                )
                for field, value in rollout_tensors.items()
            }
        )
    native_batch = compute_advantages(rollout_batch)
    _require(
        isinstance(native_batch, list) and len(native_batch) == len(selected_samples),
        "AReaL compute_advantages returned an invalid batch",
    )
    prepared: list[dict[str, Any]] = []
    for global_index, native, sample in zip(
        selected_indices, native_batch, selected_samples, strict=True
    ):
        _require(
            isinstance(native, dict),
            f"AReaL compute_advantages sample {global_index} is not a tensor dict",
        )
        expected = {
            field: torch.tensor(value, dtype=expected_dtypes[field], device=device)
            for field, value in sample["tensor_dict"].items()
        }
        for field in (
            "input_ids",
            "attention_mask",
            "loss_mask",
            "logprobs",
            "prox_logp",
            "versions",
            "rewards",
        ):
            actual = native.get(field)
            target = expected[field]
            _require(
                isinstance(actual, torch.Tensor) and actual.shape == target.shape,
                f"native AReaL {field} shape differs from S sample {global_index}",
            )
            if actual.dtype.is_floating_point or target.dtype.is_floating_point:
                matches = torch.allclose(
                    actual.to(torch.float64),
                    target.to(torch.float64),
                    rtol=0.0,
                    atol=1e-7,
                )
            else:
                matches = torch.equal(actual, target)
            _require(
                bool(matches),
                f"native AReaL {field} differs from audited S sample {global_index}",
            )
        _require(
            isinstance(native.get("kl_rewards"), torch.Tensor)
            and isinstance(native.get("tot_rewards"), torch.Tensor),
            "native AReaL reward statistics are missing",
        )
        native["advantages"] = expected["advantages"]
        native["returns"] = expected["returns"]
        prepared.append(native)
    return prepared
