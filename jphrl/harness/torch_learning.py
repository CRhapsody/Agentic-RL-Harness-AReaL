from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from jphrl.paths import require_outside_repository
from jphrl.trajectory.joint_credit_alignment import (
    JointCreditAlignmentError,
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.multi_s_frozen_training_batch import (
    ValidatedMultiSFrozenTrainingBatch,
    iter_member_s_records,
    multi_s_source_binding,
    validate_multi_s_source_binding,
)
from jphrl.trajectory.schema import JointVersion

from .controller import HarnessDecision, HarnessState
from .spec import HarnessAction

ACTION_IDS = tuple(action.value for action in HarnessAction)
POLICY_SCHEMA_VERSION = "torch-harness-categorical-v1"
CHECKPOINT_SCHEMA_VERSION = "jph.torch-harness-checkpoint.v2"
MULTI_S_CHECKPOINT_SCHEMA_VERSION = "jph.torch-harness-multi-s-checkpoint.v1"
ROLLOUT_CHECKPOINT_SCHEMA_VERSION = "jph.torch-harness-rollout-checkpoint.v1"
HARNESS_CANDIDATE_SCHEMA_VERSION = "jph.torch-harness-candidate.v2"
MULTI_S_HARNESS_CANDIDATE_SCHEMA_VERSION = "jph.torch-harness-multi-s-candidate.v1"
_STATE_FEATURE_SCHEMA_VERSION = "jph.harness-state-features.v1"
_STATE_FEATURE_DIM = 32
_SECRET_FIELD_NAMES = {
    "admin_api_key",
    "api_key",
    "authorization",
    "session_api_key",
}
_HARNESS_EVIDENCE_SCOPE = {
    "source_joint_credit_validated": True,
    "lag_zero_joint_version_validated": True,
    "behavior_state_mask_logits_replayed": True,
    "clipped_ppo_update_invoked": True,
    "finite_nonzero_gradient_observed": True,
    "parameter_digest_changed": True,
    "model_state_persisted": True,
    "optimizer_state_persisted": True,
    "rng_state_persisted": True,
    "policy_optimizer_update": False,
    "harness_optimizer_update": True,
}
_MULTI_S_HARNESS_EVIDENCE_SCOPE = {
    **_HARNESS_EVIDENCE_SCOPE,
    "validated_multi_s_batch_consumed": True,
    "every_member_s_revalidated_lag_zero": True,
    "one_adam_step_for_all_members": True,
}
_CHECKPOINT_METRIC_FIELDS = {
    "batch_size",
    "effective_batch_size",
    "loss",
    "mean_ratio",
    "clip_fraction",
    "gradient_norm",
    "parameter_digest_before",
    "parameter_digest_after",
}


class TorchHarnessLearningError(ValueError):
    """Raised when a Harness candidate cannot be trained without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TorchHarnessLearningError(message)


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
        raise TorchHarnessLearningError(
            "Harness optimizer input is not finite canonical JSON"
        ) from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_no_secret_fields(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(
                str(key).lower() not in _SECRET_FIELD_NAMES,
                f"credential field cannot enter Harness optimizer: {path}.{key}",
            )
            _assert_no_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secret_fields(item, f"{path}[{index}]")


def _state_payload(state: HarnessState) -> dict[str, object]:
    _require(type(state) is HarnessState, "Harness state must use the frozen schema")
    for name in (
        "turn",
        "remaining_tool_calls",
        "remaining_model_retries",
        "context_chars",
    ):
        value = getattr(state, name)
        _require(
            type(value) is int and value >= 0,
            f"Harness state {name} must be a non-negative integer",
        )
    _require(
        state.last_error is None or isinstance(state.last_error, str),
        "Harness state last_error must be null or a string",
    )
    _require(
        type(state.retrieval_hit) is bool,
        "Harness state retrieval_hit must be boolean",
    )
    _require(
        isinstance(state.verifier_status, str) and bool(state.verifier_status),
        "Harness state verifier_status must be non-empty",
    )
    _require(
        isinstance(state.task_domain, str) and bool(state.task_domain),
        "Harness state task_domain must be non-empty",
    )
    return asdict(state)


def _add_hashed_feature(
    features: list[float],
    *,
    namespace: str,
    value: str,
    offset: int,
    width: int,
) -> None:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode()).digest()
    index = offset + int.from_bytes(digest[:4], "big") % width
    sign = 1.0 if digest[4] & 1 else -1.0
    features[index] += sign


def encode_harness_state(state: HarnessState) -> tuple[float, ...]:
    """Encode the frozen state with a stable, answer-free feature schema."""

    _state_payload(state)
    features = [0.0] * _STATE_FEATURE_DIM
    features[0] = math.tanh(state.turn / 8.0)
    features[1] = math.tanh(state.remaining_tool_calls / 4.0)
    features[2] = math.tanh(state.remaining_model_retries / 4.0)
    features[3] = math.tanh(math.log1p(state.context_chars) / 12.0)
    features[4] = 1.0 if state.retrieval_hit else 0.0
    features[5] = 1.0 if state.last_error is not None else 0.0
    _add_hashed_feature(
        features,
        namespace="last_error",
        value=state.last_error or "<none>",
        offset=6,
        width=8,
    )
    _add_hashed_feature(
        features,
        namespace="verifier_status",
        value=state.verifier_status,
        offset=14,
        width=8,
    )
    _add_hashed_feature(
        features,
        namespace="task_domain",
        value=state.task_domain,
        offset=22,
        width=8,
    )
    features[30] = 1.0
    features[31] = 1.0 if state.turn % 2 else -1.0
    return tuple(features)


def _normalize_action_mask(action_mask: Sequence[bool] | None) -> tuple[bool, ...]:
    if action_mask is None:
        return (True,) * len(ACTION_IDS)
    result = tuple(action_mask)
    _require(
        len(result) == len(ACTION_IDS)
        and all(type(value) is bool for value in result)
        and any(result),
        "Harness action mask must be a non-empty boolean five-action mask",
    )
    return result


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous().view(torch.uint8)
    return bytes(tensor.untyped_storage())


class TorchHarnessPolicy(nn.Module):
    """Small batched categorical policy for production Harness decisions.

    The action order and state features are schema-versioned. Sampling uses an
    owned CPU generator, so checkpoint restoration does not depend on ambient
    process RNG state. ``choose`` emits the complete behavior record consumed by
    ``EpisodeTrace`` and the R/S admission path.
    """

    schema_version = POLICY_SCHEMA_VERSION
    state_feature_schema_version = _STATE_FEATURE_SCHEMA_VERSION
    action_ids = ACTION_IDS

    def __init__(self, *, seed: int = 0, hidden_size: int = 32) -> None:
        super().__init__()
        _require(type(seed) is int and seed >= 0, "Harness seed must be non-negative")
        _require(
            type(hidden_size) is int and hidden_size > 0,
            "Harness hidden size must be positive",
        )
        self.seed = seed
        self.hidden_size = hidden_size
        self.update_step = 0
        self.sample_count = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(_STATE_FEATURE_DIM, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, len(ACTION_IDS)),
            )
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def forward(self, state_features: Tensor) -> Tensor:
        _require(
            state_features.ndim == 2 and state_features.shape[-1] == _STATE_FEATURE_DIM,
            "Harness feature tensor has the wrong shape",
        )
        return self.network(state_features)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema_version.encode("utf-8"))
        digest.update(self.state_feature_schema_version.encode("utf-8"))
        digest.update(_canonical_json(ACTION_IDS))
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(_canonical_json(list(value.shape)))
            digest.update(_tensor_bytes(value))
        return digest.hexdigest()

    @property
    def version(self) -> str:
        return (
            f"{self.schema_version}-step{self.update_step:06d}-"
            f"{self.parameter_digest[:16]}"
        )

    def features_for(self, states: Sequence[HarnessState]) -> Tensor:
        _require(bool(states), "Harness policy requires at least one state")
        rows = [encode_harness_state(state) for state in states]
        dtype = next(self.parameters()).dtype
        return torch.tensor(rows, dtype=dtype, device=self.device)

    def logits_for(self, states: Sequence[HarnessState]) -> Tensor:
        return self(self.features_for(states))

    def choose(
        self,
        state: HarnessState,
        *,
        action_mask: Sequence[bool] | None = None,
        harness_loss_mask: int = 1,
    ) -> HarnessDecision:
        mask = _normalize_action_mask(action_mask)
        _require(
            type(harness_loss_mask) is int and harness_loss_mask in (0, 1),
            "Harness loss mask must be integer 0 or 1",
        )
        with torch.no_grad():
            logits = self.logits_for([state])[0].detach().cpu()
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            masked_logits = logits.masked_fill(~mask_tensor, -torch.inf)
            logprobs = torch.log_softmax(masked_logits, dim=-1)
            probabilities = torch.exp(logprobs)
            selected = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=self._generator,
                ).item()
            )
        serialized_logits = tuple(float(value) for value in logits.tolist())
        allowed_logits = [
            value for value, allowed in zip(serialized_logits, mask) if allowed
        ]
        maximum = max(allowed_logits)
        log_normalizer = maximum + math.log(
            sum(math.exp(value - maximum) for value in allowed_logits)
        )
        old_logprob = serialized_logits[selected] - log_normalizer
        state_digest = hashlib.sha256(
            _canonical_json(_state_payload(state))
        ).hexdigest()
        decision = HarnessDecision(
            decision_id=(
                f"torch-step-{self.update_step}-sample-{self.sample_count}-"
                f"{state_digest[:12]}"
            ),
            action=HarnessAction(ACTION_IDS[selected]),
            old_harness_logprob=old_logprob,
            controller_version=self.version,
            action_ids=ACTION_IDS,
            action_mask=mask,
            pre_mask_logits=serialized_logits,
            harness_loss_mask=harness_loss_mask,
        )
        self.sample_count += 1
        return decision

    def clone(self) -> TorchHarnessPolicy:
        candidate = type(self)(seed=self.seed, hidden_size=self.hidden_size)
        candidate.load_state_dict(deepcopy(self.state_dict()))
        candidate.to(self.device)
        candidate.update_step = self.update_step
        candidate.sample_count = self.sample_count
        candidate._generator.set_state(self._generator.get_state().clone())
        candidate.train(self.training)
        return candidate


def _adam_step(value: object, *, label: str) -> int:
    if isinstance(value, Tensor):
        _require(
            value.numel() == 1 and bool(torch.isfinite(value).all()),
            f"{label} must be one finite scalar",
        )
        numeric = float(value.detach().cpu().item())
    else:
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be numeric",
        )
        numeric = float(value)
    _require(
        math.isfinite(numeric) and numeric >= 0.0 and numeric.is_integer(),
        f"{label} must be a non-negative integer",
    )
    return int(numeric)


def _validate_adam_optimizer_state(
    policy: TorchHarnessPolicy,
    optimizer: torch.optim.Adam,
    *,
    allow_uninitialized_step_zero: bool,
) -> int:
    """Bind one independent Adam state to every parameter and policy step."""

    _require(
        type(policy) is TorchHarnessPolicy,
        "Harness Adam state requires TorchHarnessPolicy",
    )
    _require(
        type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1,
        "Harness optimizer must be one independent Adam parameter group",
    )
    _require(
        type(policy.update_step) is int and policy.update_step >= 0,
        "Harness policy update step is invalid",
    )
    parameters = tuple(policy.parameters())
    group = optimizer.param_groups[0]
    group_parameters = tuple(group.get("params", ()))
    _require(
        len(group_parameters) == len(parameters)
        and all(
            actual is expected for actual, expected in zip(group_parameters, parameters)
        ),
        "Harness Adam parameters differ from the policy",
    )
    learning_rate = group.get("lr")
    _require(
        isinstance(learning_rate, (int, float))
        and not isinstance(learning_rate, bool)
        and math.isfinite(float(learning_rate))
        and float(learning_rate) > 0.0,
        "Harness Adam learning rate must be finite and positive",
    )
    _require(
        not bool(group.get("amsgrad", False)),
        "Harness Adam checkpoint unexpectedly enables AMSGrad",
    )

    if not optimizer.state:
        _require(
            allow_uninitialized_step_zero and policy.update_step == 0,
            "Harness policy after step zero requires persisted Adam state",
        )
        return 0

    _require(
        policy.update_step > 0,
        "initialized Harness Adam state cannot belong to step zero",
    )
    _require(
        {id(parameter) for parameter in optimizer.state}
        == {id(parameter) for parameter in parameters},
        "Harness Adam state does not cover every policy parameter exactly once",
    )
    observed_steps: set[int] = set()
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        _require(
            isinstance(state, Mapping)
            and set(state) == {"step", "exp_avg", "exp_avg_sq"},
            "Harness Adam parameter state field set differs",
        )
        observed_steps.add(_adam_step(state["step"], label="Harness Adam step"))
        for name in ("exp_avg", "exp_avg_sq"):
            moment = state[name]
            _require(
                isinstance(moment, Tensor)
                and moment.shape == parameter.shape
                and moment.dtype == parameter.dtype
                and bool(torch.isfinite(moment).all()),
                f"Harness Adam {name} differs from its policy parameter",
            )
    _require(
        observed_steps == {policy.update_step},
        "Harness Adam step differs from the policy update step",
    )
    return policy.update_step


def _json_tensor_record(value: Tensor) -> dict[str, object]:
    tensor = value.detach().cpu().contiguous()
    _require(
        not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
        "Harness rollout checkpoint tensor is non-finite",
    )
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "values": tensor.tolist(),
    }


def build_torch_harness_rollout_checkpoint(
    policy: TorchHarnessPolicy,
) -> dict[str, object]:
    """Serialize the behavior policy needed to replay its next sampled action.

    This JSON-only checkpoint is intentionally model/RNG-only. Optimizer state
    remains in the production ``.pt`` checkpoint used by U/W.
    """

    _require(
        type(policy) is TorchHarnessPolicy,
        "rollout checkpoint requires TorchHarnessPolicy",
    )
    record: dict[str, object] = {
        "schema_version": ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
        "policy_schema_version": policy.schema_version,
        "state_feature_schema_version": policy.state_feature_schema_version,
        "action_ids": list(ACTION_IDS),
        "policy_config": {
            "seed": policy.seed,
            "hidden_size": policy.hidden_size,
        },
        "policy_metadata": {
            "update_step": policy.update_step,
            "sample_count": policy.sample_count,
            "version": policy.version,
            "parameter_digest": policy.parameter_digest,
        },
        "model_state": {
            name: _json_tensor_record(value)
            for name, value in sorted(policy.state_dict().items())
        },
        "policy_generator_state": policy._generator.get_state().cpu().tolist(),
    }
    _assert_no_secret_fields(record)
    record["record_sha256"] = _record_sha256(record)
    _canonical_json(record)
    return record


def _dtype_from_record(value: object) -> torch.dtype:
    _require(isinstance(value, str), "rollout checkpoint dtype must be a string")
    name = value.removeprefix("torch.")
    dtype = getattr(torch, name, None)
    _require(
        isinstance(dtype, torch.dtype),
        "rollout checkpoint tensor dtype is unsupported",
    )
    return dtype


def load_torch_harness_rollout_checkpoint(
    record: Mapping[str, object],
    *,
    device: str | torch.device = "cpu",
) -> TorchHarnessPolicy:
    """Restore and self-validate a JSON behavior checkpoint for exact replay."""

    required = {
        "schema_version",
        "policy_schema_version",
        "state_feature_schema_version",
        "action_ids",
        "policy_config",
        "policy_metadata",
        "model_state",
        "policy_generator_state",
        "record_sha256",
    }
    _require(set(record) == required, "rollout checkpoint field set differs")
    _assert_no_secret_fields(record)
    _canonical_json(record)
    _require(
        record.get("record_sha256") == _record_sha256(record)
        and record.get("schema_version") == ROLLOUT_CHECKPOINT_SCHEMA_VERSION
        and record.get("policy_schema_version") == POLICY_SCHEMA_VERSION
        and record.get("state_feature_schema_version") == _STATE_FEATURE_SCHEMA_VERSION
        and tuple(record.get("action_ids", ())) == ACTION_IDS,
        "rollout checkpoint schema differs",
    )
    config = record.get("policy_config")
    metadata = record.get("policy_metadata")
    model_state = record.get("model_state")
    generator_state = record.get("policy_generator_state")
    _require(
        isinstance(config, Mapping)
        and set(config) == {"seed", "hidden_size"}
        and type(config.get("seed")) is int
        and config["seed"] >= 0
        and type(config.get("hidden_size")) is int
        and config["hidden_size"] > 0,
        "rollout checkpoint policy config differs",
    )
    _require(
        isinstance(metadata, Mapping)
        and set(metadata)
        == {"update_step", "sample_count", "version", "parameter_digest"}
        and type(metadata.get("update_step")) is int
        and metadata["update_step"] >= 0
        and type(metadata.get("sample_count")) is int
        and metadata["sample_count"] >= 0
        and isinstance(metadata.get("version"), str)
        and _is_sha256(metadata.get("parameter_digest")),
        "rollout checkpoint policy metadata differs",
    )
    _require(
        isinstance(model_state, Mapping),
        "rollout checkpoint model state is missing",
    )
    _require(
        isinstance(generator_state, list)
        and bool(generator_state)
        and all(type(value) is int and 0 <= value <= 255 for value in generator_state),
        "rollout checkpoint policy RNG is invalid",
    )
    policy = TorchHarnessPolicy(
        seed=config["seed"],
        hidden_size=config["hidden_size"],
    )
    expected_state = policy.state_dict()
    _require(
        set(model_state) == set(expected_state),
        "rollout checkpoint model parameter names differ",
    )
    restored_state: dict[str, Tensor] = {}
    for name, expected in expected_state.items():
        tensor_record = model_state[name]
        _require(
            isinstance(tensor_record, Mapping)
            and set(tensor_record) == {"dtype", "shape", "values"},
            f"rollout checkpoint tensor {name} field set differs",
        )
        shape = tensor_record.get("shape")
        _require(
            isinstance(shape, list)
            and all(type(dimension) is int and dimension >= 0 for dimension in shape)
            and shape == list(expected.shape),
            f"rollout checkpoint tensor {name} shape differs",
        )
        dtype = _dtype_from_record(tensor_record.get("dtype"))
        _require(
            dtype == expected.dtype, f"rollout checkpoint tensor {name} dtype differs"
        )
        try:
            tensor = torch.tensor(tensor_record.get("values"), dtype=dtype).reshape(
                shape
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TorchHarnessLearningError(
                f"rollout checkpoint tensor {name} values are invalid"
            ) from exc
        _require(
            not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
            f"rollout checkpoint tensor {name} is non-finite",
        )
        restored_state[name] = tensor
    policy.load_state_dict(restored_state, strict=True)
    policy.update_step = metadata["update_step"]
    policy.sample_count = metadata["sample_count"]
    policy._generator.set_state(torch.tensor(generator_state, dtype=torch.uint8))
    _require(
        policy.parameter_digest == metadata["parameter_digest"]
        and policy.version == metadata["version"],
        "rollout checkpoint policy identity differs from restored state",
    )
    return policy.to(device)


@dataclass(frozen=True)
class TorchHarnessUpdateEvidence:
    transaction_id: str
    source_joint_credit_sha256: str
    parent_joint_version: JointVersion
    parent_joint_version_id: str
    behavior_version: str
    candidate_version: str
    parameter_digest_before: str
    parameter_digest_after: str
    optimizer_step_before: int
    optimizer_step_after: int
    batch_size: int
    effective_batch_size: int
    loss: float
    mean_ratio: float
    clip_fraction: float
    gradient_norm: float
    checkpoint_path: str
    checkpoint_sha256: str

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["schema_version"] = HARNESS_CANDIDATE_SCHEMA_VERSION
        record["evidence_scope"] = dict(_HARNESS_EVIDENCE_SCOPE)
        record["record_sha256"] = _record_sha256(record)
        validate_torch_harness_update_evidence(
            record,
            active_joint_version=self.parent_joint_version,
        )
        return record


@dataclass(frozen=True)
class TorchHarnessUpdateResult:
    candidate_policy: TorchHarnessPolicy
    candidate_optimizer: TorchHarnessOptimizer
    evidence: TorchHarnessUpdateEvidence


@dataclass(frozen=True)
class TorchHarnessMultiSUpdateEvidence:
    transaction_id: str
    source_joint_credit_sha256: str
    source_binding: Mapping[str, object]
    parent_joint_version: JointVersion
    parent_joint_version_id: str
    behavior_version: str
    candidate_version: str
    parameter_digest_before: str
    parameter_digest_after: str
    optimizer_step_before: int
    optimizer_step_after: int
    batch_size: int
    effective_batch_size: int
    loss: float
    mean_ratio: float
    clip_fraction: float
    gradient_norm: float
    checkpoint_path: str
    checkpoint_sha256: str

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["schema_version"] = MULTI_S_HARNESS_CANDIDATE_SCHEMA_VERSION
        record["evidence_scope"] = dict(_MULTI_S_HARNESS_EVIDENCE_SCOPE)
        record["record_sha256"] = _record_sha256(record)
        validate_torch_harness_multi_s_update_evidence(
            record,
            active_joint_version=self.parent_joint_version,
        )
        return record


@dataclass(frozen=True)
class TorchHarnessMultiSUpdateResult:
    candidate_policy: TorchHarnessPolicy
    candidate_optimizer: TorchHarnessOptimizer
    evidence: TorchHarnessMultiSUpdateEvidence


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_torch_harness_update_evidence(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
    require_checkpoint: bool = True,
) -> TorchHarnessUpdateEvidence | TorchHarnessMultiSUpdateEvidence:
    """Revalidate a persisted real Harness optimizer receipt."""

    if record.get("schema_version") == MULTI_S_HARNESS_CANDIDATE_SCHEMA_VERSION:
        return validate_torch_harness_multi_s_update_evidence(
            record,
            active_joint_version=active_joint_version,
            require_checkpoint=require_checkpoint,
        )
    _assert_no_secret_fields(record)
    evidence_fields = set(TorchHarnessUpdateEvidence.__dataclass_fields__)
    _require(
        set(record)
        == evidence_fields | {"schema_version", "evidence_scope", "record_sha256"},
        "Harness candidate evidence field set differs",
    )
    _require(
        record.get("schema_version") == HARNESS_CANDIDATE_SCHEMA_VERSION,
        "unknown Harness candidate evidence schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "Harness candidate evidence hash mismatch",
    )
    transaction_id = record.get("transaction_id")
    _require(
        isinstance(transaction_id, str) and bool(transaction_id),
        "Harness candidate transaction ID is missing",
    )
    raw_joint_version = record.get("parent_joint_version")
    _require(
        isinstance(raw_joint_version, Mapping),
        "Harness candidate parent JointVersion is missing",
    )
    try:
        parent_joint_version = JointVersion(**dict(raw_joint_version))
    except TypeError as exc:
        raise TorchHarnessLearningError(
            "Harness candidate parent JointVersion is invalid"
        ) from exc
    if active_joint_version is not None:
        _require(
            parent_joint_version == active_joint_version,
            "Harness candidate differs from lag-zero active JointVersion",
        )
    _require(
        record.get("parent_joint_version_id") == parent_joint_version.version_id
        and record.get("behavior_version") == parent_joint_version.harness_controller,
        "Harness candidate parent identity differs from JointVersion",
    )
    _require(
        _is_sha256(record.get("source_joint_credit_sha256")),
        "Harness candidate source S hash is invalid",
    )
    before_digest = record.get("parameter_digest_before")
    after_digest = record.get("parameter_digest_after")
    _require(
        _is_sha256(before_digest)
        and _is_sha256(after_digest)
        and before_digest != after_digest,
        "Harness candidate parameter digests do not prove a change",
    )
    step_before = record.get("optimizer_step_before")
    step_after = record.get("optimizer_step_after")
    _require(
        type(step_before) is int and step_before >= 0 and step_after == step_before + 1,
        "Harness candidate optimizer step did not advance exactly once",
    )
    expected_behavior_version = (
        f"{POLICY_SCHEMA_VERSION}-step{step_before:06d}-{before_digest[:16]}"
    )
    expected_candidate_version = (
        f"{POLICY_SCHEMA_VERSION}-step{step_after:06d}-{after_digest[:16]}"
    )
    _require(
        record.get("behavior_version") == expected_behavior_version
        and record.get("behavior_version") == parent_joint_version.harness_controller
        and record.get("candidate_version") == expected_candidate_version
        and record.get("candidate_version") != record.get("behavior_version"),
        "Harness behavior or candidate version differs from its parameter digest",
    )
    batch_size = record.get("batch_size")
    effective_batch_size = record.get("effective_batch_size")
    _require(
        type(batch_size) is int
        and type(effective_batch_size) is int
        and 0 < effective_batch_size <= batch_size,
        "Harness candidate batch sizes are invalid",
    )
    for field in ("loss", "mean_ratio", "clip_fraction", "gradient_norm"):
        value = record.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"Harness candidate {field} is not finite",
        )
    _require(
        float(record["gradient_norm"]) > 0.0
        and 0.0 <= float(record["clip_fraction"]) <= 1.0
        and float(record["mean_ratio"]) > 0.0,
        "Harness candidate optimizer metrics are invalid",
    )
    checkpoint_path = record.get("checkpoint_path")
    checkpoint_sha256 = record.get("checkpoint_sha256")
    _require(
        isinstance(checkpoint_path, str)
        and checkpoint_path
        and _is_sha256(checkpoint_sha256),
        "Harness candidate checkpoint identity is invalid",
    )
    if require_checkpoint:
        unresolved_path = Path(checkpoint_path).expanduser()
        _require(
            not unresolved_path.is_symlink(),
            "Harness candidate checkpoint is missing or unsafe",
        )
        path = require_outside_repository(checkpoint_path)
        _require(
            path.is_file() and not path.is_symlink(),
            "Harness candidate checkpoint is missing or unsafe",
        )
        _require(
            _checkpoint_sha256(path) == checkpoint_sha256,
            "Harness candidate checkpoint hash mismatch",
        )
        restored_policy, restored_optimizer, checkpoint = load_torch_harness_checkpoint(
            path, map_location="cpu"
        )
        del restored_optimizer  # fully checked by the checkpoint loader
        source = checkpoint["source"]
        metrics = checkpoint["update_metrics"]
        _require(
            restored_policy.version == record.get("candidate_version")
            and restored_policy.parameter_digest == after_digest
            and restored_policy.update_step == step_after,
            "Harness candidate checkpoint model differs from evidence",
        )
        _require(
            source["transaction_id"] == transaction_id
            and source["joint_credit_record_sha256"]
            == record.get("source_joint_credit_sha256")
            and source["parent_joint_version_id"]
            == record.get("parent_joint_version_id"),
            "Harness candidate checkpoint source differs from evidence",
        )
        _require(
            metrics
            == {
                "batch_size": batch_size,
                "effective_batch_size": effective_batch_size,
                "loss": record["loss"],
                "mean_ratio": record["mean_ratio"],
                "clip_fraction": record["clip_fraction"],
                "gradient_norm": record["gradient_norm"],
                "parameter_digest_before": before_digest,
                "parameter_digest_after": after_digest,
            },
            "Harness candidate checkpoint metrics differ from evidence",
        )
    _require(
        record.get("evidence_scope") == _HARNESS_EVIDENCE_SCOPE,
        "Harness candidate evidence scope differs from contract",
    )
    payload = {
        field: record[field]
        for field in evidence_fields
        if field != "parent_joint_version"
    }
    return TorchHarnessUpdateEvidence(
        parent_joint_version=parent_joint_version,
        **payload,
    )


def validate_torch_harness_multi_s_update_evidence(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
    validated_batch: ValidatedMultiSFrozenTrainingBatch | None = None,
    require_checkpoint: bool = True,
) -> TorchHarnessMultiSUpdateEvidence:
    """Revalidate one Adam receipt over an ordered validated multi-S batch."""

    _assert_no_secret_fields(record)
    evidence_fields = set(TorchHarnessMultiSUpdateEvidence.__dataclass_fields__)
    _require(
        set(record)
        == evidence_fields | {"schema_version", "evidence_scope", "record_sha256"},
        "multi-S Harness candidate evidence field set differs",
    )
    _require(
        record.get("schema_version") == MULTI_S_HARNESS_CANDIDATE_SCHEMA_VERSION,
        "unknown multi-S Harness candidate evidence schema",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "multi-S Harness candidate evidence hash mismatch",
    )
    transaction_id = record.get("transaction_id")
    _require(
        isinstance(transaction_id, str) and bool(transaction_id),
        "multi-S Harness candidate transaction ID is missing",
    )
    raw_joint_version = record.get("parent_joint_version")
    _require(
        isinstance(raw_joint_version, Mapping),
        "multi-S Harness candidate parent JointVersion is missing",
    )
    try:
        parent_joint_version = JointVersion(**dict(raw_joint_version))
    except TypeError as exc:
        raise TorchHarnessLearningError(
            "multi-S Harness candidate parent JointVersion is invalid"
        ) from exc
    if active_joint_version is not None:
        _require(
            parent_joint_version == active_joint_version,
            "multi-S Harness candidate differs from lag-zero active JointVersion",
        )
    _require(
        record.get("parent_joint_version_id") == parent_joint_version.version_id
        and record.get("behavior_version") == parent_joint_version.harness_controller,
        "multi-S Harness candidate parent identity differs from JointVersion",
    )
    try:
        source_binding = validate_multi_s_source_binding(
            record.get("source_binding"),  # type: ignore[arg-type]
            batch=validated_batch,
        )
    except ValueError as exc:
        raise TorchHarnessLearningError(str(exc)) from exc
    _require(
        source_binding["joint_version_id"] == parent_joint_version.version_id,
        "multi-S Harness source binding differs from parent JointVersion",
    )
    _require(
        record.get("source_joint_credit_sha256") == source_binding["record_sha256"],
        "multi-S Harness source digest differs from source binding",
    )
    before_digest = record.get("parameter_digest_before")
    after_digest = record.get("parameter_digest_after")
    step_before = record.get("optimizer_step_before")
    step_after = record.get("optimizer_step_after")
    _require(
        _is_sha256(before_digest)
        and _is_sha256(after_digest)
        and before_digest != after_digest
        and type(step_before) is int
        and step_before >= 0
        and step_after == step_before + 1,
        "multi-S Harness candidate did not prove exactly one optimizer step",
    )
    _require(
        record.get("behavior_version")
        == f"{POLICY_SCHEMA_VERSION}-step{step_before:06d}-{before_digest[:16]}"
        and record.get("candidate_version")
        == f"{POLICY_SCHEMA_VERSION}-step{step_after:06d}-{after_digest[:16]}",
        "multi-S Harness behavior or candidate version differs",
    )
    batch_size = record.get("batch_size")
    effective_batch_size = record.get("effective_batch_size")
    _require(
        type(batch_size) is int
        and type(effective_batch_size) is int
        and 0 < effective_batch_size <= batch_size,
        "multi-S Harness candidate batch sizes are invalid",
    )
    for field in ("loss", "mean_ratio", "clip_fraction", "gradient_norm"):
        value = record.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"multi-S Harness candidate {field} is not finite",
        )
    _require(
        float(record["gradient_norm"]) > 0.0
        and float(record["mean_ratio"]) > 0.0
        and 0.0 <= float(record["clip_fraction"]) <= 1.0,
        "multi-S Harness candidate optimizer metrics are invalid",
    )
    checkpoint_path = record.get("checkpoint_path")
    checkpoint_sha256 = record.get("checkpoint_sha256")
    _require(
        isinstance(checkpoint_path, str)
        and bool(checkpoint_path)
        and _is_sha256(checkpoint_sha256),
        "multi-S Harness candidate checkpoint identity is invalid",
    )
    if require_checkpoint:
        unresolved_path = Path(checkpoint_path).expanduser()
        _require(
            not unresolved_path.is_symlink(),
            "multi-S Harness candidate checkpoint is missing or unsafe",
        )
        path = require_outside_repository(checkpoint_path)
        _require(
            path.is_file()
            and not path.is_symlink()
            and _checkpoint_sha256(path) == checkpoint_sha256,
            "multi-S Harness candidate checkpoint hash mismatch",
        )
        restored_policy, restored_optimizer, checkpoint = load_torch_harness_checkpoint(
            path, map_location="cpu"
        )
        del restored_optimizer
        source = checkpoint["source"]
        metrics = checkpoint["update_metrics"]
        _require(
            checkpoint["schema_version"] == MULTI_S_CHECKPOINT_SCHEMA_VERSION
            and restored_policy.version == record.get("candidate_version")
            and restored_policy.parameter_digest == after_digest
            and restored_policy.update_step == step_after,
            "multi-S Harness candidate checkpoint model differs from evidence",
        )
        _require(
            source
            == {
                "transaction_id": transaction_id,
                "multi_s_source_binding": source_binding,
                "parent_joint_version_id": parent_joint_version.version_id,
            },
            "multi-S Harness candidate checkpoint source differs from evidence",
        )
        _require(
            metrics
            == {
                "batch_size": batch_size,
                "effective_batch_size": effective_batch_size,
                "loss": record["loss"],
                "mean_ratio": record["mean_ratio"],
                "clip_fraction": record["clip_fraction"],
                "gradient_norm": record["gradient_norm"],
                "parameter_digest_before": before_digest,
                "parameter_digest_after": after_digest,
            },
            "multi-S Harness candidate checkpoint metrics differ from evidence",
        )
    _require(
        record.get("evidence_scope") == _MULTI_S_HARNESS_EVIDENCE_SCOPE,
        "multi-S Harness candidate evidence scope differs from contract",
    )
    payload = {
        field: record[field]
        for field in evidence_fields
        if field not in {"parent_joint_version", "source_binding"}
    }
    return TorchHarnessMultiSUpdateEvidence(
        parent_joint_version=parent_joint_version,
        source_binding=source_binding,
        **payload,
    )


def _safe_torch_load(path: Path, *, map_location: str | torch.device) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise TorchHarnessLearningError(
            "safe Harness checkpoint loading requires torch.load(weights_only=True)"
        ) from exc
    except Exception as exc:
        raise TorchHarnessLearningError(
            "cannot safely load Harness checkpoint"
        ) from exc


def _checkpoint_payload(
    *,
    policy: TorchHarnessPolicy,
    optimizer: torch.optim.Adam,
    schema_version: str,
    source: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "policy_schema_version": policy.schema_version,
        "state_feature_schema_version": policy.state_feature_schema_version,
        "action_ids": list(ACTION_IDS),
        "policy_config": {
            "seed": policy.seed,
            "hidden_size": policy.hidden_size,
        },
        "policy_metadata": {
            "update_step": policy.update_step,
            "sample_count": policy.sample_count,
            "version": policy.version,
            "parameter_digest": policy.parameter_digest,
        },
        "model_state_dict": deepcopy(policy.state_dict()),
        "optimizer_name": "Adam",
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "rng": {
            "policy_generator_state": policy._generator.get_state().clone(),
            "torch_cpu_rng_state": torch.get_rng_state().clone(),
            # Harness sampling owns a CPU generator. Avoid initializing CUDA merely
            # to checkpoint this small, independently optimized controller.
            "torch_cuda_rng_states": [],
        },
        "source": deepcopy(dict(source)),
        "update_metrics": dict(metrics),
    }


def load_torch_harness_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TorchHarnessPolicy, torch.optim.Adam, dict[str, object]]:
    """Load and self-validate model, Adam state, and owned sampling RNG."""

    unresolved_path = Path(checkpoint_path).expanduser()
    _require(
        not unresolved_path.is_symlink(),
        "Harness checkpoint does not exist or is unsafe",
    )
    path = require_outside_repository(checkpoint_path)
    _require(
        path.is_file() and not path.is_symlink(),
        "Harness checkpoint does not exist or is unsafe",
    )
    raw = _safe_torch_load(path, map_location=map_location)
    _require(isinstance(raw, Mapping), "Harness checkpoint must be an object")
    _assert_no_secret_fields(raw, "checkpoint")
    required = {
        "schema_version",
        "policy_schema_version",
        "state_feature_schema_version",
        "action_ids",
        "policy_config",
        "policy_metadata",
        "model_state_dict",
        "optimizer_name",
        "optimizer_state_dict",
        "rng",
        "source",
        "update_metrics",
    }
    _require(set(raw) == required, "Harness checkpoint field set differs")
    checkpoint_schema = raw["schema_version"]
    _require(
        checkpoint_schema
        in {CHECKPOINT_SCHEMA_VERSION, MULTI_S_CHECKPOINT_SCHEMA_VERSION},
        "unknown Harness checkpoint schema",
    )
    _require(
        raw["policy_schema_version"] == POLICY_SCHEMA_VERSION
        and raw["state_feature_schema_version"] == _STATE_FEATURE_SCHEMA_VERSION,
        "Harness checkpoint policy schema differs",
    )
    _require(
        tuple(raw["action_ids"]) == ACTION_IDS,
        "Harness checkpoint action schema differs",
    )
    config = raw["policy_config"]
    metadata = raw["policy_metadata"]
    rng = raw["rng"]
    _require(
        isinstance(config, Mapping)
        and set(config) == {"seed", "hidden_size"}
        and type(config.get("seed")) is int
        and config["seed"] >= 0
        and type(config.get("hidden_size")) is int
        and config["hidden_size"] > 0,
        "Harness checkpoint policy config differs",
    )
    _require(
        isinstance(metadata, Mapping)
        and set(metadata)
        == {"update_step", "sample_count", "version", "parameter_digest"},
        "Harness checkpoint policy metadata differs",
    )
    _require(
        type(metadata.get("update_step")) is int
        and metadata["update_step"] > 0
        and type(metadata.get("sample_count")) is int
        and metadata["sample_count"] >= 0
        and isinstance(metadata.get("version"), str)
        and _is_sha256(metadata.get("parameter_digest")),
        "Harness checkpoint policy metadata values are invalid",
    )
    _require(
        isinstance(rng, Mapping)
        and set(rng)
        == {
            "policy_generator_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
        },
        "Harness checkpoint RNG state differs",
    )
    policy = TorchHarnessPolicy(
        seed=config["seed"],
        hidden_size=config["hidden_size"],
    )
    model_state = raw["model_state_dict"]
    expected_state = policy.state_dict()
    _require(
        isinstance(model_state, Mapping) and set(model_state) == set(expected_state),
        "Harness checkpoint model parameter names differ",
    )
    for name, expected in expected_state.items():
        value = model_state[name]
        _require(
            isinstance(value, Tensor)
            and value.shape == expected.shape
            and value.dtype == expected.dtype
            and bool(torch.isfinite(value).all()),
            f"Harness checkpoint model parameter {name} is invalid",
        )
    policy.load_state_dict(model_state, strict=True)
    policy.to(map_location)
    policy.update_step = metadata["update_step"]
    policy.sample_count = metadata["sample_count"]
    policy_generator_state = rng["policy_generator_state"]
    torch_cpu_rng_state = rng["torch_cpu_rng_state"]
    _require(
        isinstance(policy_generator_state, Tensor)
        and policy_generator_state.dtype == torch.uint8
        and policy_generator_state.ndim == 1
        and policy_generator_state.numel() > 0
        and isinstance(torch_cpu_rng_state, Tensor)
        and torch_cpu_rng_state.dtype == torch.uint8
        and torch_cpu_rng_state.ndim == 1
        and torch_cpu_rng_state.numel() > 0
        and isinstance(rng["torch_cuda_rng_states"], list)
        and not rng["torch_cuda_rng_states"],
        "Harness checkpoint RNG state is invalid",
    )
    try:
        policy._generator.set_state(policy_generator_state.cpu())
    except RuntimeError as exc:
        raise TorchHarnessLearningError(
            "Harness checkpoint policy RNG cannot be restored"
        ) from exc
    _require(
        policy.parameter_digest == metadata["parameter_digest"],
        "Harness checkpoint parameter digest mismatch",
    )
    _require(
        policy.version == metadata["version"],
        "Harness checkpoint policy version mismatch",
    )
    _require(raw["optimizer_name"] == "Adam", "Harness checkpoint is not Adam")
    optimizer_state = raw["optimizer_state_dict"]
    _require(
        isinstance(optimizer_state, Mapping),
        "Harness checkpoint optimizer state is missing",
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0)
    try:
        optimizer.load_state_dict(optimizer_state)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TorchHarnessLearningError(
            "Harness checkpoint Adam state cannot be restored"
        ) from exc
    _validate_adam_optimizer_state(
        policy,
        optimizer,
        allow_uninitialized_step_zero=False,
    )
    source = raw["source"]
    metrics = raw["update_metrics"]
    _require(isinstance(source, Mapping), "Harness checkpoint source identity differs")
    common_source_valid = (
        isinstance(source.get("transaction_id"), str)
        and bool(source["transaction_id"])
        and isinstance(source.get("parent_joint_version_id"), str)
        and bool(source["parent_joint_version_id"])
    )
    if checkpoint_schema == CHECKPOINT_SCHEMA_VERSION:
        _require(
            common_source_valid
            and set(source)
            == {
                "transaction_id",
                "joint_credit_record_sha256",
                "parent_joint_version_id",
            }
            and _is_sha256(source.get("joint_credit_record_sha256")),
            "Harness checkpoint source identity differs",
        )
    else:
        _require(
            common_source_valid
            and set(source)
            == {
                "transaction_id",
                "multi_s_source_binding",
                "parent_joint_version_id",
            },
            "multi-S Harness checkpoint source identity differs",
        )
        try:
            checkpoint_binding = validate_multi_s_source_binding(
                source.get("multi_s_source_binding"),  # type: ignore[arg-type]
            )
        except ValueError as exc:
            raise TorchHarnessLearningError(str(exc)) from exc
        _require(
            source["parent_joint_version_id"] == checkpoint_binding["joint_version_id"],
            "multi-S Harness checkpoint source JointVersion differs",
        )
    _require(
        isinstance(metrics, Mapping) and set(metrics) == _CHECKPOINT_METRIC_FIELDS,
        "Harness checkpoint metrics field set differs",
    )
    batch_size = metrics.get("batch_size")
    effective_batch_size = metrics.get("effective_batch_size")
    _require(
        type(batch_size) is int
        and type(effective_batch_size) is int
        and 0 < effective_batch_size <= batch_size,
        "Harness checkpoint batch sizes are invalid",
    )
    for name in ("loss", "mean_ratio", "clip_fraction", "gradient_norm"):
        value = metrics.get(name)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"Harness checkpoint metric {name} is invalid",
        )
    _require(
        float(metrics["mean_ratio"]) > 0.0
        and 0.0 <= float(metrics["clip_fraction"]) <= 1.0
        and float(metrics["gradient_norm"]) > 0.0
        and _is_sha256(metrics.get("parameter_digest_before"))
        and metrics.get("parameter_digest_after") == policy.parameter_digest
        and metrics["parameter_digest_before"] != metrics["parameter_digest_after"],
        "Harness checkpoint optimizer metrics are inconsistent",
    )
    return policy, optimizer, dict(raw)


@dataclass(frozen=True)
class _PreparedHarnessBatch:
    samples: tuple[Mapping[str, object], ...]
    states: tuple[HarnessState, ...]
    masks: tuple[tuple[bool, ...], ...]
    selected_indexes: tuple[int, ...]
    recorded_logits: tuple[tuple[float, ...], ...]
    old_logprobs: tuple[float, ...]
    advantages: tuple[float, ...]
    loss_masks: tuple[int, ...]


def _prepare_harness_batch(
    policy: TorchHarnessPolicy,
    samples: Sequence[object],
    *,
    expected_trainable_count: int,
) -> _PreparedHarnessBatch:
    _require(bool(samples), "Harness optimizer requires at least one admitted sample")
    _require(
        type(expected_trainable_count) is int and expected_trainable_count >= 0,
        "Harness trainable action count is invalid",
    )
    parsed_samples: list[Mapping[str, object]] = []
    states: list[HarnessState] = []
    masks: list[tuple[bool, ...]] = []
    selected_indexes: list[int] = []
    recorded_logits: list[tuple[float, ...]] = []
    old_logprobs: list[float] = []
    advantages: list[float] = []
    loss_masks: list[int] = []
    for sample in samples:
        _require(isinstance(sample, Mapping), "Harness sample must be an object")
        action = sample.get("action")
        _require(isinstance(action, Mapping), "Harness action record is missing")
        state_raw = action.get("state")
        _require(isinstance(state_raw, Mapping), "Harness state record is missing")
        _require(
            set(state_raw) == set(HarnessState.__dataclass_fields__),
            "Harness state field set differs from schema",
        )
        try:
            state = HarnessState(**dict(state_raw))
        except TypeError as exc:
            raise TorchHarnessLearningError("invalid Harness state") from exc
        _state_payload(state)
        action_ids = tuple(action.get("action_ids", ()))
        _require(
            action_ids == ACTION_IDS,
            "Harness action IDs differ from fixed five-action schema",
        )
        mask = _normalize_action_mask(action.get("action_mask"))
        selected_action = action.get("action")
        _require(
            selected_action in ACTION_IDS,
            "Harness selected action differs from fixed schema",
        )
        selected = ACTION_IDS.index(str(selected_action))
        _require(mask[selected], "chosen Harness action is masked out")
        logits = action.get("pre_mask_logits")
        _require(
            isinstance(logits, (list, tuple))
            and len(logits) == len(ACTION_IDS)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in logits
            ),
            "Harness behavior logits are invalid",
        )
        old_logprob = action.get("old_harness_logprob")
        loss_mask = action.get("harness_loss_mask")
        masked_advantage = sample.get("masked_advantage")
        _require(
            isinstance(old_logprob, (int, float))
            and not isinstance(old_logprob, bool)
            and math.isfinite(float(old_logprob)),
            "Harness old log-prob is invalid",
        )
        _require(
            type(loss_mask) is int and loss_mask in (0, 1),
            "Harness loss mask must be integer 0 or 1",
        )
        _require(
            isinstance(masked_advantage, (int, float))
            and not isinstance(masked_advantage, bool)
            and math.isfinite(float(masked_advantage)),
            "Harness masked advantage is invalid",
        )
        _require(
            action.get("harness_behavior_version") == policy.version,
            "Harness sample behavior version differs from policy snapshot",
        )
        parsed_samples.append(sample)
        states.append(state)
        masks.append(mask)
        selected_indexes.append(selected)
        recorded_logits.append(tuple(float(value) for value in logits))
        old_logprobs.append(float(old_logprob))
        advantages.append(float(masked_advantage))
        loss_masks.append(loss_mask)

    _require(
        sum(loss_masks) == expected_trainable_count,
        "Harness loss masks differ from validated S summary",
    )
    _require(sum(loss_masks) > 0, "Harness batch has zero trainable credit")
    _require(
        any(
            mask and abs(advantage) > 0.0
            for mask, advantage in zip(loss_masks, advantages)
        ),
        "Harness batch has zero effective credit",
    )
    return _PreparedHarnessBatch(
        samples=tuple(parsed_samples),
        states=tuple(states),
        masks=tuple(masks),
        selected_indexes=tuple(selected_indexes),
        recorded_logits=tuple(recorded_logits),
        old_logprobs=tuple(old_logprobs),
        advantages=tuple(advantages),
        loss_masks=tuple(loss_masks),
    )


class TorchHarnessOptimizer:
    """Clipped-PPO updater that leaves the admitted behavior snapshot untouched."""

    def __init__(
        self,
        policy: TorchHarnessPolicy,
        *,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        max_grad_norm: float = 1.0,
        _optimizer: torch.optim.Adam | None = None,
    ) -> None:
        _require(
            type(policy) is TorchHarnessPolicy,
            "Harness optimizer requires a TorchHarnessPolicy",
        )
        _require(
            math.isfinite(learning_rate) and learning_rate > 0.0,
            "Harness learning rate must be finite and positive",
        )
        _require(
            math.isfinite(clip_ratio) and 0.0 < clip_ratio < 1.0,
            "Harness PPO clip ratio must be between zero and one",
        )
        _require(
            math.isfinite(max_grad_norm) and max_grad_norm > 0.0,
            "Harness max gradient norm must be finite and positive",
        )
        self.policy = policy
        self.clip_ratio = float(clip_ratio)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = _optimizer or torch.optim.Adam(
            policy.parameters(), lr=float(learning_rate)
        )
        _require(
            type(self.optimizer) is torch.optim.Adam
            and len(self.optimizer.param_groups) == 1,
            "Harness optimizer must be one independent Adam parameter group",
        )
        _validate_adam_optimizer_state(
            self.policy,
            self.optimizer,
            allow_uninitialized_step_zero=True,
        )

    def _candidate(self) -> tuple[TorchHarnessPolicy, torch.optim.Adam]:
        _validate_adam_optimizer_state(
            self.policy,
            self.optimizer,
            allow_uninitialized_step_zero=True,
        )
        candidate = self.policy.clone()
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
        optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))
        _validate_adam_optimizer_state(
            candidate,
            optimizer,
            allow_uninitialized_step_zero=True,
        )
        return candidate, optimizer

    def _execute_prepared_batch(
        self,
        prepared: _PreparedHarnessBatch,
        *,
        checkpoint_path: str | os.PathLike[str],
        checkpoint_schema_version: str,
        checkpoint_source: Mapping[str, object],
    ) -> tuple[
        TorchHarnessPolicy,
        torch.optim.Adam,
        dict[str, object],
        Path,
    ]:
        target = require_outside_repository(checkpoint_path)
        _require(not target.exists(), "Harness checkpoint path already exists")
        target.parent.mkdir(parents=True, exist_ok=True)

        candidate, optimizer = self._candidate()
        logits = candidate.logits_for(prepared.states)
        device = logits.device
        dtype = logits.dtype
        recorded_logits_tensor = torch.tensor(
            prepared.recorded_logits,
            dtype=dtype,
            device=device,
        )
        _require(
            torch.isfinite(logits).all().item(),
            "recomputed Harness logits are not finite",
        )
        _require(
            torch.allclose(
                logits.detach(),
                recorded_logits_tensor,
                rtol=0.0,
                atol=1e-6,
            ),
            "recorded Harness logits differ from behavior snapshot",
        )
        mask_tensor = torch.tensor(prepared.masks, dtype=torch.bool, device=device)
        selected_tensor = torch.tensor(
            prepared.selected_indexes,
            dtype=torch.long,
            device=device,
        )
        masked_logits = logits.masked_fill(~mask_tensor, -torch.inf)
        current_logprobs = torch.log_softmax(masked_logits, dim=-1).gather(
            1,
            selected_tensor[:, None],
        )[:, 0]
        old_logprob_tensor = torch.tensor(
            prepared.old_logprobs,
            dtype=dtype,
            device=device,
        )
        _require(
            torch.allclose(
                current_logprobs.detach(),
                old_logprob_tensor,
                rtol=0.0,
                atol=1e-6,
            ),
            "recorded old log-prob differs from behavior snapshot",
        )
        advantage_tensor = torch.tensor(
            prepared.advantages,
            dtype=dtype,
            device=device,
        )
        loss_mask_tensor = torch.tensor(
            prepared.loss_masks,
            dtype=dtype,
            device=device,
        )
        ratio = torch.exp(current_logprobs - old_logprob_tensor)
        unclipped = ratio * advantage_tensor
        clipped = (
            torch.clamp(
                ratio,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            )
            * advantage_tensor
        )
        loss = (
            -(torch.minimum(unclipped, clipped) * loss_mask_tensor).sum()
            / loss_mask_tensor.sum()
        )
        _require(torch.isfinite(loss).item(), "Harness PPO loss is not finite")

        parameter_digest_before = self.policy.parameter_digest
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_squared = torch.zeros((), dtype=torch.float64, device=device)
        for parameter in candidate.parameters():
            if parameter.grad is not None:
                gradient_squared += parameter.grad.detach().double().square().sum()
        gradient_norm = float(torch.sqrt(gradient_squared).item())
        _require(
            math.isfinite(gradient_norm) and gradient_norm > 0.0,
            "Harness PPO gradient norm must be finite and non-zero",
        )
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), self.max_grad_norm)
        optimizer.step()
        candidate.update_step = self.policy.update_step + 1
        _validate_adam_optimizer_state(
            candidate,
            optimizer,
            allow_uninitialized_step_zero=False,
        )
        parameter_digest_after = candidate.parameter_digest
        _require(
            parameter_digest_after != parameter_digest_before,
            "Harness Adam step did not change the parameter digest",
        )
        _require(
            candidate.version != self.policy.version,
            "Harness candidate version did not advance",
        )
        metrics: dict[str, object] = {
            "batch_size": len(prepared.samples),
            "effective_batch_size": sum(prepared.loss_masks),
            "loss": float(loss.detach().item()),
            "mean_ratio": float(ratio.detach().mean().item()),
            "clip_fraction": float(
                ((ratio.detach() - 1.0).abs() > self.clip_ratio).float().mean().item()
            ),
            "gradient_norm": gradient_norm,
            "parameter_digest_before": parameter_digest_before,
            "parameter_digest_after": parameter_digest_after,
        }
        checkpoint = _checkpoint_payload(
            policy=candidate,
            optimizer=optimizer,
            schema_version=checkpoint_schema_version,
            source=checkpoint_source,
            metrics=metrics,
        )
        temporary_path: Path | None = None
        target_created = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, target)
            target_created = True
            temporary_path.unlink()
            temporary_path = None
            restored_policy, restored_optimizer, restored = (
                load_torch_harness_checkpoint(target, map_location=candidate.device)
            )
            _require(
                restored_policy.parameter_digest == parameter_digest_after
                and restored_policy.version == candidate.version,
                "Harness checkpoint model round trip differs from candidate",
            )
            _require(
                bool(restored_optimizer.state)
                and restored["schema_version"] == checkpoint_schema_version
                and restored["source"] == checkpoint_source,
                "Harness checkpoint optimizer or source round trip differs",
            )
            _require(
                torch.equal(
                    restored_policy._generator.get_state(),
                    candidate._generator.get_state(),
                ),
                "Harness checkpoint sampling RNG round trip differs",
            )
        except Exception:
            if target_created:
                target.unlink(missing_ok=True)
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return candidate, optimizer, metrics, target

    def update_from_frozen_joint_credit(
        self,
        record: Mapping[str, object],
        *,
        transaction_id: str,
        active_joint_version: JointVersion,
        checkpoint_path: str | os.PathLike[str],
    ) -> TorchHarnessUpdateResult:
        """Create one candidate from a real, lag-zero validated S record."""

        _require(
            isinstance(record, Mapping),
            "Harness optimizer requires a frozen joint-credit record",
        )
        _require(
            isinstance(transaction_id, str) and bool(transaction_id),
            "Harness optimizer transaction ID is missing",
        )
        _assert_no_secret_fields(record)
        _require(
            type(active_joint_version) is JointVersion,
            "Harness optimizer requires an active JointVersion",
        )
        try:
            audit = validate_frozen_joint_credit_alignment(
                record,
                active_joint_version=active_joint_version,
            )
        except (JointCreditAlignmentError, ValueError) as exc:
            raise TorchHarnessLearningError(str(exc)) from exc
        _require(
            self.policy.version == active_joint_version.harness_controller,
            "Harness behavior version differs from lag-zero JointVersion",
        )
        _require(
            audit["joint_version_id"] == active_joint_version.version_id,
            "S audit differs from lag-zero JointVersion",
        )
        target = require_outside_repository(checkpoint_path)
        _require(not target.exists(), "Harness checkpoint path already exists")
        target.parent.mkdir(parents=True, exist_ok=True)

        samples = record.get("harness_samples")
        _require(
            isinstance(samples, list) and bool(samples),
            "Harness optimizer requires at least one admitted sample",
        )
        states: list[HarnessState] = []
        masks: list[tuple[bool, ...]] = []
        selected_indexes: list[int] = []
        recorded_logits: list[list[float]] = []
        old_logprobs: list[float] = []
        advantages: list[float] = []
        loss_masks: list[int] = []
        for sample in samples:
            _require(isinstance(sample, Mapping), "Harness sample must be an object")
            action = sample.get("action")
            _require(isinstance(action, Mapping), "Harness action record is missing")
            state_raw = action.get("state")
            _require(isinstance(state_raw, Mapping), "Harness state record is missing")
            _require(
                set(state_raw) == set(HarnessState.__dataclass_fields__),
                "Harness state field set differs from schema",
            )
            try:
                state = HarnessState(**dict(state_raw))
            except TypeError as exc:
                raise TorchHarnessLearningError("invalid Harness state") from exc
            _state_payload(state)
            action_ids = tuple(action.get("action_ids", ()))
            _require(
                action_ids == ACTION_IDS,
                "Harness action IDs differ from fixed five-action schema",
            )
            mask = _normalize_action_mask(action.get("action_mask"))
            selected_action = action.get("action")
            _require(
                selected_action in ACTION_IDS,
                "Harness selected action differs from fixed schema",
            )
            selected = ACTION_IDS.index(str(selected_action))
            _require(mask[selected], "chosen Harness action is masked out")
            logits = action.get("pre_mask_logits")
            _require(
                isinstance(logits, (list, tuple))
                and len(logits) == len(ACTION_IDS)
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in logits
                ),
                "Harness behavior logits are invalid",
            )
            old_logprob = action.get("old_harness_logprob")
            loss_mask = action.get("harness_loss_mask")
            masked_advantage = sample.get("masked_advantage")
            _require(
                isinstance(old_logprob, (int, float))
                and not isinstance(old_logprob, bool)
                and math.isfinite(float(old_logprob)),
                "Harness old log-prob is invalid",
            )
            _require(
                type(loss_mask) is int and loss_mask in (0, 1),
                "Harness loss mask must be integer 0 or 1",
            )
            _require(
                isinstance(masked_advantage, (int, float))
                and not isinstance(masked_advantage, bool)
                and math.isfinite(float(masked_advantage)),
                "Harness masked advantage is invalid",
            )
            _require(
                action.get("harness_behavior_version") == self.policy.version,
                "Harness sample behavior version differs from policy snapshot",
            )
            states.append(state)
            masks.append(mask)
            selected_indexes.append(selected)
            recorded_logits.append([float(value) for value in logits])
            old_logprobs.append(float(old_logprob))
            advantages.append(float(masked_advantage))
            loss_masks.append(loss_mask)

        _require(
            sum(loss_masks) == int(audit["harness_trainable_action_count"]),
            "Harness loss masks differ from validated S summary",
        )
        _require(sum(loss_masks) > 0, "Harness batch has zero trainable credit")
        _require(
            any(
                mask and abs(advantage) > 0.0
                for mask, advantage in zip(loss_masks, advantages)
            ),
            "Harness batch has zero effective credit",
        )

        candidate, optimizer = self._candidate()
        logits = candidate.logits_for(states)
        device = logits.device
        dtype = logits.dtype
        recorded_logits_tensor = torch.tensor(
            recorded_logits, dtype=dtype, device=device
        )
        _require(
            torch.isfinite(logits).all().item(),
            "recomputed Harness logits are not finite",
        )
        _require(
            torch.allclose(
                logits.detach(),
                recorded_logits_tensor,
                rtol=0.0,
                atol=1e-6,
            ),
            "recorded Harness logits differ from behavior snapshot",
        )
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
        selected_tensor = torch.tensor(
            selected_indexes, dtype=torch.long, device=device
        )
        masked_logits = logits.masked_fill(~mask_tensor, -torch.inf)
        current_logprobs = torch.log_softmax(masked_logits, dim=-1).gather(
            1, selected_tensor[:, None]
        )[:, 0]
        old_logprob_tensor = torch.tensor(old_logprobs, dtype=dtype, device=device)
        _require(
            torch.allclose(
                current_logprobs.detach(),
                old_logprob_tensor,
                rtol=0.0,
                atol=1e-6,
            ),
            "recorded old log-prob differs from behavior snapshot",
        )
        advantage_tensor = torch.tensor(advantages, dtype=dtype, device=device)
        loss_mask_tensor = torch.tensor(loss_masks, dtype=dtype, device=device)
        ratio = torch.exp(current_logprobs - old_logprob_tensor)
        unclipped = ratio * advantage_tensor
        clipped = (
            torch.clamp(
                ratio,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            )
            * advantage_tensor
        )
        denominator = loss_mask_tensor.sum()
        loss = (
            -(torch.minimum(unclipped, clipped) * loss_mask_tensor).sum() / denominator
        )
        _require(torch.isfinite(loss).item(), "Harness PPO loss is not finite")

        parameter_digest_before = self.policy.parameter_digest
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_squared = torch.zeros((), dtype=torch.float64, device=device)
        for parameter in candidate.parameters():
            if parameter.grad is not None:
                gradient_squared += parameter.grad.detach().double().square().sum()
        gradient_norm_tensor = torch.sqrt(gradient_squared)
        gradient_norm = float(gradient_norm_tensor.item())
        _require(
            math.isfinite(gradient_norm) and gradient_norm > 0.0,
            "Harness PPO gradient norm must be finite and non-zero",
        )
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), self.max_grad_norm)
        optimizer.step()
        candidate.update_step = self.policy.update_step + 1
        _validate_adam_optimizer_state(
            candidate,
            optimizer,
            allow_uninitialized_step_zero=False,
        )
        parameter_digest_after = candidate.parameter_digest
        _require(
            parameter_digest_after != parameter_digest_before,
            "Harness Adam step did not change the parameter digest",
        )
        _require(
            candidate.version != self.policy.version,
            "Harness candidate version did not advance",
        )
        mean_ratio = float(ratio.detach().mean().item())
        clip_fraction = float(
            ((ratio.detach() - 1.0).abs() > self.clip_ratio).float().mean().item()
        )
        metrics: dict[str, object] = {
            "batch_size": len(samples),
            "effective_batch_size": sum(loss_masks),
            "loss": float(loss.detach().item()),
            "mean_ratio": mean_ratio,
            "clip_fraction": clip_fraction,
            "gradient_norm": gradient_norm,
            "parameter_digest_before": parameter_digest_before,
            "parameter_digest_after": parameter_digest_after,
        }
        checkpoint = _checkpoint_payload(
            policy=candidate,
            optimizer=optimizer,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            source={
                "transaction_id": transaction_id,
                "joint_credit_record_sha256": str(record["record_sha256"]),
                "parent_joint_version_id": active_joint_version.version_id,
            },
            metrics=metrics,
        )
        temporary_path: Path | None = None
        target_created = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link publication is atomic and refuses to overwrite a target
            # created after the preflight existence check.
            os.link(temporary_path, target)
            target_created = True
            temporary_path.unlink()
            temporary_path = None
            restored_policy, restored_optimizer, restored = (
                load_torch_harness_checkpoint(target, map_location=candidate.device)
            )
            _require(
                restored_policy.parameter_digest == parameter_digest_after
                and restored_policy.version == candidate.version,
                "Harness checkpoint model round trip differs from candidate",
            )
            _require(
                bool(restored_optimizer.state)
                and restored["source"]["transaction_id"] == transaction_id
                and restored["source"]["joint_credit_record_sha256"]
                == record["record_sha256"]
                and restored["source"]["parent_joint_version_id"]
                == active_joint_version.version_id,
                "Harness checkpoint optimizer or source round trip differs",
            )
            _require(
                torch.equal(
                    restored_policy._generator.get_state(),
                    candidate._generator.get_state(),
                ),
                "Harness checkpoint sampling RNG round trip differs",
            )
        except Exception:
            if target_created:
                target.unlink(missing_ok=True)
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        evidence = TorchHarnessUpdateEvidence(
            transaction_id=transaction_id,
            source_joint_credit_sha256=str(record["record_sha256"]),
            parent_joint_version=active_joint_version,
            parent_joint_version_id=active_joint_version.version_id,
            behavior_version=self.policy.version,
            candidate_version=candidate.version,
            parameter_digest_before=parameter_digest_before,
            parameter_digest_after=parameter_digest_after,
            optimizer_step_before=self.policy.update_step,
            optimizer_step_after=candidate.update_step,
            batch_size=len(samples),
            effective_batch_size=sum(loss_masks),
            loss=float(loss.detach().item()),
            mean_ratio=mean_ratio,
            clip_fraction=clip_fraction,
            gradient_norm=gradient_norm,
            checkpoint_path=str(target.resolve()),
            checkpoint_sha256=_checkpoint_sha256(target),
        )
        candidate_trainer = TorchHarnessOptimizer(
            candidate,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            clip_ratio=self.clip_ratio,
            max_grad_norm=self.max_grad_norm,
            _optimizer=optimizer,
        )
        return TorchHarnessUpdateResult(
            candidate_policy=candidate,
            candidate_optimizer=candidate_trainer,
            evidence=evidence,
        )

    def update_from_validated_multi_s_frozen_training_batch(
        self,
        batch: ValidatedMultiSFrozenTrainingBatch,
        *,
        transaction_id: str,
        active_joint_version: JointVersion,
        checkpoint_path: str | os.PathLike[str],
    ) -> TorchHarnessMultiSUpdateResult:
        """Run exactly one Adam step over every ordered Harness sample in T/U's batch."""

        _require(
            type(batch) is ValidatedMultiSFrozenTrainingBatch,
            "multi-S Harness optimizer requires a validated batch object",
        )
        _require(
            isinstance(transaction_id, str) and bool(transaction_id),
            "multi-S Harness optimizer transaction ID is missing",
        )
        _require(
            type(active_joint_version) is JointVersion,
            "multi-S Harness optimizer requires an active JointVersion",
        )
        _require(
            batch.joint_version == active_joint_version,
            "multi-S Harness batch differs from lag-zero active JointVersion",
        )
        _require(
            self.policy.version == active_joint_version.harness_controller,
            "Harness behavior version differs from lag-zero JointVersion",
        )
        try:
            source_binding = multi_s_source_binding(batch)
            validate_multi_s_source_binding(source_binding, batch=batch)
            ordered_records = tuple(iter_member_s_records(batch))
        except ValueError as exc:
            raise TorchHarnessLearningError(str(exc)) from exc
        _assert_no_secret_fields(source_binding, "source_binding")
        _require(
            len(ordered_records) == len(batch.members) >= 4,
            "multi-S Harness optimizer requires at least four ordered members",
        )

        merged_samples: list[object] = []
        trainable_action_count = 0
        policy_sample_count = 0
        member_claims: set[str] = set()
        stable_s_claims: set[str] = set()
        for expected_index, ((member_claim, s_record), member) in enumerate(
            zip(ordered_records, batch.members)
        ):
            _require(
                member.member_index == expected_index
                and member_claim == member.member_claim_sha256,
                "multi-S Harness member order or claim differs",
            )
            _require(
                member_claim not in member_claims
                and member.s_record_sha256 not in stable_s_claims,
                "multi-S Harness member claims must be unique",
            )
            member_claims.add(member_claim)
            stable_s_claims.add(member.s_record_sha256)
            _assert_no_secret_fields(s_record, f"members[{expected_index}].s_record")
            _require(
                s_record.get("record_sha256") == member.s_record_sha256
                and _record_sha256(s_record) == member.s_record_sha256,
                "multi-S Harness member S digest differs",
            )
            try:
                audit = validate_frozen_joint_credit_alignment(
                    s_record,
                    active_joint_version=active_joint_version,
                )
            except (JointCreditAlignmentError, ValueError) as exc:
                raise TorchHarnessLearningError(str(exc)) from exc
            _require(
                audit["joint_version_id"] == active_joint_version.version_id,
                "multi-S Harness member differs from lag-zero JointVersion",
            )
            admissions = s_record.get("admissions")
            _require(
                isinstance(admissions, Mapping)
                and admissions.get("policy_export_style") == "individual",
                "multi-S Harness optimizer rejects concat/post-batch records",
            )
            samples = s_record.get("harness_samples")
            policy_samples = s_record.get("policy_samples")
            _require(
                isinstance(samples, list)
                and bool(samples)
                and isinstance(policy_samples, list)
                and bool(policy_samples),
                "multi-S Harness member samples are missing",
            )
            _require(
                len(samples) == len(member.harness_decision_ids)
                and len(policy_samples) == len(member.policy_sample_ids),
                "multi-S Harness member sample counts differ from claims",
            )
            persisted_path = member.source_path.expanduser()
            _require(
                not persisted_path.is_symlink(),
                "multi-S Harness persisted S source is unsafe",
            )
            persisted_path = require_outside_repository(persisted_path)
            _require(
                persisted_path.is_file() and not persisted_path.is_symlink(),
                "multi-S Harness persisted S source is missing or unsafe",
            )
            raw = persisted_path.read_bytes()
            _require(
                hashlib.sha256(raw).hexdigest() == member.source_file_sha256,
                "multi-S Harness persisted S source file hash mismatch",
            )
            try:
                persisted_s_record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TorchHarnessLearningError(
                    "multi-S Harness persisted S source is invalid JSON"
                ) from exc
            _require(
                _canonical_json(persisted_s_record) == _canonical_json(s_record),
                "multi-S Harness persisted S source differs from embedded member",
            )
            merged_samples.extend(samples)
            policy_sample_count += len(policy_samples)
            trainable_action_count += int(audit["harness_trainable_action_count"])

        _require(
            len(merged_samples)
            == batch.harness_action_count
            == source_binding["harness_action_count"]
            and policy_sample_count
            == batch.policy_sample_count
            == source_binding["policy_sample_count"],
            "multi-S Harness merged sample counts differ from source binding",
        )
        prepared = _prepare_harness_batch(
            self.policy,
            merged_samples,
            expected_trainable_count=trainable_action_count,
        )
        checkpoint_source: dict[str, object] = {
            "transaction_id": transaction_id,
            "multi_s_source_binding": deepcopy(source_binding),
            "parent_joint_version_id": active_joint_version.version_id,
        }
        candidate, optimizer, metrics, target = self._execute_prepared_batch(
            prepared,
            checkpoint_path=checkpoint_path,
            checkpoint_schema_version=MULTI_S_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_source=checkpoint_source,
        )
        evidence = TorchHarnessMultiSUpdateEvidence(
            transaction_id=transaction_id,
            source_joint_credit_sha256=str(source_binding["record_sha256"]),
            source_binding=deepcopy(source_binding),
            parent_joint_version=active_joint_version,
            parent_joint_version_id=active_joint_version.version_id,
            behavior_version=self.policy.version,
            candidate_version=candidate.version,
            parameter_digest_before=str(metrics["parameter_digest_before"]),
            parameter_digest_after=str(metrics["parameter_digest_after"]),
            optimizer_step_before=self.policy.update_step,
            optimizer_step_after=candidate.update_step,
            batch_size=int(metrics["batch_size"]),
            effective_batch_size=int(metrics["effective_batch_size"]),
            loss=float(metrics["loss"]),
            mean_ratio=float(metrics["mean_ratio"]),
            clip_fraction=float(metrics["clip_fraction"]),
            gradient_norm=float(metrics["gradient_norm"]),
            checkpoint_path=str(target.resolve()),
            checkpoint_sha256=_checkpoint_sha256(target),
        )
        validate_torch_harness_multi_s_update_evidence(
            evidence.to_record(),
            active_joint_version=active_joint_version,
            validated_batch=batch,
        )
        candidate_trainer = TorchHarnessOptimizer(
            candidate,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            clip_ratio=self.clip_ratio,
            max_grad_norm=self.max_grad_norm,
            _optimizer=optimizer,
        )
        return TorchHarnessMultiSUpdateResult(
            candidate_policy=candidate,
            candidate_optimizer=candidate_trainer,
            evidence=evidence,
        )


__all__ = [
    "ACTION_IDS",
    "CHECKPOINT_SCHEMA_VERSION",
    "HARNESS_CANDIDATE_SCHEMA_VERSION",
    "MULTI_S_CHECKPOINT_SCHEMA_VERSION",
    "MULTI_S_HARNESS_CANDIDATE_SCHEMA_VERSION",
    "POLICY_SCHEMA_VERSION",
    "ROLLOUT_CHECKPOINT_SCHEMA_VERSION",
    "TorchHarnessLearningError",
    "TorchHarnessMultiSUpdateEvidence",
    "TorchHarnessMultiSUpdateResult",
    "TorchHarnessOptimizer",
    "TorchHarnessPolicy",
    "TorchHarnessUpdateEvidence",
    "TorchHarnessUpdateResult",
    "build_torch_harness_rollout_checkpoint",
    "encode_harness_state",
    "load_torch_harness_checkpoint",
    "load_torch_harness_rollout_checkpoint",
    "validate_torch_harness_multi_s_update_evidence",
    "validate_torch_harness_update_evidence",
]
