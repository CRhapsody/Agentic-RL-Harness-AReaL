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

from jphrl.paths import require_within_configured_root
from jphrl.trajectory.joint_credit_alignment import (
    JointCreditAlignmentError,
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.schema import JointVersion

from .controller import HarnessDecision, HarnessState
from .spec import HarnessAction

ACTION_IDS = tuple(action.value for action in HarnessAction)
POLICY_SCHEMA_VERSION = "torch-harness-categorical-v1"
CHECKPOINT_SCHEMA_VERSION = "jph.torch-harness-checkpoint.v1"
HARNESS_CANDIDATE_SCHEMA_VERSION = "jph.torch-harness-candidate.v1"
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


@dataclass(frozen=True)
class TorchHarnessUpdateEvidence:
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
) -> TorchHarnessUpdateEvidence:
    """Revalidate a persisted real Harness optimizer receipt."""

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
    expected_candidate_version = (
        f"{POLICY_SCHEMA_VERSION}-step{step_after:06d}-{after_digest[:16]}"
    )
    _require(
        record.get("candidate_version") == expected_candidate_version
        and record.get("candidate_version") != record.get("behavior_version"),
        "Harness candidate version differs from its parameter digest",
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
        path = require_within_configured_root(checkpoint_path)
        _require(
            path.is_file() and not path.is_symlink(),
            "Harness candidate checkpoint is missing or unsafe",
        )
        _require(
            _checkpoint_sha256(path) == checkpoint_sha256,
            "Harness candidate checkpoint hash mismatch",
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


def _safe_torch_load(path: Path, *, map_location: str | torch.device) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only argument
        return torch.load(path, map_location=map_location)


def _checkpoint_payload(
    *,
    policy: TorchHarnessPolicy,
    optimizer: torch.optim.Adam,
    source_joint_credit_sha256: str,
    parent_joint_version_id: str,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
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
        "source": {
            "joint_credit_record_sha256": source_joint_credit_sha256,
            "parent_joint_version_id": parent_joint_version_id,
        },
        "update_metrics": dict(metrics),
    }


def load_torch_harness_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TorchHarnessPolicy, torch.optim.Adam, dict[str, object]]:
    """Load and self-validate model, Adam state, and owned sampling RNG."""

    path = Path(checkpoint_path)
    _require(path.is_file(), "Harness checkpoint does not exist")
    raw = _safe_torch_load(path, map_location=map_location)
    _require(isinstance(raw, Mapping), "Harness checkpoint must be an object")
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
    _require(
        raw["schema_version"] == CHECKPOINT_SCHEMA_VERSION,
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
        isinstance(config, Mapping) and set(config) == {"seed", "hidden_size"},
        "Harness checkpoint policy config differs",
    )
    _require(
        isinstance(metadata, Mapping)
        and set(metadata)
        == {"update_step", "sample_count", "version", "parameter_digest"},
        "Harness checkpoint policy metadata differs",
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
        seed=int(config["seed"]),
        hidden_size=int(config["hidden_size"]),
    )
    policy.load_state_dict(raw["model_state_dict"], strict=True)
    policy.to(map_location)
    policy.update_step = int(metadata["update_step"])
    policy.sample_count = int(metadata["sample_count"])
    policy._generator.set_state(rng["policy_generator_state"].cpu())
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
    optimizer.load_state_dict(optimizer_state)
    _require(bool(optimizer.state), "Harness checkpoint Adam state is empty")
    _require(
        isinstance(rng["torch_cpu_rng_state"], Tensor)
        and isinstance(rng["torch_cuda_rng_states"], list)
        and all(isinstance(state, Tensor) for state in rng["torch_cuda_rng_states"]),
        "Harness checkpoint process RNG state is invalid",
    )
    source = raw["source"]
    metrics = raw["update_metrics"]
    _require(
        isinstance(source, Mapping)
        and set(source) == {"joint_credit_record_sha256", "parent_joint_version_id"},
        "Harness checkpoint source identity differs",
    )
    _require(isinstance(metrics, Mapping), "Harness checkpoint metrics are missing")
    return policy, optimizer, dict(raw)


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

    def _candidate(self) -> tuple[TorchHarnessPolicy, torch.optim.Adam]:
        candidate = self.policy.clone()
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
        optimizer.load_state_dict(deepcopy(self.optimizer.state_dict()))
        return candidate, optimizer

    def update_from_frozen_joint_credit(
        self,
        record: Mapping[str, object],
        *,
        active_joint_version: JointVersion,
        checkpoint_path: str | os.PathLike[str],
    ) -> TorchHarnessUpdateResult:
        """Create one candidate from a real, lag-zero validated S record."""

        _require(
            isinstance(record, Mapping),
            "Harness optimizer requires a frozen joint-credit record",
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
        target = require_within_configured_root(checkpoint_path)
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
        parameter_digest_after = candidate.parameter_digest
        _require(
            parameter_digest_after != parameter_digest_before,
            "Harness Adam step did not change the parameter digest",
        )
        _require(
            candidate.version != self.policy.version,
            "Harness candidate version did not advance",
        )
        _require(bool(optimizer.state), "Harness Adam state is empty after update")

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
            source_joint_credit_sha256=str(record["record_sha256"]),
            parent_joint_version_id=active_joint_version.version_id,
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


__all__ = [
    "ACTION_IDS",
    "CHECKPOINT_SCHEMA_VERSION",
    "HARNESS_CANDIDATE_SCHEMA_VERSION",
    "POLICY_SCHEMA_VERSION",
    "TorchHarnessLearningError",
    "TorchHarnessOptimizer",
    "TorchHarnessPolicy",
    "TorchHarnessUpdateEvidence",
    "TorchHarnessUpdateResult",
    "encode_harness_state",
    "load_torch_harness_checkpoint",
    "validate_torch_harness_update_evidence",
]
