from __future__ import annotations

"""Real held-out X observations for the single-step M0 joint update.

The evaluator consumes separately persisted RLVR workflow admissions that were
not used for the optimizer step.  It recomputes candidate Policy token
log-probabilities with the live AReaL actor and candidate Harness logits with
the restored Torch policy.  It returns measurements only; X owns thresholds,
repeatability checks, and the final verdict.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jphrl.experiments.m0_joint_runner import (
    M0AcceptanceGate,
    M0CandidateEvaluator,
    M0JointRunnerError,
    RLVRM0SourceRecords,
)
from jphrl.harness.controller import HarnessState
from jphrl.trajectory.schema import JointVersion

M0_HELDOUT_FIXTURE_SCHEMA = "jph.m0-heldout-suite-fixture.v1"
M0_DISTRIBUTED_HELDOUT_FIXTURE_SCHEMA = "jph.m0-distributed-heldout-suite-fixture.v1"
M0_HELDOUT_OBSERVATION_SCHEMA = "jph.m0-heldout-observation.v1"

_SUITE_KINDS = (
    "policy_heldout",
    "harness_offpolicy",
    "joint_safety",
    "restart_recovery",
)
_TENSOR_FIELDS = {
    "input_ids",
    "loss_mask",
    "logprobs",
    "versions",
    "attention_mask",
    "rewards",
}


class M0LiveEvaluatorError(M0JointRunnerError):
    """Raised when a held-out observation cannot be measured honestly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0LiveEvaluatorError(message)


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
        raise M0LiveEvaluatorError(
            "held-out observation is not finite canonical JSON"
        ) from exc


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain_tensor(value: object) -> list[object]:
    detach = getattr(value, "detach", None)
    _require(callable(detach), "held-out result is not a tensor")
    tensor = detach().cpu()
    return tensor.tolist()


def _finite_summary(value: object) -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - remote dependency
        raise M0LiveEvaluatorError("torch is required for M0 held-out X") from exc
    _require(isinstance(value, torch.Tensor), "held-out result is not a tensor")
    tensor = value.detach().to(dtype=torch.float64, device="cpu")
    _require(tensor.numel() > 0, "held-out result tensor is empty")
    finite = torch.isfinite(tensor)
    finite_count = int(finite.count_nonzero().item())
    total_count = int(tensor.numel())
    summary: dict[str, object] = {
        "finite_count": finite_count,
        "total_count": total_count,
        "finite_fraction": finite_count / total_count,
    }
    if finite_count == total_count:
        summary.update(
            {
                "mean": float(tensor.mean().item()),
                "minimum": float(tensor.min().item()),
                "maximum": float(tensor.max().item()),
            }
        )
    else:
        summary.update({"mean": None, "minimum": None, "maximum": None})
    _canonical_json(summary)
    return summary


def _fixture_record(
    *,
    kind: str,
    training_runner_admission_sha256: str,
    holdout_runner_admission_sha256s: Sequence[str],
) -> dict[str, object]:
    _require(kind in _SUITE_KINDS, "unknown M0 held-out suite")
    _require(
        _is_sha256(training_runner_admission_sha256),
        "training runner admission digest is invalid",
    )
    holdouts = tuple(holdout_runner_admission_sha256s)
    _require(
        bool(holdouts)
        and len(set(holdouts)) == len(holdouts)
        and all(_is_sha256(value) for value in holdouts)
        and training_runner_admission_sha256 not in holdouts,
        "held-out runner admission identity is invalid",
    )
    return {
        "schema_version": M0_HELDOUT_FIXTURE_SCHEMA,
        "suite_kind": kind,
        "training_runner_admission_sha256": training_runner_admission_sha256,
        "holdout_runner_admission_sha256s": list(holdouts),
    }


def build_m0_heldout_acceptance_gates(
    *,
    training_runner_admission_sha256: str,
    holdout_runner_admission_sha256s: Sequence[str],
) -> tuple[M0AcceptanceGate, ...]:
    """Freeze all four X suites to one disjoint set of real rollout records."""

    holdouts = tuple(holdout_runner_admission_sha256s)
    metrics = {
        "policy_heldout": "finite_candidate_token_logprob_fraction",
        "harness_offpolicy": "finite_candidate_harness_logit_fraction",
        "joint_safety": "joint_candidate_finite_fraction",
        "restart_recovery": "repeatable_candidate_observation",
    }
    gates: list[M0AcceptanceGate] = []
    for kind in _SUITE_KINDS:
        fixture = _canonical_json(
            _fixture_record(
                kind=kind,
                training_runner_admission_sha256=(
                    training_runner_admission_sha256
                ),
                holdout_runner_admission_sha256s=holdouts,
            )
        )
        gates.append(
            M0AcceptanceGate(
                kind=kind,
                suite_id=f"m0-real-{kind}-v1",
                fixture=fixture,
                metric_name=metrics[kind],
                minimum_score=1.0,
                minimum_sample_count=(
                    len(holdouts)
                    if kind in {"policy_heldout", "harness_offpolicy"}
                    else 1
                ),
            )
        )
    return tuple(gates)


def build_m0_distributed_heldout_acceptance_gates(
    *,
    training_runner_admission_sha256s: Sequence[str],
    holdout_runner_admission_sha256s: Sequence[str],
) -> tuple[M0AcceptanceGate, ...]:
    """Freeze four training and four disjoint X admissions into every suite."""

    training = tuple(training_runner_admission_sha256s)
    holdouts = tuple(holdout_runner_admission_sha256s)
    _require(
        len(training) == len(holdouts) == 4
        and len(set(training)) == len(set(holdouts)) == 4
        and set(training).isdisjoint(holdouts)
        and all(_is_sha256(value) for value in training + holdouts),
        "distributed X requires four unique training and four unique holdout admissions",
    )
    metrics = {
        "policy_heldout": "finite_candidate_token_logprob_fraction",
        "harness_offpolicy": "finite_candidate_harness_logit_fraction",
        "joint_safety": "joint_candidate_finite_fraction",
        "restart_recovery": "repeatable_candidate_observation",
    }
    gates: list[M0AcceptanceGate] = []
    for kind in _SUITE_KINDS:
        fixture = _canonical_json(
            {
                "schema_version": M0_DISTRIBUTED_HELDOUT_FIXTURE_SCHEMA,
                "suite_kind": kind,
                "training_runner_admission_sha256s": list(training),
                "holdout_runner_admission_sha256s": list(holdouts),
            }
        )
        gates.append(
            M0AcceptanceGate(
                kind=kind,
                suite_id=f"m0-distributed-real-{kind}-v1",
                fixture=fixture,
                metric_name=metrics[kind],
                minimum_score=1.0,
                minimum_sample_count=(
                    4 if kind in {"policy_heldout", "harness_offpolicy"} else 1
                ),
            )
        )
    return tuple(gates)


def _validate_rlvr_source(
    source: RLVRM0SourceRecords,
    *,
    active_joint_version: JointVersion,
) -> None:
    _require(
        type(source) is RLVRM0SourceRecords,
        "M0 held-out evaluator requires dedicated RLVR sources",
    )
    _require(
        source.active_joint_version == active_joint_version
        and _is_sha256(source.runner_admission_sha256)
        and source.runner_admission.get("record_sha256")
        == source.runner_admission_sha256,
        "held-out RLVR source differs from the training JointVersion",
    )
    try:
        from jphrl.trajectory.rlvr_workflow_admission import (
            validate_rlvr_workflow_runner_admission,
        )

        validate_rlvr_workflow_runner_admission(
            source.runner_admission,
            active_joint_version=active_joint_version,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M0LiveEvaluatorError(str(exc)) from exc


def _policy_samples(source: RLVRM0SourceRecords) -> list[Mapping[str, object]]:
    try:
        record = source.s_joint_credit["admissions"]["policy_admission_record"]
        samples = record["samples"]
    except (KeyError, TypeError) as exc:
        raise M0LiveEvaluatorError(
            "held-out RLVR Policy samples are missing"
        ) from exc
    _require(
        isinstance(samples, list)
        and bool(samples)
        and all(isinstance(sample, Mapping) for sample in samples),
        "held-out RLVR Policy samples are invalid",
    )
    return list(samples)


def _harness_state(source: RLVRM0SourceRecords) -> HarnessState:
    try:
        events = source.runner_admission["episode_trace"]["events"]
        event = events[1]
        raw = event["payload"]["state"]
    except (IndexError, KeyError, TypeError) as exc:
        raise M0LiveEvaluatorError("held-out Harness state is missing") from exc
    _require(
        isinstance(event, Mapping)
        and event.get("kind") == "harness_decision"
        and isinstance(raw, Mapping)
        and set(raw) == set(HarnessState.__dataclass_fields__),
        "held-out Harness state field set differs",
    )
    try:
        return HarnessState(**dict(raw))
    except TypeError as exc:
        raise M0LiveEvaluatorError("held-out Harness state is invalid") from exc


@dataclass(frozen=True)
class _MeasuredCandidate:
    policy: tuple[dict[str, object], ...]
    harness: tuple[dict[str, object], ...]
    actor_public_version: int
    harness_controller_version: str
    harness_parameter_sha256: str
    policy_current_state_attestation_sha256: str | None

    def joint_record(self) -> dict[str, object]:
        return {
            "schema_version": M0_HELDOUT_OBSERVATION_SCHEMA,
            "actor_public_version": self.actor_public_version,
            "harness_controller_version": self.harness_controller_version,
            "harness_parameter_sha256": self.harness_parameter_sha256,
            "policy_current_state_attestation_sha256": (
                self.policy_current_state_attestation_sha256
            ),
            "policy": list(self.policy),
            "harness": list(self.harness),
        }


class RealRlvrM0CandidateEvaluator(M0CandidateEvaluator):
    """Recompute X measurements on disjoint real RLVR workflow rollouts."""

    def __init__(
        self,
        *,
        training_source: RLVRM0SourceRecords,
        holdout_sources: Sequence[RLVRM0SourceRecords],
    ) -> None:
        _require(
            type(training_source) is RLVRM0SourceRecords,
            "training source must be a dedicated RLVR runner admission",
        )
        active = training_source.active_joint_version
        _validate_rlvr_source(training_source, active_joint_version=active)
        holdouts = tuple(holdout_sources)
        _require(bool(holdouts), "M0 X requires at least one real held-out rollout")
        for source in holdouts:
            _validate_rlvr_source(source, active_joint_version=active)
        digests = tuple(source.runner_admission_sha256 for source in holdouts)
        _require(
            len(set(digests)) == len(digests)
            and training_source.runner_admission_sha256 not in digests,
            "training and held-out RLVR admissions are not disjoint",
        )
        self._training_source = training_source
        self._holdouts = holdouts
        self._gates = build_m0_heldout_acceptance_gates(
            training_runner_admission_sha256=(
                training_source.runner_admission_sha256
            ),
            holdout_runner_admission_sha256s=digests,
        )
        self._fixture_by_kind = {gate.kind: gate.fixture for gate in self._gates}

    @property
    def acceptance_gates(self) -> tuple[M0AcceptanceGate, ...]:
        return self._gates

    def _validate_live_components(
        self,
        *,
        actor: object,
        harness_policy: object,
        joint_version: JointVersion,
        live_policy_candidate: object | None,
        policy_current_state_attestation: Mapping[str, object] | None,
        distributed_serving_export_receipt: Mapping[str, object] | None,
        live_serving_exports: object | None,
    ) -> str | None:
        actor_identity = f"{type(actor).__module__}.{type(actor).__qualname__}"
        _require(
            (
                type(actor).__name__ == "FSDPPPOActor"
                and type(actor).__module__.startswith("areal.")
            )
            or actor_identity
            == "jphrl.training.areal_distributed_policy.JPHPPOActorController",
            "M0 X requires the pinned actor or exact distributed controller",
        )
        _require(
            type(harness_policy).__name__ == "TorchHarnessPolicy"
            and type(harness_policy).__module__
            == "jphrl.harness.torch_learning",
            "M0 X requires the real Torch Harness candidate",
        )
        _require(
            isinstance(getattr(harness_policy, "version", None), str)
            and harness_policy.version == joint_version.harness_controller,
            "M0 X Harness candidate differs from the candidate JointVersion",
        )
        parameter_digest = getattr(harness_policy, "parameter_digest", None)
        _require(
            _is_sha256(parameter_digest),
            "M0 X Harness candidate parameter digest is invalid",
        )
        if actor_identity != (
            "jphrl.training.areal_distributed_policy.JPHPPOActorController"
        ):
            _require(
                live_policy_candidate is None
                and policy_current_state_attestation is None
                and distributed_serving_export_receipt is None
                and live_serving_exports is None,
                "single-rank M0 X cannot accept distributed current-state evidence",
            )
            return None
        _require(
            isinstance(policy_current_state_attestation, Mapping)
            and isinstance(distributed_serving_export_receipt, Mapping),
            "distributed M0 X current-state evidence is missing",
        )
        from jphrl.training.areal_distributed_policy import (
            require_live_remote_policy_candidate,
            validate_distributed_policy_current_state_receipt,
            validate_distributed_policy_candidate,
        )
        from jphrl.training.areal_production_worker import (
            require_live_areal_serving_export_pair,
        )

        live = require_live_remote_policy_candidate(live_policy_candidate)
        policy = validate_distributed_policy_candidate(live.receipt)
        serving = require_live_areal_serving_export_pair(live_serving_exports)
        _require(
            joint_version.policy == policy.candidate_policy_version
            and serving.candidate.joint_version == joint_version
            and serving.candidate.policy_candidate_record_sha256
            == policy.record_sha256,
            "distributed M0 X candidate/version/export lineage is crossed",
        )
        validated = validate_distributed_policy_current_state_receipt(
            policy_current_state_attestation,
            candidate=live,
            distributed_serving_export_receipt=(
                distributed_serving_export_receipt
            ),
            candidate_serving_export_lineage_sha256=(
                serving.candidate.record_sha256
            ),
        )
        _require(
            validated["candidate_serving_parameter_sha256"]
            == serving.candidate.serving_parameter_sha256,
            "distributed M0 X serving parameter digest is crossed",
        )
        return str(validated["record_sha256"])

    def _measure_policy(self, actor: object) -> tuple[dict[str, object], ...]:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - remote dependency
            raise M0LiveEvaluatorError("torch is required for M0 held-out X") from exc
        actor_identity = f"{type(actor).__module__}.{type(actor).__qualname__}"
        device = (
            "cpu"
            if actor_identity
            == "jphrl.training.areal_distributed_policy.JPHPPOActorController"
            else getattr(actor, "device", None)
        )
        _require(device is not None, "AReaL actor device is unavailable")
        dtypes = {
            "input_ids": torch.long,
            "loss_mask": torch.long,
            "logprobs": torch.float32,
            "versions": torch.long,
            "attention_mask": torch.bool,
            "rewards": torch.float32,
        }
        inputs: list[dict[str, Any]] = []
        identities: list[tuple[str, str]] = []
        for source in self._holdouts:
            samples = _policy_samples(source)
            _require(
                len(samples) == 1,
                "M0 single-turn held-out source must contain exactly one sample",
            )
            sample = samples[0]
            tensor_dict = sample.get("tensor_dict")
            _require(
                isinstance(tensor_dict, Mapping)
                and set(tensor_dict) == _TENSOR_FIELDS,
                "held-out Policy tensor contract differs from AReaL six-field input",
            )
            inputs.append(
                {
                    field: torch.tensor(value, dtype=dtypes[field], device=device)
                    for field, value in tensor_dict.items()
                }
            )
            identities.append(
                (
                    source.runner_admission_sha256,
                    str(sample.get("sample_id")),
                )
            )
        compute_logp = getattr(actor, "compute_logp", None)
        _require(callable(compute_logp), "AReaL actor compute_logp is unavailable")
        outputs = compute_logp(inputs)
        _require(
            isinstance(outputs, list) and len(outputs) == len(inputs),
            "AReaL actor returned the wrong held-out log-probability count",
        )
        measured: list[dict[str, object]] = []
        for identity, output in zip(identities, outputs):
            summary = _finite_summary(output)
            measured.append(
                {
                    "runner_admission_sha256": identity[0],
                    "sample_id": identity[1],
                    "candidate_logprobs": summary,
                }
            )
        return tuple(measured)

    def _measure_harness(
        self, harness_policy: object
    ) -> tuple[dict[str, object], ...]:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - remote dependency
            raise M0LiveEvaluatorError("torch is required for M0 held-out X") from exc
        states = [_harness_state(source) for source in self._holdouts]
        logits_for = getattr(harness_policy, "logits_for", None)
        _require(callable(logits_for), "Torch Harness logits_for is unavailable")
        with torch.no_grad():
            logits = logits_for(states)
        _require(
            isinstance(logits, torch.Tensor)
            and logits.ndim == 2
            and logits.shape[0] == len(states)
            and logits.shape[1] == 5,
            "Torch Harness returned the wrong held-out five-action logits",
        )
        measured: list[dict[str, object]] = []
        for source, row in zip(self._holdouts, logits):
            measured.append(
                {
                    "runner_admission_sha256": source.runner_admission_sha256,
                    "candidate_harness_logits": _finite_summary(row),
                }
            )
        return tuple(measured)

    def _measure(
        self,
        *,
        actor: object,
        harness_policy: object,
        joint_version: JointVersion,
        live_policy_candidate: object | None,
        policy_current_state_attestation: Mapping[str, object] | None,
        distributed_serving_export_receipt: Mapping[str, object] | None,
        live_serving_exports: object | None,
    ) -> _MeasuredCandidate:
        current_state_sha256 = self._validate_live_components(
            actor=actor,
            harness_policy=harness_policy,
            joint_version=joint_version,
            live_policy_candidate=live_policy_candidate,
            policy_current_state_attestation=policy_current_state_attestation,
            distributed_serving_export_receipt=(
                distributed_serving_export_receipt
            ),
            live_serving_exports=live_serving_exports,
        )
        get_version = getattr(actor, "get_version", None)
        _require(callable(get_version), "AReaL actor get_version is unavailable")
        actor_version = get_version()
        _require(
            type(actor_version) is int and actor_version >= 0,
            "AReaL actor public version is invalid",
        )
        policy = self._measure_policy(actor)
        harness = self._measure_harness(harness_policy)
        if current_state_sha256 is not None:
            policy = tuple(
                {
                    **item,
                    "policy_current_state_attestation_sha256": (
                        current_state_sha256
                    ),
                }
                for item in policy
            )
            harness = tuple(
                {
                    **item,
                    "policy_current_state_attestation_sha256": (
                        current_state_sha256
                    ),
                }
                for item in harness
            )
        return _MeasuredCandidate(
            policy=policy,
            harness=harness,
            actor_public_version=actor_version,
            harness_controller_version=str(harness_policy.version),
            harness_parameter_sha256=str(harness_policy.parameter_digest),
            policy_current_state_attestation_sha256=current_state_sha256,
        )

    @staticmethod
    def _minimum_fraction(values: Sequence[Mapping[str, object]], key: str) -> float:
        fractions: list[float] = []
        for value in values:
            summary = value.get(key)
            _require(isinstance(summary, Mapping), "held-out summary is missing")
            fraction = summary.get("finite_fraction")
            _require(
                isinstance(fraction, (int, float))
                and not isinstance(fraction, bool)
                and math.isfinite(float(fraction)),
                "held-out finite fraction is invalid",
            )
            fractions.append(float(fraction))
        _require(bool(fractions), "held-out finite fraction set is empty")
        return min(fractions)

    def observe(
        self,
        *,
        joint_version: JointVersion,
        gate: M0AcceptanceGate,
        actor: object,
        harness_policy: object,
        live_policy_candidate: object | None = None,
        policy_current_state_attestation: Mapping[str, object] | None = None,
        distributed_serving_export_receipt: Mapping[str, object] | None = None,
        live_serving_exports: object | None = None,
    ) -> Sequence[object]:
        _require(
            type(joint_version) is JointVersion,
            "M0 X candidate JointVersion is untyped",
        )
        gate.validate()
        _require(
            gate.kind in self._fixture_by_kind
            and gate.fixture == self._fixture_by_kind[gate.kind],
            "M0 X suite fixture differs from the frozen held-out admissions",
        )
        measured = self._measure(
            actor=actor,
            harness_policy=harness_policy,
            joint_version=joint_version,
            live_policy_candidate=live_policy_candidate,
            policy_current_state_attestation=policy_current_state_attestation,
            distributed_serving_export_receipt=(
                distributed_serving_export_receipt
            ),
            live_serving_exports=live_serving_exports,
        )
        try:
            from jphrl.training.candidate_acceptance import (
                CandidateProbeObservation,
            )
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            raise M0LiveEvaluatorError(
                "candidate observation type is unavailable"
            ) from exc

        if gate.kind == "policy_heldout":
            return tuple(
                CandidateProbeObservation(
                    sample_id=f"policy-{item['runner_admission_sha256'][:20]}",
                    metric_value=self._minimum_fraction(
                        (item,), "candidate_logprobs"
                    ),
                    output=dict(item),
                )
                for item in measured.policy
            )
        if gate.kind == "harness_offpolicy":
            return tuple(
                CandidateProbeObservation(
                    sample_id=f"harness-{item['runner_admission_sha256'][:20]}",
                    metric_value=self._minimum_fraction(
                        (item,), "candidate_harness_logits"
                    ),
                    output=dict(item),
                )
                for item in measured.harness
            )
        joint_record = measured.joint_record()
        if gate.kind == "joint_safety":
            policy_fraction = self._minimum_fraction(
                measured.policy, "candidate_logprobs"
            )
            harness_fraction = self._minimum_fraction(
                measured.harness, "candidate_harness_logits"
            )
            return (
                CandidateProbeObservation(
                    sample_id="joint-live-candidate",
                    metric_value=min(policy_fraction, harness_fraction),
                    output=joint_record,
                ),
            )
        _require(
            gate.kind == "restart_recovery",
            "unknown M0 X suite kind",
        )
        repeated = self._measure(
            actor=actor,
            harness_policy=harness_policy,
            joint_version=joint_version,
            live_policy_candidate=live_policy_candidate,
            policy_current_state_attestation=policy_current_state_attestation,
            distributed_serving_export_receipt=(
                distributed_serving_export_receipt
            ),
            live_serving_exports=live_serving_exports,
        ).joint_record()
        identical = _canonical_json(joint_record) == _canonical_json(repeated)
        return (
            CandidateProbeObservation(
                sample_id="restart-recovery-live-candidate",
                metric_value=1.0 if identical else 0.0,
                    output={
                        "schema_version": M0_HELDOUT_OBSERVATION_SCHEMA,
                        "policy_current_state_attestation_sha256": (
                            measured.policy_current_state_attestation_sha256
                        ),
                    "repeatable": identical,
                    "first_sha256": hashlib.sha256(
                        _canonical_json(joint_record)
                    ).hexdigest(),
                    "second_sha256": hashlib.sha256(
                        _canonical_json(repeated)
                    ).hexdigest(),
                },
            ),
        )


class DistributedRealRlvrM0CandidateEvaluator(RealRlvrM0CandidateEvaluator):
    """The fsdp:d4 evaluator with all four optimizer sources frozen in X."""

    def __init__(
        self,
        *,
        training_sources: Sequence[RLVRM0SourceRecords],
        holdout_sources: Sequence[RLVRM0SourceRecords],
    ) -> None:
        training = tuple(training_sources)
        holdouts = tuple(holdout_sources)
        _require(
            len(training) == len(holdouts) == 4,
            "distributed M0 X requires exactly four training and four holdouts",
        )
        active = training[0].active_joint_version
        for source in training + holdouts:
            _validate_rlvr_source(source, active_joint_version=active)
        training_digests = tuple(
            source.runner_admission_sha256 for source in training
        )
        holdout_digests = tuple(
            source.runner_admission_sha256 for source in holdouts
        )
        _require(
            len(set(training_digests)) == len(set(holdout_digests)) == 4
            and set(training_digests).isdisjoint(holdout_digests),
            "distributed M0 training and held-out admissions overlap",
        )
        self._training_source = training[0]
        self._training_sources = training
        self._holdouts = holdouts
        self._gates = build_m0_distributed_heldout_acceptance_gates(
            training_runner_admission_sha256s=training_digests,
            holdout_runner_admission_sha256s=holdout_digests,
        )
        self._fixture_by_kind = {gate.kind: gate.fixture for gate in self._gates}


__all__ = [
    "DistributedRealRlvrM0CandidateEvaluator",
    "M0_DISTRIBUTED_HELDOUT_FIXTURE_SCHEMA",
    "M0_HELDOUT_FIXTURE_SCHEMA",
    "M0_HELDOUT_OBSERVATION_SCHEMA",
    "M0LiveEvaluatorError",
    "RealRlvrM0CandidateEvaluator",
    "build_m0_heldout_acceptance_gates",
    "build_m0_distributed_heldout_acceptance_gates",
]
