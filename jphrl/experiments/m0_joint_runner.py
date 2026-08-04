from __future__ import annotations

"""One-process coordinator for the real M0 T -> U -> V -> W -> X -> Y path.

The coordinator deliberately does not manufacture either of the two external
boundaries that are still deployment-specific:

* callers must provide the P Agent Service record as well as S, because S only
  retains the P digest and cannot prove that the source route was Agent Service;
* callers must provide a factory for real ``ProductionActivationWorker``
  adapters.  A training ``FSDPPPOActor`` is not a rollout worker and must not be
  presented as one merely to make Y pass.

All optimizer and recovery evidence is emitted by the existing T--Y modules.
This module only sequences them and persists their returned records outside the
source checkout.
"""

import hashlib
import inspect
import json
import math
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from jphrl.joint_release import CandidateArtifact, JointReleaseStore, ReleaseManifest
from jphrl.paths import repository_root, require_outside_repository
from jphrl.trajectory.areal_agent_service_adapter import (
    validate_agent_service_training_record,
)
from jphrl.trajectory.joint_credit_alignment import (
    validate_frozen_joint_credit_alignment,
)
from jphrl.trajectory.schema import JointVersion

if TYPE_CHECKING:
    from jphrl.training.areal_production_worker import LiveArealServingExportPair

PINNED_AREAL_COMMIT = "fee938eada49208a5aabdbc1095730a13076a349"
M0_RUN_SUMMARY_SCHEMA = "jph.m0-live-joint-update-summary.v1"
M0_SOURCE_ROUTE_SCHEMA = "jph.m0-agent-service-source.v1"
M0_RLVR_SOURCE_ROUTE_SCHEMA = "jph.m0-rlvr-workflow-source.v1"
M0_STREAM_SCHEMA = "jph.m0-single-item-stream.v1"
M0_BOOTSTRAP_POLICY_SCHEMA = "jph.m0-measured-parent-policy.v1"
M0_BOOTSTRAP_HARNESS_SCHEMA = "jph.m0-measured-parent-harness.v1"
M0_WORKER_CLEANUP_SCHEMA = "jph.m0-production-worker-cleanup.v1"

_STAGES = ("T", "U", "V", "W", "X", "Y")
_REQUIRED_SUITE_KINDS = (
    "policy_heldout",
    "harness_offpolicy",
    "joint_safety",
    "restart_recovery",
)
_SECRET_MARKERS = {
    "access_token",
    "admin_api_key",
    "api_key",
    "authorization",
    "credential",
    "github_token",
    "password",
    "refresh_token",
    "secret",
    "session_api_key",
    "token",
}


class M0JointRunnerError(RuntimeError):
    """Raised when a live M0 run cannot preserve the T--Y contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M0JointRunnerError(message)


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
        raise M0JointRunnerError("M0 record is not finite canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            secret = normalized in _SECRET_MARKERS or normalized.endswith(
                (
                    "_api_key",
                    "_credential",
                    "_password",
                    "_secret",
                    "_token",
                )
            )
            _require(not secret, f"credential field cannot enter M0: {path}.{key}")
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _read_json(path: str | Path, *, label: str) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    _require(source.is_file() and not source.is_symlink(), f"{label} is missing")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise M0JointRunnerError(f"{label} is not strict JSON") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    _assert_no_secrets(value, label)
    _canonical_json(value)
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> Path:
    _assert_no_secrets(value)
    payload = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _joint_version(value: object) -> JointVersion:
    _require(isinstance(value, Mapping), "M0 S JointVersion must be an object")
    fields = set(JointVersion.__dataclass_fields__)
    _require(set(value) == fields, "M0 S JointVersion field set differs")
    _require(
        all(isinstance(item, str) and bool(item) for item in value.values()),
        "M0 S JointVersion fields must be non-empty strings",
    )
    try:
        return JointVersion(**dict(value))
    except TypeError as exc:  # pragma: no cover - exact fields own this branch
        raise M0JointRunnerError("M0 S JointVersion is invalid") from exc


@dataclass(frozen=True)
class AgentServiceM0SourceRecords:
    """The separately persisted P/S pair required to prove the real route."""

    p_training_record: Mapping[str, object]
    s_joint_credit: Mapping[str, object]
    active_joint_version: JointVersion
    p_record_sha256: str
    s_record_sha256: str
    session_id: str
    trajectory_id: int
    export_style: str
    rollout_bridge_sha256: str
    rollout_sglang_mem_fraction_static: float
    expected_inference_engine_version: int


@dataclass(frozen=True)
class RLVRM0SourceRecords:
    """A dedicated RLVR runner envelope with no Agent Service identity."""

    runner_admission: Mapping[str, object]
    s_joint_credit: Mapping[str, object]
    active_joint_version: JointVersion
    runner_admission_sha256: str
    rlvr_pre_batch_record_sha256: str
    episode_trace_sha256: str
    s_record_sha256: str
    rollout_bridge_sha256: str
    rollout_sglang_mem_fraction_static: float
    mem_fraction_static_source_path: str
    expected_inference_engine_version: int


M0SourceRecords = Union[AgentServiceM0SourceRecords, RLVRM0SourceRecords]  # noqa: UP007


def load_m0_agent_service_source_records(
    *,
    p_training_record_path: str | Path,
    s_joint_credit_path: str | Path,
    rollout_bridge_record_path: str | Path,
) -> AgentServiceM0SourceRecords:
    """Load P and S and prove S came from a pre-batch Agent Service route.

    Requiring P here is intentional.  Passing only S is a source-proof gap and
    is rejected by the function signature instead of being papered over with a
    synthetic session or trajectory identity.
    """

    p_record = _read_json(
        require_outside_repository(p_training_record_path),
        label="P Agent Service record",
    )
    s_record = _read_json(
        require_outside_repository(s_joint_credit_path),
        label="S joint-credit record",
    )
    bridge = _read_json(
        require_outside_repository(rollout_bridge_record_path),
        label="AReaL rollout bridge record",
    )
    try:
        from jphrl.trajectory.areal_joint_bridge import (
            validate_areal_joint_bridge_record,
        )

        p_audit = validate_agent_service_training_record(p_record)
        active = _joint_version(s_record.get("joint_version"))
        s_audit = validate_frozen_joint_credit_alignment(
            s_record,
            active_joint_version=active,
        )
        policy_binding = bridge.get("policy_binding")
        _require(
            isinstance(policy_binding, Mapping), "bridge Policy binding is missing"
        )
        expected_engine_version = policy_binding.get(
            "expected_inference_engine_version"
        )
        _require(
            type(expected_engine_version) is int and expected_engine_version >= 0,
            "bridge inference engine version is invalid",
        )
        validate_areal_joint_bridge_record(
            bridge,
            expected_policy_version=expected_engine_version,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise M0JointRunnerError(str(exc)) from exc

    admissions = s_record.get("admissions")
    identity = s_record.get("identity")
    _require(
        isinstance(admissions, Mapping) and isinstance(identity, Mapping),
        "S admissions or identity are missing",
    )
    q_record = admissions.get("policy_admission_record")
    _require(isinstance(q_record, Mapping), "S does not retain its Q record")
    q_source = q_record.get("source")
    _require(isinstance(q_source, Mapping), "S/Q source lineage is missing")
    _require(
        q_source.get("agent_service_training_record_sha256")
        == p_record.get("record_sha256")
        == identity.get("source_training_record_sha256"),
        "P and S do not share one Agent Service training record",
    )

    p_identity = p_record.get("identity")
    archive = p_record.get("training_archive")
    _require(
        isinstance(p_identity, Mapping) and isinstance(archive, Mapping),
        "P route identity or training archive is missing",
    )
    sidecar = archive.get("interaction_sidecar")
    _require(isinstance(sidecar, Mapping), "P interaction sidecar is missing")
    bindings = sidecar.get("bindings")
    _require(
        isinstance(bindings, list)
        and bool(bindings)
        and all(
            isinstance(binding, Mapping)
            and binding.get("route_kind") == "agent-service-session"
            and binding.get("session_id") == p_identity.get("session_id")
            and binding.get("trajectory_id") == p_identity.get("trajectory_id")
            for binding in bindings
        ),
        "M0 source must be a real Agent Service pre-batch route",
    )
    _require(
        p_record.get("evidence_scope")
        == {
            "pre_batch_interaction_binding": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "P record contains an optimizer claim or lacks pre-batch binding",
    )
    _require(
        s_record.get("evidence_scope")
        == {
            "policy_samples_admitted": True,
            "harness_action_samples_admitted": True,
            "policy_advantages_aligned": True,
            "harness_advantages_aligned": True,
            "policy_optimizer_update": False,
            "harness_optimizer_update": False,
        },
        "S record contains an optimizer claim or incomplete frozen credit",
    )
    _require(
        p_identity.get("episode_id") == s_audit.get("episode_id")
        and p_identity.get("joint_version_id") == active.version_id,
        "P/S episode or JointVersion differs",
    )
    bridge_version = _joint_version(bridge.get("joint_version"))
    bridge_sidecar = bridge.get("interaction_adapter_sidecar")
    _require(
        bridge_version == active
        and bridge.get("episode_id") == p_identity.get("episode_id")
        and isinstance(bridge_sidecar, Mapping),
        "rollout bridge differs from the P/S episode or JointVersion",
    )
    bridge_bindings = bridge_sidecar.get("bindings")
    _require(
        isinstance(bridge_bindings, list)
        and len(bridge_bindings) == len(bindings)
        and all(
            isinstance(bridge_binding, Mapping)
            and isinstance(p_binding, Mapping)
            and bridge_binding.get("route_kind") == "agent-service-session"
            and all(
                bridge_binding.get(field) == p_binding.get(field)
                for field in (
                    "episode_id",
                    "model_call_id",
                    "session_id",
                    "trajectory_id",
                    "interaction_id",
                    "parent_interaction_id",
                    "ordinal",
                    "joint_version_id",
                    "route_kind",
                )
            )
            for bridge_binding, p_binding in zip(bridge_bindings, bindings)
        ),
        "rollout bridge is not the same Agent Service route as P",
    )
    bridge_harness = bridge.get("harness")
    bridge_decision = (
        bridge_harness.get("decision") if isinstance(bridge_harness, Mapping) else None
    )
    _require(
        s_audit.get("policy_model_call_ids")
        == [binding["model_call_id"] for binding in bridge_bindings]
        and isinstance(bridge_decision, Mapping)
        and s_audit.get("harness_decision_ids") == [bridge_decision.get("decision_id")],
        "rollout bridge Policy/Harness decisions differ from S",
    )
    runtime = bridge["policy_binding"].get("inference_runtime_contract")
    fixed = runtime.get("fixed") if isinstance(runtime, Mapping) else None
    server_args = fixed.get("server_args") if isinstance(fixed, Mapping) else None
    mem_fraction = (
        server_args.get("mem_fraction_static")
        if isinstance(server_args, Mapping)
        else None
    )
    _require(
        isinstance(mem_fraction, (int, float))
        and not isinstance(mem_fraction, bool)
        and math.isfinite(float(mem_fraction))
        and 0.28 <= float(mem_fraction) <= 0.30,
        "recorded rollout sglang.mem_fraction_static must be in [0.28, 0.30]",
    )
    session_id = p_identity.get("session_id")
    trajectory_id = p_identity.get("trajectory_id")
    _require(
        isinstance(session_id, str)
        and bool(session_id)
        and type(trajectory_id) is int
        and trajectory_id >= 0,
        "P session or trajectory identity is missing",
    )
    export_style = p_audit.get("export_style")
    _require(export_style in {"individual", "concat"}, "P export style is invalid")
    return AgentServiceM0SourceRecords(
        p_training_record=p_record,
        s_joint_credit=s_record,
        active_joint_version=active,
        p_record_sha256=str(p_record["record_sha256"]),
        s_record_sha256=str(s_record["record_sha256"]),
        session_id=session_id,
        trajectory_id=trajectory_id,
        export_style=str(export_style),
        rollout_bridge_sha256=str(bridge["record_sha256"]),
        rollout_sglang_mem_fraction_static=float(mem_fraction),
        expected_inference_engine_version=expected_engine_version,
    )


def load_m0_rlvr_source_records(
    *,
    runner_admission_path: str | Path,
    active_joint_version: JointVersion,
) -> RLVRM0SourceRecords:
    """Load the dedicated RLVR envelope without manufacturing Agent receipts."""

    record = _read_json(
        require_outside_repository(runner_admission_path),
        label="RLVR workflow runner admission",
    )
    try:
        from jphrl.trajectory.rlvr_workflow_admission import (
            load_rlvr_workflow_runner_admission,
        )

        loaded = load_rlvr_workflow_runner_admission(
            record,
            active_joint_version=active_joint_version,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M0JointRunnerError(str(exc)) from exc
    bridge = loaded.bridge_record
    policy = bridge.get("policy_binding")
    expected_engine_version = (
        policy.get("expected_inference_engine_version")
        if isinstance(policy, Mapping)
        else None
    )
    _require(
        type(expected_engine_version) is int and expected_engine_version >= 0,
        "RLVR bridge inference engine version is invalid",
    )
    mem_fraction = loaded.rollout_sglang_mem_fraction_static
    _require(
        isinstance(mem_fraction, (int, float))
        and not isinstance(mem_fraction, bool)
        and math.isfinite(float(mem_fraction))
        and 0.28 <= float(mem_fraction) <= 0.30,
        "recorded RLVR rollout sglang.mem_fraction_static must be in [0.28, 0.30]",
    )
    pre_batch = loaded.rlvr_pre_batch_record
    episode_trace_sha256 = pre_batch.get("source", {}).get(
        "episode_trace_sha256"
    )
    _require(
        loaded.route_kind == "rlvr-workflow"
        and _is_sha256(loaded.record_sha256)
        and _is_sha256(loaded.bridge_record_sha256)
        and _is_sha256(pre_batch.get("record_sha256"))
        and _is_sha256(episode_trace_sha256)
        and _is_sha256(loaded.s_joint_credit.get("record_sha256")),
        "RLVR runner source identity is invalid",
    )
    return RLVRM0SourceRecords(
        runner_admission=record,
        s_joint_credit=loaded.s_joint_credit,
        active_joint_version=loaded.joint_version,
        runner_admission_sha256=loaded.record_sha256,
        rlvr_pre_batch_record_sha256=str(pre_batch["record_sha256"]),
        episode_trace_sha256=str(episode_trace_sha256),
        s_record_sha256=str(loaded.s_joint_credit["record_sha256"]),
        rollout_bridge_sha256=loaded.bridge_record_sha256,
        rollout_sglang_mem_fraction_static=float(mem_fraction),
        mem_fraction_static_source_path=loaded.mem_fraction_static_source_path,
        expected_inference_engine_version=expected_engine_version,
    )


@dataclass(frozen=True)
class M0ArealActorSpec:
    model_path: str
    experiment_name: str
    trial_name: str
    learning_rate: float = 1e-6
    dtype: str = "bfloat16"
    optimizer_dtype: str = "float32"
    attention_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True
    max_new_tokens: int = 512

    def validate(self) -> Path:
        model = require_outside_repository(self.model_path)
        _require(model.is_dir() and not model.is_symlink(), "model path is unsafe")
        _require(
            isinstance(self.experiment_name, str)
            and bool(self.experiment_name)
            and isinstance(self.trial_name, str)
            and bool(self.trial_name),
            "AReaL experiment or trial name is missing",
        )
        _require(
            isinstance(self.learning_rate, (int, float))
            and not isinstance(self.learning_rate, bool)
            and math.isfinite(float(self.learning_rate))
            and self.learning_rate > 0.0,
            "AReaL learning rate must be finite and positive",
        )
        _require(
            self.dtype in {"bfloat16", "float32"}
            and self.optimizer_dtype in {"bfloat16", "float32"},
            "AReaL dtype is outside the audited M0 set",
        )
        _require(
            type(self.max_new_tokens) is int and self.max_new_tokens > 0,
            "AReaL max_new_tokens must be positive",
        )
        return model


@dataclass(frozen=True)
class M0AcceptanceGate:
    kind: str
    suite_id: str
    fixture: bytes
    metric_name: str
    minimum_score: float
    minimum_sample_count: int = 1

    def validate(self) -> None:
        _require(self.kind in _REQUIRED_SUITE_KINDS, "unknown M0 gate kind")
        _require(
            isinstance(self.suite_id, str)
            and bool(self.suite_id)
            and type(self.fixture) is bytes
            and bool(self.fixture)
            and isinstance(self.metric_name, str)
            and bool(self.metric_name),
            "M0 gate identity or fixture is missing",
        )
        _require(
            isinstance(self.minimum_score, (int, float))
            and not isinstance(self.minimum_score, bool)
            and math.isfinite(float(self.minimum_score))
            and type(self.minimum_sample_count) is int
            and self.minimum_sample_count > 0,
            "M0 gate threshold is invalid",
        )

    @property
    def fixture_sha256(self) -> str:
        self.validate()
        return hashlib.sha256(self.fixture).hexdigest()


@dataclass(frozen=True)
class M0JointRunConfig:
    artifact_root: str
    project_commit: str
    areal_root: str
    transaction_id: str
    macro_step: int
    rollout_sglang_mem_fraction_static: float = 0.29
    max_new_gpu_memory_gib: float = 26.0

    def validate(self) -> Path:
        target = require_outside_repository(self.artifact_root)
        _require(not target.exists(), "M0 artifact root must be new")
        _require(_is_git_sha(self.project_commit), "project commit is not a Git SHA-1")
        _require(
            isinstance(self.transaction_id, str) and bool(self.transaction_id),
            "M0 transaction ID is missing",
        )
        _require(
            type(self.macro_step) is int and self.macro_step >= 0, "bad macro step"
        )
        _require(
            isinstance(self.rollout_sglang_mem_fraction_static, (int, float))
            and 0.28 <= float(self.rollout_sglang_mem_fraction_static) <= 0.30,
            "M0 rollout sglang.mem_fraction_static must remain in [0.28, 0.30]",
        )
        _require(
            isinstance(self.max_new_gpu_memory_gib, (int, float))
            and 0.0 < float(self.max_new_gpu_memory_gib) <= 26.0,
            "M0 per-process GPU peak limit must be at most 26 GiB",
        )
        areal_root = Path(self.areal_root).expanduser().resolve()
        _require(
            areal_root.is_dir()
            and (areal_root / ".git").is_dir()
            and repository_root() not in areal_root.parents,
            "pinned AReaL checkout is missing or nested in the project",
        )
        return target


class M0CandidateEvaluator(ABC):
    """Real held-out evaluator.  It returns measurements, never verdicts."""

    @abstractmethod
    def observe(
        self,
        *,
        joint_version: JointVersion,
        gate: M0AcceptanceGate,
        actor: object,
        harness_policy: object,
    ) -> Sequence[object]:
        """Return native CandidateProbeObservation objects for one frozen gate."""


@dataclass(frozen=True)
class M0ActivationAssets:
    """Exact V/W and live HF-export lineage supplied to the Y worker factory.

    The Policy DCP paths remain lineage only. ``serving_exports`` is the native
    proof that the pinned actor produced HF weights, while the adapter must
    still update SGLang and expose its measured tensor digest through
    ``ProductionWorkerState`` after every side effect.
    """

    parent_release: ReleaseManifest
    candidate_release: ReleaseManifest
    serving_exports: LiveArealServingExportPair
    parent_joint_version: JointVersion
    candidate_joint_version: JointVersion
    parent_policy_dcp: str
    candidate_policy_dcp: str
    policy_parent_manifest_sha256: str
    policy_candidate_manifest_sha256: str
    parent_harness_rollout_checkpoint: str
    parent_harness_checkpoint_sha256: str
    candidate_harness_checkpoint: str
    candidate_harness_checkpoint_sha256: str
    candidate_harness_parameter_sha256: str
    joint_safety_fixture: bytes
    candidate_joint_safety_output_sha256: str
    recorded_rollout_sglang_mem_fraction_static: float
    max_new_gpu_memory_gib: float


# A factory must close any controllers it starts if construction itself raises;
# once it returns, the runner owns and closes every worker in the sequence.
ProductionWorkerFactory = Callable[[M0ActivationAssets], Sequence[object]]
# Kept optional for existing programmatic callers; a real experiment CLI must
# supply a live nvidia-smi/process audit callback for both launch phases.
GPULaunchGuard = Callable[[str], None]


@dataclass(frozen=True)
class M0JointRunResult:
    artifact_root: Path
    summary_path: Path
    active_release_id: str
    candidate_joint_version: JointVersion
    production_attestation_path: Path
    production_attestation_sha256: str
    production_worker_cleanup_path: Path
    peak_gpu_memory_gib: float | None


class _StageLedger:
    def __init__(self) -> None:
        self._completed: list[str] = []

    def complete(self, stage: str) -> None:
        expected = _STAGES[len(self._completed)] if len(self._completed) < 6 else None
        _require(stage == expected, f"M0 stage order requires {expected}, got {stage}")
        self._completed.append(stage)

    @property
    def completed(self) -> tuple[str, ...]:
        return tuple(self._completed)


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise M0JointRunnerError(f"cannot inspect Git checkout {root}") from exc
    return result.stdout.strip()


def _validate_source_checkouts(config: M0JointRunConfig) -> None:
    project = repository_root()
    areal = Path(config.areal_root).expanduser().resolve()
    _require(
        _git_output(project, "rev-parse", "HEAD") == config.project_commit,
        "project HEAD differs from the requested experiment commit",
    )
    _require(
        not _git_output(project, "status", "--porcelain=v1", "--untracked-files=all"),
        "project checkout must be clean before a real M0 update",
    )
    _require(
        _git_output(areal, "rev-parse", "HEAD") == PINNED_AREAL_COMMIT,
        "AReaL checkout differs from the pinned commit",
    )
    _require(
        not _git_output(areal, "status", "--porcelain=v1", "--untracked-files=all"),
        "AReaL checkout must be clean before a real M0 update",
    )


def build_pinned_areal_actor_config(spec: M0ArealActorSpec) -> object:
    """Build the audited config with the pinned AReaL dataclass itself."""

    model_path = spec.validate()
    try:
        from areal.api.cli_args import OptimizerConfig, PPOActorConfig
    except (ImportError, ModuleNotFoundError) as exc:
        raise M0JointRunnerError(
            "pinned AReaL config dataclasses are unavailable"
        ) from exc
    optimizer = OptimizerConfig(
        type="adam",
        lr=float(spec.learning_rate),
        weight_decay=0.0,
        lr_scheduler_type="constant",
        warmup_steps_proportion=0.0,
    )
    actor_config = PPOActorConfig(
        experiment_name=spec.experiment_name,
        trial_name=spec.trial_name,
        path=str(model_path),
        backend="fsdp:d1",
        attn_impl=spec.attention_implementation,
        dtype=spec.dtype,
        optimizer_dtype=spec.optimizer_dtype,
        optimizer=optimizer,
        disable_dropout=True,
        gradient_checkpointing=spec.gradient_checkpointing,
        ppo_n_minibatches=1,
        eps_clip_higher=None,
        c_clip=None,
        m2_threshold=None,
        reward_norm=None,
        reward_scaling=1.0,
        reward_bias=0.0,
        overlong_reward_penalty=False,
        mask_no_eos_with_zero=False,
        adv_norm=None,
        kl_ctl=0.0,
        recompute_logprob=False,
        use_decoupled_loss=False,
        rejection_sampling=None,
        importance_sampling_level="token",
        log_agent_stats=False,
        use_cispo_loss=False,
        use_sapo_loss=False,
        max_new_tokens=spec.max_new_tokens,
        is_critic=False,
    )
    _require(
        type(actor_config) is PPOActorConfig,
        "actor config is not the pinned PPOActorConfig dataclass",
    )
    try:
        from jphrl.training.areal_policy_optimizer import (
            validate_m0_areal_actor_config,
        )

        carrier = type("M0PinnedActorConfigCarrier", (), {"config": actor_config})()
        validate_m0_areal_actor_config(carrier)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise M0JointRunnerError(str(exc)) from exc
    return actor_config


def initialize_pinned_areal_actor(
    spec: M0ArealActorSpec,
    *,
    inference_engine_version: int,
) -> object:
    """Instantiate and initialize the exact pinned ``FSDPPPOActor`` type."""

    spec.validate()
    _require(
        type(inference_engine_version) is int and inference_engine_version >= 0,
        "inference engine version is invalid",
    )
    required_env = {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
    _require(
        all(os.environ.get(key) == value for key, value in required_env.items()),
        "single-worker AReaL distributed environment is not initialized",
    )
    _require(
        bool(os.environ.get("MASTER_ADDR")) and bool(os.environ.get("MASTER_PORT")),
        "AReaL MASTER_ADDR/MASTER_PORT are missing",
    )
    actor_config = build_pinned_areal_actor_config(spec)
    try:
        from areal.api import FinetuneSpec
        from areal.engine.fsdp_engine import FSDPPPOActor
    except (ImportError, ModuleNotFoundError) as exc:
        raise M0JointRunnerError("pinned AReaL is unavailable") from exc

    _validate_imported_actor_source(FSDPPPOActor)
    actor = FSDPPPOActor(actor_config)
    try:
        actor.create_process_group()
        actor.initialize(
            None,
            FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1),
        )
        actor.set_version(inference_engine_version)
    except BaseException:
        try:
            actor.destroy()
        except Exception as destroy_error:
            raise M0JointRunnerError(
                "AReaL initialization failed and actor cleanup also failed"
            ) from destroy_error
        raise
    return actor


def _validate_imported_actor_source(actor_type: type[object]) -> Path:
    actor_source = Path(inspect.getfile(actor_type)).resolve()
    raw_areal_root = os.environ.get("JPH_AREAL_ROOT")
    if raw_areal_root:
        configured_areal = Path(raw_areal_root).expanduser().resolve()
        _require(
            actor_source == configured_areal
            or configured_areal in actor_source.parents,
            "imported FSDPPPOActor is outside JPH_AREAL_ROOT",
        )
    return actor_source


def _load_harness_behavior_checkpoint(
    path: str | Path,
    *,
    active_joint_version: JointVersion,
) -> tuple[object, Mapping[str, object], str]:
    source = require_outside_repository(path)
    record = _read_json(source, label="Torch Harness rollout checkpoint")
    try:
        from jphrl.harness.torch_learning import (
            TorchHarnessOptimizer,
            TorchHarnessPolicy,
            load_torch_harness_rollout_checkpoint,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise M0JointRunnerError("Torch Harness implementation is unavailable") from exc
    policy = load_torch_harness_rollout_checkpoint(record, device="cpu")
    _require(
        type(policy) is TorchHarnessPolicy, "Harness behavior type is not real Torch"
    )
    _require(
        policy.version == active_joint_version.harness_controller,
        "Harness behavior checkpoint differs from lag-zero JointVersion",
    )
    trainer = TorchHarnessOptimizer(policy)
    _require(
        type(trainer.optimizer).__module__.startswith("torch.optim"),
        "Harness optimizer is not real Torch Adam",
    )
    digest = record.get("record_sha256")
    _require(_is_sha256(digest), "Harness rollout checkpoint digest is invalid")
    return trainer, record, str(digest)


def _acceptance_spec(gates: tuple[M0AcceptanceGate, ...]) -> object:
    _require(
        tuple(gate.kind for gate in gates) == _REQUIRED_SUITE_KINDS,
        "M0 acceptance gates must be complete and ordered",
    )
    try:
        from jphrl.training.candidate_acceptance import (
            CandidateAcceptanceSpec,
            CandidateAcceptanceSuite,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise M0JointRunnerError(
            "candidate acceptance implementation is unavailable"
        ) from exc
    suites = []
    for gate in gates:
        gate.validate()
        suites.append(
            CandidateAcceptanceSuite(
                kind=gate.kind,
                suite_id=gate.suite_id,
                fixture_sha256=gate.fixture_sha256,
                metric_name=gate.metric_name,
                minimum_score=float(gate.minimum_score),
                minimum_sample_count=gate.minimum_sample_count,
            )
        )
    spec = CandidateAcceptanceSpec(tuple(suites))
    spec.validate()
    return spec


def _source_admission_sha256(source: M0SourceRecords) -> str:
    if type(source) is AgentServiceM0SourceRecords:
        return source.p_record_sha256
    _require(type(source) is RLVRM0SourceRecords, "unsupported M0 source type")
    return source.runner_admission_sha256


def _source_s_sha256(source: M0SourceRecords) -> str:
    if type(source) is AgentServiceM0SourceRecords:
        return source.s_record_sha256
    _require(type(source) is RLVRM0SourceRecords, "unsupported M0 source type")
    return source.s_record_sha256


def _persist_source_inputs(source: M0SourceRecords, inputs: Path) -> None:
    if type(source) is AgentServiceM0SourceRecords:
        _write_new_json(inputs / "p-agent-service.json", source.p_training_record)
    else:
        _require(type(source) is RLVRM0SourceRecords, "unsupported M0 source type")
        _write_new_json(
            inputs / "rlvr-runner-admission.json", source.runner_admission
        )
    _write_new_json(inputs / "s-joint-credit.json", source.s_joint_credit)


def _source_summary(source: M0SourceRecords) -> dict[str, object]:
    if type(source) is AgentServiceM0SourceRecords:
        return {
            "route_schema": M0_SOURCE_ROUTE_SCHEMA,
            "route_kind": "agent-service-session",
            "p_training_record_sha256": source.p_record_sha256,
            "s_joint_credit_sha256": source.s_record_sha256,
            "rollout_bridge_sha256": source.rollout_bridge_sha256,
            "session_id": source.session_id,
            "trajectory_id": source.trajectory_id,
            "export_style": source.export_style,
        }
    _require(type(source) is RLVRM0SourceRecords, "unsupported M0 source type")
    return {
        "route_schema": M0_RLVR_SOURCE_ROUTE_SCHEMA,
        "route_kind": "rlvr-workflow",
        "runner_admission_sha256": source.runner_admission_sha256,
        "rlvr_pre_batch_record_sha256": source.rlvr_pre_batch_record_sha256,
        "episode_trace_sha256": source.episode_trace_sha256,
        "s_joint_credit_sha256": source.s_record_sha256,
        "rollout_bridge_sha256": source.rollout_bridge_sha256,
        "session_id": None,
        "trajectory_id": None,
        "mem_fraction_static_source_path": source.mem_fraction_static_source_path,
    }


def _bootstrap_parent_release(
    store: JointReleaseStore,
    *,
    source: M0SourceRecords,
    harness_checkpoint_sha256: str,
    policy_engine_version: int,
) -> ReleaseManifest:
    _require(
        store.read_active() is None,
        "new M0 release store already has an active release",
    )
    policy = CandidateArtifact(
        component="policy",
        version=source.active_joint_version.policy,
        payload={
            "schema_version": M0_BOOTSTRAP_POLICY_SCHEMA,
            "joint_version_id": source.active_joint_version.version_id,
            "source_route_kind": _source_summary(source)["route_kind"],
            "source_admission_sha256": _source_admission_sha256(source),
            "source_s_sha256": _source_s_sha256(source),
            "observed_policy_engine_version": policy_engine_version,
        },
    )
    harness = CandidateArtifact(
        component="harness",
        version=source.active_joint_version.harness_controller,
        payload={
            "schema_version": M0_BOOTSTRAP_HARNESS_SCHEMA,
            "joint_version_id": source.active_joint_version.version_id,
            "source_route_kind": _source_summary(source)["route_kind"],
            "source_admission_sha256": _source_admission_sha256(source),
            "source_s_sha256": _source_s_sha256(source),
            "harness_rollout_checkpoint_sha256": harness_checkpoint_sha256,
        },
    )
    staged = store.stage(
        joint_version=source.active_joint_version,
        policy=policy,
        harness=harness,
        expected_active_release_id=None,
    )
    return store.activate(release_id=staged.release_id, expected_active_release_id=None)


def _stream_record(kind: str, item_sha256: str) -> dict[str, object]:
    _require(kind in {"rollout", "dataloader"}, "unknown M0 stream kind")
    _require(_is_sha256(item_sha256), "M0 stream item digest is invalid")
    record: dict[str, object] = {
        "schema_version": M0_STREAM_SCHEMA,
        "kind": kind,
        "items": [item_sha256],
    }
    record["record_sha256"] = _sha256(record)
    return record


def _run_harness_continuation_step(
    policy: object,
    optimizer: object,
    s_record: Mapping[str, object],
) -> None:
    """Run one measured Adam step; W owns version advancement and evidence."""

    try:
        import torch

        from jphrl.harness.controller import HarnessState
        from jphrl.harness.torch_learning import ACTION_IDS, TorchHarnessPolicy
    except (ImportError, ModuleNotFoundError) as exc:
        raise M0JointRunnerError("Torch Harness continuation is unavailable") from exc
    _require(type(policy) is TorchHarnessPolicy, "W Harness policy type differs")
    _require(type(optimizer) is torch.optim.Adam, "W Harness optimizer is not Adam")
    samples = s_record.get("harness_samples")
    _require(isinstance(samples, list) and bool(samples), "S has no Harness samples")
    states: list[Any] = []
    masks: list[list[bool]] = []
    selected: list[int] = []
    old_logprobs: list[float] = []
    advantages: list[float] = []
    loss_masks: list[int] = []
    for sample in samples:
        _require(isinstance(sample, Mapping), "S Harness sample is invalid")
        action = sample.get("action")
        _require(isinstance(action, Mapping), "S Harness action is missing")
        state = action.get("state")
        _require(
            isinstance(state, Mapping)
            and set(state) == set(HarnessState.__dataclass_fields__),
            "S Harness state differs from schema",
        )
        states.append(HarnessState(**dict(state)))
        action_ids = tuple(action.get("action_ids", ()))
        action_mask = action.get("action_mask")
        chosen = action.get("action")
        _require(
            action_ids == ACTION_IDS
            and isinstance(action_mask, list)
            and len(action_mask) == len(ACTION_IDS)
            and all(type(value) is bool for value in action_mask)
            and chosen in ACTION_IDS,
            "S Harness action contract differs",
        )
        masks.append(list(action_mask))
        selected.append(ACTION_IDS.index(str(chosen)))
        old_logprobs.append(float(action["old_harness_logprob"]))
        advantages.append(float(sample["masked_advantage"]))
        loss_masks.append(int(action["harness_loss_mask"]))
    _require(sum(loss_masks) > 0, "W Harness continuation has no trainable action")
    logits = policy.logits_for(states)
    device = logits.device
    dtype = logits.dtype
    mask_tensor = torch.tensor(masks, dtype=torch.bool, device=device)
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
    current = torch.log_softmax(logits.masked_fill(~mask_tensor, -torch.inf), dim=-1)
    current = current.gather(1, selected_tensor[:, None])[:, 0]
    old = torch.tensor(old_logprobs, dtype=dtype, device=device)
    advantage = torch.tensor(advantages, dtype=dtype, device=device)
    loss_mask = torch.tensor(loss_masks, dtype=dtype, device=device)
    ratio = torch.exp(current - old)
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
    loss = -(torch.minimum(ratio * advantage, clipped) * loss_mask).sum()
    loss = loss / loss_mask.sum()
    _require(bool(torch.isfinite(loss)), "W Harness continuation loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().double().square().sum()
            for parameter in policy.parameters()
            if parameter.grad is not None
        )
    )
    _require(
        bool(torch.isfinite(gradient_norm)) and float(gradient_norm.item()) > 0.0,
        "W Harness continuation has no finite gradient",
    )
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()


def _peak_gpu_memory_gib() -> float | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_reserved()) / float(1024**3)


def _enforce_gpu_peak(config: M0JointRunConfig) -> float | None:
    peak = _peak_gpu_memory_gib()
    _require(
        peak is None or peak <= float(config.max_new_gpu_memory_gib),
        "M0 process exceeded its configured GPU peak-memory gate",
    )
    return peak


def _require_areal_update_batch(value: object) -> list[dict[str, Any]]:
    _require(
        isinstance(value, list)
        and bool(value)
        and all(type(sample) is dict for sample in value),
        "AReaL ppo_update requires a non-empty list of tensor dictionaries",
    )
    return value


def _joint_safety_production_probe_sha256(
    acceptance_report: Mapping[str, object],
) -> str:
    """Return X's raw production-probe digest, never its aggregate digest."""

    critical_suites = acceptance_report.get("critical_suites")
    _require(
        isinstance(critical_suites, list),
        "X acceptance has no critical suite evidence",
    )
    joint_safety = [
        suite
        for suite in critical_suites
        if isinstance(suite, Mapping)
        and isinstance(suite.get("spec"), Mapping)
        and suite["spec"].get("kind") == "joint_safety"
    ]
    _require(len(joint_safety) == 1, "X has no unique joint-safety result")
    probe = joint_safety[0].get("probe")
    _require(isinstance(probe, Mapping), "X joint-safety probe is invalid")
    digest = probe.get("production_probe_output_sha256")
    size = probe.get("production_probe_output_size_bytes")
    observations = probe.get("observations")
    _require(
        _is_sha256(digest)
        and type(size) is int
        and size > 0
        and isinstance(observations, list)
        and len(observations) == 1
        and isinstance(observations[0], Mapping)
        and observations[0].get("production_probe_output_sha256") == digest
        and observations[0].get("production_probe_output_size_bytes") == size,
        "X joint-safety raw production-probe identity is invalid",
    )
    return str(digest)


def _attach_joint_safety_production_probe(
    observations: Sequence[object],
    *,
    gate_kind: str,
    production_probe_output: bytes,
) -> tuple[object, ...]:
    """Attach framework-owned live serving identity to X's raw measurement."""

    values = tuple(observations)
    if gate_kind != "joint_safety":
        _require(
            all(
                getattr(value, "production_probe_output", None) is None
                for value in values
            ),
            "only joint_safety may carry a production probe output",
        )
        return values
    try:
        from jphrl.training.candidate_acceptance import CandidateProbeObservation
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise M0JointRunnerError("candidate observation type is unavailable") from exc
    _require(
        type(production_probe_output) is bytes
        and bool(production_probe_output)
        and len(values) == 1
        and type(values[0]) is CandidateProbeObservation,
        "joint_safety requires exactly one native raw observation",
    )
    observation = values[0]
    supplied = observation.production_probe_output
    _require(
        supplied is None or supplied == production_probe_output,
        "joint_safety evaluator supplied crossed production probe bytes",
    )
    return (
        CandidateProbeObservation(
            sample_id=observation.sample_id,
            metric_value=observation.metric_value,
            output=observation.output,
            production_probe_output=production_probe_output,
        ),
    )


def _destroy_actor(actor: object) -> None:
    destroy = getattr(actor, "destroy", None)
    _require(callable(destroy), "AReaL actor destroy is unavailable")
    destroy()
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - actor requires torch
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_gpu_launch_guard(
    guard: GPULaunchGuard | None,
    *,
    phase: str,
) -> None:
    """Run the caller-owned live GPU audit immediately before one launch."""

    _require(
        phase in {"training-actor", "production-sglang"},
        "unknown GPU launch phase",
    )
    if guard is None:
        return
    _require(callable(guard), "GPU launch guard is not callable")
    try:
        result = guard(phase)
    except Exception as exc:
        raise M0JointRunnerError(
            f"GPU launch guard rejected phase {phase}"
        ) from exc
    _require(
        result is None,
        f"GPU launch guard for {phase} must return None after a live audit",
    )


def _initialize_actor_after_gpu_guard(
    actor_spec: M0ArealActorSpec,
    *,
    inference_engine_version: int,
    gpu_launch_guard: GPULaunchGuard | None,
) -> object:
    _run_gpu_launch_guard(gpu_launch_guard, phase="training-actor")
    return initialize_pinned_areal_actor(
        actor_spec,
        inference_engine_version=inference_engine_version,
    )


def _start_production_workers_after_gpu_guard(
    factory: ProductionWorkerFactory,
    assets: M0ActivationAssets,
    *,
    gpu_launch_guard: GPULaunchGuard | None,
) -> Sequence[object]:
    _run_gpu_launch_guard(gpu_launch_guard, phase="production-sglang")
    return factory(assets)


def _cleanup_production_workers(
    workers: Sequence[object],
    *,
    record_path: Path,
    summary_record_sha256: str | None,
    production_attestation_sha256: str | None,
    release_store: JointReleaseStore | None = None,
    expected_active_release_id: str | None = None,
    factory_completed: bool = True,
) -> Path:
    """Close every spawned rollout worker and persist non-activation evidence."""

    outcomes: list[dict[str, object]] = []
    all_returned = True
    for index, worker in enumerate(workers):
        close = getattr(worker, "close", None)
        method_name = "close"
        if not callable(close):
            close = getattr(worker, "destroy", None)
            method_name = "destroy"
        returned = False
        error_type: str | None = None
        if callable(close):
            try:
                close()
                returned = True
            except Exception as exc:  # noqa: BLE001 - attempt remaining workers
                error_type = type(exc).__name__
        else:
            method_name = "unavailable"
            error_type = "MissingCleanupMethod"
        all_returned = all_returned and returned
        outcomes.append(
            {
                "worker_index": index,
                "worker_type": f"{type(worker).__module__}.{type(worker).__qualname__}",
                "cleanup_method": method_name,
                "returned_without_error": returned,
                "error_type": error_type,
            }
        )
    active_release_id_after_cleanup: str | None = None
    active_release_unchanged: bool | None = None
    if expected_active_release_id is not None:
        _require(
            release_store is not None,
            "successful production cleanup requires the release store",
        )
        active = release_store.read_active()
        active_release_id_after_cleanup = (
            None if active is None else active.release_id
        )
        active_release_unchanged = (
            active_release_id_after_cleanup == expected_active_release_id
        )
    record: dict[str, object] = {
        "schema_version": M0_WORKER_CLEANUP_SCHEMA,
        "summary_record_sha256": summary_record_sha256,
        "production_attestation_sha256": production_attestation_sha256,
        "factory_completed": factory_completed,
        "workers": outcomes,
        "all_cleanup_calls_returned": all_returned,
        "expected_active_release_id": expected_active_release_id,
        "active_release_id_after_cleanup": active_release_id_after_cleanup,
        "active_release_unchanged": active_release_unchanged,
        "evidence_scope": {
            "cleanup_calls_returned": all_returned,
            "factory_partial_startup_cleanup_verified": (
                None if factory_completed else False
            ),
            "gpu_process_absence_verified": False,
            "active_release_rechecked": expected_active_release_id is not None,
            "production_attestation_is_point_in_time": True,
        },
    }
    record["record_sha256"] = _sha256(record)
    path = _write_new_json(record_path, record)
    _require(
        all_returned and active_release_unchanged is not False,
        "production rollout cleanup failed or changed the active release; "
        f"inspect {path}",
    )
    return path


class _M0JointUpdateRunner:
    """Execute one real, unpublished-then-activated M0 joint update."""

    def __init__(
        self,
        *,
        source: M0SourceRecords,
        actor_spec: M0ArealActorSpec,
        run_config: M0JointRunConfig,
        harness_behavior_checkpoint: str | Path,
        acceptance_gates: Sequence[M0AcceptanceGate],
        evaluator: M0CandidateEvaluator,
        production_worker_factory: ProductionWorkerFactory | None,
        gpu_launch_guard: GPULaunchGuard | None = None,
    ) -> None:
        self.source = source
        self.actor_spec = actor_spec
        self.run_config = run_config
        self.harness_behavior_checkpoint = str(
            Path(harness_behavior_checkpoint).expanduser().resolve()
        )
        self.acceptance_gates = tuple(acceptance_gates)
        self.evaluator = evaluator
        self.production_worker_factory = production_worker_factory
        self.gpu_launch_guard = gpu_launch_guard

    def _preflight(self) -> tuple[Path, object]:
        root = self.run_config.validate()
        self.actor_spec.validate()
        _require(
            type(self.evaluator) is not M0CandidateEvaluator
            and isinstance(self.evaluator, M0CandidateEvaluator),
            "M0 requires a concrete raw-observation evaluator",
        )
        _require(
            callable(self.production_worker_factory),
            "Y requires a real ProductionActivationWorker factory; the project has "
            "no built-in AReaL inference-service adapter yet",
        )
        _require(
            self.gpu_launch_guard is None or callable(self.gpu_launch_guard),
            "GPU launch guard is not callable",
        )
        _require(
            self.source.rollout_sglang_mem_fraction_static
            == float(self.run_config.rollout_sglang_mem_fraction_static),
            "runner resource setting differs from the recorded rollout launch contract",
        )
        spec = _acceptance_spec(self.acceptance_gates)
        _validate_source_checkouts(self.run_config)
        return root, spec

    def run(self) -> M0JointRunResult:
        root, acceptance_spec = self._preflight()
        root.mkdir(parents=True, mode=0o700)
        os.chmod(root, 0o700)
        ledger = _StageLedger()
        actor: object | None = None
        production_workers: tuple[object, ...] = ()
        release_store: JointReleaseStore | None = None
        expected_active_release_after_cleanup: str | None = None
        summary_record_sha256: str | None = None
        production_attestation_sha256: str | None = None
        cleanup_path = root / "production-worker-cleanup.json"
        try:
            from jphrl.harness.torch_learning import TorchHarnessOptimizer
            from jphrl.training.areal_policy_candidate import (
                run_areal_policy_candidate_update,
            )
            from jphrl.training.areal_policy_optimizer import (
                build_areal_external_advantage_batch,
                materialize_areal_ppo_update_tensors,
                validate_areal_external_advantage_batch,
            )
            from jphrl.training.areal_production_worker import (
                build_production_probe_output,
                materialize_areal_serving_export_pair,
            )
            from jphrl.training.candidate_acceptance import (
                build_production_candidate_artifacts,
                run_joint_candidate_acceptance,
            )
            from jphrl.training.joint_activation import (
                ProductionActivationWorker,
                ProductionJointActivationController,
                ProductionProbeSpec,
                authorize_production_activation,
            )
            from jphrl.training.joint_step import seal_joint_candidate_bundle
            from jphrl.training.production_checkpoint import (
                RuntimeCursorState,
                RuntimeTopology,
                capture_rank_runtime_state,
                save_production_joint_checkpoint,
                verify_exact_joint_recovery,
            )

            admission = build_areal_external_advantage_batch(
                self.source.s_joint_credit,
                active_joint_version=self.source.active_joint_version,
            )
            admission_audit = validate_areal_external_advantage_batch(
                admission,
                active_joint_version=self.source.active_joint_version,
            )
            _require(
                admission_audit.inference_engine_version
                == self.source.expected_inference_engine_version,
                "S Policy engine version differs from the recorded rollout bridge",
            )
            inputs = root / "inputs"
            _persist_source_inputs(self.source, inputs)
            _write_new_json(inputs / "t-policy-admission.json", admission)

            trainer, _behavior_checkpoint, harness_parent_sha = (
                _load_harness_behavior_checkpoint(
                    self.harness_behavior_checkpoint,
                    active_joint_version=self.source.active_joint_version,
                )
            )
            _require(
                type(trainer) is TorchHarnessOptimizer,
                "Harness trainer type differs from the production implementation",
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ModuleNotFoundError:  # pragma: no cover - actor requires torch
                pass
            actor = _initialize_actor_after_gpu_guard(
                self.actor_spec,
                inference_engine_version=admission_audit.inference_engine_version,
                gpu_launch_guard=self.gpu_launch_guard,
            )
            _enforce_gpu_peak(self.run_config)

            store = JointReleaseStore(root / "release-store")
            release_store = store
            parent_release = _bootstrap_parent_release(
                store,
                source=self.source,
                harness_checkpoint_sha256=harness_parent_sha,
                policy_engine_version=admission_audit.inference_engine_version,
            )

            policy_receipt = run_areal_policy_candidate_update(
                admission,
                source_joint_credit_record=self.source.s_joint_credit,
                actor=actor,
                active_joint_version=self.source.active_joint_version,
                candidate_root=root / "policy-candidate",
                project_root=repository_root(),
                transaction_id=self.run_config.transaction_id,
                project_commit=self.run_config.project_commit,
                areal_commit=PINNED_AREAL_COMMIT,
                device="cuda" if _peak_gpu_memory_gib() is not None else "cpu",
            )
            ledger.complete("T")
            _enforce_gpu_peak(self.run_config)

            harness_result = trainer.update_from_frozen_joint_credit(
                self.source.s_joint_credit,
                active_joint_version=self.source.active_joint_version,
                checkpoint_path=root / "harness-candidate" / "harness-candidate.pt",
            )
            harness_receipt = harness_result.evidence.to_record()
            _write_new_json(
                root / "harness-candidate" / "harness-candidate-evidence.json",
                harness_receipt,
            )
            ledger.complete("U")

            bundle = seal_joint_candidate_bundle(
                seal_root=root / "joint-candidate",
                project_root=repository_root(),
                policy_receipt=policy_receipt,
                harness_receipt=harness_receipt,
                active_joint_version=self.source.active_joint_version,
                parent_release_id=parent_release.release_id,
                macro_step_id=self.run_config.transaction_id,
                actor_public_version=actor.get_version(),
                harness_public_version=trainer.policy.version,
            )
            _write_new_json(root / "joint-candidate" / "bundle.json", bundle)
            ledger.complete("V")

            source_s_sha256 = _source_s_sha256(self.source)
            rollout_stream = _stream_record("rollout", source_s_sha256)
            dataloader_stream = _stream_record(
                "dataloader", admission_audit.record_sha256
            )
            _write_new_json(inputs / "rollout-stream.json", rollout_stream)
            _write_new_json(inputs / "dataloader-stream.json", dataloader_stream)
            topology = RuntimeTopology(
                1, 1, 1, 1, ("cuda:0" if _peak_gpu_memory_gib() is not None else "cpu",)
            )
            rank_state = capture_rank_runtime_state(
                rank=0,
                local_rank=0,
                device=topology.rank_to_device[0],
                harness_policy=harness_result.candidate_policy,
            )
            checkpoint_manifest = save_production_joint_checkpoint(
                checkpoint_root=root / "production-checkpoint",
                project_root=repository_root(),
                joint_candidate_bundle=bundle,
                actor=actor,
                harness_policy=harness_result.candidate_policy,
                topology=topology,
                rank_states=(rank_state,),
                macro_step=self.run_config.macro_step,
                rollout_cursor=RuntimeCursorState(
                    name="rollout",
                    position=0,
                    source_sha256=str(rollout_stream["record_sha256"]),
                    pending_item_sha256=source_s_sha256,
                ),
                dataloader_cursor=RuntimeCursorState(
                    name="dataloader",
                    position=0,
                    source_sha256=str(dataloader_stream["record_sha256"]),
                    pending_item_sha256=admission_audit.record_sha256,
                ),
            )

            def policy_continuation(live_actor: object) -> None:
                update_batch = materialize_areal_ppo_update_tensors(
                    admission,
                    actor=live_actor,
                    active_joint_version=self.source.active_joint_version,
                    device="cuda" if _peak_gpu_memory_gib() is not None else "cpu",
                )
                live_actor.ppo_update(_require_areal_update_batch(update_batch))

            def harness_continuation(policy: object, optimizer: object) -> None:
                _run_harness_continuation_step(
                    policy,
                    optimizer,
                    self.source.s_joint_credit,
                )

            live_recovery = verify_exact_joint_recovery(
                checkpoint_manifest,
                actor=actor,
                current_topology=topology,
                rank=0,
                admission_record=admission,
                device="cuda" if _peak_gpu_memory_gib() is not None else "cpu",
                run_policy_optimizer_step=policy_continuation,
                run_harness_optimizer_step=harness_continuation,
            )
            _write_new_json(
                root / "production-checkpoint" / "w-live-recovery.json",
                live_recovery.record,
            )
            ledger.complete("W")
            _enforce_gpu_peak(self.run_config)

            serving_exports = materialize_areal_serving_export_pair(
                actor=actor,
                policy_candidate_record=policy_receipt,
                export_root=root / "serving-exports",
                parent_joint_version=self.source.active_joint_version,
                candidate_joint_version=live_recovery.candidate_joint_version,
            )
            policy_artifact, harness_artifact = build_production_candidate_artifacts(
                joint_candidate_bundle=bundle,
                checkpoint_manifest=checkpoint_manifest,
                live_serving_exports=serving_exports,
            )
            candidate_release = store.stage(
                joint_version=live_recovery.candidate_joint_version,
                policy=policy_artifact,
                harness=harness_artifact,
                expected_active_release_id=parent_release.release_id,
            )

            gate_by_kind = {gate.kind: gate for gate in self.acceptance_gates}
            policy_checkpoints = policy_receipt["checkpoints"]
            joint_gate = gate_by_kind["joint_safety"]
            candidate_production_probe_output = build_production_probe_output(
                fixture=joint_gate.fixture,
                target_release_id=candidate_release.release_id,
                target_joint_version=live_recovery.candidate_joint_version,
                policy_engine_version=(
                    serving_exports.candidate.policy_engine_version
                ),
                policy_checkpoint_sha256=(
                    serving_exports.candidate.source_dcp_manifest_sha256
                ),
                serving_parameter_sha256=(
                    serving_exports.candidate.serving_parameter_sha256
                ),
                harness_checkpoint_sha256=str(
                    harness_receipt["checkpoint_sha256"]
                ),
                harness_parameter_sha256=str(
                    harness_receipt["parameter_digest_after"]
                ),
            )
            probes: dict[str, Callable[..., Sequence[object]]] = {}
            for kind, gate in gate_by_kind.items():

                def observe(
                    version: JointVersion,
                    _suite: object,
                    *,
                    frozen_gate: M0AcceptanceGate = gate,
                ) -> Sequence[object]:
                    _require(
                        version == live_recovery.candidate_joint_version,
                        "X evaluator received a crossed candidate JointVersion",
                    )
                    return _attach_joint_safety_production_probe(
                        self.evaluator.observe(
                            joint_version=version,
                            gate=frozen_gate,
                            actor=actor,
                            harness_policy=live_recovery.restored_harness_policy,
                        ),
                        gate_kind=frozen_gate.kind,
                        production_probe_output=candidate_production_probe_output,
                    )

                probes[kind] = observe
            live_acceptance = run_joint_candidate_acceptance(
                joint_candidate_bundle=bundle,
                checkpoint_manifest=checkpoint_manifest,
                live_exact_recovery=live_recovery,
                candidate_release_id=candidate_release.release_id,
                expected_spec=acceptance_spec,
                probes=probes,
                release_store=store,
                report_root=root / "candidate-acceptance",
                project_root=repository_root(),
                live_serving_exports=serving_exports,
            )
            ledger.complete("X")
            training_peak = _enforce_gpu_peak(self.run_config)

            candidate_probe_sha = _joint_safety_production_probe_sha256(
                live_acceptance.report
            )
            assets = M0ActivationAssets(
                parent_release=parent_release,
                candidate_release=candidate_release,
                serving_exports=serving_exports,
                parent_joint_version=self.source.active_joint_version,
                candidate_joint_version=live_recovery.candidate_joint_version,
                parent_policy_dcp=str(policy_checkpoints["parent_path"]),
                candidate_policy_dcp=str(policy_checkpoints["candidate_path"]),
                policy_parent_manifest_sha256=str(
                    policy_checkpoints["parent_manifest"]["manifest_sha256"]
                ),
                policy_candidate_manifest_sha256=str(
                    policy_checkpoints["candidate_manifest"]["manifest_sha256"]
                ),
                parent_harness_rollout_checkpoint=self.harness_behavior_checkpoint,
                parent_harness_checkpoint_sha256=harness_parent_sha,
                candidate_harness_checkpoint=str(harness_receipt["checkpoint_path"]),
                candidate_harness_checkpoint_sha256=str(
                    harness_receipt["checkpoint_sha256"]
                ),
                candidate_harness_parameter_sha256=str(
                    harness_receipt["parameter_digest_after"]
                ),
                joint_safety_fixture=joint_gate.fixture,
                candidate_joint_safety_output_sha256=str(candidate_probe_sha),
                recorded_rollout_sglang_mem_fraction_static=(
                    self.source.rollout_sglang_mem_fraction_static
                ),
                max_new_gpu_memory_gib=float(self.run_config.max_new_gpu_memory_gib),
            )
            # The held-out X evaluator has finished and V/W artifacts are sealed.
            # Release the training actor before the factory starts SGLang so the
            # single-GPU M0 path reuses, rather than overlaps, the memory budget.
            actor_to_destroy = actor
            actor = None
            _destroy_actor(actor_to_destroy)
            assert self.production_worker_factory is not None
            try:
                created_workers = _start_production_workers_after_gpu_guard(
                    self.production_worker_factory,
                    assets,
                    gpu_launch_guard=self.gpu_launch_guard,
                )
            except Exception:
                _cleanup_production_workers(
                    (),
                    record_path=cleanup_path,
                    summary_record_sha256=None,
                    production_attestation_sha256=None,
                    factory_completed=False,
                )
                raise
            _require(
                isinstance(created_workers, Sequence)
                and not isinstance(created_workers, (str, bytes, bytearray)),
                "Y worker factory must return an owned worker sequence",
            )
            workers = tuple(created_workers)
            production_workers = workers
            _require(
                bool(workers)
                and all(
                    isinstance(worker, ProductionActivationWorker) for worker in workers
                ),
                "Y worker factory did not return real typed rollout workers",
            )
            parent_outputs: list[bytes] = []
            for worker in workers:
                first = worker.run_probe(joint_gate.fixture)
                second = worker.run_probe(joint_gate.fixture)
                _require(
                    type(first) is bytes and first == second,
                    "parent production probe is not deterministic raw bytes",
                )
                parent_outputs.append(first)
            parent_hashes = {
                hashlib.sha256(value).hexdigest() for value in parent_outputs
            }
            _require(
                len(parent_hashes) == 1, "parent workers disagree on raw probe output"
            )
            parent_probe_sha = parent_hashes.pop()
            production_probes = {
                parent_release.release_id: ProductionProbeSpec(
                    probe_id=f"m0-parent-{joint_gate.suite_id}",
                    fixture=joint_gate.fixture,
                    fixture_sha256=joint_gate.fixture_sha256,
                    expected_output_sha256=parent_probe_sha,
                ),
                candidate_release.release_id: ProductionProbeSpec(
                    probe_id=f"m0-candidate-{joint_gate.suite_id}",
                    fixture=joint_gate.fixture,
                    fixture_sha256=joint_gate.fixture_sha256,
                    expected_output_sha256=str(candidate_probe_sha),
                ),
            }
            authorization = authorize_production_activation(
                store=store,
                live_candidate_acceptance=live_acceptance,
                live_exact_recovery=live_recovery,
                expected_acceptance_spec=acceptance_spec,
                joint_candidate_bundle=bundle,
                checkpoint_manifest=checkpoint_manifest,
                workers=workers,
                probes=production_probes,
            )
            activation = ProductionJointActivationController(
                store=store,
                workers=workers,
                probes=production_probes,
                project_root=repository_root(),
            ).activate(authorization)
            production_attestation_sha256 = activation.attestation_sha256
            expected_active_release_after_cleanup = activation.active_release_id
            ledger.complete("Y")
            _require(ledger.completed == _STAGES, "M0 did not complete T--Y")
            summary: dict[str, object] = {
                "schema_version": M0_RUN_SUMMARY_SCHEMA,
                "source": _source_summary(self.source),
                "stages": {
                    "completed": list(ledger.completed),
                    "policy_candidate_receipt_sha256": policy_receipt["record_sha256"],
                    "harness_candidate_receipt_sha256": harness_receipt[
                        "record_sha256"
                    ],
                    "joint_candidate_bundle_sha256": bundle["record_sha256"],
                    "production_checkpoint_manifest": str(checkpoint_manifest),
                    "live_recovery_sha256": live_recovery.record_sha256,
                    "live_acceptance_sha256": live_acceptance.record_sha256,
                    "parent_serving_export_lineage_sha256": (
                        serving_exports.parent.record_sha256
                    ),
                    "candidate_serving_export_lineage_sha256": (
                        serving_exports.candidate.record_sha256
                    ),
                    "production_attestation_sha256": activation.attestation_sha256,
                },
                "release": {
                    "parent_release_id": parent_release.release_id,
                    "candidate_release_id": candidate_release.release_id,
                    "active_release_id": activation.active_release_id,
                    "candidate_joint_version": asdict(
                        live_recovery.candidate_joint_version
                    ),
                    "candidate_joint_version_id": live_recovery.candidate_joint_version.version_id,
                },
                "resource_gate": {
                    "rollout_sglang_mem_fraction_static": float(
                        self.run_config.rollout_sglang_mem_fraction_static
                    ),
                    "max_new_gpu_memory_gib": float(
                        self.run_config.max_new_gpu_memory_gib
                    ),
                    "measured_training_peak_reserved_gib": training_peak,
                    "actor_destroyed_before_rollout_worker_start": True,
                },
                "provenance": {
                    "project_commit": self.run_config.project_commit,
                    "areal_commit": PINNED_AREAL_COMMIT,
                },
            }
            summary["record_sha256"] = _sha256(summary)
            summary_record_sha256 = str(summary["record_sha256"])
            summary_path = _write_new_json(root / "m0-summary.json", summary)
            return M0JointRunResult(
                artifact_root=root,
                summary_path=summary_path,
                active_release_id=activation.active_release_id,
                candidate_joint_version=live_recovery.candidate_joint_version,
                production_attestation_path=activation.attestation_path,
                production_attestation_sha256=activation.attestation_sha256,
                production_worker_cleanup_path=cleanup_path,
                peak_gpu_memory_gib=training_peak,
            )
        finally:
            try:
                if actor is not None:
                    _destroy_actor(actor)
            finally:
                if production_workers:
                    _cleanup_production_workers(
                        production_workers,
                        record_path=cleanup_path,
                        summary_record_sha256=summary_record_sha256,
                        production_attestation_sha256=(
                            production_attestation_sha256
                        ),
                        release_store=release_store,
                        expected_active_release_id=(
                            expected_active_release_after_cleanup
                        ),
                    )


class AgentServiceM0JointUpdateRunner(_M0JointUpdateRunner):
    """M0 entry that requires a validated Agent Service P/S source."""

    def __init__(
        self,
        *,
        source: AgentServiceM0SourceRecords,
        actor_spec: M0ArealActorSpec,
        run_config: M0JointRunConfig,
        harness_behavior_checkpoint: str | Path,
        acceptance_gates: Sequence[M0AcceptanceGate],
        evaluator: M0CandidateEvaluator,
        production_worker_factory: ProductionWorkerFactory | None,
        gpu_launch_guard: GPULaunchGuard | None = None,
    ) -> None:
        _require(
            type(source) is AgentServiceM0SourceRecords,
            "Agent Service M0 runner requires AgentServiceM0SourceRecords",
        )
        super().__init__(
            source=source,
            actor_spec=actor_spec,
            run_config=run_config,
            harness_behavior_checkpoint=harness_behavior_checkpoint,
            acceptance_gates=acceptance_gates,
            evaluator=evaluator,
            production_worker_factory=production_worker_factory,
            gpu_launch_guard=gpu_launch_guard,
        )


class RLVRM0JointUpdateRunner(_M0JointUpdateRunner):
    """M0 entry for the dedicated RLVR workflow admission envelope."""

    def __init__(
        self,
        *,
        source: RLVRM0SourceRecords,
        actor_spec: M0ArealActorSpec,
        run_config: M0JointRunConfig,
        harness_behavior_checkpoint: str | Path,
        acceptance_gates: Sequence[M0AcceptanceGate],
        evaluator: M0CandidateEvaluator,
        production_worker_factory: ProductionWorkerFactory | None,
        gpu_launch_guard: GPULaunchGuard | None = None,
    ) -> None:
        _require(
            type(source) is RLVRM0SourceRecords,
            "RLVR M0 runner requires the dedicated RLVR admission source",
        )
        super().__init__(
            source=source,
            actor_spec=actor_spec,
            run_config=run_config,
            harness_behavior_checkpoint=harness_behavior_checkpoint,
            acceptance_gates=acceptance_gates,
            evaluator=evaluator,
            production_worker_factory=production_worker_factory,
            gpu_launch_guard=gpu_launch_guard,
        )

    @classmethod
    def from_runner_admission_path(
        cls,
        *,
        runner_admission_path: str | Path,
        active_joint_version: JointVersion,
        actor_spec: M0ArealActorSpec,
        run_config: M0JointRunConfig,
        harness_behavior_checkpoint: str | Path,
        acceptance_gates: Sequence[M0AcceptanceGate],
        evaluator: M0CandidateEvaluator,
        production_worker_factory: ProductionWorkerFactory | None,
        gpu_launch_guard: GPULaunchGuard | None = None,
    ) -> RLVRM0JointUpdateRunner:
        return cls(
            source=load_m0_rlvr_source_records(
                runner_admission_path=runner_admission_path,
                active_joint_version=active_joint_version,
            ),
            actor_spec=actor_spec,
            run_config=run_config,
            harness_behavior_checkpoint=harness_behavior_checkpoint,
            acceptance_gates=acceptance_gates,
            evaluator=evaluator,
            production_worker_factory=production_worker_factory,
            gpu_launch_guard=gpu_launch_guard,
        )

    @classmethod
    def from_rlvr_bridge(
        cls, *_args: object, **_kwargs: object
    ) -> RLVRM0JointUpdateRunner:
        raise M0JointRunnerError(
            "RLVR M0 must load the dedicated pre-batch runner admission; a raw "
            "bridge cannot mint an Agent Service P/session record"
        )


__all__ = [
    "PINNED_AREAL_COMMIT",
    "AgentServiceM0JointUpdateRunner",
    "AgentServiceM0SourceRecords",
    "GPULaunchGuard",
    "M0AcceptanceGate",
    "M0ActivationAssets",
    "M0ArealActorSpec",
    "M0CandidateEvaluator",
    "M0JointRunConfig",
    "M0JointRunResult",
    "M0JointRunnerError",
    "ProductionWorkerFactory",
    "RLVRM0JointUpdateRunner",
    "RLVRM0SourceRecords",
    "build_pinned_areal_actor_config",
    "initialize_pinned_areal_actor",
    "load_m0_agent_service_source_records",
    "load_m0_rlvr_source_records",
]
