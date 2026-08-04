from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from jphrl.paths import require_outside_repository, require_within_configured_root
from jphrl.trajectory.schema import JointVersion

from .areal_policy_optimizer import (
    materialize_areal_ppo_update_tensors,
    validate_areal_policy_optimizer_source,
    validate_m0_areal_actor_config,
)

AREAL_POLICY_CANDIDATE_SCHEMA = "jph.areal-policy-candidate.v2"
PINNED_AREAL_COMMIT = "fee938eada49208a5aabdbc1095730a13076a349"
_EVIDENCE_SCOPE = {
    "policy_optimizer_update": True,
    "policy_candidate_created": True,
    "policy_weights_published": False,
    "rollout_weights_synchronized": False,
    "active_joint_version_changed": False,
    "harness_optimizer_update": False,
    "joint_publish": False,
}
_STAT_KEYS = {
    "grad_norm": "ppo_actor/update/grad_norm",
    "learning_rate": "ppo_actor/update/lr",
    "update_successful": "ppo_actor/update/update_successful",
}


class ArealPolicyCandidateError(RuntimeError):
    """Raised when a real AReaL Policy candidate cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealPolicyCandidateError(message)


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
        raise ArealPolicyCandidateError(
            "Policy candidate evidence is not finite canonical JSON"
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


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
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
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == fields, f"{label} field set differs from schema")
    return value


def _prepare_run_root(candidate_root: str | Path, project_root: str | Path) -> Path:
    del project_root  # the actual checkout boundary is not caller-controlled
    root = require_outside_repository(candidate_root)
    root.mkdir(parents=True, exist_ok=True)
    _require(root.is_dir() and not root.is_symlink(), "candidate root is invalid")
    return root


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(root: str | Path) -> dict[str, object]:
    """Hash every regular checkpoint file without trusting DCP filenames."""

    source = require_within_configured_root(root)
    _require(source.is_dir() and not source.is_symlink(), "checkpoint is missing")
    files: list[dict[str, object]] = []
    for path in sorted(source.rglob("*")):
        _require(not path.is_symlink(), "checkpoint contains a symbolic link")
        if path.is_dir():
            continue
        _require(path.is_file(), "checkpoint contains a non-regular entry")
        files.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _require(bool(files), "checkpoint has no files")
    manifest: dict[str, object] = {"files": files}
    manifest["manifest_sha256"] = _sha256(manifest)
    return manifest


def _atomic_write_json(path: Path, record: Mapping[str, object]) -> None:
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_actor_config(actor: object) -> dict[str, object]:
    validate_m0_areal_actor_config(actor)
    config = actor.config
    optimizer = config.optimizer
    backend = str(config.backend)
    _require(backend.startswith("fsdp:"), "M0 requires the audited FSDP backend")
    return {
        "adv_norm": config.adv_norm,
        "backend": backend,
        "c_clip": config.c_clip,
        "disable_dropout": config.disable_dropout,
        "eps_clip": float(config.eps_clip),
        "eps_clip_higher": config.eps_clip_higher,
        "importance_sampling_level": config.importance_sampling_level,
        "kl_ctl": float(config.kl_ctl),
        "lr": float(optimizer.lr),
        "lr_scheduler_type": optimizer.lr_scheduler_type,
        "ppo_n_minibatches": config.ppo_n_minibatches,
        "recompute_logprob": config.recompute_logprob,
        "reward_norm": config.reward_norm,
        "temperature": float(config.temperature),
        "use_decoupled_loss": config.use_decoupled_loss,
    }


def _validate_safe_config(value: object) -> Mapping[str, object]:
    config = _exact_mapping(
        value,
        {
            "adv_norm",
            "backend",
            "c_clip",
            "disable_dropout",
            "eps_clip",
            "eps_clip_higher",
            "importance_sampling_level",
            "kl_ctl",
            "lr",
            "lr_scheduler_type",
            "ppo_n_minibatches",
            "recompute_logprob",
            "reward_norm",
            "temperature",
            "use_decoupled_loss",
        },
        "Policy candidate safe actor config",
    )
    _require(
        config.get("adv_norm") is None
        and isinstance(config.get("backend"), str)
        and config["backend"].startswith("fsdp:")
        and config.get("c_clip") is None
        and config.get("disable_dropout") is True
        and _is_finite_number(config.get("eps_clip"))
        and float(config["eps_clip"]) > 0.0
        and config.get("eps_clip_higher") is None
        and config.get("importance_sampling_level") == "token"
        and _is_finite_number(config.get("kl_ctl"))
        and float(config["kl_ctl"]) == 0.0
        and _is_finite_number(config.get("lr"))
        and float(config["lr"]) > 0.0
        and config.get("lr_scheduler_type") == "constant"
        and type(config.get("ppo_n_minibatches")) is int
        and config["ppo_n_minibatches"] == 1
        and config.get("recompute_logprob") is False
        and config.get("reward_norm") is None
        and _is_finite_number(config.get("temperature"))
        and float(config["temperature"]) > 0.0
        and config.get("use_decoupled_loss") is False,
        "Policy candidate safe actor config differs from M0",
    )
    return config


def _real_areal_actor(actor: object) -> tuple[str, str]:
    try:
        from areal.engine.fsdp_engine import FSDPPPOActor
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise ArealPolicyCandidateError(
            "pinned AReaL FSDPPPOActor is unavailable"
        ) from exc
    _require(
        type(actor) is FSDPPPOActor,
        "optimizer evidence requires the exact pinned AReaL FSDPPPOActor",
    )
    optimizer = getattr(actor, "optimizer", None)
    _require(optimizer is not None, "real AReaL actor optimizer is unavailable")
    optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    _require(
        optimizer_type == "torch.optim.adamw.AdamW",
        "M0 Policy optimizer must be the pinned torch AdamW implementation",
    )
    return "areal.engine.fsdp_engine.FSDPPPOActor", optimizer_type


def _lr_scheduler_state(actor: object) -> dict[str, object]:
    scheduler = getattr(actor, "lr_scheduler", None)
    state_dict = getattr(scheduler, "state_dict", None)
    _require(callable(state_dict), "AReaL actor lr scheduler state is unavailable")
    raw_state = state_dict()
    _require(isinstance(raw_state, Mapping), "AReaL lr scheduler state is invalid")
    state = deepcopy(dict(raw_state))
    _canonical_json(state)
    return state


def _restore_lr_scheduler_state(
    actor: object,
    state: Mapping[str, object],
) -> None:
    scheduler = getattr(actor, "lr_scheduler", None)
    load_state_dict = getattr(scheduler, "load_state_dict", None)
    _require(
        callable(load_state_dict),
        "AReaL actor lr scheduler rollback is unavailable",
    )
    load_state_dict(deepcopy(dict(state)))
    _require(
        _lr_scheduler_state(actor) == state,
        "AReaL actor lr scheduler rollback is incomplete",
    )


def _require_one_lr_scheduler_step(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> None:
    _require(
        set(before) == set(after)
        and type(before.get("last_epoch")) is int
        and after.get("last_epoch") == before["last_epoch"] + 1
        and type(before.get("_step_count")) is int
        and after.get("_step_count") == before["_step_count"] + 1,
        "AReaL lr scheduler did not advance exactly once",
    )
    before_rest = dict(before)
    after_rest = dict(after)
    before_rest.pop("last_epoch")
    before_rest.pop("_step_count")
    after_rest.pop("last_epoch")
    after_rest.pop("_step_count")
    _require(
        before_rest == after_rest,
        "AReaL constant lr scheduler changed outside its step counters",
    )


def _actor_version(actor: object) -> int:
    method = getattr(actor, "get_version", None)
    _require(callable(method), "AReaL actor get_version is unavailable")
    version = method()
    _require(
        type(version) is int and version >= 0,
        "AReaL actor version must be a non-negative integer",
    )
    return version


def _probe(
    actor: object,
    admission_record: Mapping[str, object],
    *,
    active_joint_version: JointVersion,
    device: str | Any,
) -> tuple[list[list[float]], str]:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - remote runtime gate
        raise ArealPolicyCandidateError("torch is required for Policy probes") from exc
    prepared = materialize_areal_ppo_update_tensors(
        admission_record,
        actor=actor,
        active_joint_version=active_joint_version,
        device=device,
    )
    compute_logp = getattr(actor, "compute_logp", None)
    _require(callable(compute_logp), "AReaL actor compute_logp is unavailable")
    outputs = compute_logp(prepared)
    _require(isinstance(outputs, list) and bool(outputs), "Policy probe is missing")
    values: list[list[float]] = []
    for output in outputs:
        _require(isinstance(output, torch.Tensor), "Policy probe is not a tensor")
        row = [float(value) for value in output.detach().float().cpu().reshape(-1)]
        _require(
            bool(row) and all(math.isfinite(value) for value in row),
            "Policy probe contains no finite log-probabilities",
        )
        values.append(row)
    return values, _sha256(values)


def _probe_delta(left: list[list[float]], right: list[list[float]]) -> float:
    _require(
        len(left) == len(right)
        and all(len(a) == len(b) for a, b in zip(left, right, strict=True)),
        "Policy probe shapes differ",
    )
    delta = max(
        abs(a - b)
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )
    _require(math.isfinite(delta), "Policy probe delta is not finite")
    return delta


def _optimizer_stats(stats: object) -> dict[str, object]:
    _require(isinstance(stats, Mapping), "AReaL optimizer stats are missing")
    values: dict[str, object] = {}
    for output_name, key in _STAT_KEYS.items():
        value = stats.get(key)
        count = stats.get(f"{key}__count")
        _require(
            _is_finite_number(value) and count == 1,
            f"AReaL optimizer stat {key} must have exactly one finite update",
        )
        values[output_name] = float(value)
        values[f"{output_name}_count"] = int(count)
    _require(
        values["update_successful"] == 1.0,
        "AReaL optimizer reported an unsuccessful update",
    )
    _require(values["grad_norm"] > 0.0, "AReaL optimizer gradient norm is zero")
    _require(values["learning_rate"] > 0.0, "AReaL optimizer learning rate is zero")
    return values


def _optimizer_step(actor: object) -> int:
    optimizer = getattr(actor, "optimizer", None)
    state = getattr(optimizer, "state", None)
    _require(isinstance(state, Mapping), "AReaL optimizer state is unavailable")
    steps: list[int] = []
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping) or "step" not in parameter_state:
            continue
        raw_step = parameter_state["step"]
        if hasattr(raw_step, "item"):
            raw_step = raw_step.item()
        _require(
            isinstance(raw_step, (int, float))
            and not isinstance(raw_step, bool)
            and math.isfinite(float(raw_step))
            and float(raw_step).is_integer()
            and int(raw_step) >= 0,
            "AReaL optimizer contains an invalid step counter",
        )
        steps.append(int(raw_step))
    if not steps:
        return 0
    _require(
        len(set(steps)) == 1,
        "AReaL optimizer parameter step counters differ",
    )
    return steps[0]


def _validate_normalized_optimizer_stats(stats: object) -> None:
    values = _exact_mapping(
        stats,
        {
            "grad_norm",
            "grad_norm_count",
            "learning_rate",
            "learning_rate_count",
            "optimizer_step_after",
            "optimizer_step_before",
            "update_successful",
            "update_successful_count",
        },
        "Policy candidate optimizer stats",
    )
    _require(
        all(
            _is_finite_number(values.get(field))
            for field in ("grad_norm", "learning_rate", "update_successful")
        )
        and values.get("grad_norm_count") == 1
        and values.get("learning_rate_count") == 1
        and values.get("update_successful_count") == 1
        and type(values.get("optimizer_step_before")) is int
        and values.get("optimizer_step_before") >= 0
        and values.get("optimizer_step_after")
        == values.get("optimizer_step_before") + 1
        and float(values["grad_norm"]) > 0.0
        and float(values["learning_rate"]) > 0.0
        and float(values["update_successful"]) == 1.0,
        "Policy candidate optimizer stats do not prove one successful update",
    )


@dataclass(frozen=True)
class ValidatedArealPolicyCandidate:
    transaction_id: str
    episode_id: str
    parent_joint_version: JointVersion
    source_admission_sha256: str
    source_joint_credit_sha256: str
    trainable_token_count: int
    parent_engine_version: int
    reserved_candidate_engine_version: int
    candidate_policy_version: str
    record_sha256: str

    @property
    def digest(self) -> str:
        return self.record_sha256


def _run_areal_policy_candidate_update_unprotected(
    admission_record: Mapping[str, object],
    *,
    source_joint_credit_record: Mapping[str, object],
    actor: object,
    active_joint_version: JointVersion,
    candidate_root: str | Path,
    project_root: str | Path,
    transaction_id: str,
    project_commit: str,
    areal_commit: str,
    device: str | Any,
) -> dict[str, object]:
    """Perform one real AReaL PPO step but leave rollout publication untouched."""

    _require(
        areal_commit == PINNED_AREAL_COMMIT,
        "AReaL commit differs from the pinned M0 implementation",
    )
    _require(
        isinstance(transaction_id, str) and transaction_id,
        "candidate transaction ID is missing",
    )
    _require(_is_git_commit(project_commit), "project commit must be a full Git SHA-1")
    admission = validate_areal_policy_optimizer_source(
        admission_record,
        source_joint_credit_record=source_joint_credit_record,
        active_joint_version=active_joint_version,
    )
    root = _prepare_run_root(candidate_root, project_root)
    actor_type, optimizer_type = _real_areal_actor(actor)
    safe_config = _safe_actor_config(actor)
    parent_version = _actor_version(actor)
    _require(
        parent_version == admission.inference_engine_version,
        "AReaL actor version differs from admitted S rollout",
    )

    parent_path = root / "policy-parent.dcp"
    candidate_path = root / "policy-candidate.dcp"
    evidence_path = root / "policy-candidate-evidence.json"
    _require(
        not parent_path.exists()
        and not candidate_path.exists()
        and not evidence_path.exists(),
        "Policy candidate transaction paths must be new",
    )
    try:
        from areal.api import SaveLoadMeta
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise ArealPolicyCandidateError(
            "pinned AReaL SaveLoadMeta is unavailable"
        ) from exc

    save = getattr(actor, "save", None)
    load = getattr(actor, "load", None)
    export_stats = getattr(actor, "export_stats", None)
    ppo_update = getattr(actor, "ppo_update", None)
    step_scheduler = getattr(actor, "step_lr_scheduler", None)
    _require(
        all(
            callable(method)
            for method in (save, load, export_stats, ppo_update, step_scheduler)
        ),
        "AReaL actor candidate API is incomplete",
    )

    parent_meta = SaveLoadMeta(
        path=str(parent_path),
        weight_format="dcp",
        with_optim=True,
    )
    candidate_meta = SaveLoadMeta(
        path=str(candidate_path),
        weight_format="dcp",
        with_optim=True,
    )
    save(meta=parent_meta)
    parent_manifest = checkpoint_manifest(parent_path)
    pre_probe, pre_probe_sha256 = _probe(
        actor,
        admission_record,
        active_joint_version=active_joint_version,
        device=device,
    )
    optimizer_step_before = _optimizer_step(actor)
    lr_scheduler_state_before = _lr_scheduler_state(actor)
    export_stats()
    update_batch = materialize_areal_ppo_update_tensors(
        admission_record,
        actor=actor,
        active_joint_version=active_joint_version,
        device=device,
    )
    ppo_update(update_batch)
    step_scheduler()
    lr_scheduler_state_after = _lr_scheduler_state(actor)
    _require_one_lr_scheduler_step(
        lr_scheduler_state_before,
        lr_scheduler_state_after,
    )
    optimizer_stats = _optimizer_stats(export_stats())
    optimizer_step_after = _optimizer_step(actor)
    _require(
        optimizer_step_after == optimizer_step_before + 1,
        "AReaL optimizer step did not advance exactly once",
    )
    optimizer_stats.update(
        {
            "optimizer_step_before": optimizer_step_before,
            "optimizer_step_after": optimizer_step_after,
        }
    )
    _require(
        _actor_version(actor) == parent_version,
        "AReaL actor version changed before joint publication",
    )
    post_probe, post_probe_sha256 = _probe(
        actor,
        admission_record,
        active_joint_version=active_joint_version,
        device=device,
    )
    probe_delta = _probe_delta(pre_probe, post_probe)
    _require(probe_delta > 0.0, "Policy probe did not change after optimizer update")
    save(meta=candidate_meta)
    candidate_manifest = checkpoint_manifest(candidate_path)
    _require(
        parent_manifest["manifest_sha256"] != candidate_manifest["manifest_sha256"],
        "parent and candidate Policy checkpoints are identical",
    )

    load(meta=parent_meta)
    parent_reload, parent_reload_sha256 = _probe(
        actor,
        admission_record,
        active_joint_version=active_joint_version,
        device=device,
    )
    _require(
        _probe_delta(pre_probe, parent_reload) <= 1e-7,
        "parent Policy checkpoint does not reproduce its probe",
    )
    load(meta=candidate_meta)
    candidate_reload, candidate_reload_sha256 = _probe(
        actor,
        admission_record,
        active_joint_version=active_joint_version,
        device=device,
    )
    _require(
        _probe_delta(post_probe, candidate_reload) <= 1e-7,
        "candidate Policy checkpoint does not reproduce its probe",
    )
    _require(
        _actor_version(actor) == parent_version,
        "AReaL checkpoint roundtrip changed the unpublished actor version",
    )

    candidate_policy_version = (
        "areal-policy-" + str(candidate_manifest["manifest_sha256"])[:20]
    )
    record: dict[str, object] = {
        "schema_version": AREAL_POLICY_CANDIDATE_SCHEMA,
        "transaction": {
            "transaction_id": transaction_id,
            "episode_id": admission.episode_id,
            "source_admission_sha256": admission.record_sha256,
            "source_joint_credit_sha256": admission.source_joint_credit_sha256,
            "trainable_token_count": admission.trainable_token_count,
        },
        "parent": {
            "joint_version": asdict(active_joint_version),
            "joint_version_id": active_joint_version.version_id,
            "policy_engine_version": parent_version,
            "policy_version": active_joint_version.policy,
        },
        "candidate": {
            "policy_version": candidate_policy_version,
            "reserved_policy_engine_version": parent_version + 1,
        },
        "optimizer": {
            "actor_type": actor_type,
            "optimizer_type": optimizer_type,
            "config": safe_config,
            "stats": optimizer_stats,
            "lr_scheduler": {
                "state_before_sha256": _sha256(lr_scheduler_state_before),
                "state_after_sha256": _sha256(lr_scheduler_state_after),
                "step_completed": True,
            },
        },
        "checkpoints": {
            "parent_path": str(parent_path),
            "parent_manifest": parent_manifest,
            "candidate_path": str(candidate_path),
            "candidate_manifest": candidate_manifest,
        },
        "probes": {
            "pre_sha256": pre_probe_sha256,
            "post_sha256": post_probe_sha256,
            "max_abs_logprob_delta": probe_delta,
            "parent_reload_sha256": parent_reload_sha256,
            "candidate_reload_sha256": candidate_reload_sha256,
        },
        "provenance": {
            "areal_commit": areal_commit,
            "project_commit": project_commit,
        },
        "evidence_scope": dict(_EVIDENCE_SCOPE),
    }
    record["record_sha256"] = _record_sha256(record)
    validate_areal_policy_candidate(record, active_joint_version=active_joint_version)
    _atomic_write_json(evidence_path, record)
    return record


def run_areal_policy_candidate_update(
    admission_record: Mapping[str, object],
    *,
    source_joint_credit_record: Mapping[str, object],
    actor: object,
    active_joint_version: JointVersion,
    candidate_root: str | Path,
    project_root: str | Path,
    transaction_id: str,
    project_commit: str,
    areal_commit: str,
    device: str | Any,
) -> dict[str, object]:
    """Create an unpublished candidate, restoring the parent on any failure."""

    scheduler_state = _lr_scheduler_state(actor)
    try:
        return _run_areal_policy_candidate_update_unprotected(
            admission_record,
            source_joint_credit_record=source_joint_credit_record,
            actor=actor,
            active_joint_version=active_joint_version,
            candidate_root=candidate_root,
            project_root=project_root,
            transaction_id=transaction_id,
            project_commit=project_commit,
            areal_commit=areal_commit,
            device=device,
        )
    except BaseException:
        parent_path = (
            require_within_configured_root(candidate_root) / "policy-parent.dcp"
        )
        try:
            if parent_path.is_dir():
                from areal.api import SaveLoadMeta

                load = getattr(actor, "load", None)
                _require(callable(load), "AReaL actor load is unavailable for rollback")
                load(
                    meta=SaveLoadMeta(
                        path=str(parent_path),
                        weight_format="dcp",
                        with_optim=True,
                    )
                )
            _restore_lr_scheduler_state(actor, scheduler_state)
        except BaseException as rollback_error:
            raise ArealPolicyCandidateError(
                "Policy candidate failed and parent rollback also failed"
            ) from rollback_error
        raise


def validate_areal_policy_candidate(
    record: Mapping[str, object],
    *,
    active_joint_version: JointVersion | None = None,
    require_checkpoints: bool = False,
) -> ValidatedArealPolicyCandidate:
    _require(
        set(record)
        == {
            "schema_version",
            "transaction",
            "parent",
            "candidate",
            "optimizer",
            "checkpoints",
            "probes",
            "provenance",
            "evidence_scope",
            "record_sha256",
        },
        "Policy candidate field set differs from schema",
    )
    _require(
        record.get("schema_version") == AREAL_POLICY_CANDIDATE_SCHEMA,
        "Policy candidate schema differs",
    )
    _require(
        record.get("record_sha256") == _record_sha256(record),
        "Policy candidate record hash mismatch",
    )
    transaction = _exact_mapping(
        record.get("transaction"),
        {
            "transaction_id",
            "episode_id",
            "source_admission_sha256",
            "source_joint_credit_sha256",
            "trainable_token_count",
        },
        "Policy candidate transaction",
    )
    _require(
        all(
            isinstance(transaction.get(field), str) and transaction.get(field)
            for field in ("transaction_id", "episode_id")
        ),
        "Policy candidate transaction identity is missing",
    )
    source_sha256 = transaction.get("source_admission_sha256")
    _require(_is_sha256(source_sha256), "Policy candidate source hash is invalid")
    source_joint_credit_sha256 = transaction.get("source_joint_credit_sha256")
    _require(
        _is_sha256(source_joint_credit_sha256),
        "Policy candidate source S hash is invalid",
    )
    trainable_token_count = transaction.get("trainable_token_count")
    _require(
        type(trainable_token_count) is int and trainable_token_count > 0,
        "Policy candidate has no trainable tokens",
    )
    parent = _exact_mapping(
        record.get("parent"),
        {
            "joint_version",
            "joint_version_id",
            "policy_engine_version",
            "policy_version",
        },
        "Policy candidate parent",
    )
    raw_version = parent.get("joint_version")
    _require(isinstance(raw_version, Mapping), "parent JointVersion is missing")
    try:
        joint_version = JointVersion(**dict(raw_version))
    except TypeError as exc:
        raise ArealPolicyCandidateError("parent JointVersion is invalid") from exc
    _require(
        parent.get("joint_version_id") == joint_version.version_id
        and parent.get("policy_version") == joint_version.policy,
        "Policy candidate parent differs from JointVersion",
    )
    if active_joint_version is not None:
        _require(
            joint_version == active_joint_version,
            "Policy candidate differs from lag-zero active JointVersion",
        )
    parent_engine_version = parent.get("policy_engine_version")
    _require(
        type(parent_engine_version) is int and parent_engine_version >= 0,
        "parent Policy engine version is invalid",
    )
    candidate = _exact_mapping(
        record.get("candidate"),
        {"policy_version", "reserved_policy_engine_version"},
        "Policy candidate identity",
    )
    candidate_policy_version = candidate.get("policy_version")
    reserved_version = candidate.get("reserved_policy_engine_version")
    _require(
        isinstance(candidate_policy_version, str)
        and candidate_policy_version.startswith("areal-policy-")
        and reserved_version == parent_engine_version + 1,
        "Policy candidate version does not advance its parent",
    )
    optimizer = _exact_mapping(
        record.get("optimizer"),
        {
            "actor_type",
            "optimizer_type",
            "config",
            "stats",
            "lr_scheduler",
        },
        "Policy candidate optimizer",
    )
    _require(
        optimizer.get("actor_type") == "areal.engine.fsdp_engine.FSDPPPOActor"
        and optimizer.get("optimizer_type") == "torch.optim.adamw.AdamW",
        "Policy candidate optimizer identity is invalid",
    )
    _validate_safe_config(optimizer.get("config"))
    _validate_normalized_optimizer_stats(optimizer.get("stats"))
    lr_scheduler = _exact_mapping(
        optimizer.get("lr_scheduler"),
        {"state_before_sha256", "state_after_sha256", "step_completed"},
        "Policy candidate lr scheduler",
    )
    _require(
        _is_sha256(lr_scheduler.get("state_before_sha256"))
        and _is_sha256(lr_scheduler.get("state_after_sha256"))
        and lr_scheduler["state_before_sha256"] != lr_scheduler["state_after_sha256"]
        and lr_scheduler.get("step_completed") is True,
        "Policy candidate lr scheduler evidence is invalid",
    )
    checkpoints = _exact_mapping(
        record.get("checkpoints"),
        {
            "parent_path",
            "parent_manifest",
            "candidate_path",
            "candidate_manifest",
        },
        "Policy candidate checkpoints",
    )
    for kind in ("parent", "candidate"):
        path = checkpoints.get(f"{kind}_path")
        manifest = checkpoints.get(f"{kind}_manifest")
        _require(isinstance(path, str) and path, f"{kind} checkpoint path is invalid")
        manifest = _exact_mapping(
            manifest,
            {"files", "manifest_sha256"},
            f"{kind} checkpoint manifest",
        )
        _require(
            isinstance(manifest.get("files"), list)
            and bool(manifest["files"])
            and manifest.get("manifest_sha256")
            == _sha256({"files": manifest["files"]}),
            f"{kind} checkpoint manifest hash is invalid",
        )
        seen_paths: set[str] = set()
        for item in manifest["files"]:
            item = _exact_mapping(
                item,
                {"path", "size_bytes", "sha256"},
                f"{kind} checkpoint file",
            )
            relative = item.get("path")
            _require(
                isinstance(relative, str)
                and relative
                and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts
                and relative not in seen_paths,
                f"{kind} checkpoint file path is invalid",
            )
            seen_paths.add(relative)
            _require(
                type(item.get("size_bytes")) is int
                and item["size_bytes"] >= 0
                and _is_sha256(item.get("sha256")),
                f"{kind} checkpoint file digest is invalid",
            )
        if require_checkpoints:
            _require(
                checkpoint_manifest(str(path)) == manifest,
                f"{kind} checkpoint differs from its persisted manifest",
            )
    parent_manifest = checkpoints["parent_manifest"]
    candidate_manifest = checkpoints["candidate_manifest"]
    _require(
        parent_manifest["manifest_sha256"] != candidate_manifest["manifest_sha256"]
        and candidate_policy_version
        == "areal-policy-" + candidate_manifest["manifest_sha256"][:20],
        "Policy candidate identity differs from checkpoint contents",
    )
    probes = _exact_mapping(
        record.get("probes"),
        {
            "pre_sha256",
            "post_sha256",
            "max_abs_logprob_delta",
            "parent_reload_sha256",
            "candidate_reload_sha256",
        },
        "Policy candidate probes",
    )
    _require(
        all(
            _is_sha256(probes.get(field))
            for field in (
                "pre_sha256",
                "post_sha256",
                "parent_reload_sha256",
                "candidate_reload_sha256",
            )
        )
        and probes["pre_sha256"] == probes["parent_reload_sha256"]
        and probes["post_sha256"] == probes["candidate_reload_sha256"]
        and probes["pre_sha256"] != probes["post_sha256"]
        and _is_finite_number(probes.get("max_abs_logprob_delta"))
        and float(probes["max_abs_logprob_delta"]) > 0.0,
        "Policy candidate probe evidence is invalid",
    )
    provenance = _exact_mapping(
        record.get("provenance"),
        {"areal_commit", "project_commit"},
        "Policy candidate provenance",
    )
    _require(
        provenance.get("areal_commit") == PINNED_AREAL_COMMIT
        and _is_git_commit(provenance.get("project_commit")),
        "Policy candidate provenance is invalid",
    )
    _require(
        record.get("evidence_scope") == _EVIDENCE_SCOPE,
        "Policy candidate evidence scope differs from contract",
    )
    return ValidatedArealPolicyCandidate(
        transaction_id=str(transaction["transaction_id"]),
        episode_id=str(transaction["episode_id"]),
        parent_joint_version=joint_version,
        source_admission_sha256=str(source_sha256),
        source_joint_credit_sha256=str(source_joint_credit_sha256),
        trainable_token_count=trainable_token_count,
        parent_engine_version=parent_engine_version,
        reserved_candidate_engine_version=reserved_version,
        candidate_policy_version=str(candidate_policy_version),
        record_sha256=str(record["record_sha256"]),
    )
