#!/usr/bin/env python3

from __future__ import annotations

"""Deep, persisted-evidence audit for a completed live M0 T -> Y run."""

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from jphrl.experiments.m0_joint_runner import (
    M0_RUN_SUMMARY_SCHEMA,
    M0_WORKER_CLEANUP_SCHEMA,
    PINNED_AREAL_COMMIT,
)
from jphrl.experiments.m0_live_evaluator import (
    build_m0_heldout_acceptance_gates,
)
from jphrl.experiments.m0_live_joint import (
    M0_GPU_LAUNCH_AUDIT_SCHEMA,
    M0_LIVE_SELECTION_SCHEMA,
)
from jphrl.joint_release import JointReleaseStore
from jphrl.paths import require_outside_repository
from jphrl.training.areal_policy_candidate import validate_areal_policy_candidate
from jphrl.training.candidate_acceptance import (
    CandidateAcceptanceSpec,
    CandidateAcceptanceSuite,
    validate_candidate_acceptance_report,
)
from jphrl.training.joint_activation import PRODUCTION_ATTESTATION_SCHEMA
from jphrl.training.joint_step import validate_joint_candidate_bundle
from jphrl.training.production_checkpoint import (
    validate_exact_joint_recovery_evidence,
    validate_production_joint_checkpoint,
)
from jphrl.trajectory.schema import JointVersion

M0_LIVE_VERIFICATION_SCHEMA = "jph.m0-live-joint-verification.v1"
GPU_MEMORY_AUDIT_SCHEMA = "jph.gpu-memory-observation.v2"

_SECRET_FIELDS = {
    "access_token",
    "admin_api_key",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "github_token",
    "password",
    "secret",
    "session_api_key",
    "token",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _record_sha256(record: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical_json(
            {key: value for key, value in record.items() if key != "record_sha256"}
        )
    ).hexdigest()


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
            if normalized in _SECRET_FIELDS or normalized.endswith(
                ("_api_key", "_credential", "_password", "_secret", "_token")
            ):
                raise ValueError(f"credential field entered M0 evidence: {path}.{key}")
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _load_record(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe")
    if path.stat().st_mode & 0o077:
        raise ValueError(f"{label} is not private")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    _assert_no_secrets(value, label)
    if value.get("record_sha256") != _record_sha256(value):
        raise ValueError(f"{label} record SHA-256 differs")
    return value


def _single_record(root: Path, pattern: str, *, label: str) -> tuple[Path, dict[str, object]]:
    paths = tuple(sorted(root.glob(pattern)))
    if len(paths) != 1:
        raise ValueError(f"{label} must contain exactly one record")
    return paths[0], _load_record(paths[0], label=label)


def _joint_version(raw: object) -> JointVersion:
    if not isinstance(raw, Mapping) or set(raw) != set(JointVersion.__dataclass_fields__):
        raise ValueError("verification JointVersion field set differs")
    try:
        version = JointVersion(**dict(raw))
    except TypeError as exc:
        raise ValueError("verification JointVersion is invalid") from exc
    if any(not isinstance(value, str) or not value for value in raw.values()):
        raise ValueError("verification JointVersion field is empty")
    return version


def _write_verification(path: Path, record: Mapping[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def verify_m0_live_joint(
    *,
    run_root: str | Path,
    expected_project_commit: str,
    expected_areal_commit: str = PINNED_AREAL_COMMIT,
) -> dict[str, object]:
    if not _is_git_sha(expected_project_commit):
        raise ValueError("expected project commit is invalid")
    if expected_areal_commit != PINNED_AREAL_COMMIT:
        raise ValueError("expected AReaL commit differs from the M0 pin")
    root = require_outside_repository(run_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("M0 live run root is missing or unsafe")
    artifact_root = root / "joint-update"

    selection = _load_record(root / "selection" / "selection.json", label="M0 selection")
    summary = _load_record(artifact_root / "m0-summary.json", label="M0 summary")
    cleanup = _load_record(
        artifact_root / "production-worker-cleanup.json",
        label="M0 production cleanup",
    )
    memory = _load_record(root / "gpu-memory-audit.json", label="GPU memory audit")
    training_guard = _load_record(
        artifact_root / "gpu-launch-audits" / "training-actor.json",
        label="training GPU launch audit",
    )
    production_guard = _load_record(
        artifact_root / "gpu-launch-audits" / "production-sglang.json",
        label="production GPU launch audit",
    )
    _, attestation = _single_record(
        artifact_root / "release-store" / "activation-attestations",
        "activation-*.json",
        label="production activation attestation",
    )
    _, acceptance = _single_record(
        artifact_root / "candidate-acceptance",
        "acceptance-*.json",
        label="candidate acceptance",
    )

    if selection.get("schema_version") != M0_LIVE_SELECTION_SCHEMA:
        raise ValueError("M0 selection schema differs")
    active_joint_version = _joint_version(selection.get("joint_version"))
    if selection.get("joint_version_id") != active_joint_version.version_id:
        raise ValueError("M0 selection JointVersion digest differs")
    training = selection.get("training")
    holdouts = selection.get("holdouts")
    if not isinstance(training, Mapping) or not isinstance(holdouts, list):
        raise TypeError("M0 selection split is missing")
    holdout_digests = tuple(
        item.get("runner_admission_sha256")
        for item in holdouts
        if isinstance(item, Mapping)
    )
    training_digest = training.get("runner_admission_sha256")
    if (
        len(holdout_digests) != 3
        or not isinstance(training_digest, str)
        or len({training_digest, *holdout_digests}) != 4
    ):
        raise ValueError("M0 training/held-out selection is not disjoint 1:3")

    if summary.get("schema_version") != M0_RUN_SUMMARY_SCHEMA:
        raise ValueError("M0 summary schema differs")
    source = summary.get("source")
    stages = summary.get("stages")
    release = summary.get("release")
    resource = summary.get("resource_gate")
    provenance = summary.get("provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (source, stages, release, resource, provenance)
    ):
        raise ValueError("M0 summary sections are missing")
    if (
        source.get("route_kind") != "rlvr-workflow"
        or source.get("runner_admission_sha256") != training_digest
        or stages.get("completed") != ["T", "U", "V", "W", "X", "Y"]
        or provenance.get("project_commit") != expected_project_commit
        or provenance.get("areal_commit") != expected_areal_commit
        or resource.get("rollout_sglang_mem_fraction_static") != 0.29
        or resource.get("gpu_memory_limit_enforced") is not False
        or resource.get("max_new_gpu_memory_gib") is not None
        or resource.get("actor_destroyed_before_rollout_worker_start") is not True
    ):
        raise ValueError("M0 summary lineage, stages, or resource gate differs")

    policy_receipt = _load_record(
        artifact_root / "policy-candidate" / "policy-candidate-evidence.json",
        label="Policy optimizer receipt",
    )
    policy_audit = validate_areal_policy_candidate(
        policy_receipt,
        active_joint_version=active_joint_version,
        require_checkpoints=True,
    )
    harness_receipt = _load_record(
        artifact_root / "harness-candidate" / "harness-candidate-evidence.json",
        label="Harness optimizer receipt",
    )
    try:
        from jphrl.harness.torch_learning import (
            validate_torch_harness_update_evidence,
        )
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - remote dep
        raise ValueError("Torch Harness verification is unavailable") from exc
    validate_torch_harness_update_evidence(
        harness_receipt,
        active_joint_version=active_joint_version,
        require_checkpoint=True,
    )
    bundle = _load_record(
        artifact_root / "joint-candidate" / "bundle.json",
        label="joint candidate bundle",
    )
    bundle_audit = validate_joint_candidate_bundle(
        bundle,
        active_joint_version=active_joint_version,
        require_receipt_files=True,
    )
    manifest_path = Path(str(stages["production_checkpoint_manifest"])).resolve()
    if artifact_root.resolve() not in manifest_path.parents:
        raise ValueError("production checkpoint manifest escapes the M0 artifact root")
    checkpoint_audit = validate_production_joint_checkpoint(
        manifest_path,
        require_component_files=True,
    )
    recovery = _load_record(
        artifact_root / "production-checkpoint" / "w-live-recovery.json",
        label="exact recovery evidence",
    )
    recovery_audit = validate_exact_joint_recovery_evidence(
        recovery,
        manifest=manifest_path,
        current_topology=checkpoint_audit.topology,
    )

    gates = build_m0_heldout_acceptance_gates(
        training_runner_admission_sha256=training_digest,
        holdout_runner_admission_sha256s=holdout_digests,
    )
    expected_spec = CandidateAcceptanceSpec(
        tuple(
            CandidateAcceptanceSuite(
                kind=gate.kind,
                suite_id=gate.suite_id,
                fixture_sha256=gate.fixture_sha256,
                metric_name=gate.metric_name,
                minimum_score=gate.minimum_score,
                minimum_sample_count=gate.minimum_sample_count,
            )
            for gate in gates
        )
    )
    acceptance_audit = validate_candidate_acceptance_report(
        acceptance,
        expected_spec=expected_spec,
        release_store=JointReleaseStore(artifact_root / "release-store"),
        joint_candidate_bundle=bundle,
        checkpoint_manifest=manifest_path,
        exact_recovery_evidence=recovery,
        require_component_files=True,
    )

    if (
        stages.get("policy_candidate_receipt_sha256") != policy_audit.record_sha256
        or stages.get("harness_candidate_receipt_sha256")
        != harness_receipt["record_sha256"]
        or stages.get("joint_candidate_bundle_sha256") != bundle_audit.record_sha256
        or stages.get("live_recovery_sha256") != recovery_audit["record_sha256"]
        or stages.get("live_acceptance_sha256") != acceptance_audit.record_sha256
        or release.get("active_release_id") != acceptance_audit.candidate_release_id
    ):
        raise ValueError("M0 T/U/V/W/X digest chain differs from the summary")

    if (
        attestation.get("schema_version") != PRODUCTION_ATTESTATION_SCHEMA
        or stages.get("production_attestation_sha256")
        != attestation.get("record_sha256")
        or cleanup.get("schema_version") != M0_WORKER_CLEANUP_SCHEMA
        or cleanup.get("summary_record_sha256") != summary.get("record_sha256")
        or cleanup.get("production_attestation_sha256")
        != attestation.get("record_sha256")
        or cleanup.get("factory_completed") is not True
        or cleanup.get("all_cleanup_calls_returned") is not True
        or cleanup.get("active_release_unchanged") is not True
        or cleanup.get("active_release_id_after_cleanup")
        != release.get("active_release_id")
    ):
        raise ValueError("production Y attestation or cleanup chain differs")
    workers = cleanup.get("workers")
    if (
        not isinstance(workers, list)
        or not workers
        or any(
            not isinstance(worker, Mapping)
            or worker.get("returned_without_error") is not True
            for worker in workers
        )
    ):
        raise ValueError("production worker cleanup was incomplete")

    for audit, phase in (
        (training_guard, "training-actor"),
        (production_guard, "production-sglang"),
    ):
        if (
            audit.get("schema_version") != M0_GPU_LAUNCH_AUDIT_SCHEMA
            or audit.get("phase") != phase
            or audit.get("passed") is not True
            or audit.get("memory_limit_enforced") is not False
            or audit.get("compute_processes_observation_only") is not True
        ):
            raise ValueError(f"{phase} live GPU launch audit failed")
    if (
        memory.get("schema_version") != GPU_MEMORY_AUDIT_SCHEMA
        or memory.get("run_kind") != "m0-live-joint-v1"
        or memory.get("project_commit") != expected_project_commit
        or memory.get("memory_limit_enforced") is not False
        or memory.get("max_new_memory_mib") is not None
        or memory.get("passed") is not True
    ):
        raise ValueError("M0 GPU memory observation audit failed")

    record: dict[str, object] = {
        "schema_version": M0_LIVE_VERIFICATION_SCHEMA,
        "run_root": str(root),
        "project_commit": expected_project_commit,
        "areal_commit": expected_areal_commit,
        "source": {
            "training_runner_admission_sha256": training_digest,
            "holdout_runner_admission_sha256s": list(holdout_digests),
            "parent_joint_version": asdict(active_joint_version),
            "parent_joint_version_id": active_joint_version.version_id,
        },
        "candidate": {
            "joint_version_id": release.get("candidate_joint_version_id"),
            "active_release_id": release.get("active_release_id"),
        },
        "evidence": {
            "stages_completed": ["T", "U", "V", "W", "X", "Y"],
            "policy_optimizer_receipt_validated": True,
            "harness_optimizer_receipt_validated": True,
            "exact_recovery_validated": True,
            "four_live_heldout_gates_validated": True,
            "production_activation_validated": True,
            "production_worker_cleanup_validated": True,
            "gpu_launch_guards_validated": True,
            "gpu_memory_envelope_validated": True,
        },
        "digests": {
            "summary": summary["record_sha256"],
            "policy_optimizer_receipt": policy_audit.record_sha256,
            "harness_optimizer_receipt": harness_receipt["record_sha256"],
            "joint_candidate_bundle": bundle_audit.record_sha256,
            "exact_recovery": recovery_audit["record_sha256"],
            "candidate_acceptance": acceptance_audit.record_sha256,
            "production_attestation": attestation["record_sha256"],
            "production_cleanup": cleanup["record_sha256"],
            "gpu_memory_audit": memory["record_sha256"],
        },
        "passed": True,
    }
    _assert_no_secrets(record)
    record["record_sha256"] = _record_sha256(record)
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-project-commit", required=True)
    parser.add_argument("--expected-areal-commit", default=PINNED_AREAL_COMMIT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = require_outside_repository(args.output)
    if output.exists():
        raise ValueError("M0 verification output already exists")
    run_root = require_outside_repository(args.run_root)
    if output.parent != run_root:
        raise ValueError("M0 verification must remain at the run root")
    record = verify_m0_live_joint(
        run_root=run_root,
        expected_project_commit=args.expected_project_commit,
        expected_areal_commit=args.expected_areal_commit,
    )
    _write_verification(output, record)
    print(json.dumps(record, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
