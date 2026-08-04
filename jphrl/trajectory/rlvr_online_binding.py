from __future__ import annotations

"""Persistent V2-agent to RLVR pre-batch binding.

The V2 inference controller runs an agent against an OpenAI-compatible
Gateway.  The agent can observe the public completion ID, but only the
DataProxy still owns the exact ``InteractionWithTokenLogpReward`` object.  This
module joins those two observations through a private, run-scoped journal and
finalizes P/Q/R/S only inside the pre-batch callback.
"""

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from jphrl.experiments.m0_agent_service_entry import (
    _decision_record,
    _parse_decision,
    _parse_joint_version,
    _parse_state,
    _validate_harness_replay,
    _validate_prompt_and_runtime,
)
from jphrl.harness.controller import HarnessDecision, HarnessState
from jphrl.paths import require_outside_repository
from jphrl.trajectory.areal_data_proxy_pre_batch import (
    HOOK_STAGE,
    VerifiedDataProxyPreBatchHook,
    validate_pre_batch_trajectory_export,
)
from jphrl.trajectory.areal_interaction_sidecar import (
    InteractionBinding,
    build_interaction_adapter_sidecar,
)
from jphrl.trajectory.areal_joint_bridge import (
    build_areal_joint_bridge_record,
    write_areal_joint_bridge_record,
)
from jphrl.trajectory.rlvr_workflow_admission import (
    load_frozen_dual_credit_estimator_template_file,
    materialize_dual_credit_estimator_from_template,
    prepare_rlvr_workflow_joint_admission,
    rlvr_runner_admission_path_for_request,
    write_rlvr_workflow_runner_admission,
)
from jphrl.trajectory.schema import JointVersion


RLVR_AGENT_STAGED_SCHEMA_VERSION = "jph.rlvr-v2-agent-staged.v1"
RLVR_AGENT_FINALIZED_SCHEMA_VERSION = "jph.rlvr-v2-agent-finalized.v1"
RLVR_AGENT_PAIR_INDEX_SCHEMA_VERSION = "jph.rlvr-v2-agent-pair-index.v1"
_STAGED_SCOPE = {
    "v2_agent_response_id_captured": True,
    "torch_harness_sampling_replayed": True,
    "pre_batch_interaction_binding": False,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}
_FINALIZED_SCOPE = {
    "v2_agent_response_id_captured": True,
    "torch_harness_sampling_replayed": True,
    "pre_batch_interaction_binding": True,
    "p_q_r_s_admitted": True,
    "policy_optimizer_update": False,
    "harness_optimizer_update": False,
}
_SECRET_NAMES = frozenset(
    {
        "admin_api_key",
        "api_key",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "session_api_key",
        "token",
    }
)
_SECRET_SUFFIXES = (
    "_api_key",
    "_access_key",
    "_private_key",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)


class RlvrOnlineBindingError(ValueError):
    """Raised when a V2-agent observation cannot become RLVR evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RlvrOnlineBindingError(message)


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
        raise RlvrOnlineBindingError(
            "RLVR V2-agent evidence is not finite canonical JSON"
        ) from exc


def _record_sha256(record: Mapping[str, object]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            _require(
                normalized not in _SECRET_NAMES
                and not normalized.endswith(_SECRET_SUFFIXES),
                f"credential field cannot enter RLVR V2-agent evidence: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str):
        normalized = value.strip().lower()
        _require(
            not normalized.startswith(("sk-", "bearer ")),
            f"credential-shaped value cannot enter RLVR V2-agent evidence: {path}",
        )


def _public_id(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} is missing")
    result = value.strip()
    _require(
        not result.lower().startswith(("sk-", "bearer ")),
        f"{label} looks like a credential",
    )
    return result


def _journal_root(value: str | Path) -> Path:
    try:
        root = require_outside_repository(value)
    except ValueError as exc:
        raise RlvrOnlineBindingError(str(exc)) from exc
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = root.resolve()
    _require(root.is_dir() and not root.is_symlink(), "RLVR journal root is unsafe")
    root.chmod(0o700)
    _require(
        root.stat().st_mode & 0o077 == 0,
        "RLVR journal root is accessible to group or other users",
    )
    return root


def _interaction_key(interaction_id: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "jph.rlvr-v2-agent-interaction-key.v1",
                "interaction_id": interaction_id,
            }
        )
    ).hexdigest()


def _model_call_key(model_call_id: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "schema_version": "jph.rlvr-v2-agent-model-call-key.v1",
                "model_call_id": model_call_id,
            }
        )
    ).hexdigest()


def _paths(root: Path, interaction_id: str) -> tuple[Path, Path]:
    key = _interaction_key(interaction_id)
    return root / "pending" / f"{key}.json", root / "finalized" / f"{key}.json"


def _pair_index_path(root: Path, model_call_id: str) -> Path:
    return root / "by-model-call" / f"{_model_call_key(model_call_id)}.json"


def _write_new(record: Mapping[str, object], path: Path, root: Path) -> Path:
    _assert_no_secrets(record)
    payload = _canonical_json(record) + b"\n"
    resolved_parent = path.parent.resolve()
    _require(
        resolved_parent == root or root in resolved_parent.parents,
        "RLVR journal path escapes its root",
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def _read_private(path: Path, root: Path) -> dict[str, object]:
    _require(path.is_file() and not path.is_symlink(), "RLVR staged record is missing")
    resolved = path.resolve()
    _require(root in resolved.parents, "RLVR staged record escapes its root")
    _require(
        resolved.stat().st_mode & 0o077 == 0,
        "RLVR staged record is accessible to group or other users",
    )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RlvrOnlineBindingError("cannot read RLVR staged record") from exc
    _require(isinstance(value, dict), "RLVR staged record must be an object")
    return value


def _pair_index_record(*, model_call_id: str, interaction_id: str) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": RLVR_AGENT_PAIR_INDEX_SCHEMA_VERSION,
        "model_call_id": model_call_id,
        "interaction_id": interaction_id,
    }
    record["record_sha256"] = _record_sha256(record)
    return record


def _validate_pair_index(
    record: Mapping[str, object],
    *,
    model_call_id: str,
    interaction_id: str,
) -> None:
    _require(
        set(record)
        == {
            "schema_version",
            "model_call_id",
            "interaction_id",
            "record_sha256",
        }
        and record.get("schema_version") == RLVR_AGENT_PAIR_INDEX_SCHEMA_VERSION
        and record.get("record_sha256") == _record_sha256(record)
        and record.get("model_call_id") == model_call_id
        and record.get("interaction_id") == interaction_id,
        "RLVR model-call and interaction pair index differs",
    )
    _assert_no_secrets(record)


def stage_rlvr_v2_agent_response(
    *,
    journal_root: str | Path,
    task_id: int,
    interaction_id: str,
    episode_id: str,
    model_call_id: str,
    joint_version: JointVersion,
    harness_state: HarnessState,
    harness_decision: HarnessDecision,
    harness_sampling_before_checkpoint: Mapping[str, object],
    base_messages: Sequence[Mapping[str, str]],
    effective_messages: Sequence[Mapping[str, str]],
    base_input_tokens: Sequence[int],
    effective_input_tokens: Sequence[int],
    expected_policy_version: int,
    project_commit: str,
    areal_commit: str,
    behavior_snapshot_path: str,
    behavior_revision: str,
    dataset_selection: str,
    sglang_version: str,
    generation_logprob_mode: str,
    inference_runtime_contract: Mapping[str, object],
    export_style: str = "individual",
    turn_discount: float = 1.0,
) -> Path:
    """Persist all non-tensor evidence before the DataProxy export begins."""

    _require(type(joint_version) is JointVersion, "RLVR staged JointVersion is invalid")
    interaction_id = _public_id(interaction_id, "interaction ID")
    episode_id = _public_id(episode_id, "episode ID")
    model_call_id = _public_id(model_call_id, "model-call ID")
    _assert_no_secrets(inference_runtime_contract, "inference_runtime_contract")
    try:
        request_id = _validate_prompt_and_runtime(
            task_id=task_id,
            state=harness_state,
            decision=harness_decision,
            base_messages=base_messages,
            effective_messages=effective_messages,
            base_input_tokens=base_input_tokens,
            effective_input_tokens=effective_input_tokens,
            dataset_selection=dataset_selection,
            inference_runtime_contract=inference_runtime_contract,
        )
        _validate_harness_replay(
            state=harness_state,
            decision=harness_decision,
            checkpoint=harness_sampling_before_checkpoint,
            joint_version=joint_version,
        )
    except ValueError as exc:
        raise RlvrOnlineBindingError(str(exc)) from exc
    _require(
        export_style == "individual" and float(turn_discount) == 1.0,
        "formal RLVR V2-agent export must be individual with discount 1.0",
    )
    record: dict[str, object] = {
        "schema_version": RLVR_AGENT_STAGED_SCHEMA_VERSION,
        "identity": {
            "task_id": task_id,
            "request_id": request_id,
            "interaction_id": interaction_id,
            "episode_id": episode_id,
            "model_call_id": model_call_id,
            "joint_version_id": joint_version.version_id,
        },
        "joint_version": asdict(joint_version),
        "harness": {
            "state": asdict(harness_state),
            "decision": _decision_record(harness_decision),
            "sampling_before_checkpoint": deepcopy(
                dict(harness_sampling_before_checkpoint)
            ),
        },
        "prompt": {
            "base_messages": [dict(item) for item in base_messages],
            "effective_messages": [dict(item) for item in effective_messages],
            "base_input_tokens": list(base_input_tokens),
            "effective_input_tokens": list(effective_input_tokens),
        },
        "runtime": {
            "expected_policy_version": expected_policy_version,
            "project_commit": project_commit,
            "areal_commit": areal_commit,
            "behavior_snapshot_path": behavior_snapshot_path,
            "behavior_revision": behavior_revision,
            "dataset_selection": dataset_selection,
            "sglang_version": sglang_version,
            "generation_logprob_mode": generation_logprob_mode,
            "inference_runtime_contract": deepcopy(dict(inference_runtime_contract)),
        },
        "export": {"style": export_style, "turn_discount": float(turn_discount)},
        "evidence_scope": dict(_STAGED_SCOPE),
    }
    _assert_no_secrets(record)
    record["record_sha256"] = _record_sha256(record)
    validate_staged_rlvr_v2_agent_response(record)
    root = _journal_root(journal_root)
    pending, finalized = _paths(root, interaction_id)
    pair_index = _pair_index_path(root, model_call_id)
    _require(not finalized.exists(), "RLVR interaction is already finalized")
    try:
        staged_path = _write_new(record, pending, root)
    except FileExistsError as exc:
        raise RlvrOnlineBindingError("RLVR interaction is already staged") from exc
    pair_record = _pair_index_record(
        model_call_id=model_call_id,
        interaction_id=interaction_id,
    )
    try:
        _write_new(pair_record, pair_index, root)
    except FileExistsError as exc:
        try:
            existing = _read_private(pair_index, root)
            _validate_pair_index(
                existing,
                model_call_id=model_call_id,
                interaction_id=interaction_id,
            )
        except BaseException:
            staged_path.unlink(missing_ok=True)
            raise
        staged_path.unlink(missing_ok=True)
        raise RlvrOnlineBindingError(
            "RLVR model-call and interaction pair is already staged"
        ) from exc
    return staged_path


def validate_staged_rlvr_v2_agent_response(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Revalidate a staged V2-agent observation without trusting its hash alone."""

    _require(
        set(record)
        == {
            "schema_version",
            "identity",
            "joint_version",
            "harness",
            "prompt",
            "runtime",
            "export",
            "evidence_scope",
            "record_sha256",
        },
        "RLVR staged field set differs",
    )
    _require(
        record.get("schema_version") == RLVR_AGENT_STAGED_SCHEMA_VERSION
        and record.get("record_sha256") == _record_sha256(record),
        "RLVR staged schema or hash differs",
    )
    _assert_no_secrets(record)
    identity = record.get("identity")
    harness = record.get("harness")
    prompt = record.get("prompt")
    runtime = record.get("runtime")
    export = record.get("export")
    _require(
        isinstance(identity, Mapping)
        and set(identity)
        == {
            "task_id",
            "request_id",
            "interaction_id",
            "episode_id",
            "model_call_id",
            "joint_version_id",
        },
        "RLVR staged identity differs",
    )
    _require(
        isinstance(harness, Mapping)
        and set(harness)
        == {"state", "decision", "sampling_before_checkpoint"}
        and isinstance(prompt, Mapping)
        and set(prompt)
        == {
            "base_messages",
            "effective_messages",
            "base_input_tokens",
            "effective_input_tokens",
        }
        and isinstance(runtime, Mapping)
        and set(runtime)
        == {
            "expected_policy_version",
            "project_commit",
            "areal_commit",
            "behavior_snapshot_path",
            "behavior_revision",
            "dataset_selection",
            "sglang_version",
            "generation_logprob_mode",
            "inference_runtime_contract",
        }
        and isinstance(export, Mapping)
        and set(export) == {"style", "turn_discount"},
        "RLVR staged Harness, prompt, runtime, or export differs",
    )
    try:
        joint_version = _parse_joint_version(record.get("joint_version"))
        state = _parse_state(harness["state"])
        decision = _parse_decision(harness["decision"])
        _validate_harness_replay(
            state=state,
            decision=decision,
            checkpoint=harness["sampling_before_checkpoint"],
            joint_version=joint_version,
        )
        request_id = _validate_prompt_and_runtime(
            task_id=identity["task_id"],
            state=state,
            decision=decision,
            base_messages=prompt["base_messages"],
            effective_messages=prompt["effective_messages"],
            base_input_tokens=prompt["base_input_tokens"],
            effective_input_tokens=prompt["effective_input_tokens"],
            dataset_selection=runtime["dataset_selection"],
            inference_runtime_contract=runtime["inference_runtime_contract"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RlvrOnlineBindingError(str(exc)) from exc
    _require(
        identity["request_id"] == request_id
        and identity["joint_version_id"] == joint_version.version_id
        and all(
            _public_id(identity[field], field)
            for field in ("interaction_id", "episode_id", "model_call_id")
        ),
        "RLVR staged identity is crossed",
    )
    for commit_name in ("project_commit", "areal_commit", "behavior_revision"):
        commit = runtime.get(commit_name)
        _require(
            isinstance(commit, str)
            and len(commit) == 40
            and all(character in "0123456789abcdef" for character in commit),
            f"RLVR staged {commit_name} is invalid",
        )
    _require(
        type(runtime.get("expected_policy_version")) is int
        and runtime["expected_policy_version"] >= 0
        and export == {"style": "individual", "turn_discount": 1.0}
        and record.get("evidence_scope") == _STAGED_SCOPE,
        "RLVR staged runtime, export, or evidence scope differs",
    )
    return {
        "identity": dict(identity),
        "joint_version": joint_version,
        "state": state,
        "decision": decision,
        "checkpoint": deepcopy(dict(harness["sampling_before_checkpoint"])),
        "record_sha256": record["record_sha256"],
    }


def validate_finalized_rlvr_v2_agent_admission(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Validate the terminal journal marker before treating it as finalized."""

    _require(
        set(record)
        == {
            "schema_version",
            "identity",
            "staged_record_sha256",
            "bridge_record_sha256",
            "bridge_record_path",
            "runner_admission_sha256",
            "runner_admission_path",
            "evidence_scope",
            "record_sha256",
        }
        and record.get("schema_version") == RLVR_AGENT_FINALIZED_SCHEMA_VERSION
        and record.get("record_sha256") == _record_sha256(record),
        "RLVR finalized schema or hash differs",
    )
    _assert_no_secrets(record)
    identity = record.get("identity")
    _require(
        isinstance(identity, Mapping)
        and set(identity)
        == {
            "task_id",
            "request_id",
            "interaction_id",
            "episode_id",
            "model_call_id",
            "joint_version_id",
            "session_id",
            "trajectory_id",
        }
        and all(
            _public_id(identity[field], field)
            for field in (
                "request_id",
                "interaction_id",
                "episode_id",
                "model_call_id",
                "joint_version_id",
                "session_id",
            )
        )
        and type(identity["task_id"]) is int
        and type(identity["trajectory_id"]) is int
        and identity["trajectory_id"] >= 0,
        "RLVR finalized identity differs",
    )
    _require(
        all(
            isinstance(record.get(field), str)
            and len(record[field]) == 64
            and all(character in "0123456789abcdef" for character in record[field])
            for field in (
                "staged_record_sha256",
                "bridge_record_sha256",
                "runner_admission_sha256",
            )
        )
        and all(
            isinstance(record.get(field), str) and bool(record[field])
            for field in ("bridge_record_path", "runner_admission_path")
        )
        and record.get("evidence_scope") == _FINALIZED_SCOPE,
        "RLVR finalized evidence differs",
    )
    return dict(record)


class PersistentRlvrV2AgentPreBatchBinder:
    """Finalize one staged V2-agent response against one real interaction."""

    def __init__(self, journal_root: str | Path) -> None:
        self._root = _journal_root(journal_root)

    def __call__(self, event: object) -> dict[str, object]:
        _require(
            getattr(event, "hook_stage", None) == HOOK_STAGE,
            "RLVR V2-agent binder was invoked outside the pre-batch seam",
        )
        interactions = getattr(event, "exported_interactions", None)
        _require(
            isinstance(interactions, Mapping) and len(interactions) == 1,
            "formal RLVR V2-agent export requires exactly one interaction",
        )
        try:
            validate_pre_batch_trajectory_export(event)
        except ValueError as exc:
            raise RlvrOnlineBindingError(str(exc)) from exc
        interaction_id, interaction = next(iter(interactions.items()))
        interaction_id = _public_id(interaction_id, "pre-batch interaction ID")
        pending, finalized = _paths(self._root, interaction_id)
        if finalized.exists():
            validate_finalized_rlvr_v2_agent_admission(
                _read_private(finalized, self._root)
            )
            raise RlvrOnlineBindingError("RLVR interaction is already finalized")
        staged_record = _read_private(pending, self._root)
        staged = validate_staged_rlvr_v2_agent_response(staged_record)
        identity = staged["identity"]
        _require(
            identity["interaction_id"] == interaction_id,
            "pre-batch interaction differs from the staged response ID",
        )
        pair_index = _read_private(
            _pair_index_path(self._root, identity["model_call_id"]),
            self._root,
        )
        _validate_pair_index(
            pair_index,
            model_call_id=identity["model_call_id"],
            interaction_id=interaction_id,
        )
        session_id = _public_id(getattr(event, "session_id"), "pre-batch session ID")
        joint_version = staged["joint_version"]
        sidecar = build_interaction_adapter_sidecar(
            [
                InteractionBinding(
                    episode_id=identity["episode_id"],
                    model_call_id=identity["model_call_id"],
                    session_id=None,
                    trajectory_id=None,
                    interaction_id=interaction_id,
                    parent_interaction_id=None,
                    ordinal=0,
                    joint_version_id=joint_version.version_id,
                    route_kind="rlvr-workflow",
                )
            ]
        )
        runtime = staged_record["runtime"]
        prompt = staged_record["prompt"]
        bridge = build_areal_joint_bridge_record(
            task_id=identity["task_id"],
            request_id=identity["request_id"],
            joint_version=joint_version,
            expected_policy_version=runtime["expected_policy_version"],
            harness_state=staged["state"],
            harness_decision=staged["decision"],
            harness_controller_checkpoint=staged["checkpoint"],
            base_messages=prompt["base_messages"],
            effective_messages=prompt["effective_messages"],
            base_input_tokens=prompt["base_input_tokens"],
            effective_input_tokens=prompt["effective_input_tokens"],
            model_response=interaction.model_response,
            interaction=interaction,
            tensor_dict=interaction.to_tensor_dict(),
            project_commit=runtime["project_commit"],
            areal_commit=runtime["areal_commit"],
            behavior_snapshot_path=runtime["behavior_snapshot_path"],
            behavior_revision=runtime["behavior_revision"],
            dataset_selection=runtime["dataset_selection"],
            sglang_version=runtime["sglang_version"],
            generation_logprob_mode=runtime["generation_logprob_mode"],
            inference_runtime_contract=runtime["inference_runtime_contract"],
            interaction_adapter_sidecar=sidecar,
        )
        estimator_template_path = os.environ.get(
            "JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH", ""
        )
        _require(bool(estimator_template_path), "RLVR estimator template is missing")
        template = load_frozen_dual_credit_estimator_template_file(
            estimator_template_path,
            allowed_root=os.environ.get("JPH_ROOT"),
        )
        estimator = materialize_dual_credit_estimator_from_template(
            template,
            joint_version=joint_version,
            model_call_id=identity["model_call_id"],
            harness_decision_id=staged["decision"].decision_id,
        )
        admission = prepare_rlvr_workflow_joint_admission(
            bridge,
            pre_batch_interaction=interaction,
            estimator=estimator,
            active_joint_version=joint_version,
            export_style=getattr(event, "export_style"),
            turn_discount=float(getattr(event, "turn_discount")),
        )
        bridge_path = write_areal_joint_bridge_record(
            bridge,
            trace_dir=os.environ.get("JPH_AREAL_JOINT_BRIDGE_DIR", ""),
            allowed_root=os.environ.get("JPH_ROOT"),
        )
        runner_path = rlvr_runner_admission_path_for_request(
            output_dir=os.environ.get("JPH_RLVR_RUNNER_ADMISSION_DIR", ""),
            request_id=identity["request_id"],
            allowed_root=os.environ.get("JPH_ROOT"),
        )
        write_rlvr_workflow_runner_admission(
            admission.runner_admission,
            output_path=runner_path,
            allowed_root=os.environ.get("JPH_ROOT"),
            active_joint_version=joint_version,
        )
        marker: dict[str, object] = {
            "schema_version": RLVR_AGENT_FINALIZED_SCHEMA_VERSION,
            "identity": {
                **dict(identity),
                "session_id": session_id,
                "trajectory_id": getattr(event, "trajectory_id"),
            },
            "staged_record_sha256": staged["record_sha256"],
            "bridge_record_sha256": bridge["record_sha256"],
            "bridge_record_path": str(bridge_path),
            "runner_admission_sha256": admission.runner_admission["record_sha256"],
            "runner_admission_path": str(runner_path),
            "evidence_scope": dict(_FINALIZED_SCOPE),
        }
        _assert_no_secrets(marker)
        marker["record_sha256"] = _record_sha256(marker)
        validate_finalized_rlvr_v2_agent_admission(marker)
        _write_new(marker, finalized, self._root)
        return marker


async def pre_batch_finalize_rlvr_v2_agent_admission(
    *,
    session_id: str,
    trajectory_id: int,
    interactions: Mapping[str, object],
    discount: float,
    style: str,
) -> None:
    """Deployment callback loaded by the run-scoped patched DataProxy."""

    journal_root = os.environ.get("JPH_RLVR_V2_AGENT_JOURNAL_ROOT", "")
    _require(bool(journal_root), "JPH_RLVR_V2_AGENT_JOURNAL_ROOT is required")
    hook = VerifiedDataProxyPreBatchHook(
        PersistentRlvrV2AgentPreBatchBinder(journal_root)
    )
    await hook(
        session_id=session_id,
        trajectory_id=trajectory_id,
        interactions=interactions,
        discount=discount,
        style=style,
    )


__all__ = [
    "PersistentRlvrV2AgentPreBatchBinder",
    "RLVR_AGENT_FINALIZED_SCHEMA_VERSION",
    "RLVR_AGENT_PAIR_INDEX_SCHEMA_VERSION",
    "RLVR_AGENT_STAGED_SCHEMA_VERSION",
    "RlvrOnlineBindingError",
    "pre_batch_finalize_rlvr_v2_agent_admission",
    "stage_rlvr_v2_agent_response",
    "validate_finalized_rlvr_v2_agent_admission",
    "validate_staged_rlvr_v2_agent_response",
]
