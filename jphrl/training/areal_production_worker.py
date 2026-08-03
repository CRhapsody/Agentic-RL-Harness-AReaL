from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from jphrl.paths import require_outside_repository, require_within_configured_root
from jphrl.trajectory.schema import JointVersion

from .areal_policy_candidate import (
    PINNED_AREAL_COMMIT,
    checkpoint_manifest,
    validate_areal_policy_candidate,
)
from .joint_activation import (
    JointActivationError,
    ProductionActivationWorker,
    ProductionReleaseTarget,
    ProductionWorkerState,
)

SERVING_EXPORT_SCHEMA = "jph.areal-serving-export-lineage.v1"
PRODUCTION_PROBE_OUTPUT_SCHEMA = "jph.m0-production-probe-output.v1"
_LIVE_EXPORT_TOKEN = object()
_WORKER_CONSTRUCTION_TOKEN = object()
_SECRET_FIELDS = {
    "access_token",
    "admin_api_key",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "session_api_key",
    "token",
}


class ArealProductionWorkerError(JointActivationError):
    """A serving export or live rollout observation failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArealProductionWorkerError(message)


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
        raise ArealProductionWorkerError(
            "production rollout evidence is not finite canonical JSON"
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


def _assert_no_secrets(value: object, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            _require(
                normalized not in _SECRET_FIELDS
                and not normalized.endswith(
                    ("_api_key", "_password", "_secret", "_token")
                ),
                f"credential field cannot enter production rollout evidence: {path}.{key}",
            )
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest(root: Path) -> dict[str, object]:
    _require(root.is_dir() and not root.is_symlink(), "serving export is missing")
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "serving export contains a symbolic link")
        if path.is_dir():
            continue
        _require(path.is_file(), "serving export contains a non-regular entry")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    _require(bool(files), "serving export contains no files")
    result: dict[str, object] = {"files": files}
    result["manifest_sha256"] = _sha256(result)
    return result


def _tensor_bytes(value: object) -> bytes:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise ArealProductionWorkerError("torch is required for serving proof") from exc
    _require(isinstance(value, torch.Tensor), "serving parameter is not a tensor")
    tensor = value.detach().cpu().contiguous()
    _require(
        not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
        "serving parameter is non-finite",
    )
    return bytes(tensor.view(torch.uint8).untyped_storage())


def _parameter_digest(tensors: Mapping[str, object]) -> str:
    _require(bool(tensors), "serving parameter set is empty")
    digest = hashlib.sha256()
    digest.update(b"jph.areal-canonical-hf-parameter-set.v1")
    for name in sorted(tensors):
        value = tensors[name]
        _require(isinstance(name, str) and bool(name), "parameter name is invalid")
        try:
            shape = list(value.shape)  # type: ignore[attr-defined]
            dtype = str(value.dtype)  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise ArealProductionWorkerError("serving parameter is not a tensor") from exc
        digest.update(name.encode("utf-8"))
        digest.update(dtype.encode("ascii"))
        digest.update(_canonical_json(shape))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _load_safetensor_export(root: Path) -> dict[str, object]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise ArealProductionWorkerError(
            "safetensors is required to verify the AReaL HF export"
        ) from exc
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            record = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArealProductionWorkerError("serving export index is invalid") from exc
        weight_map = record.get("weight_map")
        _require(isinstance(weight_map, Mapping), "serving export weight map is missing")
        shard_names = sorted(set(weight_map.values()))
        _require(
            all(isinstance(name, str) and name for name in shard_names),
            "serving export shard name is invalid",
        )
        paths = [root / str(name) for name in shard_names]
    else:
        paths = sorted(root.glob("*.safetensors"))
    _require(bool(paths), "AReaL HF export has no safetensors weights")
    tensors: dict[str, object] = {}
    for path in paths:
        _require(
            path.is_file() and not path.is_symlink() and path.parent == root,
            "serving export weight shard is unsafe",
        )
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle:
                _require(name not in tensors, "serving export parameter is duplicated")
                tensors[name] = handle.get_tensor(name)
    if index.is_file():
        _require(set(tensors) == set(weight_map), "serving export index differs from weights")
    return tensors


def _write_new_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(record))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ArealProductionWorkerError(
                "serving lineage record already exists"
            ) from exc
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ArealServingExportAudit:
    joint_version: JointVersion
    policy_engine_version: int
    policy_candidate_record_sha256: str
    source_dcp_path: str
    source_dcp_manifest_sha256: str
    serving_export_path: str
    serving_export_manifest_sha256: str
    serving_parameter_sha256: str
    record_path: str
    record_sha256: str

    def validate(self) -> None:
        _require(
            type(self.policy_engine_version) is int and self.policy_engine_version >= 0,
            "serving export Policy engine version is invalid",
        )
        for digest in (
            self.source_dcp_manifest_sha256,
            self.policy_candidate_record_sha256,
            self.serving_export_manifest_sha256,
            self.serving_parameter_sha256,
            self.record_sha256,
        ):
            _require(_is_sha256(digest), "serving export digest is invalid")

    def to_record(self) -> dict[str, object]:
        self.validate()
        return {
            "joint_version": asdict(self.joint_version),
            "joint_version_id": self.joint_version.version_id,
            "policy_engine_version": self.policy_engine_version,
            "policy_candidate_record_sha256": self.policy_candidate_record_sha256,
            "source_dcp_path": self.source_dcp_path,
            "source_dcp_manifest_sha256": self.source_dcp_manifest_sha256,
            "serving_export_path": self.serving_export_path,
            "serving_export_manifest_sha256": self.serving_export_manifest_sha256,
            "serving_parameter_sha256": self.serving_parameter_sha256,
            "record_path": self.record_path,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True, init=False)
class LiveArealServingExportPair:
    """Process-local proof that one real actor emitted both serving exports."""

    parent: ArealServingExportAudit
    candidate: ArealServingExportAudit
    _token: object

    @classmethod
    def _create(
        cls,
        *,
        parent: ArealServingExportAudit,
        candidate: ArealServingExportAudit,
        token: object,
    ) -> LiveArealServingExportPair:
        _require(token is _LIVE_EXPORT_TOKEN, "live serving export token is invalid")
        instance = object.__new__(cls)
        object.__setattr__(instance, "parent", parent)
        object.__setattr__(instance, "candidate", candidate)
        object.__setattr__(instance, "_token", token)
        return instance


def require_live_areal_serving_export_pair(value: object) -> LiveArealServingExportPair:
    _require(
        type(value) is LiveArealServingExportPair
        and getattr(value, "_token", None) is _LIVE_EXPORT_TOKEN,
        "production rollout worker requires native live serving-export lineage",
    )
    value.parent.validate()
    value.candidate.validate()
    _require(
        value.candidate.policy_engine_version == value.parent.policy_engine_version + 1,
        "serving export candidate is not the next Policy engine version",
    )
    return value


def _materialize_one(
    *,
    actor: object,
    dcp_path: str,
    expected_dcp_sha256: str,
    export_path: Path,
    joint_version: JointVersion,
    policy_engine_version: int,
    policy_candidate_record_sha256: str,
) -> ArealServingExportAudit:
    source = require_within_configured_root(dcp_path)
    actual_dcp = checkpoint_manifest(source)
    _require(
        actual_dcp["manifest_sha256"] == expected_dcp_sha256,
        "source DCP differs from its Policy candidate manifest",
    )
    _require(not export_path.exists(), "serving export path must be new")
    try:
        from areal.api import SaveLoadMeta
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise ArealProductionWorkerError("pinned AReaL SaveLoadMeta is unavailable") from exc
    actor.load(meta=SaveLoadMeta(path=str(source), weight_format="dcp", with_optim=True))
    actor.save(
        meta=SaveLoadMeta(
            path=str(export_path),
            weight_format="hf",
            with_optim=False,
            tokenizer=getattr(actor, "tokenizer", None),
            processor=getattr(actor, "processor", None),
        )
    )
    export_manifest = _directory_manifest(export_path)
    parameter_sha256 = _parameter_digest(_load_safetensor_export(export_path))
    record_path = export_path.parent / f"{export_path.name}-lineage.json"
    record: dict[str, object] = {
        "schema_version": SERVING_EXPORT_SCHEMA,
        "areal_commit": PINNED_AREAL_COMMIT,
        "joint_version": asdict(joint_version),
        "joint_version_id": joint_version.version_id,
        "policy_engine_version": policy_engine_version,
        "policy_candidate_record_sha256": policy_candidate_record_sha256,
        "source_dcp_path": str(source),
        "source_dcp_manifest_sha256": expected_dcp_sha256,
        "serving_export_path": str(export_path),
        "serving_export_manifest": export_manifest,
        "serving_parameter_sha256": parameter_sha256,
        "exporter": "areal.engine.fsdp_engine.FSDPPPOActor.SaveLoadMeta(hf)",
    }
    _assert_no_secrets(record)
    record["record_sha256"] = _record_sha256(record)
    _write_new_json(record_path, record)
    return ArealServingExportAudit(
        joint_version=joint_version,
        policy_engine_version=policy_engine_version,
        policy_candidate_record_sha256=policy_candidate_record_sha256,
        source_dcp_path=str(source),
        source_dcp_manifest_sha256=expected_dcp_sha256,
        serving_export_path=str(export_path),
        serving_export_manifest_sha256=str(export_manifest["manifest_sha256"]),
        serving_parameter_sha256=parameter_sha256,
        record_path=str(record_path),
        record_sha256=str(record["record_sha256"]),
    )


def materialize_areal_serving_export_pair(
    *,
    actor: object,
    policy_candidate_record: Mapping[str, object],
    export_root: str | Path,
    parent_joint_version: JointVersion,
    candidate_joint_version: JointVersion,
) -> LiveArealServingExportPair:
    """Load each sealed DCP into the live actor and emit serving-compatible HF."""

    module = type(actor).__module__
    _require(
        type(actor).__name__ == "FSDPPPOActor" and module.startswith("areal."),
        "serving export requires the real pinned AReaL FSDPPPOActor",
    )
    policy = validate_areal_policy_candidate(
        policy_candidate_record,
        active_joint_version=parent_joint_version,
        require_checkpoints=True,
    )
    _require(
        candidate_joint_version.policy == policy.candidate_policy_version,
        "serving export candidate JointVersion differs from T",
    )
    checkpoints = policy_candidate_record["checkpoints"]
    root = require_outside_repository(export_root)
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    try:
        parent = _materialize_one(
            actor=actor,
            dcp_path=str(checkpoints["parent_path"]),
            expected_dcp_sha256=str(
                checkpoints["parent_manifest"]["manifest_sha256"]
            ),
            export_path=root / "policy-parent-hf",
            joint_version=parent_joint_version,
            policy_engine_version=policy.parent_engine_version,
            policy_candidate_record_sha256=policy.record_sha256,
        )
        candidate = _materialize_one(
            actor=actor,
            dcp_path=str(checkpoints["candidate_path"]),
            expected_dcp_sha256=str(
                checkpoints["candidate_manifest"]["manifest_sha256"]
            ),
            export_path=root / "policy-candidate-hf",
            joint_version=candidate_joint_version,
            policy_engine_version=policy.reserved_candidate_engine_version,
            policy_candidate_record_sha256=policy.record_sha256,
        )
    finally:
        try:
            from areal.api import SaveLoadMeta

            actor.load(
                meta=SaveLoadMeta(
                    path=str(checkpoints["candidate_path"]),
                    weight_format="dcp",
                    with_optim=True,
                )
            )
        except BaseException as exc:
            raise ArealProductionWorkerError(
                "serving export failed to restore the sealed candidate DCP"
            ) from exc
    return LiveArealServingExportPair._create(
        parent=parent,
        candidate=candidate,
        token=_LIVE_EXPORT_TOKEN,
    )


def build_production_probe_output(
    *,
    fixture: bytes,
    target_release_id: str,
    target_joint_version: JointVersion,
    policy_engine_version: int,
    policy_checkpoint_sha256: str,
    serving_parameter_sha256: str,
    harness_checkpoint_sha256: str,
    harness_parameter_sha256: str,
) -> bytes:
    """Build raw deterministic joint bytes from measured component identities."""

    _require(type(fixture) is bytes and bool(fixture), "production fixture is empty")
    for digest in (
        policy_checkpoint_sha256,
        serving_parameter_sha256,
        harness_checkpoint_sha256,
        harness_parameter_sha256,
    ):
        _require(_is_sha256(digest), "production probe component digest is invalid")
    record = {
        "schema_version": PRODUCTION_PROBE_OUTPUT_SCHEMA,
        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "release_id": target_release_id,
        "joint_version_id": target_joint_version.version_id,
        "policy_engine_version": policy_engine_version,
        "policy_checkpoint_sha256": policy_checkpoint_sha256,
        "serving_parameter_sha256": serving_parameter_sha256,
        "harness_checkpoint_sha256": harness_checkpoint_sha256,
        "harness_parameter_sha256": harness_parameter_sha256,
    }
    return _canonical_json(record)


@dataclass(frozen=True)
class HarnessServingCheckpoint:
    path: str
    checkpoint_sha256: str
    kind: str

    def validate(self) -> None:
        _require(self.kind in {"rollout_json", "candidate_pt"}, "Harness checkpoint kind is invalid")
        _require(_is_sha256(self.checkpoint_sha256), "Harness checkpoint hash is invalid")


class PinnedArealSGLangActivationWorker(ProductionActivationWorker):
    """M0 single-worker adapter with active SGLang tensor observations."""

    def __init__(
        self,
        *,
        controller: object,
        serving_exports: LiveArealServingExportPair,
        harness_checkpoints: Mapping[str, HarnessServingCheckpoint],
        observation_root: str | Path,
        parent_release_id: str,
        candidate_release_id: str,
        request_timeout_seconds: float = 120.0,
        _construction_token: object | None = None,
    ) -> None:
        _require(
            _construction_token is _WORKER_CONSTRUCTION_TOKEN,
            "use PinnedArealSGLangActivationWorker.create() so launch failure cleans up",
        )
        self._exports = require_live_areal_serving_export_pair(serving_exports)
        _require(
            type(controller).__module__
            == "areal.v2.inference_service.controller.controller"
            and type(controller).__name__ == "RolloutControllerV2",
            "production adapter requires the pinned AReaL RolloutControllerV2",
        )
        self._controller = controller
        self._timeout = float(request_timeout_seconds)
        _require(self._timeout > 0.0, "production request timeout is invalid")
        self._root = require_outside_repository(observation_root)
        self._root.mkdir(parents=True, mode=0o700)
        os.chmod(self._root, 0o700)
        self._targets = {
            parent_release_id: self._exports.parent,
            candidate_release_id: self._exports.candidate,
        }
        _require(set(harness_checkpoints) == set(self._targets), "Harness checkpoint coverage differs")
        self._harness_specs = dict(harness_checkpoints)
        for spec in self._harness_specs.values():
            _require(type(spec) is HarnessServingCheckpoint, "Harness checkpoint spec is untyped")
            spec.validate()
        urls = tuple(controller.inference_worker_urls)
        _require(len(urls) == 1, "M0 production adapter supports exactly one inference worker")
        worker_ids = controller.worker_ids
        _require(len(worker_ids) == 1, "M0 production adapter supports exactly one data proxy")
        self._inference_url = str(urls[0]).rstrip("/")
        self._data_proxy_url, routed_worker_id = next(iter(worker_ids.items()))
        self._worker_id = f"areal-v2-{routed_worker_id}"
        _require(
            bool(parent_release_id)
            and bool(candidate_release_id)
            and parent_release_id != candidate_release_id,
            "worker release binding is invalid",
        )
        self._active_release_id = parent_release_id
        self._policy_release_id = parent_release_id
        self._harness_release_id = parent_release_id
        self._harness_policy: object | None = None
        self._closed = False
        self._load_harness(parent_release_id)
        self._verify_live_policy(parent_release_id)
        self._observe_version(self._exports.parent.policy_engine_version)
        _require(not self._is_quiesced(), "worker must bootstrap in serving state")

    @classmethod
    def create(
        cls,
        *,
        controller: object,
        serving_exports: LiveArealServingExportPair,
        harness_checkpoints: Mapping[str, HarnessServingCheckpoint],
        observation_root: str | Path,
        parent_release_id: str,
        candidate_release_id: str,
        request_timeout_seconds: float = 120.0,
    ) -> PinnedArealSGLangActivationWorker:
        """Take ownership of a controller; destroy it on every init failure."""

        try:
            return cls(
                controller=controller,
                serving_exports=serving_exports,
                harness_checkpoints=harness_checkpoints,
                observation_root=observation_root,
                parent_release_id=parent_release_id,
                candidate_release_id=candidate_release_id,
                request_timeout_seconds=request_timeout_seconds,
                _construction_token=_WORKER_CONSTRUCTION_TOKEN,
            )
        except BaseException:
            try:
                controller.destroy()
            except BaseException as cleanup_error:
                raise ArealProductionWorkerError(
                    "worker initialization and AReaL controller cleanup both failed"
                ) from cleanup_error
            raise

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def _http_json(self, method: str, url: str, body: Mapping[str, object] | None = None) -> Mapping[str, object]:
        payload = None if body is None else _canonical_json(body)
        request = urllib.request.Request(
            url,
            data=payload,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
                _require(200 <= response.status < 300, "AReaL service request failed")
        except (OSError, urllib.error.URLError) as exc:
            raise ArealProductionWorkerError("AReaL service request failed") from exc
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArealProductionWorkerError("AReaL service response is invalid") from exc
        _require(isinstance(value, Mapping), "AReaL service response is not an object")
        return value

    def _health(self) -> Mapping[str, object]:
        health = self._http_json("GET", f"{self._data_proxy_url.rstrip('/')}/health")
        _require(health.get("status") == "ok", "AReaL data proxy is unhealthy")
        return health

    def _is_quiesced(self) -> bool:
        health = self._health()
        executor_paused = bool(self._controller.workflow_executor.is_paused())
        _require(bool(health.get("paused")) == executor_paused, "dispatcher/backend pause state differs")
        return executor_paused

    def _observe_version(self, expected: int) -> None:
        health = self._health()
        _require(
            self._controller.get_version() == expected and health.get("version") == expected,
            "live AReaL Policy version differs",
        )

    def _dump_live_parameters(self) -> str:
        path = self._root / f"live-parameters-{uuid4().hex}.pt"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        inode = path.stat().st_ino
        response = self._http_json(
            "POST",
            f"{self._inference_url}/awex/debug/get_parameters",
            {"save_path": str(path), "names": None},
        )
        _require(response.get("status") == "ok", "SGLang parameter dump failed")
        deadline = time.monotonic() + self._timeout
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_ino == inode
            and path.stat().st_size > 0,
            "SGLang did not safely emit live parameters",
        )
        try:
            import torch

            try:
                tensors = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - old torch
                tensors = torch.load(path, map_location="cpu")
        finally:
            path.unlink(missing_ok=True)
        _require(isinstance(tensors, Mapping), "SGLang parameter dump is invalid")
        return _parameter_digest(tensors)

    def _verify_live_policy(self, release_id: str) -> None:
        expected = self._targets[release_id]
        _require(
            self._dump_live_parameters() == expected.serving_parameter_sha256,
            "live SGLang parameters differ from the AReaL serving export",
        )

    def _load_harness(self, release_id: str) -> None:
        spec = self._harness_specs[release_id]
        path = require_within_configured_root(spec.path)
        _require(path.is_file() and not path.is_symlink(), "Harness checkpoint is missing")
        _require(_file_sha256(path) == spec.checkpoint_sha256, "Harness checkpoint hash differs")
        try:
            from jphrl.harness.torch_learning import (
                load_torch_harness_checkpoint,
                load_torch_harness_rollout_checkpoint,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ArealProductionWorkerError("Torch Harness loader is unavailable") from exc
        if spec.kind == "candidate_pt":
            policy, _optimizer, _record = load_torch_harness_checkpoint(path, map_location="cpu")
        else:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArealProductionWorkerError("Harness rollout checkpoint is invalid") from exc
            _require(isinstance(raw, Mapping), "Harness rollout checkpoint is invalid")
            policy = load_torch_harness_rollout_checkpoint(raw)
        self._harness_policy = policy
        self._harness_release_id = release_id

    def quiesce(self) -> None:
        _require(not self._closed, "production worker is closed")
        if self._is_quiesced():
            return
        self._controller.pause()
        self._controller.pause_generation()
        _require(self._is_quiesced(), "AReaL rollout worker did not quiesce")

    def install_policy(self, target: ProductionReleaseTarget) -> None:
        _require(self._is_quiesced(), "Policy install requires a quiesced worker")
        expected = self._targets.get(target.release_id)
        _require(expected is not None, "Policy target is outside the serving export pair")
        _require(
            target.policy_checkpoint_sha256 == expected.serving_parameter_sha256
            and target.policy_engine_version == expected.policy_engine_version,
            "Policy target differs from serving-export lineage",
        )
        response = self._http_json(
            "POST",
            f"{self._inference_url}/update_weights_from_disk",
            {"model_path": expected.serving_export_path, "abort_all_requests": True},
        )
        _require(
            response.get("success") is True or response.get("status") == "ok",
            "SGLang rejected the serving export",
        )
        self._controller.set_version(target.policy_engine_version)
        self._verify_live_policy(target.release_id)
        self._observe_version(target.policy_engine_version)
        self._policy_release_id = target.release_id

    def install_harness(self, target: ProductionReleaseTarget) -> None:
        _require(self._is_quiesced(), "Harness install requires a quiesced worker")
        self._load_harness(target.release_id)
        _require(
            self._harness_specs[target.release_id].checkpoint_sha256
            == target.harness_checkpoint_sha256
            and getattr(self._harness_policy, "parameter_digest", None)
            == target.harness_parameter_digest
            and getattr(self._harness_policy, "version", None)
            == target.joint_version.harness_controller,
            "loaded Harness checkpoint differs from target",
        )

    def bind_release(self, target: ProductionReleaseTarget) -> None:
        _require(self._is_quiesced(), "release binding requires a quiesced worker")
        _require(
            self._policy_release_id == target.release_id
            and self._harness_release_id == target.release_id,
            "release binding requires both components installed",
        )
        self._active_release_id = target.release_id

    def run_probe(self, fixture: bytes) -> bytes:
        target = self._targets[self._active_release_id]
        self._verify_live_policy(self._policy_release_id)
        policy = self._harness_policy
        _require(policy is not None, "Harness policy is not installed")
        spec = self._harness_specs[self._harness_release_id]
        return build_production_probe_output(
            fixture=fixture,
            target_release_id=self._active_release_id,
            target_joint_version=target.joint_version,
            policy_engine_version=target.policy_engine_version,
            policy_checkpoint_sha256=target.source_dcp_manifest_sha256,
            serving_parameter_sha256=target.serving_parameter_sha256,
            harness_checkpoint_sha256=spec.checkpoint_sha256,
            harness_parameter_sha256=str(policy.parameter_digest),
        )

    def resume(self) -> None:
        _require(not self._closed, "production worker is closed")
        if not self._is_quiesced():
            return
        self._controller.continue_generation()
        self._controller.resume()
        _require(not self._is_quiesced(), "AReaL rollout worker did not resume")

    def read_state(self) -> ProductionWorkerState:
        _require(not self._closed, "production worker is closed")
        self._verify_live_policy(self._policy_release_id)
        policy_target = self._targets[self._policy_release_id]
        active_target = self._targets[self._active_release_id]
        policy = self._harness_policy
        _require(policy is not None, "Harness policy is not installed")
        harness_spec = self._harness_specs[self._harness_release_id]
        self._observe_version(policy_target.policy_engine_version)
        return ProductionWorkerState(
            worker_id=self.worker_id,
            lifecycle_phase="quiesced" if self._is_quiesced() else "serving",
            active_release_id=self._active_release_id,
            joint_version=active_target.joint_version,
            policy_engine_version=policy_target.policy_engine_version,
            policy_checkpoint_sha256=policy_target.serving_parameter_sha256,
            harness_controller_version=str(policy.version),
            harness_checkpoint_sha256=harness_spec.checkpoint_sha256,
            harness_parameter_digest=str(policy.parameter_digest),
        )

    def close(self) -> None:
        """Idempotently destroy the controller and its LocalScheduler children."""

        if self._closed:
            return
        try:
            self._controller.destroy()
        except BaseException as exc:
            raise ArealProductionWorkerError("AReaL controller destruction failed") from exc
        _require(
            bool(getattr(self._controller, "_destroyed", False))
            and not tuple(getattr(self._controller, "workers", ()))
            and not tuple(getattr(self._controller, "_service_roles", ()))
            and not tuple(getattr(self._controller, "_server_infos", ()))
            and not tuple(getattr(self._controller, "_inf_addrs", ()))
            and not tuple(getattr(self._controller, "_data_proxy_addrs", ()))
            and not dict(getattr(self._controller, "_worker_ids", {})),
            "AReaL controller did not release all rollout workers",
        )
        self._closed = True


def launch_pinned_areal_sglang_activation_worker(
    *,
    controller_config: object,
    scheduler: object,
    server_args: Mapping[str, object],
    serving_exports: LiveArealServingExportPair,
    harness_checkpoints: Mapping[str, HarnessServingCheckpoint],
    observation_root: str | Path,
    parent_release_id: str,
    candidate_release_id: str,
    recorded_mem_fraction_static: float,
    request_timeout_seconds: float = 120.0,
) -> PinnedArealSGLangActivationWorker:
    """Launch the pinned single-GPU controller and transfer cleanup ownership.

    The caller supplies an in-memory AReaL config (including its admin key), but
    neither this function nor the worker serializes that config.  The parent HF
    export must be the model launched by SGLang; otherwise initialization fails
    before Y can observe a parent state.
    """

    exports = require_live_areal_serving_export_pair(serving_exports)
    _assert_no_secrets(server_args, "server_args")
    _require(
        type(controller_config).__module__ == "areal.api.cli_args"
        and type(controller_config).__name__ == "InferenceEngineConfig",
        "production launch requires pinned AReaL InferenceEngineConfig",
    )
    _require(
        type(scheduler).__module__ == "areal.infra.scheduler.local"
        and type(scheduler).__name__ == "LocalScheduler",
        "M0 production launch requires pinned AReaL LocalScheduler",
    )
    fraction = server_args.get("mem_fraction_static")
    _require(
        isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and float(fraction) == float(recorded_mem_fraction_static)
        and 0.28 <= float(fraction) <= 0.30,
        "production SGLang memory fraction differs from the recorded 24-26 GiB envelope",
    )
    parent_path = exports.parent.serving_export_path
    _require(
        server_args.get("model_path") == parent_path
        and getattr(controller_config, "model", None) == parent_path
        and getattr(controller_config, "tokenizer_path", None) == parent_path
        and str(getattr(controller_config, "backend", "")).startswith("sglang:d1"),
        "production SGLang launch does not use the exact parent serving export",
    )
    try:
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise ArealProductionWorkerError(
            "pinned AReaL RolloutControllerV2 is unavailable"
        ) from exc
    controller = RolloutControllerV2(config=controller_config, scheduler=scheduler)
    try:
        controller.initialize(
            role="m0-production-rollout",
            server_args=dict(server_args),
            wait=True,
        )
        controller.set_version(exports.parent.policy_engine_version)
        return PinnedArealSGLangActivationWorker.create(
            controller=controller,
            serving_exports=exports,
            harness_checkpoints=harness_checkpoints,
            observation_root=observation_root,
            parent_release_id=parent_release_id,
            candidate_release_id=candidate_release_id,
            request_timeout_seconds=request_timeout_seconds,
        )
    except BaseException:
        controller.destroy()
        raise


__all__ = [
    "PINNED_AREAL_COMMIT",
    "ArealProductionWorkerError",
    "ArealServingExportAudit",
    "HarnessServingCheckpoint",
    "LiveArealServingExportPair",
    "PinnedArealSGLangActivationWorker",
    "build_production_probe_output",
    "launch_pinned_areal_sglang_activation_worker",
    "materialize_areal_serving_export_pair",
    "require_live_areal_serving_export_pair",
]
