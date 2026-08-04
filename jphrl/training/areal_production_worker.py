from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
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


def _tensor_bytes(value: object) -> memoryview:
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
    # ``bytes(UntypedStorage)`` iterates storage one byte at a time and takes
    # minutes for a multi-GiB export.  A contiguous uint8 NumPy view exposes
    # the identical bytes through the buffer protocol, so hashlib consumes
    # them directly without a Python-level copy or changing the canonical SHA.
    return memoryview(tensor.view(torch.uint8).numpy()).cast("B")


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
            raise ArealProductionWorkerError(
                "serving parameter is not a tensor"
            ) from exc
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
        _require(
            isinstance(weight_map, Mapping), "serving export weight map is missing"
        )
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
            # ``safe_open`` exposes parameter names through ``keys()``.  Its
            # context handle is not an iterable in current safetensors
            # releases, even though older test doubles often were.
            for name in handle.keys():
                _require(name not in tensors, "serving export parameter is duplicated")
                tensors[name] = handle.get_tensor(name)
    if index.is_file():
        _require(
            set(tensors) == set(weight_map), "serving export index differs from weights"
        )
    return tensors


def _safetensor_export_parameter_names(root: Path) -> frozenset[str]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise ArealProductionWorkerError(
            "safetensors is required to verify live AReaL parameters"
        ) from exc
    index = root / "model.safetensors.index.json"
    expected_names: set[str] | None = None
    if index.is_file():
        try:
            record = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArealProductionWorkerError(
                "serving export index is invalid"
            ) from exc
        weight_map = record.get("weight_map")
        _require(
            isinstance(weight_map, Mapping)
            and all(
                isinstance(name, str)
                and bool(name)
                and isinstance(shard, str)
                and bool(shard)
                for name, shard in weight_map.items()
            ),
            "serving export weight map is invalid",
        )
        expected_names = set(weight_map)
        paths = [root / name for name in sorted(set(weight_map.values()))]
    else:
        paths = sorted(root.glob("*.safetensors"))
    _require(bool(paths), "AReaL HF export has no safetensors weights")
    names: set[str] = set()
    for path in paths:
        _require(
            path.is_file() and not path.is_symlink() and path.parent == root,
            "serving export weight shard is unsafe",
        )
        with safe_open(path, framework="pt", device="cpu") as handle:
            shard_names = set(handle.keys())
        _require(
            bool(shard_names)
            and all(isinstance(name, str) and bool(name) for name in shard_names)
            and names.isdisjoint(shard_names),
            "serving export parameter names are invalid or duplicated",
        )
        names.update(shard_names)
    if expected_names is not None:
        _require(names == expected_names, "serving export index differs from weights")
    return frozenset(names)


def _runtime_parameter_contract(root: Path) -> tuple[frozenset[str], bool]:
    config_path = root / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArealProductionWorkerError(
            "serving export model config is invalid"
        ) from exc
    _require(isinstance(config, Mapping), "serving export model config is invalid")
    tied = config.get("tie_word_embeddings")
    _require(
        type(tied) is bool,
        "serving export tie_word_embeddings contract is missing",
    )
    return _safetensor_export_parameter_names(root), tied


def _runtime_parameter_digest(
    tensors: Mapping[str, object],
    *,
    export_parameter_names: frozenset[str],
    tie_word_embeddings: bool,
) -> str:
    _require(
        type(export_parameter_names) is frozenset
        and bool(export_parameter_names)
        and all(isinstance(name, str) and bool(name) for name in export_parameter_names)
        and type(tie_word_embeddings) is bool,
        "live parameter digest contract is invalid",
    )
    normalized = dict(tensors)
    _require(
        set(normalized).issubset(export_parameter_names),
        "live SGLang parameter names contain values absent from the serving export",
    )
    missing = export_parameter_names.difference(normalized)
    output_name = "lm_head.weight"
    embedding_name = "model.embed_tokens.weight"
    if missing == {output_name}:
        _require(
            tie_word_embeddings and embedding_name in normalized,
            "live SGLang parameters omit an unapproved output weight",
        )
        # PyTorch named_parameters() removes duplicate Parameter objects.  A
        # tied Qwen output head is therefore absent from AReaL's SGLang debug
        # dump even when the HF export serializes both aliases.  Reintroduce
        # only that declared alias from the observed live embedding tensor;
        # the full export digest below still proves the two serialized values
        # are byte-identical.
        normalized[output_name] = normalized[embedding_name]
    _require(
        set(normalized) == export_parameter_names,
        "live SGLang parameter names differ from the serving export",
    )
    return _parameter_digest(normalized)


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
        raise ArealProductionWorkerError(
            "pinned AReaL SaveLoadMeta is unavailable"
        ) from exc
    actor.load(
        meta=SaveLoadMeta(path=str(source), weight_format="dcp", with_optim=True)
    )
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
    live_policy_candidate: object | None = None,
) -> LiveArealServingExportPair:
    """Load each sealed DCP into the live actor and emit serving-compatible HF."""

    module = type(actor).__module__
    if (
        module == "jphrl.training.areal_distributed_policy"
        and type(actor).__name__ == "JPHPPOActorController"
    ):
        from .areal_distributed_policy import (
            require_live_remote_policy_candidate,
            validate_distributed_policy_candidate,
        )

        live = require_live_remote_policy_candidate(live_policy_candidate)
        _require(
            live.receipt == policy_candidate_record,
            "distributed serving export live candidate differs from T",
        )
        policy = validate_distributed_policy_candidate(
            policy_candidate_record,
            active_joint_version=parent_joint_version,
            require_checkpoints=True,
        )
        _require(
            candidate_joint_version.policy == policy.candidate_policy_version,
            "distributed serving export candidate JointVersion differs from T",
        )
        root = require_outside_repository(export_root)
        aggregate = actor.materialize_m0_serving_export_pair(
            live,
            export_root=root,
        )

        def _collective_audit(
            kind: str,
            joint_version: JointVersion,
            policy_engine_version: int,
        ) -> ArealServingExportAudit:
            value = aggregate[kind]
            _require(isinstance(value, Mapping), "distributed export branch is missing")
            export_path = require_within_configured_root(value["export_path"])
            actual_manifest = _directory_manifest(export_path)
            actual_parameters = _parameter_digest(
                _load_safetensor_export(export_path)
            )
            _require(
                actual_manifest == value["export_manifest"]
                and actual_parameters == value["parameter_sha256"],
                f"distributed {kind} export differs from rank receipts",
            )
            record_path = root / f"policy-{kind}-hf-lineage.json"
            record: dict[str, object] = {
                "schema_version": SERVING_EXPORT_SCHEMA,
                "areal_commit": PINNED_AREAL_COMMIT,
                "joint_version": asdict(joint_version),
                "joint_version_id": joint_version.version_id,
                "policy_engine_version": policy_engine_version,
                "policy_candidate_record_sha256": policy.record_sha256,
                "source_dcp_path": value["dcp_path"],
                "source_dcp_manifest_sha256": value[
                    "dcp_manifest_sha256"
                ],
                "serving_export_path": str(export_path),
                "serving_export_manifest": actual_manifest,
                "serving_parameter_sha256": actual_parameters,
                "distributed_export_receipt_sha256": aggregate["record_sha256"],
                "exporter": (
                    "jphrl.training.areal_distributed_policy."
                    "JPHFSDPPPOActor.SaveLoadMeta(hf)-collective-d4"
                ),
            }
            _assert_no_secrets(record)
            record["record_sha256"] = _record_sha256(record)
            _write_new_json(record_path, record)
            return ArealServingExportAudit(
                joint_version=joint_version,
                policy_engine_version=policy_engine_version,
                policy_candidate_record_sha256=policy.record_sha256,
                source_dcp_path=str(value["dcp_path"]),
                source_dcp_manifest_sha256=str(
                    value["dcp_manifest_sha256"]
                ),
                serving_export_path=str(export_path),
                serving_export_manifest_sha256=str(
                    actual_manifest["manifest_sha256"]
                ),
                serving_parameter_sha256=actual_parameters,
                record_path=str(record_path),
                record_sha256=str(record["record_sha256"]),
            )

        parent = _collective_audit(
            "parent",
            parent_joint_version,
            policy.parent_engine_version,
        )
        candidate = _collective_audit(
            "candidate",
            candidate_joint_version,
            policy.reserved_candidate_engine_version,
        )
        return LiveArealServingExportPair._create(
            parent=parent,
            candidate=candidate,
            token=_LIVE_EXPORT_TOKEN,
        )
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
            expected_dcp_sha256=str(checkpoints["parent_manifest"]["manifest_sha256"]),
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
        _require(
            self.kind in {"rollout_json", "candidate_pt"},
            "Harness checkpoint kind is invalid",
        )
        _require(
            _is_sha256(self.checkpoint_sha256), "Harness checkpoint hash is invalid"
        )


@dataclass(frozen=True)
class _ArealDataParallelRoute:
    """Frozen one-to-one AReaL DP route; never synthesized from another worker."""

    index: int
    inference_url: str
    data_proxy_url: str
    routed_worker_id: str


def _freeze_data_parallel_routes(
    controller: object,
) -> tuple[_ArealDataParallelRoute, ...]:
    """Bind fixed-AReaL's group-ordered inference/DataProxy route vectors."""

    inference_urls = tuple(
        str(url).rstrip("/") for url in controller.inference_worker_urls
    )
    worker_ids = controller.worker_ids
    _require(isinstance(worker_ids, Mapping), "AReaL worker ID registry is invalid")
    data_proxy_urls = tuple(str(url).rstrip("/") for url in worker_ids)
    registered_worker_ids = tuple(str(worker_ids[url]) for url in worker_ids)
    controller_data_proxy_urls = tuple(
        str(url).rstrip("/") for url in getattr(controller, "_data_proxy_addrs", ())
    )
    allocation = getattr(controller, "rollout_alloc", None)
    parallel = getattr(allocation, "parallel", None)
    expected_dp = getattr(parallel, "dp_size", None)
    _require(
        getattr(allocation, "backend", None) == "sglang"
        and type(expected_dp) is int
        and expected_dp in {1, 4}
        and getattr(parallel, "tp_size", None) == 1
        and getattr(parallel, "pp_size", None) == 1,
        "production rollout must use exactly sglang:d1 or sglang:d4",
    )
    _require(
        len(inference_urls)
        == len(data_proxy_urls)
        == len(registered_worker_ids)
        == expected_dp
        and data_proxy_urls == controller_data_proxy_urls,
        "AReaL inference/DataProxy worker coverage or route order differs",
    )
    _require(
        all(inference_urls)
        and all(data_proxy_urls)
        and all(registered_worker_ids)
        and len(set(inference_urls)) == expected_dp
        and len(set(data_proxy_urls)) == expected_dp
        and len(set(registered_worker_ids)) == expected_dp,
        "AReaL inference/DataProxy worker identities are missing or duplicated",
    )
    return tuple(
        _ArealDataParallelRoute(
            index=index,
            inference_url=inference_urls[index],
            data_proxy_url=data_proxy_urls[index],
            routed_worker_id=registered_worker_ids[index],
        )
        for index in range(expected_dp)
    )


class PinnedArealSGLangActivationWorker(ProductionActivationWorker):
    """M0 adapter with exact observations from every SGLang DP replica."""

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
        _require(
            set(harness_checkpoints) == set(self._targets),
            "Harness checkpoint coverage differs",
        )
        self._harness_specs = dict(harness_checkpoints)
        for spec in self._harness_specs.values():
            _require(
                type(spec) is HarnessServingCheckpoint,
                "Harness checkpoint spec is untyped",
            )
            spec.validate()
        self._routes = _freeze_data_parallel_routes(controller)
        if len(self._routes) == 1:
            self._worker_id = f"areal-v2-{self._routes[0].routed_worker_id}"
        else:
            roster = [
                {
                    "index": route.index,
                    "inference_url": route.inference_url,
                    "data_proxy_url": route.data_proxy_url,
                    "routed_worker_id": route.routed_worker_id,
                }
                for route in self._routes
            ]
            self._worker_id = f"areal-v2-dp{len(self._routes)}-{_sha256(roster)[:16]}"
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

    @property
    def data_parallel_worker_ids(self) -> tuple[str, ...]:
        """Router identities frozen at construction, in AReaL DP-group order."""

        return tuple(route.routed_worker_id for route in self._routes)

    def _require_frozen_routes(self) -> tuple[_ArealDataParallelRoute, ...]:
        _require(
            not bool(getattr(self._controller, "_destroyed", False))
            and _freeze_data_parallel_routes(self._controller) == self._routes,
            "AReaL inference/DataProxy worker roster changed while Y was live",
        )
        return self._routes

    def _http_json(
        self, method: str, url: str, body: Mapping[str, object] | None = None
    ) -> Mapping[str, object]:
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
            raise ArealProductionWorkerError(
                "AReaL service response is invalid"
            ) from exc
        _require(isinstance(value, Mapping), "AReaL service response is not an object")
        return value

    def _health(self, route: _ArealDataParallelRoute) -> Mapping[str, object]:
        health = self._http_json("GET", f"{route.data_proxy_url}/health")
        _require(
            health.get("status") == "ok",
            f"AReaL data proxy {route.routed_worker_id} is unhealthy",
        )
        return health

    def _all_health(self) -> tuple[Mapping[str, object], ...]:
        routes = self._require_frozen_routes()
        health = tuple(self._health(route) for route in routes)
        _require(
            len(health) == len(routes),
            "AReaL DataProxy health coverage differs from the frozen roster",
        )
        return health

    def _pause_state_snapshot(self) -> tuple[bool, tuple[bool, ...]]:
        health = self._all_health()
        executor_paused = bool(self._controller.workflow_executor.is_paused())
        paused = tuple(item.get("paused") for item in health)
        _require(
            all(type(value) is bool for value in paused),
            "DataProxy pause state is invalid",
        )
        return executor_paused, paused

    def _is_quiesced(self) -> bool:
        executor_paused, paused = self._pause_state_snapshot()
        _require(
            all(value is executor_paused for value in paused),
            "dispatcher/backend pause state differs across DataProxy workers",
        )
        return executor_paused

    def _observe_version(self, expected: int) -> None:
        health = self._all_health()
        versions = tuple(item.get("version") for item in health)
        _require(
            type(expected) is int
            and type(self._controller.get_version()) is int
            and self._controller.get_version() == expected
            and all(
                type(version) is int and version == expected for version in versions
            ),
            "live AReaL Policy version differs across DataProxy workers",
        )

    def _dump_live_parameters(
        self,
        route: _ArealDataParallelRoute,
        *,
        export_parameter_names: frozenset[str],
        tie_word_embeddings: bool,
    ) -> str:
        path = self._root / (f"live-parameters-dp-{route.index:05d}-{uuid4().hex}.pt")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        inode = path.stat().st_ino
        response = self._http_json(
            "POST",
            f"{route.inference_url}/awex/debug/get_parameters",
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
        return _runtime_parameter_digest(
            tensors,
            export_parameter_names=export_parameter_names,
            tie_word_embeddings=tie_word_embeddings,
        )

    def _live_parameter_contract(
        self, expected: ArealServingExportAudit
    ) -> tuple[frozenset[str], bool]:
        return _runtime_parameter_contract(Path(expected.serving_export_path))

    def _verify_live_policy(self, release_id: str) -> None:
        expected = self._targets[release_id]
        routes = self._require_frozen_routes()
        export_parameter_names, tie_word_embeddings = (
            self._live_parameter_contract(expected)
        )
        digests = tuple(
            self._dump_live_parameters(
                route,
                export_parameter_names=export_parameter_names,
                tie_word_embeddings=tie_word_embeddings,
            )
            for route in routes
        )
        _require(
            len(digests) == len(routes) and len(set(digests)) == 1,
            "live SGLang parameters differ across workers",
        )
        _require(
            digests[0] == expected.serving_parameter_sha256,
            "live SGLang parameters differ from the AReaL serving export",
        )

    def _post_every_data_proxy(
        self,
        endpoint: str,
        body: Mapping[str, object],
        *,
        response_validator: Callable[[Mapping[str, object]], bool],
    ) -> None:
        routes = self._require_frozen_routes()
        errors: list[BaseException] = []
        for route in routes:
            try:
                response = self._http_json(
                    "POST",
                    f"{route.data_proxy_url}{endpoint}",
                    body,
                )
                _require(
                    response_validator(response),
                    f"AReaL DataProxy {route.routed_worker_id} rejected {endpoint}",
                )
            except ArealProductionWorkerError as exc:
                errors.append(exc)
        if errors:
            raise ArealProductionWorkerError(
                f"AReaL DataProxy operation {endpoint} failed on "
                f"{len(errors)} of {len(routes)} workers"
            ) from errors[0]

    def _set_all_versions(self, version: int) -> None:
        controller_error: BaseException | None = None
        try:
            self._controller.set_version(version)
        except RuntimeError as exc:
            # The fixed controller raises only when its own broadcast fails on every
            # worker.  Direct per-worker repair below is still required either way.
            controller_error = exc
        try:
            self._post_every_data_proxy(
                "/set_version",
                {"version": version},
                response_validator=lambda response: (
                    response.get("status") == "ok"
                    and response.get("version") == version
                ),
            )
            self._observe_version(version)
        except ArealProductionWorkerError:
            if controller_error is not None:
                raise ArealProductionWorkerError(
                    "AReaL controller and direct per-worker version sync both failed"
                ) from controller_error
            raise

    def _install_serving_export(self, release_id: str) -> None:
        expected = self._targets[release_id]
        routes = self._require_frozen_routes()
        errors: list[BaseException] = []
        for route in routes:
            try:
                response = self._http_json(
                    "POST",
                    f"{route.inference_url}/update_weights_from_disk",
                    {
                        "model_path": expected.serving_export_path,
                        "abort_all_requests": True,
                    },
                )
                _require(
                    response.get("success") is True or response.get("status") == "ok",
                    f"SGLang worker {route.routed_worker_id} rejected the serving export",
                )
            except ArealProductionWorkerError as exc:
                errors.append(exc)
        if errors:
            raise ArealProductionWorkerError(
                "SGLang serving export install failed on "
                f"{len(errors)} of {len(routes)} workers"
            ) from errors[0]

    def _verify_installed_policy(self, release_id: str) -> None:
        expected = self._targets[release_id]
        self._set_all_versions(expected.policy_engine_version)
        self._verify_live_policy(release_id)
        self._observe_version(expected.policy_engine_version)

    def _load_harness(self, release_id: str) -> None:
        spec = self._harness_specs[release_id]
        path = require_within_configured_root(spec.path)
        _require(
            path.is_file() and not path.is_symlink(), "Harness checkpoint is missing"
        )
        _require(
            _file_sha256(path) == spec.checkpoint_sha256,
            "Harness checkpoint hash differs",
        )
        try:
            from jphrl.harness.torch_learning import (
                load_torch_harness_checkpoint,
                load_torch_harness_rollout_checkpoint,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ArealProductionWorkerError(
                "Torch Harness loader is unavailable"
            ) from exc
        if spec.kind == "candidate_pt":
            policy, _optimizer, _record = load_torch_harness_checkpoint(
                path, map_location="cpu"
            )
        else:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArealProductionWorkerError(
                    "Harness rollout checkpoint is invalid"
                ) from exc
            _require(isinstance(raw, Mapping), "Harness rollout checkpoint is invalid")
            policy = load_torch_harness_rollout_checkpoint(raw)
        self._harness_policy = policy
        self._harness_release_id = release_id

    def quiesce(self) -> None:
        _require(not self._closed, "production worker is closed")
        executor_paused, paused = self._pause_state_snapshot()
        if executor_paused and all(paused):
            return
        self._controller.pause()
        self._post_every_data_proxy(
            "/pause_generation",
            {},
            response_validator=lambda response: (
                response.get("status") == "ok" and response.get("paused") is True
            ),
        )
        _require(self._is_quiesced(), "AReaL rollout worker did not quiesce")

    def install_policy(self, target: ProductionReleaseTarget) -> None:
        _require(self._is_quiesced(), "Policy install requires a quiesced worker")
        expected = self._targets.get(target.release_id)
        _require(
            expected is not None, "Policy target is outside the serving export pair"
        )
        _require(
            target.policy_checkpoint_sha256 == expected.serving_parameter_sha256
            and target.policy_engine_version == expected.policy_engine_version,
            "Policy target differs from serving-export lineage",
        )
        previous_release_id = self._policy_release_id
        try:
            self._install_serving_export(target.release_id)
            self._verify_installed_policy(target.release_id)
        except ArealProductionWorkerError as install_error:
            if target.release_id == previous_release_id:
                raise
            try:
                self._install_serving_export(previous_release_id)
                self._verify_installed_policy(previous_release_id)
            except ArealProductionWorkerError as rollback_error:
                raise ArealProductionWorkerError(
                    "multi-worker Policy install failed and immediate rollback was incomplete"
                ) from rollback_error
            raise ArealProductionWorkerError(
                "multi-worker Policy install failed; the prior release was restored"
            ) from install_error
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
        executor_paused, paused = self._pause_state_snapshot()
        if not executor_paused and not any(paused):
            return
        self._post_every_data_proxy(
            "/continue_generation",
            {},
            response_validator=lambda response: (
                response.get("status") == "ok" and response.get("paused") is False
            ),
        )
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
            raise ArealProductionWorkerError(
                "AReaL controller destruction failed"
            ) from exc
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
    """Launch the pinned one- or four-replica controller and transfer ownership.

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
    try:
        from areal.api.alloc_mode import ModelAllocation
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise ArealProductionWorkerError(
            "pinned AReaL ModelAllocation is unavailable"
        ) from exc
    try:
        allocation = ModelAllocation.from_str(
            str(getattr(controller_config, "backend", ""))
        )
    except (TypeError, ValueError) as exc:
        raise ArealProductionWorkerError(
            "production rollout allocation is invalid"
        ) from exc
    parallel = allocation.parallel
    _require(
        allocation.backend == "sglang"
        and parallel.dp_size in {1, 4}
        and parallel.tp_size == 1
        and parallel.pp_size == 1,
        "production rollout must use exactly sglang:d1 or sglang:d4",
    )
    fraction = server_args.get("mem_fraction_static")
    _require(
        isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and float(fraction) == float(recorded_mem_fraction_static)
        and 0.0 < float(fraction) <= 0.95,
        "production SGLang memory fraction is invalid or differs from the record",
    )
    parent_path = exports.parent.serving_export_path
    _require(
        server_args.get("model_path") == parent_path
        and getattr(controller_config, "model", None) == parent_path
        and getattr(controller_config, "tokenizer_path", None) == parent_path
        and allocation.parallel.dp_size in {1, 4},
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
