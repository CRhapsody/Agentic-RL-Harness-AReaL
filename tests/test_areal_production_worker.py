from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from jphrl.training.areal_production_worker import (
    ArealProductionWorkerError,
    HarnessServingCheckpoint,
    LiveArealServingExportPair,
    PinnedArealSGLangActivationWorker,
    _freeze_data_parallel_routes,
    _load_safetensor_export,
    build_production_probe_output,
    materialize_areal_serving_export_pair,
    require_live_areal_serving_export_pair,
)
from jphrl.training.joint_activation import ProductionReleaseTarget
from jphrl.trajectory.schema import JointVersion


def _version() -> JointVersion:
    return JointVersion(
        policy="policy-parent",
        harness_controller="harness-parent",
        harness_artifact="harness-artifact-v1",
        tool_schema="tools-v1",
        parser="parser-v1",
        environment="environment-v1",
        evaluator="evaluator-v1",
        tokenizer="tokenizer-v1",
        context_builder="context-v1",
    )


class _FakeWorkflowExecutor:
    def __init__(self) -> None:
        self.paused = False

    def is_paused(self) -> bool:
        return self.paused


class _FakeController:
    def __init__(self, worker_count: int) -> None:
        self.inference_worker_urls = [
            f"http://inference-{index}" for index in range(worker_count)
        ]
        self._data_proxy_addrs = [
            f"http://data-proxy-{index}" for index in range(worker_count)
        ]
        self._worker_ids = {
            url: f"router-worker-{index}"
            for index, url in enumerate(self._data_proxy_addrs)
        }
        self.rollout_alloc = SimpleNamespace(
            backend="sglang",
            parallel=SimpleNamespace(
                dp_size=worker_count,
                tp_size=1,
                pp_size=1,
            ),
        )
        self.workflow_executor = _FakeWorkflowExecutor()
        self._version = 7
        self._destroyed = False

    @property
    def worker_ids(self) -> dict[str, str]:
        return dict(self._worker_ids)

    def get_version(self) -> int:
        return self._version

    def set_version(self, version: int) -> None:
        # Deliberately updates only controller-local state.  The Y adapter must
        # still target every DataProxy itself.
        self._version = version

    def pause(self) -> None:
        self.workflow_executor.paused = True

    def resume(self) -> None:
        self.workflow_executor.paused = False


class _FakeServiceMesh:
    def __init__(self, controller: _FakeController) -> None:
        self.versions = {url: 7 for url in controller._data_proxy_addrs}
        self.paused = {url: False for url in controller._data_proxy_addrs}
        self.model_paths = {
            url: "/exports/parent" for url in controller.inference_worker_urls
        }
        self.calls: list[tuple[str, str, object]] = []
        self.parameter_dumps: list[str] = []
        self.fail_update_url: str | None = None
        self.fail_update_model_path: str | None = None
        self.fail_data_proxy_request_once: tuple[str, str] | None = None

    def request(
        self,
        method: str,
        url: str,
        body: object | None = None,
    ) -> dict[str, object]:
        self.calls.append((method, url, body))
        if method == "GET" and url.endswith("/health"):
            proxy = url[: -len("/health")]
            if proxy not in self.versions:
                raise OSError("missing DataProxy")
            return {
                "status": "ok",
                "paused": self.paused[proxy],
                "version": self.versions[proxy],
            }
        if method == "POST" and url.endswith("/set_version"):
            proxy = url[: -len("/set_version")]
            if self.fail_data_proxy_request_once == (proxy, "/set_version"):
                self.fail_data_proxy_request_once = None
                raise ArealProductionWorkerError("injected DataProxy failure")
            version = body["version"]
            self.versions[proxy] = version
            return {"status": "ok", "version": version}
        for endpoint, paused in (
            ("/pause_generation", True),
            ("/continue_generation", False),
        ):
            if method == "POST" and url.endswith(endpoint):
                proxy = url[: -len(endpoint)]
                if self.fail_data_proxy_request_once == (proxy, endpoint):
                    self.fail_data_proxy_request_once = None
                    raise ArealProductionWorkerError("injected DataProxy failure")
                self.paused[proxy] = paused
                return {"status": "ok", "paused": paused}
        if method == "POST" and url.endswith("/update_weights_from_disk"):
            inference = url[: -len("/update_weights_from_disk")]
            if (
                inference == self.fail_update_url
                and body["model_path"] == self.fail_update_model_path
            ):
                raise ArealProductionWorkerError("injected update failure")
            self.model_paths[inference] = body["model_path"]
            return {"success": True}
        raise AssertionError(f"unexpected fake request: {method} {url}")


def _release_targets() -> tuple[dict[str, object], dict[str, ProductionReleaseTarget]]:
    parent_version = _version()
    candidate_version = replace(
        parent_version,
        policy="policy-candidate",
        harness_controller="harness-candidate",
    )
    audits = {
        "parent": SimpleNamespace(
            joint_version=parent_version,
            policy_engine_version=7,
            serving_parameter_sha256="1" * 64,
            serving_export_path="/exports/parent",
            source_dcp_manifest_sha256="a" * 64,
        ),
        "candidate": SimpleNamespace(
            joint_version=candidate_version,
            policy_engine_version=8,
            serving_parameter_sha256="2" * 64,
            serving_export_path="/exports/candidate",
            source_dcp_manifest_sha256="b" * 64,
        ),
    }
    targets = {
        release_id: ProductionReleaseTarget(
            release_id=release_id,
            joint_version=audit.joint_version,
            policy_engine_version=audit.policy_engine_version,
            policy_checkpoint_sha256=audit.serving_parameter_sha256,
            harness_checkpoint_sha256=("3" if release_id == "parent" else "4") * 64,
            harness_parameter_digest=("5" if release_id == "parent" else "6") * 64,
        )
        for release_id, audit in audits.items()
    }
    return audits, targets


def _fake_worker(
    root: Path,
    *,
    worker_count: int = 4,
) -> tuple[
    PinnedArealSGLangActivationWorker,
    _FakeController,
    _FakeServiceMesh,
    dict[str, ProductionReleaseTarget],
]:
    controller = _FakeController(worker_count)
    mesh = _FakeServiceMesh(controller)
    audits, targets = _release_targets()
    worker = object.__new__(PinnedArealSGLangActivationWorker)
    worker._controller = controller
    worker._routes = _freeze_data_parallel_routes(controller)
    worker._worker_id = (
        f"areal-v2-{worker._routes[0].routed_worker_id}"
        if worker_count == 1
        else "areal-v2-dp4-test"
    )
    worker._timeout = 1.0
    worker._root = root
    worker._targets = audits
    worker._active_release_id = "parent"
    worker._policy_release_id = "parent"
    worker._harness_release_id = "parent"
    worker._harness_specs = {
        release_id: HarnessServingCheckpoint(
            path=f"/unused/{release_id}",
            checkpoint_sha256=target.harness_checkpoint_sha256,
            kind="rollout_json",
        )
        for release_id, target in targets.items()
    }
    worker._harness_policy = SimpleNamespace(
        version=targets["parent"].joint_version.harness_controller,
        parameter_digest=targets["parent"].harness_parameter_digest,
    )
    worker._closed = False
    worker._http_json = mesh.request

    def dump(route: object) -> str:
        mesh.parameter_dumps.append(route.routed_worker_id)
        model_path = mesh.model_paths[route.inference_url]
        return {
            "/exports/parent": "1" * 64,
            "/exports/candidate": "2" * 64,
        }[model_path]

    worker._dump_live_parameters = dump
    return worker, controller, mesh, targets


class ArealProductionWorkerTests(unittest.TestCase):
    def test_safetensor_export_uses_context_handle_keys_api(self) -> None:
        first = object()
        second = object()

        class SafeOpenHandle:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def keys(self):
                return ["model.first", "model.second"]

            def get_tensor(self, name):
                return {
                    "model.first": first,
                    "model.second": second,
                }[name]

        safetensors = ModuleType("safetensors")
        safetensors.safe_open = Mock(  # type: ignore[attr-defined]
            return_value=SafeOpenHandle()
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors").write_bytes(b"fixture")
            with patch.dict(sys.modules, {"safetensors": safetensors}):
                tensors = _load_safetensor_export(root)

        self.assertIs(tensors["model.first"], first)
        self.assertIs(tensors["model.second"], second)
        safetensors.safe_open.assert_called_once()  # type: ignore[attr-defined]

    def test_probe_output_keeps_dcp_and_live_serving_identity_distinct(self) -> None:
        parent = _version()
        candidate = replace(
            parent,
            policy="policy-candidate",
            harness_controller="harness-candidate",
        )
        arguments = {
            "fixture": b'{"prompt":"held-out"}',
            "target_release_id": "release-candidate",
            "target_joint_version": candidate,
            "policy_engine_version": 8,
            "policy_checkpoint_sha256": "1" * 64,
            "serving_parameter_sha256": "2" * 64,
            "harness_checkpoint_sha256": "3" * 64,
            "harness_parameter_sha256": "4" * 64,
        }
        first = build_production_probe_output(**arguments)
        second = build_production_probe_output(**arguments)
        record = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(record["policy_checkpoint_sha256"], "1" * 64)
        self.assertEqual(record["serving_parameter_sha256"], "2" * 64)
        self.assertNotEqual(
            record["policy_checkpoint_sha256"],
            record["serving_parameter_sha256"],
        )
        self.assertEqual(
            record["fixture_sha256"], hashlib.sha256(arguments["fixture"]).hexdigest()
        )

    def test_probe_output_rejects_empty_fixture_and_bad_digest(self) -> None:
        with self.assertRaisesRegex(ArealProductionWorkerError, "fixture"):
            build_production_probe_output(
                fixture=b"",
                target_release_id="release",
                target_joint_version=_version(),
                policy_engine_version=1,
                policy_checkpoint_sha256="1" * 64,
                serving_parameter_sha256="2" * 64,
                harness_checkpoint_sha256="3" * 64,
                harness_parameter_sha256="4" * 64,
            )
        with self.assertRaisesRegex(ArealProductionWorkerError, "digest"):
            build_production_probe_output(
                fixture=b"fixture",
                target_release_id="release",
                target_joint_version=_version(),
                policy_engine_version=1,
                policy_checkpoint_sha256="not-a-digest",
                serving_parameter_sha256="2" * 64,
                harness_checkpoint_sha256="3" * 64,
                harness_parameter_sha256="4" * 64,
            )

    def test_persisted_or_user_constructed_lineage_cannot_mint_live_pair(self) -> None:
        forged = LiveArealServingExportPair()
        with self.assertRaisesRegex(ArealProductionWorkerError, "native live"):
            require_live_areal_serving_export_pair(forged)
        with self.assertRaisesRegex(ArealProductionWorkerError, "real pinned"):
            materialize_areal_serving_export_pair(
                actor=object(),
                policy_candidate_record={},
                export_root="/tmp/unused-serving-export",
                parent_joint_version=_version(),
                candidate_joint_version=replace(
                    _version(), policy="candidate", harness_controller="candidate"
                ),
            )

    def test_worker_direct_constructor_is_blocked_for_cleanup_ownership(self) -> None:
        with self.assertRaisesRegex(ArealProductionWorkerError, r"\.create\(\)"):
            PinnedArealSGLangActivationWorker(
                controller=object(),
                serving_exports=LiveArealServingExportPair(),
                harness_checkpoints={},
                observation_root="/tmp/unused-worker-observations",
                parent_release_id="parent",
                candidate_release_id="candidate",
            )

    def test_harness_checkpoint_spec_is_exact(self) -> None:
        HarnessServingCheckpoint(
            path="/external/harness.json",
            checkpoint_sha256="a" * 64,
            kind="rollout_json",
        ).validate()
        with self.assertRaisesRegex(ArealProductionWorkerError, "kind"):
            HarnessServingCheckpoint(
                path="/external/harness.bin",
                checkpoint_sha256="a" * 64,
                kind="unknown",
            ).validate()

    def test_four_data_proxy_roster_is_frozen_and_every_replica_is_probed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker, controller, mesh, _targets = _fake_worker(Path(directory))
            output = worker.run_probe(b"held-out")

            self.assertEqual(
                worker.data_parallel_worker_ids,
                tuple(f"router-worker-{index}" for index in range(4)),
            )
            self.assertEqual(json.loads(output)["release_id"], "parent")
            self.assertEqual(
                mesh.parameter_dumps,
                [f"router-worker-{index}" for index in range(4)],
            )

            controller._worker_ids.pop(controller._data_proxy_addrs[-1])
            with self.assertRaisesRegex(
                ArealProductionWorkerError,
                "coverage|roster",
            ):
                worker.read_state()

    def test_four_replica_install_syncs_each_inference_and_data_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker, controller, mesh, targets = _fake_worker(Path(directory))
            worker.quiesce()
            worker.install_policy(targets["candidate"])

            update_urls = [
                url
                for method, url, _body in mesh.calls
                if method == "POST" and url.endswith("/update_weights_from_disk")
            ]
            version_urls = [
                url
                for method, url, _body in mesh.calls
                if method == "POST" and url.endswith("/set_version")
            ]
            self.assertEqual(
                update_urls,
                [
                    f"http://inference-{index}/update_weights_from_disk"
                    for index in range(4)
                ],
            )
            self.assertEqual(
                version_urls,
                [f"http://data-proxy-{index}/set_version" for index in range(4)],
            )
            self.assertEqual(set(mesh.model_paths.values()), {"/exports/candidate"})
            self.assertEqual(set(mesh.versions.values()), {8})
            self.assertEqual(controller.get_version(), 8)
            self.assertEqual(worker._policy_release_id, "candidate")

    def test_partial_install_attempts_every_replica_then_parent_rollback_repairs_all(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker, _controller, mesh, targets = _fake_worker(Path(directory))
            worker.quiesce()
            mesh.fail_update_url = "http://inference-1"
            mesh.fail_update_model_path = "/exports/candidate"
            with self.assertRaisesRegex(ArealProductionWorkerError, "prior release"):
                worker.install_policy(targets["candidate"])

            candidate_attempts = [
                url
                for method, url, body in mesh.calls
                if method == "POST"
                and url.endswith("/update_weights_from_disk")
                and body["model_path"] == "/exports/candidate"
            ]
            self.assertEqual(len(candidate_attempts), 4)
            self.assertEqual(worker._policy_release_id, "parent")
            parent_attempts = [
                url
                for method, url, body in mesh.calls
                if method == "POST"
                and url.endswith("/update_weights_from_disk")
                and body["model_path"] == "/exports/parent"
            ]
            self.assertEqual(len(parent_attempts), 4)
            self.assertEqual(set(mesh.model_paths.values()), {"/exports/parent"})
            self.assertEqual(set(mesh.versions.values()), {7})

    def test_partial_version_sync_rolls_weights_and_versions_back_on_every_replica(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker, controller, mesh, targets = _fake_worker(Path(directory))
            worker.quiesce()
            mesh.fail_data_proxy_request_once = (
                "http://data-proxy-2",
                "/set_version",
            )
            with self.assertRaisesRegex(ArealProductionWorkerError, "prior release"):
                worker.install_policy(targets["candidate"])

            candidate_version_attempts = [
                url
                for method, url, body in mesh.calls
                if method == "POST"
                and url.endswith("/set_version")
                and body["version"] == 8
            ]
            parent_version_attempts = [
                url
                for method, url, body in mesh.calls
                if method == "POST"
                and url.endswith("/set_version")
                and body["version"] == 7
            ]
            self.assertEqual(len(candidate_version_attempts), 4)
            self.assertEqual(len(parent_version_attempts), 4)
            self.assertEqual(set(mesh.model_paths.values()), {"/exports/parent"})
            self.assertEqual(set(mesh.versions.values()), {7})
            self.assertEqual(controller.get_version(), 7)
            self.assertEqual(worker._policy_release_id, "parent")

    def test_pause_resume_and_version_divergence_are_checked_per_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker, controller, mesh, _targets = _fake_worker(Path(directory))
            worker.quiesce()
            self.assertTrue(controller.workflow_executor.is_paused())
            self.assertEqual(set(mesh.paused.values()), {True})
            pause_urls = [
                url
                for method, url, _body in mesh.calls
                if method == "POST" and url.endswith("/pause_generation")
            ]
            self.assertEqual(len(pause_urls), 4)

            worker.resume()
            self.assertFalse(controller.workflow_executor.is_paused())
            self.assertEqual(set(mesh.paused.values()), {False})
            continue_urls = [
                url
                for method, url, _body in mesh.calls
                if method == "POST" and url.endswith("/continue_generation")
            ]
            self.assertEqual(len(continue_urls), 4)

            mesh.versions["http://data-proxy-3"] = 6
            with self.assertRaisesRegex(ArealProductionWorkerError, "version differs"):
                worker._observe_version(7)

    def test_partial_pause_and_resume_are_retryable_without_destroying_rollout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker, controller, mesh, _targets = _fake_worker(Path(directory))
            mesh.fail_data_proxy_request_once = (
                "http://data-proxy-1",
                "/pause_generation",
            )
            with self.assertRaisesRegex(ArealProductionWorkerError, "1 of 4"):
                worker.quiesce()
            self.assertTrue(controller.workflow_executor.is_paused())
            self.assertFalse(mesh.paused["http://data-proxy-1"])

            worker.quiesce()
            self.assertEqual(set(mesh.paused.values()), {True})
            mesh.fail_data_proxy_request_once = (
                "http://data-proxy-2",
                "/continue_generation",
            )
            with self.assertRaisesRegex(ArealProductionWorkerError, "1 of 4"):
                worker.resume()
            self.assertTrue(controller.workflow_executor.is_paused())
            self.assertTrue(mesh.paused["http://data-proxy-2"])

            worker.resume()
            self.assertEqual(set(mesh.paused.values()), {False})
            self.assertFalse(controller.workflow_executor.is_paused())
            self.assertFalse(worker._closed)
            self.assertFalse(controller._destroyed)

    def test_single_replica_keeps_legacy_worker_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker, controller, mesh, targets = _fake_worker(
                Path(directory),
                worker_count=1,
            )
            worker.quiesce()
            worker.install_policy(targets["candidate"])
            worker._active_release_id = "candidate"
            worker._harness_release_id = "candidate"
            worker._harness_policy = SimpleNamespace(
                version=targets["candidate"].joint_version.harness_controller,
                parameter_digest=targets["candidate"].harness_parameter_digest,
            )
            output = worker.run_probe(b"single-worker")
            worker.resume()

            self.assertEqual(worker.worker_id, "areal-v2-router-worker-0")
            self.assertEqual(worker.data_parallel_worker_ids, ("router-worker-0",))
            self.assertEqual(json.loads(output)["release_id"], "candidate")
            self.assertEqual(set(mesh.model_paths.values()), {"/exports/candidate"})
            self.assertEqual(set(mesh.versions.values()), {8})
            self.assertFalse(controller.workflow_executor.is_paused())


if __name__ == "__main__":
    unittest.main()
