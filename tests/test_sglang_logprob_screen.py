from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from jphrl.experiments.sglang_logprob_screen import (
    C0_MODE,
    C1_MODE,
    SCREEN_DATASET_SELECTION,
    artifact_tree_sha256,
    compare_screen_runs,
    write_screen_report,
)
from jphrl.experiments.sglang_cuda_graph_screen import (
    CUDA_GRAPH_DATASET_SELECTION,
    compare_cuda_graph_screen_runs,
)
from jphrl.harness.controller import HarnessState
from jphrl.harness.learning import TabularHarnessController
from jphrl.trajectory.areal_joint_bridge import (
    build_areal_joint_bridge_record,
    build_joint_version,
    deterministic_bridge_request_id,
    inference_runtime_contract_sha256,
    inject_harness_instruction,
    prompt_context_chars,
    write_areal_joint_bridge_record,
)
from jphrl.trajectory.areal_interaction_sidecar import (
    InteractionBinding,
    build_interaction_adapter_sidecar,
)
from scripts.write_sglang_logprob_screen_pointer import write_pointer


def _sha256(value: dict[str, object]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
    return hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _trajectory_binding(tensors: dict[str, object]) -> dict[str, object]:
    return {
        key: tensors[key]
        for key in (
            "input_ids",
            "loss_mask",
            "logprobs",
            "versions",
            "attention_mask",
            "rewards",
        )
    }


def _runtime_contract(
    *,
    run_id: str,
    pair_id: str,
    mode: str,
    physical_gpu_id: int,
    dataset_selection: str = SCREEN_DATASET_SELECTION,
    disable_cuda_graph: bool = False,
    runtime_schema: str = "jph.sglang-inference-runtime.v1",
    experimental_axis: str = "generation-logprob-formula-v1",
) -> dict[str, object]:
    treatment: dict[str, object] = {
        "generation_logprob_mode": mode,
        "sglang_return_original_logprob": mode == C1_MODE,
    }
    if runtime_schema == "jph.sglang-inference-runtime.v2":
        treatment = {
            "disable_cuda_graph": disable_cuda_graph,
            "experimental_axis": experimental_axis,
            **treatment,
        }
    return {
        "schema_version": runtime_schema,
        "identity": {"run_id": run_id, "screen_pair_id": pair_id},
        "fixed": {
            "areal_commit": "a" * 40,
            "areal_version": "2.0.0",
            "behavior_revision": "b" * 40,
            "clean_environment_policy": "env-i-v1",
            "cuda_runtime_version": "12.6",
            "cuda_visible_devices": str(physical_gpu_id),
            "dataset_revision": "c" * 40,
            "dataset_selection": dataset_selection,
            "driver_version": "550.54.15",
            "generation": {
                "greedy": False,
                "max_new_tokens": 64,
                "n_samples": 1,
                "temperature": 1.0,
                "top_k": 100000000,
                "top_p": 1.0,
            },
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "gpu_uuid": f"GPU-test-{physical_gpu_id}",
            "physical_gpu_id": physical_gpu_id,
            "python_version": "3.11.0",
            "project_commit": "d" * 40,
            "rollout": {
                "backend": "sglang:d1p1t1",
                "max_concurrent_rollouts": 1,
            },
            "seed": 1,
            "server_args": {
                "attention_backend": "fa3",
                "base_gpu_id": 0,
                "disable_cuda_graph": disable_cuda_graph,
                "disable_radix_cache": True,
                "dtype": "bfloat16",
                "kv_cache_dtype": "auto",
                "model_path": "/allowed/model",
                "random_seed": 1,
                "tokenizer_path": "/allowed/model",
                "tp_size": 1,
            },
            "sglang_environment": {
                "SGLANG_CACHE_DIR": "/allowed/cache/sglang",
            },
            "sglang_version": "0.5.10.post1",
            "torch_version": "2.8.0",
            "transformers_version": "4.57.1",
        },
        "treatment": treatment,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _resign_cell_audit(run_root: Path) -> None:
    path = run_root / "cell-audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["audited_tree_sha256"] = artifact_tree_sha256(run_root)
    audit["record_sha256"] = _sha256(audit)
    _write_json(path, audit)


def _build_cell(
    root: Path,
    *,
    mode: str,
    pair_id: str,
    stored_tail: tuple[float, float],
    physical_gpu_id: int = 0,
    prompt_suffix: str = "",
    output_token_delta: int = 0,
    run_id_override: str | None = None,
    dataset_selection: str = SCREEN_DATASET_SELECTION,
    disable_cuda_graph: bool = False,
    runtime_schema: str = "jph.sglang-inference-runtime.v1",
    experimental_axis: str = "generation-logprob-formula-v1",
) -> Path:
    run_id = run_id_override or f"run-{'c1' if mode == C1_MODE else 'c0'}"
    run_root = root / run_id
    run_root.mkdir(mode=0o700)
    run_root.chmod(0o700)
    score_dir = run_root / "same-backend-scores"
    score_dir.mkdir(mode=0o700)
    score_dir.chmod(0o700)
    runtime_contract = _runtime_contract(
        run_id=run_id,
        pair_id=pair_id,
        mode=mode,
        physical_gpu_id=physical_gpu_id,
        dataset_selection=dataset_selection,
        disable_cuda_graph=disable_cuda_graph,
        runtime_schema=runtime_schema,
        experimental_axis=experimental_axis,
    )
    runtime_hash = inference_runtime_contract_sha256(runtime_contract)
    manifest: dict[str, object] = {
        "schema_version": "jph.sglang-launch-manifest.v1",
        "inference_runtime_contract": runtime_contract,
        "inference_runtime_contract_sha256": runtime_hash,
    }
    manifest["record_sha256"] = _sha256(manifest)
    _write_json(run_root / "launch-manifest.json", manifest)

    for index in range(4):
        base_messages = [
            {
                "role": "user",
                "content": f"What is {index} + {index + 1}?{prompt_suffix}",
            }
        ]
        state = HarnessState(
            turn=0,
            remaining_tool_calls=0,
            remaining_model_retries=0,
            context_chars=prompt_context_chars(base_messages),
            last_error=None,
            retrieval_hit=False,
            verifier_status="not-run",
            task_domain="gsm8k",
        )
        controller = TabularHarnessController(seed=1)
        checkpoint = controller.checkpoint()
        request_id = deterministic_bridge_request_id(
            task_id=index,
            dataset_selection=dataset_selection,
            base_messages=base_messages,
        )
        decision = replace(
            controller.choose(state),
            decision_id=f"{request_id}:harness:0",
        )
        effective_messages, _ = inject_harness_instruction(
            base_messages,
            decision.action,
        )
        base_tokens = [100 + index, 200 + index]
        effective_tokens = [900, *base_tokens]
        output_tokens = [300 + index, 400 + index + output_token_delta]
        input_ids = [[*effective_tokens, *output_tokens]]
        loss_mask = [[0, 0, 0, 1, 1]]
        stored = [[0.0, 0.0, 0.0, *stored_tail]]
        versions = [[-1, -1, -1, 0, 0]]
        tensors: dict[str, object] = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "logprobs": stored,
            "versions": versions,
            "attention_mask": [[True, True, True, True, True]],
            "rewards": [1.0],
        }
        response = SimpleNamespace(
            input_tokens=effective_tokens,
            output_tokens=output_tokens,
            output_logprobs=list(stored_tail),
            output_versions=[0, 0],
            stop_reason="stop",
        )
        interaction = SimpleNamespace(
            interaction_id=request_id,
            reward=1.0,
            chat_template_type=None,
        )
        joint_version = build_joint_version(
            policy_release_id="areal-sglang@model:engine-v0",
            harness_controller_version=controller.version,
            areal_commit="a" * 40,
            behavior_revision="b" * 40,
            dataset_revision="c" * 40,
            dataset_selection=dataset_selection,
            sglang_version="0.5.10.post1",
            generation_logprob_mode=mode,
            inference_runtime_contract_sha256=runtime_hash,
        )
        episode_id = f"{runtime_contract['identity']['run_id']}:{request_id}"
        interaction_sidecar = build_interaction_adapter_sidecar(
            [
                InteractionBinding(
                    episode_id=episode_id,
                    model_call_id=f"{episode_id}:model:0",
                    session_id=None,
                    trajectory_id=None,
                    interaction_id=request_id,
                    parent_interaction_id=None,
                    ordinal=0,
                    joint_version_id=joint_version.version_id,
                    route_kind="rlvr-workflow",
                )
            ]
        )
        bridge = build_areal_joint_bridge_record(
            task_id=index,
            request_id=request_id,
            joint_version=joint_version,
            expected_policy_version=0,
            harness_state=state,
            harness_decision=decision,
            harness_controller_checkpoint=checkpoint,
            base_messages=base_messages,
            effective_messages=effective_messages,
            base_input_tokens=base_tokens,
            effective_input_tokens=effective_tokens,
            model_response=response,
            interaction=interaction,
            tensor_dict=tensors,
            project_commit="d" * 40,
            areal_commit="a" * 40,
            behavior_snapshot_path="/allowed/model",
            behavior_revision="b" * 40,
            dataset_selection=dataset_selection,
            sglang_version="0.5.10.post1",
            generation_logprob_mode=mode,
            inference_runtime_contract=runtime_contract,
            interaction_adapter_sidecar=interaction_sidecar,
        )
        write_areal_joint_bridge_record(
            bridge,
            trace_dir=run_root / "bridge-records",
            allowed_root=root,
        )
        score: dict[str, object] = {
            "schema_version": "jph.areal-same-backend-logprob.v6",
            "request_id": request_id,
            "bridge_record_sha256": bridge["record_sha256"],
            "trajectory_binding_sha256": _sha256(_trajectory_binding(tensors)),
            "scoring_origin": {
                "api": "RolloutController.compute_logp",
                "controller_api_version": "v1",
                "lifecycle": "same-controller-after-wait-before-destroy",
                "score_parser": "jph-tail-before-conversion-v1",
                "score_token_id_validation": "exact-requested-tail-v1",
                "transport_localization": "RTensor.localize-before-score-and-write-v1",
                "backend": "sglang:d1p1t1",
                "engine_version_before_score": 0,
                "engine_version_after_score": 0,
                "policy_release_id": bridge["joint_version"]["policy"],
                "behavior_revision": "b" * 40,
                "areal_commit": "a" * 40,
                "project_commit": "d" * 40,
                "generation_logprob_mode": mode,
                "dataset_selection": dataset_selection,
                "sglang_version": "0.5.10.post1",
                "inference_runtime_contract_sha256": runtime_hash,
            },
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "stored_logprobs": stored,
            "rescored_logprobs": [[0.0, 0.0, 0.0, -1.0, -2.0]],
            "versions": versions,
        }
        score["record_sha256"] = _sha256(score)
        _write_json(
            score_dir / f"same-backend-score-{index}.json",
            score,
        )
    cell_audit: dict[str, object] = {
        "schema_version": "jph.sglang-logprob-screen-cell-audit.v1",
        "run_root": str(run_root.resolve()),
        "audited_tree_sha256": artifact_tree_sha256(run_root),
        "mode_audit_before_report": {
            "mode_violations": [],
            "symlinks": [],
        },
        "sensitive_artifact_audit": {
            "unsafe_fields": [],
            "runtime_secret_matches": 0,
            "default_key_matches": 0,
        },
        "passed": True,
    }
    cell_audit["record_sha256"] = _sha256(cell_audit)
    _write_json(run_root / "cell-audit.json", cell_audit)
    return run_root


def _build_pair(
    root: Path,
    *,
    c1_gpu: int = 0,
    c1_prompt_suffix: str = "",
    c1_output_token_delta: int = 0,
    c1_stored_tail: tuple[float, float] = (-1.005, -2.01),
    runtime_schema: str = "jph.sglang-inference-runtime.v1",
) -> tuple[Path, Path]:
    pair_id = "pair-0123456789abcdef"
    c0 = _build_cell(
        root,
        mode=C0_MODE,
        pair_id=pair_id,
        stored_tail=(-1.03, -2.12),
        runtime_schema=runtime_schema,
    )
    c1 = _build_cell(
        root,
        mode=C1_MODE,
        pair_id=pair_id,
        stored_tail=c1_stored_tail,
        physical_gpu_id=c1_gpu,
        prompt_suffix=c1_prompt_suffix,
        output_token_delta=c1_output_token_delta,
        runtime_schema=runtime_schema,
    )
    return c0, c1


def _build_cuda_graph_pair(
    root: Path,
    *,
    c2b_gpu: int = 0,
    c2b_output_token_delta: int = 0,
    c2b_stored_tail: tuple[float, float] = (-1.005, -2.01),
) -> tuple[Path, Path]:
    pair_id = "cuda-pair-0123456789abcdef"
    common = {
        "mode": C0_MODE,
        "pair_id": pair_id,
        "dataset_selection": CUDA_GRAPH_DATASET_SELECTION,
        "runtime_schema": "jph.sglang-inference-runtime.v2",
        "experimental_axis": "cuda-graph-v1",
    }
    c2a = _build_cell(
        root,
        **common,
        run_id_override="run-c2a",
        stored_tail=(-1.03, -2.12),
        disable_cuda_graph=False,
    )
    c2b = _build_cell(
        root,
        **common,
        run_id_override="run-c2b",
        stored_tail=c2b_stored_tail,
        disable_cuda_graph=True,
        physical_gpu_id=c2b_gpu,
        output_token_delta=c2b_output_token_delta,
    )
    return c2a, c2b


class SGLangLogprobScreenTests(unittest.TestCase):
    def test_real_four_by_four_pair_supports_mechanism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_screen_runs(c0_root, c1_root)
            self.assertTrue(report["summary"]["mechanism_supported"])
            self.assertTrue(
                report["summary"]["paired_rescored_logprobs_within_tolerance"]
            )
            self.assertTrue(
                report["summary"]["at_least_one_active_stored_logprob_changed"]
            )
            self.assertFalse(report["claim_boundary"]["may_unlock_joint_optimizer"])

    def test_runtime_v2_formula_pair_remains_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(
                root,
                runtime_schema="jph.sglang-inference-runtime.v2",
            )
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_screen_runs(c0_root, c1_root)
            self.assertTrue(report["summary"]["mechanism_supported"])
            self.assertTrue(report["summary"]["paired_runtime_fixed_fields_equal"])

    def test_rejects_rescored_logprob_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root)
            score_path = sorted((c1_root / "same-backend-scores").glob("*.json"))[0]
            score = json.loads(score_path.read_text(encoding="utf-8"))
            score["rescored_logprobs"][0][-1] += 1e-4
            score["record_sha256"] = _sha256(score)
            _write_json(score_path, score)
            _resign_cell_audit(c1_root)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_screen_runs(c0_root, c1_root)
            self.assertFalse(
                report["summary"]["paired_rescored_logprobs_within_tolerance"]
            )
            self.assertFalse(report["summary"]["mechanism_supported"])

    def test_rejects_when_treatment_does_not_change_stored_logprobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(
                root,
                c1_stored_tail=(-1.03, -2.12),
            )
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_screen_runs(c0_root, c1_root)
            self.assertFalse(
                report["summary"]["at_least_one_active_stored_logprob_changed"]
            )
            self.assertFalse(report["summary"]["mechanism_supported"])

    def test_rejects_when_c1_is_worse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(
                root,
                c1_stored_tail=(-1.05, -2.15),
            )
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_screen_runs(c0_root, c1_root)
            self.assertFalse(report["summary"]["c1_non_worse_on_every_trace"])
            self.assertFalse(report["summary"]["mechanism_supported"])

    def test_rejects_different_physical_gpu_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root, c1_gpu=1)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_screen_runs(c0_root, c1_root)
            self.assertFalse(report["summary"]["paired_runtime_fixed_fields_equal"])
            self.assertFalse(report["summary"]["mechanism_supported"])

    def test_loader_rejects_missing_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root)
            sorted((c1_root / "same-backend-scores").glob("*.json"))[0].unlink()
            _resign_cell_audit(c1_root)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "expected four score"):
                    compare_screen_runs(c0_root, c1_root)

    def test_loader_rejects_wrong_bridge_binding_after_score_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root)
            score_path = sorted((c1_root / "same-backend-scores").glob("*.json"))[0]
            score = json.loads(score_path.read_text(encoding="utf-8"))
            score["bridge_record_sha256"] = "e" * 64
            score["record_sha256"] = _sha256(score)
            _write_json(score_path, score)
            _resign_cell_audit(c1_root)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "not bound to bridge"):
                    compare_screen_runs(c0_root, c1_root)

    def test_loader_rejects_non_private_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root)
            score_path = sorted((c1_root / "same-backend-scores").glob("*.json"))[0]
            score_path.chmod(0o644)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "not private"):
                    compare_screen_runs(c0_root, c1_root)

    def test_loader_rejects_post_audit_tree_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root)
            late_file = c1_root / "late-write.log"
            late_file.write_text("written after final audit\n", encoding="utf-8")
            late_file.chmod(0o600)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "invalid cell audit"):
                    compare_screen_runs(c0_root, c1_root)

    def test_loader_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root)
            target = sorted((c1_root / "same-backend-scores").glob("*.json"))[0]
            replacement = root / "outside-score.json"
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o600)
            target.unlink()
            target.symlink_to(replacement)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    compare_screen_runs(c0_root, c1_root)

    def test_rejects_changed_prompt_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root, c1_prompt_suffix=" changed")
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "different base prompts"):
                    compare_screen_runs(c0_root, c1_root)

    def test_rejects_changed_output_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0_root, c1_root = _build_pair(root, c1_output_token_delta=1)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "paired score input_ids changed"):
                    compare_screen_runs(c0_root, c1_root)

    def test_writer_is_private_exclusive_and_root_bounded(self) -> None:
        report = {"record_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "screen.json"
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                write_screen_report(report, output)
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(FileExistsError):
                    write_screen_report(report, output)
                with self.assertRaisesRegex(ValueError, "escapes"):
                    write_screen_report(report, root.parent / "outside.json")

    def test_pair_pointer_is_exclusive_and_canonically_root_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_id = "pair-0123456789abcdef"
            pair_root = (
                root
                / "artifacts"
                / "sglang-logprob-screen"
                / "pairs"
                / pair_id
            )
            pair_root.mkdir(parents=True, mode=0o700)
            pair_root.chmod(0o700)
            run_root = root / "artifacts" / "cell"
            run_root.mkdir(parents=True, mode=0o700)
            pointer = pair_root / "c0-run-root.txt"
            write_pointer(
                configured_root=root,
                pair_id=pair_id,
                cell="c0",
                pointer=pointer,
                run_root=run_root,
            )
            self.assertEqual(pointer.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_pointer(
                    configured_root=root,
                    pair_id=pair_id,
                    cell="c0",
                    pointer=pointer,
                    run_root=run_root,
                )
            with self.assertRaisesRegex(ValueError, "escapes"):
                write_pointer(
                    configured_root=root,
                    pair_id=pair_id,
                    cell="c0",
                    pointer=pair_root / ".." / "c0-run-root.txt",
                    run_root=run_root,
                )

    def test_cuda_graph_pair_supports_only_the_registered_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c2a_root, c2b_root = _build_cuda_graph_pair(root)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_cuda_graph_screen_runs(c2a_root, c2b_root)
            self.assertTrue(report["summary"]["mechanism_supported"])
            self.assertTrue(report["summary"]["paired_runtime_invariants_equal"])
            self.assertTrue(report["summary"]["paired_score_alignment_equal"])
            self.assertFalse(report["claim_boundary"]["may_unlock_joint_optimizer"])

    def test_cuda_graph_pair_records_changed_output_as_negative_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c2a_root, c2b_root = _build_cuda_graph_pair(
                root,
                c2b_output_token_delta=1,
            )
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_cuda_graph_screen_runs(c2a_root, c2b_root)
            self.assertFalse(report["summary"]["paired_generation_equal"])
            self.assertFalse(report["summary"]["paired_score_alignment_equal"])
            self.assertFalse(report["summary"]["mechanism_supported"])
            self.assertIsNone(
                report["traces"][0]["paired_score"][
                    "max_rescored_logprob_abs_delta"
                ]
            )

    def test_cuda_graph_cli_writes_negative_report_before_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c2a_root, c2b_root = _build_cuda_graph_pair(
                root,
                c2b_stored_tail=(-1.03, -2.12),
            )
            output = root / "comparison.json"
            project_root = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env["JPH_ROOT"] = str(root)
            env["PYTHONPATH"] = str(project_root)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        project_root
                        / "scripts"
                        / "compare_sglang_cuda_graph_screen.py"
                    ),
                    str(c2a_root),
                    str(c2b_root),
                    "--output",
                    str(output),
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["summary"]["mechanism_supported"])
            self.assertEqual(report["record_sha256"], _sha256(report))

    def test_cuda_graph_pair_rejects_another_runtime_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c2a_root, c2b_root = _build_cuda_graph_pair(root, c2b_gpu=1)
            with patch.dict(os.environ, {"JPH_ROOT": str(root)}):
                report = compare_cuda_graph_screen_runs(c2a_root, c2b_root)
            self.assertFalse(report["summary"]["paired_runtime_invariants_equal"])
            self.assertFalse(report["summary"]["mechanism_supported"])

    def test_runtime_v2_rejects_treatment_server_arg_mismatch(self) -> None:
        contract = _runtime_contract(
            run_id="run-c2b",
            pair_id="cuda-pair-0123456789abcdef",
            mode=C0_MODE,
            physical_gpu_id=0,
            dataset_selection=CUDA_GRAPH_DATASET_SELECTION,
            disable_cuda_graph=True,
            runtime_schema="jph.sglang-inference-runtime.v2",
            experimental_axis="cuda-graph-v1",
        )
        contract["fixed"]["server_args"]["disable_cuda_graph"] = False
        with self.assertRaisesRegex(
            ValueError,
            "CUDA Graph treatment differs from effective server args",
        ):
            inference_runtime_contract_sha256(contract)

    def test_cuda_graph_pointer_uses_separate_pair_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_id = "cuda-pair-0123456789abcdef"
            pair_root = (
                root
                / "artifacts"
                / "sglang-cuda-graph-screen"
                / "pairs"
                / pair_id
            )
            pair_root.mkdir(parents=True, mode=0o700)
            pair_root.chmod(0o700)
            run_root = root / "artifacts" / "cuda-cell"
            run_root.mkdir(parents=True, mode=0o700)
            pointer = pair_root / "c2a-run-root.txt"
            write_pointer(
                configured_root=root,
                pair_id=pair_id,
                pair_artifact_group="sglang-cuda-graph-screen",
                cell="c2a",
                pointer=pointer,
                run_root=run_root,
            )
            self.assertEqual(pointer.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
