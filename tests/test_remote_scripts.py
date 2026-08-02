from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

from jphrl.envs.calculator import TASKS
from jphrl.harness.controller import SmokeHarnessController
from jphrl.harness.spec import HarnessSpec
from jphrl.models.base import MockStructuredModel
from jphrl.runner import run_calculator_smoke
from jphrl.trajectory.schema import JointVersion
from scripts.verify_real_smoke_trace import verify_trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


class RemoteScriptContractTests(unittest.TestCase):
    def test_environment_freezes_root_mirrors_and_worker_defaults(self) -> None:
        text = (SCRIPTS / "remote_env.sh").read_text(encoding="utf-8")
        self.assertIn('export JPH_ROOT="/mnt/sdb/ljw/chizm"', text)
        self.assertIn('export PYTHONPATH="${JPH_PROJECT_DIR}"', text)
        self.assertIn('export PATH="${JPH_ROOT}/bin:/usr/local/cuda/bin:', text)
        self.assertNotIn("PYTHONPATH:+", text)
        self.assertIn('export HF_ENDPOINT="https://hf-mirror.com"', text)
        self.assertIn(
            'export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"',
            text,
        )
        self.assertIn('export JPH_DATALOADER_WORKERS="${JPH_DATALOADER_WORKERS:-2}"', text)
        self.assertIn('export UV_PYTHON_INSTALL_DIR="${JPH_ROOT}/runtime/python"', text)
        self.assertIn('export UV_PYTHON_BIN_DIR="${JPH_ROOT}/bin"', text)
        self.assertIn('export UV_PYTHON_PREFERENCE="only-managed"', text)
        self.assertIn('export AREAL_CACHE_DIR="${JPH_ROOT}/cache/areal"', text)
        self.assertNotIn("UV_MANAGED_PYTHON", text)
        self.assertNotIn("UV_DEFAULT_INDEX", text)

    def test_areal_bootstrap_is_pinned_and_installs_flash_after_exact_sync(self) -> None:
        text = (SCRIPTS / "bootstrap_areal_v2.sh").read_text(encoding="utf-8")
        self.assertIn("fee938eada49208a5aabdbc1095730a13076a349", text)
        self.assertIn('UV_VERSION="0.11.26"', text)
        self.assertIn('UV_BIN="${JPH_ROOT}/bin/uv"', text)
        self.assertIn("UV_UNMANAGED_INSTALL=\"${JPH_ROOT}/bin\"", text)
        self.assertNotIn("python3 -m venv", text)
        sync_position = text.index('"${UV_BIN}" sync')
        flash_install_position = text.rindex('"${UV_BIN}" pip install')
        self.assertLess(sync_position, flash_install_position)
        self.assertIn("--no-deps", text)
        self.assertIn("--locked", text)
        self.assertIn("--extra cuda", text)
        self.assertNotIn("--default-index", text)
        self.assertIn('"${UV_BIN}" pip freeze --python', text)
        self.assertIn("validate_areal_runtime.py", text)

    def test_areal_b0_is_bounded_and_does_not_max_workers(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(encoding="utf-8")
        self.assertIn("+total_train_steps=1", text)
        self.assertIn("train_dataset.num_workers=2", text)
        self.assertIn("valid_dataset.num_workers=2", text)
        self.assertIn("rollout.max_concurrent_rollouts=8", text)
        self.assertIn("rollout.max_head_offpolicyness=0", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", text)
        self.assertIn('MODEL_REPORT="${JPH_ROOT}/artifacts/bootstrap/', text)
        self.assertIn('DATASET_REPORT="${JPH_ROOT}/artifacts/bootstrap/', text)
        self.assertIn('actor.path="${MODEL_SNAPSHOT}"', text)
        self.assertIn('train_dataset.path="${DATASET_SNAPSHOT}"', text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("HF_DATASETS_OFFLINE=1", text)
        self.assertIn("TRANSFORMERS_OFFLINE=1", text)
        self.assertIn("secrets.token_urlsafe(32)", text)
        self.assertIn('JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}"', text)
        self.assertIn(
            "'+rollout.agent.admin_api_key=${oc.env:JPH_AREAL_ADMIN_API_KEY}'",
            text,
        )
        self.assertNotIn("AREAL_ALLOW_DEFAULT_ADMIN_KEY=1", text)
        self.assertIn("umask 077", text)
        self.assertIn("redact_runtime_admin_key.py", text)
        self.assertNotIn("set -x", text)
        self.assertIn('JPH_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.6', text)
        self.assertIn('export CUDACXX="${CUDA_HOME}/bin/nvcc"', text)
        self.assertIn('export PATH="${AREAL_VENV}/bin:${CUDA_HOME}/bin:${PATH}"', text)
        self.assertIn("-std=c++20", text)

    def test_real_smoke_requires_an_on_disk_trace_audit(self) -> None:
        text = (SCRIPTS / "run_remote_smoke.sh").read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", text)
        self.assertIn("verify_real_smoke_trace.py", text)
        self.assertIn("AUDIT_PATH=", text)
        self.assertIn("prefetch_hf_model.py", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("flock -n 9", text)

    def test_dataset_prefetch_pins_revision_and_audits_cache_paths(self) -> None:
        text = (SCRIPTS / "prefetch_hf_dataset.py").read_text(encoding="utf-8")
        self.assertIn('repo_type="dataset"', text)
        self.assertIn("revision=commit", text)
        self.assertIn("require_within_configured_root(snapshot)", text)
        self.assertIn('require_within_configured_root(entry["filename"])', text)

    def test_model_snapshot_validation_is_local_and_cuda_backed(self) -> None:
        text = (SCRIPTS / "validate_hf_model_snapshot.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("local_files_only=True", text)
        self.assertIn('require_within_configured_root(metadata["snapshot_path"])', text)
        self.assertIn('model.generate(', text)
        self.assertIn('torch.cuda.synchronize()', text)
        self.assertIn('"peak_memory_bytes"', text)

    def test_b0_waiter_is_bounded_and_never_kills_gpu_processes(self) -> None:
        text = (SCRIPTS / "wait_and_run_areal_b0.sh").read_text(encoding="utf-8")
        self.assertIn('JPH_B0_MAX_WAIT_SECONDS:-86400', text)
        self.assertIn('JPH_B0_POLL_SECONDS:-60', text)
        self.assertIn('JPH_B0_MIN_FREE_MEMORY_MIB:-71680', text)
        self.assertIn('JPH_B0_MAX_USED_MEMORY_MIB:-10240', text)
        self.assertIn('GPU_MEMORY_FREE < MIN_FREE_MEMORY_MIB', text)
        self.assertIn('GPU_MEMORY_USED > MAX_USED_MEMORY_MIB', text)
        self.assertIn('exec /bin/bash "${SCRIPT_DIR}/run_areal_official_b0.sh"', text)
        self.assertNotIn("pkill", text)
        self.assertNotIn("kill -", text)

    def test_runtime_admin_key_redactor_stays_in_root_and_scrubs_text(self) -> None:
        redactor = SCRIPTS / "redact_runtime_admin_key.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "run" / "config.yaml"
            artifact.parent.mkdir()
            secret = "jph-b0-test-secret"
            artifact.write_text(f"admin_api_key: {secret}\n", encoding="utf-8")
            env = {
                **os.environ,
                "JPH_ROOT": str(root),
                "JPH_AREAL_ADMIN_API_KEY": secret,
            }
            subprocess.run(
                [sys.executable, str(redactor), str(root / "run")],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            result = artifact.read_text(encoding="utf-8")
            self.assertNotIn(secret, result)
            self.assertIn("<redacted-runtime-admin-key>", result)

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [sys.executable, str(redactor), str(root.parent)],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
    def test_b0_launcher_rechecks_memory_headroom(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('JPH_B0_MIN_FREE_MEMORY_MIB:-71680', text)
        self.assertIn('JPH_B0_MAX_USED_MEMORY_MIB:-10240', text)
        self.assertIn('--query-gpu=memory.used,memory.free', text)
        self.assertIn('GPU_MEMORY_FREE < MIN_FREE_MEMORY_MIB', text)
        self.assertIn('GPU_MEMORY_USED > MAX_USED_MEMORY_MIB', text)

    def test_real_trace_audit_rejects_scripted_and_accepts_valid_hf_metadata(self) -> None:
        result = run_calculator_smoke(
            model=MockStructuredModel(),
            task=TASKS["add-17-25"],
            controller=SmokeHarnessController(),
            harness_spec=HarnessSpec(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            result.trace.write_json(path)
            with self.assertRaisesRegex(ValueError, "scripted|non-HF"):
                verify_trace(path)

            payload = json.loads(path.read_text(encoding="utf-8"))
            commit = "a" * 40
            payload["joint_version"]["policy"] = f"hf:test-policy@{commit}:config"
            payload["joint_version"]["tokenizer"] = f"hf:test-tokenizer@{commit}:tokenizer"
            version_id = JointVersion(**payload["joint_version"]).version_id
            for event in payload["events"]:
                event["joint_version_id"] = version_id
                if event["kind"] == "model_response":
                    event["payload"].update(
                        {
                            "input_token_ids": [10],
                            "output_token_ids": [20],
                            "output_token_logprobs": [-0.1],
                            "completion_loss_mask": [1],
                            "policy_kind": "causal_lm",
                            "token_metadata_status": "available",
                        }
                    )
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = verify_trace(path)
            self.assertTrue(report["ok"])
            self.assertEqual(report["completion_token_counts"], [1, 1])


if __name__ == "__main__":
    unittest.main()
