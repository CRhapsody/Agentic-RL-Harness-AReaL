import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
    def test_gpu_shell_scripts_have_no_broad_kill_or_fixed_memory_cap(self) -> None:
        gpu_markers = (
            "nvidia-smi",
            "CUDA_VISIBLE_DEVICES",
            "gpu_memory_utilization",
            "eight-GPU",
            "8gpu",
        )
        gpu_scripts = {
            path: path.read_text(encoding="utf-8")
            for path in SCRIPTS.glob("*.sh")
            if any(marker in path.read_text(encoding="utf-8") for marker in gpu_markers)
        }
        self.assertTrue(gpu_scripts)
        for path, text in gpu_scripts.items():
            with self.subTest(script=path.name):
                self.assertNotIn("killall", text)
                self.assertNotIn("pkill", text)
                self.assertNotIn("sudo kill", text)
                self.assertNotIn("SOFT_MAX_NEW_GPU_MEMORY_MIB", text)
                self.assertNotIn("HARD_MAX_NEW_GPU_MEMORY_MIB", text)
                for obsolete_limit_mib in (24576, 25600, 26624, 30720):
                    self.assertNotIn(str(obsolete_limit_mib), text)

    def test_agent_service_adapter_verifier_is_cpu_only_and_uses_premerge_hook(
        self,
    ) -> None:
        text = (SCRIPTS / "verify_areal_agent_service_adapter.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("SessionData", text)
        self.assertIn("export_session_trajectory_with_pre_batch_hook", text)
        self.assertIn("stage_agent_service_training_binding", text)
        self.assertIn("PersistentAgentServicePreBatchBinder", text)
        self.assertIn("HermesModelCallReceipt", text)
        self.assertIn("build_policy_training_admission", text)
        self.assertIn("admit_real_harness_action_samples", text)
        self.assertIn("build_frozen_joint_credit_alignment", text)
        self.assertIn('journal_root / "finalized"', text)
        self.assertIn('"policy_optimizer_update"', text)
        self.assertIn('"harness_optimizer_update"', text)
        self.assertIn('"gpu_used": False', text)
        self.assertNotIn("torch.cuda", text)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", text)
        self.assertNotIn("optimizer.step", text)

    def test_areal_joint_bridge_is_real_bounded_and_prompt_effective(self) -> None:
        launcher = (SCRIPTS / "run_areal_joint_bridge.sh").read_text(encoding="utf-8")
        runner = (SCRIPTS / "run_areal_joint_bridge_eval.py").read_text(
            encoding="utf-8"
        )
        verifier = (SCRIPTS / "verify_areal_joint_bridge.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('TASK_COUNT="${JPH_AREAL_JOINT_BRIDGE_TASKS:-4}"', launcher)
        self.assertIn(
            'TASK_OFFSET="${JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET:-0}"', launcher
        )
        self.assertIn("formal-v1 requires count=4 offset=0", launcher)
        self.assertIn("m0-torch-joint-v1 requires count=4 offset=0", launcher)
        self.assertIn(
            'JPH_HARNESS_CONTROLLER_KIND="${HARNESS_CONTROLLER_KIND}"',
            launcher,
        )
        self.assertIn("_harness_workflow_config", runner)
        self.assertIn('"harness_kind": harness_kind', runner)
        self.assertIn("gconfig.max_new_tokens=64", launcher)
        self.assertIn("rollout.max_concurrent_rollouts=1", launcher)
        self.assertIn('SGLANG_MEM_FRACTION_STATIC="0.35"', launcher)
        self.assertIn('SGLANG_MEM_FRACTION_STATIC="0.29"', launcher)
        self.assertIn(
            'sglang.mem_fraction_static="${SGLANG_MEM_FRACTION_STATIC}"',
            launcher,
        )
        self.assertIn("JPH_SGLANG_MEM_FRACTION_STATIC", launcher + runner)
        self.assertGreaterEqual(
            launcher.count("--query-gpu=memory.used,memory.free"), 2
        )
        self.assertGreaterEqual(launcher.count("--query-compute-apps"), 2)
        self.assertIn("GPU_MEMORY_USED_AT_LAUNCH", launcher)
        self.assertNotIn("MAX_NEW_GPU_MEMORY_MIB", launcher)
        self.assertNotIn("REQUIRE_EMPTY_COMPUTE_PROCESSES", launcher)
        self.assertIn("audit_gpu_memory_envelope.py", launcher)
        self.assertIn('>> "${MEMORY_SAMPLES}"', launcher)
        self.assertIn("setsid", launcher)
        self.assertNotIn("GPU memory watchdog", launcher)
        self.assertIn("stop_run_session", launcher)
        self.assertIn("redact_and_check_secret", launcher)
        self.assertIn("assert_no_run_gpu_processes", launcher)
        self.assertIn("--verify-absent", launcher)
        self.assertNotIn("rg -", launcher)
        self.assertIn("audit_gpu_memory", launcher)
        self.assertIn('--expected-count "${TASK_COUNT}"', launcher)
        self.assertIn("--expected-dataset-selection", launcher)
        self.assertIn("--expected-generation-logprob-mode", launcher)
        self.assertIn("--expected-sglang-version", launcher)
        self.assertIn('EXPECTED_SGLANG_VERSION="0.5.10.post1"', launcher)
        self.assertIn("--expected-project-commit", launcher)
        self.assertIn("status --porcelain", launcher)
        self.assertIn("--dataset-report", launcher)
        self.assertIn("verify_areal_joint_bridge.py", launcher)
        self.assertIn("ArealJointBridgeWorkflow", runner)
        self.assertIn("controller.wait(submitted, timeout=900.0)", runner)
        self.assertIn("controller.compute_logp(results)", runner)
        self.assertIn("JPHRemoteSGLangEngine", runner)
        self.assertIn("results = RTensor.localize(results)", runner)
        self.assertIn("rescored_logprobs = RTensor.localize(rescored_logprobs)", runner)
        self.assertIn('config.rollout._version != "v1"', runner)
        self.assertIn("engine_version_before_score = controller.get_version()", runner)
        self.assertIn("engine_version_after_score = controller.get_version()", runner)
        self.assertIn("JPH_AREAL_SAME_BACKEND_SCORE_DIR", launcher + runner)
        self.assertIn("same-backend-score-", runner)
        self.assertIn("recompute_behavior_logprobs", verifier)
        self.assertIn("require_all_passed=False", verifier)
        self.assertIn("_verify_same_backend_scores", verifier)
        self.assertIn("MAX_SAME_BACKEND_MEAN_IMPORTANCE_RATIO_ERROR = 0.02", verifier)
        self.assertIn("MAX_SAME_BACKEND_IMPORTANCE_RATIO_ERROR = 0.10", verifier)
        self.assertIn("_recompute_prompt_tokens", verifier)
        self.assertIn("harness_decision_changed_prompt", verifier)
        self.assertIn("secrets.token_urlsafe(32)", launcher)
        self.assertIn('JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}"', launcher)
        self.assertIn(
            "'+rollout.agent.admin_api_key=${oc.env:JPH_AREAL_ADMIN_API_KEY}'",
            launcher,
        )
        self.assertIn("redact_runtime_admin_key.py", launcher)
        self.assertIn("--same-backend-score-dir", launcher + verifier)
        self.assertNotIn("optimizer.step", launcher + runner + verifier)

    def test_m0_live_launcher_is_sequential_bounded_and_fail_closed(self) -> None:
        launcher = (SCRIPTS / "run_m0_live_joint.sh").read_text(encoding="utf-8")
        pipeline = (SCRIPTS / "run_m0_live_pipeline.sh").read_text(
            encoding="utf-8"
        )
        waiter = (SCRIPTS / "wait_and_run_m0_live_pipeline.sh").read_text(
            encoding="utf-8"
        )
        entry = (
            PROJECT_ROOT / "jphrl" / "experiments" / "m0_live_joint.py"
        ).read_text(encoding="utf-8")
        self.assertIn("umask 077", launcher)
        self.assertNotIn('MAX_NEW_GPU_MEMORY_MIB', launcher)
        self.assertNotIn('MAX_USED_MEMORY_MIB', launcher)
        self.assertNotIn('MIN_FREE_MEMORY_MIB', launcher)
        self.assertGreaterEqual(
            launcher.count("--query-gpu=memory.used,memory.free"),
            1,
        )
        self.assertIn("--query-compute-apps", launcher)
        self.assertIn('observe_gpu "preflight"', launcher)
        self.assertIn('observe_gpu "immediately-before-python"', launcher)
        self.assertIn("gpu-memory.csv", launcher)
        self.assertIn("audit_gpu_memory_envelope.py", launcher)
        self.assertIn("setsid", launcher)
        self.assertNotIn("GPU memory watchdog", launcher)
        self.assertIn("stop_run_process_group", launcher)
        self.assertIn("redact_and_check_secret", launcher)
        self.assertIn("assert_no_run_processes", launcher)
        self.assertIn("--verify-absent", launcher)
        self.assertNotIn("rg -", launcher)
        self.assertIn("verify_m0_live_joint.py", launcher)
        self.assertIn("redact_runtime_admin_key.py", launcher)
        self.assertIn("JPH_AREAL_ADMIN_API_KEY", launcher + entry)
        self.assertIn("RLVRM0JointUpdateRunner", entry)
        self.assertIn("RealRlvrM0CandidateEvaluator", entry)
        self.assertIn("LiveM0GPULaunchGuard", entry)
        self.assertIn("production-sglang", entry)
        self.assertIn("training-actor", entry)
        self.assertIn("LocalScheduler", entry)
        self.assertIn("InferenceEngineConfig", entry)
        self.assertIn('backend="sglang:d1"', entry)
        self.assertIn('rollout_sglang_mem_fraction_static=0.29', entry)
        self.assertIn('max_new_gpu_memory_gib=None', entry)
        self.assertIn("write_m0_rlvr_estimator_template.py", pipeline)
        self.assertIn("m0-torch-joint-v1", pipeline)
        self.assertIn("JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH", pipeline)
        self.assertIn("run_areal_joint_bridge.sh", pipeline)
        self.assertIn("run_m0_live_joint.sh", pipeline)
        self.assertLess(
            pipeline.index("run_areal_joint_bridge.sh"),
            pipeline.index("run_m0_live_joint.sh"),
        )
        self.assertIn('JPH_M0_POLL_SECONDS:-60', waiter)
        self.assertIn('JPH_M0_MAX_WAIT_SECONDS:-604800', waiter)
        self.assertIn("--query-gpu=memory.used,memory.free", waiter)
        self.assertIn("--query-compute-apps", waiter)
        self.assertNotIn("MAX_USED_MEMORY_MIB", waiter)
        self.assertNotIn("MIN_FREE_MEMORY_MIB", waiter)
        self.assertIn("run_m0_live_pipeline.sh", waiter)
        self.assertNotIn("pkill", waiter)
        self.assertNotIn("kill ", waiter)
        self.assertNotIn("optimizer.step", launcher + entry)

    def test_sglang_logprob_screen_is_unseen_single_variable_and_bounded(self) -> None:
        wrapper = (SCRIPTS / "run_sglang_logprob_screen.sh").read_text(encoding="utf-8")
        pair_runner = (SCRIPTS / "run_sglang_logprob_screen_pair.sh").read_text(
            encoding="utf-8"
        )
        launcher = (SCRIPTS / "run_areal_joint_bridge.sh").read_text(encoding="utf-8")
        runner = (SCRIPTS / "run_areal_joint_bridge_eval.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("JPH_AREAL_JOINT_BRIDGE_TASKS=4", wrapper)
        self.assertIn("JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET=32", wrapper)
        self.assertIn("standard-log-of-softmax-v1", wrapper)
        self.assertIn("original-log-softmax-v1", wrapper)
        self.assertIn("SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE=0", launcher)
        self.assertIn("SGLANG_RETURN_ORIGINAL_LOGPROB_VALUE=1", launcher)
        self.assertIn("SGLANG_ENV_MODE_AUDIT", launcher)
        self.assertIn("envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get()", launcher)
        self.assertIn("logprob-mechanism-screen-v1", launcher + wrapper)
        self.assertIn("exec /usr/bin/env -i", wrapper)
        self.assertIn("JPH_CLEAN_ENVIRONMENT_POLICY=env-i-v1", wrapper)
        self.assertIn("requires the env-i-v1 launch policy", launcher)
        self.assertIn("GPU_UUID", launcher)
        self.assertIn("launch-manifest.json", runner)
        self.assertIn("audit_sglang_logprob_screen_cell.py", launcher)
        self.assertIn("write_sglang_logprob_screen_pointer.py", launcher)
        self.assertIn("server_args", runner)
        self.assertIn("inference_runtime_contract_sha256", runner)
        self.assertIn("controller.destroy()", runner)
        c0_position = pair_runner.index('"${GPU_ID}" c0')
        c1_position = pair_runner.index('"${GPU_ID}" c1')
        compare_position = pair_runner.index("compare_sglang_logprob_screen.py")
        self.assertLess(c0_position, c1_position)
        self.assertLess(c1_position, compare_position)
        self.assertIn(
            'RUN_ID="${RUN_STAMP}-${SGLANG_LOGPROB_MODE}-${RUN_NONCE}"', launcher
        )
        self.assertIn("joint-bridge-${RUN_ID}", launcher)
        self.assertIn("JPH_SGLANG_DISABLE_CUDA_GRAPH=false", wrapper)
        self.assertNotIn("JPH_SGLANG_DISABLE_CUDA_GRAPH=true", wrapper)
        self.assertNotIn("optimizer.step", wrapper + pair_runner)

    def test_cuda_graph_screen_is_new_slice_single_variable_and_bounded(self) -> None:
        wrapper = (SCRIPTS / "run_sglang_cuda_graph_screen.sh").read_text(
            encoding="utf-8"
        )
        pair_runner = (SCRIPTS / "run_sglang_cuda_graph_screen_pair.sh").read_text(
            encoding="utf-8"
        )
        launcher = (SCRIPTS / "run_areal_joint_bridge.sh").read_text(encoding="utf-8")
        comparator = (SCRIPTS / "compare_sglang_cuda_graph_screen.py").read_text(
            encoding="utf-8"
        )
        config_preflight = (SCRIPTS / "validate_sglang_cuda_graph_config.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("JPH_AREAL_JOINT_BRIDGE_TASK_OFFSET=64", wrapper)
        self.assertIn("JPH_SGLANG_LOGPROB_MODE=standard-log-of-softmax-v1", wrapper)
        self.assertIn("JPH_EXPERIMENTAL_AXIS=cuda-graph-v1", wrapper)
        self.assertIn("DISABLE_CUDA_GRAPH=false", wrapper)
        self.assertIn("DISABLE_CUDA_GRAPH=true", wrapper)
        self.assertIn("exec /usr/bin/env -i", wrapper)
        self.assertIn("cuda-graph-mechanism-screen-v1", wrapper + launcher)
        self.assertIn(
            '+sglang.disable_cuda_graph="${SGLANG_DISABLE_CUDA_GRAPH}"', launcher
        )
        self.assertIn('f"+sglang.disable_cuda_graph={value}"', config_preflight)
        c2a_position = pair_runner.index('"${GPU_ID}" c2a')
        c2b_position = pair_runner.index('"${GPU_ID}" c2b')
        compare_position = pair_runner.index("compare_sglang_cuda_graph_screen.py")
        self.assertLess(c2a_position, c2b_position)
        self.assertLess(c2b_position, compare_position)
        self.assertIn("compare_cuda_graph_screen_runs", comparator)
        self.assertIn("load_expr_config", config_preflight)
        self.assertIn("SGLangConfig.build_args", config_preflight)
        self.assertIn("type(composed) is not bool", config_preflight)
        self.assertIn("type(built) is not bool", config_preflight)
        self.assertIn("torch.cuda.is_initialized()", config_preflight)
        self.assertIn("os.O_EXCL", config_preflight)
        self.assertIn("config-preflight.json", pair_runner)
        preflight_position = pair_runner.index("validate_sglang_cuda_graph_config.py")
        self.assertLess(preflight_position, c2a_position)
        self.assertNotIn(
            "optimizer.step",
            wrapper + pair_runner + comparator + config_preflight,
        )

    def test_g1_launcher_is_private_external_and_verified(self) -> None:
        text = (SCRIPTS / "run_g1_integrity.sh").read_text(encoding="utf-8")
        self.assertIn("umask 077", text)
        self.assertIn("/mnt/sdb/ljw/chizm/artifacts/g1/", text)
        self.assertIn(
            "/mnt/sdb/ljw/chizm/venvs/areal-v2.0.0/bin/python",
            text,
        )
        self.assertNotIn("/mnt/sdb/ljw/chizm/venvs/areal-v2/bin/python", text)
        self.assertIn("--version-fixtures 1000", text)
        self.assertIn("--project-commit", text)
        self.assertIn("verify_g1_integrity.py", text)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", text)

    def test_environment_freezes_root_mirrors_and_worker_defaults(self) -> None:
        text = (SCRIPTS / "remote_env.sh").read_text(encoding="utf-8")
        self.assertIn('export JPH_ROOT="/mnt/sdb/ljw/chizm"', text)
        self.assertIn('export PYTHONPATH="${JPH_PROJECT_DIR}"', text)
        self.assertIn(
            'export JPH_AREAL_ROOT="${JPH_ROOT}/src/AReaL-v2.0.0"', text
        )
        self.assertIn('export JPH_AREAL_SOURCE="${JPH_AREAL_ROOT}"', text)
        self.assertIn('export PATH="${JPH_ROOT}/bin:/usr/local/cuda/bin:', text)
        self.assertNotIn("PYTHONPATH:+", text)
        self.assertIn('export HF_ENDPOINT="https://hf-mirror.com"', text)
        self.assertIn(
            'export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"',
            text,
        )
        self.assertIn(
            'export JPH_DATALOADER_WORKERS="${JPH_DATALOADER_WORKERS:-2}"', text
        )
        self.assertIn('export UV_PYTHON_INSTALL_DIR="${JPH_ROOT}/runtime/python"', text)
        self.assertIn('export UV_PYTHON_BIN_DIR="${JPH_ROOT}/bin"', text)
        self.assertIn('export UV_PYTHON_PREFERENCE="only-managed"', text)
        self.assertIn('export AREAL_CACHE_DIR="${JPH_ROOT}/cache/areal"', text)
        self.assertNotIn("UV_MANAGED_PYTHON", text)
        self.assertNotIn("UV_DEFAULT_INDEX", text)

    def test_areal_bootstrap_is_pinned_and_installs_flash_after_exact_sync(
        self,
    ) -> None:
        text = (SCRIPTS / "bootstrap_areal_v2.sh").read_text(encoding="utf-8")
        self.assertIn("fee938eada49208a5aabdbc1095730a13076a349", text)
        self.assertIn('UV_VERSION="0.11.26"', text)
        self.assertIn('UV_BIN="${JPH_ROOT}/bin/uv"', text)
        self.assertIn('UV_UNMANAGED_INSTALL="${JPH_ROOT}/bin"', text)
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
        subprocess.run(
            ["bash", "-n", str(SCRIPTS / "run_areal_official_b0.sh")],
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertIn("TOTAL_TRAIN_STEPS=1", text)
        self.assertIn('+total_train_steps="${TOTAL_TRAIN_STEPS}"', text)
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
        self.assertIn("secrets.token_urlsafe(48)", text)
        self.assertIn('JPH_AREAL_ADMIN_API_KEY="${RUN_ADMIN_API_KEY}"', text)
        self.assertIn(
            "'+rollout.agent.admin_api_key=${oc.env:JPH_AREAL_ADMIN_API_KEY}'",
            text,
        )
        self.assertNotIn("AREAL_ALLOW_DEFAULT_ADMIN_KEY=1", text)
        self.assertIn("umask 077", text)
        self.assertIn("redact_runtime_admin_key.py", text)
        self.assertNotIn("set -x", text)
        self.assertIn("JPH_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.6", text)
        self.assertIn('export CUDACXX="${CUDA_HOME}/bin/nvcc"', text)
        self.assertIn('export PATH="${AREAL_VENV}/bin:${CUDA_HOME}/bin:${PATH}"', text)
        self.assertIn("-std=c++20", text)
        self.assertIn("mktemp --suffix=.o", text)
        self.assertIn('realpath -e "${AREAL_VENV}/bin/python"', text)
        self.assertIn(
            '"${JPH_ROOT}/runtime/python/"*/bin/python3.12',
            text,
        )
        self.assertIn('stat -c %u "${AREAL_PYTHON_REALPATH}"', text)

    def test_areal_b0_rejects_non_tmux_invocation_before_remote_setup(self) -> None:
        env = os.environ.copy()
        env.pop("TMUX", None)
        result = subprocess.run(
            ["bash", str(SCRIPTS / "run_areal_official_b0.sh")],
            check=False,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must run inside tmux", result.stderr)

    def test_areal_b0_is_one_eight_gpu_areal_job(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(encoding="utf-8")
        self.assertEqual(text.count('"${AREAL_VENV}/bin/python" examples/math/gsm8k_rl.py'), 1)
        self.assertIn("os.setsid()", text)
        self.assertIn("os.execv(sys.argv[1], sys.argv[1:])", text)
        self.assertIn("cluster.n_nodes=1", text)
        self.assertIn("cluster.n_gpus_per_node=8", text)
        self.assertIn("actor.backend=fsdp:d4p1t1", text)
        self.assertIn("rollout.backend=sglang:d4p1t1", text)
        self.assertIn("ref.backend=fsdp:d4p1t1", text)
        self.assertIn("ref.scheduling_strategy.type=colocation", text)
        self.assertIn("ref.scheduling_strategy.target=actor", text)
        self.assertIn("sglang.mem_fraction_static=0.29", text)
        self.assertIn("sglang.context_length=1024", text)
        self.assertIn("sglang.max_running_requests=2", text)
        self.assertNotIn("run_m0_live_pipeline.sh", text)
        self.assertNotIn("m0-live-parallel-8gpu", text)
        self.assertNotIn("WORLD_SIZE=1", text)

    def test_areal_b0_composes_and_persists_update_checkpoint_config(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(encoding="utf-8")
        self.assertIn("GRPOConfig, load_expr_config", text)
        self.assertIn('CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn("hydra-preflight.json", text)
        self.assertIn("actor.optimizer.lr=1.70e-5", text)
        self.assertIn("actor.optimizer.lr_scheduler_type=constant", text)
        self.assertIn("actor.optimizer.warmup_steps_proportion=0.0", text)
        self.assertIn("actor.kl_ctl=0.0", text)
        self.assertIn("SAVER_FREQ_STEPS=1", text)
        self.assertIn('saver.freq_steps="${SAVER_FREQ_STEPS}"', text)
        self.assertIn("saver.freq_epochs=null", text)
        self.assertIn("saver.freq_secs=null", text)
        self.assertIn("+saver.mode=sync", text)
        self.assertIn("recover.mode=disabled", text)
        self.assertIn('saver.fileroot="${RUN_ROOT}"', text)

    def test_areal_gpu_holder_reuses_b0_safety_and_bounds_disk(self) -> None:
        holder = (SCRIPTS / "run_areal_gpu_holder.sh").read_text(encoding="utf-8")
        launcher = (SCRIPTS / "run_areal_official_b0.sh").read_text(
            encoding="utf-8"
        )
        stopper = (SCRIPTS / "stop_areal_gpu_holder.py").read_text(
            encoding="utf-8"
        )
        subprocess.run(
            ["bash", "-n", str(SCRIPTS / "run_areal_gpu_holder.sh")],
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertIn("JPH_B0_RUN_MODE=holder", holder)
        self.assertIn('run_areal_official_b0.sh"', holder)
        self.assertNotIn("nvidia-smi", holder)
        self.assertIn("JPH_HOLDER_TOTAL_TRAIN_STEPS:-10000", launcher)
        self.assertIn("JPH_HOLDER_MAX_RUNTIME_SECONDS:-21600", launcher)
        self.assertIn("ROLLOUT_DUMP_TO_FILE=false", launcher)
        self.assertIn("SAVER_FREQ_STEPS=null", launcher)
        self.assertIn("recover.freq_steps=null", launcher)
        self.assertIn("recover.freq_epochs=null", launcher)
        self.assertIn("recover.freq_secs=null", launcher)
        self.assertIn("evaluator.freq_steps=null", launcher)
        self.assertIn("evaluator.freq_epochs=null", launcher)
        self.assertIn("evaluator.freq_secs=null", launcher)
        self.assertIn("final_state=stopped", launcher)
        self.assertIn('[[ "${RUN_MODE}" == official ]]', launcher)
        self.assertIn("holder-control.json", launcher)
        self.assertIn("stop.requested.json", launcher)
        self.assertIn("PID {pid} start time no longer matches", stopper)
        self.assertIn("os.kill(launcher_pid, signal.SIGTERM)", stopper)
        self.assertNotIn("pkill", stopper)
        self.assertNotIn("killall", holder + stopper)

    def test_areal_b0_owns_all_locks_and_only_kills_its_session(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ -z "${TMUX:-}" ]]', text)
        self.assertIn("areal-official-b0-8gpu.lock", text)
        self.assertIn("declare -A GPU_LOCK_FDS", text)
        self.assertIn('flock -n "${gpu_lock_fd}"', text)
        self.assertIn("process_start_time", text)
        self.assertIn("bind_job_identity", text)
        self.assertIn('[[ "${JOB_SESSION_ID}" == "${JOB_PID}" ]]', text)
        self.assertIn('session_pids "${JOB_SESSION_ID}"', text)
        self.assertIn('kill -TERM -- "${pids[@]}"', text)
        self.assertIn('kill -KILL -- "${pids[@]}"', text)
        self.assertIn("JOB_LEADER_REAPED", text)
        self.assertIn("areal-session-straggler-after-coordinator-exit", text)
        self.assertNotIn("nvidia-smi --gpu-reset", text)
        self.assertNotIn("killall", text)
        self.assertNotIn("pkill", text)
        self.assertIn("trap 'exit 129' HUP", text)
        self.assertIn("trap 'exit 130' INT", text)
        self.assertIn("trap 'exit 143' TERM", text)

    def test_areal_b0_persists_eight_gpu_memory_observations_without_caps(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(encoding="utf-8")
        self.assertNotIn("SOFT_MAX_NEW_GPU_MEMORY_MIB", text)
        self.assertNotIn("HARD_MAX_NEW_GPU_MEMORY_MIB", text)
        self.assertIn('AUDIT_PATH="${RUN_ROOT}/gpu-memory-audit.json"', text)
        self.assertIn("jph.areal-official-b0-gpu-memory-observation.v2", text)
        self.assertIn('MEMORY_RUN_KIND="areal-official-b0-v1"', text)
        self.assertIn('"baseline_used_mib": baseline', text)
        self.assertIn('"peak_used_mib": peak_used', text)
        self.assertIn('"peak_delta_mib": delta', text)
        self.assertIn('"memory_limit_enforced": False', text)
        self.assertIn('"max_new_memory_mib": None', text)
        self.assertIn('"sample_count": len(samples)', text)
        self.assertIn('record["record_sha256"] = hashlib.sha256(canonical)', text)
        self.assertNotIn('delta > SOFT_MAX_NEW_GPU_MEMORY_MIB', text)
        self.assertIn('kill -TERM "${B0_ORCHESTRATOR_PID}"', text)
        self.assertIn("append_final_samples", text)
        self.assertIn("write_gpu_memory_audit", text)
        self.assertNotIn('"peak_timestamp_unix":', text)
        self.assertNotIn('"watchdog_breach":', text)
        self.assertNotIn('"failure_reason": failure_reason', text)

    def test_areal_b0_pins_clean_sources_and_verifies_secret_absence(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(encoding="utf-8")
        self.assertIn("fee938eada49208a5aabdbc1095730a13076a349", text)
        self.assertGreaterEqual(text.count("status --porcelain=v1"), 2)
        self.assertIn("Project worktree must be clean before official B0", text)
        self.assertIn("Pinned AReaL worktree must be clean before official B0", text)
        self.assertIn('--verify-absent "${RUN_ROOT}" "${LOG_PATH}"', text)
        self.assertIn('RUN_ROOT="${JPH_ROOT}/artifacts/areal-b0/', text)
        self.assertIn('LOG_PATH="${JPH_ROOT}/logs/areal-b0-', text)
        self.assertIn(
            "project=%s AReaL=%s GPUs=0,1,2,3,4,5,6,7 run_root=%s",
            text,
        )
        self.assertIn('exec tee -a "${LOG_PATH}" >/dev/null', text)
        self.assertIn('NAME_RESOLVE_ROOT="${RUN_ROOT}/name-resolve"', text)
        self.assertIn('"${SCRIPT_DIR}/verify_areal_official_b0.py"', text)
        self.assertIn('--output "${RUN_ROOT}/verification.json"', text)
        self.assertNotIn("harness_optimizer_update", text)

    def test_areal_trace_is_real_bounded_and_recomputed(self) -> None:
        launcher = (SCRIPTS / "run_areal_trace_b0.sh").read_text(encoding="utf-8")
        eval_runner = (SCRIPTS / "run_areal_trace_eval.py").read_text(encoding="utf-8")
        workflow = (PROJECT_ROOT / "jphrl" / "areal_trace_workflow.py").read_text(
            encoding="utf-8"
        )
        verifier = (SCRIPTS / "verify_areal_trace.py").read_text(encoding="utf-8")
        self.assertIn("InteractionWithTokenLogpReward", workflow)
        self.assertIn("interaction.to_tensor_dict()", workflow)
        self.assertIn("return {request.rid: interaction}", workflow)
        self.assertIn("JPH_AREAL_TRACE_TASKS=1", launcher)
        self.assertIn("JPH_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.6", launcher)
        self.assertIn('export CUDACXX="${CUDA_HOME}/bin/nvcc"', launcher)
        self.assertIn(
            'export PATH="${AREAL_VENV}/bin:${CUDA_HOME}/bin:${PATH}"',
            launcher,
        )
        self.assertIn("-std=c++20", launcher)
        self.assertIn("path=config.valid_dataset.path", eval_runner)
        self.assertNotIn(
            'split="test",\n        dataset_config=config.valid_dataset',
            eval_runner,
        )
        self.assertIn("rollout.backend=sglang:d1p1t1", launcher)
        self.assertIn("sglang.mem_fraction_static=0.35", launcher)
        self.assertNotIn("rollout.agent=null", launcher)
        self.assertIn("flock -n 9", launcher)
        self.assertIn("HF_HUB_OFFLINE=1", launcher)
        self.assertIn("AutoModelForCausalLM.from_pretrained", verifier)
        self.assertIn("local_files_only=True", verifier)
        self.assertIn("max_abs_error", verifier)
        self.assertIn('"largest_errors": largest_errors', verifier)
        self.assertIn('"policy_update": False', verifier)
        self.assertIn('"harness_update": False', verifier)

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
        text = (SCRIPTS / "validate_hf_model_snapshot.py").read_text(encoding="utf-8")
        self.assertIn("local_files_only=True", text)
        self.assertIn('require_within_configured_root(metadata["snapshot_path"])', text)
        self.assertIn("model.generate(", text)
        self.assertIn("torch.cuda.synchronize()", text)
        self.assertIn('"peak_memory_bytes"', text)

    def test_b0_waiter_observes_all_cards_and_never_kills_gpu_processes(self) -> None:
        text = (SCRIPTS / "wait_and_run_areal_b0.sh").read_text(encoding="utf-8")
        self.assertIn("--query-gpu=memory.used,memory.free", text)
        self.assertIn("--query-compute-apps=pid,process_name,used_memory", text)
        self.assertIn("without a configured memory limit", text)
        self.assertNotIn("MIN_FREE_MEMORY_MIB", text)
        self.assertNotIn("MAX_USED_MEMORY_MIB", text)
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
            artifact.write_text(
                (
                    "rollout:\n"
                    f"  agent:\n    admin_api_key: {secret}\n"
                    "actor:\n  agent:\n    admin_api_key: areal-admin-key\n"
                    "evaluator:\n  agent:\n    admin_api_key: areal-admin-key\n"
                    "proxy:\n  agent:\n    admin_api_key: areal-admin-key\n"
                ),
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "JPH_ROOT": str(root),
                "JPH_AREAL_ADMIN_API_KEY": secret,
            }
            subprocess.run(
                [
                    sys.executable,
                    str(redactor),
                    "--verify-absent",
                    str(root / "run"),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            result = artifact.read_text(encoding="utf-8")
            self.assertNotIn(secret, result)
            self.assertNotIn("areal-admin-key", result)
            self.assertIn("<redacted-runtime-admin-key>", result)
            self.assertIn("<redacted-default-admin-key>", result)

            extensionless = root / "run" / "resolved-config"
            extensionless.write_text(
                f"admin_api_key={secret}\nfallback=areal-admin-key\n",
                encoding="utf-8",
            )
            subprocess.run(
                [sys.executable, str(redactor), str(extensionless)],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            extensionless_result = extensionless.read_text(encoding="utf-8")
            self.assertNotIn(secret, extensionless_result)
            self.assertNotIn("areal-admin-key", extensionless_result)

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [sys.executable, str(redactor), str(root.parent)],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )

    def test_b0_launcher_rechecks_memory_and_process_observations(self) -> None:
        text = (SCRIPTS / "run_areal_official_b0.sh").read_text(encoding="utf-8")
        self.assertIn("--query-gpu=memory.used,memory.free", text)
        self.assertIn("--query-compute-apps=pid,process_name,used_memory", text)
        self.assertIn("memory_limit_enforced=false", text)
        self.assertIn("existing_compute_processes_are_observed_only=true", text)
        self.assertNotIn("ALLOWED_EXISTING_COMPUTE_UIDS", text)
        self.assertNotIn("MIN_FREE_MEMORY_MIB", text)
        self.assertNotIn("MAX_USED_MEMORY_MIB", text)
        self.assertIn("run_all_gpu_gate preflight", text)
        self.assertIn("run_all_gpu_gate immediately-before-launch", text)
        preflight = text.index("run_all_gpu_gate preflight")
        compose = text.index("GRPOConfig, load_expr_config")
        final_gate = text.index("run_all_gpu_gate immediately-before-launch")
        launch = text.index(
            '"${AREAL_VENV}/bin/python" examples/math/gsm8k_rl.py'
        )
        self.assertLess(preflight, compose)
        self.assertLess(compose, final_gate)
        self.assertLess(final_gate, launch)

    def test_real_trace_audit_rejects_scripted_and_accepts_valid_hf_metadata(
        self,
    ) -> None:
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
            payload["joint_version"]["tokenizer"] = (
                f"hf:test-tokenizer@{commit}:tokenizer"
            )
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
