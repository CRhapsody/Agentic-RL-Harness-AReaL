from __future__ import annotations

import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jphrl.experiments.m0_agent_service_entry import _parse_decision
from jphrl.harness.controller import HarnessState
from jphrl.trajectory.areal_data_proxy_pre_batch import HOOK_STAGE
from jphrl.trajectory.rlvr_online_binding import (
    PersistentRlvrV2AgentPreBatchBinder,
    RlvrOnlineBindingError,
    stage_rlvr_v2_agent_response,
)
from jphrl.trajectory.rlvr_workflow_admission import (
    load_rlvr_workflow_runner_admission_file,
)
from scripts.write_m0_rlvr_estimator_template import (
    write_m0_rlvr_estimator_template,
)
from tests.test_rlvr_workflow_admission import _fake_areal_type_import, _source


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required")
class RlvrOnlineBindingTests(unittest.TestCase):
    def _stage(
        self,
        root: Path,
        *,
        runtime_override: object | None = None,
        interaction_id_override: str | None = None,
    ):
        bridge, interaction, joint_version, _ = _source(
            reward=1.0,
            controller_kind="torch",
        )
        binding = bridge["interaction_adapter_sidecar"]["bindings"][0]
        harness = bridge["harness"]
        prompt = bridge["prompt_binding"]
        runtime = bridge["policy_binding"]["inference_runtime_contract"]
        if runtime_override is not None:
            runtime = runtime_override
        stage_rlvr_v2_agent_response(
            journal_root=root / "journal",
            task_id=bridge["task_id"],
            interaction_id=interaction_id_override or binding["interaction_id"],
            episode_id=binding["episode_id"],
            model_call_id=binding["model_call_id"],
            joint_version=joint_version,
            harness_state=HarnessState(**harness["state"]),
            harness_decision=_parse_decision(harness["decision"]),
            harness_sampling_before_checkpoint=(
                harness["controller_checkpoint_before_decision"]
            ),
            base_messages=prompt["base_messages"],
            effective_messages=prompt["effective_messages"],
            base_input_tokens=prompt["base_input_tokens"],
            effective_input_tokens=prompt["effective_input_tokens"],
            expected_policy_version=(
                bridge["policy_binding"]["expected_inference_engine_version"]
            ),
            project_commit=bridge["origin"]["project_commit"],
            areal_commit=bridge["areal_trace"]["origin"]["areal_commit"],
            behavior_snapshot_path=(
                bridge["areal_trace"]["origin"]["behavior_snapshot_path"]
            ),
            behavior_revision=(
                bridge["areal_trace"]["origin"]["behavior_revision"]
            ),
            dataset_selection=bridge["policy_binding"]["dataset_selection"],
            sglang_version=bridge["policy_binding"]["sglang_version"],
            generation_logprob_mode=(
                bridge["policy_binding"]["generation_logprob_mode"]
            ),
            inference_runtime_contract=runtime,
        )
        return bridge, interaction, joint_version

    @staticmethod
    def _event(interaction: object, interaction_id: str) -> object:
        return SimpleNamespace(
            hook_stage=HOOK_STAGE,
            session_id="public-session-1",
            trajectory_id=7,
            exported_interactions={interaction_id: interaction},
            export_style="individual",
            turn_discount=1.0,
        )

    def test_v2_response_and_real_pre_batch_interaction_finalize_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("bridges", "runner-admissions"):
                (root / name).mkdir(mode=0o700)
            estimator_path = root / "frozen-estimator-template.json"
            write_m0_rlvr_estimator_template(
                output_path=estimator_path,
                allowed_root=root,
            )
            bridge, interaction, joint_version = self._stage(root)
            interaction_id = bridge["interaction_adapter_sidecar"]["bindings"][0][
                "interaction_id"
            ]
            environment = {
                "JPH_ROOT": str(root),
                "JPH_AREAL_JOINT_BRIDGE_DIR": str(root / "bridges"),
                "JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH": str(estimator_path),
                "JPH_RLVR_RUNNER_ADMISSION_DIR": str(root / "runner-admissions"),
            }
            binder = PersistentRlvrV2AgentPreBatchBinder(root / "journal")
            with patch.dict(os.environ, environment, clear=False), _fake_areal_type_import():
                marker = binder(self._event(interaction, interaction_id))

            self.assertEqual(marker["identity"]["interaction_id"], interaction_id)
            self.assertEqual(marker["identity"]["session_id"], "public-session-1")
            self.assertTrue(marker["evidence_scope"]["pre_batch_interaction_binding"])
            self.assertFalse(marker["evidence_scope"]["policy_optimizer_update"])
            runner_files = list((root / "runner-admissions").glob("*.json"))
            self.assertEqual(len(runner_files), 1)
            loaded = load_rlvr_workflow_runner_admission_file(
                runner_files[0],
                allowed_root=root,
                active_joint_version=joint_version,
            )
            self.assertEqual(
                loaded.rlvr_pre_batch_record["identity"]["interaction_id"],
                interaction_id,
            )
            serialized = json.dumps(marker, sort_keys=True).lower()
            self.assertNotIn("api_key", serialized)
            self.assertNotIn("authorization", serialized)
            with self.assertRaisesRegex(RlvrOnlineBindingError, "already finalized"):
                binder(self._event(interaction, interaction_id))

    def test_crossed_pre_batch_interaction_and_secret_runtime_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, interaction, _ = self._stage(root)
            binder = PersistentRlvrV2AgentPreBatchBinder(root / "journal")
            crossed = self._event(interaction, "crossed-interaction")
            with self.assertRaisesRegex(ValueError, "interaction"):
                binder(crossed)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge, _, _, _ = _source(reward=1.0, controller_kind="torch")
            runtime = dict(bridge["policy_binding"]["inference_runtime_contract"])
            runtime["session_api_key"] = "must-never-persist"
            with self.assertRaisesRegex(RlvrOnlineBindingError, "credential field"):
                self._stage(root, runtime_override=runtime)

    def test_pair_index_secret_session_post_batch_and_corrupt_final_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge, interaction, _ = self._stage(root)
            binding = bridge["interaction_adapter_sidecar"]["bindings"][0]
            with self.assertRaisesRegex(
                RlvrOnlineBindingError,
                "pair index differs",
            ):
                self._stage(
                    root,
                    interaction_id_override="another-real-interaction-id",
                )
            binder = PersistentRlvrV2AgentPreBatchBinder(root / "journal")
            post_batch = self._event(interaction, "interactions")
            with self.assertRaisesRegex(ValueError, "post-batch"):
                binder(post_batch)
            secret_session = self._event(interaction, binding["interaction_id"])
            secret_session.session_id = "Bearer must-never-persist"
            with self.assertRaisesRegex(RlvrOnlineBindingError, "credential"):
                binder(secret_session)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("bridges", "runner-admissions"):
                (root / name).mkdir(mode=0o700)
            estimator_path = root / "frozen-estimator-template.json"
            write_m0_rlvr_estimator_template(
                output_path=estimator_path,
                allowed_root=root,
            )
            bridge, interaction, _ = self._stage(root)
            interaction_id = bridge["interaction_adapter_sidecar"]["bindings"][0][
                "interaction_id"
            ]
            environment = {
                "JPH_ROOT": str(root),
                "JPH_AREAL_JOINT_BRIDGE_DIR": str(root / "bridges"),
                "JPH_RLVR_FROZEN_ESTIMATOR_TEMPLATE_PATH": str(estimator_path),
                "JPH_RLVR_RUNNER_ADMISSION_DIR": str(root / "runner-admissions"),
            }
            binder = PersistentRlvrV2AgentPreBatchBinder(root / "journal")
            with patch.dict(os.environ, environment, clear=False), _fake_areal_type_import():
                binder(self._event(interaction, interaction_id))
            finalized = next((root / "journal" / "finalized").glob("*.json"))
            finalized.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RlvrOnlineBindingError, "schema or hash"):
                binder(self._event(interaction, interaction_id))


if __name__ == "__main__":
    unittest.main()
