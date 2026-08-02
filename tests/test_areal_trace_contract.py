import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jphrl.trajectory.areal_trace_contract import (
    ArealTraceContractError,
    build_areal_trace_record,
    record_sha256,
    validate_areal_trace_record,
    write_areal_trace_record,
)


class ArealTraceContractTests(unittest.TestCase):
    def _record(self):
        response = SimpleNamespace(
            input_tokens=[10, 11],
            output_tokens=[12, 13],
            output_logprobs=[-0.123456789, -1.25],
            output_versions=[0, 0],
            stop_reason="stop",
        )
        interaction = SimpleNamespace(
            reward=1.0,
            original_reward=1.0,
            chat_template_type="hf",
        )
        tensor_dict = {
            "input_ids": [[10, 11, 12, 13]],
            "loss_mask": [[0, 0, 1, 1]],
            "logprobs": [[0.0, 0.0, -0.12345679104328156, -1.25]],
            "versions": [[-1, -1, 0, 0]],
            "attention_mask": [[True, True, True, True]],
            "rewards": [1.0],
            "original_rewards": [1.0],
        }
        return build_areal_trace_record(
            task_id=7,
            request_id="abcd1234",
            model_response=response,
            interaction=interaction,
            tensor_dict=tensor_dict,
            areal_commit="a" * 40,
            behavior_snapshot_path="/configured/models/snapshot",
            behavior_revision="b" * 40,
        )

    def test_model_response_roundtrip_accepts_only_float32_sized_drift(self) -> None:
        report = validate_areal_trace_record(
            self._record(), expected_policy_version=0
        )
        self.assertTrue(report["ok"])
        self.assertLess(report["roundtrip_logprob_max_abs_error"], 1e-6)
        self.assertEqual(report["prompt_tokens"], 2)
        self.assertEqual(report["generated_tokens"], 2)

    def test_prompt_and_output_version_contract_fails_closed(self) -> None:
        record = self._record()
        record["tensor_dict"]["versions"][0][0] = 0
        record["record_sha256"] = record_sha256(record)
        with self.assertRaisesRegex(ArealTraceContractError, "prompt=-1"):
            validate_areal_trace_record(record)

    def test_model_response_and_tensor_logprobs_must_align(self) -> None:
        record = self._record()
        record["tensor_dict"]["logprobs"][0][-1] = -2.0
        record["record_sha256"] = record_sha256(record)
        with self.assertRaisesRegex(ArealTraceContractError, "float32 tolerance"):
            validate_areal_trace_record(record)

    def test_trace_write_is_unique_and_stays_under_root(self) -> None:
        record = self._record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_areal_trace_record(
                record, trace_dir=root / "artifacts" / "trace", allowed_root=root
            )
            self.assertEqual(json.loads(path.read_text()), record)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_areal_trace_record(
                    record,
                    trace_dir=root / "artifacts" / "trace",
                    allowed_root=root,
                )
            with self.assertRaisesRegex(ArealTraceContractError, "escapes"):
                write_areal_trace_record(
                    record,
                    trace_dir=root.parent / "outside",
                    allowed_root=root,
                )


if __name__ == "__main__":
    unittest.main()
