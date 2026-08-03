from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.audit_gpu_memory_envelope import audit_gpu_memory_envelope


class GpuMemoryEnvelopeTests(unittest.TestCase):
    def test_passes_at_limit_and_persists_observation_only_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "gpu-memory.csv"
            output = root / "gpu-memory-audit.json"
            samples.write_text("1,100,80000\n2,26724,53376\n", encoding="utf-8")
            record = audit_gpu_memory_envelope(
                samples_path=samples,
                output_path=output,
                physical_gpu_id=0,
                baseline_used_mib=100,
                max_new_memory_mib=26 * 1024,
                run_kind="m0-torch-joint-v1",
                project_commit="a" * 40,
            )
            self.assertTrue(record["passed"])
            self.assertEqual(record["measured_new_memory_mib"], 26 * 1024)
            self.assertFalse(
                record["evidence_scope"]["policy_optimizer_update"]  # type: ignore[index]
            )
            self.assertEqual(json.loads(output.read_text()), record)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_exceeded_envelope_writes_failed_audit_then_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "gpu-memory.csv"
            output = root / "gpu-memory-audit.json"
            samples.write_text("1,100,80000\n2,26725,53375\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                audit_gpu_memory_envelope(
                    samples_path=samples,
                    output_path=output,
                    physical_gpu_id=0,
                    baseline_used_mib=100,
                    max_new_memory_mib=26 * 1024,
                    run_kind="m0-torch-joint-v1",
                    project_commit="a" * 40,
                )
            self.assertFalse(json.loads(output.read_text())["passed"])

    def test_empty_malformed_cross_directory_and_existing_output_fail(self) -> None:
        for payload in ("", "1,ERROR,2\n", "2,1,2\n1,1,2\n"):
            with (
                self.subTest(payload=payload),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                samples = root / "gpu-memory.csv"
                output = root / "gpu-memory-audit.json"
                samples.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit_gpu_memory_envelope(
                        samples_path=samples,
                        output_path=output,
                        physical_gpu_id=0,
                        baseline_used_mib=0,
                        max_new_memory_mib=1,
                        run_kind="m0",
                        project_commit="a" * 40,
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "gpu-memory.csv"
            samples.write_text("1,1,2\n", encoding="utf-8")
            other = root / "other"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "beside"):
                audit_gpu_memory_envelope(
                    samples_path=samples,
                    output_path=other / "audit.json",
                    physical_gpu_id=0,
                    baseline_used_mib=0,
                    max_new_memory_mib=1,
                    run_kind="m0",
                    project_commit="a" * 40,
                )
            output = root / "audit.json"
            output.write_text("already exists", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                audit_gpu_memory_envelope(
                    samples_path=samples,
                    output_path=output,
                    physical_gpu_id=0,
                    baseline_used_mib=0,
                    max_new_memory_mib=1,
                    run_kind="m0",
                    project_commit="a" * 40,
                )


if __name__ == "__main__":
    unittest.main()
