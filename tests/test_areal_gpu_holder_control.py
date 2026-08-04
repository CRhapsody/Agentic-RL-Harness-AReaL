from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import stop_areal_gpu_holder


RUN_ID = "20260804T010203Z-gpu-holder-0123456789abcdef"


class ArealGpuHolderControlTests(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(jph_root="/unused", run_id=RUN_ID, reason="manual")

    @staticmethod
    def _control() -> dict[str, object]:
        return {
            "launcher_pid": 43210,
            "launcher_start_time": 987654,
        }

    def test_stop_signals_only_the_exact_control_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            with (
                mock.patch.object(
                    stop_areal_gpu_holder,
                    "_load_control",
                    return_value=(run_root, self._control()),
                ),
                mock.patch.object(stop_areal_gpu_holder, "_validate_process") as validate,
                mock.patch.object(stop_areal_gpu_holder.os, "kill") as kill,
            ):
                stop_areal_gpu_holder.request_stop(self._args())

            validate.assert_called_once_with(43210, 987654)
            kill.assert_called_once_with(43210, signal.SIGTERM)
            request = json.loads(
                (run_root / "stop.requested.json").read_text(encoding="utf-8")
            )
            self.assertEqual(request["launcher_pid"], 43210)
            self.assertEqual(request["launcher_start_time"], 987654)
            self.assertEqual(request["requested_by_uid"], os.getuid())

    def test_start_time_recheck_failure_never_signals_or_leaves_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            with (
                mock.patch.object(
                    stop_areal_gpu_holder,
                    "_load_control",
                    return_value=(run_root, self._control()),
                ),
                mock.patch.object(
                    stop_areal_gpu_holder,
                    "_validate_process",
                    side_effect=ValueError("PID 43210 start time no longer matches"),
                ),
                mock.patch.object(stop_areal_gpu_holder.os, "kill") as kill,
            ):
                with self.assertRaisesRegex(ValueError, "start time no longer matches"):
                    stop_areal_gpu_holder.request_stop(self._args())

            kill.assert_not_called()
            self.assertFalse((run_root / "stop.requested.json").exists())


if __name__ == "__main__":
    unittest.main()
