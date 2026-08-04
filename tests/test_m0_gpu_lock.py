from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_HELPER = PROJECT_ROOT / "scripts" / "m0_gpu_lock.sh"


@unittest.skipUnless(
    Path("/proc/self/fd").is_dir() and shutil.which("flock") is not None,
    "the production M0 lock contract requires Linux /proc and flock",
)
class M0GpuLockTests(unittest.TestCase):
    def _run(self, script: str, *, root: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["JPH_ROOT"] = str(root)
        env["LOCK_HELPER"] = str(LOCK_HELPER)
        return subprocess.run(
            ["bash", "-c", script],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_inherited_descriptor_is_reused_only_for_the_same_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                r"""
set -euo pipefail
source "${LOCK_HELPER}"
jph_acquire_m0_gpu_lock 2
test "${JPH_M0_GPU_LOCK_FD}" = 9
bash -c '
  set -euo pipefail
  source "${LOCK_HELPER}"
  jph_acquire_m0_gpu_lock 2
  if jph_acquire_m0_gpu_lock 3; then
    exit 91
  fi
'
""",
                root=Path(directory),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("does not match GPU 3", result.stderr)

    def test_independent_process_cannot_take_the_held_gpu_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                r"""
set -euo pipefail
source "${LOCK_HELPER}"
jph_acquire_m0_gpu_lock 5
env -u JPH_M0_GPU_LOCK_FD JPH_ROOT="${JPH_ROOT}" LOCK_HELPER="${LOCK_HELPER}" \
  bash -c '
    set -euo pipefail
    source "${LOCK_HELPER}"
    if jph_acquire_m0_gpu_lock 5; then
      exit 92
    fi
  '
""",
                root=Path(directory),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("reserved by another JPH process", result.stderr)


if __name__ == "__main__":
    unittest.main()
