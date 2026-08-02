import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from jphrl.paths import require_within_configured_root


class PathPolicyTests(unittest.TestCase):
    def test_local_mode_allows_any_path(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                require_within_configured_root("/tmp/example"),
                Path("/tmp/example").resolve(),
            )

    def test_configured_root_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"JPH_ROOT": root}, clear=True):
                accepted = require_within_configured_root(Path(root) / "artifacts" / "x.json")
                self.assertEqual(
                    accepted,
                    (Path(root) / "artifacts" / "x.json").resolve(),
                )
                with self.assertRaises(ValueError):
                    require_within_configured_root("/tmp/escape.json")


if __name__ == "__main__":
    unittest.main()
