import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jphrl.paths import (
    repository_root,
    require_outside_repository,
    require_within_configured_root,
)


class PathPolicyTests(unittest.TestCase):
    def test_local_mode_allows_any_path(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                require_within_configured_root("/tmp/example"),
                Path("/tmp/example").resolve(),
            )

    def test_configured_root_rejects_escape(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root,
            mock.patch.dict(os.environ, {"JPH_ROOT": root}, clear=True),
        ):
            accepted = require_within_configured_root(
                Path(root) / "artifacts" / "x.json"
            )
            self.assertEqual(
                accepted,
                (Path(root) / "artifacts" / "x.json").resolve(),
            )
            with self.assertRaises(ValueError):
                require_within_configured_root("/tmp/escape.json")

    def test_runtime_artifacts_use_the_actual_git_checkout_boundary(self) -> None:
        checkout = repository_root()
        self.assertTrue((checkout / ".git").exists())
        with self.assertRaisesRegex(ValueError, "outside Git checkout"):
            require_outside_repository(checkout / "artifacts" / "forbidden.json")
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                require_outside_repository(Path(temporary) / "allowed.json"),
                (Path(temporary) / "allowed.json").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
