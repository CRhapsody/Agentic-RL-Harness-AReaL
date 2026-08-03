from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_m0_live_joint import (
    _load_record,
    _record_sha256,
    verify_m0_live_joint,
)


class VerifyM0LiveJointTests(unittest.TestCase):
    def test_private_self_hashed_record_loads_and_secret_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.json"
            record: dict[str, object] = {
                "schema_version": "test",
                "value": 7,
            }
            record["record_sha256"] = _record_sha256(record)
            path.write_text(
                json.dumps(record, allow_nan=False, sort_keys=True),
                encoding="utf-8",
            )
            path.chmod(0o600)
            self.assertEqual(_load_record(path, label="test"), record)

            secret: dict[str, object] = {
                "schema_version": "test",
                "session_api_key": "must-not-persist",
            }
            secret["record_sha256"] = _record_sha256(secret)
            path.write_text(json.dumps(secret, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "credential field"):
                _load_record(path, label="test")

    def test_incomplete_run_fails_before_granting_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "selection"):
                verify_m0_live_joint(
                    run_root=root,
                    expected_project_commit="a" * 40,
                )


if __name__ == "__main__":
    unittest.main()
