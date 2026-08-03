from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from jphrl.trajectory.rlvr_workflow_admission import (
    frozen_dual_credit_estimator_template_from_record,
)
from scripts.write_m0_rlvr_estimator_template import (
    write_m0_rlvr_estimator_template,
)


class WriteM0RlvrEstimatorTemplateTests(unittest.TestCase):
    def test_private_exact_preregistered_template_and_o_excl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "runtime" / "template.json"
            record = write_m0_rlvr_estimator_template(
                output_path=output,
                allowed_root=root,
            )
            loaded = json.loads(output.read_text(encoding="utf-8"))
            template = frozen_dual_credit_estimator_template_from_record(loaded)
            self.assertEqual(loaded, record)
            self.assertEqual(template.policy_baseline, 0.5)
            self.assertEqual(template.harness_baseline, 0.5)
            self.assertNotEqual(template.policy_source, template.harness_source)
            self.assertNotEqual(
                template.policy_baseline_snapshot_id,
                template.harness_baseline_snapshot_id,
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(output.parent.stat().st_mode & 0o777, 0o700)
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_m0_rlvr_estimator_template(
                    output_path=output,
                    allowed_root=root,
                )

    def test_output_must_remain_inside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root.parent / f"{root.name}-outside-template.json"
            try:
                with self.assertRaisesRegex(ValueError, "escapes"):
                    write_m0_rlvr_estimator_template(
                        output_path=outside,
                        allowed_root=root,
                    )
            finally:
                if outside.exists():
                    os.unlink(outside)


if __name__ == "__main__":
    unittest.main()
