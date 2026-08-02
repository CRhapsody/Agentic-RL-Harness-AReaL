from __future__ import annotations

import argparse
import json
from pathlib import Path

from jphrl.experiments.sglang_logprob_screen import (
    compare_screen_runs,
    write_screen_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the paired C0/C1 SGLang generation log-prob screen"
    )
    parser.add_argument("c0_root", type=Path)
    parser.add_argument("c1_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_screen_runs(args.c0_root, args.c1_root)
    write_screen_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    if not report["summary"]["mechanism_supported"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
