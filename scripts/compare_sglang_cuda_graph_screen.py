from __future__ import annotations

import argparse
import json
from pathlib import Path

from jphrl.experiments.sglang_cuda_graph_screen import (
    compare_cuda_graph_screen_runs,
    write_screen_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired C2a/C2b SGLang CUDA Graph screen cells"
    )
    parser.add_argument("c2a_root", type=Path)
    parser.add_argument("c2b_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_cuda_graph_screen_runs(args.c2a_root, args.c2b_root)
    write_screen_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    if not report["summary"]["mechanism_supported"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
