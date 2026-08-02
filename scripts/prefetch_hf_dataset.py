from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download

from jphrl.paths import require_within_configured_root


def _commit_from_snapshot(snapshot: Path) -> str:
    commit = snapshot.name
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"Hugging Face snapshot did not resolve to a commit hash: {snapshot}")
    return commit


def _split_report(dataset: DatasetDict) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, split in dataset.items():
        cache_files = []
        for entry in split.cache_files:
            path = require_within_configured_root(entry["filename"])
            cache_files.append(str(path))
        report[name] = {
            "num_rows": split.num_rows,
            "columns": list(split.column_names),
            "cache_files": cache_files,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve, cache, materialize, and report one pinned HF dataset"
    )
    parser.add_argument("dataset")
    parser.add_argument("--config")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = Path(
        snapshot_download(
            repo_id=args.dataset,
            repo_type="dataset",
            revision=args.revision,
        )
    ).resolve()
    require_within_configured_root(snapshot)
    commit = _commit_from_snapshot(snapshot)

    dataset = load_dataset(
        args.dataset,
        args.config,
        revision=commit,
    )
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError("Expected load_dataset without split= to return DatasetDict")

    report = {
        "dataset": args.dataset,
        "config": args.config,
        "requested_revision": args.revision,
        "resolved_commit": commit,
        "snapshot_path": str(snapshot),
        "splits": _split_report(dataset),
    }
    destination = require_within_configured_root(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
