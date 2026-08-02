from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from jphrl.paths import require_within_configured_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and cache one pinned HF model snapshot")
    parser.add_argument("model")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
        )
    ).resolve()
    require_within_configured_root(snapshot)
    commit = snapshot.name
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"Hugging Face snapshot did not resolve to a commit hash: {snapshot}")

    report = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_commit": commit,
        "snapshot_path": str(snapshot),
    }
    destination = require_within_configured_root(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
