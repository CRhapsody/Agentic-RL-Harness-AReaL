from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jphrl.paths import require_within_configured_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load one pinned local model snapshot and run a short CUDA generation"
    )
    parser.add_argument("--snapshot-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot_report = require_within_configured_root(args.snapshot_report)
    metadata = json.loads(snapshot_report.read_text(encoding="utf-8"))
    snapshot = require_within_configured_root(metadata["snapshot_path"])
    if not snapshot.is_dir():
        raise RuntimeError(f"Model snapshot directory is missing: {snapshot}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model snapshot smoke")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).eval().to("cuda:0")
    encoded = tokenizer("Compute 17 + 25.", return_tensors="pt").to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    prompt_tokens = encoded["input_ids"].shape[-1]
    completion = generated[0, prompt_tokens:].detach().cpu().tolist()
    report = {
        "ok": True,
        "model": metadata["model"],
        "resolved_commit": metadata["resolved_commit"],
        "snapshot_path": str(snapshot),
        "torch_dtype": str(next(model.parameters()).dtype),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        "prompt_token_count": prompt_tokens,
        "completion_token_ids": completion,
        "completion_text": tokenizer.decode(completion),
    }
    destination = require_within_configured_root(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
