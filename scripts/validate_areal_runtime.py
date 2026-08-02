from __future__ import annotations

import json
import os
import sys

import areal
import flash_attn
from flash_attn import flash_attn_func
import sglang
import torch
import transformers


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the pinned AReaL environment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Flash Attention smoke must expose exactly one preselected idle GPU")

    query = torch.randn((1, 8, 1, 64), device="cuda", dtype=torch.bfloat16)
    output = flash_attn_func(query, query, query, dropout_p=0.0, causal=False)
    torch.cuda.synchronize()
    if output.shape != query.shape or not torch.isfinite(output).all().item():
        raise RuntimeError("Flash Attention kernel returned an invalid result")

    report = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "sglang": sglang.__version__,
        "flash_attn": flash_attn.__version__,
        "areal": areal.__file__,
        "physical_gpu_id": os.environ.get("JPH_PHYSICAL_GPU_ID"),
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "visible_gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "flash_attention_output_shape": list(output.shape),
        "flash_attention_finite": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
