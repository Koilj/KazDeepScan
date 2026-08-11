from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kds.models import XlsrSlsClassifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run XLS-R + SLS forward smoke test from local weights."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    arguments = parser.parse_args()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if not arguments.model_dir.is_dir():
        raise ValueError(f"Local XLS-R directory does not exist: {arguments.model_dir}")
    if arguments.precision == "bf16" and arguments.device != "cuda":
        raise ValueError("bf16 smoke test requires CUDA in this project.")

    dtype = torch.bfloat16 if arguments.precision == "bf16" else torch.float32
    model = (
        XlsrSlsClassifier.from_pretrained(str(arguments.model_dir))
        .to(arguments.device, dtype=dtype)
        .eval()
    )
    input_values = torch.randn((1, 64_600), device=arguments.device, dtype=dtype) * 0.02
    attention_mask = torch.ones_like(input_values, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_values, attention_mask)
    if arguments.device == "cuda":
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "device": arguments.device,
                "precision": arguments.precision,
                "input_shape": list(input_values.shape),
                "output_shape": list(logits.shape),
                "finite": bool(torch.isfinite(logits).all()),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
