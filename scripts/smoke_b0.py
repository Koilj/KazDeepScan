from __future__ import annotations

import argparse
import json

import torch

from kds.models import B0LogMelCnn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an untrained B0 forward-pass smoke test.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    device = (
        "cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    model = B0LogMelCnn().to(device).eval()
    waveform = torch.randn((1, 64_600), device=device) * 0.02
    with torch.inference_mode():
        logit = model(waveform)
    if device == "cuda":
        torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "device": device,
                "input_shape": list(waveform.shape),
                "output_shape": list(logit.shape),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
