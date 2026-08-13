"""Narrow text-only contract for a future pinned VoxCPM2 smoke/synthesis process."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

VOXCPM2_FIXED_SEED = 20_260_814


class VoxCPM2TextOnlyError(ValueError):
    """Raised when an input attempts to leave the frozen text-only route."""


class VoxCPM2GenerateProtocol(Protocol):
    def generate(self, **kwargs: object) -> Any: ...


@dataclass(frozen=True, slots=True)
class BoundText:
    literal_sha256: str
    collapse_whitespace_sha256: str


def collapse_whitespace(text: str) -> str:
    """Apply the only admitted deterministic text transform before upstream code."""

    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise VoxCPM2TextOnlyError("Text must be a non-empty NUL-free string.")
    return " ".join(text.split())


def bind_text(text: str) -> BoundText:
    canonical = collapse_whitespace(text)
    return BoundText(
        literal_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        collapse_whitespace_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def offline_environment() -> Mapping[str, str]:
    """Return environment flags required in addition to an outer network block."""

    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
    }


def local_model_load_kwargs(model_root: Path) -> dict[str, object]:
    """Expose only the pinned local, no-denoiser, no-LoRA load route."""

    root = model_root.resolve(strict=True)
    if not root.is_dir():
        raise VoxCPM2TextOnlyError(f"Model root is not a directory: {root}")
    return {
        "hf_model_id": str(root),
        "load_denoiser": False,
        "zipenhancer_model_id": None,
        "local_files_only": True,
        "optimize": False,
        "device": "cuda",
        "lora_config": None,
        "lora_weights_path": None,
    }


def generation_kwargs(text: str, expected: BoundText) -> dict[str, object]:
    """Bind literal/canonical text and return the complete one-attempt parameter set."""

    actual = bind_text(text)
    if actual != expected:
        raise VoxCPM2TextOnlyError("Literal or collapse-whitespace text binding mismatch.")
    return {
        "text": collapse_whitespace(text),
        "prompt_wav_path": None,
        "prompt_text": None,
        "reference_wav_path": None,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "min_len": 2,
        "max_len": 4096,
        "normalize": False,
        "denoise": False,
        "retry_badcase": False,
        "retry_badcase_max_times": 1,
        "streaming": False,
        "seed": VOXCPM2_FIXED_SEED,
    }


def synthesize_text_only(
    model: VoxCPM2GenerateProtocol, text: str, expected: BoundText
) -> Any:
    """Call a supplied already-loaded model through the frozen text-only surface."""

    return model.generate(**generation_kwargs(text, expected))
