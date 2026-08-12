"""Shared local inference primitives for the pinned KazakhTTS route."""

from __future__ import annotations

import importlib.metadata
import os
import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from packaging.version import Version

from kds.data.kazakhtts import ExtractedKazakhTtsRuntime, KazakhTtsRuntime
from kds.data.research_tts import ResearchTtsError


def resolve_kazakhtts_device(value: str) -> torch.device:
    resolved = ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ResearchTtsError("KazakhTTS CUDA was requested but is unavailable.")
    return device


def _disable_unused_english_g2p_download_route() -> None:
    """ESPnet imports g2p_en eagerly even though this char model declares g2p=null."""

    module = types.ModuleType("g2p_en")

    class DisabledG2p:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("g2p_en is disabled for the pinned KazakhTTS char model.")

    module.G2p = DisabledG2p  # type: ignore[attr-defined]
    sys.modules["g2p_en"] = module


def load_kazakhtts_models(
    runtime: KazakhTtsRuntime,
    extracted: ExtractedKazakhTtsRuntime,
    device: torch.device,
) -> tuple[Any, Any]:
    """Load exact acoustic/vocoder checkpoints with the audited compatibility shims."""

    if Version(torch.__version__.split("+", maxsplit=1)[0]) < Version("2.6"):
        raise ResearchTtsError("KazakhTTS requires torch>=2.6 for weights-only load by default.")
    for distribution, expected in (
        ("espnet", runtime.espnet_version),
        ("parallel-wavegan", runtime.parallel_wavegan_version),
    ):
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise ResearchTtsError(
                f"KazakhTTS runtime needs {distribution}=={expected}, got {actual}."
            )
    _disable_unused_english_g2p_download_route()
    import scipy.signal as signal  # type: ignore[import-untyped]

    signal.kaiser = signal.windows.kaiser
    from espnet2.bin.tts_inference import Text2Speech  # type: ignore[import-untyped]
    from parallel_wavegan.utils import load_model  # type: ignore[import-untyped]

    previous_directory = Path.cwd()
    try:
        os.chdir(extracted.acoustic_config.parents[2])
        text_to_speech = Text2Speech(
            extracted.acoustic_config,
            extracted.acoustic_checkpoint,
            device=str(device),
            threshold=runtime.threshold,
            minlenratio=runtime.min_length_ratio,
            maxlenratio=runtime.max_length_ratio,
            use_att_constraint=runtime.use_attention_constraint,
            backward_window=runtime.backward_window,
            forward_window=runtime.forward_window,
            speed_control_alpha=runtime.speed_control_alpha,
        )
    finally:
        os.chdir(previous_directory)
    text_to_speech.spc2wav = None
    try:
        vocoder_config_value: object = yaml.safe_load(
            extracted.vocoder_config.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ResearchTtsError(f"Cannot safely read KazakhTTS vocoder config: {error}") from error
    if not isinstance(vocoder_config_value, dict):
        raise ResearchTtsError("KazakhTTS vocoder config must be a mapping.")
    vocoder = load_model(
        str(extracted.vocoder_checkpoint),
        config=cast(dict[str, object], vocoder_config_value),
    )
    vocoder = vocoder.to(device).eval()
    vocoder.remove_weight_norm()
    return text_to_speech, vocoder


def synthesize_kazakhtts_waveform(text_to_speech: Any, vocoder: Any, text: str) -> np.ndarray:
    """Synthesize one finite mono waveform without writing or normalizing an asset."""

    with torch.inference_mode():
        result = text_to_speech(text)
        features = result["feat_gen"]
        waveform = vocoder.inference(features).reshape(-1).float().cpu().numpy()
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ResearchTtsError("KazakhTTS produced invalid waveform samples.")
    return cast(np.ndarray, waveform)
