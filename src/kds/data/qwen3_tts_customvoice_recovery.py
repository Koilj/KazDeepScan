"""Absolute-output adapter for a new, separately authorized Qwen recovery route.

The historical Qwen wrapper is deliberately left untouched because it is bound to
the failed v4-final attempt.  This small adapter reuses its model/runtime checks
but resolves the WAV target before CrispASR is started from its own working
directory.  It is only usable from a new recovery contract that pins this file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]

from kds.data import qwen3_tts_customvoice as historical
from kds.data.research_tts import ResearchTtsModel


def load_recovery_qwen3_tts_customvoice(
    model_root: Path, model: ResearchTtsModel
) -> historical.Qwen3TtsCustomVoice:
    """Load the exact historical route; the output-path fix is applied only at run time."""

    return historical.load_qwen3_tts_customvoice(model_root, model)


def synthesize_to_absolute_file(
    runtime: historical.Qwen3TtsCustomVoice,
    prepared: historical.Qwen3TtsCustomVoiceText,
    output_path: Path,
) -> None:
    """Generate one checked WAV, passing CrispASR a project-absolute output path."""

    resolved = output_path.resolve(strict=False)
    if resolved.exists() or not resolved.parent.is_dir():
        raise historical.Qwen3TtsCustomVoiceError(
            "Recovery Qwen output must be a new file under an existing directory."
        )
    completed = subprocess.run(
        runtime.command_for(prepared, resolved),
        cwd=runtime.executable.parent,
        env=historical._cuda_environment(runtime.cuda_library_dirs),
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-1200:]
        raise historical.Qwen3TtsCustomVoiceError(
            f"Pinned recovery Qwen runtime failed with exit code {completed.returncode}: {stderr}"
        )
    if not resolved.is_file():
        raise historical.Qwen3TtsCustomVoiceError(
            "Pinned recovery Qwen runtime produced no WAV output."
        )
    try:
        info = sf.info(resolved)
    except RuntimeError as error:
        raise historical.Qwen3TtsCustomVoiceError(
            "Pinned recovery Qwen output is not readable audio."
        ) from error
    if info.samplerate != runtime.sample_rate or info.channels != 1 or info.frames <= 0:
        raise historical.Qwen3TtsCustomVoiceError(
            "Pinned recovery Qwen output must be non-empty 24 kHz mono audio."
        )
