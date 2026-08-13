from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.qwen3_tts_customvoice import (
    Qwen3TtsCustomVoice,
    Qwen3TtsCustomVoiceError,
    _contains_customvoice_backend,
)


def _runtime() -> Qwen3TtsCustomVoice:
    return Qwen3TtsCustomVoice(
        executable=Path("/models/crispasr"),
        talker_path=Path("/models/talker.gguf"),
        codec_path=Path("/models/codec.gguf"),
        cuda_library_dirs=(Path("/venv/cudart"), Path("/venv/cublas")),
        fixed_speaker_name="aiden",
        sample_rate=24_000,
        target_language="ru",
        temperature=0.9,
        max_new_tokens=512,
    )


def test_prepare_text_preserves_literal_source_text_and_derives_stable_seed() -> None:
    runtime = _runtime()

    first = runtime.prepare_text("  Привет\nмир!  ")
    second = runtime.prepare_text("  Привет\nмир!  ")

    assert first.source_text == "  Привет\nмир!  "
    assert first.seed == second.seed


def test_prepare_text_rejects_empty_or_nul_text() -> None:
    runtime = _runtime()

    with pytest.raises(Qwen3TtsCustomVoiceError, match="non-empty"):
        runtime.prepare_text("  \n")
    with pytest.raises(Qwen3TtsCustomVoiceError, match="NUL"):
        runtime.prepare_text("привет\x00мир")


def test_command_locks_customvoice_without_reference_or_instruction(tmp_path: Path) -> None:
    runtime = _runtime()
    command = runtime.command_for(runtime.prepare_text("Привет, мир."), tmp_path / "out.wav")

    assert command == (
        "/models/crispasr",
        "--backend",
        "qwen3-tts-customvoice",
        "--model",
        "/models/talker.gguf",
        "--codec-model",
        "/models/codec.gguf",
        "--voice",
        "aiden",
        "--target-lang",
        "ru",
        "--seed",
        str(runtime.prepare_text("Привет, мир.").seed),
        "--temperature",
        "0.9",
        "--max-new-tokens",
        "512",
        "--tts",
        "Привет, мир.",
        "--tts-output",
        str(tmp_path / "out.wav"),
        "--gpu-backend",
        "cuda",
        "--no-prints",
    )
    forbidden = {"--ref-text", "--instruct", "--auto-download", "--voice-dir"}
    assert forbidden.isdisjoint(command)


def test_command_refuses_non_wav_output(tmp_path: Path) -> None:
    with pytest.raises(Qwen3TtsCustomVoiceError, match=".wav"):
        _runtime().command_for(_runtime().prepare_text("Привет"), tmp_path / "out.flac")


def test_backend_inventory_requires_the_object_returned_by_crispasr() -> None:
    assert _contains_customvoice_backend(
        {"backends": [{"name": "qwen3-tts-customvoice"}]}
    )
    assert not _contains_customvoice_backend([{"name": "qwen3-tts-customvoice"}])
    assert not _contains_customvoice_backend({"backends": [{"name": "whisper"}]})
