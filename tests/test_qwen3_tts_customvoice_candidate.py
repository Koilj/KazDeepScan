from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kds.data.manifest import ManifestRow, validate_manifest
from kds.data.qwen3_tts_customvoice import Qwen3TtsCustomVoice
from kds.data.qwen3_tts_customvoice_candidate import (
    QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID,
    Qwen3TtsCustomVoiceCandidateError,
    qwen3_tts_customvoice_spoof_row,
)
from kds.data.research_tts import load_research_tts_model_lock


def _base(text: str) -> ManifestRow:
    return ManifestRow(
        sample_id="voxforge_ru_mdc_2026_05:submission:test:prompt:ru_0001",
        relative_path="processed/base.wav",
        sha256="a" * 64,
        split="test",
        label="bonafide",
        language="ru",
        code_switch="unknown",
        parent_group_id="voxforge_ru_mdc_2026_05:contributor:test",
        source_name="voxforge_ru_mdc_2026_05",
        source_license="GPL-3.0-or-later",
        rights_basis="Pinned VoxForge source",
        speaker_pseudo_id="voxforge_ru_mdc_2026_05:contributor:test",
        text_id="ru_0001",
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="voxforge_submission_read_speech",
        original_sr=48_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-13T18:13:54Z",
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


def _model():
    return load_research_tts_model_lock(
        Path("configs/research/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1_models.json")
    ).models[0]


def test_spoof_row_preserves_frozen_literal_text_and_fixed_aiden_route() -> None:
    text = "Точный исходный текст."
    base = _base(text)
    runtime = _runtime()
    row = qwen3_tts_customvoice_spoof_row(
        base_row=base,
        model=_model(),
        runtime=runtime,
        prepared=runtime.prepare_text(text),
        relative_path="raw/qwen/test.wav",
        sha256="b" * 64,
        duration_s=2.0,
        created_at="2026-08-13T20:00:00Z",
    )

    validate_manifest([row])
    assert row.source_name == QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
    assert row.text_id == base.text_id
    assert row.text_hash == base.text_hash
    assert row.voice_id == "qwen3_tts_customvoice:aiden"
    assert row.original_sr == 24_000
    assert "reference_audio=forbidden" in row.augmentation_chain


def test_spoof_row_rejects_text_or_seed_change() -> None:
    base = _base("Точный исходный текст.")
    runtime = _runtime()

    with pytest.raises(Qwen3TtsCustomVoiceCandidateError, match="literal text or seed"):
        qwen3_tts_customvoice_spoof_row(
            base_row=base,
            model=_model(),
            runtime=runtime,
            prepared=runtime.prepare_text("Другой текст."),
            relative_path="raw/qwen/test.wav",
            sha256="b" * 64,
            duration_s=2.0,
            created_at="2026-08-13T20:00:00Z",
        )
