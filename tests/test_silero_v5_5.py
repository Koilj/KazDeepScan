from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from kds.data.manifest import ManifestRow, validate_manifest
from kds.data.research_tts import ResearchTtsModel, load_research_tts_model_lock
from kds.data.silero_v5_5 import (
    SILERO_V5_5_FIXED_SPEAKER,
    SILERO_V5_5_SOURCE_ID,
    SileroV55Error,
    SileroV55Runtime,
    inspect_silero_v5_5_package,
    load_silero_v5_5_runtime,
    normalize_silero_v5_5_text,
    silero_v5_5_spoof_row,
    synthesize_silero_v5_5,
)
from tests.factories import manifest_mapping


def _runtime_and_model() -> tuple[SileroV55Runtime, ResearchTtsModel]:
    lock = load_research_tts_model_lock(
        Path("configs/research/silero_v5_5_ru_eugene_v1_models.json")
    )
    assert len(lock.models) == 1
    return load_silero_v5_5_runtime(lock.models[0]), lock.models[0]


def _base_row() -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="common_voice_ru_v24:1",
            split="test",
            label="bonafide",
            language="ru",
            code_switch="unknown",
            source_name="common_voice_ru_v24",
            source_license="CC0-1.0",
            text_id="sentence-1",
            text_hash="b" * 64,
        ),
        2,
    )


def test_silero_v5_5_lock_is_fixed_to_text_only_eugene() -> None:
    runtime, model = _runtime_and_model()

    assert runtime.sample_rate == 48_000
    assert runtime.fixed_speaker == SILERO_V5_5_FIXED_SPEAKER
    assert model.generator_family == "silero_v5_5_ru_torchpackage_tts"
    assert sum(artifact.expected_size_bytes for artifact in model.artifacts) < 2 * 1024**3


def test_silero_v5_5_text_contract_rejects_control_and_lexical_loss() -> None:
    assert normalize_silero_v5_5_text("  Ёж — это… тест;  ") == "Ёж — это… тест;"

    with pytest.raises(SileroV55Error, match="unsupported characters"):
        normalize_silero_v5_5_text("Модель 5.5")
    with pytest.raises(SileroV55Error, match="unsupported characters"):
        normalize_silero_v5_5_text("тест +акцент")


def test_silero_v5_5_spoof_row_preserves_common_voice_text() -> None:
    _runtime, model = _runtime_and_model()
    base = _base_row()

    spoof = silero_v5_5_spoof_row(
        base_row=base,
        model=model,
        relative_path="raw/silero_v5_5_ru_eugene_v1/slices/test/s.wav",
        sha256="c" * 64,
        duration_s=1.5,
        original_sr=48_000,
        created_at="2026-08-13T00:00:00Z",
        device="local_cpu_silero_v5_5_ru_eugene",
    )

    validate_manifest([spoof])
    assert spoof.source_name == SILERO_V5_5_SOURCE_ID
    assert spoof.text_hash == base.text_hash
    assert spoof.code_switch == "unknown"
    assert spoof.voice_id.endswith(":eugene")
    assert spoof.clone_consent_id == "not_applicable:fixed-pretrained-tts-no-reference-audio"


def test_silero_v5_5_synthesis_requires_wav_output_suffix(tmp_path: Path) -> None:
    runtime, _model = _runtime_and_model()

    with pytest.raises(SileroV55Error, match=".wav filename"):
        synthesize_silero_v5_5(
            model=object(),
            text="Точный текст",
            runtime=runtime,
            output=tmp_path / "invalid.wav.part",
        )


def test_silero_v5_5_zip_inspection_rejects_unsafe_paths(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.pt"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("questions_public_eugene_int_intensity/../../outside", b"unsafe")

    with pytest.raises(SileroV55Error, match="Unsafe"):
        inspect_silero_v5_5_package(archive)
