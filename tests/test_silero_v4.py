from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from kds.data.manifest import ManifestRow, validate_manifest
from kds.data.research_tts import load_research_tts_model_lock
from kds.data.silero_v4 import (
    SILERO_V4_SOURCE_ID,
    SileroV4Error,
    assign_silero_v4_profiles,
    inspect_silero_v4_package,
    load_silero_v4_runtime,
    normalize_silero_v4_text,
    silero_v4_spoof_row,
)
from tests.factories import manifest_mapping


def _runtime_and_model():
    lock = load_research_tts_model_lock(Path("configs/research/silero_v4_cyrillic_v1_models.json"))
    assert len(lock.models) == 1
    return load_silero_v4_runtime(lock.models[0]), lock.models[0]


def _base_row(*, language: str, sample_id: str) -> ManifestRow:
    source_name = "google_fleurs_ru_v1" if language == "ru" else "google_fleurs_kk_v1"
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id=sample_id,
            split="test",
            label="bonafide",
            language=language,
            code_switch="false",
            source_name=source_name,
            source_license="CC-BY-4.0",
            text_id=f"{source_name}:prompt:1",
            text_hash="b" * 64,
        ),
        2,
    )


def test_silero_v4_lock_has_only_fixed_ru_and_kk_profiles() -> None:
    runtime, model = _runtime_and_model()

    assert model.generator_family == "fastpitch_hifigan_torchscript_tts"
    assert runtime.sample_rate == 48_000
    assert [profile.voice_id for profile in runtime.profiles_by_language["ru"]] == ["b_ru"]
    assert [profile.voice_id for profile in runtime.profiles_by_language["kk"]] == [
        "kz_M1",
        "kz_M2",
        "kz_F1",
        "kz_F2",
        "kz_F3",
    ]
    assert "random" not in {
        profile.voice_id
        for profiles in runtime.profiles_by_language.values()
        for profile in profiles
    }
    assert sum(artifact.expected_size_bytes for artifact in model.artifacts) < 2 * 1024**3


def test_silero_v4_text_normalizer_refuses_semantic_loss() -> None:
    assert normalize_silero_v4_text('«Сәлем», мир;') == "сәлем, мир,"

    with pytest.raises(SileroV4Error, match="unsupported characters"):
        normalize_silero_v4_text("Модель 2.0")
    with pytest.raises(SileroV4Error, match="unsupported characters"):
        normalize_silero_v4_text("тест API")


def test_silero_v4_profiles_are_deterministic_and_spoof_row_preserves_text() -> None:
    runtime, model = _runtime_and_model()
    ru_row = _base_row(language="ru", sample_id="google_fleurs_ru_v1:1")
    kk_row = _base_row(language="kk", sample_id="google_fleurs_kk_v1:2")

    assignments = assign_silero_v4_profiles([kk_row, ru_row], runtime)
    by_id = {row.sample_id: profile for row, profile in assignments}
    assert by_id[ru_row.sample_id].voice_id == "b_ru"
    assert by_id[kk_row.sample_id].voice_id == "kz_M1"

    spoof = silero_v4_spoof_row(
        base_row=kk_row,
        model=model,
        profile=by_id[kk_row.sample_id],
        relative_path="raw/fleurs_ru_kk_v1_silero_v4/slices/test/kk.wav",
        sha256="c" * 64,
        duration_s=1.5,
        original_sr=48_000,
        created_at="2026-08-11T00:00:00Z",
        device="local_cpu_silero_v4_fastpitch_hifigan",
    )
    validate_manifest([spoof])
    assert spoof.source_name == SILERO_V4_SOURCE_ID
    assert spoof.text_hash == kk_row.text_hash
    assert spoof.code_switch == "false"
    assert spoof.clone_consent_id == "not_applicable:fixed-pretrained-tts-no-reference-audio"


def test_silero_v4_zip_inspection_rejects_unsafe_paths(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.pt"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("v4_cyrillic/../../outside", b"unsafe")

    with pytest.raises(SileroV4Error, match="Unsafe"):
        inspect_silero_v4_package(archive)
