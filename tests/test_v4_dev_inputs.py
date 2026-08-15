from __future__ import annotations

from pathlib import Path

from kds.data.manifest import ManifestRow, validate_manifest
from kds.data.research_tts import load_research_tts_model_lock
from kds.data.silero_v4 import load_silero_v4_runtime
from kds.data.v4_dev_inputs import (
    V4_KK_DEV_SILERO_SOURCE_ID,
    build_v4_combined_dev_manifest,
    freeze_v4_kk_dev_pairs,
    v4_kk_dev_silero_spoof_row,
)
from tests.factories import manifest_mapping


def _ksc_row() -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="ksc_slr102:dev-1",
            relative_path="processed/source.wav",
            sha256="a" * 64,
            split="dev",
            label="bonafide",
            language="kk",
            code_switch="unknown",
            source_name="ksc_slr102",
            source_license="CC-BY-4.0",
            text_id="ksc_slr102:dev-1",
            text_hash="b" * 64,
        ),
        2,
    )


def _pyara_row() -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="pyara_ru_v7:dev-1",
            relative_path="processed/pyara.wav",
            sha256="d" * 64,
            split="dev",
            label="bonafide",
            language="ru",
            source_name="pyara_ru_v7",
            source_license="CC-BY-NC-SA-4.0",
            text_id="pyara_ru_v7:dev-1",
            text_hash="e" * 64,
        ),
        2,
    )


def test_v4_kk_dev_silero_pair_retains_kk_dev_role() -> None:
    source = _ksc_row()
    model = load_research_tts_model_lock(
        Path("configs/research/silero_v4_cyrillic_v1_models.json")
    ).models[0]
    profile = load_silero_v4_runtime(model).profiles_by_language["kk"][0]
    spoof = v4_kk_dev_silero_spoof_row(
        base_row=source,
        model=model,
        profile=profile,
        relative_path="processed/spoof.wav",
        sha256="c" * 64,
        duration_s=1.0,
        original_sr=16_000,
        created_at="2026-08-15T00:00:00Z",
        device="local_cuda_silero_v4_fastpitch_hifigan",
    )

    frozen = freeze_v4_kk_dev_pairs(
        [source], [spoof], source_ranks={source.text_id: 1}, target_pairs=1
    )
    combined = build_v4_combined_dev_manifest([_pyara_row()], frozen)

    validate_manifest(combined)
    assert spoof.source_name == V4_KK_DEV_SILERO_SOURCE_ID
    assert spoof.split == "dev"
    assert spoof.text_hash == source.text_hash
    assert [row.language for row in combined] == ["kk", "kk", "ru"]
