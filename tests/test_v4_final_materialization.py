from __future__ import annotations

from pathlib import Path
from typing import cast

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel
from kds.data.v4_final_materialization import (
    KK_SPOOF_ID,
    PROTOCOL_ID,
    RU_SOURCE_ID,
    _history,
    _load_selection,
    _spoof_row,
    load_plan,
)


def _base_row() -> ManifestRow:
    return ManifestRow(
        sample_id="common_voice_ru_v24:clip",
        relative_path="raw/source.mp3",
        sha256="a" * 64,
        split="test",
        label="bonafide",
        language="ru",
        code_switch="false",
        parent_group_id="common_voice_ru_v24:client:test",
        source_name=RU_SOURCE_ID,
        source_license="CC0-1.0",
        rights_basis="test",
        speaker_pseudo_id="common_voice_ru_v24:client:test",
        text_id="common_voice_ru_v24:sentence:test",
        text_hash="b" * 64,
        duration_s=3.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="test",
        original_sr=48_000,
        codec="mp3",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-15T22:00:00+06:00",
    )


def _model() -> ResearchTtsModel:
    return ResearchTtsModel(
        model_id="locked-model",
        destination="locked-model",
        generator_family="test-family",
        generator_name="test-generator",
        generator_version="v1",
        license="test-license",
        source_url="https://example.invalid/model",
        runtime={},
        artifacts=(),
    )


def test_final_materialization_plan_loads_all_pinned_inputs() -> None:
    root = Path(__file__).resolve().parents[1]

    plan = load_plan(
        root / "configs/research/v4/xlsr_sls_model_v4_final_materialization_v1.json",
        root,
    )

    assert plan.path == "configs/research/v4/xlsr_sls_model_v4_final_materialization_v1.json"
    assert plan.inputs["metadata_selection"].rows == 1000
    assert plan.outputs["pair_lock_manifest"].endswith("final_pairs_frozen_v1.csv")
    selected = _load_selection(plan, root)
    assert len(selected) == 1000
    assert all(row.synthesis_seed.isdecimal() for row in selected if row.language == "ru")
    assert all(not row.synthesis_seed for row in selected if row.language == "kk")
    _historical_exact, _historical_signatures, history = _history(plan, root)
    assert cast(int, history["unique_audio_hashes"]) >= 84_605


def test_final_spoof_row_binds_explicit_audio_hash_and_one_shot_route() -> None:
    row = _spoof_row(
        _base_row(),
        _model(),
        KK_SPOOF_ID,
        "raw/v4/final.wav",
        "c" * 64,
        2.5,
        22_050,
        "2026-08-15T22:00:00+06:00",
        "1",
        "d" * 64,
        "cuda:0",
    )

    assert row.sha256 == "c" * 64
    assert row.source_name == KK_SPOOF_ID
    assert PROTOCOL_ID.split("-v1")[0] in row.sample_id or row.sample_id.startswith(KK_SPOOF_ID)
    assert "reference_audio=forbidden" in row.augmentation_chain
