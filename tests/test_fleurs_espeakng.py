from __future__ import annotations

from pathlib import Path

from kds.data.espeakng import load_espeakng_runtime
from kds.data.fleurs_espeakng import (
    FLEURS_RU_ESPEAKNG_SOURCE_ID,
    fleurs_ru_espeakng_spoof_row,
    select_fleurs_ru_espeakng_base,
)
from kds.data.manifest import load_manifest, validate_manifest
from kds.data.research_tts import load_research_tts_model_lock


def test_fleurs_ru_espeakng_reserves_every_existing_silero_text_group() -> None:
    full = load_manifest(Path("data/manifests/fleurs_ru_ru_v1_test_ready_300.csv"))
    existing = load_manifest(Path("data/manifests/fleurs_ru_v1_silero_v4_test_214.csv"))

    selected, held = select_fleurs_ru_espeakng_base(full, existing)

    assert len(selected) == 75
    assert len(held) == 214
    assert {row.text_hash for row in selected}.isdisjoint(row.text_hash for row in held)
    assert {row.sample_id for row in held} == {
        row.sample_id for row in existing if row.label == "bonafide"
    }


def test_fleurs_ru_espeakng_spoof_preserves_exact_text_provenance() -> None:
    full = load_manifest(Path("data/manifests/fleurs_ru_ru_v1_test_ready_300.csv"))
    existing = load_manifest(Path("data/manifests/fleurs_ru_v1_silero_v4_test_214.csv"))
    base = select_fleurs_ru_espeakng_base(full, existing)[0][0]
    lock = load_research_tts_model_lock(Path("configs/research/espeakng_ru_v1_models.json"))
    model = lock.models[0]
    profile = load_espeakng_runtime(model).profiles[0]

    spoof = fleurs_ru_espeakng_spoof_row(
        base_row=base,
        model=model,
        profile=profile,
        relative_path="raw/fleurs_ru_v1_espeakng/slices/v1/espeakng_russian_formant/a.wav",
        sha256="a" * 64,
        duration_s=1.0,
        original_sr=22_050,
        created_at="2026-08-11T00:00:00Z",
    )

    validate_manifest([spoof])
    assert spoof.source_name == FLEURS_RU_ESPEAKNG_SOURCE_ID
    assert spoof.text_id == base.text_id
    assert spoof.text_hash == base.text_hash
