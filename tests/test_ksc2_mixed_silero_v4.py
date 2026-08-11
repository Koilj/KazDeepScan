from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from kds.data.ksc2_mixed_silero_v4 import (
    KSC2_MIXED_SILERO_V4_SOURCE_ID,
    build_paired_mixed_candidate_rows,
    mixed_silero_v4_spoof_row,
)
from kds.data.manifest import ManifestRow, validate_manifest
from kds.data.research_tts import load_research_tts_model_lock
from kds.data.silero_v4 import load_silero_v4_runtime
from tests.factories import manifest_mapping


def _base_row() -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="ksc2_v1:Test/podcasts/row",
            relative_path="processed/mixed.wav",
            sha256="a" * 64,
            split="test",
            label="bonafide",
            language="mixed",
            code_switch="true",
            source_name="ksc2_v1",
            source_license="CC-BY-4.0",
            text_id="ksc2_v1:transcript:" + "b" * 64,
            text_hash="b" * 64,
        ),
        2,
    )


def test_mixed_silero_pair_preserves_input_text_provenance() -> None:
    base = _base_row()
    lock = load_research_tts_model_lock(Path("configs/research/silero_v4_cyrillic_v1_models.json"))
    model = lock.models[0]
    profile = load_silero_v4_runtime(model).profiles_by_language["kk"][0]
    raw = mixed_silero_v4_spoof_row(
        base_row=base,
        model=model,
        profile=profile,
        relative_path="raw/ksc2_mixed_v1_silero_v4/slices/v1/s.wav",
        sha256="c" * 64,
        duration_s=1.0,
        original_sr=48_000,
        created_at="2026-08-11T00:00:00Z",
        device="local_cuda_silero_v4_fastpitch_hifigan",
    )

    validate_manifest([raw])
    assert raw.source_name == KSC2_MIXED_SILERO_V4_SOURCE_ID
    assert raw.language == "mixed"
    assert raw.code_switch == "true"
    assert raw.text_hash == base.text_hash
    assert "intended_input_text_only" in raw.augmentation_chain
    assert build_paired_mixed_candidate_rows(
        base_rows=[base],
        raw_spoof_rows=[raw],
        ready_spoof_rows=[replace(raw, relative_path="processed/s.wav", sha256="d" * 64)],
        text_rejected_base_ids=set(),
        audio_rejected_spoof_ids=set(),
    ) == [base, replace(raw, relative_path="processed/s.wav", sha256="d" * 64)]
