from __future__ import annotations

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

from kds.data.manifest import ManifestRow


def _paired_bonafide_rows(*args: Any, **kwargs: Any) -> list[ManifestRow]:
    script_path = Path("scripts/build_ksc_derived_kk_test.py")
    spec = spec_from_file_location("kds_test_manifest_builder", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(list[ManifestRow], module._paired_bonafide_rows(*args, **kwargs))


def _base_row() -> ManifestRow:
    return ManifestRow(
        sample_id="ksc_slr102:u1",
        relative_path="processed/ksc_slr102/u1.wav",
        sha256="a" * 64,
        split="test",
        label="bonafide",
        language="kk",
        code_switch="false",
        parent_group_id="ksc_slr102:source-split:test",
        source_name="ksc_slr102",
        source_license="CC-BY-4.0",
        rights_basis="research",
        speaker_pseudo_id="ksc_slr102:unknown",
        text_id="ksc_slr102:u1",
        text_hash="b" * 64,
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="corpus",
        original_sr=16_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-10T00:00:00Z",
    )


def test_paired_builder_accepts_explicit_new_derived_source() -> None:
    base = _base_row()
    spoof = replace(
        base,
        sample_id="ksc_derived_kk_v2_kazemotts:s1",
        relative_path="processed/ksc_derived_kk_v2_kazemotts/s1.wav",
        sha256="c" * 64,
        label="spoof",
        source_name="ksc_derived_kk_v2_kazemotts",
        source_license="CC-BY-4.0",
        generator_family="gradtts_hifigan_emotional_tts",
        generator_name="KazEmoTTS",
        generator_version="0db250b2",
        voice_id="M1_neutral",
        clone_consent_id="not_applicable:pretrained-tts-no-local-voice-cloning",
        capture_route="offline_neural_tts",
    )

    assert _paired_bonafide_rows(
        [base],
        [spoof],
        spoof_source_name="ksc_derived_kk_v2_kazemotts",
    ) == [base]
