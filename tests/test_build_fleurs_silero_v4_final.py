from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow, write_manifest
from kds.data.research_tts import load_research_tts_model_lock
from kds.data.silero_v4 import load_silero_v4_runtime, silero_v4_spoof_row
from scripts.build_fleurs_silero_v4_final import build_final_rows
from tests.factories import manifest_mapping


def _base_row() -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="google_fleurs_ru_v1:1",
            relative_path="processed/ru/base.wav",
            sha256="a" * 64,
            split="test",
            label="bonafide",
            language="ru",
            code_switch="false",
            source_name="google_fleurs_ru_v1",
            source_license="CC-BY-4.0",
            text_id="google_fleurs_ru_v1:prompt:1",
            text_hash="b" * 64,
        ),
        2,
    )


def _spoof_row(base: ManifestRow) -> ManifestRow:
    lock = load_research_tts_model_lock(Path("configs/research/silero_v4_cyrillic_v1_models.json"))
    model = lock.models[0]
    runtime = load_silero_v4_runtime(model)
    return silero_v4_spoof_row(
        base_row=base,
        model=model,
        profile=runtime.profiles_by_language["ru"][0],
        relative_path="processed/ru/spoof.wav",
        sha256="c" * 64,
        duration_s=1.0,
        original_sr=16_000,
        created_at="2026-08-11T00:00:00Z",
        device="local_cpu_silero_v4_fastpitch_hifigan",
    )


def test_builder_requires_complete_rejection_accounting_and_exact_pair(tmp_path: Path) -> None:
    base = _base_row()
    spoof = _spoof_row(base)
    base_manifest = tmp_path / "base.csv"
    raw_manifest = tmp_path / "raw.csv"
    ready_manifest = tmp_path / "ready.csv"
    write_manifest(base_manifest, [base])
    write_manifest(raw_manifest, [spoof])
    write_manifest(ready_manifest, [spoof])
    text_report = {
        "base_manifest_sha256": {str(base_manifest): sha256_file(base_manifest)},
        "rejected_rows": [],
    }
    audio_report = {"input_manifest": str(raw_manifest), "rejected_rows": []}

    final_rows = build_final_rows(
        base_rows=[base],
        raw_spoof_rows=[spoof],
        ready_spoof_rows=[spoof],
        text_rejection_report=text_report,
        audio_rejection_report=audio_report,
        base_manifest=base_manifest,
        raw_manifest=raw_manifest,
        language="ru",
    )

    assert [row.label for row in final_rows] == ["bonafide", "spoof"]
    assert final_rows[0].text_hash == final_rows[1].text_hash

    with pytest.raises(ValueError, match="invalid ru"):
        build_final_rows(
            base_rows=[base],
            raw_spoof_rows=[],
            ready_spoof_rows=[],
            text_rejection_report=text_report,
            audio_rejection_report=audio_report,
            base_manifest=base_manifest,
            raw_manifest=raw_manifest,
            language="ru",
        )


def test_builder_requires_text_report_hash_of_exact_base_manifest(tmp_path: Path) -> None:
    base = _base_row()
    spoof = _spoof_row(base)
    base_manifest = tmp_path / "base.csv"
    raw_manifest = tmp_path / "raw.csv"
    write_manifest(base_manifest, [base])
    write_manifest(raw_manifest, [spoof])

    with pytest.raises(ValueError, match="not pinned"):
        build_final_rows(
            base_rows=[base],
            raw_spoof_rows=[spoof],
            ready_spoof_rows=[spoof],
            text_rejection_report={"base_manifest_sha256": {}, "rejected_rows": []},
            audio_rejection_report={"input_manifest": str(raw_manifest), "rejected_rows": []},
            base_manifest=base_manifest,
            raw_manifest=raw_manifest,
            language="ru",
        )
