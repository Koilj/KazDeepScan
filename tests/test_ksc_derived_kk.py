from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.ksc_derived_kk import (
    SynthesisProfile,
    assign_synthesis_profiles,
    load_verified_ksc_transcript,
    merge_prepared_ksc_rows,
    select_ksc_bonafide_rows,
)
from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel
from tests.factories import manifest_mapping


def _row(index: int, *, split: str = "test", label: str = "bonafide") -> ManifestRow:
    text = f"мәтін {index}"
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id=f"ksc_slr102:u{index}",
            source_name="ksc_slr102",
            split=split,
            label=label,
            language="kk",
            sha256=f"{index:064x}",
            parent_group_id=f"ksc:{split}",
            speaker_pseudo_id=f"ksc:{split}:unknown",
            text_id=f"ksc_slr102:u{index}",
            text_hash=hashlib.sha256(text.encode()).hexdigest(),
            generator_family="family" if label == "spoof" else "",
            generator_name="name" if label == "spoof" else "",
            generator_version="1" if label == "spoof" else "",
            voice_id="voice" if label == "spoof" else "",
        ),
        row_number=index + 2,
    )


def _model(model_id: str, family: str) -> ResearchTtsModel:
    return ResearchTtsModel(
        model_id=model_id,
        destination=model_id,
        generator_family=family,
        generator_name=model_id,
        generator_version="1",
        license="research",
        source_url="https://example.test/model",
        runtime={"kind": "test"},
        artifacts=(),
    )


def test_derived_ksc_selection_is_deterministic_and_excludes_non_test_rows() -> None:
    rows = [_row(index) for index in range(8)] + [_row(9, split="dev")]

    first = select_ksc_bonafide_rows(rows, limit=5, seed="fixed")
    second = select_ksc_bonafide_rows(rows, limit=5, seed="fixed")

    assert first == second
    assert len(first) == 5
    assert all(row.split == "test" and row.label == "bonafide" for row in first)
    with pytest.raises(ValueError, match="Need 9"):
        select_ksc_bonafide_rows(rows, limit=9, seed="fixed")


def test_derived_ksc_assignments_balance_families_then_voices() -> None:
    rows = [_row(index) for index in range(14)]
    piper = _model("piper", "piper_neural_tts")
    mms = _model("mms", "mms_vits_tts")
    profiles = [
        SynthesisProfile(piper, "piper-a", 0),
        SynthesisProfile(piper, "piper-b", 1),
        SynthesisProfile(mms, "mms-default", None),
    ]

    assignments = assign_synthesis_profiles(rows, profiles)
    family_counts = Counter(profile.model.generator_family for _row, profile in assignments)
    voice_counts = Counter(profile.voice_id for _row, profile in assignments)

    assert family_counts == {"mms_vits_tts": 7, "piper_neural_tts": 7}
    assert voice_counts == {"mms-default": 7, "piper-a": 4, "piper-b": 3}


def test_verified_ksc_transcript_must_match_manifest_hash(tmp_path: Path) -> None:
    row = _row(1)
    transcripts = tmp_path / "Transcriptions"
    transcripts.mkdir()
    (transcripts / "u1.txt").write_text("мәтін   1\n", encoding="utf-8")

    assert load_verified_ksc_transcript(row, tmp_path) == "мәтін 1"
    (transcripts / "u1.txt").write_text("басқа мәтін", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_ksc_transcript(row, tmp_path)


def test_derived_ksc_merge_reuses_only_matching_prior_ready_rows() -> None:
    raw_first = _row(1)
    raw_second = _row(2)
    first_ready = replace(
        raw_first,
        relative_path="processed/first.wav",
        sha256="a" * 64,
        duration_s=3.0,
        codec="wav",
    )
    second_ready = replace(
        raw_second,
        relative_path="processed/second.wav",
        sha256="b" * 64,
        duration_s=3.0,
        codec="wav",
    )

    merged, reused_ids = merge_prepared_ksc_rows(
        [raw_first, raw_second], [first_ready], [second_ready]
    )

    assert merged == [first_ready, second_ready]
    assert reused_ids == {raw_second.sample_id}
    bad_ready = replace(second_ready, text_hash="wrong")
    with pytest.raises(ValueError, match="does not match raw provenance"):
        merge_prepared_ksc_rows([raw_second], [], [bad_ready])
