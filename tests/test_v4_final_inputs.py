from __future__ import annotations

import hashlib

import pytest

from kds.data.fleurs import FleursRecord
from kds.data.manifest import ManifestRow
from kds.data.v4_final_inputs import (
    CandidateIdentity,
    V4FinalInputError,
    _kk_candidates,
    select_distinct_metadata_candidates,
)


def _candidate(sample_id: str, group: str, text: str) -> CandidateIdentity:
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return CandidateIdentity(
        language="ru",
        sample_id=sample_id,
        source_member=f"{sample_id}.mp3",
        source_split="test",
        parent_group_id=group,
        speaker_pseudo_id=group,
        text_id=f"text:{sample_id}",
        text_hash=text_hash,
        synthesis_text_sha256=text_hash,
        synthesis_seed="0",
        normalization_operations="literal_utf8_source_text",
    )


def _history_row(*, sample_id: str, text_hash: str, parent_group_id: str) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        relative_path=f"processed/{sample_id}.wav",
        sha256=hashlib.sha256(sample_id.encode()).hexdigest(),
        split="test",
        label="bonafide",
        language="kk",
        code_switch="false",
        parent_group_id=parent_group_id,
        source_name="google_fleurs_kk_v1",
        source_license="CC-BY-4.0",
        rights_basis="test",
        speaker_pseudo_id="google_fleurs_kk_v1:unknown",
        text_id=parent_group_id,
        text_hash=text_hash,
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="test",
        capture_route="test",
        original_sr=16000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-15T00:00:00Z",
    )


def test_final_metadata_selection_is_deterministic_and_group_text_disjoint() -> None:
    selected = select_distinct_metadata_candidates(
        (
            _candidate("s1", "g1", "one"),
            _candidate("s2", "g1", "two"),
            _candidate("s3", "g2", "one"),
            _candidate("s4", "g3", "four"),
        ),
        seed="v4-final-test",
        target=2,
    )

    assert len(selected) == 2
    assert len({item.parent_group_id for item in selected}) == 2
    assert len({item.text_hash for item in selected}) == 2
    assert selected == select_distinct_metadata_candidates(
        (
            _candidate("s1", "g1", "one"),
            _candidate("s2", "g1", "two"),
            _candidate("s3", "g2", "one"),
            _candidate("s4", "g3", "four"),
        ),
        seed="v4-final-test",
        target=2,
    )


def test_final_metadata_selection_fails_closed_for_insufficient_disjoint_capacity() -> None:
    with pytest.raises(V4FinalInputError, match="insufficient disjoint source/text capacity"):
        select_distinct_metadata_candidates(
            (_candidate("s1", "g1", "one"), _candidate("s2", "g1", "two")),
            seed="v4-final-test",
            target=2,
        )


def test_fleurs_unknown_speaker_placeholder_does_not_block_fresh_prompt_groups() -> None:
    record = FleursRecord(
        locale="kk_kz",
        language="kk",
        source_split="train",
        prompt_id="100",
        filename="100.wav",
        raw_transcript="Алматы",
        transcript="Алматы",
        character_transcript="Алматы",
        samples=16_000,
        gender="FEMALE",
    )
    unrelated = _history_row(
        sample_id="google_fleurs_kk_v1:old",
        text_hash=hashlib.sha256(b"old").hexdigest(),
        parent_group_id="google_fleurs_kk_v1:prompt:old",
    )

    candidates, audit = _kk_candidates((record,), (unrelated,))

    assert len(candidates) == 1
    assert candidates[0].speaker_pseudo_id == "google_fleurs_kk_v1:unknown"
    assert audit["eligible_text_groups"] == 1
    assert audit["historical_overlap_counts"] == {
        "parent_group_id": 0,
        "sample_id": 0,
        "text_hash": 0,
    }
