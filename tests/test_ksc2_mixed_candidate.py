from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.ksc2_mixed_candidate import (
    Ksc2MixedAudioInfo,
    Ksc2MixedCandidateError,
    load_published_mixed_review,
    mixed_bonafide_rows,
    select_mixed_smoke_evidence,
)
from kds.data.manifest import validate_manifest


def test_published_review_builds_only_mixed_bonafide_candidates() -> None:
    evidence = load_published_mixed_review(
        Path("data/manifests/ksc2_test_mixed_ai_review_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json"),
    )
    audio = {
        item.annotation_id: Ksc2MixedAudioInfo(duration_s=1.0, original_sr=16_000, codec="flac")
        for item in evidence
    }
    rows = mixed_bonafide_rows(evidence, audio, created_at="2026-08-11T00:00:00Z")

    validate_manifest(rows)
    assert len(rows) == 32
    assert {row.label for row in rows} == {"bonafide"}
    assert {row.language for row in rows} == {"mixed"}
    assert {row.code_switch for row in rows} == {"true"}
    assert {row.speaker_pseudo_id for row in rows} == {"ksc2_v1:unknown"}


def test_mixed_candidate_rejects_missing_audio_metadata() -> None:
    evidence = load_published_mixed_review(
        Path("data/manifests/ksc2_test_mixed_ai_review_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json"),
    )

    with pytest.raises(Ksc2MixedCandidateError, match="invalid audio metadata"):
        mixed_bonafide_rows(evidence, {}, created_at="2026-08-11T00:00:00Z")


def test_smoke_selection_is_deterministic_and_covers_components() -> None:
    evidence = load_published_mixed_review(
        Path("data/manifests/ksc2_test_mixed_ai_review_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json"),
    )

    selected = select_mixed_smoke_evidence(evidence, limit=5, seed="20260811")

    assert len(selected) == 5
    assert len({item.annotation_id for item in selected}) == 5
    assert {item.component for item in selected} == {
        "Test/podcasts",
        "Test/radio",
        "Test/talkshow",
    }
    assert selected == select_mixed_smoke_evidence(evidence, limit=5, seed="20260811")


def test_stage_c_delta_review_profile_is_accepted(tmp_path: Path) -> None:
    from scripts.publish_ksc2_ai_mixed_review import main as publish_ai_review

    review = tmp_path / "review.csv"
    receipt = tmp_path / "receipt.json"
    assert publish_ai_review(
        [
            "--packet",
            "data/manifests/ksc2_test_mixed_annotation_v1.csv",
            "--packet-receipt",
            "data/licenses/ksc2_test_mixed_annotation_v1_receipt.json",
            "--packet-lock",
            "data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json",
            "--reviewed-at",
            "2026-08-12T21:00:00+05:00",
            "--decision-set",
            "v2_delta",
            "--output-csv",
            str(review),
            "--output-receipt",
            str(receipt),
        ]
    ) == 0

    evidence = load_published_mixed_review(review, receipt)
    assert len(evidence) == 59
    assert {item.review_method for item in evidence} == {
        "single_ai_transcript_semantic_review_v2_delta"
    }
