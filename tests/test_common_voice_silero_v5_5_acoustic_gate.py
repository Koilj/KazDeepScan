from __future__ import annotations

from pathlib import Path

from kds.data.manifest import load_manifest
from kds.eval.common_voice_silero_v5_5_acoustic_gate import (
    build_packet,
    evaluate,
    read_reviews,
)


def test_build_packet_requires_exact_84_asset_candidate() -> None:
    candidate = load_manifest(
        Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairs_v1.csv")
    )
    transcripts = {
        row.sample_id: "Точный русский текст"
        for row in candidate
        if row.label == "bonafide"
    }

    packet = build_packet(candidate, transcripts)

    assert len(packet) == 84
    assert {row.label for row in packet} == {"bonafide", "spoof"}


def test_completed_review_forms_authorize_the_technical_gate() -> None:
    packet = Path(
        "data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_packet_v1.csv"
    )
    reviews = [
        *read_reviews(
            Path(
                "data/manifests/"
                "common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_review_reviewer_a_v1.csv"
            )
        ),
        *read_reviews(
            Path(
                "data/manifests/"
                "common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_review_reviewer_b_v1.csv"
            )
        ),
    ]

    report, results = evaluate(packet, reviews)

    assert report["all_assets_acoustically_verified"] is True
    assert report["evaluation_contract_authorized"] is True
    assert {result.decision for result in results} == {"pass"}
