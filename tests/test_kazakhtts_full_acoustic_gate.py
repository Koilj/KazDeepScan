from __future__ import annotations

import csv
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.kazakhtts_text import KAZAKHTTS_TEXT_NORMALIZER_ID
from kds.eval.kazakhtts_full_acoustic_gate import (
    PACKET_FIELDS,
    PROTOCOL_ID,
    REVIEW_FIELDS,
    KazakhTtsFullPacketRow,
    evaluate_kazakhtts_full_acoustic_gate,
    read_kazakhtts_full_reviews,
    write_kazakhtts_full_acoustic_packet,
    write_kazakhtts_full_review_template,
)


def _packet(tmp_path: Path) -> tuple[KazakhTtsFullPacketRow, ...]:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"exact-test-bytes")
    digest = sha256_file(audio)
    rows = []
    languages = ("kk",) * 60 + ("mixed",) * 57 + ("ru",) * 50
    for index, language in enumerate(languages):
        rows.append(
            KazakhTtsFullPacketRow(
                protocol_id=PROTOCOL_ID,
                pairing_receipt_sha256="a" * 64,
                candidate_manifest_sha256="b" * 64,
                normalization_plan_sha256="c" * 64,
                sample_id=f"sample-{index}",
                language=language,
                text_id=f"text-{index}",
                source_text_hash="d" * 64,
                audio_path=audio.as_posix(),
                audio_sha256=digest,
                source_text="исходный текст",
                synthesis_text="текст синтеза",
                synthesis_text_sha256="e" * 64,
                normalizer_id=KAZAKHTTS_TEXT_NORMALIZER_ID,
            )
        )
    return tuple(rows)


def test_full_gate_requires_two_exact_complete_reviews(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.csv"
    write_kazakhtts_full_acoustic_packet(packet_path, _packet(tmp_path))
    template_path = tmp_path / "template.csv"
    write_kazakhtts_full_review_template(template_path, packet_path, "reviewer_a")
    template = read_kazakhtts_full_reviews(template_path)
    assert len(template) == 167
    assert {row.review_status for row in template} == {"inconclusive"}

    reviews_path = tmp_path / "reviews.csv"
    with reviews_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for item in _packet(tmp_path):
            for reviewer in ("reviewer_a", "reviewer_b"):
                writer.writerow(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "packet_sha256": sha256_file(packet_path),
                        "sample_id": item.sample_id,
                        "language": item.language,
                        "audio_sha256": item.audio_sha256,
                        "reviewer_pseudo_id": reviewer,
                        "review_status": "pass",
                        "speech_intelligible": "yes",
                        "lexical_content_preserved": "yes",
                        "language_preserved": "yes",
                        "severe_artifacts": "no",
                        "notes": "",
                    }
                )
    report = evaluate_kazakhtts_full_acoustic_gate(
        packet_path, read_kazakhtts_full_reviews(reviews_path)
    )
    assert report["all_assets_acoustically_verified"] is True
    assert report["passed_by_language"] == {"kk": 60, "mixed": 57, "ru": 50}
    assert report["detector_inference_authorized"] is False


def test_full_packet_schema_is_frozen() -> None:
    assert PACKET_FIELDS[0] == "protocol_id"
    assert PACKET_FIELDS[-1] == "normalizer_id"
