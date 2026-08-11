from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow
from kds.eval.tone_speak_acoustic_gate import (
    TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID,
    TONE_SPEAK_ACOUSTIC_GATE_REVIEW_FIELDS,
    build_tone_speak_acoustic_packet,
    evaluate_tone_speak_acoustic_gate,
    read_tone_speak_acoustic_reviews,
    write_tone_speak_acoustic_packet,
    write_tone_speak_acoustic_review_template,
)


def _candidate() -> tuple[list[ManifestRow], dict[str, str]]:
    rows: list[ManifestRow] = []
    transcripts: dict[str, str] = {}
    for index in range(100):
        voice = "alloy" if index % 2 == 0 else "coral"
        sample_id = f"tone_speak_ru_v1:{index:05d}_{voice}"
        transcript = f"Русский тестовый текст номер {index}."
        text_hash = hashlib.sha256(" ".join(transcript.split()).encode("utf-8")).hexdigest()
        transcripts[sample_id] = transcript
        rows.append(
            ManifestRow(
                sample_id=sample_id,
                relative_path=f"processed/{index:02x}/{index:064x}.wav",
                sha256=f"{index:064x}",
                split="ood",
                label="spoof",
                language="ru",
                code_switch="false",
                parent_group_id=f"tone_speak_ru_v1:text:{text_hash}",
                source_name="tone_speak_ru_v1",
                source_license="Apache-2.0",
                rights_basis="test",
                speaker_pseudo_id=f"tone_speak_ru_v1:voice:{voice}",
                text_id=f"tone_speak_ru_v1:text:{text_hash}",
                text_hash=text_hash,
                duration_s=3.0,
                generator_family="neural_tts",
                generator_name="openai_gpt_4o_mini_tts",
                generator_version="source_card_unpinned",
                voice_id=f"tone_speak_ru_v1:voice:{voice}",
                clone_consent_id="",
                device="unknown",
                capture_route="openai_tts_source_release",
                original_sr=24_000,
                codec="wav",
                augmentation_chain="",
                augmentation_seed="",
                created_at="2026-08-11T00:00:00Z",
            )
        )
    return rows, transcripts


def test_tone_speak_packet_requires_two_independent_passes(tmp_path: Path) -> None:
    candidate, transcripts = _candidate()
    packet = build_tone_speak_acoustic_packet(candidate, transcripts)
    assert len(packet) == 100

    packet_path = tmp_path / "packet.csv"
    write_tone_speak_acoustic_packet(packet_path, packet)
    packet_hash = sha256_file(packet_path)
    template_path = tmp_path / "reviewer-a.csv"
    write_tone_speak_acoustic_review_template(template_path, packet_path, "reviewer_a")
    assert len(read_tone_speak_acoustic_reviews(template_path)) == 100

    reviews_path = tmp_path / "reviews.csv"
    with reviews_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TONE_SPEAK_ACOUSTIC_GATE_REVIEW_FIELDS)
        writer.writeheader()
        for item in packet:
            for reviewer in ("reviewer_a", "reviewer_b"):
                writer.writerow(
                    {
                        "protocol_id": TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID,
                        "packet_sha256": packet_hash,
                        "text_hash": item.text_hash,
                        "sample_id": item.sample_id,
                        "audio_sha256": item.audio_sha256,
                        "reviewer_pseudo_id": reviewer,
                        "review_status": "pass",
                        "russian_audible": "yes",
                        "lexical_content_preserved": "yes",
                        "notes": "",
                    }
                )
    report, results = evaluate_tone_speak_acoustic_gate(
        packet_path, read_tone_speak_acoustic_reviews(reviews_path)
    )
    assert report["all_assets_acoustically_verified"] is True
    assert report["final_or_product_eligible"] is False
    assert {item.decision for item in results} == {"pass"}
