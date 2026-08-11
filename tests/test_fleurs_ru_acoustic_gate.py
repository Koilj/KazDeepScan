from __future__ import annotations

import csv
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import load_manifest
from kds.eval.fleurs_ru_acoustic_gate import (
    FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID,
    FLEURS_RU_ACOUSTIC_GATE_REVIEW_FIELDS,
    build_fleurs_ru_acoustic_packet,
    evaluate_fleurs_ru_acoustic_gate,
    read_fleurs_ru_acoustic_reviews,
    write_fleurs_ru_acoustic_packet,
    write_fleurs_ru_acoustic_review_template,
)


def test_ru_packet_binds_every_pair_and_requires_two_reviews(tmp_path: Path) -> None:
    candidate = load_manifest(Path("data/manifests/fleurs_ru_v1_espeakng_test_75.csv"))
    transcripts = {
        row.sample_id: "проверяемая русская фраза"
        for row in candidate
        if row.label == "bonafide"
    }
    packet = build_fleurs_ru_acoustic_packet(candidate, transcripts)
    assert len(packet) == 150
    assert {item.label for item in packet} == {"bonafide", "spoof"}

    packet_path = tmp_path / "packet.csv"
    write_fleurs_ru_acoustic_packet(packet_path, packet)
    packet_hash = sha256_file(packet_path)
    template_path = tmp_path / "reviewer-a.csv"
    write_fleurs_ru_acoustic_review_template(template_path, packet_path, "reviewer_a")
    template = read_fleurs_ru_acoustic_reviews(template_path)
    assert len(template) == 150
    assert {item.review_status for item in template} == {"inconclusive"}

    reviews_path = tmp_path / "reviews.csv"
    with reviews_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLEURS_RU_ACOUSTIC_GATE_REVIEW_FIELDS)
        writer.writeheader()
        for item in packet:
            for reviewer in ("reviewer_a", "reviewer_b"):
                writer.writerow(
                    {
                        "protocol_id": FLEURS_RU_ACOUSTIC_GATE_PROTOCOL_ID,
                        "packet_sha256": packet_hash,
                        "text_hash": item.text_hash,
                        "label": item.label,
                        "sample_id": item.sample_id,
                        "audio_sha256": item.audio_sha256,
                        "reviewer_pseudo_id": reviewer,
                        "review_status": "pass",
                        "russian_audible": "yes",
                        "lexical_content_preserved": "yes",
                        "notes": "",
                    }
                )
    report, results = evaluate_fleurs_ru_acoustic_gate(
        packet_path, read_fleurs_ru_acoustic_reviews(reviews_path)
    )
    assert report["all_assets_acoustically_verified"] is True
    assert report["final_or_product_eligible"] is False
    assert {item.decision for item in results} == {"pass"}
