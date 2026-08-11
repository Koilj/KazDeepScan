from __future__ import annotations

import csv
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.ksc2_mixed_candidate import load_published_mixed_review
from kds.data.manifest import load_manifest
from kds.eval.mixed_acoustic_gate import (
    MIXED_ACOUSTIC_GATE_PROTOCOL_ID,
    MIXED_ACOUSTIC_GATE_REVIEW_FIELDS,
    build_mixed_acoustic_gate_packet,
    evaluate_mixed_acoustic_gate,
    load_pair_lock,
    read_mixed_acoustic_gate_reviews,
    write_mixed_acoustic_gate_packet,
    write_mixed_acoustic_gate_review_template,
)


def test_gate_packet_binds_all_frozen_assets_and_requires_two_reviews(tmp_path: Path) -> None:
    candidate = load_manifest(Path("data/manifests/ksc2_mixed_v1_silero_v4_candidate_30.csv"))
    evidence = load_published_mixed_review(
        Path("data/manifests/ksc2_test_mixed_ai_review_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json"),
    )
    packet = build_mixed_acoustic_gate_packet(
        candidate,
        evidence,
        load_pair_lock(Path("data/licenses/ksc2_mixed_v1_silero_v4_pair_lock.json")),
    )
    assert len(packet) == 60
    assert {item.label for item in packet} == {"bonafide", "spoof"}
    assert all(item.ru_evidence_tokens and item.kk_evidence_tokens for item in packet)

    packet_path = tmp_path / "packet.csv"
    write_mixed_acoustic_gate_packet(packet_path, packet)
    packet_hash = sha256_file(packet_path)
    template_path = tmp_path / "reviewer-a.csv"
    write_mixed_acoustic_gate_review_template(template_path, packet_path, "reviewer_a")
    template = read_mixed_acoustic_gate_reviews(template_path)
    assert len(template) == 60
    assert {item.review_status for item in template} == {"inconclusive"}
    reviews_path = tmp_path / "reviews.csv"
    with reviews_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MIXED_ACOUSTIC_GATE_REVIEW_FIELDS)
        writer.writeheader()
        for item in packet:
            for reviewer in ("reviewer_a", "reviewer_b"):
                writer.writerow(
                    {
                        "protocol_id": MIXED_ACOUSTIC_GATE_PROTOCOL_ID,
                        "packet_sha256": packet_hash,
                        "annotation_id": item.annotation_id,
                        "label": item.label,
                        "sample_id": item.sample_id,
                        "audio_sha256": item.audio_sha256,
                        "reviewer_pseudo_id": reviewer,
                        "review_status": "pass",
                        "ru_evidence_audible": "yes",
                        "kk_evidence_audible": "yes",
                        "lexical_content_preserved": "yes",
                        "notes": "",
                    }
                )
    report, results = evaluate_mixed_acoustic_gate(
        packet_path, read_mixed_acoustic_gate_reviews(reviews_path)
    )
    assert report["all_assets_acoustically_verified"] is True
    assert report["final_or_product_eligible"] is False
    assert {item.decision for item in results} == {"pass"}
