from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow, load_manifest
from kds.eval.fleurs_kk_acoustic_gate import (
    FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID,
    FLEURS_KK_ACOUSTIC_GATE_REVIEW_FIELDS,
    FleursKkAcousticGateError,
    FleursKkAcousticPacketRow,
    build_fleurs_kk_acoustic_packet,
    evaluate_fleurs_kk_acoustic_gate,
    read_fleurs_kk_acoustic_reviews,
    write_fleurs_kk_acoustic_packet,
    write_fleurs_kk_acoustic_review_template,
)


def _candidate_with_transcripts() -> tuple[list[ManifestRow], dict[str, str]]:
    candidate = load_manifest(Path("data/manifests/fleurs_kk_v1_silero_v4_test_152.csv"))
    transcript_by_old_hash = {
        text_hash: f"қазақша тексеру мәтіні {index}"
        for index, text_hash in enumerate(sorted({row.text_hash for row in candidate}))
    }
    new_hash_by_old_hash = {
        old_hash: hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        for old_hash, transcript in transcript_by_old_hash.items()
    }
    changed = [
        replace(row, text_hash=new_hash_by_old_hash[row.text_hash]) for row in candidate
    ]
    transcripts = {
        row.sample_id: transcript_by_old_hash[original.text_hash]
        for row, original in zip(changed, candidate, strict=True)
        if row.label == "bonafide"
    }
    return changed, transcripts


def _write_reviews(
    path: Path,
    packet: tuple[FleursKkAcousticPacketRow, ...],
    packet_hash: str,
    *,
    omit_last: bool = False,
    failed_sample_id: str = "",
) -> None:
    rows = list(packet)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FLEURS_KK_ACOUSTIC_GATE_REVIEW_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for item in rows[:-1] if omit_last else rows:
            failed = item.sample_id == failed_sample_id
            writer.writerow(
                {
                    "protocol_id": FLEURS_KK_ACOUSTIC_GATE_PROTOCOL_ID,
                    "packet_sha256": packet_hash,
                    "text_hash": item.text_hash,
                    "label": item.label,
                    "sample_id": item.sample_id,
                    "audio_sha256": item.audio_sha256,
                    "relative_path": item.relative_path,
                    "input_transcript": item.input_transcript,
                    "reviewer_pseudo_id": path.stem,
                    "review_status": "fail" if failed else "pass",
                    "audio_audible": "yes",
                    "kazakh_text_matches": "no" if failed else "yes",
                    "no_obvious_defects": "yes",
                    "notes": "lexical mismatch" if failed else "",
                }
            )


def test_kk_packet_and_receipt_require_two_complete_reviews(tmp_path: Path) -> None:
    candidate, transcripts = _candidate_with_transcripts()
    packet = build_fleurs_kk_acoustic_packet(candidate, transcripts)
    assert len(packet) == 304
    assert {item.label for item in packet} == {"bonafide", "spoof"}

    packet_path = tmp_path / "packet.csv"
    write_fleurs_kk_acoustic_packet(packet_path, packet)
    packet_hash = sha256_file(packet_path)
    template_path = tmp_path / "reviewer-template.csv"
    write_fleurs_kk_acoustic_review_template(template_path, packet_path, "reviewer_template")
    template = read_fleurs_kk_acoustic_reviews(template_path)
    assert len(template) == 304
    assert {item.review_status for item in template} == {"inconclusive"}
    assert {item.audio_audible for item in template} == {"unknown"}
    assert all(item.relative_path and item.input_transcript for item in template)

    with pytest.raises(FleursKkAcousticGateError, match="decision contract"):
        evaluate_fleurs_kk_acoustic_gate(packet_path, template + template)

    review_a = tmp_path / "reviewer_a.csv"
    review_b = tmp_path / "reviewer_b.csv"
    _write_reviews(review_a, packet, packet_hash)
    _write_reviews(review_b, packet, packet_hash)
    reviews = read_fleurs_kk_acoustic_reviews(review_a) + read_fleurs_kk_acoustic_reviews(review_b)
    report, results = evaluate_fleurs_kk_acoustic_gate(packet_path, reviews)
    assert report["all_assets_acoustically_verified"] is True
    assert report["evidence_timing"] == "post_inference"
    assert report["metric_status_changed"] is False
    assert report["final_or_product_eligible"] is False
    assert {item.decision for item in results} == {"pass"}


def test_kk_gate_records_a_complete_failed_decision_without_promoting_metric(
    tmp_path: Path,
) -> None:
    candidate, transcripts = _candidate_with_transcripts()
    packet = build_fleurs_kk_acoustic_packet(candidate, transcripts)
    packet_path = tmp_path / "packet.csv"
    write_fleurs_kk_acoustic_packet(packet_path, packet)
    packet_hash = sha256_file(packet_path)
    review_a = tmp_path / "reviewer_a.csv"
    review_b = tmp_path / "reviewer_b.csv"
    _write_reviews(review_a, packet, packet_hash, failed_sample_id=packet[0].sample_id)
    _write_reviews(review_b, packet, packet_hash)
    reviews = read_fleurs_kk_acoustic_reviews(review_a) + read_fleurs_kk_acoustic_reviews(review_b)
    report, results = evaluate_fleurs_kk_acoustic_gate(packet_path, reviews)
    assert report["all_assets_acoustically_verified"] is False
    assert report["metric_status_changed"] is False
    assert sum(result.decision == "not_eligible" for result in results) == 1


def test_kk_gate_refuses_incomplete_or_contradictory_reviews(tmp_path: Path) -> None:
    candidate, transcripts = _candidate_with_transcripts()
    packet = build_fleurs_kk_acoustic_packet(candidate, transcripts)
    packet_path = tmp_path / "packet.csv"
    write_fleurs_kk_acoustic_packet(packet_path, packet)
    packet_hash = sha256_file(packet_path)
    review_a = tmp_path / "reviewer_a.csv"
    review_b = tmp_path / "reviewer_b.csv"
    _write_reviews(review_a, packet, packet_hash)
    _write_reviews(review_b, packet, packet_hash, omit_last=True)
    reviews = read_fleurs_kk_acoustic_reviews(review_a) + read_fleurs_kk_acoustic_reviews(review_b)
    with pytest.raises(FleursKkAcousticGateError, match="requires 608 review rows"):
        evaluate_fleurs_kk_acoustic_gate(packet_path, reviews)

    contradictory = list(read_fleurs_kk_acoustic_reviews(review_a))
    contradictory[0] = replace(contradictory[0], kazakh_text_matches="no")
    complete = tuple(contradictory) + read_fleurs_kk_acoustic_reviews(review_a)
    with pytest.raises(FleursKkAcousticGateError, match="decision contract"):
        evaluate_fleurs_kk_acoustic_gate(packet_path, complete)
