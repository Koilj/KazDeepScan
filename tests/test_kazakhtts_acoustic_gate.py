from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from kds.eval.kazakhtts_acoustic_gate import (
    build_kazakhtts_acoustic_packet,
    evaluate_kazakhtts_acoustic_gate,
    read_kazakhtts_reviews,
    write_kazakhtts_acoustic_packet,
    write_kazakhtts_review_template,
)


def _smoke_report(tmp_path: Path) -> Path:
    outputs = []
    for language, status in (
        ("kk", "officially_supported"),
        ("ru", "conditional_acoustic_smoke_only"),
        ("mixed", "conditional_acoustic_smoke_only"),
    ):
        audio = tmp_path / f"{language}.wav"
        audio.write_bytes(f"audio-{language}".encode())
        outputs.append(
            {
                "case_id": f"{language}-case",
                "language": language,
                "support_status": status,
                "relative_path": str(audio),
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "text": f"text-{language}",
            }
        )
    report = tmp_path / "smoke.json"
    report.write_text(
        json.dumps(
            {
                "technical_smoke_passed": True,
                "detector_inference_performed": False,
                "acoustic_language_gate_passed": False,
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    return report


def _complete_review(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        assert fields is not None
        rows = list(reader)
    for row in rows:
        row.update(
            {
                "review_status": "pass",
                "speech_intelligible": "yes",
                "text_preserved": "yes",
                "language_preserved": "yes",
                "severe_artifacts": "no",
                "notes": "heard in full",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_kazakhtts_acoustic_gate_requires_two_complete_distinct_reviews(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.csv"
    write_kazakhtts_acoustic_packet(
        packet_path, build_kazakhtts_acoustic_packet(_smoke_report(tmp_path))
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_kazakhtts_review_template(first, packet_path, "listener-a")
    write_kazakhtts_review_template(second, packet_path, "listener-b")
    _complete_review(first)
    _complete_review(second)

    report = evaluate_kazakhtts_acoustic_gate(
        packet_path, (*read_kazakhtts_reviews(first), *read_kazakhtts_reviews(second))
    )

    assert report["approved_input_languages"] == ["kk", "mixed", "ru"]
    assert report["all_languages_passed"] is True
    assert report["detector_inference_authorized"] is False


def test_kazakhtts_acoustic_gate_default_forms_are_fail_closed(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.csv"
    write_kazakhtts_acoustic_packet(
        packet_path, build_kazakhtts_acoustic_packet(_smoke_report(tmp_path))
    )
    review = tmp_path / "review.csv"
    write_kazakhtts_review_template(review, packet_path, "listener-a")

    report = evaluate_kazakhtts_acoustic_gate(packet_path, read_kazakhtts_reviews(review))

    assert report["approved_input_languages"] == []
    assert report["all_languages_passed"] is False
