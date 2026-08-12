from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from kds.data.ksc2_mixed_review import (
    AI_REVIEW_FIELDS,
    CURATED_MIXED_DECISIONS_V2_DELTA,
    curated_mixed_rows,
    load_candidate_packet,
    sha256_file,
)
from scripts.publish_ksc2_ai_mixed_review import main as publish_ai_review


def test_curated_ai_review_is_locked_positive_only() -> None:
    packet = load_candidate_packet(
        Path("data/manifests/ksc2_test_mixed_annotation_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_receipt.json"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json"),
    )
    rows = curated_mixed_rows(packet, "2026-08-11T00:00:00Z")

    assert len(packet.rows) == 2632
    assert len(rows) == 32
    assert {row["language"] for row in rows} == {"mixed"}
    assert {row["code_switch"] for row in rows} == {"true"}
    assert {row["review_method"] for row in rows} == {"single_ai_transcript_semantic_review_v1"}
    assert {row["component"] for row in rows} == {
        "Test/podcasts",
        "Test/radio",
        "Test/talkshow",
    }
    assert all(row["ru_evidence_tokens"] and row["kk_evidence_tokens"] for row in rows)


def test_published_ai_review_artifacts_match_the_locked_decisions() -> None:
    packet = load_candidate_packet(
        Path("data/manifests/ksc2_test_mixed_annotation_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_receipt.json"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json"),
    )
    expected_rows = curated_mixed_rows(packet, "2026-08-11T00:00:00Z")
    csv_path = Path("data/manifests/ksc2_test_mixed_ai_review_v1.csv")
    receipt_path = Path("data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(AI_REVIEW_FIELDS)
        assert list(reader) == expected_rows
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["candidate_rows"] == len(packet.rows)
    assert receipt["confirmed_mixed_rows"] == len(expected_rows)
    assert receipt["output_csv_sha256"] == sha256_file(csv_path)


def test_publisher_replays_explicit_ai_review_without_runtime_inference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_csv = tmp_path / "review.csv"
    output_receipt = tmp_path / "receipt.json"
    exit_code = publish_ai_review(
        [
            "--packet",
            "data/manifests/ksc2_test_mixed_annotation_v1.csv",
            "--packet-receipt",
            "data/licenses/ksc2_test_mixed_annotation_v1_receipt.json",
            "--packet-lock",
            "data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json",
            "--reviewed-at",
            "2026-08-11T00:00:00Z",
            "--output-csv",
            str(output_csv),
            "--output-receipt",
            str(output_receipt),
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["explicit_ai_review_decisions"] == 32
    assert result["output_csv_sha256"] == sha256_file(output_csv)
    receipt = json.loads(output_receipt.read_text(encoding="utf-8"))
    assert receipt["rule"].endswith("not a binary training manifest.")


def test_stage_c_delta_is_explicit_and_disjoint_from_v1() -> None:
    packet = load_candidate_packet(
        Path("data/manifests/ksc2_test_mixed_annotation_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_receipt.json"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json"),
    )
    v1 = curated_mixed_rows(packet, "2026-08-11T00:00:00Z")
    delta = curated_mixed_rows(
        packet,
        "2026-08-12T21:00:00+05:00",
        decisions=CURATED_MIXED_DECISIONS_V2_DELTA,
        review_method="single_ai_transcript_semantic_review_v2_delta",
        reviewer="codex_language_review_v2",
    )

    assert len(delta) == 59
    assert {row["annotation_id"] for row in v1}.isdisjoint(
        row["annotation_id"] for row in delta
    )
    assert {row["review_method"] for row in delta} == {
        "single_ai_transcript_semantic_review_v2_delta"
    }
    assert all(row["ru_evidence_tokens"] and row["kk_evidence_tokens"] for row in delta)


def test_publisher_can_publish_stage_c_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_csv = tmp_path / "review_v2_delta.csv"
    output_receipt = tmp_path / "receipt_v2_delta.json"
    exit_code = publish_ai_review(
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
            str(output_csv),
            "--output-receipt",
            str(output_receipt),
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["explicit_ai_review_decisions"] == 59
    receipt = json.loads(output_receipt.read_text(encoding="utf-8"))
    assert receipt["review_method"] == "single_ai_transcript_semantic_review_v2_delta"


def test_published_stage_c_delta_matches_locked_decisions() -> None:
    packet = load_candidate_packet(
        Path("data/manifests/ksc2_test_mixed_annotation_v1.csv"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_receipt.json"),
        Path("data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json"),
    )
    expected = curated_mixed_rows(
        packet,
        "2026-08-12T20:47:13+05:00",
        decisions=CURATED_MIXED_DECISIONS_V2_DELTA,
        review_method="single_ai_transcript_semantic_review_v2_delta",
        reviewer="codex_language_review_v2",
    )
    csv_path = Path("data/manifests/ksc2_test_mixed_ai_review_v2_delta.csv")
    receipt_path = Path("data/licenses/ksc2_test_mixed_ai_review_v2_delta_receipt.json")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == expected
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["confirmed_mixed_rows"] == 59
    assert receipt["output_csv_sha256"] == sha256_file(csv_path)
