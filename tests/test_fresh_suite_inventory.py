from __future__ import annotations

from dataclasses import replace

import pytest

from kds.data.fleurs import FleursRecord
from kds.data.manifest import ManifestRow
from kds.eval.fresh_suite_inventory import (
    FreshSuiteInventoryError,
    audit_fleurs_locale_inventory,
    audit_ksc2_mixed_inventory,
)


def _manifest_row(
    sample_id: str,
    text_hash: str,
    *,
    source_name: str,
    language: str,
    code_switch: str = "false",
) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        relative_path=f"processed/{text_hash[:2]}/{text_hash}.wav",
        sha256="a" * 64,
        split="test",
        label="bonafide",
        language=language,
        code_switch=code_switch,
        parent_group_id=f"{source_name}:group:{text_hash}",
        source_name=source_name,
        source_license="CC-BY-4.0",
        rights_basis="test",
        speaker_pseudo_id=f"{source_name}:unknown",
        text_id=f"{source_name}:text:{text_hash}",
        text_hash=text_hash,
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="source",
        capture_route="source",
        original_sr=16_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-12T00:00:00Z",
    )


def _fleurs_record(filename: str, transcript: str) -> FleursRecord:
    return FleursRecord(
        locale="ru_ru",
        language="ru",
        source_split="test",
        prompt_id=filename.removesuffix(".wav"),
        filename=filename,
        raw_transcript=transcript,
        transcript=transcript,
        character_transcript=" ".join(transcript),
        samples=16_000,
        gender="FEMALE",
    )


def test_fleurs_inventory_separates_release_capacity_from_ready_capacity() -> None:
    records = [_fleurs_record("1.wav", "один"), _fleurs_record("2.wav", "два")]
    ready = [
        _manifest_row(
            "google_fleurs_ru_v1:1",
            records[0].text_hash,
            source_name="google_fleurs_ru_v1",
            language="ru",
        )
    ]

    result = audit_fleurs_locale_inventory(
        locale="ru_ru", test_records=records, ready_rows=ready, exposed_rows=ready
    )

    assert result["source_test_unique_text_groups"] == 2
    assert result["fresh_release_unique_text_groups_before_extraction_and_qa"] == 1
    assert result["fresh_qa_ready_unique_text_groups"] == 0
    assert result["all_current_qa_ready_text_groups_exposed"] is True


def test_fleurs_inventory_rejects_changed_release_binding() -> None:
    record = _fleurs_record("1.wav", "один")
    row = _manifest_row(
        "google_fleurs_ru_v1:1",
        "b" * 64,
        source_name="google_fleurs_ru_v1",
        language="ru",
    )

    with pytest.raises(FreshSuiteInventoryError, match="not bound"):
        audit_fleurs_locale_inventory(
            locale="ru_ru", test_records=[record], ready_rows=[row], exposed_rows=[row]
        )


def _candidate(annotation_id: str, transcript_hash: str) -> dict[str, str]:
    return {
        "annotation_id": annotation_id,
        "annotation_state": "pending",
        "language": "unknown",
        "code_switch": "unknown",
        "source_name": "ksc2_v1",
        "audio_sha256": "c" * 64,
        "transcript_sha256": transcript_hash,
        "archive_audio_member": f"ISSAI_KSC2/{annotation_id}.flac",
    }


def _review(candidate: dict[str, str]) -> dict[str, str]:
    return {
        **candidate,
        "language": "mixed",
        "code_switch": "true",
    }


def test_mixed_inventory_reports_one_unexposed_ready_asset() -> None:
    first_hash = "d" * 64
    second_hash = "e" * 64
    candidates = [
        _candidate("ksc2_v1:first", first_hash),
        _candidate("ksc2_v1:second", second_hash),
    ]
    reviews = [_review(row) for row in candidates]
    ready = [
        _manifest_row(
            "ksc2_v1:first",
            first_hash,
            source_name="ksc2_v1",
            language="mixed",
            code_switch="true",
        ),
        _manifest_row(
            "ksc2_v1:second",
            second_hash,
            source_name="ksc2_v1",
            language="mixed",
            code_switch="true",
        ),
    ]

    result = audit_ksc2_mixed_inventory(
        candidate_rows=candidates,
        reviewed_rows=reviews,
        ready_rows=ready,
        exposed_rows=[ready[0]],
    )

    assert result["pending_semantic_review_rows"] == 0
    assert result["fresh_qa_ready_mixed_rows"] == 1
    assert result["fresh_qa_ready_annotation_ids"] == ["ksc2_v1:second"]


def test_mixed_inventory_rejects_unreviewed_ready_asset() -> None:
    transcript_hash = "f" * 64
    candidate = _candidate("ksc2_v1:first", transcript_hash)
    ready = _manifest_row(
        "ksc2_v1:first",
        transcript_hash,
        source_name="ksc2_v1",
        language="mixed",
        code_switch="true",
    )
    unrelated = replace(ready, sample_id="ksc2_v1:unrelated")

    with pytest.raises(FreshSuiteInventoryError, match="without semantic review"):
        audit_ksc2_mixed_inventory(
            candidate_rows=[candidate],
            reviewed_rows=[_review(candidate)],
            ready_rows=[ready, unrelated],
            exposed_rows=[ready],
        )
