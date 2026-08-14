from __future__ import annotations

import hashlib

from kds.data.denis import DENIS_SOURCE_ID, DenisRecord
from kds.eval.denis_selection import DENIS_SINGLE_SPEAKER_GROUP, select_denis_pre_qa_candidate


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _records() -> tuple[DenisRecord, ...]:
    rows: list[DenisRecord] = []
    for category_index, category in enumerate(("General", "Chat", "CustomerService")):
        for index in range(30):
            identity = f"{category}:{index}"
            rows.append(
                DenisRecord(
                    sample_id=f"{DENIS_SOURCE_ID}:{identity}",
                    member_stem=f"ru-RU/{category}/{index:010d}",
                    category=category,
                    literal_text_sha256=_hash(f"literal:{identity}"),
                    whitespace_canonical_text_sha256=_hash(f"canonical:{identity}"),
                    nfkc_whitespace_canonical_text_sha256=_hash(f"nfkc:{identity}"),
                    audio_sha256=_hash(f"audio:{identity}"),
                    audio_size_bytes=1000 + index,
                    decoded_frames=120_000 + category_index * 10_000 + index,
                    sample_rate_hz=48_000,
                    channels=2,
                    decoded_container="OGG",
                    decoded_subtype="OPUS",
                )
            )
    return tuple(rows)


def _screen(record_count: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": "denis-1-0-mdc-source-exposure-screen-v1",
        "source": {
            "source_id": DENIS_SOURCE_ID,
            "records": record_count,
            "source_provided_speaker_groups": 1,
        },
        "strict_single_speaker_group_exclusion": {
            "direct_identity_overlap_found": False,
            "surviving_records": record_count,
            "surviving_source_speaker_groups": 1,
        },
        "historical_likely_speaker_lineage": {"status": "likely_exposed_fail_closed"},
        "claims": {
            "exact_source_sample_audio_and_text_absent_from_historical_project_scope": True,
            "new_direct_human_source": True,
            "historical_likely_speaker_lineage_exposure": True,
            "speaker_independent": False,
            "candidate_selection_performed": False,
            "disk_extraction_performed": False,
            "detector_inference_performed": False,
        },
    }


def test_denis_selection_is_balanced_deterministic_and_single_speaker() -> None:
    records = _records()

    first = select_denis_pre_qa_candidate(
        records=records,
        source_exposure_screen=_screen(len(records)),
        selection_seed="denis-fixture-v1",
        requested_records=79,
        target_pairs=79,
        selected_at="2026-08-14T04:00:00+05:00",
    )
    second = select_denis_pre_qa_candidate(
        records=tuple(reversed(records)),
        source_exposure_screen=_screen(len(records)),
        selection_seed="denis-fixture-v1",
        requested_records=79,
        target_pairs=79,
        selected_at="2026-08-14T04:00:00+05:00",
    )

    assert first == second
    assert len(first.entries) == 79
    counts = first.receipt["selection_policy"]["selected_category_counts"]
    assert sorted(counts.values()) == [26, 26, 27]
    assert {entry.parent_group_id for entry in first.entries} == {
        DENIS_SINGLE_SPEAKER_GROUP
    }
    assert len({entry.sample_id for entry in first.entries}) == 79
    assert len({entry.whitespace_canonical_text_sha256 for entry in first.entries}) == 79
    assert first.receipt["claims"]["speaker_independent"] is False
