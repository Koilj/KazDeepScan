from __future__ import annotations

from pathlib import Path

from kds.data.ksc2 import Ksc2AnnotationCandidate
from scripts.prepare_ksc2_mixed_annotation import annotation_rows, load_ksc2_annotation_lock


def test_annotation_rows_are_explicitly_pending_not_auto_labelled() -> None:
    rows = annotation_rows(
        [
            Ksc2AnnotationCandidate(
                candidate_id="Test/podcasts/episode_1",
                component="Test/podcasts",
                archive_audio_member="ISSAI_KSC2/Test/podcasts/episode_1.flac",
                archive_transcript_member="ISSAI_KSC2/Test/podcasts/episode_1.txt",
                audio_relative_path="assets/Test/podcasts/asset.flac",
                audio_sha256="a" * 64,
                transcript="бір сөз",
                transcript_sha256="b" * 64,
            )
        ],
        slice_name="mixed_annotation_v1",
        archive_sha256="c" * 64,
        source_license="CC-BY-4.0",
        source_lock_sha256="d" * 64,
    )

    assert rows[0]["annotation_state"] == "pending"
    assert rows[0]["language"] == "unknown"
    assert rows[0]["code_switch"] == "unknown"
    assert rows[0]["audio_relative_path"] == (
        "raw/ksc2_v1/slices/mixed_annotation_v1/assets/Test/podcasts/asset.flac"
    )


def test_ksc2_annotation_lock_reads_pinned_archive_contract() -> None:
    archive_hash, license_value, lock_hash = load_ksc2_annotation_lock(
        Path("data/licenses/ksc2_v1_artifact_lock.json")
    )

    assert archive_hash == "43d1ee6725d737a438125a13997a0abde4159de84ef17d1706fe7921e8632cbe"
    assert license_value == "CC-BY-4.0"
    assert len(lock_hash) == 64
