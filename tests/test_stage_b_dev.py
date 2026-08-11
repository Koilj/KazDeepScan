from __future__ import annotations

from dataclasses import replace

from kds.data.manifest import ManifestRow
from kds.data.stage_b_dev import filter_stage_b_calibration_rows, filter_stage_b_dev_rows
from tests.factories import manifest_mapping


def _row(sample_id: str, split: str, label: str, text_hash: str) -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id=sample_id,
            relative_path=f"{sample_id}.wav",
            sha256=(sample_id.encode().hex() + "0" * 64)[:64],
            split=split,
            label=label,
            parent_group_id=f"parent-{sample_id}",
            speaker_pseudo_id=f"speaker-{sample_id}",
            text_id=f"text-{sample_id}",
            text_hash=text_hash,
            source_name="train-source" if split == "train" else "dev-source",
            generator_family="tts" if label == "spoof" else "",
            generator_name="test-tts" if label == "spoof" else "",
            generator_version="1" if label == "spoof" else "",
            voice_id=f"voice-{sample_id}" if label == "spoof" else "",
        ),
        row_number=2,
    )


def test_stage_b_dev_filter_removes_complete_text_group_shared_with_train() -> None:
    train = [
        _row("train-real", "train", "bonafide", "a" * 64),
        _row("train-fake", "train", "spoof", "b" * 64),
    ]
    candidate = [
        _row("shared-real", "dev", "bonafide", "a" * 64),
        _row("paired-fake", "dev", "spoof", "a" * 64),
        _row("fresh-real", "dev", "bonafide", "c" * 64),
        _row("fresh-fake", "dev", "spoof", "d" * 64),
    ]

    selected, report = filter_stage_b_dev_rows(train, candidate)

    assert {row.sample_id for row in selected} == {"fresh-real", "fresh-fake"}
    assert report.excluded_rows == 2
    assert report.reason_counts == {"text_hash": 2}


def test_stage_b_dev_filter_removes_duplicate_asset() -> None:
    train = [
        _row("train-real", "train", "bonafide", "a" * 64),
        _row("train-fake", "train", "spoof", "b" * 64),
    ]
    candidate = [
        replace(
            _row("dev-real", "dev", "bonafide", "c" * 64),
            sha256=train[0].sha256,
        ),
        _row("fresh-real", "dev", "bonafide", "e" * 64),
        _row("fresh-fake", "dev", "spoof", "d" * 64),
    ]

    selected, report = filter_stage_b_dev_rows(train, candidate)

    assert {row.sample_id for row in selected} == {"fresh-real", "fresh-fake"}
    assert report.reason_counts == {"sha256": 1}


def test_stage_b_calibration_filter_excludes_all_earlier_dev_roles() -> None:
    history = [
        _row("stage-b-train", "train", "bonafide", "a" * 64),
        _row("stage-a-dev", "dev", "spoof", "b" * 64),
        _row("stage-b-dev", "dev", "bonafide", "c" * 64),
    ]
    candidate = [
        _row("shares-stage-a-text", "dev", "bonafide", "b" * 64),
        _row("paired-stage-a-text", "dev", "spoof", "b" * 64),
        _row("fresh-real", "dev", "bonafide", "d" * 64),
        _row("fresh-fake", "dev", "spoof", "e" * 64),
    ]

    selected, report = filter_stage_b_calibration_rows(history, candidate)

    assert {row.sample_id for row in selected} == {"fresh-real", "fresh-fake"}
    assert report.excluded_rows == 2
    assert report.reason_counts == {"text_hash": 2}
