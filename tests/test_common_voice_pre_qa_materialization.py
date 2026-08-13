from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from kds.data.assets import sha256_file
from kds.data.common_voice import CommonVoiceRecord
from scripts.materialize_common_voice_ru_v24_silero_v5_5_pre_qa import (
    FrozenPreQaSelectionRow,
    bind_frozen_pre_qa_selection,
    load_frozen_pre_qa_selection,
)


def _write_frozen_selection(project: Path) -> tuple[Path, Path]:
    manifests = project / "data" / "manifests"
    model_lock = project / "configs" / "research"
    manifests.mkdir(parents=True)
    model_lock.mkdir(parents=True)
    parent_paths = {
        "metadata_exposure_screen": manifests / "metadata.json",
        "literal_text_screen": manifests / "literal.json",
        "silero_v5_5_model_lock": model_lock / "model.json",
    }
    for path in parent_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    selection = manifests / "selection.csv"
    text_hash = hashlib.sha256("Точный текст".encode()).hexdigest()
    with selection.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "selection_rank",
                "sample_id",
                "clip_name",
                "source_split",
                "parent_group_id",
                "speaker_pseudo_id",
                "text_id",
                "text_hash",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "selection_rank": 1,
                "sample_id": "common_voice_ru_v24:clip-1",
                "clip_name": "clip-1.mp3",
                "source_split": "test",
                "parent_group_id": "common_voice_ru_v24:client:client-1",
                "speaker_pseudo_id": "common_voice_ru_v24:client:client-1",
                "text_id": "sentence-1",
                "text_hash": text_hash,
            }
        )
    receipt = manifests / "selection.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-selection-v1",
                "output_selection": {
                    "path": "data/manifests/selection.csv",
                    "sha256": sha256_file(selection),
                    "rows": 1,
                },
                "selection_policy": {
                    "kind": "seeded_two_stage_one_record_per_client_group",
                    "selected_records": 1,
                    "selected_client_groups": 1,
                    "post_selection_backfill": False,
                    "selection_uses_audio_or_duration": False,
                    "selection_uses_detector_or_model_output": False,
                    "selection_uses_model_metrics_or_final_errors": False,
                },
                "claims": {
                    "selection_frozen": True,
                    "audio_extraction_performed": False,
                    "future_extraction_must_use_only_selected_clip_names": True,
                    "qa_rejects_must_not_trigger_backfill": True,
                },
                "inputs": {
                    name: {
                        "path": path.relative_to(project).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for name, path in parent_paths.items()
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return selection, receipt


def test_frozen_pre_qa_selection_binds_exact_archive_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    selection_csv, selection_receipt = _write_frozen_selection(project)
    monkeypatch.chdir(project)

    selection = load_frozen_pre_qa_selection(
        selection_csv.relative_to(project), selection_receipt.relative_to(project), Path(".")
    )
    record = CommonVoiceRecord(
        clip_name="clip-1.mp3",
        split="test",
        client_id="client-1",
        sentence_id="sentence-1",
        sentence="Точный текст",
    )

    assert selection == (
        FrozenPreQaSelectionRow(
            selection_rank=1,
            sample_id="common_voice_ru_v24:clip-1",
            clip_name="clip-1.mp3",
            source_split="test",
            parent_group_id="common_voice_ru_v24:client:client-1",
            speaker_pseudo_id="common_voice_ru_v24:client:client-1",
            text_id="sentence-1",
            text_hash=hashlib.sha256("Точный текст".encode()).hexdigest(),
        ),
    )
    assert bind_frozen_pre_qa_selection(selection, [record]) == (record,)


def test_frozen_pre_qa_selection_refuses_changed_archive_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    selection_csv, selection_receipt = _write_frozen_selection(project)
    monkeypatch.chdir(project)
    selection = load_frozen_pre_qa_selection(
        selection_csv.relative_to(project), selection_receipt.relative_to(project), Path(".")
    )
    changed_record = CommonVoiceRecord(
        clip_name="clip-1.mp3",
        split="test",
        client_id="client-1",
        sentence_id="sentence-1",
        sentence="Измененный текст",
    )

    with pytest.raises(ValueError, match="differs"):
        bind_frozen_pre_qa_selection(selection, [changed_record])
