from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kds.data.common_voice import CommonVoiceRecord
from kds.data.manifest import ManifestRow, write_manifest
from scripts.bind_common_voice_ru_v24_silero_v5_5_pre_qa_text import (
    CommonVoiceSileroV55TextBindingError,
    bind_literal_texts,
    load_ready_pre_qa_candidate,
)


def _row(*, text: str = "Точный текст", suffix: str = "") -> ManifestRow:
    return ManifestRow(
        sample_id=f"common_voice_ru_v24:clip{suffix}",
        relative_path="processed/clip.wav",
        sha256=hashlib.sha256(f"asset{suffix}".encode()).hexdigest(),
        split="test",
        label="bonafide",
        language="ru",
        code_switch="unknown",
        parent_group_id=f"common_voice_ru_v24:client:group{suffix}",
        source_name="common_voice_ru_v24",
        source_license="CC0-1.0",
        rights_basis="test rights",
        speaker_pseudo_id=f"common_voice_ru_v24:client:group{suffix}",
        text_id=f"sentence-id{suffix}",
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="crowdsourced_web_recording",
        original_sr=48_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-13T17:12:59+05:00",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(project: Path, manifest: Path, *, ready_rows: int = 75) -> Path:
    raw = project / "data/manifests/raw.csv"
    raw.write_text("raw\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "protocol_id": "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-materialization-v1",
        "archive": {
            "expected_size_bytes": 7008716262,
            "expected_sha256": "9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea",
            "identity_verified_before_metadata_read_and_extraction": True,
        },
        "selection": {"one_record_per_client_group": True, "post_selection_backfill": False},
        "outputs": {
            "raw_manifest": {"path": "data/manifests/raw.csv", "sha256": _sha(raw), "rows": 80},
            "ready_manifest": {
                "path": "data/manifests/ready.csv",
                "sha256": _sha(manifest),
                "rows": ready_rows,
            },
        },
        "technical_qa": {
            "raw_rows": 80,
            "ready_rows": 75,
            "reused_rows": 0,
            "replacement_or_backfill": False,
        },
        "claims": {
            "synthetic_audio_generated": False,
            "acoustic_review_performed": False,
            "detector_inference_performed": False,
            "detector_inference_authorized": False,
            "future_synthesis_must_use_only_ready_frozen_texts": True,
        },
    }
    receipt_path = project / "data/manifests/materialization.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path


def test_load_ready_pre_qa_candidate_requires_pinned_75_row_manifest(tmp_path: Path) -> None:
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    ready = manifests / "ready.csv"
    write_manifest(
        ready,
        [_row(text=f"Точный текст {index}", suffix=f"-{index}") for index in range(75)],
    )
    receipt = _write_receipt(tmp_path, ready)

    rows = load_ready_pre_qa_candidate(
        ready_manifest=ready, materialization_receipt=receipt, project_root=tmp_path
    )

    assert len(rows) == 75


def test_load_ready_pre_qa_candidate_rejects_wrong_ready_count(tmp_path: Path) -> None:
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    ready = manifests / "ready.csv"
    write_manifest(ready, [_row()])
    receipt = _write_receipt(tmp_path, ready, ready_rows=1)

    with pytest.raises(CommonVoiceSileroV55TextBindingError, match="exact 75-row"):
        load_ready_pre_qa_candidate(
            ready_manifest=ready, materialization_receipt=receipt, project_root=tmp_path
        )


def test_bind_literal_texts_rejects_nonliteral_normalization() -> None:
    row = _row(text="Точный текст")
    record = CommonVoiceRecord(
        clip_name="clip.mp3",
        split="test",
        client_id="client",
        sentence_id="sentence-id",
        sentence="Точный  текст",
    )

    with pytest.raises(CommonVoiceSileroV55TextBindingError, match="exact literal"):
        bind_literal_texts((row,), (record,))
