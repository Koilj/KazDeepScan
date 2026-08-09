from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from kds.data.ruasd_catalog import (
    RuAsdCatalogError,
    audit_ruasd_collection,
    load_ruasd_artifact_catalog,
    write_ruasd_audit_report,
)


def _write_archive(path: Path) -> None:
    records = (
        {
            "sample_id": "raw_real_common_voice_a",
            "label": "real",
            "group": "raw",
            "subset": "CommonVoice",
            "source_type": "real_speech",
            "filename": "raw_real_common_voice_a.wav",
            "audio_relpath": "raw/real/CommonVoice/audio/raw_real_common_voice_a.wav",
            "speakers": -1,
            "transcription": "Проверочная фраза",
        },
        {
            "sample_id": "raw_fake_vosk_a",
            "label": "fake",
            "group": "raw",
            "subset": "voskTTS",
            "source_type": "tts",
            "filename": "raw_fake_vosk_a.wav",
            "audio_relpath": "raw/fake/voskTTS/audio/raw_fake_vosk_a.wav",
            "speakers": 4,
            "model": "vosk-model-tts-ru",
            "true_lines": "Проверочная фраза",
        },
        {
            "sample_id": "augmented_real_a",
            "label": "real",
            "group": "augmented",
            "subset": "",
            "source_type": "augmented_audio",
            "filename": "augmented_real_a.wav",
            "audio_relpath": "augmented/real/augmented_real_a.wav",
            "speakers": None,
        },
    )
    with tarfile.open(path, mode="w") as archive:
        for record in records:
            payload = json.dumps(record).encode()
            metadata = tarfile.TarInfo(f"{record['sample_id']}.json")
            metadata.size = len(payload)
            archive.addfile(metadata, io.BytesIO(payload))
            audio = b"minimal-wav-bytes"
            audio_info = tarfile.TarInfo(f"{record['sample_id']}.wav")
            audio_info.size = len(audio)
            archive.addfile(audio_info, io.BytesIO(audio))


def _write_catalog(path: Path, archive: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "archive_name",
                "expected_size_bytes",
                "sha256",
                "pinned_revision",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "archive_name": archive.name,
                "expected_size_bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "pinned_revision": "revision",
                "source_url": "https://example.test/ruasd",
            }
        )


def test_collection_audit_counts_metadata_without_extracting_audio(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "ruasd-000000.tar"
    _write_archive(archive)
    catalog_path = tmp_path / "catalog.csv"
    _write_catalog(catalog_path, archive)

    events: list[tuple[int, int, str]] = []
    audit = audit_ruasd_collection(
        archive_dir,
        load_ruasd_artifact_catalog(catalog_path),
        verify_sha256=True,
        progress_callback=lambda completed, total, name: events.append((completed, total, name)),
    )

    assert audit.archive_count == 1
    assert audit.sha256_verified_archives == 1
    assert audit.records == 3
    assert audit.record_counts == {
        "fake/raw/tts": 1,
        "real/augmented/augmented_audio": 1,
        "real/raw/real_speech": 1,
    }
    assert audit.speaker_counts == {
        "fake/raw/known": 1,
        "real/augmented/unknown": 1,
        "real/raw/unknown": 1,
    }
    assert audit.text_counts["fake/raw/source_text_present"] == 1
    assert audit.text_counts["real/augmented/source_text_missing"] == 1
    assert events == [(1, 1, "ruasd-000000.tar")]


def test_collection_audit_rejects_missing_pinned_archive(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "ruasd-000000.tar"
    _write_archive(archive)
    catalog_path = tmp_path / "catalog.csv"
    _write_catalog(catalog_path, archive)
    content = catalog_path.read_text(encoding="utf-8")
    catalog_path.write_text(
        content
        + "ruasd-000001.tar,1,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,revision,https://example.test/ruasd\n",
        encoding="utf-8",
    )

    with pytest.raises(RuAsdCatalogError, match="missing=ruasd-000001.tar"):
        audit_ruasd_collection(archive_dir, load_ruasd_artifact_catalog(catalog_path))


def test_audit_report_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "ruasd-000000.tar"
    _write_archive(archive)
    catalog_path = tmp_path / "catalog.csv"
    _write_catalog(catalog_path, archive)
    audit = audit_ruasd_collection(archive_dir, load_ruasd_artifact_catalog(catalog_path))
    report = tmp_path / "report.json"

    write_ruasd_audit_report(report, audit)

    assert json.loads(report.read_text(encoding="utf-8"))["records"] == 3
    with pytest.raises(RuAsdCatalogError, match="Unsafe RuASD audit report destination"):
        write_ruasd_audit_report(report, audit)
