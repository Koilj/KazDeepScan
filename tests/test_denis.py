from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import kds.data.denis as denis


def _opus_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.zeros((4_800, 2), dtype=np.float32),
        48_000,
        format="OGG",
        subtype="OPUS",
    )
    return buffer.getvalue()


def _add_directory(archive: tarfile.TarFile, name: str) -> None:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    archive.addfile(member)


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _fixture_archive(tmp_path: Path, *, include_audio: bool = True) -> Path:
    path = tmp_path / "browser-download-name.tar.gz"
    category = "fixture_General"
    with tarfile.open(path, "w:gz") as archive:
        _add_directory(archive, denis.DENIS_ARCHIVE_ROOT)
        _add_directory(archive, f"{denis.DENIS_ARCHIVE_ROOT}/{category}")
        stem = f"{denis.DENIS_ARCHIVE_ROOT}/{category}/0000000001"
        _add_bytes(archive, f"{stem}.txt", "Тестовая строка\u00a0 ".encode())
        if include_audio:
            _add_bytes(archive, f"{stem}.webm", _opus_bytes())
    return path


def _pin_fixture(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(denis, "DENIS_ARCHIVE_EXPECTED_SIZE_BYTES", path.stat().st_size)
    monkeypatch.setattr(
        denis,
        "DENIS_ARCHIVE_EXPECTED_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(denis, "DENIS_EXPECTED_RECORDS_BY_CATEGORY", {"fixture_General": 1})


def test_audit_denis_archive_accepts_paired_ogg_opus_with_webm_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture_archive(tmp_path)
    _pin_fixture(monkeypatch, path)

    audit = denis.audit_denis_archive(path)

    assert audit.paired_records == 1
    assert audit.literal_unique_texts == 1
    assert audit.decoded_container_counts == {"OGG": 1}
    assert audit.decoded_subtype_counts == {"OPUS": 1}
    assert audit.sample_rate_counts_hz == {"48000": 1}
    assert audit.channel_counts == {"2": 1}
    assert audit.gzip_crc_verified is True
    assert audit.gzip_uncompressed_bytes > 0
    assert audit.tar_stream_fully_read is True
    assert audit.text_members_with_nbsp == 1
    assert audit.text_members_with_trailing_whitespace == 1
    assert audit.disk_extraction_performed is False
    assert audit.detector_inference_performed is False


def test_audit_denis_archive_rejects_unpaired_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture_archive(tmp_path, include_audio=False)
    _pin_fixture(monkeypatch, path)

    with pytest.raises(denis.DenisArchiveAuditError, match="unpaired"):
        denis.audit_denis_archive(path)
