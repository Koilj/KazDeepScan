from __future__ import annotations

import hashlib
import io
import tarfile
import wave
from pathlib import Path

import pytest

import kds.data.voxforge as voxforge


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 160)
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _add_directory(archive: tarfile.TarFile, name: str) -> None:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    archive.addfile(member)


def _fixture_archive(tmp_path: Path, *, include_prompt: bool = True) -> Path:
    path = tmp_path / "voxforge-russian-9a8495d3.tar.gz"
    root = "voxforge-ru"
    submission = "tester-20260512-abc"
    gpl = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
    with tarfile.open(path, "w:gz") as archive:
        _add_directory(archive, root)
        _add_directory(archive, f"{root}/{submission}")
        _add_directory(archive, f"{root}/{submission}/wav")
        _add_bytes(archive, f"{root}/{submission}/wav/ru_0001.wav", _wav_bytes())
        _add_directory(archive, f"{root}/{submission}/etc")
        _add_bytes(archive, f"{root}/{submission}/etc/GPL_license.txt", gpl.encode())
        _add_bytes(archive, f"{root}/{submission}/etc/README", b"User Name:Tester\n")
        _add_bytes(
            archive,
            f"{root}/{submission}/etc/PROMPTS",
            (f"{submission}/mfc/ru_0001 Test prompt\n" if include_prompt else "").encode(),
        )
        _add_bytes(
            archive,
            f"{root}/{submission}/etc/prompts-original",
            b"ru_0001 Test prompt.\n",
        )
        _add_bytes(archive, f"{root}/{submission}/LICENSE", gpl.encode())
    return path


def _pin_fixture(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(voxforge, "VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES", path.stat().st_size)
    monkeypatch.setattr(
        voxforge,
        "VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_audit_voxforge_ru_archive_validates_transcript_bound_wav_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture_archive(tmp_path)
    _pin_fixture(monkeypatch, path)

    audit = voxforge.audit_voxforge_ru_archive(path)

    assert audit.intake_status == "accepted_source_level_only"
    assert audit.submissions == 1
    assert audit.wav_files == audit.prompt_rows == audit.original_prompt_rows == 1
    assert audit.source_provided_contributor_groups == 1
    assert audit.sample_rates_hz == {"16000": 1}
    assert audit.extraction_performed is False


def test_audit_voxforge_ru_archive_rejects_wav_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture_archive(tmp_path, include_prompt=False)
    _pin_fixture(monkeypatch, path)

    with pytest.raises(voxforge.VoxForgeRuAuditError, match="no prompt rows"):
        voxforge.audit_voxforge_ru_archive(path)
