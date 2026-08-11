from __future__ import annotations

import hashlib
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

import kds.data.openstt_rhvoice as rhvoice


def _write_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, BytesIO(payload))


def _fixture_release(
    tmp_path: Path,
    *,
    include_link: bool = False,
    repeat_first: bool = False,
    different_duplicate_payload: bool = False,
) -> tuple[Path, Path]:
    archive = tmp_path / rhvoice.OPENSTT_RHVOICE_ARCHIVE_NAME
    manifest = tmp_path / rhvoice.OPENSTT_RHVOICE_MANIFEST_NAME
    root = rhvoice.OPENSTT_RHVOICE_ARCHIVE_ROOT
    first_row = f"{root}/a/01/first.opus,{root}/a/01/first.txt,0.25\n"
    second_row = f"{root}/b/02/second.opus,{root}/b/02/second.txt,0.50\n"
    manifest.write_text(
        first_row + (first_row if repeat_first else "") + second_row,
        encoding="utf-8",
    )
    with tarfile.open(archive, "w:gz") as tar_archive:
        _write_member(tar_archive, f"{root}/a/01/first.opus", b"opus-one")
        _write_member(tar_archive, f"{root}/a/01/first.txt", b"first")
        if repeat_first:
            duplicate_opus = b"different-opus" if different_duplicate_payload else b"opus-one"
            _write_member(tar_archive, f"{root}/a/01/first.opus", duplicate_opus)
            _write_member(tar_archive, f"{root}/a/01/first.txt", b"first")
        _write_member(tar_archive, f"{root}/b/02/second.opus", b"opus-two")
        _write_member(tar_archive, f"{root}/b/02/second.txt", b"second")
        if include_link:
            link = tarfile.TarInfo(f"{root}/bad-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/outside"
            tar_archive.addfile(link)
    return archive, manifest


def _pin_fixture(monkeypatch: pytest.MonkeyPatch, archive: Path, manifest: Path) -> None:
    monkeypatch.setattr(
        rhvoice, "OPENSTT_RHVOICE_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size
    )
    monkeypatch.setattr(
        rhvoice,
        "OPENSTT_RHVOICE_ARCHIVE_EXPECTED_MD5",
        hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        rhvoice, "OPENSTT_RHVOICE_MANIFEST_EXPECTED_SIZE_BYTES", manifest.stat().st_size
    )
    monkeypatch.setattr(
        rhvoice,
        "OPENSTT_RHVOICE_MANIFEST_EXPECTED_MD5",
        hashlib.md5(manifest.read_bytes(), usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        rhvoice, "OPENSTT_RHVOICE_EXPECTED_MANIFEST_ROWS", len(manifest.read_text().splitlines())
    )


def test_audit_openstt_rhvoice_archive_streams_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest = _fixture_release(tmp_path)
    _pin_fixture(monkeypatch, archive, manifest)

    audit = rhvoice.audit_openstt_rhvoice_archive(archive, manifest)

    assert audit.manifest_rows == 2
    assert audit.manifest_unique_pairs == 2
    assert audit.manifest_duplicate_rows == 0
    assert audit.manifest_duration_sum == "0.75"
    assert audit.archive_opus_files == 2
    assert audit.archive_transcript_files == 2
    assert audit.archive_regular_files == 4


def test_audit_openstt_rhvoice_archive_rejects_nonregular_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest = _fixture_release(tmp_path, include_link=True)
    _pin_fixture(monkeypatch, archive, manifest)

    with pytest.raises(rhvoice.OpenSttRhvoiceAuditError, match="non-regular"):
        rhvoice.audit_openstt_rhvoice_archive(archive, manifest)


def test_audit_openstt_rhvoice_archive_counts_exact_duplicate_manifest_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest = _fixture_release(tmp_path, repeat_first=True)
    _pin_fixture(monkeypatch, archive, manifest)

    audit = rhvoice.audit_openstt_rhvoice_archive(archive, manifest)

    assert audit.manifest_rows == 3
    assert audit.manifest_unique_pairs == 2
    assert audit.manifest_duplicate_paths == 1
    assert audit.manifest_duplicate_rows == 1
    assert audit.archive_duplicate_opus_members == 1
    assert audit.archive_duplicate_transcript_members == 1
    assert audit.archive_duplicate_payloads_verified == 2


def test_audit_openstt_rhvoice_archive_rejects_nonidentical_repeated_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest = _fixture_release(
        tmp_path, repeat_first=True, different_duplicate_payload=True
    )
    _pin_fixture(monkeypatch, archive, manifest)

    with pytest.raises(rhvoice.OpenSttRhvoiceAuditError, match="differs byte-for-byte"):
        rhvoice.audit_openstt_rhvoice_archive(archive, manifest)
