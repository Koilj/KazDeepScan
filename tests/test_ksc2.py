from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from kds.data.ksc2 import (
    KSC2_ARCHIVE_BASENAME,
    Ksc2AuditError,
    audit_ksc2_archive,
    extract_ksc2_mixed_annotation_candidates,
    extract_ksc2_selected_audio,
    scan_ksc2_text_candidates,
    write_ksc2_audit_report,
)


def _write_parts(directory: Path, archive_members: list[tuple[str, bytes]]) -> tuple[int, ...]:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        root = tarfile.TarInfo("ISSAI_KSC2/")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        directory_member = tarfile.TarInfo("ISSAI_KSC2/Test/crowdsourced/")
        directory_member.type = tarfile.DIRTYPE
        archive.addfile(directory_member)
        for name, payload in archive_members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    compressed = gzip.compress(stream.getvalue())
    split = len(compressed) // 2
    suffixes = (
        "partaa",
        "partab",
        "partac",
        "partad",
        "partae",
        "partaf",
        "partag",
        "partah",
        "partai",
        "partaj",
    )
    chunks = [compressed[:split], compressed[split:]] + [b""] * 8
    for suffix, payload in zip(suffixes, chunks, strict=True):
        (directory / f"{KSC2_ARCHIVE_BASENAME}.{suffix}").write_bytes(payload)
    return tuple(len(chunk) for chunk in chunks)


def test_ksc2_audit_streams_parts_once_and_collects_layout(tmp_path: Path) -> None:
    expected_sizes = _write_parts(
        tmp_path,
        [
            ("ISSAI_KSC2/Test/crowdsourced/a.flac", b"flac"),
            ("ISSAI_KSC2/Test/crowdsourced/a.txt", b"text"),
            ("ISSAI_KSC2/Test/crowdsourced/b.flac.flac", b"flac"),
            ("ISSAI_KSC2/Test/crowdsourced/b.txt.txt", b"text"),
            ("ISSAI_KSC2/Test/metadata.csv", b"id,text\n1,hello\n"),
        ],
    )

    report = audit_ksc2_archive(tmp_path, expected_sizes=expected_sizes)

    assert report.regular_files == 5
    assert report.files_by_extension == {".csv": 1, ".flac": 2, ".txt": 2}
    assert report.files_by_component == {"Test/crowdsourced": 4, "Test/metadata.csv": 1}
    assert report.audio_files == 2
    assert report.transcript_files == 2
    assert report.unpaired_audio_examples == ()
    assert report.unpaired_transcript_examples == ()
    assert report.metadata_files == 1
    assert report.metadata_member_examples == ("ISSAI_KSC2/Test/metadata.csv",)
    assert len(report.parts) == 10
    assert len(report.compressed_sha256) == 64


def test_ksc2_audit_reports_part_progress(tmp_path: Path) -> None:
    expected_sizes = _write_parts(
        tmp_path,
        [
            ("ISSAI_KSC2/Test/crowdsourced/a.flac", b"flac"),
            ("ISSAI_KSC2/Test/crowdsourced/a.txt", b"text"),
        ],
    )
    events: list[tuple[int, int, str]] = []

    audit_ksc2_archive(
        tmp_path,
        expected_sizes=expected_sizes,
        progress_callback=lambda completed, total, name: events.append(
            (completed, total, name)
        ),
    )

    assert events[0] == (1, 10, f"{KSC2_ARCHIVE_BASENAME}.partaa")
    assert events[-1] == (10, 10, f"{KSC2_ARCHIVE_BASENAME}.partaj")


def test_ksc2_audit_report_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    expected_sizes = _write_parts(
        tmp_path,
        [
            ("ISSAI_KSC2/Test/crowdsourced/a.flac", b"flac"),
            ("ISSAI_KSC2/Test/crowdsourced/a.txt", b"text"),
        ],
    )
    audit = audit_ksc2_archive(tmp_path, expected_sizes=expected_sizes)
    report = tmp_path / "report.json"

    write_ksc2_audit_report(report, audit)

    assert json.loads(report.read_text(encoding="utf-8"))["audio_files"] == 1
    with pytest.raises(Ksc2AuditError, match="Unsafe KSC2 audit report destination"):
        write_ksc2_audit_report(report, audit)


def test_ksc2_text_scan_keeps_only_allowed_paired_train_components(tmp_path: Path) -> None:
    expected_sizes = _write_parts(
        tmp_path,
        [
            ("ISSAI_KSC2/Train/radio/a.flac", b"flac"),
            ("ISSAI_KSC2/Train/radio/a.txt", "  Қазақ   мәтіні\n".encode()),
            ("ISSAI_KSC2/Train/radio/orphan.txt", b"orphan"),
            ("ISSAI_KSC2/Test/crowdsourced/b.flac", b"flac"),
            ("ISSAI_KSC2/Test/crowdsourced/b.txt", b"ignored"),
        ],
    )
    archive_hash = audit_ksc2_archive(tmp_path, expected_sizes=expected_sizes).compressed_sha256

    candidates = scan_ksc2_text_candidates(
        tmp_path,
        allowed_components=frozenset({"Train/radio"}),
        expected_compressed_sha256=archive_hash,
        expected_sizes=expected_sizes,
    )

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "Train/radio/a"
    assert candidates[0].canonical_text_sha256 == hashlib.sha256(
        "Қазақ мәтіні".encode()
    ).hexdigest()


def test_ksc2_selected_audio_extraction_is_allow_listed_and_hash_bound(
    tmp_path: Path,
) -> None:
    expected_sizes = _write_parts(
        tmp_path,
        [
            ("ISSAI_KSC2/Train/radio/a.flac", b"selected-flac"),
            ("ISSAI_KSC2/Train/radio/a.txt", b"selected text"),
            ("ISSAI_KSC2/Train/radio/b.flac", b"not-selected"),
        ],
    )
    archive_hash = audit_ksc2_archive(tmp_path, expected_sizes=expected_sizes).compressed_sha256
    output = tmp_path / "selected"

    extracted = extract_ksc2_selected_audio(
        tmp_path,
        output,
        selected_members=frozenset({"ISSAI_KSC2/Train/radio/a.flac"}),
        expected_compressed_sha256=archive_hash,
        expected_sizes=expected_sizes,
    )

    assert len(extracted) == 1
    assert extracted[0].sha256 == hashlib.sha256(b"selected-flac").hexdigest()
    assert (output / "Train/radio/a.flac").read_bytes() == b"selected-flac"
    assert not (output / "Train/radio/b.flac").exists()


def test_ksc2_audit_rejects_symlink_members(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        root = tarfile.TarInfo("ISSAI_KSC2/")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        link = tarfile.TarInfo("ISSAI_KSC2/Test/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    compressed = gzip.compress(stream.getvalue())
    suffixes = (
        "partaa",
        "partab",
        "partac",
        "partad",
        "partae",
        "partaf",
        "partag",
        "partah",
        "partai",
        "partaj",
    )
    chunks = [compressed] + [b""] * 9
    for suffix, payload in zip(suffixes, chunks, strict=True):
        (tmp_path / f"{KSC2_ARCHIVE_BASENAME}.{suffix}").write_bytes(payload)

    with pytest.raises(Ksc2AuditError, match="unsupported member type"):
        audit_ksc2_archive(tmp_path, expected_sizes=tuple(len(chunk) for chunk in chunks))


def test_ksc2_annotation_extracts_only_priority_components_as_unlabelled_candidates(
    tmp_path: Path,
) -> None:
    expected_sizes = _write_parts(
        tmp_path,
        [
            ("ISSAI_KSC2/Test/podcasts/p.flac", b"podcast-audio"),
            ("ISSAI_KSC2/Test/podcasts/p.txt", "подкаст текст".encode()),
            ("ISSAI_KSC2/Test/talkshow/t.flac", b"talkshow-audio"),
            ("ISSAI_KSC2/Test/talkshow/t.txt", "ток-шоу текст".encode()),
            ("ISSAI_KSC2/Test/radio/r.flac", b"radio-audio"),
            ("ISSAI_KSC2/Test/radio/r.txt", "радио текст".encode()),
            ("ISSAI_KSC2/Test/tv_news/n.flac", b"news-audio"),
            ("ISSAI_KSC2/Test/tv_news/n.txt", "новости".encode()),
        ],
    )
    archive_hash = audit_ksc2_archive(tmp_path, expected_sizes=expected_sizes).compressed_sha256

    output = tmp_path / "annotation-stage"
    candidates = extract_ksc2_mixed_annotation_candidates(
        tmp_path,
        output,
        expected_compressed_sha256=archive_hash,
        expected_sizes=expected_sizes,
    )

    assert [candidate.candidate_id for candidate in candidates] == [
        "Test/podcasts/p",
        "Test/radio/r",
        "Test/talkshow/t",
    ]
    assert {candidate.component for candidate in candidates} == {
        "Test/podcasts",
        "Test/radio",
        "Test/talkshow",
    }
    assert {candidate.transcript for candidate in candidates} == {
        "подкаст текст",
        "ток-шоу текст",
        "радио текст",
    }
    for candidate in candidates:
        assert (output / candidate.audio_relative_path).is_file()
        assert "tv_news" not in candidate.audio_relative_path
