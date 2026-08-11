"""Read-only intake audit for the OpenSTT RHVoice Russian-address archive.

This module deliberately validates only the downloaded release artifact and its
official, headerless manifest.  It does not extract audio, create a training or
evaluation manifest, or establish source independence for a final protocol.
"""

from __future__ import annotations

import csv
import hashlib
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

OPENSTT_RHVOICE_SOURCE_ID = "openstt_rhvoice_addresses_v1"
OPENSTT_RHVOICE_ARCHIVE_NAME = "tts_russian_addresses_rhvoice_4voices.tar.gz"
OPENSTT_RHVOICE_MANIFEST_NAME = "tts_russian_addresses_rhvoice_4voices.csv"
OPENSTT_RHVOICE_ARCHIVE_ROOT = "tts_russian_addresses_rhvoice_4voices"
OPENSTT_RHVOICE_ARCHIVE_EXPECTED_SIZE_BYTES = 13_862_699_423
OPENSTT_RHVOICE_ARCHIVE_EXPECTED_MD5 = "2bdd0e26d972f60a0e54dafeef642264"
OPENSTT_RHVOICE_MANIFEST_EXPECTED_SIZE_BYTES = 220_255_453
OPENSTT_RHVOICE_MANIFEST_EXPECTED_MD5 = "628c2974eeb2edfba4a560445d9dc628"
OPENSTT_RHVOICE_EXPECTED_MANIFEST_ROWS = 1_741_838


class OpenSttRhvoiceAuditError(ValueError):
    """Raised when the downloaded OpenSTT RHVoice release is not exactly as expected."""


@dataclass(frozen=True, slots=True)
class OpenSttRhvoiceArtifactReceipt:
    filename: str
    size_bytes: int
    md5: str


@dataclass(frozen=True, slots=True)
class OpenSttRhvoiceArchiveAudit:
    source_id: str
    archive: OpenSttRhvoiceArtifactReceipt
    manifest: OpenSttRhvoiceArtifactReceipt
    manifest_rows: int
    manifest_unique_pairs: int
    manifest_duplicate_paths: int
    manifest_duplicate_rows: int
    manifest_duration_sum: str
    archive_directories: int
    archive_regular_files: int
    archive_content_bytes: int
    archive_opus_files: int
    archive_transcript_files: int
    archive_duplicate_opus_members: int
    archive_duplicate_transcript_members: int
    archive_duplicate_payloads_verified: int


@dataclass(frozen=True, slots=True)
class _ManifestInventory:
    rows: int
    unique_pairs: int
    duplicate_paths: int
    duplicate_rows: int
    duration_sum: Decimal


@dataclass(frozen=True, slots=True)
class _ArchiveInventory:
    directories: int
    regular_files: int
    content_bytes: int
    opus_files: int
    transcript_files: int
    duplicate_opus_members: int
    duplicate_transcript_members: int
    duplicate_payloads_verified: int


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact(
    path: Path, *, expected_name: str, expected_size: int, expected_md5: str
) -> OpenSttRhvoiceArtifactReceipt:
    if not path.is_file():
        raise OpenSttRhvoiceAuditError(f"OpenSTT RHVoice artifact does not exist: {path}")
    if path.name != expected_name:
        raise OpenSttRhvoiceAuditError(
            f"Unexpected OpenSTT RHVoice filename: expected {expected_name!r}, got {path.name!r}."
        )
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise OpenSttRhvoiceAuditError(
            f"OpenSTT RHVoice size mismatch for {path.name}: expected {expected_size}, "
            f"got {actual_size}."
        )
    actual_md5 = _md5_file(path)
    if actual_md5 != expected_md5:
        raise OpenSttRhvoiceAuditError(
            f"OpenSTT RHVoice MD5 mismatch for {path.name}: expected {expected_md5}, "
            f"got {actual_md5}."
        )
    return OpenSttRhvoiceArtifactReceipt(path.name, actual_size, actual_md5)


def _safe_path(value: str, *, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or not path.parts
        or path.parts[0] != OPENSTT_RHVOICE_ARCHIVE_ROOT
    ):
        raise OpenSttRhvoiceAuditError(f"{context}: unsafe or unexpected path {value!r}.")
    return path


def _prepare_membership_database(connection: sqlite3.Connection) -> None:
    """Create a bounded-memory, temporary index for complete archive membership."""

    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        PRAGMA cache_size = -32768;
        CREATE TABLE expected (
            stem TEXT PRIMARY KEY,
            duration TEXT NOT NULL,
            duplicate_path INTEGER NOT NULL DEFAULT 0,
            occurrence_count INTEGER NOT NULL DEFAULT 1
        ) WITHOUT ROWID;
        CREATE TABLE archive_member (
            stem TEXT PRIMARY KEY,
            expected_occurrence_count INTEGER NOT NULL,
            opus_count INTEGER NOT NULL DEFAULT 0,
            transcript_count INTEGER NOT NULL DEFAULT 0,
            opus_sha256 TEXT,
            transcript_sha256 TEXT
        ) WITHOUT ROWID;
        """
    )


def _load_expected_stems(manifest: Path, connection: sqlite3.Connection) -> _ManifestInventory:
    """Index the headerless official manifest without holding its paths in RAM.

    The index is a temporary SQLite database. The official CSV has exact duplicate
    rows, which are counted but never treated as additional assets; a duplicate
    with a different duration is rejected. Transcript paths are deterministic and
    checked before the duplicate test.
    """

    duration_sum = Decimal(0)
    manifest_rows = 0
    unique_pairs = 0
    duplicate_paths = 0
    duplicate_rows = 0
    try:
        connection.execute("BEGIN")
        with manifest.open("r", encoding="utf-8-sig", newline="") as file_handle:
            reader = csv.reader(file_handle)
            for row_number, row in enumerate(reader, start=1):
                manifest_rows = row_number
                if len(row) != 3:
                    raise OpenSttRhvoiceAuditError(
                        f"{manifest.name}:{row_number}: expected exactly three CSV fields."
                )
                audio_value, transcript_value, duration_value = row
                audio_path = _safe_path(audio_value, context=f"{manifest.name}:{row_number}")
                transcript_path = _safe_path(
                    transcript_value, context=f"{manifest.name}:{row_number}"
                )
                if audio_path.suffix != ".opus" or transcript_path.suffix != ".txt":
                    raise OpenSttRhvoiceAuditError(
                        f"{manifest.name}:{row_number}: expected paired .opus and .txt paths."
                    )
                if transcript_path != audio_path.with_suffix(".txt"):
                    raise OpenSttRhvoiceAuditError(
                        f"{manifest.name}:{row_number}: transcript path is not paired with "
                        "audio path."
                    )
                try:
                    duration = Decimal(duration_value)
                except InvalidOperation as error:
                    raise OpenSttRhvoiceAuditError(
                        f"{manifest.name}:{row_number}: invalid duration {duration_value!r}."
                    ) from error
                if not duration.is_finite() or duration <= 0:
                    raise OpenSttRhvoiceAuditError(
                        f"{manifest.name}:{row_number}: duration must be finite and positive."
                    )
                stem = (
                    audio_path.with_suffix("")
                    .relative_to(OPENSTT_RHVOICE_ARCHIVE_ROOT)
                    .as_posix()
                )
                try:
                    connection.execute(
                        "INSERT INTO expected (stem, duration) VALUES (?, ?)",
                        (stem, duration_value),
                    )
                except sqlite3.IntegrityError:
                    existing = connection.execute(
                        "SELECT duration, duplicate_path FROM expected WHERE stem = ?", (stem,)
                    ).fetchone()
                    if existing is None or existing[0] != duration_value:
                        raise OpenSttRhvoiceAuditError(
                            f"{manifest.name}:{row_number}: conflicting duplicate path "
                            f"{audio_value!r}."
                        ) from None
                    duplicate_rows += 1
                    if existing[1] == 0:
                        duplicate_paths += 1
                        connection.execute(
                            """
                            UPDATE expected
                            SET duplicate_path = 1, occurrence_count = occurrence_count + 1
                            WHERE stem = ?
                            """,
                            (stem,),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE expected
                            SET occurrence_count = occurrence_count + 1
                            WHERE stem = ?
                            """,
                            (stem,),
                        )
                else:
                    unique_pairs += 1
                duration_sum += duration
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    if manifest_rows != OPENSTT_RHVOICE_EXPECTED_MANIFEST_ROWS:
        raise OpenSttRhvoiceAuditError(
            "OpenSTT RHVoice manifest row count mismatch: "
            f"expected {OPENSTT_RHVOICE_EXPECTED_MANIFEST_ROWS}, got {manifest_rows}."
        )
    if not unique_pairs:
        raise OpenSttRhvoiceAuditError("OpenSTT RHVoice manifest contains no paired rows.")
    return _ManifestInventory(
        rows=manifest_rows,
        unique_pairs=unique_pairs,
        duplicate_paths=duplicate_paths,
        duplicate_rows=duplicate_rows,
        duration_sum=duration_sum,
    )


def _member_sha256(tar_archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    """Hash one streamed TAR member without extracting it to the filesystem."""

    member_handle = tar_archive.extractfile(member)
    if member_handle is None:
        raise OpenSttRhvoiceAuditError(
            f"OpenSTT RHVoice TAR member cannot be opened for hashing: {member.name!r}."
        )
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        while chunk := member_handle.read(1024 * 1024):
            digest.update(chunk)
            bytes_read += len(chunk)
    finally:
        member_handle.close()
    if bytes_read != member.size:
        raise OpenSttRhvoiceAuditError(
            f"OpenSTT RHVoice TAR member has an unexpected byte count: {member.name!r}."
        )
    return digest.hexdigest()


def _audit_archive(archive: Path, connection: sqlite3.Connection) -> _ArchiveInventory:
    directories = 0
    regular_files = 0
    content_bytes = 0
    opus_files = 0
    transcript_files = 0
    duplicate_opus_members = 0
    duplicate_transcript_members = 0
    duplicate_payloads_verified = 0
    try:
        connection.execute("BEGIN")
        with tarfile.open(archive, mode="r|gz") as tar_archive:
            for member in tar_archive:
                # ``tarfile`` retains every header it has returned, even in stream mode.
                # Keep only the current ``member`` reference for this multi-million-member TAR.
                retained_members = getattr(tar_archive, "members", None)
                if isinstance(retained_members, list):
                    retained_members.clear()
                path = _safe_path(member.name.rstrip("/"), context="OpenSTT RHVoice TAR")
                if member.isdir():
                    directories += 1
                    continue
                if not member.isfile():
                    raise OpenSttRhvoiceAuditError(
                        f"OpenSTT RHVoice TAR has a non-regular member: {member.name!r}."
                    )
                suffix = path.suffix
                if suffix not in {".opus", ".txt"}:
                    raise OpenSttRhvoiceAuditError(
                        f"OpenSTT RHVoice TAR has an unexpected file type: {member.name!r}."
                    )
                stem = path.with_suffix("").relative_to(OPENSTT_RHVOICE_ARCHIVE_ROOT).as_posix()
                member_row = connection.execute(
                    """
                    SELECT expected_occurrence_count, opus_count, transcript_count,
                           opus_sha256, transcript_sha256
                    FROM archive_member
                    WHERE stem = ?
                    """,
                    (stem,),
                ).fetchone()
                if member_row is None:
                    expected_row = connection.execute(
                        "SELECT occurrence_count FROM expected WHERE stem = ?", (stem,)
                    ).fetchone()
                    if expected_row is None:
                        raise OpenSttRhvoiceAuditError(
                            "OpenSTT RHVoice TAR has a member missing from the manifest: "
                            f"{member.name!r}."
                        )
                    expected_occurrence_count = expected_row[0]
                    opus_count = 0
                    transcript_count = 0
                    opus_sha256: str | None = None
                    transcript_sha256: str | None = None
                else:
                    (
                        expected_occurrence_count,
                        opus_count,
                        transcript_count,
                        opus_sha256,
                        transcript_sha256,
                    ) = member_row
                current_count = opus_count if suffix == ".opus" else transcript_count
                if current_count >= expected_occurrence_count:
                    raise OpenSttRhvoiceAuditError(
                        "OpenSTT RHVoice TAR has more repeated members than its manifest: "
                        f"{member.name!r}."
                    )
                member_sha256: str | None = None
                if expected_occurrence_count > 1:
                    member_sha256 = _member_sha256(tar_archive, member)
                    baseline_sha256 = opus_sha256 if suffix == ".opus" else transcript_sha256
                    if baseline_sha256 is not None and member_sha256 != baseline_sha256:
                        raise OpenSttRhvoiceAuditError(
                            "OpenSTT RHVoice TAR duplicate member differs byte-for-byte: "
                            f"{member.name!r}."
                        )
                    if current_count > 0:
                        duplicate_payloads_verified += 1
                if suffix == ".opus":
                    opus_count += 1
                    if current_count > 0:
                        duplicate_opus_members += 1
                    if member_sha256 is not None and opus_sha256 is None:
                        opus_sha256 = member_sha256
                else:
                    transcript_count += 1
                    if current_count > 0:
                        duplicate_transcript_members += 1
                    if member_sha256 is not None and transcript_sha256 is None:
                        transcript_sha256 = member_sha256
                if member_row is None:
                    connection.execute(
                        """
                        INSERT INTO archive_member (
                            stem, expected_occurrence_count, opus_count, transcript_count,
                            opus_sha256, transcript_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stem,
                            expected_occurrence_count,
                            opus_count,
                            transcript_count,
                            opus_sha256,
                            transcript_sha256,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE archive_member
                        SET opus_count = ?, transcript_count = ?,
                            opus_sha256 = ?, transcript_sha256 = ?
                        WHERE stem = ?
                        """,
                        (opus_count, transcript_count, opus_sha256, transcript_sha256, stem),
                    )
                regular_files += 1
                content_bytes += member.size
                if suffix == ".opus":
                    opus_files += 1
                else:
                    transcript_files += 1
        connection.commit()
    except (OSError, EOFError, tarfile.TarError) as error:
        connection.rollback()
        raise OpenSttRhvoiceAuditError(
            f"OpenSTT RHVoice TAR cannot be streamed safely: {error}"
        ) from error
    except Exception:
        connection.rollback()
        raise
    incomplete_row = connection.execute(
        """
        SELECT expected.stem
        FROM expected
        LEFT JOIN archive_member ON archive_member.stem = expected.stem
        WHERE COALESCE(archive_member.opus_count, 0) != expected.occurrence_count
           OR COALESCE(archive_member.transcript_count, 0) != expected.occurrence_count
        LIMIT 1
        """
    ).fetchone()
    if incomplete_row is not None:
        raise OpenSttRhvoiceAuditError(
            "OpenSTT RHVoice TAR does not match its manifest exactly; incomplete pair below root: "
            f"{incomplete_row[0]!r}."
        )
    return _ArchiveInventory(
        directories=directories,
        regular_files=regular_files,
        content_bytes=content_bytes,
        opus_files=opus_files,
        transcript_files=transcript_files,
        duplicate_opus_members=duplicate_opus_members,
        duplicate_transcript_members=duplicate_transcript_members,
        duplicate_payloads_verified=duplicate_payloads_verified,
    )


def audit_openstt_rhvoice_archive(archive: Path, manifest: Path) -> OpenSttRhvoiceArchiveAudit:
    """Validate both pinned artifacts and stream the TAR against its complete manifest.

    The gzip TAR is traversed sequentially without extraction.  This is only an
    artifact-integrity gate; source licensing, generator provenance, overlap, and
    final-protocol eligibility remain separate decisions.
    """

    archive_receipt = _validate_artifact(
        archive,
        expected_name=OPENSTT_RHVOICE_ARCHIVE_NAME,
        expected_size=OPENSTT_RHVOICE_ARCHIVE_EXPECTED_SIZE_BYTES,
        expected_md5=OPENSTT_RHVOICE_ARCHIVE_EXPECTED_MD5,
    )
    manifest_receipt = _validate_artifact(
        manifest,
        expected_name=OPENSTT_RHVOICE_MANIFEST_NAME,
        expected_size=OPENSTT_RHVOICE_MANIFEST_EXPECTED_SIZE_BYTES,
        expected_md5=OPENSTT_RHVOICE_MANIFEST_EXPECTED_MD5,
    )
    with tempfile.TemporaryDirectory(prefix="kds-openstt-rhvoice-audit-") as temporary_directory:
        database_path = Path(temporary_directory) / "membership.sqlite3"
        with sqlite3.connect(database_path) as connection:
            _prepare_membership_database(connection)
            inventory = _load_expected_stems(manifest, connection)
            archive_inventory = _audit_archive(archive, connection)
    if (
        archive_inventory.opus_files != inventory.rows
        or archive_inventory.transcript_files != inventory.rows
    ):
        raise OpenSttRhvoiceAuditError("OpenSTT RHVoice TAR has an unexpected paired member count.")
    return OpenSttRhvoiceArchiveAudit(
        source_id=OPENSTT_RHVOICE_SOURCE_ID,
        archive=archive_receipt,
        manifest=manifest_receipt,
        manifest_rows=inventory.rows,
        manifest_unique_pairs=inventory.unique_pairs,
        manifest_duplicate_paths=inventory.duplicate_paths,
        manifest_duplicate_rows=inventory.duplicate_rows,
        manifest_duration_sum=str(inventory.duration_sum),
        archive_directories=archive_inventory.directories,
        archive_regular_files=archive_inventory.regular_files,
        archive_content_bytes=archive_inventory.content_bytes,
        archive_opus_files=archive_inventory.opus_files,
        archive_transcript_files=archive_inventory.transcript_files,
        archive_duplicate_opus_members=archive_inventory.duplicate_opus_members,
        archive_duplicate_transcript_members=archive_inventory.duplicate_transcript_members,
        archive_duplicate_payloads_verified=archive_inventory.duplicate_payloads_verified,
    )
