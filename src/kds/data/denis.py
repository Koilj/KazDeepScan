"""Fail-closed read-only intake for the pinned MDC/Open Home Foundation Denis 1.0 archive.

The archive uses ``.webm`` member names, but every audio payload is actually an Ogg/Opus
bitstream.  The intake therefore verifies both the source filename binding and the decoded
container instead of trusting the suffix.  Audio is fully decoded in memory and never extracted
to the project tree.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import statistics
import tarfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import sha256_file

DENIS_SOURCE_ID = "denis_1_0_mdc"
DENIS_DOWNLOADED_ARCHIVE_NAME = "1764973737766-ru_RU-denis.tar.gz"
DENIS_SOURCE_CARD_ARCHIVE_NAME = "denis-1-0-3f60c388.tar.gz"
DENIS_ARCHIVE_EXPECTED_SIZE_BYTES = 109_594_943
DENIS_ARCHIVE_EXPECTED_SHA256 = (
    "75e2c63c5082df7623c6a98c529718b22015dfbd2d38a1ea328635f4dd4ccf9b"
)
DENIS_ARCHIVE_ROOT = "ru-RU"
DENIS_EXPECTED_RECORDS_BY_CATEGORY = {
    "0000000001_0300000050_General": 550,
    "3000000001_3000000300_Chat": 300,
    "4000000001_4000000300_CustomerService": 300,
}
DENIS_SOURCE_URL = "https://mozilladatacollective.com/datasets/cmiup9seu01flnv076fexaqp9"
DENIS_SOURCE_LICENSE = "CC0-1.0"
DENIS_DURATION_FEASIBILITY_SECONDS = Decimal("2.5")


class DenisArchiveAuditError(ValueError):
    """Raised when the local Denis archive cannot safely enter the project."""


@dataclass(frozen=True, slots=True)
class DenisRecord:
    """Privacy-minimal identity for one transcript/audio pair in the unextracted archive."""

    sample_id: str
    member_stem: str
    category: str
    literal_text_sha256: str
    whitespace_canonical_text_sha256: str
    nfkc_whitespace_canonical_text_sha256: str
    audio_sha256: str
    audio_size_bytes: int
    decoded_frames: int
    sample_rate_hz: int
    channels: int
    decoded_container: str
    decoded_subtype: str

    @property
    def duration_seconds(self) -> Decimal:
        return Decimal(self.decoded_frames) / Decimal(self.sample_rate_hz)


@dataclass(frozen=True, slots=True)
class DenisArchiveAudit:
    """Aggregate evidence from one exact, fully streamed source archive."""

    source_id: str
    downloaded_archive_name: str
    source_card_archive_name: str
    archive_size_bytes: int
    archive_sha256: str
    archive_root: str
    archive_members: int
    regular_files: int
    directories: int
    gzip_crc_verified: bool
    gzip_uncompressed_bytes: int
    tar_stream_fully_read: bool
    regular_file_bytes: int
    text_files: int
    audio_files: int
    paired_records: int
    orphan_record_stems: int
    unsafe_paths: int
    duplicate_member_paths: int
    casefold_duplicate_member_paths: int
    records_by_category: dict[str, int]
    text_payload_bytes: int
    audio_payload_bytes: int
    empty_texts: int
    multiline_texts: int
    nul_texts: int
    literal_unique_texts: int
    whitespace_canonical_unique_texts: int
    nfkc_whitespace_canonical_unique_texts: int
    text_members_with_nbsp: int
    text_members_with_trailing_whitespace: int
    filename_suffix_counts: dict[str, int]
    decoded_container_counts: dict[str, int]
    decoded_subtype_counts: dict[str, int]
    sample_rate_counts_hz: dict[str, int]
    channel_counts: dict[str, int]
    fully_decoded_audio_files: int
    decode_failures: int
    decoded_frames_total: int
    duration_total_seconds: str
    duration_min_seconds: str
    duration_median_seconds: str
    duration_max_seconds: str
    duration_at_least_2_5_seconds: int
    duration_below_2_5_seconds: int
    member_inventory_sha256: str
    record_identity_fingerprint: str
    source_provided_speaker_groups: int
    speaker_metadata_embedded: bool
    license_file_embedded: bool
    duration_metadata_embedded: bool
    duration_derived_from_audio: bool
    disk_extraction_performed: bool
    audio_payload_decoded_in_memory: bool
    candidate_selection_performed: bool
    tts_inference_performed: bool
    detector_inference_performed: bool
    intake_status: str

    def receipt(self, *, audited_at: str) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": 1,
                "audited_at": audited_at,
                "source_url": DENIS_SOURCE_URL,
                "license": DENIS_SOURCE_LICENSE,
                "rights_scope": (
                    "personal_research_only; no re-identification, re-hosting, product, "
                    "training, calibration, synthesis, or detector-inference authorization"
                ),
                "rights_evidence": (
                    "MDC dataset card declares CC0-1.0 and names Open Home Foundation as "
                    "steward; MDC provider terms require the provider to hold necessary rights, "
                    "permissions, and consents. The archive itself embeds no license file."
                ),
                "duration_basis": (
                    "fully decoded PCM frame count divided by the declared 48000 Hz sample "
                    "rate, after the Opus decoder applies codec pre-skip"
                ),
                "feasibility": {
                    "minimum_ready_pairs": 60,
                    "target_pairs": 79,
                    "pre_qa_rows_at_least_2_5_seconds": self.duration_at_least_2_5_seconds,
                    "minimum_60_feasible_before_vad_and_acoustic_qa": (
                        self.duration_at_least_2_5_seconds >= 60
                    ),
                    "target_79_feasible_before_vad_and_acoustic_qa": (
                        self.duration_at_least_2_5_seconds >= 79
                    ),
                },
                "limitations": [
                    "The browser-assigned download name differs from the archive name shown on "
                    "the source card; exact local bytes and SHA-256 are authoritative here.",
                    "Members use a .webm suffix but decode consistently as Ogg/Opus; the mismatch "
                    "is disclosed and must not be rewritten silently.",
                    "The archive provides no embedded license, duration table, speaker table, or "
                    "independent consent audit.",
                    "The source card describes one speaker, so this source cannot establish "
                    "speaker robustness.",
                    "Duration feasibility is not VAD, acoustic review, candidate readiness, or an "
                    "authorization to select, synthesize, or run the detector.",
                    "Raw audio and transcripts remain outside Git; no member was extracted to "
                    "disk by this intake.",
                ],
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class DenisArchiveInspection:
    """Aggregate receipt data plus per-record hashes used by the exposure screen."""

    audit: DenisArchiveAudit
    records: tuple[DenisRecord, ...]


def _validate_archive_identity(archive: Path) -> None:
    if not archive.is_file():
        raise DenisArchiveAuditError(f"Denis archive does not exist: {archive}")
    actual_size = archive.stat().st_size
    if actual_size != DENIS_ARCHIVE_EXPECTED_SIZE_BYTES:
        raise DenisArchiveAuditError(
            "Denis archive size mismatch: "
            f"expected {DENIS_ARCHIVE_EXPECTED_SIZE_BYTES}, got {actual_size}."
        )
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != DENIS_ARCHIVE_EXPECTED_SHA256:
        raise DenisArchiveAuditError(
            "Denis archive SHA-256 mismatch: "
            f"expected {DENIS_ARCHIVE_EXPECTED_SHA256}, got {actual_sha256}."
        )


def _verify_gzip_crc(archive: Path) -> int:
    uncompressed_bytes = 0
    try:
        with gzip.open(archive, mode="rb") as compressed:
            while payload := compressed.read(1024 * 1024):
                uncompressed_bytes += len(payload)
    except (OSError, gzip.BadGzipFile) as error:
        raise DenisArchiveAuditError(
            f"Denis archive does not pass a complete gzip CRC read: {archive}."
        ) from error
    if uncompressed_bytes <= 0:
        raise DenisArchiveAuditError("Denis gzip stream is empty.")
    return uncompressed_bytes


def _member_parts(member_name: str) -> tuple[str, ...]:
    path = PurePosixPath(member_name)
    if (
        not member_name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in member_name
        or any(not part for part in path.parts)
    ):
        raise DenisArchiveAuditError(f"Unsafe TAR member path: {member_name!r}.")
    return path.parts


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    file_object = archive.extractfile(member)
    if file_object is None:
        raise DenisArchiveAuditError(f"Cannot read TAR member: {member.name!r}.")
    try:
        payload = file_object.read()
    finally:
        file_object.close()
    if len(payload) != member.size:
        raise DenisArchiveAuditError(
            f"TAR member size differs while reading {member.name!r}: "
            f"expected {member.size}, got {len(payload)}."
        )
    return payload


def _decode_text(payload: bytes, member_name: str) -> tuple[str, str, str, str]:
    try:
        literal = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DenisArchiveAuditError(f"Transcript is not UTF-8: {member_name!r}.") from error
    if not literal or "\x00" in literal:
        raise DenisArchiveAuditError(f"Transcript is empty or contains NUL: {member_name!r}.")
    if "\r" in literal or "\n" in literal:
        raise DenisArchiveAuditError(
            f"Transcript is unexpectedly multiline: {member_name!r}."
        )
    whitespace_canonical = " ".join(literal.split())
    nfkc_whitespace_canonical = " ".join(unicodedata.normalize("NFKC", literal).split())
    if not whitespace_canonical or not nfkc_whitespace_canonical:
        raise DenisArchiveAuditError(f"Transcript canonicalizes to blank: {member_name!r}.")
    return literal, whitespace_canonical, nfkc_whitespace_canonical, _sha256(payload)


def _decode_audio(payload: bytes, member_name: str) -> tuple[int, int, int, str, str]:
    try:
        with sf.SoundFile(io.BytesIO(payload)) as audio:
            sample_rate = int(audio.samplerate)
            channels = int(audio.channels)
            decoded_container = str(audio.format)
            decoded_subtype = str(audio.subtype)
            declared_frames = int(audio.frames)
            decoded_frames = 0
            while True:
                block = audio.read(65_536, dtype="float32", always_2d=True)
                if len(block) == 0:
                    break
                decoded_frames += len(block)
    except (OSError, RuntimeError) as error:
        raise DenisArchiveAuditError(
            f"Audio payload cannot be fully decoded: {member_name!r}."
        ) from error
    if (
        sample_rate <= 0
        or channels <= 0
        or decoded_frames <= 0
        or decoded_frames != declared_frames
    ):
        raise DenisArchiveAuditError(
            f"Audio payload has inconsistent decoded metadata: {member_name!r}."
        )
    return decoded_frames, sample_rate, channels, decoded_container, decoded_subtype


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return _sha256(encoded)


def _seconds(value: Decimal) -> str:
    return f"{value:.6f}"


def inspect_denis_archive(archive_path: Path) -> DenisArchiveInspection:
    """Verify the exact archive, every pair, and every audio payload without disk extraction."""

    _validate_archive_identity(archive_path)
    gzip_uncompressed_bytes = _verify_gzip_crc(archive_path)
    archive_members = 0
    regular_files = 0
    directories = 0
    regular_file_bytes = 0
    member_paths: set[str] = set()
    casefold_member_paths: set[str] = set()
    directory_paths: set[str] = set()
    member_inventory: list[dict[str, object]] = []
    pairs: dict[str, dict[str, tuple[tarfile.TarInfo, bytes]]] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                archive_members += 1
                parts = _member_parts(member.name)
                if parts[0] != DENIS_ARCHIVE_ROOT:
                    raise DenisArchiveAuditError(
                        f"Unexpected Denis archive root: {member.name!r}."
                    )
                if not (member.isdir() or member.isfile()):
                    raise DenisArchiveAuditError(
                        f"Unsafe TAR member type for {member.name!r}: {member.type!r}."
                    )
                if member.name in member_paths:
                    raise DenisArchiveAuditError(
                        f"Duplicate TAR member path: {member.name!r}."
                    )
                casefold_name = member.name.casefold()
                if casefold_name in casefold_member_paths:
                    raise DenisArchiveAuditError(
                        f"Case-fold duplicate TAR member path: {member.name!r}."
                    )
                member_paths.add(member.name)
                casefold_member_paths.add(casefold_name)
                member_inventory.append(
                    {
                        "name": member.name,
                        "size": member.size,
                        "type": "directory" if member.isdir() else "file",
                    }
                )
                if member.isdir():
                    directories += 1
                    directory_paths.add(member.name)
                    continue
                regular_files += 1
                regular_file_bytes += member.size
                if len(parts) != 3:
                    raise DenisArchiveAuditError(
                        f"Unexpected Denis regular member path: {member.name!r}."
                    )
                category, filename = parts[1:]
                if category not in DENIS_EXPECTED_RECORDS_BY_CATEGORY:
                    raise DenisArchiveAuditError(
                        f"Unexpected Denis category: {category!r}."
                    )
                path = PurePosixPath(filename)
                if not path.stem.isdecimal() or path.suffix not in {".txt", ".webm"}:
                    raise DenisArchiveAuditError(
                        f"Unexpected Denis record member: {member.name!r}."
                    )
                stem = f"{DENIS_ARCHIVE_ROOT}/{category}/{path.stem}"
                by_suffix = pairs.setdefault(stem, {})
                if path.suffix in by_suffix:
                    raise DenisArchiveAuditError(
                        f"Duplicate Denis pair suffix for {stem!r}: {path.suffix}."
                    )
                by_suffix[path.suffix] = (member, _read_member(archive, member))

            expected_directories = {
                DENIS_ARCHIVE_ROOT,
                *(f"{DENIS_ARCHIVE_ROOT}/{name}" for name in DENIS_EXPECTED_RECORDS_BY_CATEGORY),
            }
            if directory_paths != expected_directories:
                raise DenisArchiveAuditError(
                    "Denis directory layout differs: "
                    f"missing={sorted(expected_directories.difference(directory_paths))}, "
                    f"unexpected={sorted(directory_paths.difference(expected_directories))}."
                )
            orphan_stems = [
                stem for stem, values in pairs.items() if set(values) != {".txt", ".webm"}
            ]
            if orphan_stems:
                raise DenisArchiveAuditError(
                    f"Denis archive has unpaired transcript/audio stems: {orphan_stems[:5]}."
                )

            records: list[DenisRecord] = []
            literal_texts: set[str] = set()
            whitespace_texts: set[str] = set()
            nfkc_texts: set[str] = set()
            records_by_category: Counter[str] = Counter()
            suffix_counts: Counter[str] = Counter()
            container_counts: Counter[str] = Counter()
            subtype_counts: Counter[str] = Counter()
            sample_rate_counts: Counter[int] = Counter()
            channel_counts: Counter[int] = Counter()
            text_payload_bytes = 0
            audio_payload_bytes = 0
            text_members_with_nbsp = 0
            text_members_with_trailing_whitespace = 0
            durations: list[Decimal] = []
            for stem in sorted(pairs):
                text_member, text_payload = pairs[stem][".txt"]
                audio_member, audio_payload = pairs[stem][".webm"]
                literal, whitespace_text, nfkc_text, literal_hash = _decode_text(
                    text_payload, text_member.name
                )
                decoded_frames, sample_rate, channels, container, subtype = _decode_audio(
                    audio_payload, audio_member.name
                )
                if "\u00a0" in literal:
                    text_members_with_nbsp += 1
                if literal != literal.rstrip():
                    text_members_with_trailing_whitespace += 1
                literal_texts.add(literal)
                whitespace_texts.add(whitespace_text)
                nfkc_texts.add(nfkc_text)
                text_payload_bytes += len(text_payload)
                audio_payload_bytes += len(audio_payload)
                category = PurePosixPath(stem).parts[1]
                records_by_category[category] += 1
                suffix_counts[PurePosixPath(audio_member.name).suffix] += 1
                container_counts[container] += 1
                subtype_counts[subtype] += 1
                sample_rate_counts[sample_rate] += 1
                channel_counts[channels] += 1
                record = DenisRecord(
                    sample_id=f"{DENIS_SOURCE_ID}:{stem.removeprefix(DENIS_ARCHIVE_ROOT + '/')}",
                    member_stem=stem,
                    category=category,
                    literal_text_sha256=literal_hash,
                    whitespace_canonical_text_sha256=_sha256(whitespace_text.encode("utf-8")),
                    nfkc_whitespace_canonical_text_sha256=_sha256(
                        nfkc_text.encode("utf-8")
                    ),
                    audio_sha256=_sha256(audio_payload),
                    audio_size_bytes=len(audio_payload),
                    decoded_frames=decoded_frames,
                    sample_rate_hz=sample_rate,
                    channels=channels,
                    decoded_container=container,
                    decoded_subtype=subtype,
                )
                records.append(record)
                durations.append(record.duration_seconds)
    except (OSError, tarfile.TarError) as error:
        raise DenisArchiveAuditError(f"Cannot read Denis TAR archive: {archive_path}.") from error

    actual_by_category = {
        key: records_by_category[key] for key in sorted(records_by_category)
    }
    if actual_by_category != DENIS_EXPECTED_RECORDS_BY_CATEGORY:
        raise DenisArchiveAuditError(
            "Denis record counts by category differ: "
            f"expected {DENIS_EXPECTED_RECORDS_BY_CATEGORY}, got {actual_by_category}."
        )
    record_payload = [
        {
            "sample_id": record.sample_id,
            "member_stem": record.member_stem,
            "literal_text_sha256": record.literal_text_sha256,
            "whitespace_canonical_text_sha256": (
                record.whitespace_canonical_text_sha256
            ),
            "nfkc_whitespace_canonical_text_sha256": (
                record.nfkc_whitespace_canonical_text_sha256
            ),
            "audio_sha256": record.audio_sha256,
            "audio_size_bytes": record.audio_size_bytes,
            "decoded_frames": record.decoded_frames,
            "sample_rate_hz": record.sample_rate_hz,
            "channels": record.channels,
            "decoded_container": record.decoded_container,
            "decoded_subtype": record.decoded_subtype,
        }
        for record in records
    ]
    total_duration = sum(durations, start=Decimal(0))
    audit = DenisArchiveAudit(
        source_id=DENIS_SOURCE_ID,
        downloaded_archive_name=archive_path.name,
        source_card_archive_name=DENIS_SOURCE_CARD_ARCHIVE_NAME,
        archive_size_bytes=DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
        archive_sha256=DENIS_ARCHIVE_EXPECTED_SHA256,
        archive_root=DENIS_ARCHIVE_ROOT,
        archive_members=archive_members,
        regular_files=regular_files,
        directories=directories,
        gzip_crc_verified=True,
        gzip_uncompressed_bytes=gzip_uncompressed_bytes,
        tar_stream_fully_read=True,
        regular_file_bytes=regular_file_bytes,
        text_files=len(records),
        audio_files=len(records),
        paired_records=len(records),
        orphan_record_stems=0,
        unsafe_paths=0,
        duplicate_member_paths=0,
        casefold_duplicate_member_paths=0,
        records_by_category=actual_by_category,
        text_payload_bytes=text_payload_bytes,
        audio_payload_bytes=audio_payload_bytes,
        empty_texts=0,
        multiline_texts=0,
        nul_texts=0,
        literal_unique_texts=len(literal_texts),
        whitespace_canonical_unique_texts=len(whitespace_texts),
        nfkc_whitespace_canonical_unique_texts=len(nfkc_texts),
        text_members_with_nbsp=text_members_with_nbsp,
        text_members_with_trailing_whitespace=text_members_with_trailing_whitespace,
        filename_suffix_counts={key: suffix_counts[key] for key in sorted(suffix_counts)},
        decoded_container_counts={
            key: container_counts[key] for key in sorted(container_counts)
        },
        decoded_subtype_counts={
            key: subtype_counts[key] for key in sorted(subtype_counts)
        },
        sample_rate_counts_hz={
            str(key): sample_rate_counts[key] for key in sorted(sample_rate_counts)
        },
        channel_counts={str(key): channel_counts[key] for key in sorted(channel_counts)},
        fully_decoded_audio_files=len(records),
        decode_failures=0,
        decoded_frames_total=sum(record.decoded_frames for record in records),
        duration_total_seconds=_seconds(total_duration),
        duration_min_seconds=_seconds(min(durations)),
        duration_median_seconds=_seconds(statistics.median(durations)),
        duration_max_seconds=_seconds(max(durations)),
        duration_at_least_2_5_seconds=sum(
            value >= DENIS_DURATION_FEASIBILITY_SECONDS for value in durations
        ),
        duration_below_2_5_seconds=sum(
            value < DENIS_DURATION_FEASIBILITY_SECONDS for value in durations
        ),
        member_inventory_sha256=_fingerprint(
            sorted(member_inventory, key=lambda row: str(row["name"]))
        ),
        record_identity_fingerprint=_fingerprint(record_payload),
        source_provided_speaker_groups=1,
        speaker_metadata_embedded=False,
        license_file_embedded=False,
        duration_metadata_embedded=False,
        duration_derived_from_audio=True,
        disk_extraction_performed=False,
        audio_payload_decoded_in_memory=True,
        candidate_selection_performed=False,
        tts_inference_performed=False,
        detector_inference_performed=False,
        intake_status="accepted_source_level_only",
    )
    return DenisArchiveInspection(audit=audit, records=tuple(records))


def audit_denis_archive(archive_path: Path) -> DenisArchiveAudit:
    """Return only the aggregate source receipt data for the exact archive."""

    return inspect_denis_archive(archive_path).audit
