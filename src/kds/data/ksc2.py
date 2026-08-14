"""Read-only, multipart-safe audit for the pinned Kazakh Speech Corpus 2 archive.

KSC2 is distributed as ten sequential parts of one gzip-compressed TAR archive.
The parts must never be concatenated to a large temporary file and TAR members
must never be extracted before the archive's layout has been audited.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

KSC2_ARCHIVE_BASENAME = "ISSAI_KSC2.tar.gz"
KSC2_ARCHIVE_ROOT = "ISSAI_KSC2"
KSC2_PART_SUFFIXES = tuple(f"parta{letter}" for letter in "abcdefghij")
KSC2_PART_EXPECTED_SIZES = (8_388_608_000,) * 9 + (5_311_650_212,)
KSC2_AUDIO_SUFFIXES = frozenset({".flac", ".wav", ".mp3", ".ogg"})
KSC2_TRANSCRIPT_SUFFIX = ".txt"
KSC2_METADATA_SUFFIXES = frozenset({".csv", ".json", ".tsv", ".yaml", ".yml"})
KSC2_METADATA_MEMBER_EXAMPLE_LIMIT = 100
KSC2_MIXED_ANNOTATION_COMPONENTS = frozenset({"Test/podcasts", "Test/talkshow", "Test/radio"})
KSC2_ANNOTATION_MAX_MEMBER_BYTES = 64 * 1024 * 1024
Ksc2AuditProgressCallback = Callable[[int, int, str], None]


class Ksc2AuditError(ValueError):
    """Raised when the multipart KSC2 archive cannot be safely audited."""


@dataclass(frozen=True, slots=True)
class Ksc2PartReceipt:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Ksc2ArchiveAudit:
    archive_root: str
    compressed_bytes: int
    compressed_sha256: str
    parts: tuple[Ksc2PartReceipt, ...]
    directories: int
    regular_files: int
    content_bytes: int
    files_by_extension: dict[str, int]
    files_by_component: dict[str, int]
    audio_files: int
    transcript_files: int
    unpaired_audio_files: int
    unpaired_transcript_files: int
    unpaired_audio_examples: tuple[str, ...]
    unpaired_transcript_examples: tuple[str, ...]
    metadata_files: int
    metadata_member_examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ksc2AnnotationCandidate:
    """One unlabelled, paired priority-component candidate for later evidence review."""

    candidate_id: str
    component: str
    archive_audio_member: str
    archive_transcript_member: str
    audio_relative_path: str
    audio_sha256: str
    transcript: str
    transcript_sha256: str


@dataclass(frozen=True, slots=True)
class Ksc2TextCandidate:
    """One exact paired KSC2 member with text identities but no extracted audio."""

    candidate_id: str
    component: str
    archive_audio_member: str
    archive_transcript_member: str
    transcript_sha256: str
    canonical_text_sha256: str


@dataclass(frozen=True, slots=True)
class Ksc2ExtractedAudio:
    """One selected KSC2 source asset extracted during an exact full-archive pass."""

    archive_member: str
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Ksc2ExtractedTranscript:
    """One selected KSC2 UTF-8 transcript extracted during an exact full-archive pass."""

    archive_member: str
    relative_path: str
    sha256: str
    size_bytes: int


class _MultipartReader:
    """Expose ordered archive parts as one read-only binary stream and hash every byte once."""

    def __init__(
        self,
        parts: tuple[Path, ...],
        *,
        progress_callback: Ksc2AuditProgressCallback | None = None,
    ) -> None:
        self._parts = parts
        self._index = 0
        self._handle: BinaryIO | None = None
        self._current_digest: hashlib._Hash | None = None
        self._part_receipts: list[Ksc2PartReceipt] = []
        self._archive_digest = hashlib.sha256()
        self._closed = False
        self._progress_callback = progress_callback

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self._index >= len(self._parts):
            return False
        path = self._parts[self._index]
        self._index += 1
        self._handle = path.open("rb")
        self._current_digest = hashlib.sha256()
        return True

    def _close_current(self) -> None:
        if self._handle is None or self._current_digest is None:
            return
        handle = self._handle
        self._handle = None
        handle.close()
        path = self._parts[self._index - 1]
        self._part_receipts.append(
            Ksc2PartReceipt(
                filename=path.name,
                size_bytes=path.stat().st_size,
                sha256=self._current_digest.hexdigest(),
            )
        )
        if self._progress_callback is not None:
            self._progress_callback(len(self._part_receipts), len(self._parts), path.name)
        self._current_digest = None

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while remaining != 0:
            if self._handle is None and not self._open_next():
                break
            assert self._handle is not None
            request_size = -1 if remaining < 0 else remaining
            chunk = self._handle.read(request_size)
            if not chunk:
                self._close_current()
                continue
            assert self._current_digest is not None
            self._current_digest.update(chunk)
            self._archive_digest.update(chunk)
            chunks.append(chunk)
            if remaining > 0:
                remaining -= len(chunk)
        return b"".join(chunks)

    @property
    def compressed_sha256(self) -> str:
        return self._archive_digest.hexdigest()

    @property
    def compressed_bytes(self) -> int:
        return sum(receipt.size_bytes for receipt in self._part_receipts) + (
            self._parts[self._index - 1].stat().st_size if self._handle is not None else 0
        )

    @property
    def part_receipts(self) -> tuple[Ksc2PartReceipt, ...]:
        return tuple(self._part_receipts)

    def close(self) -> None:
        if self._closed:
            return
        self._close_current()
        self._closed = True


def ksc2_part_paths(
    parts_directory: Path, *, expected_sizes: tuple[int, ...] = KSC2_PART_EXPECTED_SIZES
) -> tuple[Path, ...]:
    """Return exactly the ordered, pinned-size KSC2 archive parts."""

    if not parts_directory.is_dir():
        raise Ksc2AuditError(f"KSC2 parts directory does not exist: {parts_directory}")
    if len(expected_sizes) != len(KSC2_PART_SUFFIXES):
        raise ValueError("KSC2 expected part-size contract has the wrong length.")
    paths = tuple(
        parts_directory / f"{KSC2_ARCHIVE_BASENAME}.{suffix}" for suffix in KSC2_PART_SUFFIXES
    )
    for path, expected_size in zip(paths, expected_sizes, strict=True):
        if not path.is_file():
            raise Ksc2AuditError(f"KSC2 archive part is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise Ksc2AuditError(
                f"KSC2 part-size mismatch for {path.name}: expected {expected_size}, "
                f"got {actual_size}."
            )
    unexpected = sorted(
        path.name
        for path in parts_directory.glob(f"{KSC2_ARCHIVE_BASENAME}.part*")
        if path not in paths
    )
    if unexpected:
        raise Ksc2AuditError("Unexpected KSC2 archive parts: " + ", ".join(unexpected))
    return paths


def _safe_member_path(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if (
        not member.name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in member.name
        or path.parts[0] != KSC2_ARCHIVE_ROOT
    ):
        raise Ksc2AuditError(f"Unsafe or unexpected KSC2 TAR member: {member.name!r}.")
    return path


def _component_name(path: PurePosixPath) -> str:
    relative = path.parts[1:]
    if not relative:
        return "_root"
    return "/".join(relative[:2])


def _logical_member_stem(path: PurePosixPath, suffix: str) -> tuple[str, bool]:
    """Strip one or more repeated type suffixes from a paired KSC2 member name.

    The release contains a TV-news subset whose paired members are named
    ``<id>.flac.flac`` and ``<id>.txt.txt``.  A single ``with_suffix`` would
    falsely call those valid pairs unrelated.
    """

    normalized = path
    stripped = 0
    while normalized.suffix.lower() == suffix:
        normalized = normalized.with_suffix("")
        stripped += 1
    return normalized.as_posix(), stripped > 1


def audit_ksc2_archive(
    parts_directory: Path,
    *,
    expected_sizes: tuple[int, ...] = KSC2_PART_EXPECTED_SIZES,
    progress_callback: Ksc2AuditProgressCallback | None = None,
) -> Ksc2ArchiveAudit:
    """Stream every part once, validate gzip/TAR structure, and return a compact receipt.

    The combined compressed SHA-256 and each part SHA-256 are calculated during the sole
    sequential traversal.  No archive member is extracted or opened as an audio asset.
    """

    paths = ksc2_part_paths(parts_directory, expected_sizes=expected_sizes)
    reader = _MultipartReader(paths, progress_callback=progress_callback)
    directories = 0
    regular_files = 0
    content_bytes = 0
    files_by_extension: Counter[str] = Counter()
    files_by_component: Counter[str] = Counter()
    audio_members: set[str] = set()
    transcript_members: set[str] = set()
    metadata_files = 0
    metadata_member_examples: list[str] = []
    try:
        with gzip.GzipFile(fileobj=cast(BinaryIO, reader), mode="rb") as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode="r|") as archive:
                for member in archive:
                    path = _safe_member_path(member)
                    if member.isdir():
                        directories += 1
                        continue
                    if not member.isfile():
                        raise Ksc2AuditError(
                            f"KSC2 archive contains an unsupported member type: {member.name!r}."
                        )
                    regular_files += 1
                    content_bytes += member.size
                    extension = path.suffix.lower() or "<none>"
                    files_by_extension[extension] += 1
                    files_by_component[_component_name(path)] += 1
                    if extension in KSC2_AUDIO_SUFFIXES:
                        member_stem, _repeated_suffix = _logical_member_stem(path, extension)
                        audio_members.add(member_stem)
                    elif extension == KSC2_TRANSCRIPT_SUFFIX:
                        member_stem, _repeated_suffix = _logical_member_stem(
                            path, KSC2_TRANSCRIPT_SUFFIX
                        )
                        transcript_members.add(member_stem)
                    if extension in KSC2_METADATA_SUFFIXES:
                        metadata_files += 1
                        if len(metadata_member_examples) < KSC2_METADATA_MEMBER_EXAMPLE_LIMIT:
                            metadata_member_examples.append(path.as_posix())
        # Iterating the streaming TAR reaches gzip EOF and therefore verifies the gzip CRC.
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as error:
        raise Ksc2AuditError(f"KSC2 multipart archive cannot be read safely: {error}") from error
    finally:
        reader.close()
    receipts = reader.part_receipts
    if len(receipts) != len(paths):
        raise Ksc2AuditError("KSC2 audit did not consume every archive part.")
    if regular_files == 0:
        raise Ksc2AuditError("KSC2 archive contains no regular files.")
    unpaired_audio = sorted(audio_members.difference(transcript_members))
    unpaired_transcripts = sorted(transcript_members.difference(audio_members))
    return Ksc2ArchiveAudit(
        archive_root=KSC2_ARCHIVE_ROOT,
        compressed_bytes=sum(receipt.size_bytes for receipt in receipts),
        compressed_sha256=reader.compressed_sha256,
        parts=receipts,
        directories=directories,
        regular_files=regular_files,
        content_bytes=content_bytes,
        files_by_extension=dict(sorted(files_by_extension.items())),
        files_by_component=dict(sorted(files_by_component.items())),
        audio_files=len(audio_members),
        transcript_files=len(transcript_members),
        unpaired_audio_files=len(unpaired_audio),
        unpaired_transcript_files=len(unpaired_transcripts),
        unpaired_audio_examples=tuple(unpaired_audio[:KSC2_METADATA_MEMBER_EXAMPLE_LIMIT]),
        unpaired_transcript_examples=tuple(
            unpaired_transcripts[:KSC2_METADATA_MEMBER_EXAMPLE_LIMIT]
        ),
        metadata_files=metadata_files,
        metadata_member_examples=tuple(metadata_member_examples),
    )


def write_ksc2_audit_report(path: Path, audit: Ksc2ArchiveAudit) -> None:
    """Atomically publish a new KSC2 JSON receipt without replacing history."""

    if path.exists() or not path.parent.is_dir():
        raise Ksc2AuditError(f"Unsafe KSC2 audit report destination: {path}")
    try:
        with tempfile.TemporaryDirectory(prefix="kds-ksc2-report-", dir=path.parent) as stage_dir:
            staged_path = Path(stage_dir) / path.name
            staged_path.write_text(
                json.dumps(asdict(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shutil.move(str(staged_path), path)
    except OSError as error:
        raise Ksc2AuditError(f"Cannot write KSC2 audit report: {path}") from error


def scan_ksc2_text_candidates(
    parts_directory: Path,
    *,
    allowed_components: frozenset[str],
    expected_compressed_sha256: str,
    expected_sizes: tuple[int, ...] = KSC2_PART_EXPECTED_SIZES,
    progress_callback: Ksc2AuditProgressCallback | None = None,
) -> tuple[Ksc2TextCandidate, ...]:
    """Read exact paired text identities for selected components without extracting audio."""

    if not allowed_components or any(
        not component.startswith("Train/")
        or component.endswith("/crowdsourced")
        or component.endswith("/tts")
        for component in allowed_components
    ):
        raise Ksc2AuditError("KSC2 text scan requires explicit nonlegacy Train components.")
    expected_hash = expected_compressed_sha256.lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise Ksc2AuditError("KSC2 text scan expected SHA-256 is invalid.")
    paths = ksc2_part_paths(parts_directory, expected_sizes=expected_sizes)
    reader = _MultipartReader(paths, progress_callback=progress_callback)
    audio_by_stem: dict[str, tuple[str, str]] = {}
    text_by_stem: dict[str, tuple[str, str, str]] = {}
    try:
        with gzip.GzipFile(fileobj=cast(BinaryIO, reader), mode="rb") as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode="r|") as archive:
                for member in archive:
                    path = _safe_member_path(member)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise Ksc2AuditError(
                            f"KSC2 archive contains an unsupported member type: {member.name!r}."
                        )
                    component = _component_name(path)
                    if component not in allowed_components:
                        continue
                    extension = path.suffix.lower()
                    if extension in KSC2_AUDIO_SUFFIXES:
                        logical_stem, _repeated = _logical_member_stem(path, extension)
                        if logical_stem in audio_by_stem:
                            raise Ksc2AuditError(
                                f"Duplicate KSC2 selected audio member: {logical_stem!r}."
                            )
                        audio_by_stem[logical_stem] = (component, path.as_posix())
                    elif extension == KSC2_TRANSCRIPT_SUFFIX:
                        logical_stem, _repeated = _logical_member_stem(
                            path, KSC2_TRANSCRIPT_SUFFIX
                        )
                        if logical_stem in text_by_stem:
                            raise Ksc2AuditError(
                                f"Duplicate KSC2 selected transcript member: {logical_stem!r}."
                            )
                        transcript, transcript_sha256 = _read_text_member_and_hash(
                            archive, member
                        )
                        canonical_text = " ".join(
                            unicodedata.normalize("NFKC", transcript).split()
                        )
                        if not canonical_text:
                            raise Ksc2AuditError(
                                f"KSC2 selected transcript is empty: {member.name!r}."
                            )
                        text_by_stem[logical_stem] = (
                            path.as_posix(),
                            transcript_sha256,
                            hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
                        )
        # Reaching gzip EOF verifies the stream CRC.
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as error:
        raise Ksc2AuditError(f"KSC2 selected-text scan failed: {error}") from error
    finally:
        reader.close()
    if len(reader.part_receipts) != len(paths) or reader.compressed_sha256 != expected_hash:
        raise Ksc2AuditError("KSC2 selected-text scan did not verify the exact full archive.")
    audio_stems = set(audio_by_stem)
    text_stems = set(text_by_stem)
    audio_only = audio_stems.difference(text_stems)
    text_only = text_stems.difference(audio_stems)
    # The pinned release has one known transcript-only Train/radio member. It is excluded.
    if audio_only or len(text_only) > 1:
        raise Ksc2AuditError(
            "KSC2 selected components have unexpected unpaired members: "
            f"audio_only={len(audio_only)}, text_only={len(text_only)}."
        )
    candidates: list[Ksc2TextCandidate] = []
    for logical_stem in sorted(audio_stems.intersection(text_stems)):
        component, audio_member = audio_by_stem[logical_stem]
        transcript_member, transcript_sha256, canonical_text_sha256 = text_by_stem[
            logical_stem
        ]
        candidate_id = logical_stem.removeprefix(f"{KSC2_ARCHIVE_ROOT}/")
        candidates.append(
            Ksc2TextCandidate(
                candidate_id=candidate_id,
                component=component,
                archive_audio_member=audio_member,
                archive_transcript_member=transcript_member,
                transcript_sha256=transcript_sha256,
                canonical_text_sha256=canonical_text_sha256,
            )
        )
    if not candidates:
        raise Ksc2AuditError("KSC2 selected components contain no paired candidates.")
    return tuple(candidates)


def _copy_member_and_hash(archive: tarfile.TarFile, member: tarfile.TarInfo, output: Path) -> str:
    """Copy a selected regular TAR member without using extract/extractall."""

    if member.size > KSC2_ANNOTATION_MAX_MEMBER_BYTES:
        raise Ksc2AuditError(
            f"KSC2 annotation candidate is too large: {member.name!r} ({member.size} bytes)."
        )
    source = archive.extractfile(member)
    if source is None:
        raise Ksc2AuditError(f"KSC2 cannot read selected TAR member: {member.name!r}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with output.open("xb") as destination:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                destination.write(chunk)
    finally:
        source.close()
    return digest.hexdigest()


def _read_text_member_and_hash(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> tuple[str, str]:
    if member.size > KSC2_ANNOTATION_MAX_MEMBER_BYTES:
        raise Ksc2AuditError(
            f"KSC2 annotation transcript is too large: {member.name!r} ({member.size} bytes)."
        )
    source = archive.extractfile(member)
    if source is None:
        raise Ksc2AuditError(f"KSC2 cannot read selected TAR member: {member.name!r}.")
    try:
        payload = source.read()
    finally:
        source.close()
    try:
        transcript = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Ksc2AuditError(
            f"KSC2 annotation transcript is not valid UTF-8: {member.name!r}."
        ) from error
    return transcript, hashlib.sha256(payload).hexdigest()


def extract_ksc2_mixed_annotation_candidates(
    parts_directory: Path,
    output_directory: Path,
    *,
    expected_compressed_sha256: str,
    expected_sizes: tuple[int, ...] = KSC2_PART_EXPECTED_SIZES,
) -> tuple[Ksc2AnnotationCandidate, ...]:
    """Extract only unlabelled KSC2 mixed-annotation candidates into an empty staging directory.

    The sole permitted components are the three highest code-switch-rate Test paths in the
    KSC2 paper. The routine validates every TAR member type/path while streaming the entire
    multipart archive and publishes no language decision: each returned candidate remains
    ``pending`` for later evidence review.
    """

    expected_hash = expected_compressed_sha256.lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise ValueError(
            "KSC2 expected compressed SHA-256 must be a 64-character lowercase hex digest."
        )
    if output_directory.exists():
        raise Ksc2AuditError(
            f"KSC2 annotation output directory already exists: {output_directory}."
        )
    output_directory.mkdir(parents=True)
    paths = ksc2_part_paths(parts_directory, expected_sizes=expected_sizes)
    reader = _MultipartReader(paths)
    audio_by_logical_member: dict[str, tuple[str, str, str, str]] = {}
    transcript_by_logical_member: dict[str, tuple[str, str, str]] = {}
    try:
        with gzip.GzipFile(fileobj=cast(BinaryIO, reader), mode="rb") as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode="r|") as archive:
                for member in archive:
                    path = _safe_member_path(member)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise Ksc2AuditError(
                            f"KSC2 archive contains an unsupported member type: {member.name!r}."
                        )
                    component = _component_name(path)
                    extension = path.suffix.lower()
                    if component not in KSC2_MIXED_ANNOTATION_COMPONENTS:
                        continue
                    if extension in KSC2_AUDIO_SUFFIXES:
                        logical_member, _repeated_suffix = _logical_member_stem(path, extension)
                        if logical_member in audio_by_logical_member:
                            raise Ksc2AuditError(
                                "Duplicate KSC2 annotation audio logical member: "
                                f"{logical_member!r}."
                            )
                        filename = (
                            hashlib.sha256(logical_member.encode("utf-8")).hexdigest() + extension
                        )
                        relative_path = (Path("assets") / Path(component) / filename).as_posix()
                        asset_hash = _copy_member_and_hash(
                            archive, member, output_directory / relative_path
                        )
                        audio_by_logical_member[logical_member] = (
                            component,
                            path.as_posix(),
                            relative_path,
                            asset_hash,
                        )
                    elif extension == KSC2_TRANSCRIPT_SUFFIX:
                        logical_member, _repeated_suffix = _logical_member_stem(
                            path, KSC2_TRANSCRIPT_SUFFIX
                        )
                        if logical_member in transcript_by_logical_member:
                            raise Ksc2AuditError(
                                "Duplicate KSC2 annotation transcript logical member: "
                                f"{logical_member!r}."
                            )
                        transcript, transcript_hash = _read_text_member_and_hash(archive, member)
                        transcript_by_logical_member[logical_member] = (
                            path.as_posix(),
                            transcript,
                            transcript_hash,
                        )
        # Reaching TAR EOF also verifies the gzip CRC.
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as error:
        raise Ksc2AuditError(f"KSC2 annotation archive cannot be read safely: {error}") from error
    finally:
        reader.close()

    receipts = reader.part_receipts
    if len(receipts) != len(paths):
        raise Ksc2AuditError("KSC2 annotation extraction did not consume every archive part.")
    if reader.compressed_sha256 != expected_hash:
        raise Ksc2AuditError(
            "KSC2 annotation archive SHA-256 mismatch: "
            f"expected {expected_hash}, got {reader.compressed_sha256}."
        )
    audio_ids = set(audio_by_logical_member)
    transcript_ids = set(transcript_by_logical_member)
    if audio_ids != transcript_ids:
        raise Ksc2AuditError(
            "KSC2 priority components have unpaired selected members: "
            f"audio_only={len(audio_ids.difference(transcript_ids))}, "
            f"transcript_only={len(transcript_ids.difference(audio_ids))}."
        )
    if not audio_ids:
        raise Ksc2AuditError("KSC2 priority components have no paired annotation candidates.")

    candidates: list[Ksc2AnnotationCandidate] = []
    for logical_member in sorted(audio_ids):
        component, archive_audio_member, relative_path, asset_hash = audio_by_logical_member[
            logical_member
        ]
        archive_transcript_member, transcript, transcript_hash = transcript_by_logical_member[
            logical_member
        ]
        if not logical_member.startswith(f"{KSC2_ARCHIVE_ROOT}/"):
            raise Ksc2AuditError(f"Invalid KSC2 annotation logical member: {logical_member!r}.")
        candidates.append(
            Ksc2AnnotationCandidate(
                candidate_id=logical_member.removeprefix(f"{KSC2_ARCHIVE_ROOT}/"),
                component=component,
                archive_audio_member=archive_audio_member,
                archive_transcript_member=archive_transcript_member,
                audio_relative_path=relative_path,
                audio_sha256=asset_hash,
                transcript=transcript,
                transcript_sha256=transcript_hash,
            )
        )
    return tuple(candidates)


def extract_ksc2_selected_audio(
    parts_directory: Path,
    output_directory: Path,
    *,
    selected_members: frozenset[str],
    expected_compressed_sha256: str,
    expected_sizes: tuple[int, ...] = KSC2_PART_EXPECTED_SIZES,
    progress_callback: Ksc2AuditProgressCallback | None = None,
) -> tuple[Ksc2ExtractedAudio, ...]:
    """Extract an exact member allow-list while revalidating the full multipart archive."""

    expected_hash = expected_compressed_sha256.lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise Ksc2AuditError("KSC2 extraction expected SHA-256 is invalid.")
    if not selected_members or output_directory.exists() or not output_directory.parent.is_dir():
        raise Ksc2AuditError("Unsafe KSC2 selected-audio extraction destination.")
    safe_members: dict[str, PurePosixPath] = {}
    for member_name in selected_members:
        path = PurePosixPath(member_name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member_name
            or not path.parts
            or path.parts[0] != KSC2_ARCHIVE_ROOT
            or not _component_name(path).startswith("Train/")
            or path.suffix.lower() not in KSC2_AUDIO_SUFFIXES
        ):
            raise Ksc2AuditError(f"Unsafe selected KSC2 audio member: {member_name!r}.")
        safe_members[member_name] = path
    paths = ksc2_part_paths(parts_directory, expected_sizes=expected_sizes)
    reader = _MultipartReader(paths, progress_callback=progress_callback)
    extracted: dict[str, Ksc2ExtractedAudio] = {}
    output_directory.mkdir()
    try:
        with gzip.GzipFile(fileobj=cast(BinaryIO, reader), mode="rb") as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode="r|") as archive:
                for member in archive:
                    path = _safe_member_path(member)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise Ksc2AuditError(
                            f"KSC2 archive contains an unsupported member type: {member.name!r}."
                        )
                    if member.name not in safe_members:
                        continue
                    if member.name in extracted:
                        raise Ksc2AuditError(
                            f"Duplicate selected KSC2 audio member: {member.name!r}."
                        )
                    relative_path = path.relative_to(KSC2_ARCHIVE_ROOT)
                    output_path = output_directory / Path(*relative_path.parts)
                    digest = _copy_member_and_hash(archive, member, output_path)
                    extracted[member.name] = Ksc2ExtractedAudio(
                        archive_member=member.name,
                        relative_path=relative_path.as_posix(),
                        sha256=digest,
                        size_bytes=member.size,
                    )
        # Reaching EOF verifies gzip CRC as well as every TAR member path/type.
    except Ksc2AuditError:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as error:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise Ksc2AuditError(f"KSC2 selected-audio extraction failed: {error}") from error
    finally:
        reader.close()
    if (
        len(reader.part_receipts) != len(paths)
        or reader.compressed_sha256 != expected_hash
        or set(extracted) != set(selected_members)
    ):
        shutil.rmtree(output_directory, ignore_errors=True)
        raise Ksc2AuditError(
            "KSC2 extraction did not verify the full archive or materialize every selected asset."
        )
    return tuple(extracted[name] for name in sorted(extracted))


def extract_ksc2_selected_transcripts(
    parts_directory: Path,
    output_directory: Path,
    *,
    selected_members: frozenset[str],
    expected_compressed_sha256: str,
    expected_sizes: tuple[int, ...] = KSC2_PART_EXPECTED_SIZES,
    progress_callback: Ksc2AuditProgressCallback | None = None,
) -> tuple[Ksc2ExtractedTranscript, ...]:
    """Extract an exact Train transcript allow-list while revalidating the multipart archive."""

    expected_hash = expected_compressed_sha256.lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise Ksc2AuditError("KSC2 transcript extraction expected SHA-256 is invalid.")
    if not selected_members or output_directory.exists() or not output_directory.parent.is_dir():
        raise Ksc2AuditError("Unsafe KSC2 transcript extraction destination.")
    for member_name in selected_members:
        path = PurePosixPath(member_name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member_name
            or not path.parts
            or path.parts[0] != KSC2_ARCHIVE_ROOT
            or not _component_name(path).startswith("Train/")
            or path.suffix.lower() != ".txt"
        ):
            raise Ksc2AuditError(f"Unsafe selected KSC2 transcript member: {member_name!r}.")
    paths = ksc2_part_paths(parts_directory, expected_sizes=expected_sizes)
    reader = _MultipartReader(paths, progress_callback=progress_callback)
    extracted: dict[str, Ksc2ExtractedTranscript] = {}
    output_directory.mkdir()
    try:
        with gzip.GzipFile(fileobj=cast(BinaryIO, reader), mode="rb") as gzip_stream:
            with tarfile.open(fileobj=gzip_stream, mode="r|") as archive:
                for member in archive:
                    path = _safe_member_path(member)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise Ksc2AuditError(
                            f"KSC2 archive contains an unsupported member type: {member.name!r}."
                        )
                    if member.name not in selected_members:
                        continue
                    if member.name in extracted:
                        raise Ksc2AuditError(
                            f"Duplicate selected KSC2 transcript member: {member.name!r}."
                        )
                    relative_path = path.relative_to(KSC2_ARCHIVE_ROOT)
                    output_path = output_directory / Path(*relative_path.parts)
                    digest = _copy_member_and_hash(archive, member, output_path)
                    try:
                        output_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError as error:
                        raise Ksc2AuditError(
                            f"Selected KSC2 transcript is not UTF-8: {member.name!r}."
                        ) from error
                    extracted[member.name] = Ksc2ExtractedTranscript(
                        archive_member=member.name,
                        relative_path=relative_path.as_posix(),
                        sha256=digest,
                        size_bytes=member.size,
                    )
    except Ksc2AuditError:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as error:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise Ksc2AuditError(f"KSC2 transcript extraction failed: {error}") from error
    finally:
        reader.close()
    if (
        len(reader.part_receipts) != len(paths)
        or reader.compressed_sha256 != expected_hash
        or set(extracted) != set(selected_members)
    ):
        shutil.rmtree(output_directory, ignore_errors=True)
        raise Ksc2AuditError(
            "KSC2 transcript extraction did not verify the full archive or selected allow-list."
        )
    return tuple(extracted[name] for name in sorted(extracted))
