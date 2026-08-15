"""Auditable intake primitives for Kazakh Speech Corpus / OpenSLR SLR102.

The official FLAC archive contains audio, paired UTF-8 transcripts, and the
original KSC split metadata.  This module reads that layout directly and never
invents speaker or code-switch labels where the corpus does not supply them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TypeVar

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.manifest import ManifestRow

KSC_SOURCE_ID = "ksc_slr102"
KSC_ARCHIVE_NAME = "ISSAI_KSC_335RS_v1.1_flac.tar.gz"
KSC_ARCHIVE_EXPECTED_SIZE_BYTES = 19_092_377_812
KSC_ARCHIVE_ROOT = "ISSAI_KSC_335RS_v1.1_flac"
KSC_AUDIO_DIRECTORY = f"{KSC_ARCHIVE_ROOT}/Audios_flac"
KSC_TRANSCRIPT_DIRECTORY = f"{KSC_ARCHIVE_ROOT}/Transcriptions"
KSC_METADATA_DIRECTORY = f"{KSC_ARCHIVE_ROOT}/Meta"
KSC_METADATA_CHECKPOINT_DIRECTORY = f"{KSC_METADATA_DIRECTORY}/.ipynb_checkpoints"
KSC_METADATA_SPLITS: Mapping[str, str] = {
    "train": "train.csv",
    "dev": "dev.csv",
    "test": "test.csv",
}
KSC_SOURCE_SPLITS: Mapping[str, str] = {"train": "train", "dev": "dev", "test": "test"}
KSC_SOURCE_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
KSC_RIGHTS_BASIS = "OpenSLR SLR102 / KSC, CC-BY-4.0; attribution retained"
KSC_CAPTURE_ROUTE = "crowdsourced_web_recording"
KSC_UNKNOWN = "unknown"


class KscIngestionError(ValueError):
    """Raised when a KSC artifact cannot safely become a project manifest."""


@dataclass(frozen=True, slots=True)
class KscMetadataIndexRecord:
    utterance_id: str
    split: str
    device_id: str
    code_switch: str = KSC_UNKNOWN


@dataclass(frozen=True, slots=True)
class KscMetadataRecord(KscMetadataIndexRecord):
    transcript: str = ""


@dataclass(frozen=True, slots=True)
class ExtractedKscAsset:
    utterance_id: str
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int
    codec: str


@dataclass(frozen=True, slots=True)
class KscArchiveReport:
    archive: Path
    audio_files: int
    transcript_files: int
    metadata_files: int


RecordT = TypeVar("RecordT", bound=KscMetadataIndexRecord)


def _safe_utterance_id(value: str) -> str:
    utterance_id = value.strip()
    path = PurePosixPath(utterance_id)
    if (
        not utterance_id
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != utterance_id
        or "\\" in utterance_id
        or utterance_id in {".", ".."}
    ):
        raise KscIngestionError(f"Invalid KSC utterance id: {value!r}.")
    return utterance_id


def _safe_path_below(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise KscIngestionError(f"Path escapes its declared root: {relative_path!r}.") from error
    return candidate


def _canonical_transcript(text: str) -> str:
    return " ".join(text.split())


def _load_transcript(metadata_root: Path, utterance_id: str) -> str:
    path = _safe_path_below(metadata_root, f"Transcriptions/{utterance_id}.txt")
    if not path.is_file():
        raise KscIngestionError(f"Missing KSC transcript for utterance {utterance_id!r}: {path}")
    try:
        transcript = _canonical_transcript(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise KscIngestionError(f"KSC transcript is not valid UTF-8: {path}") from error
    if not transcript:
        raise KscIngestionError(f"KSC transcript is empty: {path}")
    return transcript


def _requested_splits(splits: Iterable[str]) -> tuple[str, ...]:
    requested_splits = tuple(splits)
    if not requested_splits:
        raise KscIngestionError("At least one KSC source split is required.")
    unknown_splits = sorted(set(requested_splits).difference(KSC_METADATA_SPLITS))
    if unknown_splits:
        raise KscIngestionError(f"Unsupported KSC source splits: {', '.join(unknown_splits)}.")
    return requested_splits


def _parse_metadata_csv(
    content: str, split: str, source_description: str
) -> list[KscMetadataIndexRecord]:
    reader: csv.DictReader[str] | None = None
    for delimiter in (",", "\t", " "):
        candidate = csv.DictReader(io.StringIO(content), delimiter=delimiter, skipinitialspace=True)
        if candidate.fieldnames is not None and "uttID" in candidate.fieldnames:
            reader = candidate
            break
    if reader is None:
        raise KscIngestionError(
            f"KSC metadata must contain an 'uttID' column: {source_description}"
        )

    records: list[KscMetadataIndexRecord] = []
    for row_number, row in enumerate(reader, start=2):
        raw_id = row.get("uttID") or ""
        try:
            utterance_id = _safe_utterance_id(raw_id)
        except KscIngestionError as error:
            raise KscIngestionError(f"{source_description}:{row_number}: {error}") from error
        device_id = (row.get("deviceID") or KSC_UNKNOWN).strip() or KSC_UNKNOWN
        records.append(
            KscMetadataIndexRecord(
                utterance_id=utterance_id,
                split=split,
                device_id=device_id,
            )
        )
    return records


def _validate_unique_metadata_ids(records: Iterable[KscMetadataIndexRecord]) -> None:
    seen_ids: set[str] = set()
    for record in records:
        if record.utterance_id in seen_ids:
            raise KscIngestionError(
                f"Duplicate KSC utterance id across source splits: {record.utterance_id!r}."
            )
        seen_ids.add(record.utterance_id)


def attach_ksc_transcripts(
    records: Iterable[KscMetadataIndexRecord], transcript_root: Path
) -> list[KscMetadataRecord]:
    return [
        KscMetadataRecord(
            utterance_id=record.utterance_id,
            split=record.split,
            device_id=record.device_id,
            transcript=_load_transcript(transcript_root, record.utterance_id),
            code_switch=record.code_switch,
        )
        for record in records
    ]


def load_ksc_metadata(metadata_root: Path, splits: Iterable[str]) -> list[KscMetadataRecord]:
    """Load a directory copy of KSC metadata and paired UTF-8 transcripts."""

    if not metadata_root.is_dir():
        raise KscIngestionError(f"KSC metadata root does not exist: {metadata_root}")
    requested_splits = _requested_splits(splits)

    index_records: list[KscMetadataIndexRecord] = []
    for split in requested_splits:
        metadata_name = KSC_METADATA_SPLITS[split]
        metadata_path = _safe_path_below(metadata_root, f"Meta/{metadata_name}")
        if not metadata_path.is_file():
            raise KscIngestionError(f"Missing KSC split metadata: {metadata_path}")
        try:
            content = metadata_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise KscIngestionError(f"KSC metadata is not valid UTF-8: {metadata_path}") from error
        index_records.extend(_parse_metadata_csv(content, split, str(metadata_path)))
    _validate_unique_metadata_ids(index_records)
    return attach_ksc_transcripts(index_records, metadata_root)


def select_ksc_records(
    records: Iterable[RecordT],
    limit: int | None,
    seed: str,
    *,
    excluded_utterance_ids: Iterable[str] = (),
) -> list[RecordT]:
    """Choose a reproducible source-balanced subset without changing source splits."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when it is set.")
    if not seed:
        raise ValueError("seed must not be empty.")
    excluded = {_safe_utterance_id(utterance_id) for utterance_id in excluded_utterance_ids}
    grouped: dict[str, list[RecordT]] = {}
    for record in records:
        if record.utterance_id in excluded:
            continue
        grouped.setdefault(record.split, []).append(record)
    selected: list[RecordT] = []
    for split, split_records in sorted(grouped.items()):
        ranked = sorted(
            split_records,
            key=lambda item: hashlib.sha256(f"{seed}:{item.utterance_id}".encode()).digest(),
        )
        if limit is not None and len(ranked) < limit:
            raise KscIngestionError(
                f"KSC source split {split!r} has only {len(ranked)} records after exclusions; "
                f"need {limit}."
            )
        selected.extend(ranked if limit is None else ranked[:limit])
    return selected


def select_ksc_records_from_archive_excluding_texts(
    archive: Path,
    records: Iterable[KscMetadataIndexRecord],
    limit: int,
    seed: str,
    *,
    excluded_utterance_ids: Iterable[str] = (),
    excluded_text_hashes: Iterable[str] = (),
    transcript_filter: Callable[[str], bool] | None = None,
) -> tuple[list[KscMetadataIndexRecord], KscArchiveReport]:
    """Select fresh KSC records by text before audio extraction can publish a slice.

    The archive is streamed once to inspect every member and rank only transcripts whose hashes
    are absent from frozen manifests and that pass an optional caller-supplied text predicate.
    This avoids the unsafe pattern of extracting audio first and discovering a text collision or
    unsupported synthesis text only after a destination has become visible.
    """

    if limit <= 0:
        raise ValueError("limit must be positive.")
    if not seed:
        raise ValueError("seed must not be empty.")
    _validate_archive_size(archive)
    excluded_ids = {_safe_utterance_id(value) for value in excluded_utterance_ids}
    candidates = [record for record in records if record.utterance_id not in excluded_ids]
    _validate_unique_metadata_ids(candidates)
    ranked = sorted(
        candidates,
        key=lambda item: hashlib.sha256(f"{seed}:{item.utterance_id}".encode()).digest(),
    )
    if len(ranked) < limit:
        raise KscIngestionError(
            f"KSC source has only {len(ranked)} records after ID exclusions; need {limit}."
        )
    rank_by_id = {record.utterance_id: index for index, record in enumerate(ranked)}
    record_by_id = {record.utterance_id: record for record in ranked}
    excluded_hashes = set(excluded_text_hashes)
    eligible_by_text_hash: dict[str, str] = {}
    audio_ids: set[str] = set()
    transcript_ids: set[str] = set()
    metadata_files: set[str] = set()
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if not _is_expected_member(member):
                    raise KscIngestionError(f"Unexpected KSC archive member: {member.name!r}.")
                audio_id = _member_utterance_id(member, KSC_AUDIO_DIRECTORY, ".flac")
                if audio_id is not None:
                    if audio_id in audio_ids:
                        raise KscIngestionError(f"Duplicate KSC audio member: {audio_id!r}.")
                    audio_ids.add(audio_id)
                    continue
                transcript_id = _member_utterance_id(member, KSC_TRANSCRIPT_DIRECTORY, ".txt")
                if transcript_id is not None:
                    if transcript_id in transcript_ids:
                        raise KscIngestionError(
                            f"Duplicate KSC transcript member: {transcript_id!r}."
                        )
                    transcript_ids.add(transcript_id)
                    if transcript_id not in rank_by_id:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        raise KscIngestionError(
                            f"Cannot read KSC transcript member: {member.name!r}."
                        )
                    try:
                        with source:
                            transcript = _canonical_transcript(source.read().decode("utf-8"))
                    except UnicodeDecodeError as error:
                        raise KscIngestionError(
                            f"KSC transcript is not valid UTF-8: {member.name}"
                        ) from error
                    if not transcript:
                        raise KscIngestionError(f"KSC transcript is empty: {member.name}")
                    text_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                    if text_hash in excluded_hashes:
                        continue
                    if transcript_filter is not None and not transcript_filter(transcript):
                        continue
                    prior_id = eligible_by_text_hash.get(text_hash)
                    if prior_id is None or rank_by_id[transcript_id] < rank_by_id[prior_id]:
                        eligible_by_text_hash[text_hash] = transcript_id
                    continue
                if member.name.startswith(f"{KSC_METADATA_DIRECTORY}/") and member.isfile():
                    metadata_files.add(member.name)
    except (tarfile.TarError, OSError) as error:
        raise KscIngestionError(f"KSC archive cannot be read safely: {error}") from error
    if audio_ids != transcript_ids:
        raise KscIngestionError("KSC audio and transcript utterance ids do not match exactly.")
    expected_metadata = {_metadata_member_name(name) for name in KSC_METADATA_SPLITS.values()}
    if metadata_files != expected_metadata:
        raise KscIngestionError(
            "KSC archive does not contain exactly train/dev/test metadata files."
        )
    selected = sorted(
        (record_by_id[value] for value in eligible_by_text_hash.values()),
        key=lambda item: rank_by_id[item.utterance_id],
    )[:limit]
    if len(selected) < limit:
        raise KscIngestionError(
            f"KSC source has only {len(selected)} unique-text, text-disjoint records after "
            "exclusions; "
            f"need {limit}."
        )
    return selected, KscArchiveReport(
        archive=archive,
        audio_files=len(audio_ids),
        transcript_files=len(transcript_ids),
        metadata_files=len(metadata_files),
    )


def _validate_archive_size(archive: Path) -> None:
    if not archive.is_file():
        raise KscIngestionError(f"KSC archive does not exist: {archive}")
    actual_size = archive.stat().st_size
    if actual_size != KSC_ARCHIVE_EXPECTED_SIZE_BYTES:
        raise KscIngestionError(
            "KSC archive size mismatch: "
            f"expected {KSC_ARCHIVE_EXPECTED_SIZE_BYTES} bytes, got {actual_size}. "
            "Refusing extraction; acquire a clean archive before continuing."
        )


def _member_utterance_id(member: tarfile.TarInfo, directory: str, suffix: str) -> str | None:
    path = PurePosixPath(member.name)
    expected_directory = PurePosixPath(directory)
    if (
        not member.isfile()
        or len(path.parts) != 3
        or path.parts[:2] != expected_directory.parts
        or path.suffix.lower() != suffix
    ):
        return None
    return _safe_utterance_id(path.stem)


def _is_expected_member(member: tarfile.TarInfo) -> bool:
    if member.isdir():
        return member.name.rstrip("/") in {
            KSC_ARCHIVE_ROOT,
            KSC_AUDIO_DIRECTORY,
            KSC_TRANSCRIPT_DIRECTORY,
            KSC_METADATA_DIRECTORY,
            KSC_METADATA_CHECKPOINT_DIRECTORY,
        }
    if not member.isfile():
        return False
    if _member_utterance_id(member, KSC_AUDIO_DIRECTORY, ".flac") is not None:
        return True
    if _member_utterance_id(member, KSC_TRANSCRIPT_DIRECTORY, ".txt") is not None:
        return True
    return member.name in {_metadata_member_name(name) for name in KSC_METADATA_SPLITS.values()}


def _metadata_member_name(filename: str) -> str:
    return f"{KSC_METADATA_DIRECTORY}/{filename}"


def inspect_ksc_archive(archive: Path) -> KscArchiveReport:
    """Validate expected size, gzip CRC, TAR structure, and paired KSC assets."""

    _validate_archive_size(archive)
    audio_files = 0
    transcript_files = 0
    audio_ids: set[str] = set()
    transcript_ids: set[str] = set()
    metadata_files: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not _is_expected_member(member):
                    raise KscIngestionError(f"Unexpected KSC archive member: {member.name!r}.")
                audio_id = _member_utterance_id(member, KSC_AUDIO_DIRECTORY, ".flac")
                if audio_id is not None:
                    if audio_id in audio_ids:
                        raise KscIngestionError(
                            f"Duplicate KSC audio member for utterance {audio_id!r}."
                        )
                    audio_ids.add(audio_id)
                    audio_files += 1
                transcript_id = _member_utterance_id(member, KSC_TRANSCRIPT_DIRECTORY, ".txt")
                if transcript_id is not None:
                    if transcript_id in transcript_ids:
                        raise KscIngestionError(
                            f"Duplicate KSC transcript for utterance {transcript_id!r}."
                        )
                    transcript_ids.add(transcript_id)
                    transcript_files += 1
                if member.name.startswith(f"{KSC_METADATA_DIRECTORY}/") and member.isfile():
                    metadata_files.add(member.name)
    except (tarfile.TarError, OSError) as error:
        raise KscIngestionError(f"KSC archive cannot be read safely: {error}") from error
    if audio_files == 0:
        raise KscIngestionError("KSC archive contains no FLAC audio members.")
    if audio_ids != transcript_ids:
        raise KscIngestionError("KSC audio and transcript utterance ids do not match exactly.")
    expected_metadata = {_metadata_member_name(name) for name in KSC_METADATA_SPLITS.values()}
    if metadata_files != expected_metadata:
        raise KscIngestionError(
            "KSC archive does not contain exactly train/dev/test metadata files."
        )
    return KscArchiveReport(
        archive=archive,
        audio_files=audio_files,
        transcript_files=transcript_files,
        metadata_files=len(metadata_files),
    )


def load_ksc_metadata_from_archive(
    archive: Path, splits: Iterable[str]
) -> list[KscMetadataIndexRecord]:
    """Read requested original KSC split CSVs directly from a verified archive."""

    _validate_archive_size(archive)
    requested_splits = _requested_splits(splits)
    wanted_members = {
        _metadata_member_name(KSC_METADATA_SPLITS[split]): split for split in requested_splits
    }
    contents: dict[str, str] = {}
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not _is_expected_member(member):
                    raise KscIngestionError(f"Unexpected KSC archive member: {member.name!r}.")
                split = wanted_members.get(member.name)
                if split is None:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    raise KscIngestionError(f"Cannot read KSC metadata member: {member.name!r}.")
                try:
                    with source:
                        contents[split] = source.read().decode("utf-8")
                except UnicodeDecodeError as error:
                    raise KscIngestionError(
                        f"KSC metadata is not valid UTF-8: {member.name}"
                    ) from error
    except (tarfile.TarError, OSError) as error:
        raise KscIngestionError(f"KSC metadata cannot be read safely: {error}") from error

    missing_splits = sorted(set(requested_splits).difference(contents))
    if missing_splits:
        raise KscIngestionError(
            f"KSC archive is missing metadata for: {', '.join(missing_splits)}."
        )
    records = [
        record
        for split in requested_splits
        for record in _parse_metadata_csv(
            contents[split], split, _metadata_member_name(KSC_METADATA_SPLITS[split])
        )
    ]
    _validate_unique_metadata_ids(records)
    return records


def _audio_member_name(utterance_id: str) -> str:
    return f"{KSC_AUDIO_DIRECTORY}/{_safe_utterance_id(utterance_id)}.flac"


def _transcript_member_name(utterance_id: str) -> str:
    return f"{KSC_TRANSCRIPT_DIRECTORY}/{_safe_utterance_id(utterance_id)}.txt"


def extract_ksc_audio_slice(
    archive: Path,
    utterance_ids: Iterable[str],
    destination: Path,
    *,
    excluded_text_hashes: Iterable[str] = (),
) -> dict[str, Path]:
    """Extract requested FLACs and paired transcripts without ``extractall``.

    The destination is made visible only after the whole compressed TAR stream
    has been checked.  This prevents a partial corpus slice from being mistaken
    for a validated one if the archive is truncated or contains unsafe members.
    """

    requested_ids = {_safe_utterance_id(value) for value in utterance_ids}
    excluded_hashes = set(excluded_text_hashes)
    if not requested_ids:
        raise KscIngestionError("No KSC utterance ids were requested for extraction.")
    if destination.exists():
        raise KscIngestionError(f"Refusing to overwrite KSC extraction destination: {destination}")
    if not destination.parent.is_dir():
        raise KscIngestionError(f"KSC extraction parent does not exist: {destination.parent}")
    _validate_archive_size(archive)

    extracted: dict[str, Path] = {}
    extracted_transcripts: set[str] = set()
    seen_audio_ids: set[str] = set()
    seen_transcript_ids: set[str] = set()
    seen_metadata_files: set[str] = set()
    try:
        with tempfile.TemporaryDirectory(prefix="kds-ksc-", dir=destination.parent) as stage_dir:
            stage = Path(stage_dir)
            with tarfile.open(archive, mode="r|gz") as tar:
                for member in tar:
                    if not _is_expected_member(member):
                        raise KscIngestionError(f"Unexpected KSC archive member: {member.name!r}.")
                    audio_id = _member_utterance_id(member, KSC_AUDIO_DIRECTORY, ".flac")
                    transcript_id = _member_utterance_id(member, KSC_TRANSCRIPT_DIRECTORY, ".txt")
                    if member.name.startswith(f"{KSC_METADATA_DIRECTORY}/") and member.isfile():
                        seen_metadata_files.add(member.name)
                    utterance_id = audio_id or transcript_id
                    if utterance_id is None:
                        continue
                    seen_ids = seen_audio_ids if audio_id is not None else seen_transcript_ids
                    if utterance_id in seen_ids:
                        raise KscIngestionError(
                            f"Duplicate KSC member for utterance {utterance_id!r}."
                        )
                    seen_ids.add(utterance_id)
                    if utterance_id not in requested_ids:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        raise KscIngestionError(f"Cannot read KSC audio member: {member.name!r}.")
                    subdirectory = "Audios_flac" if audio_id is not None else "Transcriptions"
                    suffix = ".flac" if audio_id is not None else ".txt"
                    output_path = stage / subdirectory / f"{utterance_id}{suffix}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with source, output_path.open("xb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                    if audio_id is not None:
                        extracted[utterance_id] = output_path
                    else:
                        extracted_transcripts.add(utterance_id)
            missing_audio = requested_ids.difference(extracted)
            missing_transcripts = requested_ids.difference(extracted_transcripts)
            missing = sorted(missing_audio.union(missing_transcripts))
            if seen_audio_ids != seen_transcript_ids:
                raise KscIngestionError(
                    "KSC audio and transcript utterance ids do not match exactly."
                )
            expected_metadata = {
                _metadata_member_name(name) for name in KSC_METADATA_SPLITS.values()
            }
            if seen_metadata_files != expected_metadata:
                raise KscIngestionError(
                    "KSC archive does not contain exactly train/dev/test metadata files."
                )
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "..." if len(missing) > 5 else ""
                raise KscIngestionError(
                    f"Requested KSC audio is absent from archive: {preview}{suffix}"
                )
            reused_text_hashes: set[str] = set()
            for utterance_id in requested_ids:
                transcript_path = stage / "Transcriptions" / f"{utterance_id}.txt"
                try:
                    transcript = _canonical_transcript(transcript_path.read_text(encoding="utf-8"))
                except UnicodeDecodeError as error:
                    raise KscIngestionError(
                        f"KSC transcript is not valid UTF-8: {transcript_path}"
                    ) from error
                if not transcript:
                    raise KscIngestionError(f"KSC transcript is empty: {transcript_path}")
                text_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                if text_hash in excluded_hashes:
                    reused_text_hashes.add(text_hash)
            if reused_text_hashes:
                raise KscIngestionError(
                    "KSC selection overlaps a frozen manifest by transcript text hash; "
                    f"found {len(reused_text_hashes)} collisions."
                )
            stage.replace(destination)
    except (tarfile.TarError, OSError) as error:
        raise KscIngestionError(f"KSC extraction failed safely: {error}") from error
    return {
        utterance_id: destination / "Audios_flac" / f"{utterance_id}.flac"
        for utterance_id in extracted
    }


def inspect_extracted_ksc_audio(path: Path) -> tuple[float, int, str]:
    """Read stable FLAC properties required by the project manifest."""

    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise KscIngestionError(f"Cannot inspect extracted KSC audio: {path}") from error
    duration = float(info.duration)
    sample_rate = int(info.samplerate)
    codec = str(info.format).lower()
    if duration <= 0 or sample_rate <= 0 or codec != "flac":
        raise KscIngestionError(f"Invalid KSC FLAC properties for {path}.")
    return duration, sample_rate, codec


def ksc_manifest_rows(
    records: Iterable[KscMetadataRecord],
    assets: Mapping[str, ExtractedKscAsset],
    created_at: str | None = None,
) -> list[ManifestRow]:
    """Create bona-fide rows while retaining KSC's original split protocol."""

    timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows: list[ManifestRow] = []
    for record in records:
        asset = assets.get(record.utterance_id)
        if asset is None:
            raise KscIngestionError(
                f"No extracted asset for KSC utterance {record.utterance_id!r}."
            )
        transcript_hash = hashlib.sha256(record.transcript.encode("utf-8")).hexdigest()
        split = KSC_SOURCE_SPLITS[record.split]
        source_partition = f"{KSC_SOURCE_ID}:source-split:{record.split}"
        rows.append(
            ManifestRow(
                sample_id=f"{KSC_SOURCE_ID}:{record.utterance_id}",
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split=split,
                label="bonafide",
                language="kk",
                code_switch=record.code_switch,
                parent_group_id=source_partition,
                source_name=KSC_SOURCE_ID,
                source_license=KSC_SOURCE_LICENSE,
                rights_basis=KSC_RIGHTS_BASIS,
                speaker_pseudo_id=f"{source_partition}:speaker-unknown",
                text_id=f"{KSC_SOURCE_ID}:{record.utterance_id}",
                text_hash=transcript_hash,
                duration_s=asset.duration_s,
                generator_family="",
                generator_name="",
                generator_version="",
                voice_id="",
                clone_consent_id="",
                device=KSC_UNKNOWN,
                capture_route=KSC_CAPTURE_ROUTE,
                original_sr=asset.original_sr,
                codec=asset.codec,
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows
