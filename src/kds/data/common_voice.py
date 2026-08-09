"""Safe, auditable intake primitives for Common Voice Russian v24."""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TypeVar

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.manifest import ManifestRow

COMMON_VOICE_RU_V24_SOURCE_ID = "common_voice_ru_v24"
COMMON_VOICE_RU_V24_ARCHIVE_NAME = "cv-corpus-24.0-2025-12-05-ru.tar.gz"
COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES = 7_008_716_262
COMMON_VOICE_RU_V24_ARCHIVE_ROOT = "cv-corpus-24.0-2025-12-05"
COMMON_VOICE_RU_V24_LOCALE = "ru"
COMMON_VOICE_RU_V24_DIRECTORY = (
    f"{COMMON_VOICE_RU_V24_ARCHIVE_ROOT}/{COMMON_VOICE_RU_V24_LOCALE}"
)
COMMON_VOICE_RU_V24_CLIPS_DIRECTORY = f"{COMMON_VOICE_RU_V24_DIRECTORY}/clips"
COMMON_VOICE_RU_V24_METADATA_FILENAMES = frozenset(
    {
        "clip_durations.tsv",
        "dev.tsv",
        "invalidated.tsv",
        "other.tsv",
        "reported.tsv",
        "test.tsv",
        "train.tsv",
        "unvalidated_sentences.tsv",
        "validated.tsv",
        "validated_sentences.tsv",
    }
)
COMMON_VOICE_RU_V24_SOURCE_SPLITS: Mapping[str, str] = {
    "train": "train",
    "dev": "dev",
    "test": "test",
}
COMMON_VOICE_RU_V24_REQUIRED_COLUMNS = frozenset(
    {"client_id", "path", "sentence_id", "sentence", "locale"}
)
COMMON_VOICE_RU_V24_SOURCE_LICENSE = "CC0-1.0"
COMMON_VOICE_RU_V24_SOURCE_URL = (
    "https://dev.mozilladatacollective.com/datasets/cmj8l8ct700o5nlovbdnv58yr"
)
COMMON_VOICE_RU_V24_RIGHTS_BASIS = (
    "Common Voice RU v24, CC0-1.0; owner-authorized personal research; "
    "no speaker identification or re-hosting"
)
COMMON_VOICE_RU_V24_CAPTURE_ROUTE = "crowdsourced_web_recording"
COMMON_VOICE_UNKNOWN = "unknown"


class CommonVoiceIngestionError(ValueError):
    """Raised when a Common Voice artifact cannot safely become a project manifest."""


@dataclass(frozen=True, slots=True)
class CommonVoiceRecord:
    clip_name: str
    split: str
    client_id: str
    sentence_id: str
    sentence: str


@dataclass(frozen=True, slots=True)
class ExtractedCommonVoiceAsset:
    clip_name: str
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int


@dataclass(frozen=True, slots=True)
class CommonVoiceArchiveReport:
    archive: Path
    audio_files: int
    metadata_files: int


RecordT = TypeVar("RecordT", bound=CommonVoiceRecord)


def _safe_clip_name(value: str) -> str:
    clip_name = value.strip()
    path = PurePosixPath(clip_name)
    if (
        not clip_name
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != clip_name
        or path.suffix.lower() != ".mp3"
        or "\\" in clip_name
        or clip_name in {".", ".."}
    ):
        raise CommonVoiceIngestionError(f"Invalid Common Voice clip path: {value!r}.")
    return clip_name


def _safe_path_below(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise CommonVoiceIngestionError(
            f"Path escapes its declared root: {relative_path!r}."
        ) from error
    return candidate


def _canonical_sentence(value: str) -> str:
    return " ".join(value.split())


def _requested_splits(splits: Iterable[str]) -> tuple[str, ...]:
    requested_splits = tuple(splits)
    if not requested_splits:
        raise CommonVoiceIngestionError("At least one Common Voice source split is required.")
    unknown_splits = sorted(set(requested_splits).difference(COMMON_VOICE_RU_V24_SOURCE_SPLITS))
    if unknown_splits:
        raise CommonVoiceIngestionError(
            f"Unsupported Common Voice source splits: {', '.join(unknown_splits)}."
        )
    return requested_splits


def _metadata_member_name(split: str) -> str:
    return f"{COMMON_VOICE_RU_V24_DIRECTORY}/{split}.tsv"


def _parse_metadata_tsv(
    content: str, split: str, source_description: str
) -> list[CommonVoiceRecord]:
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    if reader.fieldnames is None:
        raise CommonVoiceIngestionError(
            f"Common Voice metadata has no header: {source_description}"
        )
    missing = sorted(COMMON_VOICE_RU_V24_REQUIRED_COLUMNS.difference(reader.fieldnames))
    if missing:
        raise CommonVoiceIngestionError(
            f"Common Voice metadata is missing columns {', '.join(missing)}: {source_description}"
        )

    records: list[CommonVoiceRecord] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            clip_name = _safe_clip_name(row.get("path") or "")
        except CommonVoiceIngestionError as error:
            raise CommonVoiceIngestionError(
                f"{source_description}:{row_number}: {error}"
            ) from error
        client_id = (row.get("client_id") or "").strip()
        sentence_id = (row.get("sentence_id") or "").strip()
        sentence = _canonical_sentence(row.get("sentence") or "")
        locale = (row.get("locale") or "").strip().lower()
        blank = [
            name
            for name, value in (
                ("client_id", client_id),
                ("sentence_id", sentence_id),
                ("sentence", sentence),
            )
            if not value
        ]
        if blank:
            raise CommonVoiceIngestionError(
                f"{source_description}:{row_number}: blank values: {', '.join(blank)}."
            )
        if locale != COMMON_VOICE_RU_V24_LOCALE:
            raise CommonVoiceIngestionError(
                f"{source_description}:{row_number}: expected locale 'ru', got {locale!r}."
            )
        records.append(
            CommonVoiceRecord(
                clip_name=clip_name,
                split=split,
                client_id=client_id,
                sentence_id=sentence_id,
                sentence=sentence,
            )
        )
    return records


def _validate_unique_clip_names(records: Iterable[CommonVoiceRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.clip_name in seen:
            raise CommonVoiceIngestionError(
                f"Duplicate Common Voice clip path across source splits: {record.clip_name!r}."
            )
        seen.add(record.clip_name)


def select_common_voice_records(
    records: Iterable[RecordT], limit: int | None, seed: str
) -> list[RecordT]:
    """Choose a reproducible source-balanced subset before local leakage-safe splitting."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when it is set.")
    if not seed:
        raise ValueError("seed must not be empty.")
    grouped: dict[str, list[RecordT]] = {}
    for record in records:
        grouped.setdefault(record.split, []).append(record)
    selected: list[RecordT] = []
    for _split, split_records in sorted(grouped.items()):
        ranked = sorted(
            split_records,
            key=lambda item: hashlib.sha256(f"{seed}:{item.clip_name}".encode()).digest(),
        )
        selected.extend(ranked if limit is None else ranked[:limit])
    return selected


def _validate_archive_size(archive: Path) -> None:
    if not archive.is_file():
        raise CommonVoiceIngestionError(f"Common Voice archive does not exist: {archive}")
    actual_size = archive.stat().st_size
    if actual_size != COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES:
        raise CommonVoiceIngestionError(
            "Common Voice archive size mismatch: "
            f"expected {COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES} bytes, "
            f"got {actual_size}. Refusing extraction; acquire a clean archive before continuing."
        )


def _member_clip_name(member: tarfile.TarInfo) -> str | None:
    path = PurePosixPath(member.name)
    expected_directory = PurePosixPath(COMMON_VOICE_RU_V24_CLIPS_DIRECTORY)
    if (
        not member.isfile()
        or len(path.parts) != 4
        or path.parts[:3] != expected_directory.parts
        or path.suffix.lower() != ".mp3"
    ):
        return None
    return _safe_clip_name(path.name)


def _is_expected_member(member: tarfile.TarInfo) -> bool:
    if member.isdir():
        return member.name.rstrip("/") in {
            COMMON_VOICE_RU_V24_ARCHIVE_ROOT,
            COMMON_VOICE_RU_V24_DIRECTORY,
            COMMON_VOICE_RU_V24_CLIPS_DIRECTORY,
        }
    if not member.isfile():
        return False
    if _member_clip_name(member) is not None:
        return True
    return member.name in {
        f"{COMMON_VOICE_RU_V24_DIRECTORY}/{filename}"
        for filename in COMMON_VOICE_RU_V24_METADATA_FILENAMES
    }


def inspect_common_voice_archive(archive: Path) -> CommonVoiceArchiveReport:
    """Validate expected size, gzip CRC, strict tar layout, and unique MP3 members."""

    _validate_archive_size(archive)
    audio_names: set[str] = set()
    metadata_files: set[str] = set()
    expected_metadata = {
        f"{COMMON_VOICE_RU_V24_DIRECTORY}/{filename}"
        for filename in COMMON_VOICE_RU_V24_METADATA_FILENAMES
    }
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not _is_expected_member(member):
                    raise CommonVoiceIngestionError(
                        f"Unexpected Common Voice archive member: {member.name!r}."
                    )
                clip_name = _member_clip_name(member)
                if clip_name is not None:
                    if clip_name in audio_names:
                        raise CommonVoiceIngestionError(
                            f"Duplicate Common Voice audio member: {clip_name!r}."
                        )
                    audio_names.add(clip_name)
                if member.name in expected_metadata:
                    metadata_files.add(member.name)
    except (tarfile.TarError, OSError) as error:
        raise CommonVoiceIngestionError(
            f"Common Voice archive cannot be read safely: {error}"
        ) from error
    if not audio_names:
        raise CommonVoiceIngestionError("Common Voice archive contains no MP3 audio members.")
    if metadata_files != expected_metadata:
        raise CommonVoiceIngestionError(
            "Common Voice archive does not contain exactly the expected metadata files."
        )
    return CommonVoiceArchiveReport(
        archive=archive, audio_files=len(audio_names), metadata_files=len(metadata_files)
    )


def load_common_voice_metadata_from_archive(
    archive: Path, splits: Iterable[str]
) -> list[CommonVoiceRecord]:
    """Read requested official Common Voice split TSVs from a verified archive."""

    _validate_archive_size(archive)
    requested_splits = _requested_splits(splits)
    wanted_members = {_metadata_member_name(split): split for split in requested_splits}
    expected_metadata = {
        f"{COMMON_VOICE_RU_V24_DIRECTORY}/{filename}"
        for filename in COMMON_VOICE_RU_V24_METADATA_FILENAMES
    }
    contents: dict[str, str] = {}
    seen_metadata: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not _is_expected_member(member):
                    raise CommonVoiceIngestionError(
                        f"Unexpected Common Voice archive member: {member.name!r}."
                    )
                if member.name in expected_metadata:
                    seen_metadata.add(member.name)
                split = wanted_members.get(member.name)
                if split is None:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    raise CommonVoiceIngestionError(
                        f"Cannot read Common Voice metadata member: {member.name!r}."
                    )
                try:
                    with source:
                        contents[split] = source.read().decode("utf-8-sig")
                except UnicodeDecodeError as error:
                    raise CommonVoiceIngestionError(
                        f"Common Voice metadata is not valid UTF-8: {member.name}"
                    ) from error
    except (tarfile.TarError, OSError) as error:
        raise CommonVoiceIngestionError(
            f"Common Voice metadata cannot be read safely: {error}"
        ) from error
    missing_splits = sorted(set(requested_splits).difference(contents))
    if missing_splits:
        raise CommonVoiceIngestionError(
            f"Common Voice archive is missing metadata for: {', '.join(missing_splits)}."
        )
    if seen_metadata != expected_metadata:
        raise CommonVoiceIngestionError(
            "Common Voice archive does not contain exactly the expected metadata files."
        )
    records = [
        record
        for split in requested_splits
        for record in _parse_metadata_tsv(contents[split], split, _metadata_member_name(split))
    ]
    _validate_unique_clip_names(records)
    return records


def extract_common_voice_audio_slice(
    archive: Path, clip_names: Iterable[str], destination: Path
) -> dict[str, Path]:
    """Extract selected MP3s only after complete stream and layout validation."""

    requested_names = {_safe_clip_name(value) for value in clip_names}
    if not requested_names:
        raise CommonVoiceIngestionError("No Common Voice clip paths were requested for extraction.")
    if destination.exists():
        raise CommonVoiceIngestionError(
            f"Refusing to overwrite Common Voice extraction destination: {destination}"
        )
    if not destination.parent.is_dir():
        raise CommonVoiceIngestionError(
            f"Common Voice extraction parent does not exist: {destination.parent}"
        )
    _validate_archive_size(archive)

    extracted: dict[str, Path] = {}
    seen_audio_names: set[str] = set()
    expected_metadata = {
        f"{COMMON_VOICE_RU_V24_DIRECTORY}/{filename}"
        for filename in COMMON_VOICE_RU_V24_METADATA_FILENAMES
    }
    seen_metadata: set[str] = set()
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-common-voice-", dir=destination.parent
        ) as stage_dir:
            stage = Path(stage_dir)
            with tarfile.open(archive, mode="r|gz") as tar:
                for member in tar:
                    if not _is_expected_member(member):
                        raise CommonVoiceIngestionError(
                            f"Unexpected Common Voice archive member: {member.name!r}."
                        )
                    if member.name in expected_metadata:
                        seen_metadata.add(member.name)
                    clip_name = _member_clip_name(member)
                    if clip_name is None:
                        continue
                    if clip_name in seen_audio_names:
                        raise CommonVoiceIngestionError(
                            f"Duplicate Common Voice audio member: {clip_name!r}."
                        )
                    seen_audio_names.add(clip_name)
                    if clip_name not in requested_names:
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        raise CommonVoiceIngestionError(
                            f"Cannot read Common Voice audio member: {member.name!r}."
                        )
                    output_path = stage / "clips" / clip_name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with source, output_path.open("xb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                    extracted[clip_name] = output_path
            missing = sorted(requested_names.difference(extracted))
            if not seen_audio_names:
                raise CommonVoiceIngestionError(
                    "Common Voice archive contains no MP3 audio members."
                )
            if seen_metadata != expected_metadata:
                raise CommonVoiceIngestionError(
                    "Common Voice archive does not contain exactly the expected metadata files."
                )
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "..." if len(missing) > 5 else ""
                raise CommonVoiceIngestionError(
                    f"Requested Common Voice audio is absent from archive: {preview}{suffix}"
                )
            stage.replace(destination)
    except (tarfile.TarError, OSError) as error:
        raise CommonVoiceIngestionError(
            f"Common Voice extraction failed safely: {error}"
        ) from error
    return {clip_name: destination / "clips" / clip_name for clip_name in extracted}


def inspect_extracted_common_voice_audio(path: Path) -> tuple[float, int]:
    """Read duration and sample rate of an extracted MP3 for the raw manifest."""

    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise CommonVoiceIngestionError(
            f"Cannot inspect extracted Common Voice audio: {path}"
        ) from error
    duration = float(info.duration)
    sample_rate = int(info.samplerate)
    if duration <= 0 or sample_rate <= 0:
        raise CommonVoiceIngestionError(f"Invalid Common Voice MP3 properties for {path}.")
    return duration, sample_rate


def common_voice_manifest_rows(
    records: Iterable[CommonVoiceRecord],
    assets: Mapping[str, ExtractedCommonVoiceAsset],
    created_at: str | None = None,
) -> list[ManifestRow]:
    """Create Common Voice bona-fide rows before leakage-safe local split assignment."""

    timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows: list[ManifestRow] = []
    for record in records:
        asset = assets.get(record.clip_name)
        if asset is None:
            raise CommonVoiceIngestionError(
                f"No extracted asset for Common Voice clip {record.clip_name!r}."
            )
        client_group = f"{COMMON_VOICE_RU_V24_SOURCE_ID}:client:{record.client_id}"
        rows.append(
            ManifestRow(
                sample_id=(
                    f"{COMMON_VOICE_RU_V24_SOURCE_ID}:{Path(record.clip_name).stem}"
                ),
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split=COMMON_VOICE_RU_V24_SOURCE_SPLITS[record.split],
                label="bonafide",
                language="ru",
                code_switch="unknown",
                parent_group_id=client_group,
                source_name=COMMON_VOICE_RU_V24_SOURCE_ID,
                source_license=COMMON_VOICE_RU_V24_SOURCE_LICENSE,
                rights_basis=COMMON_VOICE_RU_V24_RIGHTS_BASIS,
                speaker_pseudo_id=client_group,
                text_id=record.sentence_id,
                text_hash=hashlib.sha256(record.sentence.encode("utf-8")).hexdigest(),
                duration_s=asset.duration_s,
                generator_family="",
                generator_name="",
                generator_version="",
                voice_id="",
                clone_consent_id="",
                device=COMMON_VOICE_UNKNOWN,
                capture_route=COMMON_VOICE_RU_V24_CAPTURE_ROUTE,
                original_sr=asset.original_sr,
                codec="mp3",
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows
