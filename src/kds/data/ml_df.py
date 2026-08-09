"""Safe OOD intake primitives for the Italian subset of ML-DF v1."""

from __future__ import annotations

import csv
import hashlib
import io
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TypeVar

import py7zr
import soundfile as sf  # type: ignore[import-untyped]
from py7zr.exceptions import CrcError, DecompressionBombError

from kds.data.manifest import ManifestRow

ML_DF_IT_SOURCE_ID = "ml_df_it_v1"
ML_DF_IT_ARCHIVE_NAME = "dataset_IT.7z"
ML_DF_IT_ARCHIVE_EXPECTED_SIZE_BYTES = 1_485_098_719
ML_DF_IT_ARCHIVE_EXPECTED_MD5 = "c3ce93f9566605e0a5ad2e3cda099d7d"
ML_DF_IT_ARCHIVE_EXPECTED_UNCOMPRESSED_BYTES = 2_290_807_586
ML_DF_MAX_OOD_SLICE_UNCOMPRESSED_BYTES = 2 * 1024**3
ML_DF_METADATA_ARCHIVE_NAME = "metadata.zip"
ML_DF_METADATA_EXPECTED_MD5 = "25cc69e8d9234a22c1f38222e0bfdebf"
ML_DF_METADATA_FILENAMES = frozenset(
    {"metadata_DE.csv", "metadata_EN.csv", "metadata_ES.csv", "metadata_FR.csv", "metadata_IT.csv"}
)
ML_DF_IT_METADATA_NAME = "metadata_IT.csv"
ML_DF_IT_DIRECTORY = "dataset_IT"
ML_DF_IT_ARCHIVE_NON_AUDIO_MEMBERS = frozenset(
    {"dataset_IT", "metadata", "metadata/metadata_IT.csv"}
)
ML_DF_IT_SOURCE_LICENSE = "CC-BY-4.0"
ML_DF_IT_SOURCE_URL = "https://zenodo.org/records/17098081"
ML_DF_IT_RIGHTS_BASIS = "ML-DF v1 CC-BY-4.0; source MLS is CC-BY-4.0"
ML_DF_IT_GENERATOR_FAMILIES = {
    "VITS": "tts",
    "ZMM-TTS": "tts",
    "LVC-VC": "vc",
    "DDDM-VC": "vc",
}
ML_DF_UNKNOWN = "unknown"


class MlDfIngestionError(ValueError):
    """Raised when ML-DF assets cannot safely become an OOD manifest."""


@dataclass(frozen=True, slots=True)
class MlDfRecord:
    relative_path: str
    tool: str
    gender: str
    source_group: str
    target_speaker_id: str

    @property
    def label(self) -> str:
        return "bonafide" if self.tool == "bonafide" else "spoof"

    @property
    def content_pseudo_id(self) -> str:
        stem_parts = Path(self.relative_path).stem.split("_")
        if self.label == "bonafide":
            return "_".join(stem_parts[:-1])
        return "_".join(stem_parts[:5])


@dataclass(frozen=True, slots=True)
class ExtractedMlDfAsset:
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int


@dataclass(frozen=True, slots=True)
class MlDfArchiveReport:
    archive: Path
    audio_files: int


RecordT = TypeVar("RecordT", bound=MlDfRecord)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_audio_path(value: str) -> str:
    relative_path = value.strip()
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != ML_DF_IT_DIRECTORY
        or path.suffix.lower() != ".wav"
        or "\\" in relative_path
        or ".." in path.parts
    ):
        raise MlDfIngestionError(f"Invalid ML-DF audio path: {value!r}.")
    return relative_path


def _validate_archive_size(archive: Path) -> None:
    if not archive.is_file():
        raise MlDfIngestionError(f"ML-DF archive does not exist: {archive}")
    actual_size = archive.stat().st_size
    if actual_size != ML_DF_IT_ARCHIVE_EXPECTED_SIZE_BYTES:
        raise MlDfIngestionError(
            "ML-DF archive size mismatch: "
            f"expected {ML_DF_IT_ARCHIVE_EXPECTED_SIZE_BYTES} bytes, got {actual_size}."
        )
    actual_md5 = _md5_file(archive)
    if actual_md5 != ML_DF_IT_ARCHIVE_EXPECTED_MD5:
        raise MlDfIngestionError(
            "ML-DF archive MD5 mismatch: "
            f"expected {ML_DF_IT_ARCHIVE_EXPECTED_MD5}, got {actual_md5}."
        )


def load_ml_df_it_metadata(metadata_archive: Path) -> list[MlDfRecord]:
    """Load the official metadata ZIP after validating its MD5 and exact member whitelist."""

    if not metadata_archive.is_file():
        raise MlDfIngestionError(f"ML-DF metadata archive does not exist: {metadata_archive}")
    actual_md5 = _md5_file(metadata_archive)
    if actual_md5 != ML_DF_METADATA_EXPECTED_MD5:
        raise MlDfIngestionError(
            "ML-DF metadata MD5 mismatch: "
            f"expected {ML_DF_METADATA_EXPECTED_MD5}, got {actual_md5}."
        )
    try:
        with zipfile.ZipFile(metadata_archive) as archive:
            member_names = {info.filename for info in archive.infolist() if not info.is_dir()}
            if member_names != ML_DF_METADATA_FILENAMES:
                raise MlDfIngestionError("ML-DF metadata ZIP has an unexpected member layout.")
            content = archive.read(ML_DF_IT_METADATA_NAME).decode("utf-8-sig")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise MlDfIngestionError(f"ML-DF metadata ZIP cannot be read safely: {error}") from error
    reader = csv.DictReader(io.StringIO(content), delimiter=" ", skipinitialspace=True)
    required_columns = {"wav_file", "tool", "gender", "group", "speaker"}
    if reader.fieldnames is None or set(reader.fieldnames) != required_columns:
        raise MlDfIngestionError("ML-DF Italian metadata has an unexpected header.")
    records: list[MlDfRecord] = []
    seen_paths: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        try:
            relative_path = _safe_audio_path(row.get("wav_file") or "")
        except MlDfIngestionError as error:
            raise MlDfIngestionError(f"metadata_IT.csv:{row_number}: {error}") from error
        tool = (row.get("tool") or "").strip()
        gender = (row.get("gender") or "").strip()
        source_group = (row.get("group") or "").strip()
        target_speaker_id = (row.get("speaker") or "").strip()
        if tool not in {"bonafide", *ML_DF_IT_GENERATOR_FAMILIES}:
            raise MlDfIngestionError(f"metadata_IT.csv:{row_number}: unsupported tool {tool!r}.")
        if gender not in {"F", "M"} or source_group != "train" or not target_speaker_id:
            raise MlDfIngestionError(f"metadata_IT.csv:{row_number}: invalid provenance fields.")
        if relative_path in seen_paths:
            raise MlDfIngestionError(
                f"metadata_IT.csv:{row_number}: duplicate audio path {relative_path!r}."
            )
        seen_paths.add(relative_path)
        records.append(
            MlDfRecord(relative_path, tool, gender, source_group, target_speaker_id)
        )
    if not records:
        raise MlDfIngestionError("ML-DF Italian metadata contains no records.")
    return records


def inspect_ml_df_archive(archive: Path, records: Iterable[MlDfRecord]) -> MlDfArchiveReport:
    """Validate archive size, CRC, path whitelist, and one-to-one metadata membership."""

    _validate_archive_size(archive)
    expected_paths = {record.relative_path for record in records}
    if not expected_paths:
        raise MlDfIngestionError("ML-DF metadata has no expected audio paths.")
    try:
        with py7zr.SevenZipFile(archive, mode="r") as seven_zip:
            info_by_path = {info.filename: info for info in seven_zip.list()}
            audio_paths = {
                path
                for path, info in info_by_path.items()
                if info.is_file and path.endswith(".wav")
            }
            if audio_paths != expected_paths:
                unexpected = sorted(audio_paths.difference(expected_paths))
                missing = sorted(expected_paths.difference(audio_paths))
                detail = (unexpected or missing)[:3]
                raise MlDfIngestionError(
                    "ML-DF archive and metadata paths do not match exactly: " + ", ".join(detail)
                )
            non_audio_members = set(info_by_path).difference(audio_paths)
            if non_audio_members != ML_DF_IT_ARCHIVE_NON_AUDIO_MEMBERS:
                raise MlDfIngestionError("ML-DF archive has an unexpected non-audio member layout.")
            archive_uncompressed_size = sum(
                info.uncompressed for info in info_by_path.values() if info.is_file
            )
            if archive_uncompressed_size != ML_DF_IT_ARCHIVE_EXPECTED_UNCOMPRESSED_BYTES:
                raise MlDfIngestionError(
                    "ML-DF archive uncompressed size does not match the verified release."
                )
            if any(
                not info.is_file or info.is_symlink or _safe_audio_path(path) != path
                for path, info in info_by_path.items()
                if path in audio_paths
            ):
                raise MlDfIngestionError("ML-DF archive contains an unsafe or non-file member.")
            if (
                not info_by_path[ML_DF_IT_DIRECTORY].is_directory
                or not info_by_path["metadata"].is_directory
                or not info_by_path["metadata/metadata_IT.csv"].is_file
                or info_by_path["metadata/metadata_IT.csv"].is_symlink
            ):
                raise MlDfIngestionError("ML-DF archive has an unsafe metadata member layout.")
            crc_result = seven_zip.test()
    except (CrcError, DecompressionBombError, OSError, py7zr.Bad7zFile) as error:
        raise MlDfIngestionError(f"ML-DF archive cannot be read safely: {error}") from error
    if crc_result is False:
        raise MlDfIngestionError("ML-DF archive CRC validation failed.")
    return MlDfArchiveReport(archive=archive, audio_files=len(expected_paths))


def select_ml_df_ood_records(
    records: Iterable[RecordT],
    bonafide_limit: int,
    spoof_limit_per_generator: int,
    seed: str,
) -> list[RecordT]:
    """Choose a balanced OOD subset: bona-fide plus equal samples per fake generator."""

    if bonafide_limit <= 0 or spoof_limit_per_generator <= 0 or not seed:
        raise ValueError("Selection limits and seed must be positive and non-empty.")
    grouped: dict[str, list[RecordT]] = {}
    for record in records:
        grouped.setdefault(record.tool, []).append(record)
    selected: list[RecordT] = []
    expected_tools = ("bonafide", *sorted(ML_DF_IT_GENERATOR_FAMILIES))
    for tool in expected_tools:
        candidates = grouped.get(tool, [])
        limit = bonafide_limit if tool == "bonafide" else spoof_limit_per_generator
        if len(candidates) < limit:
            raise MlDfIngestionError(f"ML-DF has only {len(candidates)} rows for tool {tool!r}.")
        selected.extend(
            sorted(
                candidates,
                key=lambda item: hashlib.sha256(f"{seed}:{item.relative_path}".encode()).digest(),
            )[:limit]
        )
    return selected


def extract_ml_df_audio_slice(
    archive: Path, records: Iterable[MlDfRecord], destination: Path
) -> dict[str, Path]:
    """Extract selected verified WAV paths to staging and publish them atomically."""

    selected_paths = {record.relative_path for record in records}
    if not selected_paths:
        raise MlDfIngestionError("No ML-DF records were requested for extraction.")
    if destination.exists():
        raise MlDfIngestionError(f"Refusing to overwrite ML-DF destination: {destination}")
    if not destination.parent.is_dir():
        raise MlDfIngestionError(f"ML-DF extraction parent does not exist: {destination.parent}")
    _validate_archive_size(archive)
    try:
        with tempfile.TemporaryDirectory(prefix="kds-ml-df-", dir=destination.parent) as stage_dir:
            stage = Path(stage_dir)
            with py7zr.SevenZipFile(
                archive, mode="r", max_extract_size=ML_DF_IT_ARCHIVE_EXPECTED_UNCOMPRESSED_BYTES
            ) as seven_zip:
                info_by_path = {info.filename: info for info in seven_zip.list()}
                if not selected_paths.issubset(info_by_path):
                    raise MlDfIngestionError("ML-DF archive is missing a requested audio asset.")
                archive_uncompressed_size = sum(
                    info.uncompressed for info in info_by_path.values() if info.is_file
                )
                if archive_uncompressed_size != ML_DF_IT_ARCHIVE_EXPECTED_UNCOMPRESSED_BYTES:
                    raise MlDfIngestionError(
                        "ML-DF archive uncompressed size does not match the verified release."
                    )
                selected_size = sum(
                    info_by_path[path].uncompressed for path in selected_paths
                )
                if selected_size > ML_DF_MAX_OOD_SLICE_UNCOMPRESSED_BYTES:
                    raise MlDfIngestionError(
                        "Requested ML-DF OOD slice exceeds the safe extraction limit."
                    )
                seven_zip.extract(path=stage, targets=selected_paths)
            extracted_paths = {
                path.relative_to(stage).as_posix()
                for path in stage.rglob("*.wav")
                if path.is_file()
            }
            if extracted_paths != selected_paths:
                raise MlDfIngestionError(
                    "ML-DF extraction did not produce exactly the requested files."
                )
            stage.replace(destination)
    except (CrcError, DecompressionBombError, OSError, py7zr.Bad7zFile) as error:
        raise MlDfIngestionError(f"ML-DF extraction failed safely: {error}") from error
    return {relative_path: destination / relative_path for relative_path in selected_paths}


def inspect_extracted_ml_df_audio(path: Path) -> tuple[float, int]:
    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise MlDfIngestionError(f"Cannot inspect extracted ML-DF audio: {path}") from error
    duration_s = float(info.duration)
    sample_rate = int(info.samplerate)
    if duration_s <= 0 or sample_rate != 16_000:
        raise MlDfIngestionError(f"Invalid ML-DF WAV properties for {path}.")
    return duration_s, sample_rate


def ml_df_ood_manifest_rows(
    records: Iterable[MlDfRecord],
    assets: Mapping[str, ExtractedMlDfAsset],
    created_at: str | None = None,
) -> list[ManifestRow]:
    """Build an Italian cross-lingual OOD manifest, never a target-language training split."""

    timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows: list[ManifestRow] = []
    for record in records:
        asset = assets.get(record.relative_path)
        if asset is None:
            raise MlDfIngestionError(f"No extracted ML-DF asset for {record.relative_path!r}.")
        target_voice = f"{ML_DF_IT_SOURCE_ID}:target-speaker:{record.target_speaker_id}"
        content_pseudo_id = f"{ML_DF_IT_SOURCE_ID}:content:{record.content_pseudo_id}"
        generator_family = (
            "" if record.label == "bonafide" else ML_DF_IT_GENERATOR_FAMILIES[record.tool]
        )
        rows.append(
            ManifestRow(
                sample_id=f"{ML_DF_IT_SOURCE_ID}:{Path(record.relative_path).stem}",
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split="ood",
                label=record.label,
                language="other",
                code_switch="unknown",
                parent_group_id=target_voice,
                source_name=ML_DF_IT_SOURCE_ID,
                source_license=ML_DF_IT_SOURCE_LICENSE,
                rights_basis=ML_DF_IT_RIGHTS_BASIS,
                speaker_pseudo_id=target_voice,
                text_id=content_pseudo_id,
                text_hash=hashlib.sha256(content_pseudo_id.encode()).hexdigest(),
                duration_s=asset.duration_s,
                generator_family=generator_family,
                generator_name="" if record.label == "bonafide" else record.tool,
                generator_version="" if record.label == "bonafide" else "ML-DF-v1",
                voice_id="" if record.label == "bonafide" else target_voice,
                clone_consent_id="",
                device=ML_DF_UNKNOWN,
                capture_route="synthetic_ml_df",
                original_sr=asset.original_sr,
                codec="wav",
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows
