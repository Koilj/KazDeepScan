"""Safe, research-only intake primitives for the locally verified PyAra v7 archive."""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TypeVar

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.manifest import ManifestRow

PYARA_SOURCE_ID = "pyara_ru_v7"
PYARA_ARCHIVE_NAME = "archive.zip"
PYARA_ARCHIVE_EXPECTED_SIZE_BYTES = 28_092_611_663
PYARA_ARCHIVE_EXPECTED_SHA256 = "dadf5b795adbd6d635e74f4f9662c3e9a425c88bd76f26731f9e6adbad278b91"
PYARA_ARCHIVE_ROOT = "final_dataset"
PYARA_METADATA_MEMBER = f"{PYARA_ARCHIVE_ROOT}/final_dataset.tsv"
PYARA_EXPECTED_AUDIO_FILES = 201_778
PYARA_SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
PYARA_SOURCE_URL = "https://www.kaggle.com/datasets/alep079/pyara/versions/7"
PYARA_RIGHTS_BASIS = (
    "PyAra v7 CC-BY-NC-SA-4.0; owner-authorized personal research; "
    "text-leakage-safe only, no verified speaker identifiers"
)
PYARA_UNKNOWN = "unknown"


class PyAraIngestionError(ValueError):
    """Raised when the PyAra archive cannot safely become a research manifest."""


@dataclass(frozen=True, slots=True)
class PyAraRecord:
    relative_path: str
    label: str
    sentence: str
    algorithm: str
    duration_s: float

    @property
    def record_id(self) -> str:
        return Path(self.relative_path).stem


@dataclass(frozen=True, slots=True)
class ExtractedPyAraAsset:
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int


@dataclass(frozen=True, slots=True)
class PyAraArchiveReport:
    archive: Path
    audio_files: int
    real_files: int
    fake_files: int


RecordT = TypeVar("RecordT", bound=PyAraRecord)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive(archive: Path) -> None:
    if not archive.is_file():
        raise PyAraIngestionError(f"PyAra archive does not exist: {archive}")
    if archive.stat().st_size != PYARA_ARCHIVE_EXPECTED_SIZE_BYTES:
        raise PyAraIngestionError("PyAra archive size does not match the verified artifact.")
    if _sha256_file(archive) != PYARA_ARCHIVE_EXPECTED_SHA256:
        raise PyAraIngestionError("PyAra archive SHA-256 does not match the verified artifact.")


def _safe_relative_audio_path(value: str) -> str:
    path = PurePosixPath(value.strip())
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] not in {"Fake", "Real"}
        or path.suffix.lower() != ".wav"
        or ".." in path.parts
        or "\\" in value
    ):
        raise PyAraIngestionError(f"Invalid PyAra audio path: {value!r}.")
    return path.as_posix()


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _archive_audio_path(info: zipfile.ZipInfo) -> str | None:
    path = PurePosixPath(info.filename)
    if info.is_dir() or _is_symlink(info):
        return None
    if path.parts == (PYARA_ARCHIVE_ROOT, "final_dataset.tsv"):
        return ""
    if len(path.parts) != 3 or path.parts[0] != PYARA_ARCHIVE_ROOT:
        return None
    return _safe_relative_audio_path(PurePosixPath(*path.parts[1:]).as_posix())


def _canonical_sentence(value: str) -> str:
    return " ".join(value.split())


def _load_metadata(archive: zipfile.ZipFile) -> list[PyAraRecord]:
    try:
        with archive.open(PYARA_METADATA_MEMBER) as source:
            reader = csv.DictReader(
                io.TextIOWrapper(source, encoding="utf-8-sig", newline=""), delimiter="\t"
            )
            required = {"path", "sentence", "age", "gender", "fake", "algorithm", "length"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise PyAraIngestionError("PyAra metadata has an unexpected header.")
            records: list[PyAraRecord] = []
            seen_paths: set[str] = set()
            for number, row in enumerate(reader, start=2):
                relative_path = _safe_relative_audio_path(row.get("path") or "")
                sentence = _canonical_sentence(row.get("sentence") or "")
                fake = (row.get("fake") or "").strip()
                algorithm = (row.get("algorithm") or "").strip()
                try:
                    duration_s = float((row.get("length") or "").strip())
                except ValueError as error:
                    raise PyAraIngestionError(
                        f"PyAra metadata row {number}: invalid length."
                    ) from error
                label = "spoof" if fake == "1" else "bonafide" if fake == "0" else ""
                expected_directory = "Fake" if label == "spoof" else "Real"
                if (
                    not sentence
                    or duration_s <= 0
                    or not label
                    or relative_path.split("/", 1)[0] != expected_directory
                    or (label == "bonafide" and algorithm)
                    or (label == "spoof" and not algorithm)
                    or relative_path in seen_paths
                ):
                    raise PyAraIngestionError(f"PyAra metadata row {number}: invalid provenance.")
                seen_paths.add(relative_path)
                records.append(PyAraRecord(relative_path, label, sentence, algorithm, duration_s))
    except (KeyError, OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise PyAraIngestionError(f"PyAra metadata cannot be read safely: {error}") from error
    return records


def inspect_pyara_archive(archive_path: Path) -> tuple[PyAraArchiveReport, list[PyAraRecord]]:
    """Verify checksum, ZIP layout and exact TSV-to-WAV membership without extracting files."""

    _validate_archive(archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise PyAraIngestionError("PyAra ZIP contains duplicate members.")
            audio_paths: set[str] = set()
            metadata_count = 0
            for info in infos:
                archive_path_value = _archive_audio_path(info)
                if archive_path_value is None:
                    raise PyAraIngestionError(f"Unexpected PyAra ZIP member: {info.filename!r}.")
                if not archive_path_value:
                    metadata_count += 1
                    continue
                if info.file_size <= 0 or archive_path_value in audio_paths:
                    raise PyAraIngestionError(f"Invalid PyAra WAV member: {info.filename!r}.")
                audio_paths.add(archive_path_value)
            records = _load_metadata(archive)
    except (OSError, zipfile.BadZipFile) as error:
        raise PyAraIngestionError(f"PyAra ZIP cannot be read safely: {error}") from error
    record_paths = {record.relative_path for record in records}
    if (
        metadata_count != 1
        or len(audio_paths) != PYARA_EXPECTED_AUDIO_FILES
        or record_paths != audio_paths
    ):
        raise PyAraIngestionError("PyAra WAV members and TSV metadata do not match exactly.")
    return (
        PyAraArchiveReport(
            archive_path,
            len(audio_paths),
            sum(record.label == "bonafide" for record in records),
            sum(record.label == "spoof" for record in records),
        ),
        records,
    )


def select_pyara_records(
    records: Iterable[RecordT], real_limit: int, fake_limit_per_algorithm: int, seed: str
) -> list[RecordT]:
    """Select a deterministic class-balanced raw slice with every fake algorithm represented."""

    if real_limit <= 0 or fake_limit_per_algorithm <= 0 or not seed:
        raise ValueError("Selection limits and seed must be positive and non-empty.")
    grouped: dict[str, list[RecordT]] = {}
    for record in records:
        key = "real" if record.label == "bonafide" else record.algorithm
        grouped.setdefault(key, []).append(record)
    selected: list[RecordT] = []
    for key, candidates in sorted(grouped.items()):
        limit = real_limit if key == "real" else fake_limit_per_algorithm
        if len(candidates) < limit:
            raise PyAraIngestionError(f"PyAra has only {len(candidates)} rows for {key!r}.")
        selected.extend(
            sorted(
                candidates,
                key=lambda item: hashlib.sha256(f"{seed}:{item.relative_path}".encode()).digest(),
            )[:limit]
        )
    return selected


def extract_pyara_audio_slice(
    archive_path: Path, records: Iterable[PyAraRecord], destination: Path
) -> dict[str, Path]:
    """Atomically extract only selected WAV files, never using ZipFile.extract/extractall."""

    selected_paths = {record.relative_path for record in records}
    if not selected_paths or destination.exists() or not destination.parent.is_dir():
        raise PyAraIngestionError(f"Unsafe PyAra extraction destination: {destination}")
    _validate_archive(archive_path)
    try:
        with tempfile.TemporaryDirectory(prefix="kds-pyara-", dir=destination.parent) as stage_dir:
            stage = Path(stage_dir)
            extracted: set[str] = set()
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    relative_path = _archive_audio_path(info)
                    if relative_path not in selected_paths:
                        continue
                    output_path = stage / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, output_path.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    extracted.add(relative_path)
            if extracted != selected_paths:
                raise PyAraIngestionError("PyAra extraction did not produce every selected WAV.")
            stage.replace(destination)
    except (OSError, zipfile.BadZipFile) as error:
        raise PyAraIngestionError(f"PyAra extraction failed safely: {error}") from error
    return {relative_path: destination / relative_path for relative_path in selected_paths}


def inspect_extracted_pyara_audio(path: Path) -> tuple[float, int]:
    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise PyAraIngestionError(f"Cannot inspect PyAra audio: {path}") from error
    if info.duration <= 0 or info.samplerate <= 0:
        raise PyAraIngestionError(f"Invalid PyAra audio properties: {path}")
    return float(info.duration), int(info.samplerate)


def pyara_manifest_rows(
    records: Iterable[PyAraRecord],
    assets: Mapping[str, ExtractedPyAraAsset],
    created_at: str | None = None,
) -> list[ManifestRow]:
    """Build research rows with text-safe groups; source does not supply speaker identities."""

    timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows: list[ManifestRow] = []
    for record in records:
        asset = assets.get(record.relative_path)
        if asset is None:
            raise PyAraIngestionError(f"Missing PyAra asset: {record.relative_path!r}.")
        record_key = f"{PYARA_SOURCE_ID}:source-record:{record.record_id}"
        text_hash = hashlib.sha256(record.sentence.encode()).hexdigest()
        rows.append(
            ManifestRow(
                sample_id=f"{PYARA_SOURCE_ID}:{record.record_id}",
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split="train",
                label=record.label,
                language="ru",
                code_switch="unknown",
                parent_group_id=record_key,
                source_name=PYARA_SOURCE_ID,
                source_license=PYARA_SOURCE_LICENSE,
                rights_basis=PYARA_RIGHTS_BASIS,
                speaker_pseudo_id=record_key,
                text_id=f"{PYARA_SOURCE_ID}:text:{text_hash}",
                text_hash=text_hash,
                duration_s=asset.duration_s,
                generator_family="unspecified_synthesis" if record.label == "spoof" else "",
                generator_name=record.algorithm,
                generator_version="PyAra-v7" if record.label == "spoof" else "",
                voice_id=f"{PYARA_SOURCE_ID}:voice:{PYARA_UNKNOWN}"
                if record.label == "spoof"
                else "",
                clone_consent_id="",
                device=PYARA_UNKNOWN,
                capture_route="pyara_source_audio",
                original_sr=asset.original_sr,
                codec="wav",
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows
