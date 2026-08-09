"""Safe OOD-only intake primitives for one verified Russian RuASD shard."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TypeVar

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.manifest import ManifestRow

RUASD_SOURCE_ID = "ruasd_ru_v1_shard000000"
RUASD_ARCHIVE_NAME = "ruasd-000000.tar"
RUASD_ARCHIVE_EXPECTED_SIZE_BYTES = 999_813_120
RUASD_ARCHIVE_EXPECTED_SHA256 = "956efb0e1281ada0dcee6f2ed9498c454552be88b3e9784e52e70c3ef4dfcd67"
RUASD_SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
RUASD_SOURCE_URL = "https://huggingface.co/datasets/lab260/RuASD"
RUASD_RIGHTS_BASIS = "RuASD CC-BY-NC-SA-4.0; personal research; OOD only"
RUASD_CAPTURE_ROUTE = "synthetic_tts_ruasd"
RUASD_UNKNOWN = "unknown"


class RuAsdIngestionError(ValueError):
    """Raised when a RuASD shard cannot safely become an OOD manifest."""


@dataclass(frozen=True, slots=True)
class RuAsdRecord:
    sample_id: str
    generator_name: str


@dataclass(frozen=True, slots=True)
class ExtractedRuAsdAsset:
    sample_id: str
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int


RecordT = TypeVar("RecordT", bound=RuAsdRecord)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive(archive: Path) -> None:
    if not archive.is_file():
        raise RuAsdIngestionError(f"RuASD archive does not exist: {archive}")
    if archive.stat().st_size != RUASD_ARCHIVE_EXPECTED_SIZE_BYTES:
        raise RuAsdIngestionError("RuASD archive size does not match the verified shard.")
    actual_sha256 = _sha256_file(archive)
    if actual_sha256 != RUASD_ARCHIVE_EXPECTED_SHA256:
        raise RuAsdIngestionError("RuASD archive SHA-256 does not match the verified shard.")


def _safe_member_name(name: str) -> tuple[str, str]:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name != name
        or "\\" in name
        or path.suffix.lower() not in {".json", ".wav"}
    ):
        raise RuAsdIngestionError(f"Unexpected RuASD member path: {name!r}.")
    return path.stem, path.suffix.lower()


def _read_json_record(member: tarfile.TarInfo, archive: tarfile.TarFile) -> RuAsdRecord:
    if member.size <= 0 or member.size > 1_000_000:
        raise RuAsdIngestionError(f"Unsafe RuASD metadata size for {member.name!r}.")
    source = archive.extractfile(member)
    if source is None:
        raise RuAsdIngestionError(f"Cannot read RuASD metadata: {member.name!r}.")
    try:
        with source:
            raw = json.loads(source.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuAsdIngestionError(f"Invalid RuASD metadata: {member.name!r}.") from error
    if not isinstance(raw, Mapping):
        raise RuAsdIngestionError(f"RuASD metadata is not an object: {member.name!r}.")
    sample_id = str(raw.get("sample_id") or "").strip()
    generator_name = str(raw.get("subset") or "").strip()
    filename = str(raw.get("filename") or "").strip()
    audio_relpath = str(raw.get("audio_relpath") or "").strip()
    audio_path = PurePosixPath(audio_relpath)
    if (
        sample_id != Path(member.name).stem
        or raw.get("label") != "fake"
        or raw.get("group") != "raw"
        or raw.get("source_type") != "tts"
        or not generator_name
        or PurePosixPath(filename).name != filename
        or audio_path.is_absolute()
        or ".." in audio_path.parts
        or "\\" in audio_relpath
        or audio_path.name != filename
        or not filename.endswith(".wav")
    ):
        raise RuAsdIngestionError(f"Unexpected RuASD fake metadata: {member.name!r}.")
    return RuAsdRecord(sample_id=sample_id, generator_name=generator_name)


def load_ruasd_fake_records(archive_path: Path) -> list[RuAsdRecord]:
    """Verify an exact archive and return all paired raw TTS fake records."""

    _validate_archive(archive_path)
    audio_ids: set[str] = set()
    records: dict[str, RuAsdRecord] = {}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or member.issym() or member.islnk():
                    raise RuAsdIngestionError(f"Unsafe RuASD archive member: {member.name!r}.")
                sample_id, suffix = _safe_member_name(member.name)
                if suffix == ".wav":
                    if member.size <= 0 or sample_id in audio_ids:
                        raise RuAsdIngestionError(f"Invalid RuASD audio member: {member.name!r}.")
                    audio_ids.add(sample_id)
                else:
                    if sample_id in records:
                        raise RuAsdIngestionError(
                            f"Duplicate RuASD metadata member: {member.name!r}."
                        )
                    records[sample_id] = _read_json_record(member, archive)
    except (OSError, tarfile.TarError) as error:
        raise RuAsdIngestionError(f"RuASD archive cannot be read safely: {error}") from error
    if not records or set(records) != audio_ids:
        raise RuAsdIngestionError("RuASD archive does not contain one JSON/WAV pair per record.")
    return list(records.values())


def select_ruasd_ood_records(
    records: Iterable[RecordT], limit_per_generator: int, seed: str
) -> list[RecordT]:
    """Select the same deterministic limit for every available fake generator."""

    if limit_per_generator <= 0 or not seed:
        raise ValueError("limit_per_generator and seed must be positive and non-empty.")
    grouped: dict[str, list[RecordT]] = {}
    for record in records:
        grouped.setdefault(record.generator_name, []).append(record)
    selected: list[RecordT] = []
    for generator_name, candidates in sorted(grouped.items()):
        if len(candidates) < limit_per_generator:
            raise RuAsdIngestionError(
                f"RuASD has only {len(candidates)} rows for generator {generator_name!r}."
            )
        selected.extend(
            sorted(
                candidates,
                key=lambda item: hashlib.sha256(f"{seed}:{item.sample_id}".encode()).digest(),
            )[:limit_per_generator]
        )
    return selected


def extract_ruasd_ood_slice(
    archive_path: Path, records: Iterable[RuAsdRecord], destination: Path
) -> dict[str, Path]:
    """Atomically extract exactly the requested direct-file WAV members."""

    selected_ids = {record.sample_id for record in records}
    if not selected_ids:
        raise RuAsdIngestionError("No RuASD records were selected for extraction.")
    if destination.exists() or not destination.parent.is_dir():
        raise RuAsdIngestionError(f"Unsafe RuASD destination: {destination}")
    _validate_archive(archive_path)
    try:
        with tempfile.TemporaryDirectory(prefix="kds-ruasd-", dir=destination.parent) as stage_dir:
            stage = Path(stage_dir)
            extracted: set[str] = set()
            with tarfile.open(archive_path, mode="r:") as archive:
                for member in archive:
                    if not member.isfile() or member.issym() or member.islnk():
                        raise RuAsdIngestionError(f"Unsafe RuASD archive member: {member.name!r}.")
                    sample_id, suffix = _safe_member_name(member.name)
                    if suffix != ".wav" or sample_id not in selected_ids:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuAsdIngestionError(f"Cannot read RuASD audio: {member.name!r}.")
                    with source, (stage / member.name).open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    extracted.add(sample_id)
            if extracted != selected_ids:
                raise RuAsdIngestionError("RuASD extraction did not produce every requested WAV.")
            stage.replace(destination)
    except (OSError, tarfile.TarError) as error:
        raise RuAsdIngestionError(f"RuASD extraction failed safely: {error}") from error
    return {sample_id: destination / f"{sample_id}.wav" for sample_id in selected_ids}


def inspect_extracted_ruasd_audio(path: Path) -> tuple[float, int]:
    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise RuAsdIngestionError(f"Cannot inspect RuASD audio: {path}") from error
    if info.duration <= 0 or info.samplerate <= 0:
        raise RuAsdIngestionError(f"Invalid RuASD audio properties: {path}")
    return float(info.duration), int(info.samplerate)


def ruasd_ood_manifest_rows(
    records: Iterable[RuAsdRecord],
    assets: Mapping[str, ExtractedRuAsdAsset],
    created_at: str | None = None,
) -> list[ManifestRow]:
    """Build fake-only Russian OOD rows; they are never eligible for training splits."""

    timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows: list[ManifestRow] = []
    for record in records:
        asset = assets.get(record.sample_id)
        if asset is None:
            raise RuAsdIngestionError(f"Missing RuASD asset: {record.sample_id!r}.")
        content_id = f"{RUASD_SOURCE_ID}:content:{record.sample_id}"
        rows.append(
            ManifestRow(
                sample_id=f"{RUASD_SOURCE_ID}:{record.sample_id}",
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split="ood",
                label="spoof",
                language="ru",
                code_switch="unknown",
                parent_group_id=content_id,
                source_name=RUASD_SOURCE_ID,
                source_license=RUASD_SOURCE_LICENSE,
                rights_basis=RUASD_RIGHTS_BASIS,
                speaker_pseudo_id=f"{RUASD_SOURCE_ID}:speaker:{RUASD_UNKNOWN}",
                text_id=content_id,
                text_hash=hashlib.sha256(content_id.encode()).hexdigest(),
                duration_s=asset.duration_s,
                generator_family="tts",
                generator_name=record.generator_name,
                generator_version="unspecified_by_source",
                voice_id=f"{RUASD_SOURCE_ID}:voice:{RUASD_UNKNOWN}",
                clone_consent_id="",
                device=RUASD_UNKNOWN,
                capture_route=RUASD_CAPTURE_ROUTE,
                original_sr=asset.original_sr,
                codec="wav",
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows
