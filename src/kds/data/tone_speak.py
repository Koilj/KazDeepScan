"""Streaming integrity audit for the pinned ToneSpeak Russian TTS release.

The audit establishes only that the downloaded Parquet release matches the published
Hugging Face revision and has the dataset-card-described row structure.  It neither
creates a model manifest nor upgrades the source to product/final provenance.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow

TONE_SPEAK_SOURCE_ID = "tone_speak_ru_v1"
TONE_SPEAK_DATASET = "Vikhrmodels/ToneSpeak"
TONE_SPEAK_REVISION = "d40f94cd5c7dcf756a8c59a1c465b834220bec56"
TONE_SPEAK_EXPECTED_COLUMNS = ("audio", "text", "text_description", "voice_name")
TONE_SPEAK_EXPECTED_SAMPLE_RATE = 24_000
TONE_SPEAK_EXPECTED_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}
)


class ToneSpeakAuditError(ValueError):
    """Raised when a ToneSpeak release artifact or record is unexpected."""


@dataclass(frozen=True, slots=True)
class ToneSpeakExpectedArtifact:
    relative_path: str
    split: str | None
    expected_rows: int | None
    size_bytes: int
    sha256: str


TONE_SPEAK_EXPECTED_ARTIFACTS = (
    ToneSpeakExpectedArtifact(
        ".gitattributes",
        None,
        None,
        2_461,
        "e7a120ab07b1bc5b486be249e9fc6c83d59448d0093e1dfebe95d1566d9cafc0",
    ),
    ToneSpeakExpectedArtifact(
        "README.md",
        None,
        None,
        3_741,
        "da0d2a6fcbe4b84edc69a5fd44985ec94991f4e80c5c69d20317a25651517df6",
    ),
    ToneSpeakExpectedArtifact(
        "data/train-00000-of-00004.parquet",
        "train",
        1_575,
        380_768_912,
        "7196f1aae6aa27da08813d2161a94e3af146669b0883bbfef2e404b6108b9f89",
    ),
    ToneSpeakExpectedArtifact(
        "data/train-00001-of-00004.parquet",
        "train",
        1_575,
        378_847_350,
        "1d79738cefae26918e02398ce0e6cd2d86de5fd8e67586f9a6c534a6d60b9987",
    ),
    ToneSpeakExpectedArtifact(
        "data/train-00002-of-00004.parquet",
        "train",
        1_574,
        378_720_674,
        "ebc2236a6fd3628b5710e360ad4469b3a0a9c3f95e7dbb169bc659187afef337",
    ),
    ToneSpeakExpectedArtifact(
        "data/train-00003-of-00004.parquet",
        "train",
        1_574,
        380_371_736,
        "dc03b849110eabdcdc5226b79ab4106ca436bf65e0ec80c7c0ff7b541c85b208",
    ),
    ToneSpeakExpectedArtifact(
        "data/validation-00000-of-00001.parquet",
        "validation",
        700,
        167_675_791,
        "25da50ec9165c208330801c758ac35ce054bcefecba2be63348159fa585ee54b",
    ),
)


@dataclass(frozen=True, slots=True)
class ToneSpeakArtifactReceipt:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ToneSpeakRecord:
    """One embedded ToneSpeak row after the full release has passed its audit."""

    source_split: str
    parquet_path: str
    embedded_path: str
    text: str
    text_hash: str
    voice_name: str


@dataclass(frozen=True, slots=True)
class ToneSpeakExtractedAsset:
    embedded_path: str
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int
    codec: str


@dataclass(frozen=True, slots=True)
class ToneSpeakAudit:
    source_id: str
    dataset: str
    revision: str
    artifacts: tuple[ToneSpeakArtifactReceipt, ...]
    artifact_total_bytes: int
    rows_by_split: dict[str, int]
    voice_counts_by_split: dict[str, dict[str, int]]
    audio_records: int
    audio_total_bytes: int
    audio_total_duration_s: float
    audio_min_duration_s: float
    audio_max_duration_s: float
    audio_sample_rates: tuple[int, ...]
    unique_audio_paths: int
    duplicate_audio_payloads: int
    unique_normalized_texts: int
    duplicate_normalized_texts: int
    cross_split_normalized_texts: int
    records_without_cyrillic: int


def _load_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as error:
        raise ToneSpeakAuditError(
            "ToneSpeak Parquet audit needs pyarrow>=18,<23 in the active environment."
        ) from error
    return parquet


def _normalized_text_hash(value: str) -> str:
    return hashlib.sha256(" ".join(value.split()).encode("utf-8")).hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or len(path.parts) != 1
        or path.suffix.lower() != ".mp3"
    ):
        raise ToneSpeakAuditError(f"Unsafe or unexpected embedded audio path: {value!r}.")
    return path


def _validate_release_tree(root: Path) -> tuple[ToneSpeakArtifactReceipt, ...]:
    if not root.is_dir():
        raise ToneSpeakAuditError(f"ToneSpeak artifact root does not exist: {root}")
    expected_by_path = {item.relative_path: item for item in TONE_SPEAK_EXPECTED_ARTIFACTS}
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        if path.is_symlink():
            raise ToneSpeakAuditError(f"ToneSpeak artifact tree has a symlink: {relative!r}.")
        if path.is_file():
            actual_paths.add(relative)
        elif not path.is_dir():
            raise ToneSpeakAuditError(
                f"ToneSpeak artifact tree has an unexpected entry: {relative!r}."
            )
    if actual_paths != set(expected_by_path):
        missing = sorted(set(expected_by_path).difference(actual_paths))
        unexpected = sorted(actual_paths.difference(expected_by_path))
        raise ToneSpeakAuditError(
            "ToneSpeak artifact tree differs from the pinned release: "
            f"missing={missing}, unexpected={unexpected}."
        )
    receipts: list[ToneSpeakArtifactReceipt] = []
    for relative_path, expected in sorted(expected_by_path.items()):
        path = root / relative_path
        size_bytes = path.stat().st_size
        if size_bytes != expected.size_bytes:
            raise ToneSpeakAuditError(
                f"ToneSpeak size mismatch for {relative_path}: expected {expected.size_bytes}, "
                f"got {size_bytes}."
            )
        digest = sha256_file(path)
        if digest != expected.sha256:
            raise ToneSpeakAuditError(
                f"ToneSpeak SHA-256 mismatch for {relative_path}: expected {expected.sha256}, "
                f"got {digest}."
            )
        receipts.append(ToneSpeakArtifactReceipt(relative_path, size_bytes, digest))
    return tuple(receipts)


def _expected_audio_path(path: PurePosixPath, voice_name: str) -> None:
    stem = path.stem
    if not stem.endswith(f"_{voice_name}") or not stem.removesuffix(f"_{voice_name}").isdecimal():
        raise ToneSpeakAuditError(
            f"Embedded MP3 path {path.as_posix()!r} does not encode voice {voice_name!r}."
        )


def _schema_metadata_sample_rate(schema: Any) -> int:
    metadata = schema.metadata or {}
    encoded = metadata.get(b"huggingface")
    if encoded is None:
        raise ToneSpeakAuditError("ToneSpeak Parquet schema has no Hugging Face feature metadata.")
    try:
        payload = json.loads(encoded.decode("utf-8"))
        sample_rate = payload["info"]["features"]["audio"]["sampling_rate"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ToneSpeakAuditError(
            "ToneSpeak Parquet feature metadata has an unexpected schema."
        ) from error
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ToneSpeakAuditError(
            "ToneSpeak audio sampling rate metadata must be a positive integer."
        )
    return sample_rate


def _validate_schema(parquet_file: Any, *, relative_path: str) -> None:
    schema = parquet_file.schema_arrow
    if tuple(schema.names) != TONE_SPEAK_EXPECTED_COLUMNS:
        raise ToneSpeakAuditError(
            f"ToneSpeak schema columns differ in {relative_path}: {tuple(schema.names)!r}."
        )
    expected_types = (
        "struct<bytes: binary, path: string>",
        "string",
        "string",
        "string",
    )
    actual_types = tuple(str(field.type) for field in schema)
    if actual_types != expected_types:
        raise ToneSpeakAuditError(
            f"ToneSpeak schema types differ in {relative_path}: {actual_types!r}."
        )
    metadata_sample_rate = _schema_metadata_sample_rate(schema)
    if metadata_sample_rate != TONE_SPEAK_EXPECTED_SAMPLE_RATE:
        raise ToneSpeakAuditError(
            "ToneSpeak feature metadata sample rate differs: "
            f"expected {TONE_SPEAK_EXPECTED_SAMPLE_RATE}, got {metadata_sample_rate}."
        )


def _iter_rows(parquet_file: Any) -> Iterable[dict[str, object]]:
    for batch in parquet_file.iter_batches(batch_size=32):
        yield from batch.to_pylist()


def _inspect_audio(audio_bytes: bytes, *, context: str) -> tuple[int, int]:
    try:
        with sf.SoundFile(BytesIO(audio_bytes)) as audio_file:
            if audio_file.format != "MP3":
                raise ToneSpeakAuditError(
                    f"ToneSpeak {context}: embedded audio format is "
                    f"{audio_file.format!r}, not MP3."
                )
            return audio_file.samplerate, audio_file.frames
    except RuntimeError as error:
        raise ToneSpeakAuditError(
            f"ToneSpeak {context}: embedded MP3 cannot be opened."
        ) from error


def audit_tone_speak_release(root: Path) -> ToneSpeakAudit:
    """Verify every pinned file and all embedded MP3 records without extraction."""

    receipts = _validate_release_tree(root)
    parquet = _load_parquet()
    rows_by_split: Counter[str] = Counter()
    voice_counts: dict[str, Counter[str]] = {}
    audio_paths: set[str] = set()
    audio_hashes: set[str] = set()
    text_hashes: dict[str, set[str]] = {}
    audio_total_bytes = 0
    audio_total_duration_s = 0.0
    audio_min_duration_s = float("inf")
    audio_max_duration_s = 0.0
    duplicate_audio_payloads = 0
    duplicate_normalized_texts = 0
    records_without_cyrillic = 0
    audio_sample_rates: set[int] = set()
    expected_by_path = {item.relative_path: item for item in TONE_SPEAK_EXPECTED_ARTIFACTS}

    for relative_path, expected in sorted(expected_by_path.items()):
        if expected.split is None:
            continue
        parquet_file = parquet.ParquetFile(root / relative_path)
        _validate_schema(parquet_file, relative_path=relative_path)
        row_count = parquet_file.metadata.num_rows
        if row_count != expected.expected_rows:
            raise ToneSpeakAuditError(
                f"ToneSpeak row count mismatch for {relative_path}: "
                f"expected {expected.expected_rows}, "
                f"got {row_count}."
            )
        split_voice_counts = voice_counts.setdefault(expected.split, Counter())
        for row_number, row in enumerate(_iter_rows(parquet_file), start=1):
            context = f"{relative_path}:{row_number}"
            audio = row["audio"]
            text = row["text"]
            description = row["text_description"]
            voice_name = row["voice_name"]
            if not isinstance(audio, dict):
                raise ToneSpeakAuditError(f"ToneSpeak {context}: audio must be a struct.")
            audio_bytes = audio.get("bytes")
            audio_path = audio.get("path")
            if not isinstance(audio_bytes, bytes) or not audio_bytes:
                raise ToneSpeakAuditError(f"ToneSpeak {context}: audio bytes are missing.")
            if not isinstance(audio_path, str):
                raise ToneSpeakAuditError(f"ToneSpeak {context}: audio path is missing.")
            if not isinstance(text, str) or not " ".join(text.split()):
                raise ToneSpeakAuditError(f"ToneSpeak {context}: text is blank.")
            if not isinstance(description, str) or not " ".join(description.split()):
                raise ToneSpeakAuditError(f"ToneSpeak {context}: text_description is blank.")
            if not isinstance(voice_name, str) or voice_name not in TONE_SPEAK_EXPECTED_VOICES:
                raise ToneSpeakAuditError(
                    f"ToneSpeak {context}: unsupported voice_name {voice_name!r}."
                )
            safe_path = _safe_relative_path(audio_path)
            _expected_audio_path(safe_path, voice_name)
            if audio_path in audio_paths:
                raise ToneSpeakAuditError(
                    f"ToneSpeak {context}: duplicate audio path {audio_path!r}."
                )
            audio_paths.add(audio_path)
            audio_digest = hashlib.sha256(audio_bytes).hexdigest()
            if audio_digest in audio_hashes:
                duplicate_audio_payloads += 1
            audio_hashes.add(audio_digest)
            text_digest = _normalized_text_hash(text)
            prior_splits = text_hashes.setdefault(text_digest, set())
            if prior_splits:
                duplicate_normalized_texts += 1
            prior_splits.add(expected.split)
            if not any("\u0400" <= character <= "\u052f" for character in text):
                records_without_cyrillic += 1
            sample_rate, frames = _inspect_audio(audio_bytes, context=context)
            if sample_rate != TONE_SPEAK_EXPECTED_SAMPLE_RATE or frames <= 0:
                raise ToneSpeakAuditError(
                    f"ToneSpeak {context}: expected positive "
                    f"{TONE_SPEAK_EXPECTED_SAMPLE_RATE} Hz MP3, "
                    f"got {sample_rate} Hz and {frames} frames."
                )
            duration_s = frames / sample_rate
            rows_by_split[expected.split] += 1
            split_voice_counts[voice_name] += 1
            audio_total_bytes += len(audio_bytes)
            audio_total_duration_s += duration_s
            audio_min_duration_s = min(audio_min_duration_s, duration_s)
            audio_max_duration_s = max(audio_max_duration_s, duration_s)
            audio_sample_rates.add(sample_rate)

    if not audio_paths:
        raise ToneSpeakAuditError("ToneSpeak contains no audio records.")
    if set(rows_by_split) != {"train", "validation"}:
        raise ToneSpeakAuditError(f"ToneSpeak unexpected split inventory: {dict(rows_by_split)!r}.")
    for split, counts in voice_counts.items():
        missing_voices = sorted(TONE_SPEAK_EXPECTED_VOICES.difference(counts))
        if missing_voices:
            raise ToneSpeakAuditError(
                f"ToneSpeak split {split!r} lacks advertised voices: {missing_voices}."
            )
    cross_split_normalized_texts = sum(len(splits) > 1 for splits in text_hashes.values())
    if cross_split_normalized_texts:
        raise ToneSpeakAuditError(
            "ToneSpeak has normalized text groups shared by train and validation: "
            f"{cross_split_normalized_texts}."
        )
    if records_without_cyrillic:
        raise ToneSpeakAuditError(
            "ToneSpeak has text records without a Cyrillic code point: "
            f"{records_without_cyrillic}."
        )
    return ToneSpeakAudit(
        source_id=TONE_SPEAK_SOURCE_ID,
        dataset=TONE_SPEAK_DATASET,
        revision=TONE_SPEAK_REVISION,
        artifacts=receipts,
        artifact_total_bytes=sum(item.size_bytes for item in receipts),
        rows_by_split=dict(sorted(rows_by_split.items())),
        voice_counts_by_split={
            split: dict(sorted(counts.items())) for split, counts in sorted(voice_counts.items())
        },
        audio_records=len(audio_paths),
        audio_total_bytes=audio_total_bytes,
        audio_total_duration_s=audio_total_duration_s,
        audio_min_duration_s=audio_min_duration_s,
        audio_max_duration_s=audio_max_duration_s,
        audio_sample_rates=tuple(sorted(audio_sample_rates)),
        unique_audio_paths=len(audio_paths),
        duplicate_audio_payloads=duplicate_audio_payloads,
        unique_normalized_texts=len(text_hashes),
        duplicate_normalized_texts=duplicate_normalized_texts,
        cross_split_normalized_texts=cross_split_normalized_texts,
        records_without_cyrillic=records_without_cyrillic,
    )


def load_tone_speak_records(root: Path, *, source_split: str) -> list[ToneSpeakRecord]:
    """Read one audited source split while retaining text and source audio identities.

    Callers must run :func:`audit_tone_speak_release` first. This loader intentionally does not
    write audio or create a project manifest.
    """

    if source_split not in {"train", "validation"}:
        raise ToneSpeakAuditError(f"Unsupported ToneSpeak source split: {source_split!r}.")
    parquet = _load_parquet()
    records: list[ToneSpeakRecord] = []
    expected_by_path = {item.relative_path: item for item in TONE_SPEAK_EXPECTED_ARTIFACTS}
    for relative_path, expected in sorted(expected_by_path.items()):
        if expected.split != source_split:
            continue
        parquet_file = parquet.ParquetFile(root / relative_path)
        _validate_schema(parquet_file, relative_path=relative_path)
        if parquet_file.metadata.num_rows != expected.expected_rows:
            raise ToneSpeakAuditError(
                f"ToneSpeak row count changed after audit for {relative_path!r}."
            )
        for row_number, row in enumerate(_iter_rows(parquet_file), start=1):
            context = f"{relative_path}:{row_number}"
            audio = row["audio"]
            text = row["text"]
            voice_name = row["voice_name"]
            if not isinstance(audio, dict):
                raise ToneSpeakAuditError(f"ToneSpeak {context}: audio must be a struct.")
            audio_path = audio.get("path")
            if not isinstance(audio_path, str):
                raise ToneSpeakAuditError(f"ToneSpeak {context}: audio path is missing.")
            if not isinstance(text, str) or not " ".join(text.split()):
                raise ToneSpeakAuditError(f"ToneSpeak {context}: text is blank.")
            if not isinstance(voice_name, str) or voice_name not in TONE_SPEAK_EXPECTED_VOICES:
                raise ToneSpeakAuditError(
                    f"ToneSpeak {context}: unsupported voice_name {voice_name!r}."
                )
            safe_path = _safe_relative_path(audio_path)
            _expected_audio_path(safe_path, voice_name)
            records.append(
                ToneSpeakRecord(
                    source_split=source_split,
                    parquet_path=relative_path,
                    embedded_path=audio_path,
                    text=" ".join(text.split()),
                    text_hash=_normalized_text_hash(text),
                    voice_name=voice_name,
                )
            )
    if not records:
        raise ToneSpeakAuditError(f"ToneSpeak source split is empty: {source_split!r}.")
    paths = [record.embedded_path for record in records]
    if len(paths) != len(set(paths)):
        raise ToneSpeakAuditError(
            f"ToneSpeak source split has duplicate embedded paths: {source_split!r}."
        )
    return records


def select_tone_speak_validation_records(
    records: Iterable[ToneSpeakRecord],
    *,
    per_voice: int,
    seed: str,
    excluded_text_hashes: Iterable[str] = (),
) -> list[ToneSpeakRecord]:
    """Select a balanced, text-group-unique held-out slice from validation only."""

    if per_voice <= 0:
        raise ValueError("ToneSpeak per_voice must be positive.")
    if not seed:
        raise ValueError("ToneSpeak selection seed must not be empty.")
    record_list = list(records)
    if not record_list or any(record.source_split != "validation" for record in record_list):
        raise ToneSpeakAuditError("ToneSpeak OOD selection accepts validation records only.")
    if {record.voice_name for record in record_list}.difference(TONE_SPEAK_EXPECTED_VOICES):
        raise ToneSpeakAuditError("ToneSpeak records contain an unsupported voice name.")
    blocked = set(excluded_text_hashes)
    grouped: dict[str, list[ToneSpeakRecord]] = {}
    for record in record_list:
        if record.text_hash not in blocked:
            grouped.setdefault(record.text_hash, []).append(record)
    for text_hash, group in grouped.items():
        if len(group) != 1:
            raise ToneSpeakAuditError(
                "ToneSpeak validation has a repeated normalized text group: "
                f"{text_hash}."
            )
    selected: list[ToneSpeakRecord] = []
    for voice_name in sorted(TONE_SPEAK_EXPECTED_VOICES):
        candidates = [
            record
            for group in grouped.values()
            for record in group
            if record.voice_name == voice_name
        ]
        ranked = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item.source_split}:{voice_name}:{item.text_hash}:{item.embedded_path}".encode()
            ).digest(),
        )
        if len(ranked) < per_voice:
            raise ToneSpeakAuditError(
                f"ToneSpeak voice {voice_name!r} has only {len(ranked)} eligible validation rows; "
                f"need {per_voice}."
            )
        selected.extend(ranked[:per_voice])
    selected.sort(key=lambda item: (item.voice_name, item.embedded_path))
    text_hashes = [record.text_hash for record in selected]
    if len(text_hashes) != len(set(text_hashes)):
        raise ToneSpeakAuditError("ToneSpeak selection reuses a normalized text group.")
    return selected


def extract_tone_speak_audio_slice(
    root: Path, selected_records: Iterable[ToneSpeakRecord], destination: Path
) -> dict[str, Path]:
    """Atomically write selected embedded MP3 assets, never using bulk extraction."""

    selected = list(selected_records)
    if not selected:
        raise ToneSpeakAuditError("ToneSpeak extraction requires at least one selected record.")
    if destination.exists() or not destination.parent.is_dir():
        raise ToneSpeakAuditError(
            f"ToneSpeak destination must be absent below an existing parent: {destination}."
        )
    selected_by_path = {record.embedded_path: record for record in selected}
    if len(selected_by_path) != len(selected):
        raise ToneSpeakAuditError("ToneSpeak extraction selection has duplicate embedded paths.")
    parquet = _load_parquet()
    written: set[str] = set()
    with tempfile.TemporaryDirectory(
        prefix=".tone-speak-stage-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "slice"
        stage.mkdir()
        for parquet_path in sorted({record.parquet_path for record in selected}):
            parquet_file = parquet.ParquetFile(root / parquet_path)
            _validate_schema(parquet_file, relative_path=parquet_path)
            for row in _iter_rows(parquet_file):
                audio = row["audio"]
                if not isinstance(audio, dict):
                    raise ToneSpeakAuditError(f"ToneSpeak {parquet_path}: audio must be a struct.")
                audio_path = audio.get("path")
                audio_bytes = audio.get("bytes")
                if audio_path not in selected_by_path:
                    continue
                if not isinstance(audio_bytes, bytes) or not audio_bytes:
                    raise ToneSpeakAuditError(
                        f"ToneSpeak {parquet_path}: selected audio bytes are missing."
                    )
                if not isinstance(audio_path, str) or audio_path in written:
                    raise ToneSpeakAuditError(
                        f"ToneSpeak {parquet_path}: duplicate selected audio path {audio_path!r}."
                    )
                output = stage / _safe_relative_path(audio_path).name
                with output.open("xb") as handle:
                    handle.write(audio_bytes)
                written.add(audio_path)
        missing = sorted(set(selected_by_path).difference(written))
        if missing:
            raise ToneSpeakAuditError(
                f"ToneSpeak extraction could not find {len(missing)} selected embedded assets."
            )
        if destination.exists():
            raise ToneSpeakAuditError(
                "ToneSpeak destination appeared while extraction was staging."
            )
        stage.replace(destination)
    return {path: destination / PurePosixPath(path).name for path in selected_by_path}


def inspect_extracted_tone_speak_audio(
    path: Path, *, embedded_path: str, data_root: Path
) -> ToneSpeakExtractedAsset:
    """Validate one extracted source MP3 before it becomes a raw manifest asset."""

    try:
        info = sf.info(path)
    except RuntimeError as error:
        raise ToneSpeakAuditError(f"Cannot inspect extracted ToneSpeak MP3: {path}") from error
    if (
        info.format != "MP3"
        or info.samplerate != TONE_SPEAK_EXPECTED_SAMPLE_RATE
        or info.frames <= 0
        or info.duration <= 0
    ):
        raise ToneSpeakAuditError(f"Extracted ToneSpeak audio has invalid properties: {path}")
    try:
        relative_path = (
            path.resolve(strict=True).relative_to(data_root.resolve(strict=True)).as_posix()
        )
    except ValueError as error:
        raise ToneSpeakAuditError(f"Extracted ToneSpeak path escapes data root: {path}") from error
    return ToneSpeakExtractedAsset(
        embedded_path=embedded_path,
        relative_path=relative_path,
        sha256=sha256_file(path),
        duration_s=float(info.duration),
        original_sr=int(info.samplerate),
        codec="mp3",
    )


def tone_speak_ood_manifest_rows(
    records: Iterable[ToneSpeakRecord],
    assets: Mapping[str, ToneSpeakExtractedAsset],
    *,
    created_at: str | None = None,
) -> list[ManifestRow]:
    """Build research-only OOD spoof rows from extracted validation source assets."""

    selected = list(records)
    if not selected or any(record.source_split != "validation" for record in selected):
        raise ToneSpeakAuditError(
            "ToneSpeak OOD manifest requires non-empty validation records only."
        )
    text_hashes = [record.text_hash for record in selected]
    if len(text_hashes) != len(set(text_hashes)):
        raise ToneSpeakAuditError("ToneSpeak OOD manifest must not reuse a normalized text group.")
    timestamp = created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    rows: list[ManifestRow] = []
    for record in selected:
        asset = assets.get(record.embedded_path)
        if asset is None or asset.embedded_path != record.embedded_path:
            raise ToneSpeakAuditError(
                f"ToneSpeak OOD manifest lacks extracted asset {record.embedded_path!r}."
            )
        rows.append(
            ManifestRow(
                sample_id=f"{TONE_SPEAK_SOURCE_ID}:{Path(record.embedded_path).stem}",
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split="ood",
                label="spoof",
                language="ru",
                code_switch="false",
                parent_group_id=f"{TONE_SPEAK_SOURCE_ID}:text:{record.text_hash}",
                source_name=TONE_SPEAK_SOURCE_ID,
                source_license="Apache-2.0",
                rights_basis=(
                    f"ToneSpeak pinned revision {TONE_SPEAK_REVISION}; dataset card declares "
                    "GPT-4.1 mini text generation and GPT-4o mini TTS"
                ),
                speaker_pseudo_id=f"{TONE_SPEAK_SOURCE_ID}:voice:{record.voice_name}",
                text_id=f"{TONE_SPEAK_SOURCE_ID}:text:{record.text_hash}",
                text_hash=record.text_hash,
                duration_s=asset.duration_s,
                generator_family="neural_tts",
                generator_name="openai_gpt_4o_mini_tts",
                generator_version="source_card_unpinned",
                voice_id=f"{TONE_SPEAK_SOURCE_ID}:voice:{record.voice_name}",
                clone_consent_id="",
                device="unknown",
                capture_route="openai_tts_source_release",
                original_sr=asset.original_sr,
                codec=asset.codec,
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows
