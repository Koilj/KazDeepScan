"""Safe, reproducible intake primitives for the pinned Google FLEURS release.

FLEURS distributes one gzip-compressed TAR archive and one headerless TSV per
source split.  A sentence is usually read by more than one participant, while
speaker IDs are intentionally not published.  This module therefore treats a
normalised transcript as an indivisible group and never turns gender, prompt
ID, or a filename into a purported speaker identity.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow

FLEURS_REVISION = "4683b04af03d2d9549064c7d72060a9a94bb6046"
FLEURS_LICENSE = "CC-BY-4.0"
FLEURS_SOURCE_URL = f"https://huggingface.co/datasets/google/fleurs/tree/{FLEURS_REVISION}"
FLEURS_SOURCE_SPLITS = ("train", "dev", "test")
FLEURS_TSV_COLUMNS = 7
FLEURS_GENDERS = frozenset({"FEMALE", "MALE"})
FLEURS_AUDIO_COPY_CHUNK_BYTES = 1024 * 1024


class FleursIngestionError(ValueError):
    """Raised when a FLEURS release cannot safely become a project manifest."""


@dataclass(frozen=True, slots=True)
class FleursArtifactSpec:
    relative_path: str
    expected_size_bytes: int
    lfs_sha256: str = ""
    git_blob_sha1: str = ""

    def __post_init__(self) -> None:
        if self.expected_size_bytes <= 0:
            raise ValueError("FLEURS artifact size must be positive.")
        if bool(self.lfs_sha256) == bool(self.git_blob_sha1):
            raise ValueError(
                "FLEURS artifact must pin exactly one of LFS SHA-256 or Git blob SHA-1."
            )


@dataclass(frozen=True, slots=True)
class FleursLocaleSpec:
    locale: str
    language: str
    source_id: str
    artifacts: tuple[FleursArtifactSpec, ...]

    def __post_init__(self) -> None:
        if self.locale not in {"kk_kz", "ru_ru"}:
            raise ValueError(f"Unsupported FLEURS locale: {self.locale!r}.")
        if self.language not in {"kk", "ru"}:
            raise ValueError(f"Unsupported FLEURS language: {self.language!r}.")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate FLEURS artifact paths for {self.locale}.")
        expected_paths = {
            f"data/{self.locale}/audio/{split}.tar.gz" for split in FLEURS_SOURCE_SPLITS
        } | {f"data/{self.locale}/{split}.tsv" for split in FLEURS_SOURCE_SPLITS}
        if set(paths) != expected_paths:
            raise ValueError(f"FLEURS artifact set is incomplete for {self.locale}.")

    def artifact(self, relative_path: str) -> FleursArtifactSpec:
        for artifact in self.artifacts:
            if artifact.relative_path == relative_path:
                return artifact
        raise KeyError(relative_path)


def _locale_spec(
    locale: str,
    language: str,
    source_id: str,
    audio: Mapping[str, tuple[int, str]],
    tsv: Mapping[str, tuple[int, str]],
) -> FleursLocaleSpec:
    artifacts = tuple(
        [
            FleursArtifactSpec(
                relative_path=f"data/{locale}/audio/{split}.tar.gz",
                expected_size_bytes=audio[split][0],
                lfs_sha256=audio[split][1],
            )
            for split in FLEURS_SOURCE_SPLITS
        ]
        + [
            FleursArtifactSpec(
                relative_path=f"data/{locale}/{split}.tsv",
                expected_size_bytes=tsv[split][0],
                git_blob_sha1=tsv[split][1],
            )
            for split in FLEURS_SOURCE_SPLITS
        ]
    )
    return FleursLocaleSpec(
        locale=locale,
        language=language,
        source_id=source_id,
        artifacts=artifacts,
    )


FLEURS_LOCALE_SPECS: Mapping[str, FleursLocaleSpec] = {
    "kk_kz": _locale_spec(
        "kk_kz",
        "kk",
        "google_fleurs_kk_v1",
        {
            "train": (
                1_992_291_838,
                "4d28cb336b1c3207b993b96a301e23aceb513739ea0079ba9dd19e30da661393",
            ),
            "dev": (
                248_842_953,
                "a0481dd445d9ae8424eb870751f1ab51d82f0b7f5566318ffa5adab7f9c9f69d",
            ),
            "test": (
                628_328_020,
                "f29331a9731a00f544e2cb7d1dbcbb5a4258fd200323811ec837d15dd3c2e8a0",
            ),
        },
        {
            "train": (2_812_850, "dd766bfd389fb4b02bccd5039fbc022195724b29"),
            "dev": (312_534, "e9eb4ca68346f8132c56b63502df7ac7c5094e16"),
            "test": (787_758, "8dabc83e97e1916f47c1b7e5150aa04bb033aacd"),
        },
    ),
    "ru_ru": _locale_spec(
        "ru_ru",
        "ru",
        "google_fleurs_ru_v1",
        {
            "train": (
                1_427_743_798,
                "1966e34c17d612317f03916fb45159c793947de77e1988fe76b3b53cf6c40b44",
            ),
            "dev": (
                187_377_526,
                "4dc533cd0f312f749bd6d0fe968f3183288d938e07ed97dc141d9f10570457f9",
            ),
            "test": (
                433_142_634,
                "8a0d6a0d23c3421f50c575bcf65d875cd19dadae2f7cabb415024f81b65178b1",
            ),
        },
        {
            "train": (2_358_557, "ad3fbc79df6727f4a168b5bd71ed14eb92cc264d"),
            "dev": (316_328, "8e6be8c301fcdf2df3a99219c9c6ab643cf10e36"),
            "test": (735_258, "1b1b37112dcbc6089433084255785f03f9fce445"),
        },
    ),
}


@dataclass(frozen=True, slots=True)
class FleursRecord:
    locale: str
    language: str
    source_split: str
    prompt_id: str
    filename: str
    raw_transcript: str
    transcript: str
    character_transcript: str
    samples: int
    gender: str

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.transcript.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FleursExtractedAsset:
    filename: str
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int
    codec: str


@dataclass(frozen=True, slots=True)
class FleursArchiveReport:
    archive: Path
    source_split: str
    audio_files: int


@dataclass(frozen=True, slots=True)
class FleursReleaseReport:
    locale: str
    artifacts: dict[str, str]
    source_splits: dict[str, int]
    unique_text_groups: dict[str, int]
    archives: tuple[FleursArchiveReport, ...]


def fleurs_locale_spec(locale: str) -> FleursLocaleSpec:
    try:
        return FLEURS_LOCALE_SPECS[locale]
    except KeyError as error:
        raise FleursIngestionError(
            f"Unsupported FLEURS locale {locale!r}; expected one of {sorted(FLEURS_LOCALE_SPECS)}."
        ) from error


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or value in {".", ".."}
    ):
        raise FleursIngestionError(f"Unsafe FLEURS relative path: {value!r}.")
    return path


def _release_path(release_root: Path, relative_path: str) -> Path:
    root = release_root.resolve(strict=True)
    candidate = (root / _safe_relative_path(relative_path)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FleursIngestionError(
            f"FLEURS path escapes release root: {relative_path!r}."
        ) from error
    return candidate


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - Git's published object identity is SHA-1.
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FLEURS_AUDIO_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(release_root: Path, artifact: FleursArtifactSpec) -> str:
    path = _release_path(release_root, artifact.relative_path)
    if not path.is_file():
        raise FleursIngestionError(f"Missing FLEURS artifact: {path}")
    actual_size = path.stat().st_size
    if actual_size != artifact.expected_size_bytes:
        raise FleursIngestionError(
            f"FLEURS artifact size mismatch for {artifact.relative_path}: expected "
            f"{artifact.expected_size_bytes}, got {actual_size}."
        )
    if artifact.lfs_sha256:
        actual_hash = sha256_file(path)
        if actual_hash != artifact.lfs_sha256:
            raise FleursIngestionError(
                f"FLEURS LFS SHA-256 mismatch for {artifact.relative_path}: expected "
                f"{artifact.lfs_sha256}, got {actual_hash}."
            )
        return actual_hash
    actual_blob_id = _git_blob_sha1(path)
    if actual_blob_id != artifact.git_blob_sha1:
        raise FleursIngestionError(
            f"FLEURS Git blob SHA-1 mismatch for {artifact.relative_path}: expected "
            f"{artifact.git_blob_sha1}, got {actual_blob_id}."
        )
    return sha256_file(path)


def _canonical_transcript(value: str) -> str:
    return " ".join(value.split())


def _safe_filename(value: str) -> str:
    path = _safe_relative_path(value)
    if len(path.parts) != 1 or path.suffix.lower() != ".wav" or not path.stem.isdecimal():
        raise FleursIngestionError(f"Invalid FLEURS audio filename: {value!r}.")
    return value


def _read_fleurs_tsv(
    path: Path, *, locale: str, language: str, source_split: str
) -> list[FleursRecord]:
    records: list[FleursRecord] = []
    filenames: set[str] = set()
    prompt_transcripts: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            # FLEURS TSV is not RFC-4180 CSV: a leading double quote can be part of an
            # ordinary transcript.  Treat it as literal text instead of allowing a quote to
            # consume tabs/newlines from following records.
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row_number, fields in enumerate(reader, start=1):
                if len(fields) != FLEURS_TSV_COLUMNS:
                    raise FleursIngestionError(
                        f"{path}:{row_number}: expected {FLEURS_TSV_COLUMNS} TSV columns, got "
                        f"{len(fields)}."
                    )
                prompt_id, filename, raw, transcript, chars, samples, gender = fields
                prompt_id = prompt_id.strip()
                if not prompt_id.isdecimal():
                    raise FleursIngestionError(f"{path}:{row_number}: invalid FLEURS prompt ID.")
                filename = _safe_filename(filename.strip())
                if filename in filenames:
                    raise FleursIngestionError(
                        f"{path}:{row_number}: duplicate filename {filename!r}."
                    )
                filenames.add(filename)
                raw = _canonical_transcript(raw)
                transcript = _canonical_transcript(transcript)
                chars = _canonical_transcript(chars)
                if not raw or not transcript or not chars:
                    raise FleursIngestionError(f"{path}:{row_number}: empty transcript field.")
                previous = prompt_transcripts.setdefault(prompt_id, transcript)
                if previous != transcript:
                    raise FleursIngestionError(
                        f"{path}:{row_number}: prompt ID {prompt_id!r} has conflicting transcripts."
                    )
                try:
                    sample_count = int(samples)
                except ValueError as error:
                    raise FleursIngestionError(
                        f"{path}:{row_number}: samples must be a positive integer."
                    ) from error
                if sample_count <= 0:
                    raise FleursIngestionError(f"{path}:{row_number}: samples must be positive.")
                gender = gender.strip().upper()
                if gender not in FLEURS_GENDERS:
                    raise FleursIngestionError(
                        f"{path}:{row_number}: unsupported gender value {gender!r}."
                    )
                records.append(
                    FleursRecord(
                        locale=locale,
                        language=language,
                        source_split=source_split,
                        prompt_id=prompt_id,
                        filename=filename,
                        raw_transcript=raw,
                        transcript=transcript,
                        character_transcript=chars,
                        samples=sample_count,
                        gender=gender,
                    )
                )
    except UnicodeDecodeError as error:
        raise FleursIngestionError(f"FLEURS TSV is not valid UTF-8: {path}") from error
    if not records:
        raise FleursIngestionError(f"FLEURS TSV has no rows: {path}")
    return records


def _expected_archive_member(split: str, member: tarfile.TarInfo) -> str | None:
    path = PurePosixPath(member.name)
    if member.isdir():
        if path.as_posix().rstrip("/") != split:
            raise FleursIngestionError(f"Unexpected FLEURS archive directory: {member.name!r}.")
        return None
    if not member.isfile() or len(path.parts) != 2 or path.parts[0] != split:
        raise FleursIngestionError(f"Unexpected FLEURS archive member: {member.name!r}.")
    return _safe_filename(path.name)


def _inspect_archive(
    archive: Path, split: str, expected_filenames: set[str]
) -> FleursArchiveReport:
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                filename = _expected_archive_member(split, member)
                if filename is None:
                    continue
                if filename in seen:
                    raise FleursIngestionError(
                        f"Duplicate FLEURS archive member {filename!r} in {archive}."
                    )
                seen.add(filename)
    except (OSError, tarfile.TarError) as error:
        raise FleursIngestionError(
            f"Cannot safely read FLEURS archive {archive}: {error}"
        ) from error
    if seen != expected_filenames:
        missing = sorted(expected_filenames.difference(seen))
        unexpected = sorted(seen.difference(expected_filenames))
        details: list[str] = []
        if missing:
            details.append(f"missing={len(missing)}")
        if unexpected:
            details.append(f"unexpected={len(unexpected)}")
        raise FleursIngestionError(
            f"FLEURS archive/TSV membership mismatch for {archive}: {', '.join(details)}."
        )
    return FleursArchiveReport(archive=archive, source_split=split, audio_files=len(seen))


def inspect_fleurs_release(
    release_root: Path, locale: str, *, spec: FleursLocaleSpec | None = None
) -> tuple[FleursReleaseReport, dict[str, list[FleursRecord]]]:
    """Verify every pinned artifact and TAR/TSV correspondence without extraction.

    Audio LFS objects are verified by SHA-256, TSV files by their published Git blob object ID,
    and all artifacts by exact size.  Traversing a gzip TAR to EOF also validates its CRC.
    """

    locale_spec = spec if spec is not None else fleurs_locale_spec(locale)
    if locale_spec.locale != locale:
        raise FleursIngestionError("FLEURS locale and supplied locale spec disagree.")
    if not release_root.is_dir():
        raise FleursIngestionError(f"FLEURS release root does not exist: {release_root}")
    artifact_hashes = {
        artifact.relative_path: _verify_artifact(release_root, artifact)
        for artifact in locale_spec.artifacts
    }
    records_by_split: dict[str, list[FleursRecord]] = {}
    archives: list[FleursArchiveReport] = []
    for split in FLEURS_SOURCE_SPLITS:
        tsv_path = _release_path(release_root, f"data/{locale}/{split}.tsv")
        records = _read_fleurs_tsv(
            tsv_path,
            locale=locale,
            language=locale_spec.language,
            source_split=split,
        )
        records_by_split[split] = records
        archive_path = _release_path(release_root, f"data/{locale}/audio/{split}.tar.gz")
        archives.append(
            _inspect_archive(archive_path, split, {record.filename for record in records})
        )
    report = FleursReleaseReport(
        locale=locale,
        artifacts=artifact_hashes,
        source_splits={split: len(records_by_split[split]) for split in FLEURS_SOURCE_SPLITS},
        unique_text_groups={
            split: len({record.text_hash for record in records_by_split[split]})
            for split in FLEURS_SOURCE_SPLITS
        },
        archives=tuple(archives),
    )
    return report, records_by_split


def verified_fleurs_test_transcripts(
    release_root: Path, rows: Iterable[ManifestRow]
) -> dict[str, str]:
    """Return source-verbatim test transcripts only after revalidating the full FLEURS release."""

    selected = list(rows)
    source_locales = {
        "google_fleurs_ru_v1": "ru_ru",
        "google_fleurs_kk_v1": "kk_kz",
    }
    by_source: dict[str, list[ManifestRow]] = {}
    for row in selected:
        locale = source_locales.get(row.source_name)
        if locale is None or row.split != "test" or row.label != "bonafide":
            raise FleursIngestionError(
                "Transcript verification accepts only FLEURS ru/kk test bona-fide rows."
            )
        by_source.setdefault(row.source_name, []).append(row)
    transcripts: dict[str, str] = {}
    for source_name, source_rows in by_source.items():
        locale = source_locales[source_name]
        _report, records_by_split = inspect_fleurs_release(release_root, locale)
        records = {
            record.filename.removesuffix(".wav"): record for record in records_by_split["test"]
        }
        for row in source_rows:
            source_prefix = f"{row.source_name}:"
            if not row.sample_id.startswith(source_prefix):
                raise FleursIngestionError(
                    f"FLEURS sample has an invalid source prefix: {row.sample_id!r}"
                )
            record = records.get(row.sample_id.removeprefix(source_prefix))
            if record is None:
                raise FleursIngestionError(
                    f"FLEURS test transcript is missing for {row.sample_id!r}"
                )
            expected_text_id = f"{row.source_name}:prompt:{record.prompt_id}"
            if row.text_id != expected_text_id or row.text_hash != record.text_hash:
                raise FleursIngestionError(
                    f"FLEURS transcript provenance mismatch for {row.sample_id!r}."
                )
            transcripts[row.sample_id] = record.transcript
    if len(transcripts) != len(selected):
        raise FleursIngestionError("FLEURS transcript verification did not cover every base row.")
    return transcripts


def select_fleurs_records(
    records: Iterable[FleursRecord],
    limit: int,
    seed: str,
    *,
    excluded_filenames: Iterable[str] = (),
    excluded_text_hashes: Iterable[str] = (),
) -> list[FleursRecord]:
    """Choose one deterministic recording per text group before any extraction is published."""

    if limit <= 0:
        raise ValueError("FLEURS selection limit must be positive.")
    if not seed:
        raise ValueError("FLEURS selection seed must not be empty.")
    record_list = list(records)
    if not record_list:
        raise FleursIngestionError("Cannot select from an empty FLEURS record list.")
    locales = {record.locale for record in record_list}
    splits = {record.source_split for record in record_list}
    if len(locales) != 1 or len(splits) != 1:
        raise FleursIngestionError(
            "FLEURS selection must use exactly one locale and one source split."
        )
    blocked_filenames = {_safe_filename(value) for value in excluded_filenames}
    blocked_text_hashes = set(excluded_text_hashes)
    grouped: dict[str, list[FleursRecord]] = {}
    for record in record_list:
        if record.filename in blocked_filenames or record.text_hash in blocked_text_hashes:
            continue
        grouped.setdefault(record.text_hash, []).append(record)
    selected_candidates: list[FleursRecord] = []
    for text_hash, group in grouped.items():
        selected_candidates.append(
            min(
                group,
                key=lambda item: hashlib.sha256(
                    f"{seed}:{item.locale}:{item.source_split}:{text_hash}:{item.filename}".encode()
                ).digest(),
            )
        )
    selected = sorted(
        selected_candidates,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item.locale}:{item.source_split}:{item.text_hash}".encode()
        ).digest(),
    )[:limit]
    if len(selected) < limit:
        raise FleursIngestionError(
            f"FLEURS has only {len(selected)} eligible text groups after exclusions; need {limit}."
        )
    return selected


def _safe_destination(destination: Path) -> Path:
    if destination.exists():
        raise FleursIngestionError(f"Refusing to overwrite FLEURS destination: {destination}")
    if not destination.parent.is_dir():
        raise FleursIngestionError(
            f"FLEURS destination parent does not exist: {destination.parent}"
        )
    return destination.parent.resolve(strict=True)


def extract_fleurs_audio_slice(
    release_root: Path,
    locale: str,
    source_split: str,
    selected_records: Iterable[FleursRecord],
    destination: Path,
) -> dict[str, Path]:
    """Atomically extract a selected, verified FLEURS split without ``extractall``."""

    if source_split not in FLEURS_SOURCE_SPLITS:
        raise FleursIngestionError(f"Unsupported FLEURS source split: {source_split!r}.")
    selected = list(selected_records)
    if not selected:
        raise FleursIngestionError("FLEURS extraction requires at least one selected record.")
    if any(record.locale != locale or record.source_split != source_split for record in selected):
        raise FleursIngestionError(
            "FLEURS selected records do not match the requested locale/split."
        )
    filenames = [record.filename for record in selected]
    if len(filenames) != len(set(filenames)):
        raise FleursIngestionError("FLEURS extraction selection has duplicate filenames.")
    parent = _safe_destination(destination)
    archive = _release_path(release_root, f"data/{locale}/audio/{source_split}.tar.gz")
    selected_paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix=".fleurs-stage-", dir=parent) as temporary_dir:
        stage = Path(temporary_dir) / "slice"
        stage.mkdir()
        try:
            with tarfile.open(archive, mode="r:gz") as tar:
                for member in tar:
                    filename = _expected_archive_member(source_split, member)
                    if filename is None or filename not in set(filenames):
                        continue
                    source = tar.extractfile(member)
                    if source is None:
                        raise FleursIngestionError(
                            f"Cannot read selected FLEURS member {member.name!r}."
                        )
                    output = stage / filename
                    with source, output.open("xb") as handle:
                        shutil.copyfileobj(source, handle, length=FLEURS_AUDIO_COPY_CHUNK_BYTES)
                    selected_paths[filename] = output
        except (OSError, tarfile.TarError) as error:
            raise FleursIngestionError(
                f"Cannot extract FLEURS archive {archive}: {error}"
            ) from error
        missing = sorted(set(filenames).difference(selected_paths))
        if missing:
            raise FleursIngestionError(
                f"FLEURS archive is missing {len(missing)} selected audio members."
            )
        destination_path = parent / destination.name
        if destination_path.exists():
            raise FleursIngestionError(
                f"FLEURS destination appeared while extraction was staging: {destination_path}"
            )
        stage.replace(destination_path)
    return {filename: destination / filename for filename in filenames}


def inspect_extracted_fleurs_audio(path: Path) -> tuple[float, int, str]:
    try:
        info = sf.info(path)
    except RuntimeError as error:
        raise FleursIngestionError(
            f"Cannot inspect extracted FLEURS WAV {path}: {error}"
        ) from error
    if info.samplerate <= 0 or info.frames <= 0 or info.duration <= 0:
        raise FleursIngestionError(f"FLEURS WAV has invalid audio properties: {path}")
    if info.samplerate != 16_000:
        raise FleursIngestionError(f"FLEURS WAV must be 16 kHz, got {info.samplerate} Hz: {path}")
    return float(info.duration), int(info.samplerate), "wav"


def fleurs_manifest_rows(
    records: Iterable[FleursRecord],
    assets: Mapping[str, FleursExtractedAsset],
    *,
    manifest_split: str,
    created_at: str | None = None,
) -> list[ManifestRow]:
    if manifest_split not in {"test", "ood"}:
        raise FleursIngestionError(
            "FLEURS final candidates may use only split='test' or split='ood'."
        )
    record_list = list(records)
    if not record_list:
        raise FleursIngestionError("Cannot build a FLEURS manifest with no records.")
    locale = record_list[0].locale
    locale_spec = fleurs_locale_spec(locale)
    if any(
        record.locale != locale or record.language != locale_spec.language for record in record_list
    ):
        raise FleursIngestionError("FLEURS manifest records contain multiple locales or languages.")
    text_hashes = [record.text_hash for record in record_list]
    if len(text_hashes) != len(set(text_hashes)):
        raise FleursIngestionError(
            "FLEURS manifest must contain at most one recording per text group."
        )
    timestamp = created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    rows: list[ManifestRow] = []
    for record in record_list:
        try:
            asset = assets[record.filename]
        except KeyError as error:
            raise FleursIngestionError(
                f"Missing extracted asset for FLEURS filename {record.filename!r}."
            ) from error
        source_prefix = f"{locale_spec.source_id}:"
        rows.append(
            ManifestRow(
                sample_id=f"{source_prefix}{record.filename.removesuffix('.wav')}",
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split=manifest_split,
                label="bonafide",
                language=record.language,
                code_switch="false",
                parent_group_id=f"{source_prefix}prompt:{record.prompt_id}",
                source_name=locale_spec.source_id,
                source_license=FLEURS_LICENSE,
                rights_basis=(
                    "Google FLEURS pinned release "
                    f"{FLEURS_REVISION}; CC-BY-4.0 attribution retained"
                ),
                speaker_pseudo_id=f"{source_prefix}unknown",
                text_id=f"{source_prefix}prompt:{record.prompt_id}",
                text_hash=record.text_hash,
                duration_s=asset.duration_s,
                generator_family="",
                generator_name="",
                generator_version="",
                voice_id="",
                clone_consent_id="",
                device="unknown",
                capture_route="read_speech_corpus",
                original_sr=asset.original_sr,
                codec=asset.codec,
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows
