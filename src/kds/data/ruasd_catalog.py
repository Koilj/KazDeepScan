"""Read-only integrity checks and metadata audit for a full RuASD release."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

CATALOG_FIELDS = frozenset(
    {"archive_name", "expected_size_bytes", "sha256", "pinned_revision", "source_url"}
)
KNOWN_LABELS = frozenset({"real", "fake"})
KNOWN_GROUPS = frozenset({"raw", "augmented"})
KNOWN_SOURCE_TYPES = frozenset({"tts", "real_speech", "augmented_audio"})
UNKNOWN_SPEAKER_VALUES = frozenset({"", "-1", "unknown", "none", "null"})
AuditProgressCallback = Callable[[int, int, str], None]


class RuAsdCatalogError(ValueError):
    """Raised when the RuASD artifact catalog or release layout is unsafe."""


@dataclass(frozen=True, slots=True)
class RuAsdArchiveSpec:
    archive_name: str
    expected_size_bytes: int
    sha256: str
    pinned_revision: str
    source_url: str


@dataclass(frozen=True, slots=True)
class RuAsdCollectionAudit:
    archive_count: int
    sha256_verified_archives: int
    records: int
    record_counts: dict[str, int]
    subset_counts: dict[str, int]
    model_counts: dict[str, int]
    speaker_counts: dict[str, int]
    text_counts: dict[str, int]

    def as_mapping(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_ruasd_artifact_catalog(path: Path) -> dict[str, RuAsdArchiveSpec]:
    """Load the pinned official artifact list used to validate local archives."""

    if not path.is_file():
        raise RuAsdCatalogError(f"RuASD artifact catalog does not exist: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or set(reader.fieldnames) != CATALOG_FIELDS:
                raise RuAsdCatalogError("RuASD artifact catalog has an unexpected header.")
            specs: dict[str, RuAsdArchiveSpec] = {}
            for row_number, row in enumerate(reader, start=2):
                spec = _catalog_spec_from_row(row, row_number)
                if spec.archive_name in specs:
                    raise RuAsdCatalogError(
                        f"Duplicate RuASD archive in catalog: {spec.archive_name!r}."
                    )
                specs[spec.archive_name] = spec
    except csv.Error as error:
        raise RuAsdCatalogError(f"Cannot parse RuASD artifact catalog: {path}") from error
    if not specs:
        raise RuAsdCatalogError("RuASD artifact catalog is empty.")
    return specs


def audit_ruasd_collection(
    archive_dir: Path,
    catalog: Mapping[str, RuAsdArchiveSpec],
    *,
    verify_sha256: bool = False,
    progress_callback: AuditProgressCallback | None = None,
) -> RuAsdCollectionAudit:
    """Audit every pinned archive without extracting or decoding audio files."""

    archive_paths = _validate_archive_set(archive_dir, catalog)
    record_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    total_records = 0
    verified_archives = 0

    for completed_archives, archive_name in enumerate(sorted(catalog), start=1):
        spec = catalog[archive_name]
        archive_path = archive_paths[archive_name]
        _validate_archive_size(archive_path, spec)
        if verify_sha256:
            if sha256_file(archive_path) != spec.sha256:
                raise RuAsdCatalogError(
                    f"RuASD archive SHA-256 does not match catalog: {archive_name!r}."
                )
            verified_archives += 1
        summaries = _audit_single_archive(archive_path)
        total_records += len(summaries)
        for summary in summaries:
            record_key = "/".join((summary.label, summary.group, summary.source_type))
            record_counts[record_key] += 1
            if summary.subset:
                subset_counts[f"{summary.label}/{summary.group}/{summary.subset}"] += 1
            if summary.model:
                model_counts[f"{summary.label}/{summary.group}/{summary.model}"] += 1
            speaker_key = "known" if summary.speaker_group is not None else "unknown"
            speaker_counts[f"{summary.label}/{summary.group}/{speaker_key}"] += 1
            text_key = "source_text_present" if summary.has_source_text else "source_text_missing"
            text_counts[f"{summary.label}/{summary.group}/{text_key}"] += 1
        if progress_callback is not None:
            progress_callback(completed_archives, len(catalog), archive_name)

    return RuAsdCollectionAudit(
        archive_count=len(catalog),
        sha256_verified_archives=verified_archives,
        records=total_records,
        record_counts=dict(sorted(record_counts.items())),
        subset_counts=dict(sorted(subset_counts.items())),
        model_counts=dict(sorted(model_counts.items())),
        speaker_counts=dict(sorted(speaker_counts.items())),
        text_counts=dict(sorted(text_counts.items())),
    )


def write_ruasd_audit_report(path: Path, audit: RuAsdCollectionAudit) -> None:
    """Atomically publish one new JSON report; never overwrite a prior audit."""

    if path.exists() or not path.parent.is_dir():
        raise RuAsdCatalogError(f"Unsafe RuASD audit report destination: {path}")
    try:
        with tempfile.TemporaryDirectory(prefix="kds-ruasd-report-", dir=path.parent) as stage_dir:
            staged_path = Path(stage_dir) / path.name
            staged_path.write_text(
                json.dumps(audit.as_mapping(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shutil.move(str(staged_path), path)
    except OSError as error:
        raise RuAsdCatalogError(f"Cannot write RuASD audit report: {path}") from error


@dataclass(frozen=True, slots=True)
class _RuAsdRecordSummary:
    label: str
    group: str
    source_type: str
    subset: str
    model: str
    speaker_group: str | None
    has_source_text: bool


def _catalog_spec_from_row(row: Mapping[str, str | None], row_number: int) -> RuAsdArchiveSpec:
    def value(name: str) -> str:
        candidate = (row.get(name) or "").strip()
        if not candidate:
            raise RuAsdCatalogError(f"RuASD catalog row {row_number}: {name} is empty.")
        return candidate

    archive_name = value("archive_name")
    path = PurePosixPath(archive_name)
    if len(path.parts) != 1 or path.name != archive_name or not archive_name.endswith(".tar"):
        raise RuAsdCatalogError(
            f"RuASD catalog row {row_number}: unsafe archive name {archive_name!r}."
        )
    try:
        expected_size_bytes = int(value("expected_size_bytes"))
    except ValueError as error:
        raise RuAsdCatalogError(
            f"RuASD catalog row {row_number}: expected_size_bytes must be an integer."
        ) from error
    sha256 = value("sha256").lower()
    if expected_size_bytes <= 0 or len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuAsdCatalogError(f"RuASD catalog row {row_number}: invalid size or SHA-256.")
    return RuAsdArchiveSpec(
        archive_name=archive_name,
        expected_size_bytes=expected_size_bytes,
        sha256=sha256,
        pinned_revision=value("pinned_revision"),
        source_url=value("source_url"),
    )


def _validate_archive_set(
    archive_dir: Path, catalog: Mapping[str, RuAsdArchiveSpec]
) -> dict[str, Path]:
    if not archive_dir.is_dir():
        raise RuAsdCatalogError(f"RuASD archive directory does not exist: {archive_dir}")
    available = {
        path.name: path
        for path in archive_dir.iterdir()
        if path.is_file() and path.name.startswith("ruasd-") and path.name.endswith(".tar")
    }
    expected = set(catalog)
    missing = sorted(expected.difference(available))
    unexpected = sorted(set(available).difference(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if unexpected:
            details.append(f"unexpected={','.join(unexpected)}")
        raise RuAsdCatalogError(
            "RuASD archive set does not match the pinned catalog: " + "; ".join(details)
        )
    return {name: available[name] for name in catalog}


def _validate_archive_size(path: Path, spec: RuAsdArchiveSpec) -> None:
    if path.stat().st_size != spec.expected_size_bytes:
        raise RuAsdCatalogError(
            f"RuASD archive size does not match catalog: {spec.archive_name!r}."
        )


def _audit_single_archive(archive_path: Path) -> list[_RuAsdRecordSummary]:
    audio_ids: set[str] = set()
    records: dict[str, _RuAsdRecordSummary] = {}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                sample_id, suffix = _safe_member_name(member)
                if suffix == ".wav":
                    if member.size <= 0 or sample_id in audio_ids:
                        raise RuAsdCatalogError(f"Invalid RuASD audio member: {member.name!r}.")
                    audio_ids.add(sample_id)
                    continue
                if sample_id in records:
                    raise RuAsdCatalogError(f"Duplicate RuASD metadata member: {member.name!r}.")
                records[sample_id] = _read_record_summary(member, archive)
    except (OSError, tarfile.TarError) as error:
        raise RuAsdCatalogError(f"RuASD archive cannot be read safely: {archive_path}") from error
    if not records or set(records) != audio_ids:
        raise RuAsdCatalogError(
            f"RuASD archive lacks exact one-to-one JSON/WAV pairs: {archive_path.name!r}."
        )
    return list(records.values())


def _safe_member_name(member: tarfile.TarInfo) -> tuple[str, str]:
    path = PurePosixPath(member.name)
    if (
        not member.isfile()
        or member.issym()
        or member.islnk()
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != member.name
        or "\\" in member.name
        or path.suffix.lower() not in {".json", ".wav"}
    ):
        raise RuAsdCatalogError(f"Unsafe RuASD archive member: {member.name!r}.")
    return path.stem, path.suffix.lower()


def _read_record_summary(
    member: tarfile.TarInfo, archive: tarfile.TarFile
) -> _RuAsdRecordSummary:
    if member.size <= 0 or member.size > 1_000_000:
        raise RuAsdCatalogError(f"Unsafe RuASD metadata size: {member.name!r}.")
    source = archive.extractfile(member)
    if source is None:
        raise RuAsdCatalogError(f"Cannot read RuASD metadata: {member.name!r}.")
    try:
        with source:
            raw = json.loads(source.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuAsdCatalogError(f"Invalid RuASD metadata: {member.name!r}.") from error
    if not isinstance(raw, Mapping):
        raise RuAsdCatalogError(f"RuASD metadata is not an object: {member.name!r}.")

    sample_id = _text_field(raw, "sample_id")
    filename = _text_field(raw, "filename")
    audio_relpath = _text_field(raw, "audio_relpath")
    audio_path = PurePosixPath(audio_relpath)
    label = _text_field(raw, "label")
    group = _text_field(raw, "group")
    source_type = _text_field(raw, "source_type")
    if (
        sample_id != PurePosixPath(member.name).stem
        or PurePosixPath(filename).name != filename
        or audio_path.is_absolute()
        or ".." in audio_path.parts
        or "\\" in audio_relpath
        or audio_path.name != filename
        or label not in KNOWN_LABELS
        or group not in KNOWN_GROUPS
        or source_type not in KNOWN_SOURCE_TYPES
    ):
        raise RuAsdCatalogError(f"Unexpected RuASD metadata values: {member.name!r}.")
    subset = _optional_text_field(raw, "subset")
    model = _optional_text_field(raw, "model")
    if group == "raw" and not subset:
        raise RuAsdCatalogError(f"Raw RuASD record has no source subset: {member.name!r}.")
    if group == "augmented" and source_type != "augmented_audio":
        raise RuAsdCatalogError(f"Unexpected augmented RuASD record: {member.name!r}.")
    if group == "raw" and label == "fake" and source_type != "tts":
        raise RuAsdCatalogError(f"Unexpected raw fake RuASD record: {member.name!r}.")
    if group == "raw" and label == "real" and source_type != "real_speech":
        raise RuAsdCatalogError(f"Unexpected raw real RuASD record: {member.name!r}.")
    return _RuAsdRecordSummary(
        label=label,
        group=group,
        source_type=source_type,
        subset=subset,
        model=model,
        speaker_group=_speaker_group(raw.get("speakers")),
        has_source_text=bool(
            _optional_text_field(raw, "true_lines")
            or _optional_text_field(raw, "transcription")
        ),
    )


def _text_field(raw: Mapping[object, object], name: str) -> str:
    value = _optional_text_field(raw, name)
    if not value:
        raise RuAsdCatalogError(f"RuASD metadata field is empty: {name!r}.")
    return value


def _optional_text_field(raw: Mapping[object, object], name: str) -> str:
    value = raw.get(name)
    return value.strip() if isinstance(value, str) else ""


def _speaker_group(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    normalized = str(value).strip()
    return None if normalized.lower() in UNKNOWN_SPEAKER_VALUES else normalized
