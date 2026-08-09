from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

MANIFEST_FIELD_ORDER = (
    "sample_id",
    "relative_path",
    "sha256",
    "split",
    "label",
    "language",
    "code_switch",
    "parent_group_id",
    "source_name",
    "source_license",
    "rights_basis",
    "speaker_pseudo_id",
    "text_id",
    "text_hash",
    "duration_s",
    "generator_family",
    "generator_name",
    "generator_version",
    "voice_id",
    "clone_consent_id",
    "device",
    "capture_route",
    "original_sr",
    "codec",
    "augmentation_chain",
    "augmentation_seed",
    "created_at",
)
REQUIRED_FIELDS = frozenset(MANIFEST_FIELD_ORDER)
SPLITS = frozenset({"train", "dev", "test", "ood"})
LABELS = frozenset({"bonafide", "spoof"})
LANGUAGES = frozenset({"ru", "kk", "mixed", "other"})
OPTIONAL_EMPTY_FIELDS = frozenset(
    {
        "generator_family",
        "generator_name",
        "generator_version",
        "voice_id",
        "clone_consent_id",
        "augmentation_chain",
        "augmentation_seed",
    }
)


class ManifestError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class ManifestRow:
    sample_id: str
    relative_path: str
    sha256: str
    split: str
    label: str
    language: str
    code_switch: str
    parent_group_id: str
    source_name: str
    source_license: str
    rights_basis: str
    speaker_pseudo_id: str
    text_id: str
    text_hash: str
    duration_s: float
    generator_family: str
    generator_name: str
    generator_version: str
    voice_id: str
    clone_consent_id: str
    device: str
    capture_route: str
    original_sr: int
    codec: str
    augmentation_chain: str
    augmentation_seed: str
    created_at: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], row_number: int) -> ManifestRow:
        missing = sorted(REQUIRED_FIELDS.difference(mapping))
        if missing:
            raise ManifestError([f"Row {row_number}: missing columns: {', '.join(missing)}."])

        def value(name: str) -> str:
            return (mapping.get(name) or "").strip()

        mandatory = REQUIRED_FIELDS.difference(OPTIONAL_EMPTY_FIELDS)
        blank = sorted(name for name in mandatory if not value(name))
        if blank:
            raise ManifestError([f"Row {row_number}: blank required values: {', '.join(blank)}."])

        sha256 = value("sha256").lower()
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ManifestError(
                [f"Row {row_number}: sha256 must be a 64-character lowercase hex digest."]
            )

        relative_path = value("relative_path")
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or "\\" in relative_path
            or relative_path in {"", "."}
        ):
            raise ManifestError(
                [
                    f"Row {row_number}: relative_path must be a portable path below the audio root."
                ]
            )

        split = value("split")
        label = value("label")
        language = value("language")
        issues: list[str] = []
        if split not in SPLITS:
            issues.append(f"Row {row_number}: split must be one of {sorted(SPLITS)}.")
        if label not in LABELS:
            issues.append(f"Row {row_number}: label must be one of {sorted(LABELS)}.")
        if language not in LANGUAGES:
            issues.append(f"Row {row_number}: language must be one of {sorted(LANGUAGES)}.")
        if language == "other" and split != "ood":
            issues.append(f"Row {row_number}: language=other is permitted only in the ood split.")
        if value("code_switch").lower() not in {"true", "false", "unknown"}:
            issues.append(f"Row {row_number}: code_switch must be true, false, or unknown.")
        try:
            duration_s = float(value("duration_s"))
            if duration_s <= 0:
                issues.append(f"Row {row_number}: duration_s must be positive.")
        except ValueError:
            duration_s = 0.0
            issues.append(f"Row {row_number}: duration_s must be numeric.")
        try:
            original_sr = int(value("original_sr"))
            if original_sr <= 0:
                issues.append(f"Row {row_number}: original_sr must be positive.")
        except ValueError:
            original_sr = 0
            issues.append(f"Row {row_number}: original_sr must be an integer.")
        try:
            datetime.fromisoformat(value("created_at").replace("Z", "+00:00"))
        except ValueError:
            issues.append(f"Row {row_number}: created_at must be an ISO-8601 timestamp.")

        generator_fields = ("generator_family", "generator_name", "generator_version", "voice_id")
        if label == "spoof":
            absent = [field for field in generator_fields if not value(field)]
            if absent:
                issues.append(
                    f"Row {row_number}: spoof requires provenance fields: {', '.join(absent)}."
                )
        elif label == "bonafide":
            present = [field for field in generator_fields if value(field)]
            if present:
                issues.append(
                    f"Row {row_number}: bonafide must not contain generator fields: "
                    f"{', '.join(present)}."
                )
        if issues:
            raise ManifestError(issues)

        return cls(
            sample_id=value("sample_id"),
            relative_path=relative_path,
            sha256=sha256,
            split=split,
            label=label,
            language=language,
            code_switch=value("code_switch").lower(),
            parent_group_id=value("parent_group_id"),
            source_name=value("source_name"),
            source_license=value("source_license"),
            rights_basis=value("rights_basis"),
            speaker_pseudo_id=value("speaker_pseudo_id"),
            text_id=value("text_id"),
            text_hash=value("text_hash"),
            duration_s=duration_s,
            generator_family=value("generator_family"),
            generator_name=value("generator_name"),
            generator_version=value("generator_version"),
            voice_id=value("voice_id"),
            clone_consent_id=value("clone_consent_id"),
            device=value("device"),
            capture_route=value("capture_route"),
            original_sr=original_sr,
            codec=value("codec"),
            augmentation_chain=value("augmentation_chain"),
            augmentation_seed=value("augmentation_seed"),
            created_at=value("created_at"),
        )


def load_manifest(path: Path) -> list[ManifestRow]:
    if not path.is_file():
        raise ManifestError([f"Manifest does not exist: {path}"])
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise ManifestError(["Manifest has no header."])
        missing = sorted(REQUIRED_FIELDS.difference(reader.fieldnames))
        if missing:
            raise ManifestError([f"Manifest missing required columns: {', '.join(missing)}."])
        rows: list[ManifestRow] = []
        issues: list[str] = []
        for row_number, mapping in enumerate(reader, start=2):
            try:
                rows.append(ManifestRow.from_mapping(mapping, row_number))
            except ManifestError as error:
                issues.extend(error.issues)
    if not rows and not issues:
        issues.append("Manifest contains no data rows.")
    if issues:
        raise ManifestError(issues)
    return rows


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    """Write a validated manifest once, refusing accidental replacement of an existing file."""

    rows = list(rows)
    validate_manifest(rows)
    if path.exists():
        raise ManifestError([f"Refusing to overwrite existing manifest: {path}"])
    if not path.parent.is_dir():
        raise ManifestError([f"Manifest output directory does not exist: {path.parent}"])
    with path.open("x", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=MANIFEST_FIELD_ORDER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _cross_split_values(rows: Iterable[ManifestRow], field: str) -> list[str]:
    splits_by_value: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_value[getattr(row, field)].add(row.split)
    return [
        f"Leakage: {field}={value!r} appears in multiple splits: {', '.join(sorted(splits))}."
        for value, splits in sorted(splits_by_value.items())
        if len(splits) > 1
    ]


def validate_manifest(rows: Iterable[ManifestRow], require_ood_generator: bool = False) -> None:
    rows = list(rows)
    issues: list[str] = []

    for field in ("sample_id", "sha256"):
        seen: set[str] = set()
        for row in rows:
            value = getattr(row, field)
            if value in seen:
                issues.append(f"Duplicate {field}: {value!r}.")
            seen.add(value)

    for field in ("parent_group_id", "speaker_pseudo_id", "text_hash"):
        issues.extend(_cross_split_values(rows, field))

    if require_ood_generator:
        ood_families = {
            row.generator_family for row in rows if row.split == "ood" and row.label == "spoof"
        }
        if not ood_families:
            issues.append("No spoof generator family is assigned to the ood split.")
        seen_elsewhere = {
            row.generator_family
            for row in rows
            if row.split != "ood" and row.label == "spoof"
        }
        overlapping = sorted(ood_families.intersection(seen_elsewhere))
        if overlapping:
            issues.append(
                "OOD generator families also occur in train/dev/test: "
                + ", ".join(overlapping)
                + "."
            )

    if issues:
        raise ManifestError(issues)
