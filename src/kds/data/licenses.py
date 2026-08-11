"""Validation for the auditable local data-source license ledger."""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from kds.data.manifest import ManifestError, ManifestRow, validate_manifest

LICENSE_LEDGER_FIELD_ORDER = (
    "source_id",
    "usage_scope",
    "train_dev_test_use",
    "ood_evaluation_use",
    "bonafide_group_provenance",
    "spoof_voice_group_provenance",
    "license",
    "source_url",
    "artifact_name",
    "expected_size_bytes",
    "last_modified_utc",
    "sha256",
    "rights_basis",
    "status",
    "notes",
)
LICENSE_LEDGER_REQUIRED_FIELDS = frozenset(LICENSE_LEDGER_FIELD_ORDER)
# ``owner_authorized_personal_research`` is a project-scope decision. It never replaces
# restrictions imposed by the source licence, datasheet, privacy law, or contributor consent.
APPROVED_LICENSE_STATUSES = frozenset({"verified", "owner_authorized_personal_research"})
PROTOCOL_USE_VALUES = frozenset({"product_allowed", "research_only", "prohibited"})
GROUP_PROVENANCE_VALUES = frozenset({"verified", "source_provided", "unknown", "not_applicable"})
TRAINING_PURPOSES = frozenset({"research", "product"})
TRAINING_SPLITS = ("train", "dev", "test")
PRODUCT_PROTOCOL_SPLITS = (*TRAINING_SPLITS, "ood")


class LicenseLedgerError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


class TrainingProtocolError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class LicenseLedgerEntry:
    source_id: str
    usage_scope: str
    train_dev_test_use: str
    ood_evaluation_use: str
    bonafide_group_provenance: str
    spoof_voice_group_provenance: str
    license: str
    source_url: str
    artifact_name: str
    expected_size_bytes: int
    last_modified_utc: str
    sha256: str
    rights_basis: str
    status: str
    notes: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str], row_number: int) -> LicenseLedgerEntry:
        missing = sorted(LICENSE_LEDGER_REQUIRED_FIELDS.difference(mapping))
        if missing:
            raise LicenseLedgerError(
                [f"Ledger row {row_number}: missing columns: {', '.join(missing)}."]
            )

        def value(name: str) -> str:
            return (mapping.get(name) or "").strip()

        required_nonempty = LICENSE_LEDGER_REQUIRED_FIELDS.difference(
            {"last_modified_utc", "sha256"}
        )
        blank = sorted(name for name in required_nonempty if not value(name))
        issues: list[str] = []
        if blank:
            issues.append(f"Ledger row {row_number}: blank required values: {', '.join(blank)}.")

        try:
            expected_size_bytes = int(value("expected_size_bytes"))
            if expected_size_bytes <= 0:
                issues.append(f"Ledger row {row_number}: expected_size_bytes must be positive.")
        except ValueError:
            expected_size_bytes = 0
            issues.append(f"Ledger row {row_number}: expected_size_bytes must be an integer.")

        last_modified_utc = value("last_modified_utc")
        if last_modified_utc:
            try:
                datetime.fromisoformat(last_modified_utc.replace("Z", "+00:00"))
            except ValueError:
                issues.append(
                    f"Ledger row {row_number}: last_modified_utc must be an ISO-8601 timestamp."
                )

        sha256 = value("sha256").lower()
        if sha256 and (
            len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256)
        ):
            issues.append(f"Ledger row {row_number}: sha256 must be a 64-character hex digest.")

        for field_name in ("train_dev_test_use", "ood_evaluation_use"):
            field_value = value(field_name).lower()
            if field_value not in PROTOCOL_USE_VALUES:
                issues.append(
                    f"Ledger row {row_number}: {field_name} must be one of "
                    f"{sorted(PROTOCOL_USE_VALUES)}."
                )
        for field_name in ("bonafide_group_provenance", "spoof_voice_group_provenance"):
            field_value = value(field_name).lower()
            if field_value not in GROUP_PROVENANCE_VALUES:
                issues.append(
                    f"Ledger row {row_number}: {field_name} must be one of "
                    f"{sorted(GROUP_PROVENANCE_VALUES)}."
                )

        status = value("status").lower()
        if status in APPROVED_LICENSE_STATUSES and not sha256:
            issues.append(f"Ledger row {row_number}: approved status requires an archive SHA-256.")
        if issues:
            raise LicenseLedgerError(issues)

        return cls(
            source_id=value("source_id"),
            usage_scope=value("usage_scope"),
            train_dev_test_use=value("train_dev_test_use").lower(),
            ood_evaluation_use=value("ood_evaluation_use").lower(),
            bonafide_group_provenance=value("bonafide_group_provenance").lower(),
            spoof_voice_group_provenance=value("spoof_voice_group_provenance").lower(),
            license=value("license"),
            source_url=value("source_url"),
            artifact_name=value("artifact_name"),
            expected_size_bytes=expected_size_bytes,
            last_modified_utc=last_modified_utc,
            sha256=sha256,
            rights_basis=value("rights_basis"),
            status=status,
            notes=value("notes"),
        )


def load_license_ledger(path: Path) -> dict[str, LicenseLedgerEntry]:
    """Load a ledger and reject malformed or duplicate source records."""

    if not path.is_file():
        raise LicenseLedgerError([f"License ledger does not exist: {path}"])
    entries: dict[str, LicenseLedgerEntry] = {}
    issues: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise LicenseLedgerError(["License ledger has no header."])
        missing = sorted(LICENSE_LEDGER_REQUIRED_FIELDS.difference(reader.fieldnames))
        if missing:
            raise LicenseLedgerError(
                [f"License ledger missing required columns: {', '.join(missing)}."]
            )
        for row_number, mapping in enumerate(reader, start=2):
            try:
                entry = LicenseLedgerEntry.from_mapping(mapping, row_number)
            except LicenseLedgerError as error:
                issues.extend(error.issues)
                continue
            if entry.source_id in entries:
                issues.append(f"Duplicate license ledger source_id: {entry.source_id!r}.")
                continue
            entries[entry.source_id] = entry
    if not entries and not issues:
        issues.append("License ledger contains no data rows.")
    if issues:
        raise LicenseLedgerError(issues)
    return entries


def write_license_ledger_snapshot(
    path: Path,
    entries: Mapping[str, LicenseLedgerEntry],
    *,
    source_ids: Iterable[str],
) -> tuple[str, ...]:
    """Write a deterministic, minimal and immutable ledger for one frozen protocol."""

    selected_ids = tuple(sorted(set(source_ids)))
    missing = tuple(source_id for source_id in selected_ids if source_id not in entries)
    if not selected_ids:
        raise LicenseLedgerError(["A license ledger snapshot needs at least one source_id."])
    if missing:
        raise LicenseLedgerError(
            ["License ledger snapshot sources are missing: " + ", ".join(missing) + "."]
        )
    if path.exists() or not path.parent.is_dir():
        raise LicenseLedgerError([f"Unsafe license ledger snapshot destination: {path}"])

    try:
        with tempfile.TemporaryDirectory(prefix="kds-ledger-snapshot-", dir=path.parent) as stage:
            staged = Path(stage) / path.name
            with staged.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=LICENSE_LEDGER_FIELD_ORDER,
                    lineterminator="\n",
                )
                writer.writeheader()
                for source_id in selected_ids:
                    writer.writerow(asdict(entries[source_id]))
            load_license_ledger(staged)
            shutil.move(staged, path)
    except OSError as error:
        raise LicenseLedgerError([f"Cannot write license ledger snapshot: {path}"]) from error
    return selected_ids


def validate_manifest_licenses(
    rows: Iterable[ManifestRow], ledger: Mapping[str, LicenseLedgerEntry]
) -> None:
    """Require every manifest source to be explicitly approved in the local ledger."""

    issues: list[str] = []
    for source_id in sorted({row.source_name for row in rows}):
        entry = ledger.get(source_id)
        if entry is None:
            issues.append(f"Manifest source {source_id!r} is missing from the license ledger.")
            continue
        if entry.status not in APPROVED_LICENSE_STATUSES:
            issues.append(
                f"Manifest source {source_id!r} is not approved in the license ledger "
                f"(status={entry.status!r})."
            )
    if issues:
        raise LicenseLedgerError(issues)


@dataclass(frozen=True, slots=True)
class TrainingProtocolReport:
    """The explicit protocol classification recorded with a training checkpoint."""

    purpose: str
    split_counts: dict[str, int]
    source_ids: tuple[str, ...]


def validate_training_protocol(
    rows: Iterable[ManifestRow],
    ledger: Mapping[str, LicenseLedgerEntry],
    *,
    purpose: str,
) -> TrainingProtocolReport:
    """Reject an underspecified binary training protocol before model training starts.

    A manifest field called ``speaker_pseudo_id`` is not proof that it represents a speaker.
    The source policy in the ledger must declare provenance separately.  A product protocol also
    needs an independent binary OOD split and verified spoof voice groups.
    """

    if purpose not in TRAINING_PURPOSES:
        raise TrainingProtocolError([f"purpose must be one of {sorted(TRAINING_PURPOSES)}."])

    rows = list(rows)
    try:
        validate_manifest(rows)
    except ManifestError as error:
        raise TrainingProtocolError(error.issues) from error
    try:
        validate_manifest_licenses(rows, ledger)
    except LicenseLedgerError as error:
        raise TrainingProtocolError(error.issues) from error

    required_splits = PRODUCT_PROTOCOL_SPLITS if purpose == "product" else TRAINING_SPLITS
    protocol_rows = [row for row in rows if row.split in required_splits]
    split_counts = {
        split: sum(row.split == split for row in protocol_rows) for split in required_splits
    }
    issues: list[str] = []
    for split, count in split_counts.items():
        if count == 0:
            issues.append(f"Protocol requires at least one row in split={split!r}.")
            continue
        labels = {row.label for row in protocol_rows if row.split == split}
        if labels != {"bonafide", "spoof"}:
            issues.append(f"split={split!r} must include both bonafide and spoof rows.")

    checked_use_keys: set[tuple[str, str]] = set()
    checked_group_keys: set[tuple[str, str]] = set()
    for row in protocol_rows:
        entry = ledger[row.source_name]
        use_field = "ood_evaluation_use" if row.split == "ood" else "train_dev_test_use"
        declared_use = getattr(entry, use_field)
        use_key = (row.source_name, use_field)
        if use_key not in checked_use_keys:
            checked_use_keys.add(use_key)
            if purpose == "research":
                if declared_use not in {"research_only", "product_allowed"}:
                    issues.append(
                        f"Source {row.source_name!r} is prohibited for {use_field} "
                        "in a research protocol."
                    )
            else:
                if declared_use != "product_allowed":
                    issues.append(
                        f"Source {row.source_name!r} is not product-allowed for {use_field} "
                        f"(value={declared_use!r})."
                    )
                if entry.status != "verified":
                    issues.append(
                        f"Source {row.source_name!r} needs ledger status='verified' "
                        "for product use."
                    )
                if entry.usage_scope != "commercial_clean":
                    issues.append(
                        f"Source {row.source_name!r} needs usage_scope='commercial_clean' "
                        "for product use."
                    )

        if purpose != "product":
            continue
        group_key = (row.source_name, row.label)
        if group_key in checked_group_keys:
            continue
        checked_group_keys.add(group_key)
        provenance_field = (
            "bonafide_group_provenance"
            if row.label == "bonafide"
            else "spoof_voice_group_provenance"
        )
        provenance = getattr(entry, provenance_field)
        if provenance != "verified":
            issues.append(
                f"Source {row.source_name!r} has no verified {provenance_field} "
                f"for label={row.label!r} (value={provenance!r})."
            )

    if purpose == "product":
        ood_families = {
            row.generator_family
            for row in protocol_rows
            if row.split == "ood" and row.label == "spoof"
        }
        if not ood_families:
            issues.append("No spoof generator family is assigned to the ood split.")
        seen_elsewhere = {
            row.generator_family
            for row in protocol_rows
            if row.split != "ood" and row.label == "spoof"
        }
        overlapping_families = sorted(ood_families.intersection(seen_elsewhere))
        if overlapping_families:
            issues.append(
                "OOD generator families also occur in train/dev/test: "
                + ", ".join(overlapping_families)
                + "."
            )
        voice_splits: dict[str, set[str]] = {}
        for row in protocol_rows:
            if (
                row.label != "spoof"
                or ledger[row.source_name].spoof_voice_group_provenance != "verified"
            ):
                continue
            voice_splits.setdefault(row.voice_id, set()).add(row.split)
        for voice_id, splits in sorted(voice_splits.items()):
            if len(splits) > 1:
                issues.append(
                    "Leakage: spoof voice_id="
                    f"{voice_id!r} appears in multiple splits: {', '.join(sorted(splits))}."
                )

    if issues:
        raise TrainingProtocolError(issues)
    return TrainingProtocolReport(
        purpose=purpose,
        split_counts=split_counts,
        source_ids=tuple(sorted({row.source_name for row in protocol_rows})),
    )
