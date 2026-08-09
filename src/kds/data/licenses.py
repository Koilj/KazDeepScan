"""Validation for the auditable local data-source license ledger."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kds.data.manifest import ManifestRow

LICENSE_LEDGER_FIELD_ORDER = (
    "source_id",
    "usage_scope",
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


class LicenseLedgerError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class LicenseLedgerEntry:
    source_id: str
    usage_scope: str
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
            issues.append(
                f"Ledger row {row_number}: blank required values: {', '.join(blank)}."
            )

        try:
            expected_size_bytes = int(value("expected_size_bytes"))
            if expected_size_bytes <= 0:
                issues.append(
                    f"Ledger row {row_number}: expected_size_bytes must be positive."
                )
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

        status = value("status").lower()
        if status in APPROVED_LICENSE_STATUSES and not sha256:
            issues.append(
                f"Ledger row {row_number}: approved status requires an archive SHA-256."
            )
        if issues:
            raise LicenseLedgerError(issues)

        return cls(
            source_id=value("source_id"),
            usage_scope=value("usage_scope"),
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
