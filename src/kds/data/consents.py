"""Validation for a local, pseudonymous registry of product-data consents.

The registry deliberately contains no names, contacts, scans, signatures, or recordings.  The
mapping to the signed agreement stays in an access-controlled system outside the repository.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CONSENT_REGISTRY_FIELD_ORDER = (
    "consent_record_id",
    "speaker_pseudo_id",
    "language",
    "collection_version",
    "product_training_authorized",
    "synthetic_derivatives_authorized",
    "commercial_deployment_authorized",
    "status",
    "signed_at",
    "revoked_at",
)
CONSENT_REGISTRY_REQUIRED_FIELDS = frozenset(CONSENT_REGISTRY_FIELD_ORDER)
CONSENT_LANGUAGES = frozenset({"ru", "kk", "mixed"})
CONSENT_STATUSES = frozenset({"active", "revoked"})
_OPAQUE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,127}$")


class ConsentRegistryError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class ConsentRegistryEntry:
    consent_record_id: str
    speaker_pseudo_id: str
    language: str
    collection_version: str
    product_training_authorized: bool
    synthetic_derivatives_authorized: bool
    commercial_deployment_authorized: bool
    status: str
    signed_at: str
    revoked_at: str

    @classmethod
    def from_mapping(cls, mapping: dict[str, str], row_number: int) -> ConsentRegistryEntry:
        missing = sorted(CONSENT_REGISTRY_REQUIRED_FIELDS.difference(mapping))
        if missing:
            raise ConsentRegistryError(
                [f"Consent row {row_number}: missing columns: {', '.join(missing)}."]
            )

        def value(name: str) -> str:
            return (mapping.get(name) or "").strip()

        blank = sorted(
            field
            for field in CONSENT_REGISTRY_REQUIRED_FIELDS.difference({"revoked_at"})
            if not value(field)
        )
        issues: list[str] = []
        if blank:
            issues.append(f"Consent row {row_number}: blank required values: {', '.join(blank)}.")
        for field in ("consent_record_id", "speaker_pseudo_id"):
            if value(field) and _OPAQUE_ID.fullmatch(value(field)) is None:
                issues.append(
                    f"Consent row {row_number}: {field} must be an opaque portable identifier."
                )

        language = value("language").lower()
        if language not in CONSENT_LANGUAGES:
            issues.append(
                f"Consent row {row_number}: language must be one of {sorted(CONSENT_LANGUAGES)}."
            )
        status = value("status").lower()
        if status not in CONSENT_STATUSES:
            issues.append(
                f"Consent row {row_number}: status must be one of {sorted(CONSENT_STATUSES)}."
            )

        bool_values: dict[str, bool] = {}
        for field in (
            "product_training_authorized",
            "synthetic_derivatives_authorized",
            "commercial_deployment_authorized",
        ):
            raw_value = value(field).lower()
            if raw_value not in {"true", "false"}:
                issues.append(f"Consent row {row_number}: {field} must be true or false.")
            bool_values[field] = raw_value == "true"

        for field in ("signed_at", "revoked_at"):
            timestamp = value(field)
            if not timestamp:
                continue
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                issues.append(f"Consent row {row_number}: {field} must be an ISO-8601 timestamp.")
        revoked_at = value("revoked_at")
        if status == "active" and revoked_at:
            issues.append(f"Consent row {row_number}: active consent must not have revoked_at.")
        if status == "revoked" and not revoked_at:
            issues.append(f"Consent row {row_number}: revoked consent requires revoked_at.")
        if issues:
            raise ConsentRegistryError(issues)

        return cls(
            consent_record_id=value("consent_record_id"),
            speaker_pseudo_id=value("speaker_pseudo_id"),
            language=language,
            collection_version=value("collection_version"),
            product_training_authorized=bool_values["product_training_authorized"],
            synthetic_derivatives_authorized=bool_values["synthetic_derivatives_authorized"],
            commercial_deployment_authorized=bool_values["commercial_deployment_authorized"],
            status=status,
            signed_at=value("signed_at"),
            revoked_at=revoked_at,
        )


def load_consent_registry(path: Path) -> list[ConsentRegistryEntry]:
    """Load the local registry and reject duplicate active speaker records."""

    if not path.is_file():
        raise ConsentRegistryError([f"Consent registry does not exist: {path}"])
    entries: list[ConsentRegistryEntry] = []
    issues: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise ConsentRegistryError(["Consent registry has no header."])
        missing = sorted(CONSENT_REGISTRY_REQUIRED_FIELDS.difference(reader.fieldnames))
        if missing:
            raise ConsentRegistryError(
                [f"Consent registry missing required columns: {', '.join(missing)}."]
            )
        for row_number, mapping in enumerate(reader, start=2):
            try:
                entries.append(ConsentRegistryEntry.from_mapping(mapping, row_number))
            except ConsentRegistryError as error:
                issues.extend(error.issues)

    if not entries and not issues:
        issues.append("Consent registry contains no data rows.")
    active_speakers: set[str] = set()
    for entry in entries:
        if entry.status != "active":
            continue
        if entry.speaker_pseudo_id in active_speakers:
            issues.append(
                f"Duplicate active consent for speaker_pseudo_id: {entry.speaker_pseudo_id!r}."
            )
        active_speakers.add(entry.speaker_pseudo_id)
    if issues:
        raise ConsentRegistryError(issues)
    return entries


def product_eligible_speaker_ids(entries: Iterable[ConsentRegistryEntry]) -> frozenset[str]:
    """Return active speakers whose consent expressly covers the planned product corpus use."""

    issues: list[str] = []
    eligible: set[str] = set()
    for entry in entries:
        if entry.status != "active":
            continue
        missing_scopes = [
            field
            for field, is_authorized in (
                ("product_training_authorized", entry.product_training_authorized),
                ("synthetic_derivatives_authorized", entry.synthetic_derivatives_authorized),
                ("commercial_deployment_authorized", entry.commercial_deployment_authorized),
            )
            if not is_authorized
        ]
        if missing_scopes:
            issues.append(
                f"Active speaker {entry.speaker_pseudo_id!r} lacks required product scopes: "
                + ", ".join(missing_scopes)
                + "."
            )
            continue
        eligible.add(entry.speaker_pseudo_id)
    if issues:
        raise ConsentRegistryError(issues)
    return frozenset(eligible)
