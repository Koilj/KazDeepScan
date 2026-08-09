"""Explicit, source-disjoint protocols for personal-research experiments.

The regular training manifest protects against sample, text and group leakage.  That is not
enough when a model can learn corpus-specific artefacts.  A source-mixed matrix therefore keeps
the source corpus used for train, development selection and final evaluation disjoint, while
retaining the original split recorded by each source manifest.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kds.data.licenses import (
    APPROVED_LICENSE_STATUSES,
    LicenseLedgerEntry,
    LicenseLedgerError,
    validate_manifest_licenses,
)
from kds.data.manifest import SPLITS, ManifestError, ManifestRow, load_manifest, validate_manifest

MATRIX_SCHEMA_VERSION = 1
MATRIX_ROLES = ("train", "dev", "test")
_ALLOWED_SOURCE_SPLITS: Mapping[str, frozenset[str]] = {
    "train": frozenset({"train"}),
    "dev": frozenset({"dev"}),
    # A source explicitly marked OOD remains OOD. It may be the final untouched matrix test.
    "test": frozenset({"test", "ood"}),
}


class SourceMatrixError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class SourceMatrixRole:
    name: str
    manifest_path: Path
    source_split: str
    expected_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceMixedResearchMatrix:
    protocol_id: str
    roles: tuple[SourceMatrixRole, ...]


@dataclass(frozen=True, slots=True)
class SourceMatrixRoleReport:
    name: str
    manifest_path: str
    source_split: str
    rows: int
    label_counts: dict[str, int]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceMixedResearchMatrixReport:
    protocol_id: str
    purpose: str
    roles: tuple[SourceMatrixRoleReport, ...]

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted({source for role in self.roles for source in role.source_ids}))


def load_source_mixed_research_matrix(path: Path) -> SourceMixedResearchMatrix:
    """Load a versioned JSON matrix without silently accepting misspelled fields."""

    if not path.is_file():
        raise SourceMatrixError([f"Source matrix does not exist: {path}"])
    try:
        raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceMatrixError([f"Cannot read source matrix {path}: {error}"]) from error
    if not isinstance(raw_value, dict):
        raise SourceMatrixError(["Source matrix root must be a JSON object."])
    raw = cast(dict[str, object], raw_value)
    expected_keys = {"schema_version", "protocol_id", "purpose", "roles"}
    unknown_keys = sorted(set(raw).difference(expected_keys))
    missing_keys = sorted(expected_keys.difference(raw))
    if unknown_keys or missing_keys:
        issues: list[str] = []
        if missing_keys:
            issues.append("Source matrix missing fields: " + ", ".join(missing_keys) + ".")
        if unknown_keys:
            issues.append("Source matrix has unknown fields: " + ", ".join(unknown_keys) + ".")
        raise SourceMatrixError(issues)

    schema_version = raw["schema_version"]
    if schema_version != MATRIX_SCHEMA_VERSION:
        raise SourceMatrixError(
            [f"schema_version must be {MATRIX_SCHEMA_VERSION!r}, got {schema_version!r}."]
        )
    protocol_id = _required_string(raw, "protocol_id", "Source matrix")
    purpose = _required_string(raw, "purpose", "Source matrix")
    if purpose != "research":
        raise SourceMatrixError(
            ["Source-mixed matrices currently support purpose='research' only."]
        )
    raw_roles = raw["roles"]
    if not isinstance(raw_roles, list):
        raise SourceMatrixError(["Source matrix roles must be a JSON array."])

    roles = tuple(_parse_role(value, index, path.parent) for index, value in enumerate(raw_roles))
    role_names = [role.name for role in roles]
    if tuple(sorted(role_names)) != tuple(sorted(MATRIX_ROLES)):
        raise SourceMatrixError(
            [
                "Source matrix must define exactly these roles once: "
                + ", ".join(MATRIX_ROLES)
                + "."
            ]
        )
    return SourceMixedResearchMatrix(protocol_id=protocol_id, roles=roles)


def select_matrix_rows(matrix: SourceMixedResearchMatrix) -> dict[str, list[ManifestRow]]:
    """Load the source-owned split selected by every role in a validated matrix."""

    selections: dict[str, list[ManifestRow]] = {}
    for role in matrix.roles:
        rows = load_manifest(role.manifest_path)
        selections[role.name] = [row for row in rows if row.split == role.source_split]
    return selections


def validate_source_mixed_research_matrix(
    matrix: SourceMixedResearchMatrix, ledger: Mapping[str, LicenseLedgerEntry]
) -> SourceMixedResearchMatrixReport:
    """Validate source, label and rights isolation for a research experiment matrix."""

    issues: list[str] = []
    reports: list[SourceMatrixRoleReport] = []
    source_roles: dict[str, set[str]] = {}
    try:
        selections = select_matrix_rows(matrix)
    except ManifestError as error:
        raise SourceMatrixError(error.issues) from error

    # Corpus isolation complements, rather than replaces, the ordinary asset/group/text checks.
    # The selected source-owned splits must also be safe when considered as one experiment.
    combined_rows = [row for role in matrix.roles for row in selections[role.name]]
    try:
        validate_manifest(combined_rows)
    except ManifestError as error:
        issues.extend(error.issues)

    for role in matrix.roles:
        try:
            full_manifest = load_manifest(role.manifest_path)
            validate_manifest(full_manifest)
            rows = selections[role.name]
            if not rows:
                issues.append(
                    f"Matrix role={role.name!r} selects no rows with split={role.source_split!r}."
                )
                continue
            actual_source_ids = tuple(sorted({row.source_name for row in rows}))
            if actual_source_ids != role.expected_source_ids:
                issues.append(
                    f"Matrix role={role.name!r} expected sources "
                    f"{list(role.expected_source_ids)!r}, "
                    f"found {list(actual_source_ids)!r}."
                )
            labels = {row.label for row in rows}
            if labels != {"bonafide", "spoof"}:
                issues.append(
                    f"Matrix role={role.name!r} must include both bonafide and spoof rows."
                )
            try:
                validate_manifest_licenses(rows, ledger)
            except LicenseLedgerError as error:
                issues.extend(error.issues)

            use_field = (
                "ood_evaluation_use" if role.source_split == "ood" else "train_dev_test_use"
            )
            for source_id in actual_source_ids:
                source_roles.setdefault(source_id, set()).add(role.name)
                entry = ledger.get(source_id)
                if entry is None or entry.status not in APPROVED_LICENSE_STATUSES:
                    continue
                declared_use = getattr(entry, use_field)
                if declared_use not in {"research_only", "product_allowed"}:
                    issues.append(
                        f"Source {source_id!r} is prohibited for {use_field} in a research matrix."
                    )

            reports.append(
                SourceMatrixRoleReport(
                    name=role.name,
                    manifest_path=str(role.manifest_path),
                    source_split=role.source_split,
                    rows=len(rows),
                    label_counts={
                        label: sum(row.label == label for row in rows)
                        for label in ("bonafide", "spoof")
                    },
                    source_ids=actual_source_ids,
                )
            )
        except ManifestError as error:
            issues.extend(error.issues)

    for source_id, roles in sorted(source_roles.items()):
        if len(roles) > 1:
            issues.append(
                f"Source leakage: source_name={source_id!r} is used by multiple matrix roles: "
                + ", ".join(sorted(roles))
                + "."
            )
    if issues:
        raise SourceMatrixError(issues)
    return SourceMixedResearchMatrixReport(
        protocol_id=matrix.protocol_id,
        purpose="research",
        roles=tuple(reports),
    )


def _parse_role(value: object, index: int, base_directory: Path) -> SourceMatrixRole:
    label = f"Source matrix role {index + 1}"
    if not isinstance(value, dict):
        raise SourceMatrixError([f"{label} must be a JSON object."])
    raw = cast(dict[str, object], value)
    expected_keys = {"name", "manifest", "source_split", "expected_source_ids"}
    unknown_keys = sorted(set(raw).difference(expected_keys))
    missing_keys = sorted(expected_keys.difference(raw))
    if unknown_keys or missing_keys:
        issues: list[str] = []
        if missing_keys:
            issues.append(f"{label} missing fields: " + ", ".join(missing_keys) + ".")
        if unknown_keys:
            issues.append(f"{label} has unknown fields: " + ", ".join(unknown_keys) + ".")
        raise SourceMatrixError(issues)

    name = _required_string(raw, "name", label)
    if name not in MATRIX_ROLES:
        raise SourceMatrixError([f"{label} name must be one of {list(MATRIX_ROLES)!r}."])
    source_split = _required_string(raw, "source_split", label)
    if source_split not in SPLITS:
        raise SourceMatrixError([f"{label} source_split must be one of {sorted(SPLITS)!r}."])
    if source_split not in _ALLOWED_SOURCE_SPLITS[name]:
        raise SourceMatrixError(
            [
                f"{label} name={name!r} cannot select source split={source_split!r}; "
                f"allowed: {sorted(_ALLOWED_SOURCE_SPLITS[name])!r}."
            ]
        )
    manifest = _required_string(raw, "manifest", label)
    manifest_path = Path(manifest)
    if manifest_path.is_absolute():
        raise SourceMatrixError([f"{label} manifest path must be relative to the matrix file."])
    source_ids_value = raw["expected_source_ids"]
    if not isinstance(source_ids_value, list) or not source_ids_value:
        raise SourceMatrixError([f"{label} expected_source_ids must be a non-empty JSON array."])
    source_ids: list[str] = []
    for source_id in source_ids_value:
        if not isinstance(source_id, str) or not source_id.strip():
            raise SourceMatrixError(
                [f"{label} expected_source_ids must contain non-empty strings."]
            )
        source_ids.append(source_id.strip())
    if len(source_ids) != len(set(source_ids)):
        raise SourceMatrixError([f"{label} expected_source_ids must not contain duplicates."])
    return SourceMatrixRole(
        name=name,
        manifest_path=(base_directory / manifest_path).resolve(),
        source_split=source_split,
        expected_source_ids=tuple(sorted(source_ids)),
    )


def _required_string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw[name]
    if not isinstance(value, str) or not value.strip():
        raise SourceMatrixError([f"{label} field {name!r} must be a non-empty string."])
    return value.strip()
