"""Fail-closed project-exposure audit for a selected evaluation subset.

The candidate source manifest is not itself evidence of prior model use.  Its
selected rows are therefore removed only from that one source manifest while
the audit still compares them with the source's remaining rows and every
other project manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest

COMPARISON_FIELDS = (
    "sample_id",
    "sha256",
    "text_hash",
    "parent_group_id",
    "speaker_pseudo_id",
)
INVENTORY_BLOCKING_FIELDS = ("sample_id", "sha256", "text_hash")


class CandidateExposureError(ValueError):
    """Raised when an evaluation subset cannot prove its project exposure scope."""


def _manifest_values(value: object) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "manifest":
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    result.append(cast(str, item["path"]))
            result.extend(_manifest_values(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_manifest_values(item))
    return result


def _relative_to_root(path: Path, project_root: Path, label: str) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise CandidateExposureError(f"{label} escapes the project root: {path}") from error


def _resolve_manifest(project_root: Path, config: Path, value: str) -> Path:
    try:
        path = (config.parent / value).resolve(strict=True)
    except OSError as error:
        raise CandidateExposureError(
            f"Cannot resolve manifest reference {value!r} from {config}."
        ) from error
    _relative_to_root(path, project_root, "Configured manifest")
    if path.suffix != ".csv":
        raise CandidateExposureError(f"Configured manifest is not a CSV: {value!r}.")
    return path


def _manifest_like_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            fields = next(csv.reader(file_handle), [])
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CandidateExposureError(
            f"Cannot inspect manifest inventory file {path}: {error}"
        ) from error
    return REQUIRED_FIELDS.issubset(fields)


def _rows_by_value(rows: Sequence[ManifestRow], field: str) -> dict[str, list[ManifestRow]]:
    result: dict[str, list[ManifestRow]] = {}
    for row in rows:
        result.setdefault(cast(str, getattr(row, field)), []).append(row)
    return result


def _overlaps(
    candidate: Sequence[ManifestRow], prior: Sequence[ManifestRow]
) -> dict[str, list[str]]:
    candidate_maps = {field: _rows_by_value(candidate, field) for field in COMPARISON_FIELDS}
    prior_maps = {field: _rows_by_value(prior, field) for field in COMPARISON_FIELDS}
    return {
        field: sorted(set(candidate_maps[field]).intersection(prior_maps[field]))
        for field in COMPARISON_FIELDS
    }


def _rows_fingerprint(rows: Sequence[ManifestRow]) -> str:
    canonical = [asdict(row) for row in sorted(rows, key=lambda row: row.sample_id)]
    encoded = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_source_binding(
    candidate_rows: Sequence[ManifestRow], source_rows: Sequence[ManifestRow]
) -> None:
    source_by_sample = {row.sample_id: row for row in source_rows}
    issues = [
        row.sample_id
        for row in candidate_rows
        if source_by_sample.get(row.sample_id) != row
    ]
    if issues:
        raise CandidateExposureError(
            "Candidate rows are absent from or changed against the declared source manifest: "
            + ", ".join(sorted(issues))
            + "."
        )


def _require_related_source_binding(
    candidate_rows: Sequence[ManifestRow], source_rows: Sequence[ManifestRow], source_path: Path
) -> None:
    source_by_sample = {row.sample_id: row for row in source_rows}
    fields = (
        "text_hash",
        "parent_group_id",
        "speaker_pseudo_id",
        "source_name",
        "label",
        "language",
    )
    issues = [
        row.sample_id
        for row in candidate_rows
        if (
            (related := source_by_sample.get(row.sample_id)) is None
            or any(getattr(related, field) != getattr(row, field) for field in fields)
        )
    ]
    if issues:
        raise CandidateExposureError(
            f"Related source manifest {source_path} cannot prove candidate lineage for: "
            + ", ".join(sorted(issues))
            + "."
        )


def configured_role_scope(
    project_root: Path, config_root: Path
) -> tuple[list[ManifestRow], list[dict[str, object]], list[dict[str, object]]]:
    configs = sorted(config_root.glob("*.json"))
    if not configs:
        raise CandidateExposureError("Exposure audit found no research configs.")
    paths: set[Path] = set()
    config_bindings: list[dict[str, object]] = []
    for config in configs:
        try:
            raw: object = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CandidateExposureError(
                f"Cannot read research config {config}: {error}"
            ) from error
        values = _manifest_values(raw)
        config_bindings.append(
            {
                "path": _relative_to_root(config, project_root, "Research config"),
                "sha256": sha256_file(config),
                "manifest_references": len(values),
            }
        )
        paths.update(_resolve_manifest(project_root, config, value) for value in values)
    rows: list[ManifestRow] = []
    manifest_bindings: list[dict[str, object]] = []
    for path in sorted(paths):
        loaded = load_manifest(path)
        rows.extend(loaded)
        manifest_bindings.append(
            {
                "path": _relative_to_root(path, project_root, "Configured manifest"),
                "sha256": sha256_file(path),
                "rows": len(loaded),
            }
        )
    return rows, config_bindings, manifest_bindings


def _inventory_scope(
    *,
    project_root: Path,
    manifest_root: Path,
    candidate_path: Path,
    source_path: Path,
    related_source_paths: Sequence[Path],
    candidate_rows: Sequence[ManifestRow],
) -> tuple[list[ManifestRow], list[dict[str, object]], dict[Path, int]]:
    candidate_sample_ids = {row.sample_id for row in candidate_rows}
    rows: list[ManifestRow] = []
    bindings: list[dict[str, object]] = []
    candidate_source_paths = {source_path, *related_source_paths}
    source_rows_removed: dict[Path, int] = {}
    for path in sorted(manifest_root.rglob("*.csv")):
        if not _manifest_like_csv(path):
            continue
        resolved_path = path.resolve()
        if resolved_path == candidate_path and resolved_path not in candidate_source_paths:
            continue
        loaded = load_manifest(path)
        included = loaded
        removed = 0
        if resolved_path == source_path:
            _require_exact_source_binding(candidate_rows, loaded)
            included = [row for row in loaded if row.sample_id not in candidate_sample_ids]
            removed = len(loaded) - len(included)
            if removed != len(candidate_rows):
                raise CandidateExposureError(
                    "Declared source manifest did not remove exactly the selected candidate rows."
                )
        elif resolved_path in related_source_paths:
            _require_related_source_binding(candidate_rows, loaded, path)
            included = [row for row in loaded if row.sample_id not in candidate_sample_ids]
            removed = len(loaded) - len(included)
            if removed != len(candidate_rows):
                raise CandidateExposureError(
                    "Related source manifest did not remove exactly the selected candidate rows."
                )
        if resolved_path in candidate_source_paths:
            source_rows_removed[resolved_path] = removed
        rows.extend(included)
        bindings.append(
            {
                "path": _relative_to_root(path, project_root, "Inventory manifest"),
                "sha256": sha256_file(path),
                "rows": len(loaded),
                "candidate_rows_excluded": removed,
            }
        )
    if not bindings:
        raise CandidateExposureError("Exposure audit found no valid manifests in the inventory.")
    return rows, bindings, source_rows_removed


def audit_candidate_project_exposure(
    *,
    candidate_manifest: Path,
    candidate_split: str,
    source_manifest: Path,
    related_source_manifests: Sequence[Path] = (),
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
    created_at: str,
) -> dict[str, object]:
    """Build an immutable-ready receipt for a candidate's configured-role exposure."""

    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandidateExposureError("created_at must be an ISO-8601 timestamp.") from error
    project_root = project_root.resolve(strict=True)
    config_root = config_root.resolve(strict=True)
    manifest_root = manifest_root.resolve(strict=True)
    candidate_path = candidate_manifest.resolve(strict=True)
    source_path = source_manifest.resolve(strict=True)
    related_source_paths = tuple(
        path.resolve(strict=True) for path in related_source_manifests
    )
    if (
        source_path in related_source_paths
        or len(set(related_source_paths)) != len(related_source_paths)
    ):
        raise CandidateExposureError(
            "Related source manifests must be distinct from each other and source."
        )
    for path, label in (
        (config_root, "Configuration directory"),
        (manifest_root, "Manifest inventory directory"),
        (candidate_path, "Candidate manifest"),
        (source_path, "Candidate source manifest"),
        *((path, "Related candidate source manifest") for path in related_source_paths),
    ):
        _relative_to_root(path, project_root, label)
    candidate_all = load_manifest(candidate_path)
    candidate = [row for row in candidate_all if row.split == candidate_split]
    if not candidate:
        raise CandidateExposureError(
            f"Candidate manifest has no rows in split {candidate_split!r}."
        )
    source = load_manifest(source_path)
    _require_exact_source_binding(candidate, source)
    configured_rows, config_bindings, configured_manifest_bindings = configured_role_scope(
        project_root, config_root
    )
    inventory_rows, inventory_bindings, source_rows_removed = _inventory_scope(
        project_root=project_root,
        manifest_root=manifest_root,
        candidate_path=candidate_path,
        source_path=source_path,
        related_source_paths=related_source_paths,
        candidate_rows=candidate,
    )
    configured_overlaps = _overlaps(candidate, configured_rows)
    inventory_overlaps = _overlaps(candidate, inventory_rows)
    if any(configured_overlaps.values()) or any(
        inventory_overlaps[field] for field in INVENTORY_BLOCKING_FIELDS
    ):
        details: list[str] = []
        for scope, overlaps, fields in (
            ("configured", configured_overlaps, COMPARISON_FIELDS),
            ("inventory", inventory_overlaps, INVENTORY_BLOCKING_FIELDS),
        ):
            details.extend(
                f"{scope}.{field}={len(overlaps[field])}" for field in fields if overlaps[field]
            )
        raise CandidateExposureError(
            "Candidate overlaps an existing project role or non-candidate inventory row: "
            + ", ".join(details)
            + "."
        )
    candidate_sources = sorted({row.source_name for row in candidate})
    configured_sources = {row.source_name for row in configured_rows}
    return {
        "schema_version": 1,
        "protocol_id": "candidate-project-exposure-v1",
        "created_at": created_at,
        "candidate": {
            "manifest": {
                "path": _relative_to_root(candidate_path, project_root, "Candidate manifest"),
                "sha256": sha256_file(candidate_path),
            },
            "selected_split": candidate_split,
            "rows": len(candidate),
            "rows_sha256": _rows_fingerprint(candidate),
            "labels": sorted({row.label for row in candidate}),
            "languages": sorted({row.language for row in candidate}),
            "source_names": candidate_sources,
        },
        "candidate_source_manifests": [
            {
                "path": _relative_to_root(path, project_root, "Candidate source manifest"),
                "sha256": sha256_file(path),
                "binding": "exact" if path == source_path else "same_sample_text_group_lineage",
                "selected_candidate_rows_excluded_from_inventory": source_rows_removed[path],
            }
            for path in (source_path, *related_source_paths)
        ],
        "scope": {
            "configuration_directory": _relative_to_root(
                config_root, project_root, "Configuration directory"
            ),
            "configuration_files": config_bindings,
            "configured_manifests": configured_manifest_bindings,
            "configured_rows": len(configured_rows),
            "manifest_inventory_directory": _relative_to_root(
                manifest_root, project_root, "Manifest inventory directory"
            ),
            "inventory_manifests": inventory_bindings,
            "non_candidate_inventory_rows": len(inventory_rows),
            "comparison_fields": list(COMPARISON_FIELDS),
            "inventory_blocking_fields": list(INVENTORY_BLOCKING_FIELDS),
            "inventory_group_or_speaker_overlap_policy": (
                "disclose_only; inventory availability is not a configured model role"
            ),
        },
        "configured_role_overlap_counts": {
            field: len(values) for field, values in configured_overlaps.items()
        },
        "configured_role_overlaps": configured_overlaps,
        "inventory_overlap_counts": {
            field: len(values) for field, values in inventory_overlaps.items()
        },
        "inventory_overlaps": inventory_overlaps,
        "configured_source_name_overlap": sorted(
            set(candidate_sources).intersection(configured_sources)
        ),
        "claims": {
            "exact_assets_absent_from_prior_configured_roles": True,
            "exact_texts_absent_from_prior_configured_roles": True,
            "candidate_parent_groups_absent_from_prior_configured_roles": True,
            "candidate_speaker_pseudo_ids_absent_from_prior_configured_roles": True,
            "candidate_source_not_used_in_prior_configured_roles": (
                not set(candidate_sources).intersection(configured_sources)
            ),
            "candidate_rows_excluded_only_from_declared_source_manifests": True,
            "exact_assets_absent_from_non_candidate_inventory": not any(
                inventory_overlaps[field] for field in ("sample_id", "sha256")
            ),
            "exact_texts_absent_from_non_candidate_inventory": not inventory_overlaps["text_hash"],
            "inventory_parent_group_overlap_disclosed": bool(inventory_overlaps["parent_group_id"]),
            "inventory_speaker_pseudo_id_overlap_disclosed": bool(
                inventory_overlaps["speaker_pseudo_id"]
            ),
            "detector_inference_performed": False,
            "detector_inference_authorized": False,
        },
    }
