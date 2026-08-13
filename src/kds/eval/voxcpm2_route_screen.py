"""Project-history screen for the first official OpenBMB VoxCPM generator route."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest
from kds.data.voxcpm2 import VOXCPM2_MODEL_REVISION, VOXCPM2_SOURCE_REVISION


class VoxCPM2RouteScreenError(ValueError):
    """Raised when the manifest-history scope cannot be screened exactly."""


def _is_manifest(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise VoxCPM2RouteScreenError(f"Cannot inspect CSV {path}: {error}") from error
    return REQUIRED_FIELDS.issubset(header)


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except ValueError as error:
        raise VoxCPM2RouteScreenError(f"Manifest escapes project root: {path}") from error


def _voxcpm_match(row: ManifestRow) -> bool:
    values = (row.generator_family, row.generator_name, row.generator_version, row.voice_id)
    return any("voxcpm" in value.casefold() for value in values)


def screen_voxcpm2_project_history(
    *,
    project_root: Path,
    manifest_root: Path,
    created_at: str,
    artifact_receipt_path: Path,
) -> dict[str, object]:
    """Bind the complete manifest inventory and fail closed on any prior VoxCPM route."""

    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise VoxCPM2RouteScreenError("created_at must be an ISO-8601 timestamp.") from error
    project_root = project_root.resolve(strict=True)
    manifest_root = manifest_root.resolve(strict=True)
    artifact_receipt_path = artifact_receipt_path.resolve(strict=True)
    bindings: list[dict[str, object]] = []
    rows: list[ManifestRow] = []
    for path in sorted(manifest_root.rglob("*.csv")):
        if not _is_manifest(path):
            continue
        loaded = load_manifest(path)
        rows.extend(loaded)
        bindings.append(
            {
                "path": _relative(path, project_root),
                "sha256": sha256_file(path),
                "rows": len(loaded),
            }
        )
    if not bindings:
        raise VoxCPM2RouteScreenError("No valid manifests were found.")
    spoof_rows = [row for row in rows if row.label == "spoof"]
    matches = [row for row in spoof_rows if _voxcpm_match(row)]
    return {
        "schema_version": 1,
        "protocol_id": "voxcpm2-official-project-history-screen-v1",
        "created_at": created_at,
        "candidate_route": {
            "generator_family": "openbmb_voxcpm2_official_text_only",
            "model_revision": VOXCPM2_MODEL_REVISION,
            "source_revision": VOXCPM2_SOURCE_REVISION,
            "artifact_receipt": {
                "path": _relative(artifact_receipt_path, project_root),
                "sha256": sha256_file(artifact_receipt_path),
            },
        },
        "scope": {
            "manifest_root": _relative(manifest_root, project_root),
            "manifest_files": len(bindings),
            "manifest_rows": len(rows),
            "unique_sample_ids": len({row.sample_id for row in rows}),
            "spoof_rows": len(spoof_rows),
            "unique_spoof_sample_ids": len({row.sample_id for row in spoof_rows}),
            "unique_generator_families": len(
                {row.generator_family for row in spoof_rows if row.generator_family}
            ),
            "unique_generator_names": len(
                {row.generator_name for row in spoof_rows if row.generator_name}
            ),
            "unique_generator_versions": len(
                {row.generator_version for row in spoof_rows if row.generator_version}
            ),
            "unique_voice_ids": len({row.voice_id for row in spoof_rows if row.voice_id}),
            "manifest_bindings": bindings,
        },
        "voxcpm_history": {
            "matching_manifest_rows": len(matches),
            "matching_unique_sample_ids": len({row.sample_id for row in matches}),
            "matching_generator_families": dict(
                sorted(Counter(row.generator_family for row in matches).items())
            ),
            "match_rule": (
                "case-insensitive substring 'voxcpm' in generator_family, generator_name, "
                "generator_version, or voice_id"
            ),
        },
        "claims": {
            "exact_voxcpm_route_absent_from_manifest_history": not matches,
            "new_project_generator_family": not matches,
            "absolute_architecture_novelty": "not_claimed_historical_metadata_is_not_universal",
            "training_data_overlap": "unverified",
            "runtime_or_synthesis_performed": False,
            "detector_inference_performed": False,
            "detector_inference_authorized": False,
        },
    }
