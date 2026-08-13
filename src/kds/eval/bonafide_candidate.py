"""Deterministically select a bona-fide evaluation subset absent from configured roles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow, load_manifest
from kds.eval.candidate_exposure import (
    COMPARISON_FIELDS,
    CandidateExposureError,
    configured_role_scope,
)


@dataclass(frozen=True, slots=True)
class UnexposedBonafideCandidate:
    """A selection and the receipt content that explains every exclusion."""

    rows: tuple[ManifestRow, ...]
    receipt: dict[str, object]


def _relative_to_root(path: Path, project_root: Path, label: str) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise CandidateExposureError(f"{label} escapes the project root: {path}") from error


def _rows_fingerprint(rows: Sequence[ManifestRow]) -> str:
    canonical = [asdict(row) for row in sorted(rows, key=lambda row: row.sample_id)]
    return hashlib.sha256(
        json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def select_unexposed_bonafide_candidate(
    *,
    source_manifest: Path,
    source_split: str,
    project_root: Path,
    config_root: Path,
    created_at: str,
) -> UnexposedBonafideCandidate:
    """Exclude any candidate whose protected values occur in prior configured roles."""

    project_root = project_root.resolve(strict=True)
    config_root = config_root.resolve(strict=True)
    source_path = source_manifest.resolve(strict=True)
    _relative_to_root(config_root, project_root, "Configuration directory")
    _relative_to_root(source_path, project_root, "Source manifest")
    source_rows = load_manifest(source_path)
    candidates = [row for row in source_rows if row.split == source_split]
    if not candidates:
        raise CandidateExposureError(f"Source manifest has no rows in split {source_split!r}.")
    labels = {row.label for row in candidates}
    if labels != {"bonafide"}:
        raise CandidateExposureError(
            "Unexposed bona-fide candidate selection requires only bonafide source rows."
        )
    configured_rows, config_bindings, manifest_bindings = configured_role_scope(
        project_root, config_root
    )
    configured_values = {
        field: {getattr(row, field) for row in configured_rows} for field in COMPARISON_FIELDS
    }
    direct_overlap_fields_by_sample: dict[str, list[str]] = {}
    tainted_parent_groups: set[str] = set()
    tainted_speakers: set[str] = set()
    for row in candidates:
        overlap_fields = [
            field for field in COMPARISON_FIELDS if getattr(row, field) in configured_values[field]
        ]
        if overlap_fields:
            direct_overlap_fields_by_sample[row.sample_id] = overlap_fields
            tainted_parent_groups.add(row.parent_group_id)
            tainted_speakers.add(row.speaker_pseudo_id)
    included: list[ManifestRow] = []
    exclusions: list[tuple[str, list[str], bool]] = []
    for row in candidates:
        direct_overlap_fields = direct_overlap_fields_by_sample.get(row.sample_id, [])
        group_tainted = (
            row.parent_group_id in tainted_parent_groups
            or row.speaker_pseudo_id in tainted_speakers
        )
        if direct_overlap_fields or group_tainted:
            exclusions.append((row.sample_id, direct_overlap_fields, group_tainted))
        else:
            included.append(row)
    if not included:
        raise CandidateExposureError("No candidate rows remain after configured-role exclusion.")
    return UnexposedBonafideCandidate(
        rows=tuple(included),
        receipt={
            "schema_version": 1,
            "protocol_id": "unexposed-bonafide-candidate-selection-v1",
            "created_at": created_at,
            "source_manifest": {
                "path": _relative_to_root(source_path, project_root, "Source manifest"),
                "sha256": sha256_file(source_path),
                "selected_split": source_split,
                "candidate_rows_before_configured_role_exclusion": len(candidates),
                "candidate_rows_before_exclusion_sha256": _rows_fingerprint(candidates),
            },
            "selection_policy": {
                "kind": "exclude_configured_role_overlap_and_entire_candidate_group",
                "comparison_fields": list(COMPARISON_FIELDS),
                "group_fields": ["parent_group_id", "speaker_pseudo_id"],
                "metric_or_model_output_used": False,
            },
            "scope": {
                "configuration_directory": _relative_to_root(
                    config_root, project_root, "Configuration directory"
                ),
                "configuration_files": config_bindings,
                "configured_manifests": manifest_bindings,
                "configured_rows": len(configured_rows),
            },
            "excluded_rows": [
                {
                    "sample_id": sample_id,
                    "direct_overlap_fields": overlap_fields,
                    "group_tainted_by_direct_overlap": group_tainted,
                }
                for sample_id, overlap_fields, group_tainted in exclusions
            ],
            "exclusion_counts": {
                field: sum(field in overlap_fields for _, overlap_fields, _ in exclusions)
                for field in COMPARISON_FIELDS
            },
            "group_tainted_rows": sum(group_tainted for _, _, group_tainted in exclusions),
            "selected_rows": len(included),
            "selected_rows_sha256": _rows_fingerprint(included),
            "claims": {
                "selection_did_not_use_detector_inference": True,
                "selection_did_not_use_model_metrics_or_final_errors": True,
                "selected_rows_are_bonafide": True,
            },
        },
    )
