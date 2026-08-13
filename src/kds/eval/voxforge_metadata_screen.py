"""Pre-extraction project-exposure screen for the pinned VoxForge Russian archive."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest
from kds.data.voxforge import VOXFORGE_RU_SOURCE_ID, VoxForgeRuRecord
from kds.eval.candidate_exposure import CandidateExposureError, configured_role_scope

METADATA_COMPARISON_FIELDS = (
    "sample_id",
    "prompt_text_hash",
    "original_prompt_text_hash",
    "parent_group_id",
    "speaker_pseudo_id",
)


@dataclass(frozen=True, slots=True)
class VoxForgeMetadataIdentity:
    """One reproducible identity available before project extraction."""

    sample_id: str
    prompt_text_hash: str
    original_prompt_text_hash: str
    parent_group_id: str
    speaker_pseudo_id: str


@dataclass(frozen=True, slots=True)
class VoxForgeMetadataScreen:
    """Source-wide capacity after strict whole-contributor-group exclusion."""

    identities: tuple[VoxForgeMetadataIdentity, ...]
    surviving: tuple[VoxForgeMetadataIdentity, ...]
    receipt: dict[str, object]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def voxforge_metadata_identity(record: VoxForgeRuRecord) -> VoxForgeMetadataIdentity:
    """Construct privacy-preserving source keys without exposing contributor aliases."""

    group = f"{VOXFORGE_RU_SOURCE_ID}:contributor:{_hash(record.contributor_alias)}"
    return VoxForgeMetadataIdentity(
        sample_id=f"{VOXFORGE_RU_SOURCE_ID}:{record.submission_id}:{record.prompt_id}",
        prompt_text_hash=_hash(record.prompt_text),
        original_prompt_text_hash=_hash(record.original_prompt_text),
        parent_group_id=group,
        speaker_pseudo_id=group,
    )


def _relative_to_root(path: Path, project_root: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except ValueError as error:
        raise CandidateExposureError(f"{label} escapes the project root: {path}") from error


def _manifest_like_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            fields = next(csv.reader(file_handle), [])
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CandidateExposureError(
            f"Cannot inspect manifest inventory file {path}: {error}"
        ) from error
    return REQUIRED_FIELDS.issubset(fields)


def _load_inventory(
    *, project_root: Path, manifest_root: Path
) -> tuple[list[ManifestRow], list[dict[str, object]]]:
    rows: list[ManifestRow] = []
    bindings: list[dict[str, object]] = []
    for path in sorted(manifest_root.rglob("*.csv")):
        if not _manifest_like_csv(path):
            continue
        loaded = load_manifest(path)
        rows.extend(loaded)
        bindings.append(
            {
                "path": _relative_to_root(path, project_root, "Manifest inventory"),
                "sha256": sha256_file(path),
                "rows": len(loaded),
            }
        )
    if not bindings:
        raise CandidateExposureError("VoxForge metadata screen found no valid manifest inventory.")
    return rows, bindings


def _values(rows: Sequence[ManifestRow]) -> dict[str, set[str]]:
    text_hashes = {row.text_hash for row in rows}
    return {
        "sample_id": {row.sample_id for row in rows},
        "prompt_text_hash": text_hashes,
        "original_prompt_text_hash": text_hashes,
        "parent_group_id": {row.parent_group_id for row in rows},
        "speaker_pseudo_id": {row.speaker_pseudo_id for row in rows},
    }


def _identity_values(identity: VoxForgeMetadataIdentity) -> dict[str, str]:
    return {
        "sample_id": identity.sample_id,
        "prompt_text_hash": identity.prompt_text_hash,
        "original_prompt_text_hash": identity.original_prompt_text_hash,
        "parent_group_id": identity.parent_group_id,
        "speaker_pseudo_id": identity.speaker_pseudo_id,
    }


def screen_voxforge_ru_metadata(
    *,
    records: Sequence[VoxForgeRuRecord],
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
    created_at: str,
) -> VoxForgeMetadataScreen:
    """Fail closed on any exact historic identity overlap and taint the whole contributor group."""

    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandidateExposureError("created_at must be an ISO-8601 timestamp.") from error
    if not records:
        raise CandidateExposureError("VoxForge metadata screen has no source records.")
    project_root = project_root.resolve(strict=True)
    config_root = config_root.resolve(strict=True)
    manifest_root = manifest_root.resolve(strict=True)
    _relative_to_root(config_root, project_root, "Configuration directory")
    _relative_to_root(manifest_root, project_root, "Manifest inventory directory")

    identities = tuple(voxforge_metadata_identity(record) for record in records)
    sample_ids = [identity.sample_id for identity in identities]
    if len(sample_ids) != len(set(sample_ids)):
        raise CandidateExposureError("VoxForge metadata screen has duplicate sample IDs.")

    configured_rows, config_bindings, configured_manifest_bindings = configured_role_scope(
        project_root, config_root
    )
    inventory_rows, inventory_bindings = _load_inventory(
        project_root=project_root, manifest_root=manifest_root
    )
    configured_values = _values(configured_rows)
    inventory_values = _values(inventory_rows)
    configured_overlap_rows: Counter[str] = Counter()
    inventory_overlap_rows: Counter[str] = Counter()
    tainted_groups: set[str] = set()

    for identity in identities:
        values = _identity_values(identity)
        configured_overlap = [
            field
            for field in METADATA_COMPARISON_FIELDS
            if values[field] in configured_values[field]
        ]
        inventory_overlap = [
            field
            for field in METADATA_COMPARISON_FIELDS
            if values[field] in inventory_values[field]
        ]
        configured_overlap_rows.update(configured_overlap)
        inventory_overlap_rows.update(inventory_overlap)
        if configured_overlap or inventory_overlap:
            tainted_groups.add(identity.parent_group_id)

    surviving = tuple(
        identity for identity in identities if identity.parent_group_id not in tainted_groups
    )
    if not surviving:
        raise CandidateExposureError(
            "No VoxForge records remain after strict historical contributor-group exclusion."
        )
    source_groups = {identity.parent_group_id for identity in identities}
    surviving_groups = {identity.parent_group_id for identity in surviving}
    source_texts = {identity.prompt_text_hash for identity in identities}
    surviving_texts = {identity.prompt_text_hash for identity in surviving}
    return VoxForgeMetadataScreen(
        identities=identities,
        surviving=surviving,
        receipt={
            "schema_version": 1,
            "protocol_id": "voxforge-ru-mdc-metadata-exposure-screen-v1",
            "created_at": created_at,
            "candidate_state": "source-wide metadata screen only; no WAV extraction or selection",
            "source": {
                "source_id": VOXFORGE_RU_SOURCE_ID,
                "metadata_records": len(identities),
                "contributor_groups": len(source_groups),
                "canonical_prompt_text_groups": len(source_texts),
                "metadata_comparison_fields": list(METADATA_COMPARISON_FIELDS),
                "unavailable_pre_extraction_field": "sha256",
            },
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
                "inventory_rows": len(inventory_rows),
            },
            "direct_overlap_record_counts": {
                "configured_roles": {
                    field: configured_overlap_rows[field] for field in METADATA_COMPARISON_FIELDS
                },
                "manifest_inventory": {
                    field: inventory_overlap_rows[field] for field in METADATA_COMPARISON_FIELDS
                },
            },
            "strict_group_exclusion": {
                "rule": (
                    "Any exact sample, transcript-layer, parent-group or speaker-pseudo-ID "
                    "overlap taints the complete source-provided contributor group."
                ),
                "tainted_contributor_groups": len(tainted_groups),
                "surviving_records": len(surviving),
                "surviving_contributor_groups": len(surviving_groups),
                "surviving_canonical_prompt_text_groups": len(surviving_texts),
                "excluded_records": len(identities) - len(surviving),
            },
            "claims": {
                "metadata_identities_absent_from_historical_project_scope": not tainted_groups,
                "source_independent": False,
                "speaker_independent": False,
                "candidate_selection_performed": False,
                "audio_extraction_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
            },
        },
    )
