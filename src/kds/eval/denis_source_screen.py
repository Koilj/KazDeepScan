"""Source-wide project-exposure screen for the pinned single-speaker Denis 1.0 archive."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.denis import DENIS_ARCHIVE_EXPECTED_SHA256, DENIS_SOURCE_ID, DenisRecord
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest
from kds.eval.candidate_exposure import CandidateExposureError, configured_role_scope

DENIS_HISTORICAL_GENERATOR_NAME = "piperTTS"
DENIS_HISTORICAL_GENERATOR_VERSION = "ru_RU-denis-medium"


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


def _is_historical_denis_route(row: ManifestRow) -> bool:
    return row.generator_name == DENIS_HISTORICAL_GENERATOR_NAME and (
        row.generator_version == DENIS_HISTORICAL_GENERATOR_VERSION
        or row.voice_id == DENIS_HISTORICAL_GENERATOR_VERSION
    )


def _route_summary(rows: Sequence[ManifestRow]) -> dict[str, object]:
    route_rows = [row for row in rows if _is_historical_denis_route(row)]
    unique_by_sample: dict[str, ManifestRow] = {}
    for row in route_rows:
        previous = unique_by_sample.get(row.sample_id)
        if previous is not None and (
            previous.split != row.split
            or previous.label != row.label
            or previous.text_hash != row.text_hash
        ):
            raise CandidateExposureError(
                f"Historical Denis route sample {row.sample_id!r} has conflicting manifest rows."
            )
        unique_by_sample[row.sample_id] = row
    unique_rows = list(unique_by_sample.values())
    return {
        "manifest_rows": len(route_rows),
        "unique_sample_ids": len(unique_rows),
        "splits_by_unique_sample_id": dict(
            sorted(Counter(row.split for row in unique_rows).items())
        ),
        "labels_by_unique_sample_id": dict(
            sorted(Counter(row.label for row in unique_rows).items())
        ),
    }


def _load_inventory(
    *, project_root: Path, manifest_root: Path
) -> tuple[list[ManifestRow], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[ManifestRow] = []
    bindings: list[dict[str, object]] = []
    route_bindings: list[dict[str, object]] = []
    for path in sorted(manifest_root.rglob("*.csv")):
        if not _manifest_like_csv(path):
            continue
        loaded = load_manifest(path)
        rows.extend(loaded)
        binding = {
            "path": _relative_to_root(path, project_root, "Manifest inventory"),
            "sha256": sha256_file(path),
            "rows": len(loaded),
        }
        bindings.append(binding)
        route_rows = sum(_is_historical_denis_route(row) for row in loaded)
        if route_rows:
            route_bindings.append({**binding, "historical_denis_route_rows": route_rows})
    if not bindings:
        raise CandidateExposureError("Denis source screen found no valid manifest inventory.")
    return rows, bindings, route_bindings


def _direct_overlaps(
    records: Sequence[DenisRecord], prior_rows: Sequence[ManifestRow]
) -> tuple[dict[str, int], dict[str, int]]:
    sample_ids = {row.sample_id for row in prior_rows}
    audio_hashes = {row.sha256 for row in prior_rows}
    text_hashes = {row.text_hash for row in prior_rows}
    candidate_counts = {
        "sample_id": sum(record.sample_id in sample_ids for record in records),
        "audio_sha256": sum(record.audio_sha256 in audio_hashes for record in records),
        "literal_text_sha256": sum(
            record.literal_text_sha256 in text_hashes for record in records
        ),
        "whitespace_canonical_text_sha256": sum(
            record.whitespace_canonical_text_sha256 in text_hashes for record in records
        ),
        "nfkc_whitespace_canonical_text_sha256": sum(
            record.nfkc_whitespace_canonical_text_sha256 in text_hashes for record in records
        ),
    }
    candidate_sample_ids = {record.sample_id for record in records}
    candidate_audio_hashes = {record.audio_sha256 for record in records}
    candidate_text_hashes = {
        text_hash
        for record in records
        for text_hash in (
            record.literal_text_sha256,
            record.whitespace_canonical_text_sha256,
            record.nfkc_whitespace_canonical_text_sha256,
        )
    }
    prior_counts = {
        "sample_id": sum(row.sample_id in candidate_sample_ids for row in prior_rows),
        "audio_sha256": sum(row.sha256 in candidate_audio_hashes for row in prior_rows),
        "any_candidate_text_sha256": sum(
            row.text_hash in candidate_text_hashes for row in prior_rows
        ),
    }
    return candidate_counts, prior_counts


def screen_denis_source_records(
    *,
    records: Sequence[DenisRecord],
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
    created_at: str,
    source_audit_receipt: dict[str, str],
) -> dict[str, object]:
    """Screen all exact source identities and disclose likely historical speaker lineage."""

    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandidateExposureError("created_at must be an ISO-8601 timestamp.") from error
    if not records:
        raise CandidateExposureError("Denis source screen has no records.")
    if len({record.sample_id for record in records}) != len(records):
        raise CandidateExposureError("Denis source records have duplicate sample IDs.")
    project_root = project_root.resolve(strict=True)
    config_root = config_root.resolve(strict=True)
    manifest_root = manifest_root.resolve(strict=True)
    configured_rows, config_bindings, configured_manifest_bindings = configured_role_scope(
        project_root, config_root
    )
    inventory_rows, inventory_bindings, route_manifest_bindings = _load_inventory(
        project_root=project_root, manifest_root=manifest_root
    )
    configured_candidate, configured_prior = _direct_overlaps(records, configured_rows)
    inventory_candidate, inventory_prior = _direct_overlaps(records, inventory_rows)
    direct_overlap = any(configured_candidate.values()) or any(inventory_candidate.values())
    configured_route = _route_summary(configured_rows)
    inventory_route = _route_summary(inventory_rows)
    historical_route_exposed = (
        cast(int, configured_route["unique_sample_ids"]) > 0
        or cast(int, inventory_route["unique_sample_ids"]) > 0
    )
    surviving_records = 0 if direct_overlap else len(records)
    return {
        "schema_version": 1,
        "protocol_id": "denis-1-0-mdc-source-exposure-screen-v1",
        "created_at": created_at,
        "source": {
            "source_id": DENIS_SOURCE_ID,
            "archive_sha256": DENIS_ARCHIVE_EXPECTED_SHA256,
            "records": len(records),
            "source_provided_speaker_groups": 1,
            "source_audit_receipt": source_audit_receipt,
        },
        "candidate_state": (
            "source-wide exact-identity and historical-lineage screen only; no selection, "
            "disk extraction, TTS, or detector inference"
        ),
        "scope": {
            "configuration_directory": _relative_to_root(
                config_root, project_root, "Configuration directory"
            ),
            "configuration_files": config_bindings,
            "configured_manifests": configured_manifest_bindings,
            "configured_rows": len(configured_rows),
            "configured_unique_sample_ids": len({row.sample_id for row in configured_rows}),
            "manifest_inventory_directory": _relative_to_root(
                manifest_root, project_root, "Manifest inventory directory"
            ),
            "manifest_inventory": inventory_bindings,
            "manifest_inventory_rows": len(inventory_rows),
            "manifest_inventory_unique_sample_ids": len(
                {row.sample_id for row in inventory_rows}
            ),
        },
        "direct_overlap_candidate_record_counts": {
            "configured_roles": configured_candidate,
            "manifest_inventory": inventory_candidate,
        },
        "direct_overlap_prior_manifest_row_counts": {
            "configured_roles": configured_prior,
            "manifest_inventory": inventory_prior,
        },
        "strict_single_speaker_group_exclusion": {
            "rule": (
                "Any exact source sample, audio, or literal/canonical transcript collision "
                "taints the complete one-speaker source."
            ),
            "direct_identity_overlap_found": direct_overlap,
            "tainted_source_speaker_groups": 1 if direct_overlap else 0,
            "surviving_records": surviving_records,
            "surviving_source_speaker_groups": 0 if direct_overlap else 1,
        },
        "historical_likely_speaker_lineage": {
            "generator_name": DENIS_HISTORICAL_GENERATOR_NAME,
            "generator_version": DENIS_HISTORICAL_GENERATOR_VERSION,
            "configured_scope": configured_route,
            "manifest_inventory_scope": inventory_route,
            "route_manifest_bindings": route_manifest_bindings,
            "status": "likely_exposed_fail_closed" if historical_route_exposed else "not_found",
            "interpretation": (
                "The source card links Denis 1.0 to an available Piper voice and the official "
                "Piper model card links ru_RU-denis-medium to OHF voice data. This is not a "
                "cryptographic archive-to-checkpoint binding, but it is sufficient to prohibit "
                "speaker-disjoint and speaker-independent claims."
            ),
        },
        "claims": {
            "exact_source_sample_audio_and_text_absent_from_historical_project_scope": (
                not direct_overlap
            ),
            "new_direct_human_source": not direct_overlap,
            "historical_likely_speaker_lineage_exposure": historical_route_exposed,
            "speaker_disjoint": False,
            "speaker_independent": False,
            "speaker_robust": False,
            "candidate_selection_performed": False,
            "disk_extraction_performed": False,
            "tts_inference_performed": False,
            "detector_inference_performed": False,
            "detector_inference_authorized": False,
        },
    }
