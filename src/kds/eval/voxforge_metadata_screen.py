"""Pre-extraction project-exposure screen for the pinned VoxForge Russian archive."""

from __future__ import annotations

import csv
import hashlib
import json
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
    source_sample_match_key: str
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


@dataclass(frozen=True, slots=True)
class VoxForgePreQaSelectionEntry:
    """One frozen metadata-only VoxForge record for a later QA candidate."""

    selection_rank: int
    sample_id: str
    submission_pseudo_id: str
    prompt_id: str
    parent_group_id: str
    speaker_pseudo_id: str
    prompt_text_hash: str
    original_prompt_text_hash: str


@dataclass(frozen=True, slots=True)
class VoxForgePreQaSelection:
    """A deterministic one-record-per-text-and-contributor pre-QA selection."""

    entries: tuple[VoxForgePreQaSelectionEntry, ...]
    receipt: dict[str, object]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def voxforge_metadata_identity(record: VoxForgeRuRecord) -> VoxForgeMetadataIdentity:
    """Construct privacy-preserving source keys without exposing contributor aliases."""

    group = f"{VOXFORGE_RU_SOURCE_ID}:contributor:{_hash(record.contributor_alias)}"
    submission = _hash(record.submission_id)
    return VoxForgeMetadataIdentity(
        sample_id=(
            f"{VOXFORGE_RU_SOURCE_ID}:submission:{submission}:prompt:{record.prompt_id}"
        ),
        source_sample_match_key=(
            f"{VOXFORGE_RU_SOURCE_ID}:{record.submission_id}:{record.prompt_id}"
        ),
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
        "sample_id": identity.source_sample_match_key,
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


def _selection_rank(seed: str, domain: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{domain}\x00{value}".encode()).digest()


def _selection_entry_payload(entry: VoxForgePreQaSelectionEntry) -> dict[str, object]:
    return {
        "selection_rank": entry.selection_rank,
        "sample_id": entry.sample_id,
        "submission_pseudo_id": entry.submission_pseudo_id,
        "prompt_id": entry.prompt_id,
        "parent_group_id": entry.parent_group_id,
        "speaker_pseudo_id": entry.speaker_pseudo_id,
        "prompt_text_hash": entry.prompt_text_hash,
        "original_prompt_text_hash": entry.original_prompt_text_hash,
    }


def _selection_fingerprint(entries: Sequence[VoxForgePreQaSelectionEntry]) -> str:
    return hashlib.sha256(
        json.dumps(
            [_selection_entry_payload(entry) for entry in entries],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def select_voxforge_ru_mdc_pre_qa_candidate(
    *,
    records: Sequence[VoxForgeRuRecord],
    metadata_screen: VoxForgeMetadataScreen,
    selection_seed: str,
    requested_text_groups: int,
    selected_at: str,
) -> VoxForgePreQaSelection:
    """Freeze distinct text and conservative contributor groups before WAV extraction.

    The seeded maximum bipartite matching uses only source metadata. It first fixes the requested
    canonical prompt-text groups by seed rank, matches each to a distinct contributor group, and
    then ranks records inside every assigned text/group cell. It never reads WAV payloads or
    ranks on audio, QA, model output, metrics, or final-run errors.
    """

    try:
        datetime.fromisoformat(selected_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandidateExposureError("selected_at must be an ISO-8601 timestamp.") from error
    if not selection_seed or "\x00" in selection_seed:
        raise CandidateExposureError("selection_seed must be non-empty and contain no NUL byte.")
    if requested_text_groups <= 0:
        raise CandidateExposureError("requested_text_groups must be positive.")
    if not records or not metadata_screen.surviving:
        raise CandidateExposureError("VoxForge pre-QA selection has no metadata-screen survivors.")

    identities = tuple(voxforge_metadata_identity(record) for record in records)
    if identities != metadata_screen.identities:
        raise CandidateExposureError(
            "VoxForge metadata screen does not exactly bind the supplied source records."
        )
    records_by_sample: dict[str, VoxForgeRuRecord] = {}
    for record, identity in zip(records, identities, strict=True):
        if identity.sample_id in records_by_sample:
            raise CandidateExposureError("VoxForge source records contain duplicate sample IDs.")
        records_by_sample[identity.sample_id] = record
    survivor_records = tuple(
        records_by_sample[identity.sample_id] for identity in metadata_screen.surviving
    )
    strict_counts = metadata_screen.receipt.get("strict_group_exclusion")
    if not isinstance(strict_counts, dict) or (
        strict_counts.get("surviving_records") != len(survivor_records)
        or strict_counts.get("surviving_contributor_groups")
        != len({voxforge_metadata_identity(record).parent_group_id for record in survivor_records})
    ):
        raise CandidateExposureError(
            "VoxForge metadata-screen receipt does not match its supplied survivors."
        )

    records_by_text_group: dict[str, dict[str, list[VoxForgeRuRecord]]] = {}
    original_hash_by_text: dict[str, str] = {}
    for record in survivor_records:
        identity = voxforge_metadata_identity(record)
        current_original_hash = original_hash_by_text.setdefault(
            identity.prompt_text_hash, identity.original_prompt_text_hash
        )
        if current_original_hash != identity.original_prompt_text_hash:
            raise CandidateExposureError(
                "One canonical VoxForge prompt text maps to multiple original transcript hashes."
            )
        records_by_text_group.setdefault(identity.prompt_text_hash, {}).setdefault(
            identity.parent_group_id, []
        ).append(record)
    if requested_text_groups > len(records_by_text_group):
        raise CandidateExposureError(
            "requested_text_groups exceeds metadata-screen canonical prompt-text capacity."
        )

    ranked_texts = sorted(
        records_by_text_group,
        key=lambda text_hash: (
            _selection_rank(selection_seed, "prompt-text-group", text_hash),
            text_hash,
        ),
    )
    selected_texts = ranked_texts[:requested_text_groups]
    group_to_text: dict[str, str] = {}

    def augment(text_hash: str, seen_groups: set[str]) -> bool:
        groups = sorted(
            records_by_text_group[text_hash],
            key=lambda group_id: (
                _selection_rank(
                    selection_seed,
                    f"contributor-for-prompt-text:{text_hash}",
                    group_id,
                ),
                group_id,
            ),
        )
        for group_id in groups:
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            prior_text = group_to_text.get(group_id)
            if prior_text is None or augment(prior_text, seen_groups):
                group_to_text[group_id] = text_hash
                return True
        return False

    for text_hash in selected_texts:
        if not augment(text_hash, set()):
            raise CandidateExposureError(
                "Requested VoxForge text groups cannot be matched to distinct contributor groups."
            )
    text_to_group = {text_hash: group_id for group_id, text_hash in group_to_text.items()}
    if set(text_to_group) != set(selected_texts):
        raise CandidateExposureError("VoxForge matching did not retain every selected text group.")

    entries: list[VoxForgePreQaSelectionEntry] = []
    for selection_rank, text_hash in enumerate(selected_texts, start=1):
        group_id = text_to_group[text_hash]
        selected_record = min(
            records_by_text_group[text_hash][group_id],
            key=lambda record: (
                _selection_rank(
                    selection_seed,
                    f"record-in-text-contributor:{text_hash}:{group_id}",
                    voxforge_metadata_identity(record).sample_id,
                ),
                voxforge_metadata_identity(record).sample_id,
            ),
        )
        identity = voxforge_metadata_identity(selected_record)
        entries.append(
            VoxForgePreQaSelectionEntry(
                selection_rank=selection_rank,
                sample_id=identity.sample_id,
                submission_pseudo_id=_hash(selected_record.submission_id),
                prompt_id=selected_record.prompt_id,
                parent_group_id=identity.parent_group_id,
                speaker_pseudo_id=identity.speaker_pseudo_id,
                prompt_text_hash=identity.prompt_text_hash,
                original_prompt_text_hash=identity.original_prompt_text_hash,
            )
        )
    if len({entry.sample_id for entry in entries}) != len(entries) or (
        len({entry.parent_group_id for entry in entries}) != len(entries)
        or len({entry.prompt_text_hash for entry in entries}) != len(entries)
        or len({entry.original_prompt_text_hash for entry in entries}) != len(entries)
    ):
        raise CandidateExposureError(
            "VoxForge pre-QA selection does not retain unique sample, contributor, and text groups."
        )

    frozen_entries = tuple(entries)
    survivor_group_count = len(
        {voxforge_metadata_identity(record).parent_group_id for record in survivor_records}
    )
    return VoxForgePreQaSelection(
        entries=frozen_entries,
        receipt={
            "schema_version": 1,
            "protocol_id": "voxforge-ru-mdc-pre-qa-selection-v1",
            "selected_at": selected_at,
            "candidate_state": (
                "immutable metadata selection only; no WAV extraction, synthesis, QA, "
                "acoustic review, pairing, or detector inference"
            ),
            "survivor_pool": {
                "source_id": VOXFORGE_RU_SOURCE_ID,
                "metadata_screen_records": len(survivor_records),
                "metadata_screen_contributor_groups": survivor_group_count,
                "metadata_screen_canonical_prompt_text_groups": len(records_by_text_group),
            },
            "selection_policy": {
                "kind": "seeded_maximum_text_to_contributor_matching",
                "selection_seed": selection_seed,
                "requested_text_groups": requested_text_groups,
                "selected_records": len(frozen_entries),
                "selected_contributor_groups": len(
                    {entry.parent_group_id for entry in frozen_entries}
                ),
                "selected_canonical_prompt_text_groups": len(
                    {entry.prompt_text_hash for entry in frozen_entries}
                ),
                "selected_original_prompt_text_groups": len(
                    {entry.original_prompt_text_hash for entry in frozen_entries}
                ),
                "text_group_ranking": (
                    "ascending SHA-256(seed + NUL + 'prompt-text-group' + NUL + text_hash), "
                    "then text_hash"
                ),
                "contributor_matching": (
                    "seeded deterministic augmenting-path maximum matching of selected canonical "
                    "prompt-text groups to distinct contributor groups"
                ),
                "record_ranking": (
                    "ascending SHA-256(seed + NUL + 'record-in-text-contributor:' + "
                    "text_hash + ':' + parent_group_id + NUL + sample_id), then sample_id"
                ),
                "exactly_one_record_per_selected_contributor_group": True,
                "exactly_one_record_per_selected_canonical_prompt_text_group": True,
                "post_selection_backfill": False,
                "selection_uses_audio_or_duration": False,
                "selection_uses_detector_or_model_output": False,
                "selection_uses_model_metrics_or_final_errors": False,
            },
            "selected_entries_fingerprint_sha256": _selection_fingerprint(frozen_entries),
            "claims": {
                "metadata_exposure_screen_required": True,
                "audio_extraction_performed": False,
                "synthetic_audio_generated": False,
                "qa_or_acoustic_review_performed": False,
                "pairing_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "selection_frozen": True,
                "future_extraction_must_use_only_selected_source_members": True,
                "qa_rejects_must_not_trigger_backfill": True,
            },
        },
    )
