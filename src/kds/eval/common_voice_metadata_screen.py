"""Screen unmaterialized Common Voice RU test metadata against all project history.

This is deliberately a pre-extraction gate.  It has no audio hash or duration because the
selected source clips have not been extracted yet; sample, transcript and source client-group
identities must nevertheless be absent from every configured role and manifest inventory.
"""

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
from kds.data.common_voice import COMMON_VOICE_RU_V24_SOURCE_ID, CommonVoiceRecord
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest
from kds.data.silero_v5_5 import (
    SILERO_V5_5_FIXED_SPEAKER,
    SILERO_V5_5_SAMPLE_RATE,
    SILERO_V5_5_TEXT_NORMALIZER_ID,
    SileroV55Error,
    normalize_silero_v5_5_text,
)
from kds.eval.candidate_exposure import CandidateExposureError, configured_role_scope

METADATA_COMPARISON_FIELDS = (
    "sample_id",
    "text_hash",
    "parent_group_id",
    "speaker_pseudo_id",
)


@dataclass(frozen=True, slots=True)
class CommonVoiceMetadataIdentity:
    """Pre-extraction identities reproducible from one validated TSV row."""

    clip_name: str
    sample_id: str
    text_hash: str
    parent_group_id: str
    speaker_pseudo_id: str


@dataclass(frozen=True, slots=True)
class CommonVoiceMetadataScreen:
    """A source-wide availability screen; it does not select or materialize clips."""

    identities: tuple[CommonVoiceMetadataIdentity, ...]
    surviving: tuple[CommonVoiceMetadataIdentity, ...]
    receipt: dict[str, object]


@dataclass(frozen=True, slots=True)
class CommonVoiceTextCompatibilityScreen:
    """Literal-text-compatible records after fail-closed client-group exclusion."""

    surviving: tuple[CommonVoiceRecord, ...]
    receipt: dict[str, object]


@dataclass(frozen=True, slots=True)
class CommonVoicePreQaSelectionEntry:
    """One frozen, unmaterialized Common Voice record for a later QA candidate."""

    selection_rank: int
    sample_id: str
    clip_name: str
    source_split: str
    parent_group_id: str
    speaker_pseudo_id: str
    text_id: str
    text_hash: str


@dataclass(frozen=True, slots=True)
class CommonVoicePreQaSelection:
    """A deterministic one-record-per-client-group pre-QA selection."""

    entries: tuple[CommonVoicePreQaSelectionEntry, ...]
    receipt: dict[str, object]


def common_voice_metadata_identity(record: CommonVoiceRecord) -> CommonVoiceMetadataIdentity:
    """Derive the fields that exist before extracting a Common Voice audio member."""

    if record.split != "test":
        raise CandidateExposureError(
            "Common Voice metadata screen accepts only source split 'test'."
        )
    group = f"{COMMON_VOICE_RU_V24_SOURCE_ID}:client:{record.client_id}"
    return CommonVoiceMetadataIdentity(
        clip_name=record.clip_name,
        sample_id=f"{COMMON_VOICE_RU_V24_SOURCE_ID}:{Path(record.clip_name).stem}",
        text_hash=hashlib.sha256(record.sentence.encode("utf-8")).hexdigest(),
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
        raise CandidateExposureError("Metadata screen found no valid manifest inventory files.")
    return rows, bindings


def _values(rows: Sequence[ManifestRow]) -> dict[str, set[str]]:
    return {
        field: {str(getattr(row, field)) for row in rows}
        for field in METADATA_COMPARISON_FIELDS
    }


def _identity_values(identity: CommonVoiceMetadataIdentity) -> dict[str, str]:
    return {
        "sample_id": identity.sample_id,
        "text_hash": identity.text_hash,
        "parent_group_id": identity.parent_group_id,
        "speaker_pseudo_id": identity.speaker_pseudo_id,
    }


def screen_common_voice_ru_test_metadata(
    *,
    records: Sequence[CommonVoiceRecord],
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
    created_at: str,
) -> CommonVoiceMetadataScreen:
    """Fail closed on any historical identity overlap and taint the whole client group.

    The result is source-wide capacity evidence only.  A later immutable selection must still
    bind an explicit seed, requested size and the exact survivors it materializes.
    """

    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandidateExposureError("created_at must be an ISO-8601 timestamp.") from error
    if not records:
        raise CandidateExposureError("Common Voice metadata screen has no records.")
    project_root = project_root.resolve(strict=True)
    config_root = config_root.resolve(strict=True)
    manifest_root = manifest_root.resolve(strict=True)
    _relative_to_root(config_root, project_root, "Configuration directory")
    _relative_to_root(manifest_root, project_root, "Manifest inventory directory")

    identities = tuple(common_voice_metadata_identity(record) for record in records)
    sample_ids = [identity.sample_id for identity in identities]
    if len(sample_ids) != len(set(sample_ids)):
        raise CandidateExposureError("Common Voice metadata screen has duplicate sample IDs.")

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
            "No Common Voice test records remain after strict historical group exclusion."
        )
    source_groups = {identity.parent_group_id for identity in identities}
    surviving_groups = {identity.parent_group_id for identity in surviving}
    return CommonVoiceMetadataScreen(
        identities=identities,
        surviving=surviving,
        receipt={
            "schema_version": 1,
            "protocol_id": "common-voice-ru-v24-test-metadata-exposure-screen-v1",
            "created_at": created_at,
            "candidate_state": "source-wide metadata screen only; no clip selection or extraction",
            "source": {
                "source_id": COMMON_VOICE_RU_V24_SOURCE_ID,
                "source_split": "test",
                "metadata_records": len(identities),
                "client_groups": len(source_groups),
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
                "policy": (
                    "exclude a whole Common Voice client group if any source test record has a "
                    "sample/text/parent-group/speaker-pseudo-ID overlap in configured roles or "
                    "manifest inventory"
                ),
                "tainted_client_groups": len(tainted_groups),
                "surviving_client_groups": len(surviving_groups),
                "surviving_records": len(surviving),
                "excluded_records_due_to_direct_or_group_overlap": len(identities) - len(surviving),
            },
            "claims": {
                "audio_extraction_performed": False,
                "synthetic_audio_generated": False,
                "qa_or_acoustic_review_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "selection_frozen": False,
                "future_selection_must_bind_seed_count_and_survivors": True,
            },
        },
    )


def screen_silero_v5_5_literal_text_compatibility(
    *,
    records: Sequence[CommonVoiceRecord],
    metadata_screen: CommonVoiceMetadataScreen,
) -> CommonVoiceTextCompatibilityScreen:
    """Reject every client group containing a text outside the fixed V5.5 wrapper grammar.

    The upstream package must never see an incompatible source text or an external lexical
    rewrite. A client group is atomic: one incompatible transcript makes all of its source
    records unavailable for a future candidate, even if its other texts happen to be compatible.
    """

    records_by_sample: dict[str, CommonVoiceRecord] = {}
    for record in records:
        identity = common_voice_metadata_identity(record)
        if identity.sample_id in records_by_sample:
            raise CandidateExposureError(
                "Common Voice literal-text screen has duplicate source sample IDs."
            )
        records_by_sample[identity.sample_id] = record
    metadata_survivor_ids = {identity.sample_id for identity in metadata_screen.surviving}
    if not metadata_survivor_ids:
        raise CandidateExposureError("Metadata screen has no survivors for literal-text screening.")
    missing = sorted(metadata_survivor_ids.difference(records_by_sample))
    if missing:
        raise CandidateExposureError(
            "Metadata screen survivors are absent from supplied source records: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )

    incompatible: list[dict[str, str]] = []
    tainted_groups: set[str] = set()
    reason_counts: Counter[str] = Counter()
    candidate_records = [
        records_by_sample[sample_id] for sample_id in sorted(metadata_survivor_ids)
    ]
    for record in candidate_records:
        identity = common_voice_metadata_identity(record)
        try:
            normalized = normalize_silero_v5_5_text(record.sentence)
            if normalized != record.sentence:
                raise SileroV55Error(
                    "Silero V5.5 literal-text contract would change the source transcript."
                )
        except SileroV55Error as error:
            reason = str(error)
            tainted_groups.add(identity.parent_group_id)
            reason_counts[reason] += 1
            incompatible.append(
                {
                    "sample_id": identity.sample_id,
                    "parent_group_id": identity.parent_group_id,
                    "reason": reason,
                }
            )
    surviving = tuple(
        record
        for record in candidate_records
        if common_voice_metadata_identity(record).parent_group_id not in tainted_groups
    )
    if not surviving:
        raise CandidateExposureError(
            "No Common Voice records remain after literal-text client-group exclusion."
        )
    surviving_groups = {
        common_voice_metadata_identity(record).parent_group_id for record in surviving
    }
    return CommonVoiceTextCompatibilityScreen(
        surviving=surviving,
        receipt={
            "schema_version": 1,
            "protocol_id": "common-voice-ru-v24-silero-v5-5-literal-text-screen-v1",
            "candidate_state": (
                "source-wide text compatibility screen only; no clip selection or extraction"
            ),
            "input_metadata_screen": {
                "source_id": COMMON_VOICE_RU_V24_SOURCE_ID,
                "source_split": "test",
                "records_before_literal_text_screen": len(candidate_records),
                "client_groups_before_literal_text_screen": len(
                    {
                        common_voice_metadata_identity(record).parent_group_id
                        for record in candidate_records
                    }
                ),
            },
            "literal_text_contract": {
                "normalizer_id": SILERO_V5_5_TEXT_NORMALIZER_ID,
                "fixed_speaker": SILERO_V5_5_FIXED_SPEAKER,
                "sample_rate": SILERO_V5_5_SAMPLE_RATE,
                "external_lexical_rewrite": "forbidden",
                "normalized_text_must_equal_source_sentence": True,
                "reference_audio": "forbidden",
                "voice_cloning": "forbidden",
            },
            "direct_incompatible_records": incompatible,
            "direct_incompatibility_reason_counts": dict(sorted(reason_counts.items())),
            "strict_group_exclusion": {
                "policy": (
                    "exclude a whole Common Voice client group if any metadata-screen survivor "
                    "fails the fixed V5.5 literal-text wrapper grammar"
                ),
                "direct_incompatible_records": len(incompatible),
                "tainted_client_groups": len(tainted_groups),
                "surviving_records": len(surviving),
                "surviving_client_groups": len(surviving_groups),
                "excluded_records_due_to_direct_or_group_incompatibility": (
                    len(candidate_records) - len(surviving)
                ),
            },
            "claims": {
                "audio_extraction_performed": False,
                "synthetic_audio_generated": False,
                "qa_or_acoustic_review_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "selection_frozen": False,
                "future_selection_must_bind_seed_count_and_survivors": True,
            },
        },
    )


def _selection_rank(seed: str, domain: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{domain}\x00{value}".encode()).digest()


def _selection_entry_payload(entry: CommonVoicePreQaSelectionEntry) -> dict[str, object]:
    return {
        "selection_rank": entry.selection_rank,
        "sample_id": entry.sample_id,
        "clip_name": entry.clip_name,
        "source_split": entry.source_split,
        "parent_group_id": entry.parent_group_id,
        "speaker_pseudo_id": entry.speaker_pseudo_id,
        "text_id": entry.text_id,
        "text_hash": entry.text_hash,
    }


def _selection_fingerprint(entries: Sequence[CommonVoicePreQaSelectionEntry]) -> str:
    return hashlib.sha256(
        json.dumps(
            [_selection_entry_payload(entry) for entry in entries],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def select_common_voice_ru_v24_silero_v5_5_pre_qa_candidate(
    *,
    compatibility_screen: CommonVoiceTextCompatibilityScreen,
    selection_seed: str,
    requested_client_groups: int,
    selected_at: str,
) -> CommonVoicePreQaSelection:
    """Freeze one literal-text-compatible test record for each selected client group.

    This selection deliberately happens before extraction.  It ranks all compatible client
    groups from a public seed, then independently ranks records within each selected group.
    It never inspects audio, model output, QA outcomes, or final-run errors.
    """

    try:
        datetime.fromisoformat(selected_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandidateExposureError("selected_at must be an ISO-8601 timestamp.") from error
    if not selection_seed or "\x00" in selection_seed:
        raise CandidateExposureError("selection_seed must be non-empty and contain no NUL byte.")
    if requested_client_groups <= 0:
        raise CandidateExposureError("requested_client_groups must be positive.")
    if not compatibility_screen.surviving:
        raise CandidateExposureError("Literal-text compatibility screen has no survivors.")

    by_group: dict[str, list[CommonVoiceRecord]] = {}
    seen_sample_ids: set[str] = set()
    for record in compatibility_screen.surviving:
        identity = common_voice_metadata_identity(record)
        if identity.sample_id in seen_sample_ids:
            raise CandidateExposureError(
                "Literal-text compatibility screen has duplicate source sample IDs."
            )
        seen_sample_ids.add(identity.sample_id)
        by_group.setdefault(identity.parent_group_id, []).append(record)
    if requested_client_groups > len(by_group):
        raise CandidateExposureError(
            "requested_client_groups exceeds literal-text-compatible client-group capacity."
        )

    ranked_groups = sorted(
        by_group,
        key=lambda group_id: (_selection_rank(selection_seed, "client-group", group_id), group_id),
    )
    entries: list[CommonVoicePreQaSelectionEntry] = []
    for selection_rank, group_id in enumerate(ranked_groups[:requested_client_groups], start=1):
        selected_record = min(
            by_group[group_id],
            key=lambda record: (
                _selection_rank(
                    selection_seed,
                    f"record-in-client-group:{group_id}",
                    common_voice_metadata_identity(record).sample_id,
                ),
                common_voice_metadata_identity(record).sample_id,
            ),
        )
        identity = common_voice_metadata_identity(selected_record)
        entries.append(
            CommonVoicePreQaSelectionEntry(
                selection_rank=selection_rank,
                sample_id=identity.sample_id,
                clip_name=selected_record.clip_name,
                source_split=selected_record.split,
                parent_group_id=identity.parent_group_id,
                speaker_pseudo_id=identity.speaker_pseudo_id,
                text_id=selected_record.sentence_id,
                text_hash=identity.text_hash,
            )
        )
    if len({entry.parent_group_id for entry in entries}) != len(entries):
        raise CandidateExposureError("Pre-QA selection does not have unique client groups.")

    compatibility_counts = compatibility_screen.receipt.get("strict_group_exclusion")
    if not isinstance(compatibility_counts, dict):
        raise CandidateExposureError("Literal-text compatibility receipt has invalid group counts.")
    if (
        compatibility_counts.get("surviving_records") != len(compatibility_screen.surviving)
        or compatibility_counts.get("surviving_client_groups") != len(by_group)
    ):
        raise CandidateExposureError(
            "Literal-text compatibility receipt does not match its supplied survivor records."
        )

    frozen_entries = tuple(entries)
    return CommonVoicePreQaSelection(
        entries=frozen_entries,
        receipt={
            "schema_version": 1,
            "protocol_id": "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-selection-v1",
            "selected_at": selected_at,
            "candidate_state": (
                "immutable metadata selection only; no audio extraction, synthesis, QA, "
                "acoustic review, or detector inference"
            ),
            "survivor_pool": {
                "source_id": COMMON_VOICE_RU_V24_SOURCE_ID,
                "source_split": "test",
                "literal_text_compatible_records": len(compatibility_screen.surviving),
                "literal_text_compatible_client_groups": len(by_group),
            },
            "selection_policy": {
                "kind": "seeded_two_stage_one_record_per_client_group",
                "selection_seed": selection_seed,
                "requested_client_groups": requested_client_groups,
                "selected_records": len(frozen_entries),
                "selected_client_groups": len({entry.parent_group_id for entry in frozen_entries}),
                "group_ranking": (
                    "ascending SHA-256(seed + NUL + 'client-group' + NUL + parent_group_id), "
                    "then parent_group_id"
                ),
                "record_ranking": (
                    "ascending SHA-256(seed + NUL + 'record-in-client-group:' + "
                    "parent_group_id + NUL + sample_id), then sample_id"
                ),
                "exactly_one_record_per_selected_client_group": True,
                "post_selection_backfill": False,
                "selection_uses_audio_or_duration": False,
                "selection_uses_detector_or_model_output": False,
                "selection_uses_model_metrics_or_final_errors": False,
            },
            "selected_entries_fingerprint_sha256": _selection_fingerprint(frozen_entries),
            "claims": {
                "historical_exposure_and_literal_text_screens_required": True,
                "audio_extraction_performed": False,
                "synthetic_audio_generated": False,
                "qa_or_acoustic_review_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "selection_frozen": True,
                "future_extraction_must_use_only_selected_clip_names": True,
                "qa_rejects_must_not_trigger_backfill": True,
            },
        },
    )
