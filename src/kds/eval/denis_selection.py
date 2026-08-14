"""Deterministic metadata-only Denis pre-QA selection governance."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

from kds.data.denis import DENIS_SOURCE_ID, DenisRecord
from kds.eval.candidate_exposure import CandidateExposureError

DENIS_TARGET_PAIRS = 79
DENIS_SINGLE_SPEAKER_GROUP = f"{DENIS_SOURCE_ID}:speaker:single"


@dataclass(frozen=True, slots=True)
class DenisPreQaSelectionEntry:
    """One exact source identity selected before audio quality or VAD is observed."""

    selection_rank: int
    sample_id: str
    member_stem: str
    category: str
    parent_group_id: str
    speaker_pseudo_id: str
    text_id: str
    literal_text_sha256: str
    whitespace_canonical_text_sha256: str
    nfkc_whitespace_canonical_text_sha256: str
    source_audio_sha256: str
    source_audio_size_bytes: int


@dataclass(frozen=True, slots=True)
class DenisPreQaSelection:
    """Frozen entries and their governance receipt before source extraction."""

    entries: tuple[DenisPreQaSelectionEntry, ...]
    receipt: dict[str, object]


def _rank(seed: str, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def _fingerprint(entries: Sequence[DenisPreQaSelectionEntry]) -> str:
    payload = json.dumps(
        [asdict(entry) for entry in entries],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_parent_screen(screen: Mapping[str, object], record_count: int) -> None:
    source = screen.get("source")
    exclusion = screen.get("strict_single_speaker_group_exclusion")
    lineage = screen.get("historical_likely_speaker_lineage")
    claims = screen.get("claims")
    if (
        screen.get("schema_version") != 1
        or screen.get("protocol_id") != "denis-1-0-mdc-source-exposure-screen-v1"
        or not isinstance(source, Mapping)
        or source.get("source_id") != DENIS_SOURCE_ID
        or source.get("records") != record_count
        or source.get("source_provided_speaker_groups") != 1
        or not isinstance(exclusion, Mapping)
        or exclusion.get("direct_identity_overlap_found") is not False
        or exclusion.get("surviving_records") != record_count
        or exclusion.get("surviving_source_speaker_groups") != 1
        or not isinstance(lineage, Mapping)
        or lineage.get("status") != "likely_exposed_fail_closed"
        or not isinstance(claims, Mapping)
        or claims.get(
            "exact_source_sample_audio_and_text_absent_from_historical_project_scope"
        )
        is not True
        or claims.get("new_direct_human_source") is not True
        or claims.get("historical_likely_speaker_lineage_exposure") is not True
        or claims.get("speaker_independent") is not False
        or claims.get("candidate_selection_performed") is not False
        or claims.get("disk_extraction_performed") is not False
        or claims.get("detector_inference_performed") is not False
    ):
        raise CandidateExposureError(
            "Denis pre-QA selection requires the exact accepted source-exposure screen."
        )


def select_denis_pre_qa_candidate(
    *,
    records: Sequence[DenisRecord],
    source_exposure_screen: Mapping[str, object],
    selection_seed: str,
    requested_records: int,
    target_pairs: int,
    selected_at: str,
) -> DenisPreQaSelection:
    """Balance source categories and rank records without duration/audio-quality signals."""

    try:
        datetime.fromisoformat(selected_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise CandidateExposureError("selected_at must be an ISO-8601 timestamp.") from error
    if not selection_seed:
        raise CandidateExposureError("Denis selection_seed must be non-empty.")
    if not records or requested_records <= 0 or target_pairs <= 0:
        raise CandidateExposureError("Denis selection requires records and positive sizes.")
    if requested_records != target_pairs:
        raise CandidateExposureError(
            "Denis no-backfill contract requires the frozen selection to equal the target."
        )
    if requested_records > len(records):
        raise CandidateExposureError("Requested Denis rows exceed source capacity.")
    if len({record.sample_id for record in records}) != len(records):
        raise CandidateExposureError("Denis source records have duplicate sample IDs.")
    for field in (
        "literal_text_sha256",
        "whitespace_canonical_text_sha256",
        "nfkc_whitespace_canonical_text_sha256",
        "audio_sha256",
    ):
        if len({getattr(record, field) for record in records}) != len(records):
            raise CandidateExposureError(f"Denis source records have duplicate {field} values.")
    _require_parent_screen(source_exposure_screen, len(records))

    by_category: dict[str, list[DenisRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)
    category_order = sorted(
        by_category,
        key=lambda category: (_rank(selection_seed, "category", category), category),
    )
    if requested_records < len(category_order):
        raise CandidateExposureError("Requested Denis rows cannot cover every category.")
    base, remainder = divmod(requested_records, len(category_order))
    category_quotas = {
        category: base + (index < remainder)
        for index, category in enumerate(category_order)
    }
    selected_by_category: dict[str, list[DenisRecord]] = {}
    for category in category_order:
        quota = category_quotas[category]
        if quota > len(by_category[category]):
            raise CandidateExposureError(
                f"Denis category {category!r} cannot satisfy its balanced quota {quota}."
            )
        selected_by_category[category] = sorted(
            by_category[category],
            key=lambda record: (
                _rank(selection_seed, f"record-in-category:{category}", record.sample_id),
                record.sample_id,
            ),
        )[:quota]

    selected_records: list[DenisRecord] = []
    for offset in range(max(category_quotas.values())):
        for category in category_order:
            category_records = selected_by_category[category]
            if offset < len(category_records):
                selected_records.append(category_records[offset])
    if len(selected_records) != requested_records:
        raise CandidateExposureError("Denis category interleave produced an invalid row count.")

    entries = tuple(
        DenisPreQaSelectionEntry(
            selection_rank=rank,
            sample_id=record.sample_id,
            member_stem=record.member_stem,
            category=record.category,
            parent_group_id=DENIS_SINGLE_SPEAKER_GROUP,
            speaker_pseudo_id=DENIS_SINGLE_SPEAKER_GROUP,
            text_id=f"{DENIS_SOURCE_ID}:text:{record.whitespace_canonical_text_sha256}",
            literal_text_sha256=record.literal_text_sha256,
            whitespace_canonical_text_sha256=record.whitespace_canonical_text_sha256,
            nfkc_whitespace_canonical_text_sha256=(
                record.nfkc_whitespace_canonical_text_sha256
            ),
            source_audio_sha256=record.audio_sha256,
            source_audio_size_bytes=record.audio_size_bytes,
        )
        for rank, record in enumerate(selected_records, start=1)
    )
    selected_counts = dict(sorted(Counter(entry.category for entry in entries).items()))
    if max(selected_counts.values()) - min(selected_counts.values()) > 1:
        raise CandidateExposureError("Denis selection category counts are not balanced.")

    return DenisPreQaSelection(
        entries=entries,
        receipt={
            "schema_version": 1,
            "protocol_id": "denis-1-0-mdc-pre-qa-selection-v1",
            "selected_at": selected_at,
            "candidate_state": (
                "immutable metadata-only target selection; no disk extraction, audio-quality "
                "selection, QA/VAD, synthesis, acoustic review, pairing, or detector inference"
            ),
            "survivor_pool": {
                "source_id": DENIS_SOURCE_ID,
                "source_records": len(records),
                "source_categories": dict(sorted(Counter(r.category for r in records).items())),
                "source_speaker_groups": 1,
            },
            "selection_policy": {
                "kind": "seeded_category_balanced_round_robin",
                "selection_seed": selection_seed,
                "target_pairs": target_pairs,
                "requested_records": requested_records,
                "selected_records": len(entries),
                "selected_category_counts": selected_counts,
                "selected_speaker_groups": 1,
                "category_order": category_order,
                "category_ranking": (
                    "ascending SHA-256(seed + NUL + 'category' + NUL + category), then category"
                ),
                "record_ranking": (
                    "within each category, ascending SHA-256(seed + NUL + "
                    "'record-in-category:' + category + NUL + sample_id), then sample_id"
                ),
                "interleave": "one ranked row per seeded category order per round",
                "literal_and_canonical_text_hashes_bound_before_materialization": True,
                "single_source_speaker_group_retained_for_every_row": True,
                "post_selection_replacement_or_backfill": False,
                "selection_uses_audio_or_duration": False,
                "selection_uses_audio_quality_or_vad": False,
                "selection_uses_detector_or_model_output": False,
                "selection_uses_model_metrics_or_final_errors": False,
            },
            "selected_entries_fingerprint_sha256": _fingerprint(entries),
            "claims": {
                "source_exposure_screen_required": True,
                "selection_frozen": True,
                "audio_extraction_performed": False,
                "technical_decode_qa_vad_performed": False,
                "synthetic_audio_generated": False,
                "acoustic_review_performed": False,
                "pairing_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "future_extraction_must_use_only_selected_source_members": True,
                "qa_rejects_must_not_trigger_replacement_or_backfill": True,
                "external_human_source_holdout": True,
                "training_data_overlap_unverified": True,
                "single_speaker": True,
                "speaker_independent": False,
            },
        },
    )
