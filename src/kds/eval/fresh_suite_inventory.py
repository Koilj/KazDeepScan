"""Fail-closed inventory checks for a future asset-level-blind research suite."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from kds.data.fleurs import FleursRecord, fleurs_locale_spec
from kds.data.manifest import ManifestRow


class FreshSuiteInventoryError(ValueError):
    """Raised when a claimed fresh-source inventory cannot be reproduced safely."""


def audit_fleurs_locale_inventory(
    *,
    locale: str,
    test_records: Iterable[FleursRecord],
    ready_rows: Sequence[ManifestRow],
    exposed_rows: Sequence[ManifestRow],
) -> dict[str, object]:
    """Count release-level and QA-ready FLEURS text groups after project exposure."""

    spec = fleurs_locale_spec(locale)
    records = list(test_records)
    if not records:
        raise FreshSuiteInventoryError(f"FLEURS {locale} test inventory is empty.")
    if any(record.locale != locale or record.source_split != "test" for record in records):
        raise FreshSuiteInventoryError(
            f"FLEURS {locale} inventory contains a different locale or source split."
        )

    release_by_sample_id: dict[str, FleursRecord] = {}
    for record in records:
        sample_id = f"{spec.source_id}:{record.filename.removesuffix('.wav')}"
        if sample_id in release_by_sample_id:
            raise FreshSuiteInventoryError(
                f"FLEURS {locale} release contains duplicate sample ID {sample_id!r}."
            )
        release_by_sample_id[sample_id] = record

    ready = _validated_fleurs_source_rows(
        ready_rows,
        source_id=spec.source_id,
        language=spec.language,
        release_by_sample_id=release_by_sample_id,
        role="ready",
    )
    exposed = _validated_fleurs_source_rows(
        exposed_rows,
        source_id=spec.source_id,
        language=spec.language,
        release_by_sample_id=release_by_sample_id,
        role="exposed",
    )
    ready_texts = {row.text_hash for row in ready}
    exposed_texts = {row.text_hash for row in exposed}
    release_texts = {record.text_hash for record in records}
    fresh_release_texts = release_texts.difference(exposed_texts)
    fresh_ready_texts = ready_texts.difference(exposed_texts)

    return {
        "locale": locale,
        "language": spec.language,
        "source_id": spec.source_id,
        "source_test_rows": len(records),
        "source_test_unique_text_groups": len(release_texts),
        "qa_ready_rows": len(ready),
        "qa_ready_unique_text_groups": len(ready_texts),
        "exposed_bonafide_rows": len(exposed),
        "exposed_unique_text_groups": len(exposed_texts),
        "fresh_release_unique_text_groups_before_extraction_and_qa": len(fresh_release_texts),
        "fresh_qa_ready_unique_text_groups": len(fresh_ready_texts),
        "all_current_qa_ready_text_groups_exposed": not fresh_ready_texts,
    }


def audit_ksc2_mixed_inventory(
    *,
    candidate_rows: Sequence[Mapping[str, str]],
    reviewed_rows: Sequence[Mapping[str, str]],
    ready_rows: Sequence[ManifestRow],
    exposed_rows: Sequence[ManifestRow],
) -> dict[str, object]:
    """Count pending, reviewed, QA-ready and already exposed KSC2 mixed assets."""

    candidates = _unique_custom_rows(candidate_rows, "annotation_id", "KSC2 candidate")
    for annotation_id, row in candidates.items():
        expected = {
            "annotation_state": "pending",
            "language": "unknown",
            "code_switch": "unknown",
            "source_name": "ksc2_v1",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise FreshSuiteInventoryError(
                    f"KSC2 candidate {annotation_id!r} field {field!r} must be {value!r}."
                )

    reviewed = _unique_custom_rows(reviewed_rows, "annotation_id", "KSC2 review")
    for annotation_id, row in reviewed.items():
        candidate = candidates.get(annotation_id)
        if candidate is None:
            raise FreshSuiteInventoryError(
                f"KSC2 review references an unknown candidate {annotation_id!r}."
            )
        if row.get("language") != "mixed" or row.get("code_switch") != "true":
            raise FreshSuiteInventoryError(
                f"KSC2 review {annotation_id!r} is not an explicit mixed/code-switch decision."
            )
        for field in ("audio_sha256", "transcript_sha256", "archive_audio_member"):
            if row.get(field) != candidate.get(field):
                raise FreshSuiteInventoryError(
                    f"KSC2 review {annotation_id!r} changes candidate field {field!r}."
                )

    ready = _validated_ksc2_manifest_rows(ready_rows, role="ready")
    exposed = _validated_ksc2_manifest_rows(exposed_rows, role="exposed")
    ready_ids = {row.sample_id for row in ready}
    reviewed_ids = set(reviewed)
    if not ready_ids.issubset(reviewed_ids):
        unknown = sorted(ready_ids.difference(reviewed_ids))
        raise FreshSuiteInventoryError(
            f"KSC2 ready manifest contains {len(unknown)} rows without semantic review."
        )
    for ready_row in ready:
        reviewed_row = reviewed[ready_row.sample_id]
        if ready_row.text_hash != reviewed_row.get("transcript_sha256"):
            raise FreshSuiteInventoryError(
                f"KSC2 ready row {ready_row.sample_id!r} changes the reviewed transcript hash."
            )

    exposed_ids = {row.sample_id for row in exposed}
    if not exposed_ids.issubset(ready_ids):
        unknown = sorted(exposed_ids.difference(ready_ids))
        raise FreshSuiteInventoryError(
            f"KSC2 exposed manifest contains {len(unknown)} rows outside QA-ready review."
        )
    fresh_ready_ids = sorted(ready_ids.difference(exposed_ids))

    return {
        "source_id": "ksc2_v1",
        "candidate_rows": len(candidates),
        "semantically_reviewed_mixed_rows": len(reviewed),
        "pending_semantic_review_rows": len(candidates) - len(reviewed),
        "qa_ready_mixed_rows": len(ready_ids),
        "exposed_bonafide_mixed_rows": len(exposed_ids),
        "fresh_qa_ready_mixed_rows": len(fresh_ready_ids),
        "fresh_qa_ready_annotation_ids": fresh_ready_ids,
    }


def _validated_fleurs_source_rows(
    rows: Sequence[ManifestRow],
    *,
    source_id: str,
    language: str,
    release_by_sample_id: Mapping[str, FleursRecord],
    role: str,
) -> list[ManifestRow]:
    selected: list[ManifestRow] = []
    seen_sample_ids: set[str] = set()
    for row in rows:
        if row.source_name.startswith("google_fleurs_") and row.source_name != source_id:
            raise FreshSuiteInventoryError(
                f"FLEURS {role} inventory mixes source {row.source_name!r} with {source_id!r}."
            )
        if row.source_name != source_id:
            continue
        if row.label != "bonafide" or row.language != language or row.split != "test":
            raise FreshSuiteInventoryError(
                f"FLEURS {role} row {row.sample_id!r} has an invalid role, label or language."
            )
        if row.sample_id in seen_sample_ids:
            raise FreshSuiteInventoryError(
                f"FLEURS {role} inventory repeats sample ID {row.sample_id!r}."
            )
        seen_sample_ids.add(row.sample_id)
        record = release_by_sample_id.get(row.sample_id)
        if record is None or row.text_hash != record.text_hash:
            raise FreshSuiteInventoryError(
                f"FLEURS {role} row {row.sample_id!r} is not bound to the pinned release."
            )
        selected.append(row)
    if not selected:
        raise FreshSuiteInventoryError(
            f"FLEURS {role} inventory has no {source_id} bona-fide rows."
        )
    return selected


def _unique_custom_rows(
    rows: Sequence[Mapping[str, str]], key: str, label: str
) -> dict[str, Mapping[str, str]]:
    if not rows:
        raise FreshSuiteInventoryError(f"{label} inventory is empty.")
    result: dict[str, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        value = row.get(key, "").strip()
        if not value:
            raise FreshSuiteInventoryError(f"{label} row {row_number} has an empty {key!r}.")
        if value in result:
            raise FreshSuiteInventoryError(f"{label} repeats {key} {value!r}.")
        result[value] = row
    return result


def _validated_ksc2_manifest_rows(
    rows: Sequence[ManifestRow], *, role: str
) -> list[ManifestRow]:
    selected: list[ManifestRow] = []
    seen: set[str] = set()
    for row in rows:
        if row.source_name != "ksc2_v1":
            continue
        if (
            row.label != "bonafide"
            or row.language != "mixed"
            or row.code_switch != "true"
            or row.split != "test"
        ):
            raise FreshSuiteInventoryError(
                f"KSC2 {role} row {row.sample_id!r} is not mixed bona-fide test evidence."
            )
        if row.sample_id in seen:
            raise FreshSuiteInventoryError(
                f"KSC2 {role} inventory repeats sample ID {row.sample_id!r}."
            )
        seen.add(row.sample_id)
        selected.append(row)
    if not selected:
        raise FreshSuiteInventoryError(f"KSC2 {role} inventory has no bona-fide mixed rows.")
    return selected
