"""Build a development role that is leakage-safe against the selected training rows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from kds.data.manifest import ManifestError, ManifestRow, validate_manifest

STAGE_B_LEAKAGE_FIELDS = (
    "sample_id",
    "sha256",
    "parent_group_id",
    "speaker_pseudo_id",
    "text_hash",
)


@dataclass(frozen=True, slots=True)
class ExcludedStageBDevRow:
    sample_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageBDevFilterReport:
    candidate_rows: int
    selected_rows: int
    excluded_rows: int
    label_counts: dict[str, int]
    reason_counts: dict[str, int]
    exclusions: tuple[ExcludedStageBDevRow, ...]


def filter_stage_b_dev_rows(
    train_rows: list[ManifestRow], candidate_rows: list[ManifestRow]
) -> tuple[list[ManifestRow], StageBDevFilterReport]:
    """Remove every candidate group that shares an ordinary leakage key with train."""

    validate_manifest(train_rows)
    if not train_rows or {row.split for row in train_rows} != {"train"}:
        raise ManifestError(["Stage-B train selection must contain only split='train' rows."])
    return _filter_stage_b_dev_rows(train_rows, candidate_rows)


def filter_stage_b_calibration_rows(
    historical_rows: list[ManifestRow], candidate_rows: list[ManifestRow]
) -> tuple[list[ManifestRow], StageBDevFilterReport]:
    """Publish calibration candidates only if they are disjoint from all earlier Stage-B roles.

    Stage-B's epoch-selection dev is not a valid calibration set.  The calibration candidates
    must therefore be checked against the selected train rows, the original Stage-A dev and the
    Stage-B selection dev at once.  The historical rows may legitimately have several split
    labels; unlike :func:`filter_stage_b_dev_rows`, this helper deliberately does not relabel
    them to train.
    """

    validate_manifest(historical_rows)
    if not historical_rows:
        raise ManifestError(["Stage-B calibration history must contain at least one row."])
    return _filter_stage_b_dev_rows(historical_rows, candidate_rows)


def _filter_stage_b_dev_rows(
    reference_rows: list[ManifestRow], candidate_rows: list[ManifestRow]
) -> tuple[list[ManifestRow], StageBDevFilterReport]:
    """Filter one new dev role against a previously fixed set of observations."""

    validate_manifest(candidate_rows)
    if not candidate_rows or {row.split for row in candidate_rows} != {"dev"}:
        raise ManifestError(
            ["Stage-B dev/calibration candidate must contain only split='dev' rows."]
        )
    reference_values = {
        field: {getattr(row, field) for row in reference_rows} for field in STAGE_B_LEAKAGE_FIELDS
    }
    excluded_text_hashes = {
        row.text_hash
        for row in candidate_rows
        if any(
            getattr(row, field) in reference_values[field] for field in STAGE_B_LEAKAGE_FIELDS
        )
    }
    selected: list[ManifestRow] = []
    exclusions: list[ExcludedStageBDevRow] = []
    reason_counts: Counter[str] = Counter()
    for row in candidate_rows:
        reasons = tuple(
            field
            for field in STAGE_B_LEAKAGE_FIELDS
            if getattr(row, field) in reference_values[field]
        )
        if not reasons and row.text_hash in excluded_text_hashes:
            reasons = (*reasons, "paired_text_group")
        if reasons:
            exclusions.append(ExcludedStageBDevRow(sample_id=row.sample_id, reasons=reasons))
            reason_counts.update(reasons)
        else:
            selected.append(row)
    if {row.label for row in selected} != {"bonafide", "spoof"}:
        raise ManifestError(
            ["Filtered Stage-B dev/calibration role must retain both bonafide and spoof rows."]
        )
    validate_manifest([*reference_rows, *selected])
    return (
        selected,
        StageBDevFilterReport(
            candidate_rows=len(candidate_rows),
            selected_rows=len(selected),
            excluded_rows=len(exclusions),
            label_counts={
                label: sum(row.label == label for row in selected)
                for label in ("bonafide", "spoof")
            },
            reason_counts=dict(sorted(reason_counts.items())),
            exclusions=tuple(exclusions),
        ),
    )
