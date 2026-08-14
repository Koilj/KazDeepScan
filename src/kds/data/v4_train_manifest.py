"""Invariant checks for the frozen XLS-R+SLS model-v4 training manifest."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kds.data.manifest import ManifestError, ManifestRow, validate_manifest


class V4TrainManifestError(ValueError):
    """Raised when frozen v4 source and spoof inputs cannot form one train manifest."""


@dataclass(frozen=True, slots=True)
class V4TrainAssemblyReport:
    """Auditable facts about one deterministic train-manifest assembly."""

    source_rows: int
    spoof_rows: int
    combined_rows: int
    source_cell_counts: dict[str, int]
    spoof_cell_counts: dict[str, int]
    combined_cell_counts: dict[str, int]
    shared_text_hashes: int


def v4_cell_counts(rows: Sequence[ManifestRow]) -> dict[str, int]:
    """Return deterministic language/label accounting for manifest rows."""

    return dict(sorted(Counter(f"{row.language}/{row.label}" for row in rows).items()))


def build_v4_combined_train_manifest(
    source_rows: Sequence[ManifestRow],
    spoof_rows: Sequence[ManifestRow],
    *,
    expected_source_cells: Mapping[str, int],
    expected_spoof_cells: Mapping[str, int],
    expected_combined_cells: Mapping[str, int],
    expected_shared_text_hashes: int,
) -> tuple[tuple[ManifestRow, ...], V4TrainAssemblyReport]:
    """Validate and deterministically merge the two immutable v4 train inputs.

    Text overlap is permitted only because both inputs are already fixed to the same ``train``
    role. It is not a speaker, source-lineage, or cross-split independence claim.
    """

    if expected_shared_text_hashes < 0:
        raise V4TrainManifestError("expected_shared_text_hashes must be non-negative.")
    source = tuple(source_rows)
    spoof = tuple(spoof_rows)
    if not source or not spoof:
        raise V4TrainManifestError("Both frozen v4 train inputs must be non-empty.")
    try:
        validate_manifest(source)
        validate_manifest(spoof)
    except ManifestError as error:
        raise V4TrainManifestError(error.issues) from error

    source_cells = v4_cell_counts(source)
    spoof_cells = v4_cell_counts(spoof)
    if source_cells != dict(sorted(expected_source_cells.items())):
        raise V4TrainManifestError(
            "Frozen source cell counts "
            f"{source_cells!r} do not match {dict(expected_source_cells)!r}."
        )
    if spoof_cells != dict(sorted(expected_spoof_cells.items())):
        raise V4TrainManifestError(
            f"Frozen spoof cell counts {spoof_cells!r} do not match {dict(expected_spoof_cells)!r}."
        )
    if any(row.split != "train" for row in (*source, *spoof)):
        raise V4TrainManifestError("Frozen v4 combined input contains a non-train row.")

    _require_disjoint(source, spoof, "sample_id")
    _require_disjoint(source, spoof, "sha256")
    _require_disjoint(source, spoof, "relative_path")
    _require_disjoint(source, spoof, "parent_group_id")
    shared_text_hashes = len(
        {row.text_hash for row in source}.intersection(row.text_hash for row in spoof)
    )
    if shared_text_hashes != expected_shared_text_hashes:
        raise V4TrainManifestError(
            "Frozen v4 text overlap is not the contract-pinned value: "
            f"{shared_text_hashes} != {expected_shared_text_hashes}."
        )

    combined = tuple(
        sorted(
            (*source, *spoof),
            key=lambda row: (row.language, row.label, row.sample_id),
        )
    )
    try:
        validate_manifest(combined)
    except ManifestError as error:
        raise V4TrainManifestError(error.issues) from error
    combined_cells = v4_cell_counts(combined)
    if combined_cells != dict(sorted(expected_combined_cells.items())):
        raise V4TrainManifestError(
            "Combined v4 train cell counts "
            f"{combined_cells!r} do not match {dict(expected_combined_cells)!r}."
        )
    return (
        combined,
        V4TrainAssemblyReport(
            source_rows=len(source),
            spoof_rows=len(spoof),
            combined_rows=len(combined),
            source_cell_counts=source_cells,
            spoof_cell_counts=spoof_cells,
            combined_cell_counts=combined_cells,
            shared_text_hashes=shared_text_hashes,
        ),
    )


def _require_disjoint(
    left: Sequence[ManifestRow], right: Sequence[ManifestRow], field: str
) -> None:
    overlap = {getattr(row, field) for row in left}.intersection(
        getattr(row, field) for row in right
    )
    if overlap:
        example = min(overlap)
        raise V4TrainManifestError(
            f"Frozen v4 inputs overlap on {field}: {len(overlap)} values; example={example!r}."
        )
