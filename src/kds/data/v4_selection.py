"""Frozen metadata selection for the KazDeepScan XLS-R+SLS model v4 train pool."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import sha256_file
from kds.data.ksc2 import Ksc2TextCandidate
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest
from kds.data.ruasd_catalog import (
    RuAsdArchiveSpec,
    _validate_archive_set,
    _validate_archive_size,
    load_ruasd_archive_records,
)
from kds.data.ruasd_research import RuAsdResearchRecord, _eligible_records

V4_SELECTION_SCHEMA_VERSION = 2
V4_CANDIDATE_FIELDS = (
    "selection_rank",
    "target_state",
    "role",
    "language",
    "label",
    "pair_id",
    "candidate_id",
    "source_id",
    "source_lineage_id",
    "source_component",
    "archive_audio_member",
    "archive_transcript_member",
    "text_hash",
    "canonical_text_hash",
    "parent_group_id",
    "speaker_group_status",
    "generator_route_id",
    "generator_family",
    "raw_audio_sha256",
    "decoded_audio_sha256",
    "asset_state",
)


class V4SelectionError(ValueError):
    """Raised when v4 metadata candidates cannot be selected fail-closed."""


V4SelectionProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class V4ExposureInventory:
    manifest_bindings: tuple[dict[str, object], ...]
    rows: int
    sample_ids: frozenset[str]
    audio_sha256: frozenset[str]
    text_hashes: frozenset[str]
    parent_group_ids: frozenset[str]
    speaker_group_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class V4SelectionConfig:
    protocol_id: str
    capacity_receipt_path: str
    capacity_receipt_sha256: str
    selection_seed: str
    target_rows_per_cell: int
    candidate_rows_per_cell: int
    ruasd_excluded_subsets: frozenset[str]
    ruasd_require_source_text: bool
    ruasd_min_per_stratum: int
    ruasd_max_per_stratum: int
    ksc2_component_quotas: dict[str, int]
    kk_generator_quotas: dict[str, int]
    kk_generator_families: dict[str, str]
    roles: dict[str, object]
    source_lineage_roots: dict[str, tuple[str, ...]]
    tts_family_roots: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class V4CandidateRow:
    selection_rank: int
    target_state: str
    role: str
    language: str
    label: str
    pair_id: str
    candidate_id: str
    source_id: str
    source_lineage_id: str
    source_component: str
    archive_audio_member: str
    archive_transcript_member: str
    text_hash: str
    canonical_text_hash: str
    parent_group_id: str
    speaker_group_status: str
    generator_route_id: str
    generator_family: str
    raw_audio_sha256: str
    decoded_audio_sha256: str
    asset_state: str


@dataclass(frozen=True, slots=True)
class V4SourceSelection:
    rows: tuple[V4CandidateRow, ...]
    available_by_stratum: dict[str, int]
    selected_by_stratum: dict[str, int]
    rejection_counts: dict[str, int]


def load_v4_selection_config(path: Path) -> V4SelectionConfig:
    raw = _json_object(path, "v4 selection config")
    _expect_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "purpose",
            "capacity_receipt",
            "selection",
            "roles",
            "role_isolation",
            "claims",
        },
        "v4 selection config",
    )
    if (
        raw["schema_version"] != V4_SELECTION_SCHEMA_VERSION
        or raw["purpose"] != "personal_research"
    ):
        raise V4SelectionError("v4 selection config schema/purpose is invalid.")
    capacity = _mapping(raw["capacity_receipt"], "capacity_receipt")
    _expect_keys(capacity, {"path", "sha256", "required_decision"}, "capacity_receipt")
    if capacity["required_decision"] != "proceed_24k":
        raise V4SelectionError("v4 selection requires capacity decision proceed_24k.")
    selection = _mapping(raw["selection"], "selection")
    _expect_keys(
        selection,
        {
            "seed",
            "target_rows_per_cell",
            "candidate_rows_per_cell",
            "ruasd",
            "ksc2",
            "kk_spoof_generators",
        },
        "selection",
    )
    target = _positive_int(selection["target_rows_per_cell"], "target_rows_per_cell")
    candidate = _positive_int(
        selection["candidate_rows_per_cell"], "candidate_rows_per_cell"
    )
    if candidate <= target:
        raise V4SelectionError("candidate_rows_per_cell must exceed the frozen target.")
    ruasd = _mapping(selection["ruasd"], "selection.ruasd")
    _expect_keys(
        ruasd,
        {"excluded_subsets", "require_source_text", "min_per_stratum", "max_per_stratum"},
        "selection.ruasd",
    )
    if ruasd["require_source_text"] is not True:
        raise V4SelectionError("v4 RuASD selection must require source text.")
    minimum = _positive_int(ruasd["min_per_stratum"], "ruasd.min_per_stratum")
    maximum = _positive_int(ruasd["max_per_stratum"], "ruasd.max_per_stratum")
    if minimum > maximum:
        raise V4SelectionError("RuASD min_per_stratum exceeds max_per_stratum.")
    ksc2 = _mapping(selection["ksc2"], "selection.ksc2")
    _expect_keys(ksc2, {"component_quotas"}, "selection.ksc2")
    component_quotas = _positive_int_mapping(ksc2["component_quotas"], "component_quotas")
    if sum(component_quotas.values()) != candidate:
        raise V4SelectionError("KSC2 component quotas do not sum to candidate_rows_per_cell.")
    generator_values = selection["kk_spoof_generators"]
    if not isinstance(generator_values, list) or not generator_values:
        raise V4SelectionError("kk_spoof_generators must be a non-empty array.")
    generator_quotas: dict[str, int] = {}
    generator_families: dict[str, str] = {}
    for index, value in enumerate(generator_values, start=1):
        item = _mapping(value, f"kk_spoof_generators[{index}]")
        _expect_keys(item, {"route_id", "generator_family", "quota"}, "KK generator")
        route_id = _string(item["route_id"], "KK generator route_id")
        if route_id in generator_quotas:
            raise V4SelectionError(f"Duplicate KK generator route: {route_id!r}.")
        generator_quotas[route_id] = _positive_int(item["quota"], "KK generator quota")
        generator_families[route_id] = _string(
            item["generator_family"], "KK generator family"
        )
    if sum(generator_quotas.values()) != candidate or len(set(generator_families.values())) < 4:
        raise V4SelectionError("KK generator quotas/families do not satisfy the v4 contract.")
    roles = _mapping(raw["roles"], "roles")
    _expect_keys(roles, {"train", "dev", "calibration", "final"}, "roles")
    source_lineage_roots, tts_family_roots = _load_role_isolation(
        raw["role_isolation"], generator_families
    )
    claims = _mapping(raw["claims"], "claims")
    expected_claims: dict[str, object] = {
        "speaker_independence": "not_verified_speaker_independent",
        "calibration_language_scope": "ru_only",
        "kk_probability_claim": False,
        "historical_final_assets_allowed": False,
        "synthesis_authorized": False,
        "training_authorized": False,
    }
    if claims != expected_claims:
        raise V4SelectionError("v4 selection claims differ from the fail-closed contract.")
    return V4SelectionConfig(
        protocol_id=_string(raw["protocol_id"], "protocol_id"),
        capacity_receipt_path=_safe_path(capacity["path"], "capacity receipt path"),
        capacity_receipt_sha256=_sha256(capacity["sha256"], "capacity receipt sha256"),
        selection_seed=_string(selection["seed"], "selection seed"),
        target_rows_per_cell=target,
        candidate_rows_per_cell=candidate,
        ruasd_excluded_subsets=frozenset(
            _string_tuple(ruasd["excluded_subsets"], "ruasd.excluded_subsets")
        ),
        ruasd_require_source_text=True,
        ruasd_min_per_stratum=minimum,
        ruasd_max_per_stratum=maximum,
        ksc2_component_quotas=component_quotas,
        kk_generator_quotas=generator_quotas,
        kk_generator_families=generator_families,
        roles=roles,
        source_lineage_roots=source_lineage_roots,
        tts_family_roots=tts_family_roots,
    )


def load_v4_exposure_inventory(manifest_root: Path, project_root: Path) -> V4ExposureInventory:
    root = project_root.resolve(strict=True)
    manifest_directory = manifest_root.resolve(strict=True)
    _relative_to_root(manifest_directory, root, "Manifest root")
    rows: list[ManifestRow] = []
    bindings: list[dict[str, object]] = []
    for path in sorted(manifest_directory.rglob("*.csv")):
        if not _manifest_like_csv(path):
            continue
        loaded = load_manifest(path)
        rows.extend(loaded)
        bindings.append(
            {
                "path": _relative_to_root(path.resolve(strict=True), root, "Manifest"),
                "sha256": sha256_file(path),
                "rows": len(loaded),
            }
        )
    if not rows:
        raise V4SelectionError("v4 exposure inventory has no manifest rows.")
    return V4ExposureInventory(
        manifest_bindings=tuple(bindings),
        rows=len(rows),
        sample_ids=frozenset(row.sample_id for row in rows),
        audio_sha256=frozenset(row.sha256 for row in rows),
        text_hashes=frozenset(row.text_hash for row in rows),
        parent_group_ids=frozenset(row.parent_group_id for row in rows),
        speaker_group_ids=frozenset(row.speaker_pseudo_id for row in rows),
    )


def select_v4_ruasd_candidates(
    archive_dir: Path,
    catalog: Mapping[str, RuAsdArchiveSpec],
    *,
    config: V4SelectionConfig,
    exposure: V4ExposureInventory,
    progress_callback: V4SelectionProgressCallback | None = None,
) -> V4SourceSelection:
    """Select source-text-present non-CommonVoice RuASD rows without materializing audio."""

    archive_paths = _validate_archive_set(archive_dir, catalog)
    best_by_text: dict[str, tuple[int, RuAsdResearchRecord]] = {}
    rejections: Counter[str] = Counter()
    for completed, archive_name in enumerate(sorted(catalog), start=1):
        _validate_archive_size(archive_paths[archive_name], catalog[archive_name])
        metadata = load_ruasd_archive_records(archive_paths[archive_name])
        for record in _eligible_records(archive_name, metadata):
            if record.subset in config.ruasd_excluded_subsets:
                rejections["excluded_source_subset"] += 1
                continue
            if config.ruasd_require_source_text and not record.has_source_text:
                rejections["source_text_missing"] += 1
                continue
            project_sample_id = f"ruasd_ru_v1_full:{record.record_key}"
            if project_sample_id in exposure.sample_ids:
                rejections["historical_sample_id"] += 1
                continue
            if record.text_hash in exposure.text_hashes:
                rejections["historical_text_hash"] += 1
                continue
            rank = _rank(config.selection_seed, "ruasd-unique-text", record.record_key)
            prior = best_by_text.get(record.text_hash)
            if prior is None or (rank, record.record_key) < (prior[0], prior[1].record_key):
                if prior is not None:
                    rejections["duplicate_candidate_text"] += 1
                best_by_text[record.text_hash] = (rank, record)
            else:
                rejections["duplicate_candidate_text"] += 1
        if progress_callback is not None:
            progress_callback(completed, len(catalog), archive_name)
    unique_records = [value[1] for value in best_by_text.values()]
    counts = Counter(record.stratum for record in unique_records)
    quotas = _bounded_quotas(
        counts,
        total_per_label=config.candidate_rows_per_cell,
        minimum=config.ruasd_min_per_stratum,
        maximum=config.ruasd_max_per_stratum,
    )
    selected = _select_ranked_ruasd(unique_records, quotas, config.selection_seed)
    rows: list[V4CandidateRow] = []
    ranks_by_label: Counter[str] = Counter()
    for record in sorted(
        selected,
        key=lambda item: (
            item.label,
            _rank(config.selection_seed, f"ru-{item.label}", item.record_key),
            item.record_key,
        ),
    ):
        ranks_by_label[record.label] += 1
        selection_rank = ranks_by_label[record.label]
        candidate_id = f"ruasd_ru_v1_full:{record.record_key}"
        rows.append(
            V4CandidateRow(
                selection_rank=selection_rank,
                target_state=(
                    "target" if selection_rank <= config.target_rows_per_cell else "reserve"
                ),
                role="train",
                language="ru",
                label=record.label,
                pair_id=f"v4:ru:{record.label}:{record.record_key}",
                candidate_id=candidate_id,
                source_id="ruasd_ru_v1_full",
                source_lineage_id="ruasd_ru_v1_full:raw:train_only",
                source_component=record.subset,
                archive_audio_member=f"{record.sample_id}.wav",
                archive_transcript_member=f"{record.sample_id}.json",
                text_hash=record.text_hash,
                canonical_text_hash=record.text_hash,
                parent_group_id=f"ruasd_ru_v1_full:source-record:{record.record_key}",
                speaker_group_status="unknown_by_source",
                generator_route_id=record.subset if record.label == "spoof" else "",
                generator_family="source_tts_route_unverified_family"
                if record.label == "spoof"
                else "",
                raw_audio_sha256="",
                decoded_audio_sha256="",
                asset_state="source_audio_unmaterialized",
            )
        )
    selected_counts = Counter(record.stratum for record in selected)
    _require_cell_counts(rows, config.candidate_rows_per_cell, cells=("ru/bonafide", "ru/spoof"))
    return V4SourceSelection(
        rows=tuple(rows),
        available_by_stratum=dict(sorted(counts.items())),
        selected_by_stratum=dict(sorted(selected_counts.items())),
        rejection_counts=dict(sorted(rejections.items())),
    )


def select_v4_ksc2_candidates(
    candidates: Sequence[Ksc2TextCandidate],
    *,
    config: V4SelectionConfig,
    exposure: V4ExposureInventory,
) -> V4SourceSelection:
    """Select component-balanced KSC2 train pairs and assign fresh spoof generator routes."""

    best_by_text: dict[str, tuple[int, Ksc2TextCandidate]] = {}
    rejections: Counter[str] = Counter()
    allowed_components = set(config.ksc2_component_quotas)
    for candidate in candidates:
        if candidate.component not in allowed_components:
            raise V4SelectionError(
                f"KSC2 scanner returned disallowed component: {candidate.component!r}."
            )
        sample_id = f"ksc2_v1:{candidate.candidate_id}"
        if sample_id in exposure.sample_ids:
            rejections["historical_sample_id"] += 1
            continue
        if (
            candidate.transcript_sha256 in exposure.text_hashes
            or candidate.canonical_text_sha256 in exposure.text_hashes
        ):
            rejections["historical_text_hash"] += 1
            continue
        rank = _rank(config.selection_seed, "ksc2-unique-text", candidate.candidate_id)
        prior = best_by_text.get(candidate.canonical_text_sha256)
        if prior is None or (rank, candidate.candidate_id) < (prior[0], prior[1].candidate_id):
            if prior is not None:
                rejections["duplicate_candidate_text"] += 1
            best_by_text[candidate.canonical_text_sha256] = (rank, candidate)
        else:
            rejections["duplicate_candidate_text"] += 1
    by_component: dict[str, list[Ksc2TextCandidate]] = {
        component: [] for component in config.ksc2_component_quotas
    }
    for _rank_value, candidate in best_by_text.values():
        by_component[candidate.component].append(candidate)
    selected: list[Ksc2TextCandidate] = []
    for component, quota in sorted(config.ksc2_component_quotas.items()):
        ranked = sorted(
            by_component[component],
            key=lambda item: (
                _rank(config.selection_seed, f"ksc2:{component}", item.candidate_id),
                item.candidate_id,
            ),
        )
        if len(ranked) < quota:
            raise V4SelectionError(
                f"KSC2 component {component!r} has {len(ranked)} eligible rows, needs {quota}."
            )
        selected.extend(ranked[:quota])
    selected = sorted(
        selected,
        key=lambda item: (
            _rank(config.selection_seed, "kk-train", item.candidate_id),
            item.candidate_id,
        ),
    )
    # Interleave route quotas deterministically instead of creating long family blocks.
    route_queues: dict[str, int] = dict(config.kk_generator_quotas)
    interleaved_routes: list[str] = []
    route_order = sorted(
        route_queues,
        key=lambda route_id: (
            _rank(config.selection_seed, "generator-order", route_id),
            route_id,
        ),
    )
    while len(interleaved_routes) < len(selected):
        for route_id in route_order:
            if route_queues[route_id] > 0:
                interleaved_routes.append(route_id)
                route_queues[route_id] -= 1
    rows: list[V4CandidateRow] = []
    for selection_rank, (candidate, route_id) in enumerate(
        zip(selected, interleaved_routes, strict=True), start=1
    ):
        target_state = "target" if selection_rank <= config.target_rows_per_cell else "reserve"
        pair_id = f"v4:kk:{candidate.candidate_id}"
        rows.append(
            V4CandidateRow(
                selection_rank=selection_rank,
                target_state=target_state,
                role="train",
                language="kk",
                label="bonafide",
                pair_id=pair_id,
                candidate_id=f"ksc2_v1:{candidate.candidate_id}",
                source_id="ksc2_v1",
                source_lineage_id="ksc2_v1:nonlegacy_train:train_only",
                source_component=candidate.component,
                archive_audio_member=candidate.archive_audio_member,
                archive_transcript_member=candidate.archive_transcript_member,
                text_hash=candidate.transcript_sha256,
                canonical_text_hash=candidate.canonical_text_sha256,
                parent_group_id=f"ksc2_v1:recording:{candidate.candidate_id}",
                speaker_group_status="unknown_by_source",
                generator_route_id="",
                generator_family="",
                raw_audio_sha256="",
                decoded_audio_sha256="",
                asset_state="source_audio_unmaterialized",
            )
        )
        rows.append(
            V4CandidateRow(
                selection_rank=selection_rank,
                target_state=target_state,
                role="train",
                language="kk",
                label="spoof",
                pair_id=pair_id,
                candidate_id=f"xlsr_sls_model_v4_kk_spoof:{candidate.candidate_id}",
                source_id=f"xlsr_sls_model_v4_kk_spoof:{route_id}",
                source_lineage_id=(
                    f"ksc2_v1:nonlegacy_train:text_only:{route_id}:train_only"
                ),
                source_component=candidate.component,
                archive_audio_member="",
                archive_transcript_member=candidate.archive_transcript_member,
                text_hash=candidate.transcript_sha256,
                canonical_text_hash=candidate.canonical_text_sha256,
                parent_group_id=f"xlsr_sls_model_v4_kk_spoof:route:{route_id}",
                speaker_group_status="unknown_by_source",
                generator_route_id=route_id,
                generator_family=config.kk_generator_families[route_id],
                raw_audio_sha256="",
                decoded_audio_sha256="",
                asset_state="synthesis_planned_not_authorized",
            )
        )
    available_counts = {component: len(values) for component, values in by_component.items()}
    selected_counts = Counter(candidate.component for candidate in selected)
    _require_cell_counts(rows, config.candidate_rows_per_cell, cells=("kk/bonafide", "kk/spoof"))
    return V4SourceSelection(
        rows=tuple(rows),
        available_by_stratum=dict(sorted(available_counts.items())),
        selected_by_stratum=dict(sorted(selected_counts.items())),
        rejection_counts=dict(sorted(rejections.items())),
    )


def publish_v4_train_candidate_selection(
    *,
    output_csv: Path,
    output_receipt: Path,
    rows: Sequence[V4CandidateRow],
    config_path: Path,
    config: V4SelectionConfig,
    exposure: V4ExposureInventory,
    ruasd_selection: V4SourceSelection,
    ksc2_selection: V4SourceSelection,
    created_at: str,
    source_bindings: Mapping[str, Mapping[str, object]],
) -> None:
    """Publish one immutable metadata-only candidate packet and accounting receipt."""

    if (
        output_csv.exists()
        or output_receipt.exists()
        or not output_csv.parent.is_dir()
        or not output_receipt.parent.is_dir()
        or output_csv == output_receipt
    ):
        raise V4SelectionError("Unsafe v4 selection output destinations.")
    _validate_combined_rows(rows, config)
    csv_published = False
    try:
        with (
            tempfile.TemporaryDirectory(
                prefix="kds-v4-selection-csv-", dir=output_csv.parent
            ) as csv_stage_name,
            tempfile.TemporaryDirectory(
                prefix="kds-v4-selection-receipt-", dir=output_receipt.parent
            ) as receipt_stage_name,
        ):
            staged_csv = Path(csv_stage_name) / output_csv.name
            staged_receipt = Path(receipt_stage_name) / output_receipt.name
            with staged_csv.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=V4_CANDIDATE_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(asdict(row) for row in rows)
            receipt = {
                "schema_version": V4_SELECTION_SCHEMA_VERSION,
                "protocol_id": config.protocol_id,
                "created_at": created_at,
                "state": "metadata_selection_frozen_audio_hashes_and_qa_pending",
                "claims": {
                    "audio_extraction_performed": False,
                    "synthesis_performed": False,
                    "training_performed": False,
                    "ready_train_rows_certified": False,
                    "speaker_independence": "not_verified_speaker_independent",
                    "historical_sample_or_text_collisions_selected": False,
                    "historical_exact_audio_collision_status": (
                        "not_checked_audio_unmaterialized"
                    ),
                },
                "bindings": {
                    "config": {
                        "path": config_path.as_posix(),
                        "sha256": sha256_file(config_path),
                    },
                    "capacity_receipt": {
                        "path": config.capacity_receipt_path,
                        "sha256": config.capacity_receipt_sha256,
                    },
                    "sources": source_bindings,
                    "project_history_manifests": list(exposure.manifest_bindings),
                    "project_history_rows_with_version_duplicates": exposure.rows,
                },
                "selection": {
                    "seed": config.selection_seed,
                    "target_rows_per_cell": config.target_rows_per_cell,
                    "candidate_rows_per_cell": config.candidate_rows_per_cell,
                    "candidate_csv": output_csv.as_posix(),
                    "candidate_csv_sha256": sha256_file(staged_csv),
                    "rows": len(rows),
                    "cell_counts": _cell_counts(rows),
                    "ruasd": {
                        "available_by_stratum": ruasd_selection.available_by_stratum,
                        "selected_by_stratum": ruasd_selection.selected_by_stratum,
                        "rejection_counts": ruasd_selection.rejection_counts,
                    },
                    "ksc2": {
                        "available_by_component": ksc2_selection.available_by_stratum,
                        "selected_by_component": ksc2_selection.selected_by_stratum,
                        "rejection_counts": ksc2_selection.rejection_counts,
                    },
                    "kk_generator_quotas": config.kk_generator_quotas,
                },
                "roles": config.roles,
                "role_isolation": {
                    "source_lineage_roots": config.source_lineage_roots,
                    "tts_family_roots": config.tts_family_roots,
                },
                "next_gate": (
                    "materialize source audio; compute raw/decoded hashes and audio fingerprints; "
                    "run QA/VAD and cross-role leakage closure before synthesis"
                ),
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # Hard-link publication is atomic per file and fails if a concurrent
            # writer has already claimed either write-once destination.
            os.link(staged_csv, output_csv)
            csv_published = True
            os.link(staged_receipt, output_receipt)
    except OSError as error:
        if csv_published:
            try:
                if output_csv.samefile(staged_csv):
                    output_csv.unlink()
            except OSError:
                pass
        raise V4SelectionError(f"Cannot publish v4 metadata selection: {error}") from error


def _bounded_quotas(
    counts: Mapping[str, int], *, total_per_label: int, minimum: int, maximum: int
) -> dict[str, int]:
    by_label: dict[str, dict[str, int]] = {"bonafide": {}, "spoof": {}}
    for stratum, count in counts.items():
        label = stratum.split("/", 1)[0]
        if label not in by_label:
            raise V4SelectionError(f"Unexpected RuASD label stratum: {stratum!r}.")
        by_label[label][stratum] = count
    result: dict[str, int] = {}
    for label, strata in by_label.items():
        if not strata or minimum * len(strata) > total_per_label:
            raise V4SelectionError(f"Cannot satisfy minimum RuASD quotas for {label}.")
        quotas = {name: minimum for name in strata}
        if any(strata[name] < minimum for name in strata):
            raise V4SelectionError(f"A RuASD {label} stratum is below the frozen minimum.")
        remaining = total_per_label - sum(quotas.values())
        order = sorted(
            strata,
            key=lambda name: (-(strata[name] - minimum), name),
        )
        while remaining:
            progressed = False
            for name in order:
                ceiling = min(maximum, strata[name])
                if quotas[name] < ceiling:
                    quotas[name] += 1
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                raise V4SelectionError(
                    f"Cannot satisfy bounded RuASD quota for {label}; "
                    f"available={dict(sorted(strata.items()))}, maximum={maximum}."
                )
        result.update(quotas)
    return result


def _select_ranked_ruasd(
    records: Sequence[RuAsdResearchRecord], quotas: Mapping[str, int], seed: str
) -> list[RuAsdResearchRecord]:
    heaps: dict[str, list[tuple[int, str, RuAsdResearchRecord]]] = {
        stratum: [] for stratum in quotas
    }
    for record in records:
        quota = quotas.get(record.stratum)
        if quota is None:
            continue
        rank = _rank(seed, f"ruasd:{record.stratum}", record.record_key)
        entry = (-rank, record.record_key, record)
        heap = heaps[record.stratum]
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = [record for heap in heaps.values() for _rank_value, _key, record in heap]
    actual = Counter(record.stratum for record in selected)
    if dict(actual) != dict(quotas):
        raise V4SelectionError("RuASD deterministic selection missed a stratum quota.")
    return selected


def _validate_combined_rows(rows: Sequence[V4CandidateRow], config: V4SelectionConfig) -> None:
    if not rows:
        raise V4SelectionError("v4 candidate selection is empty.")
    _require_cell_counts(
        rows,
        config.candidate_rows_per_cell,
        cells=("ru/bonafide", "ru/spoof", "kk/bonafide", "kk/spoof"),
    )
    candidate_ids = [row.candidate_id for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise V4SelectionError("v4 candidate selection has duplicate candidate_id values.")
    roles_by_text: dict[str, set[str]] = {}
    for row in rows:
        roles_by_text.setdefault(row.canonical_text_hash, set()).add(row.role)
    if any(len(roles) > 1 for roles in roles_by_text.values()):
        raise V4SelectionError("v4 candidate text crosses roles.")


def _require_cell_counts(
    rows: Sequence[V4CandidateRow], expected: int, *, cells: Sequence[str]
) -> None:
    counts = _cell_counts(rows)
    for cell in cells:
        if counts.get(cell) != expected:
            raise V4SelectionError(
                f"v4 candidate cell {cell!r} has {counts.get(cell, 0)} rows, expected {expected}."
            )


def _cell_counts(rows: Sequence[V4CandidateRow]) -> dict[str, int]:
    return dict(sorted(Counter(f"{row.language}/{row.label}" for row in rows).items()))


def _rank(seed: str, namespace: str, value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest(), "big"
    )


def _manifest_like_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return REQUIRED_FIELDS.issubset(next(csv.reader(handle), []))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4SelectionError(f"Cannot inspect historical manifest {path}: {error}") from error


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4SelectionError(f"Cannot read {label} {path}: {error}") from error
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4SelectionError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _expect_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        raise V4SelectionError(f"{label} fields differ: missing={missing}, unknown={unknown}.")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V4SelectionError(f"{label} must be a non-empty string.")
    return value.strip()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise V4SelectionError(f"{label} must be an array.")
    result = tuple(_string(item, label) for item in value)
    if len(result) != len(set(result)):
        raise V4SelectionError(f"{label} contains duplicates.")
    return result


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4SelectionError(f"{label} must be a positive integer.")
    return value


def _positive_int_mapping(value: object, label: str) -> dict[str, int]:
    raw = _mapping(value, label)
    return {key: _positive_int(item, f"{label}.{key}") for key, item in raw.items()}


def _load_role_isolation(
    value: object, generator_families: Mapping[str, str]
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    isolation = _mapping(value, "role_isolation")
    _expect_keys(
        isolation,
        {"source_lineage_roots", "tts_family_roots"},
        "role_isolation",
    )
    expected_roles = {"train", "dev", "calibration", "final"}
    result: list[dict[str, tuple[str, ...]]] = []
    for field in ("source_lineage_roots", "tts_family_roots"):
        raw = _mapping(isolation[field], f"role_isolation.{field}")
        _expect_keys(raw, expected_roles, f"role_isolation.{field}")
        roots = {
            role: _string_tuple(raw[role], f"role_isolation.{field}.{role}")
            for role in sorted(expected_roles)
        }
        owners: dict[str, str] = {}
        for role, values in roots.items():
            for root in values:
                prior = owners.setdefault(root, role)
                if prior != role:
                    raise V4SelectionError(
                        f"Role-isolation root {root!r} crosses {prior!r} and {role!r}."
                    )
        result.append(roots)
    source_roots, tts_roots = result
    missing_train_families = set(generator_families.values()).difference(
        tts_roots["train"]
    )
    if missing_train_families:
        raise V4SelectionError(
            "KK train generator families are missing from role isolation: "
            f"{sorted(missing_train_families)}."
        )
    return source_roots, tts_roots


def _safe_path(value: object, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text or text in {"", "."}:
        raise V4SelectionError(f"{label} must be a safe relative path.")
    return path.as_posix()


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise V4SelectionError(f"{label} must be a SHA-256 digest.")
    return digest


def _relative_to_root(path: Path, root: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise V4SelectionError(f"{label} escapes project root: {path}") from error
