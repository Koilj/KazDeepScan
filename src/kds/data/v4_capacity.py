"""Fail-closed local-capacity gate for the KazDeepScan XLS-R+SLS v4 protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import sha256_file
from kds.data.licenses import (
    APPROVED_LICENSE_STATUSES,
    LicenseLedgerEntry,
    load_license_ledger,
)
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest
from kds.data.research_tts import (
    ResearchTtsModelLock,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)

V4_GATE_A_SCHEMA_VERSION = 1
V4_GATE_A_DECISIONS = frozenset(
    {"proceed_24k", "proceed_20k_to_29999", "stop_local_capacity_exhausted"}
)


class V4CapacityError(ValueError):
    """Raised when Gate A cannot prove its inputs or conservative capacity."""


@dataclass(frozen=True, slots=True)
class V4Targets:
    preferred_train_rows: int
    minimum_train_rows: int
    maximum_train_rows: int
    preferred_rows_per_cell: int
    minimum_rows_per_cell: int
    final_rows_per_cell: int
    minimum_train_tts_families: int


@dataclass(frozen=True, slots=True)
class V4TtsRoute:
    route_id: str
    ledger_source_id: str
    lock_path: str
    model_root: str


@dataclass(frozen=True, slots=True)
class V4GateAConfig:
    protocol_id: str
    purpose: str
    inventory_globs: tuple[str, ...]
    targets: V4Targets
    ruasd_ledger_source_id: str
    ruasd_catalog_path: str
    ruasd_raw_bonafide_key: str
    ruasd_raw_spoof_key: str
    ruasd_excluded_subset_keys: tuple[str, ...]
    ruasd_history_source_ids: tuple[str, ...]
    common_voice_ledger_source_id: str
    ksc2_ledger_source_id: str
    ksc2_artifact_lock_path: str
    ksc2_allowed_train_components: tuple[str, ...]
    ksc2_history_source_ids: tuple[str, ...]
    tts_routes: tuple[V4TtsRoute, ...]


def load_v4_gate_a_config(path: Path) -> V4GateAConfig:
    """Load the exact v4 Gate A contract and reject implicit defaults."""

    raw = _json_object(path, "v4 Gate A config")
    _expect_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "purpose",
            "inventory",
            "targets",
            "sources",
            "claims",
        },
        "v4 Gate A config",
    )
    if raw["schema_version"] != V4_GATE_A_SCHEMA_VERSION:
        raise V4CapacityError(
            f"v4 Gate A schema_version must be {V4_GATE_A_SCHEMA_VERSION}."
        )
    if raw["purpose"] != "personal_research":
        raise V4CapacityError("v4 Gate A purpose must be personal_research.")

    inventory = _mapping(raw["inventory"], "inventory")
    _expect_keys(inventory, {"file_globs"}, "inventory")
    inventory_globs = _string_tuple(inventory["file_globs"], "inventory.file_globs")
    if not inventory_globs:
        raise V4CapacityError("inventory.file_globs must not be empty.")

    targets_raw = _mapping(raw["targets"], "targets")
    target_fields = {
        "preferred_train_rows",
        "minimum_train_rows",
        "maximum_train_rows",
        "preferred_rows_per_cell",
        "minimum_rows_per_cell",
        "final_rows_per_cell",
        "minimum_train_tts_families",
    }
    _expect_keys(targets_raw, target_fields, "targets")
    targets = V4Targets(
        **{name: _positive_int(targets_raw[name], f"targets.{name}") for name in target_fields}
    )
    if (
        targets.minimum_train_rows != targets.minimum_rows_per_cell * 4
        or targets.preferred_train_rows != targets.preferred_rows_per_cell * 4
        or not (
            targets.minimum_train_rows
            <= targets.preferred_train_rows
            <= targets.maximum_train_rows
        )
    ):
        raise V4CapacityError("v4 target totals and four balanced cell targets disagree.")

    sources = _mapping(raw["sources"], "sources")
    _expect_keys(sources, {"ruasd", "common_voice", "ksc2", "tts_routes"}, "sources")
    ruasd = _mapping(sources["ruasd"], "sources.ruasd")
    _expect_keys(
        ruasd,
        {
            "ledger_source_id",
            "catalog_path",
            "raw_bonafide_record_key",
            "raw_spoof_record_key",
            "excluded_subset_keys",
            "history_source_ids",
        },
        "sources.ruasd",
    )
    common_voice = _mapping(sources["common_voice"], "sources.common_voice")
    _expect_keys(common_voice, {"ledger_source_id"}, "sources.common_voice")
    ksc2 = _mapping(sources["ksc2"], "sources.ksc2")
    _expect_keys(
        ksc2,
        {
            "ledger_source_id",
            "artifact_lock_path",
            "allowed_train_components",
            "history_source_ids",
        },
        "sources.ksc2",
    )
    tts_values = sources["tts_routes"]
    if not isinstance(tts_values, list) or not tts_values:
        raise V4CapacityError("sources.tts_routes must be a non-empty array.")
    tts_routes = tuple(_parse_tts_route(value, index) for index, value in enumerate(tts_values, 1))
    if len({route.route_id for route in tts_routes}) != len(tts_routes):
        raise V4CapacityError("sources.tts_routes contains duplicate route_id values.")

    claims = _mapping(raw["claims"], "claims")
    expected_claims: dict[str, object] = {
        "local_sources_only": True,
        "speaker_independence": "not_verified_speaker_independent",
        "capacity_is_pre_qa": True,
        "training_authorized_by_gate_a": False,
        "synthesis_authorized_by_gate_a": False,
    }
    _expect_keys(claims, set(expected_claims), "claims")
    if claims != expected_claims:
        raise V4CapacityError("v4 Gate A claims must retain the fail-closed contract.")

    return V4GateAConfig(
        protocol_id=_string(raw["protocol_id"], "protocol_id"),
        purpose="personal_research",
        inventory_globs=inventory_globs,
        targets=targets,
        ruasd_ledger_source_id=_string(ruasd["ledger_source_id"], "ruasd.ledger_source_id"),
        ruasd_catalog_path=_safe_path(ruasd["catalog_path"], "ruasd.catalog_path"),
        ruasd_raw_bonafide_key=_string(
            ruasd["raw_bonafide_record_key"], "ruasd.raw_bonafide_record_key"
        ),
        ruasd_raw_spoof_key=_string(
            ruasd["raw_spoof_record_key"], "ruasd.raw_spoof_record_key"
        ),
        ruasd_excluded_subset_keys=_string_tuple(
            ruasd["excluded_subset_keys"], "ruasd.excluded_subset_keys"
        ),
        ruasd_history_source_ids=_string_tuple(
            ruasd["history_source_ids"], "ruasd.history_source_ids"
        ),
        common_voice_ledger_source_id=_string(
            common_voice["ledger_source_id"], "common_voice.ledger_source_id"
        ),
        ksc2_ledger_source_id=_string(ksc2["ledger_source_id"], "ksc2.ledger_source_id"),
        ksc2_artifact_lock_path=_safe_path(
            ksc2["artifact_lock_path"], "ksc2.artifact_lock_path"
        ),
        ksc2_allowed_train_components=_string_tuple(
            ksc2["allowed_train_components"], "ksc2.allowed_train_components"
        ),
        ksc2_history_source_ids=_string_tuple(
            ksc2["history_source_ids"], "ksc2.history_source_ids"
        ),
        tts_routes=tts_routes,
    )


def audit_v4_local_capacity(
    *,
    config_path: Path,
    project_root: Path,
    license_ledger_path: Path,
    ruasd_audit_path: Path,
    ksc2_audit_path: Path,
    common_voice_screen_path: Path,
    audited_at: str,
) -> dict[str, object]:
    """Verify Gate A inputs and return an immutable-ready local-capacity receipt."""

    _iso_timestamp(audited_at)
    root = project_root.resolve(strict=True)
    config_resolved = config_path.resolve(strict=True)
    ledger_resolved = license_ledger_path.resolve(strict=True)
    _relative_to_root(config_resolved, root, "Gate A config")
    _relative_to_root(ledger_resolved, root, "License ledger")
    config = load_v4_gate_a_config(config_resolved)
    ledger = load_license_ledger(ledger_resolved)

    inventory = _project_history_inventory(root, config.inventory_globs)
    manifest_rows = cast(list[ManifestRow], inventory.pop("_manifest_rows"))
    history_counts = _history_counts(manifest_rows)

    ruasd_report = _json_object(ruasd_audit_path, "RuASD audit")
    ksc2_report = _json_object(ksc2_audit_path, "KSC2 audit")
    common_voice_report = _json_object(common_voice_screen_path, "Common Voice screen")

    license_receipts = _verify_licenses_and_pins(
        root=root,
        config=config,
        ledger=ledger,
        ruasd_report=ruasd_report,
        ksc2_report=ksc2_report,
        common_voice_report=common_voice_report,
    )
    model_receipts, verified_families = _verify_tts_routes(root, config, ledger)

    ruasd_counts = _int_mapping(ruasd_report.get("record_counts"), "RuASD record_counts")
    ruasd_subset_counts = _int_mapping(
        ruasd_report.get("subset_counts"), "RuASD subset_counts"
    )
    raw_ru_bonafide = _required_count(ruasd_counts, config.ruasd_raw_bonafide_key)
    raw_ru_spoof = _required_count(ruasd_counts, config.ruasd_raw_spoof_key)
    excluded_common_voice = sum(
        _required_count(ruasd_subset_counts, key) for key in config.ruasd_excluded_subset_keys
    )
    historical_ru_bonafide = _history_source_label_count(
        history_counts, config.ruasd_history_source_ids, "bonafide"
    )
    historical_ru_spoof = _history_source_label_count(
        history_counts, config.ruasd_history_source_ids, "spoof"
    )
    ru_bonafide_capacity = max(
        0, raw_ru_bonafide - excluded_common_voice - historical_ru_bonafide
    )
    ru_spoof_capacity = max(0, raw_ru_spoof - historical_ru_spoof)

    files_by_component = _int_mapping(
        ksc2_report.get("files_by_component"), "KSC2 files_by_component"
    )
    component_pairs = {
        component: _required_count(files_by_component, component) // 2
        for component in config.ksc2_allowed_train_components
    }
    historical_kk_bonafide = _history_source_label_count(
        history_counts, config.ksc2_history_source_ids, "bonafide"
    )
    kk_bonafide_capacity = max(0, sum(component_pairs.values()) - historical_kk_bonafide)
    enough_tts_families = len(verified_families) >= config.targets.minimum_train_tts_families
    kk_spoof_capacity = kk_bonafide_capacity if enough_tts_families else 0

    common_voice_exclusion = _mapping(
        common_voice_report.get("strict_group_exclusion"),
        "Common Voice strict_group_exclusion",
    )
    cv_survivors = _positive_or_zero_int(
        common_voice_exclusion.get("surviving_records"),
        "Common Voice surviving_records",
    )

    cells = {
        "ru/bonafide": ru_bonafide_capacity,
        "ru/spoof": ru_spoof_capacity,
        "kk/bonafide": kk_bonafide_capacity,
        "kk/spoof": kk_spoof_capacity,
    }
    if all(value >= config.targets.preferred_rows_per_cell for value in cells.values()):
        decision = "proceed_24k"
    elif all(value >= config.targets.minimum_rows_per_cell for value in cells.values()):
        decision = "proceed_20k_to_29999"
    else:
        decision = "stop_local_capacity_exhausted"
    if decision not in V4_GATE_A_DECISIONS:
        raise AssertionError("Unexpected Gate A decision.")

    return {
        "schema_version": V4_GATE_A_SCHEMA_VERSION,
        "protocol_id": config.protocol_id,
        "audited_at": audited_at,
        "decision": decision,
        "decision_scope": (
            "candidate selection may start; exact row eligibility and QA remain pending"
            if decision != "stop_local_capacity_exhausted"
            else "no v4 data preparation, synthesis, or training may start"
        ),
        "claims": {
            "local_sources_only": True,
            "new_dataset_search_performed": False,
            "audio_extraction_performed_by_gate_a": False,
            "synthetic_audio_generated_by_gate_a": False,
            "training_performed_by_gate_a": False,
            "ready_train_rows_certified": False,
            "capacity_is_conservative_pre_qa_candidate_upper_bound": True,
            "speaker_independence": "not_verified_speaker_independent",
        },
        "bindings": {
            "config": _file_binding(config_resolved, root),
            "license_ledger": _file_binding(ledger_resolved, root),
            "ruasd_audit": _external_file_binding(ruasd_audit_path),
            "ksc2_audit": _external_file_binding(ksc2_audit_path),
            "common_voice_screen": _external_file_binding(common_voice_screen_path),
        },
        "project_history_inventory": inventory,
        "history_manifest_unique_rows": history_counts,
        "license_and_source_identity": license_receipts,
        "verified_tts_routes": model_receipts,
        "capacity": {
            "targets": {
                "preferred_train_rows": config.targets.preferred_train_rows,
                "minimum_train_rows": config.targets.minimum_train_rows,
                "maximum_train_rows": config.targets.maximum_train_rows,
                "preferred_rows_per_cell": config.targets.preferred_rows_per_cell,
                "minimum_rows_per_cell": config.targets.minimum_rows_per_cell,
            },
            "cells": {
                "ru/bonafide": {
                    "raw_ruasd_records": raw_ru_bonafide,
                    "excluded_ruasd_common_voice_strata": excluded_common_voice,
                    "conservative_historical_row_exclusion": historical_ru_bonafide,
                    "pre_qa_candidate_upper_bound": ru_bonafide_capacity,
                },
                "ru/spoof": {
                    "raw_ruasd_records": raw_ru_spoof,
                    "conservative_historical_row_exclusion": historical_ru_spoof,
                    "pre_qa_candidate_upper_bound": ru_spoof_capacity,
                },
                "kk/bonafide": {
                    "ksc2_allowed_component_pairs": component_pairs,
                    "conservative_historical_row_exclusion": historical_kk_bonafide,
                    "pre_qa_candidate_upper_bound": kk_bonafide_capacity,
                },
                "kk/spoof": {
                    "fresh_text_capacity_from_kk_bonafide": kk_bonafide_capacity,
                    "verified_generator_families": sorted(verified_families),
                    "minimum_generator_families": config.targets.minimum_train_tts_families,
                    "pre_qa_synthesis_upper_bound": kk_spoof_capacity,
                },
            },
            "common_voice_ru_final_reservoir": {
                "fresh_group_screen_surviving_records": cv_survivors,
                "required_final_rows": config.targets.final_rows_per_cell,
                "capacity_check_passed": cv_survivors >= config.targets.final_rows_per_cell,
            },
        },
        "next_gate": {
            "name": "v4_role_selection_and_row_level_leakage_graph",
            "required_before_synthesis": True,
            "required_before_training": True,
            "required_fields": [
                "raw_and_decoded_audio_sha256",
                "canonical_audio_fingerprint",
                "exact_and_normalized_text_hash",
                "source_lineage_id",
                "parent_group_id",
                "available_speaker_or_session_group",
                "generator_family_and_voice",
            ],
        },
    }


def write_v4_capacity_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    """Write one versioned Gate A receipt without replacing an existing result."""

    if path.exists() or not path.parent.is_dir():
        raise V4CapacityError(f"Unsafe v4 Gate A receipt destination: {path}")
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise V4CapacityError(f"Cannot write v4 Gate A receipt: {path}") from error


def _project_history_inventory(root: Path, patterns: Sequence[str]) -> dict[str, object]:
    paths: set[Path] = set()
    for pattern in patterns:
        pure = PurePosixPath(pattern)
        if pure.is_absolute() or ".." in pure.parts or "\\" in pattern:
            raise V4CapacityError(f"Unsafe project-history glob: {pattern!r}.")
        paths.update(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise V4CapacityError("Project-history inventory is empty.")
    bindings = [_file_binding(path.resolve(strict=True), root) for path in sorted(paths)]
    manifest_rows: list[ManifestRow] = []
    manifest_files = 0
    for path in sorted(paths):
        if path.suffix.lower() == ".csv" and _manifest_like_csv(path):
            loaded = load_manifest(path)
            manifest_rows.extend(loaded)
            manifest_files += 1
    if not manifest_rows:
        raise V4CapacityError("Project-history inventory contains no valid manifest rows.")
    canonical = json.dumps(bindings, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    row_keys = sorted(
        {
            (
                row.sample_id,
                row.sha256,
                row.text_hash,
                row.parent_group_id,
                row.speaker_pseudo_id,
                row.source_name,
                row.label,
                row.language,
                row.generator_family,
                row.voice_id,
            )
            for row in manifest_rows
        }
    )
    rows_digest = hashlib.sha256(
        json.dumps(row_keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "file_globs": list(patterns),
        "files": bindings,
        "file_count": len(bindings),
        "files_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "manifest_files": manifest_files,
        "manifest_rows_loaded_with_version_duplicates": len(manifest_rows),
        "manifest_exposure_keys": len(row_keys),
        "manifest_exposure_keys_sha256": rows_digest,
        "_manifest_rows": manifest_rows,
    }


def _history_counts(rows: Sequence[ManifestRow]) -> dict[str, object]:
    labels_by_source_sample: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row.source_name, row.sample_id)
        prior = labels_by_source_sample.setdefault(key, row.label)
        if prior != row.label:
            raise V4CapacityError(
                "Historical sample changes label across manifests: "
                f"{row.source_name}/{row.sample_id}."
            )
    counts: Counter[tuple[str, str]] = Counter(
        (source, label) for (source, _sample), label in labels_by_source_sample.items()
    )
    by_source: dict[str, dict[str, int]] = {}
    for source, label in sorted(counts):
        by_source.setdefault(source, {})[label] = counts[(source, label)]
    return {
        "unique_source_sample_ids": len(labels_by_source_sample),
        "by_source_and_label": by_source,
    }


def _verify_licenses_and_pins(
    *,
    root: Path,
    config: V4GateAConfig,
    ledger: Mapping[str, LicenseLedgerEntry],
    ruasd_report: Mapping[str, object],
    ksc2_report: Mapping[str, object],
    common_voice_report: Mapping[str, object],
) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    ruasd_entry = _approved_research_entry(ledger, config.ruasd_ledger_source_id)
    ruasd_catalog = _resolve_project_path(root, config.ruasd_catalog_path)
    if sha256_file(ruasd_catalog) != ruasd_entry.sha256:
        raise V4CapacityError("RuASD catalog hash disagrees with the approved license ledger.")
    catalog_archives, catalog_compressed_bytes = _ruasd_catalog_totals(ruasd_catalog)
    if catalog_compressed_bytes != ruasd_entry.expected_size_bytes:
        raise V4CapacityError("RuASD catalog byte total disagrees with the approved ledger.")
    if _positive_int(ruasd_report.get("archive_count"), "RuASD archive_count") != 250:
        raise V4CapacityError("RuASD audit must cover exactly 250 pinned archives.")
    if _positive_int(
        ruasd_report.get("sha256_verified_archives"), "RuASD sha256_verified_archives"
    ) != 250:
        raise V4CapacityError("RuASD Gate A audit did not SHA-256 verify every archive.")
    if catalog_archives != 250:
        raise V4CapacityError("RuASD Gate A catalog must contain exactly 250 archives.")
    receipts.append(_license_receipt(ruasd_entry, ruasd_catalog, root))

    ksc2_entry = _approved_research_entry(ledger, config.ksc2_ledger_source_id)
    ksc2_lock_path = _resolve_project_path(root, config.ksc2_artifact_lock_path)
    if sha256_file(ksc2_lock_path) != ksc2_entry.sha256:
        raise V4CapacityError("KSC2 artifact-lock hash disagrees with the approved ledger.")
    ksc2_lock = _json_object(ksc2_lock_path, "KSC2 artifact lock")
    multipart = _mapping(ksc2_lock.get("multipart_archive"), "KSC2 multipart_archive")
    for field in ("compressed_bytes", "compressed_sha256"):
        if ksc2_report.get(field) != multipart.get(field):
            raise V4CapacityError(f"Fresh KSC2 audit disagrees with artifact lock: {field}.")
    if ksc2_report.get("unpaired_audio_files") != 0:
        raise V4CapacityError("KSC2 audit contains unpaired audio files.")
    if ksc2_report.get("parts") != multipart.get("parts"):
        raise V4CapacityError("Fresh KSC2 audit part receipts disagree with artifact lock.")
    receipts.append(_license_receipt(ksc2_entry, ksc2_lock_path, root))

    cv_entry = _approved_research_entry(ledger, config.common_voice_ledger_source_id)
    archive = _mapping(common_voice_report.get("archive"), "Common Voice archive")
    if (
        archive.get("expected_size_bytes") != cv_entry.expected_size_bytes
        or archive.get("expected_sha256") != cv_entry.sha256
        or archive.get("identity_verified_before_metadata_read") is not True
    ):
        raise V4CapacityError("Common Voice screen is not bound to the approved exact archive.")
    _verify_common_voice_screen_scope(root, common_voice_report)
    receipts.append(
        {
            "source_id": cv_entry.source_id,
            "status": cv_entry.status,
            "train_dev_test_use": cv_entry.train_dev_test_use,
            "archive_size_bytes": cv_entry.expected_size_bytes,
            "archive_sha256": cv_entry.sha256,
            "fresh_metadata_screen_verified_archive_before_read": True,
        }
    )
    return receipts


def _verify_common_voice_screen_scope(
    root: Path, common_voice_report: Mapping[str, object]
) -> None:
    scope = _mapping(common_voice_report.get("scope"), "Common Voice scope")
    if scope.get("configuration_directory") != "configs/research":
        raise V4CapacityError("Common Voice screen used an unexpected configuration directory.")
    if scope.get("manifest_inventory_directory") != "data/manifests":
        raise V4CapacityError("Common Voice screen used an unexpected manifest directory.")
    expected_configs = {
        _relative_to_root(path.resolve(strict=True), root, "Research config"): sha256_file(path)
        for path in sorted((root / "configs/research").glob("*.json"))
    }
    actual_configs = _screen_bindings(
        scope.get("configuration_files"), "Common Voice configuration_files"
    )
    if actual_configs != expected_configs:
        raise V4CapacityError("Common Voice screen is stale against current research configs.")
    manifest_paths = [
        path
        for path in sorted((root / "data/manifests").rglob("*.csv"))
        if _manifest_like_csv(path)
    ]
    expected_manifests = {
        _relative_to_root(path.resolve(strict=True), root, "Manifest inventory"): sha256_file(path)
        for path in manifest_paths
    }
    actual_manifests = _screen_bindings(
        scope.get("inventory_manifests"), "Common Voice inventory_manifests"
    )
    if actual_manifests != expected_manifests:
        raise V4CapacityError("Common Voice screen is stale against current manifest history.")


def _screen_bindings(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, list):
        raise V4CapacityError(f"{label} must be an array.")
    result: dict[str, str] = {}
    for index, item in enumerate(value, start=1):
        raw = _mapping(item, f"{label}[{index}]")
        path = _string(raw.get("path"), f"{label}[{index}].path")
        digest = _string(raw.get("sha256"), f"{label}[{index}].sha256")
        if path in result:
            raise V4CapacityError(f"{label} contains duplicate path {path!r}.")
        result[path] = digest
    return result


def _ruasd_catalog_totals(path: Path) -> tuple[int, int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "expected_size_bytes" not in reader.fieldnames:
                raise V4CapacityError("RuASD catalog lacks expected_size_bytes.")
            sizes: list[int] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    size = int((row.get("expected_size_bytes") or "").strip())
                except ValueError as error:
                    raise V4CapacityError(
                        f"RuASD catalog row {row_number} has invalid expected_size_bytes."
                    ) from error
                if size <= 0:
                    raise V4CapacityError(
                        f"RuASD catalog row {row_number} has non-positive size."
                    )
                sizes.append(size)
    except (OSError, csv.Error) as error:
        raise V4CapacityError(f"Cannot read RuASD catalog totals: {path}") from error
    return len(sizes), sum(sizes)


def _verify_tts_routes(
    root: Path,
    config: V4GateAConfig,
    ledger: Mapping[str, LicenseLedgerEntry],
) -> tuple[list[dict[str, object]], set[str]]:
    receipts: list[dict[str, object]] = []
    families: set[str] = set()
    for route in config.tts_routes:
        entry = _approved_research_entry(ledger, route.ledger_source_id)
        lock_path = _resolve_project_path(root, route.lock_path)
        model_root = _resolve_project_path(root, route.model_root, require_directory=True)
        if sha256_file(lock_path) != entry.sha256:
            raise V4CapacityError(
                f"TTS lock hash for {route.route_id!r} disagrees with the approved ledger."
            )
        lock = load_research_tts_model_lock(lock_path)
        verified = verify_research_tts_model_lock(model_root, lock)
        model_receipts = _model_receipts(lock, verified)
        families.update(model.generator_family for model in lock.models)
        receipts.append(
            {
                "route_id": route.route_id,
                "ledger_source_id": route.ledger_source_id,
                "lock": _file_binding(lock_path, root),
                "model_root": _relative_to_root(model_root, root, "TTS model root"),
                "models": model_receipts,
                "status": "verified_exact_local_bytes",
            }
        )
    return receipts, families


def _model_receipts(
    lock: ResearchTtsModelLock, verified: Mapping[str, Mapping[str, Path]]
) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for model in lock.models:
        verified_paths = verified.get(model.model_id)
        if verified_paths is None or len(verified_paths) != len(model.artifacts):
            raise V4CapacityError(f"Incomplete verified TTS model: {model.model_id!r}.")
        receipts.append(
            {
                "model_id": model.model_id,
                "generator_family": model.generator_family,
                "generator_name": model.generator_name,
                "artifact_count": len(model.artifacts),
                "verified_bytes": sum(artifact.expected_size_bytes for artifact in model.artifacts),
            }
        )
    return receipts


def _history_source_label_count(
    history_counts: Mapping[str, object], source_ids: Sequence[str], label: str
) -> int:
    by_source = _mapping(history_counts.get("by_source_and_label"), "history by source")
    count = 0
    for source_id in source_ids:
        value = by_source.get(source_id, {})
        source = _mapping(value, f"history source {source_id}")
        count += _positive_or_zero_int(source.get(label, 0), f"history {source_id}/{label}")
    return count


def _approved_research_entry(
    ledger: Mapping[str, LicenseLedgerEntry], source_id: str
) -> LicenseLedgerEntry:
    entry = ledger.get(source_id)
    if entry is None:
        raise V4CapacityError(f"Required source is absent from license ledger: {source_id!r}.")
    if entry.status not in APPROVED_LICENSE_STATUSES:
        raise V4CapacityError(f"Required source is not approved: {source_id!r}.")
    if entry.train_dev_test_use not in {"research_only", "product_allowed"}:
        raise V4CapacityError(f"Required source prohibits research training: {source_id!r}.")
    return entry


def _license_receipt(entry: LicenseLedgerEntry, artifact: Path, root: Path) -> dict[str, object]:
    return {
        "source_id": entry.source_id,
        "status": entry.status,
        "train_dev_test_use": entry.train_dev_test_use,
        "artifact": _file_binding(artifact, root),
    }


def _parse_tts_route(value: object, index: int) -> V4TtsRoute:
    raw = _mapping(value, f"sources.tts_routes[{index}]")
    _expect_keys(
        raw,
        {"route_id", "ledger_source_id", "lock_path", "model_root"},
        f"sources.tts_routes[{index}]",
    )
    return V4TtsRoute(
        route_id=_string(raw["route_id"], f"tts route {index}.route_id"),
        ledger_source_id=_string(
            raw["ledger_source_id"], f"tts route {index}.ledger_source_id"
        ),
        lock_path=_safe_path(raw["lock_path"], f"tts route {index}.lock_path"),
        model_root=_safe_path(raw["model_root"], f"tts route {index}.model_root"),
    )


def _manifest_like_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return REQUIRED_FIELDS.issubset(next(csv.reader(handle), []))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4CapacityError(f"Cannot inspect project-history CSV {path}: {error}") from error


def _file_binding(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": _relative_to_root(path, root, "Inventory file"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _external_file_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _resolve_project_path(
    root: Path, value: str, *, require_directory: bool = False
) -> Path:
    candidate = (root / value).resolve(strict=True)
    _relative_to_root(candidate, root, "Configured project path")
    if require_directory and not candidate.is_dir():
        raise V4CapacityError(f"Configured path is not a directory: {value!r}.")
    if not require_directory and not candidate.is_file():
        raise V4CapacityError(f"Configured path is not a file: {value!r}.")
    return candidate


def _relative_to_root(path: Path, root: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise V4CapacityError(f"{label} escapes project root: {path}") from error


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4CapacityError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise V4CapacityError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4CapacityError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _int_mapping(value: object, label: str) -> dict[str, int]:
    raw = _mapping(value, label)
    result: dict[str, int] = {}
    for key, count in raw.items():
        result[key] = _positive_or_zero_int(count, f"{label}.{key}")
    return result


def _required_count(counts: Mapping[str, int], key: str) -> int:
    if key not in counts:
        raise V4CapacityError(f"Required audited count is absent: {key!r}.")
    return counts[key]


def _expect_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise V4CapacityError(f"{label} has unexpected fields ({'; '.join(details)}).")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V4CapacityError(f"{label} must be a non-empty string.")
    return value.strip()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise V4CapacityError(f"{label} must be an array.")
    result = tuple(_string(item, label) for item in value)
    if len(result) != len(set(result)):
        raise V4CapacityError(f"{label} contains duplicates.")
    return result


def _safe_path(value: object, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text or text in {"", "."}:
        raise V4CapacityError(f"{label} must be a safe project-relative path.")
    return path.as_posix()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4CapacityError(f"{label} must be a positive integer.")
    return value


def _positive_or_zero_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise V4CapacityError(f"{label} must be a non-negative integer.")
    return value


def _iso_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4CapacityError("audited_at must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise V4CapacityError("audited_at must include a timezone.")
