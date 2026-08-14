"""Exact source-audio materialization for the canonical XLS-R+SLS model v4 pool."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestRow, validate_manifest, write_manifest
from kds.data.ruasd_catalog import (
    RuAsdArchiveSpec,
    _validate_archive_set,
    _validate_archive_size,
    load_ruasd_archive_records,
)
from kds.data.ruasd_research import RuAsdResearchRecord, _eligible_records
from kds.data.v4_selection import V4_CANDIDATE_FIELDS, V4ExposureInventory

V4_RAW_INVENTORY_SCHEMA_VERSION = 1
V4_RAW_INVENTORY_FIELDS = (
    "selection_rank",
    "target_state",
    "language",
    "label",
    "candidate_id",
    "pair_id",
    "source_id",
    "source_lineage_id",
    "source_component",
    "archive_audio_member",
    "text_hash",
    "canonical_text_hash",
    "parent_group_id",
    "raw_relative_path",
    "raw_audio_sha256",
    "raw_size_bytes",
    "duration_s",
    "original_sr",
    "codec",
    "eligibility_status",
    "rejection_reason",
    "duplicate_of_candidate_id",
    "historical_exact_hash_matches",
)


class V4MaterializationError(ValueError):
    """Raised when source bytes cannot be bound and materialized fail closed."""


V4MaterializationProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class V4SourceCandidate:
    selection_rank: int
    target_state: str
    language: str
    label: str
    candidate_id: str
    pair_id: str
    source_id: str
    source_lineage_id: str
    source_component: str
    archive_audio_member: str
    text_hash: str
    canonical_text_hash: str
    parent_group_id: str

    @property
    def cell(self) -> str:
        return f"{self.language}/{self.label}"


@dataclass(frozen=True, slots=True)
class V4RawAsset:
    candidate: V4SourceCandidate
    raw_relative_path: str
    raw_audio_sha256: str
    raw_size_bytes: int
    duration_s: float
    original_sr: int
    codec: str


@dataclass(frozen=True, slots=True)
class V4RawDecision:
    asset: V4RawAsset
    eligibility_status: str
    rejection_reason: str
    duplicate_of_candidate_id: str
    historical_exact_hash_matches: int


def load_v4_source_candidates(
    candidate_csv: Path,
    governance_receipt: Path,
) -> tuple[V4SourceCandidate, ...]:
    """Load only canonical v2 source rows and reject a superseded or mutated packet."""

    governance = _json_object(governance_receipt, "v4 selection governance")
    canonical = _mapping(governance.get("canonical_packet"), "canonical_packet")
    csv_binding = _mapping(canonical.get("candidate_csv"), "candidate_csv")
    if (
        governance.get("schema_version") != 1
        or governance.get("status")
        != "canonical_metadata_selection_v2_audio_gate_pending"
        or canonical.get("version") != 2
        or canonical.get("source_audio_materialization_authorized") is not True
        or canonical.get("synthesis_authorized") is not False
        or csv_binding.get("path") != candidate_csv.as_posix()
        or csv_binding.get("rows") != 28_800
        or sha256_file(candidate_csv) != _sha256(csv_binding.get("sha256"), "candidate CSV")
    ):
        raise V4MaterializationError("Canonical v4 selection governance binding is invalid.")
    try:
        with candidate_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != V4_CANDIDATE_FIELDS:
                raise V4MaterializationError("v4 candidate CSV schema is invalid.")
            mappings = list(reader)
    except OSError as error:
        raise V4MaterializationError("Cannot read canonical v4 candidate CSV.") from error
    if len(mappings) != 28_800:
        raise V4MaterializationError("Canonical v4 candidate CSV must contain 28,800 rows.")
    source_rows: list[V4SourceCandidate] = []
    ranks_by_cell: dict[str, list[int]] = {}
    candidate_ids: set[str] = set()
    for row_number, mapping in enumerate(mappings, start=2):
        language = (mapping.get("language") or "").strip()
        label = (mapping.get("label") or "").strip()
        source_id = (mapping.get("source_id") or "").strip()
        is_source = language == "ru" or (language == "kk" and label == "bonafide")
        if not is_source:
            continue
        try:
            row = V4SourceCandidate(
                selection_rank=int(mapping.get("selection_rank") or ""),
                target_state=(mapping.get("target_state") or "").strip(),
                language=language,
                label=label,
                candidate_id=(mapping.get("candidate_id") or "").strip(),
                pair_id=(mapping.get("pair_id") or "").strip(),
                source_id=source_id,
                source_lineage_id=(mapping.get("source_lineage_id") or "").strip(),
                source_component=(mapping.get("source_component") or "").strip(),
                archive_audio_member=(mapping.get("archive_audio_member") or "").strip(),
                text_hash=_sha256(mapping.get("text_hash"), "candidate text hash"),
                canonical_text_hash=_sha256(
                    mapping.get("canonical_text_hash"), "candidate canonical text hash"
                ),
                parent_group_id=(mapping.get("parent_group_id") or "").strip(),
            )
        except ValueError as error:
            raise V4MaterializationError(
                f"v4 source candidate row {row_number} is invalid."
            ) from error
        expected_source = "ruasd_ru_v1_full" if language == "ru" else "ksc2_v1"
        expected_state = "target" if row.selection_rank <= 6000 else "reserve"
        if (
            mapping.get("role") != "train"
            or row.source_id != expected_source
            or row.label not in {"bonafide", "spoof"}
            or (language == "kk" and row.label != "bonafide")
            or not row.archive_audio_member
            or row.selection_rank not in range(1, 7201)
            or row.target_state != expected_state
            or mapping.get("raw_audio_sha256") != ""
            or mapping.get("decoded_audio_sha256") != ""
            or mapping.get("asset_state") != "source_audio_unmaterialized"
        ):
            raise V4MaterializationError(
                f"v4 source candidate row {row_number} violates the frozen source contract."
            )
        if row.candidate_id in candidate_ids:
            raise V4MaterializationError("v4 source candidates contain duplicate IDs.")
        candidate_ids.add(row.candidate_id)
        ranks_by_cell.setdefault(row.cell, []).append(row.selection_rank)
        source_rows.append(row)
    expected_cells = {"ru/bonafide", "ru/spoof", "kk/bonafide"}
    if set(ranks_by_cell) != expected_cells or any(
        sorted(ranks) != list(range(1, 7201)) for ranks in ranks_by_cell.values()
    ):
        raise V4MaterializationError("v4 source candidate cells/ranks are incomplete.")
    return tuple(source_rows)


def bind_v4_ruasd_records(
    candidates: Sequence[V4SourceCandidate],
    archive_dir: Path,
    catalog: Mapping[str, RuAsdArchiveSpec],
    *,
    progress_callback: V4MaterializationProgressCallback | None = None,
) -> tuple[RuAsdResearchRecord, ...]:
    """Rebind every selected RuASD row to current exact metadata before extraction."""

    selected = {candidate.candidate_id: candidate for candidate in candidates}
    if not selected or any(candidate.source_id != "ruasd_ru_v1_full" for candidate in candidates):
        raise V4MaterializationError("RuASD binding received a non-RuASD candidate.")
    archive_paths = _validate_archive_set(archive_dir, catalog)
    bound: dict[str, RuAsdResearchRecord] = {}
    for completed, archive_name in enumerate(sorted(catalog), start=1):
        _validate_archive_size(archive_paths[archive_name], catalog[archive_name])
        for record in _eligible_records(
            archive_name, load_ruasd_archive_records(archive_paths[archive_name])
        ):
            candidate_id = f"ruasd_ru_v1_full:{record.record_key}"
            candidate = selected.get(candidate_id)
            if candidate is None:
                continue
            if (
                candidate.label != record.label
                or candidate.source_component != record.subset
                or candidate.archive_audio_member != f"{record.sample_id}.wav"
                or candidate.text_hash != record.text_hash
                or not record.has_source_text
            ):
                raise V4MaterializationError(
                    f"RuASD source metadata changed for {candidate_id!r}."
                )
            bound[candidate_id] = record
        if progress_callback is not None:
            progress_callback(completed, len(catalog), archive_name)
    if set(bound) != set(selected):
        raise V4MaterializationError("RuASD metadata binding did not recover every source row.")
    return tuple(bound[candidate.candidate_id] for candidate in candidates)


def inspect_v4_raw_asset(
    candidate: V4SourceCandidate,
    path: Path,
    raw_relative_path: str,
) -> V4RawAsset:
    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise V4MaterializationError(f"Cannot inspect v4 raw audio: {path}") from error
    duration = float(info.duration)
    sample_rate = int(info.samplerate)
    if not math.isfinite(duration) or duration <= 0 or sample_rate <= 0 or path.stat().st_size <= 0:
        raise V4MaterializationError(f"Invalid v4 raw audio properties: {path}")
    return V4RawAsset(
        candidate=candidate,
        raw_relative_path=raw_relative_path,
        raw_audio_sha256=sha256_file(path),
        raw_size_bytes=path.stat().st_size,
        duration_s=duration,
        original_sr=sample_rate,
        codec=path.suffix.lower().removeprefix("."),
    )


def decide_v4_raw_exact_eligibility(
    assets: Sequence[V4RawAsset],
    historical_audio_sha256: frozenset[str],
    *,
    target_per_cell: int = 6000,
) -> tuple[V4RawDecision, ...]:
    """Reject historical/within-pool raw duplicates in frozen priority order."""

    ordered = sorted(
        assets,
        key=lambda asset: (
            asset.candidate.language,
            asset.candidate.label,
            asset.candidate.selection_rank,
            asset.candidate.candidate_id,
        ),
    )
    owner_by_hash: dict[str, V4RawAsset] = {}
    decisions: list[V4RawDecision] = []
    for asset in ordered:
        historical_matches = int(asset.raw_audio_sha256 in historical_audio_sha256)
        prior = owner_by_hash.get(asset.raw_audio_sha256)
        if historical_matches:
            status = "rejected"
            reason = "historical_exact_raw_audio_hash"
            duplicate_of = ""
        elif prior is not None:
            if (
                prior.candidate.language != asset.candidate.language
                or prior.candidate.label != asset.candidate.label
            ):
                raise V4MaterializationError(
                    "Exact raw audio has conflicting language/label assignments: "
                    f"{prior.candidate.candidate_id!r}, {asset.candidate.candidate_id!r}."
                )
            status = "rejected"
            reason = "within_pool_exact_raw_audio_duplicate"
            duplicate_of = prior.candidate.candidate_id
        else:
            status = "eligible_for_decode_qa"
            reason = ""
            duplicate_of = ""
            owner_by_hash[asset.raw_audio_sha256] = asset
        decisions.append(
            V4RawDecision(
                asset=asset,
                eligibility_status=status,
                rejection_reason=reason,
                duplicate_of_candidate_id=duplicate_of,
                historical_exact_hash_matches=historical_matches,
            )
        )
    eligible_counts = Counter(
        decision.asset.candidate.cell
        for decision in decisions
        if decision.eligibility_status == "eligible_for_decode_qa"
    )
    for cell in ("ru/bonafide", "ru/spoof", "kk/bonafide"):
        if eligible_counts[cell] < target_per_cell:
            raise V4MaterializationError(
                f"Raw exact-audio gate leaves {eligible_counts[cell]} rows in {cell}; "
                f"needs {target_per_cell}."
            )
    return tuple(decisions)


def v4_raw_manifest_rows(
    decisions: Sequence[V4RawDecision], *, created_at: str
) -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    for decision in decisions:
        if decision.eligibility_status != "eligible_for_decode_qa":
            continue
        asset = decision.asset
        candidate = asset.candidate
        is_ruasd = candidate.source_id == "ruasd_ru_v1_full"
        is_spoof = candidate.label == "spoof"
        speaker_group = (
            "ruasd_ru_v1_full:unknown-spoof-voice-group"
            if is_ruasd and is_spoof
            else "ruasd_ru_v1_full:unknown-bonafide-speaker-group"
            if is_ruasd
            else "ksc2_v1:unknown-speaker-group"
        )
        rows.append(
            ManifestRow(
                sample_id=candidate.candidate_id,
                relative_path=asset.raw_relative_path,
                sha256=asset.raw_audio_sha256,
                split="train",
                label=candidate.label,
                language=candidate.language,
                code_switch="unknown",
                parent_group_id=candidate.parent_group_id,
                source_name=candidate.source_id,
                source_license="CC-BY-NC-SA-4.0" if is_ruasd else "CC-BY-4.0",
                rights_basis=(
                    "Pinned RuASD full raw release; personal research only; unknown speaker/voice"
                    if is_ruasd
                    else "Pinned ISSAI KSC2 nonlegacy Train component; CC-BY-4.0"
                ),
                speaker_pseudo_id=speaker_group,
                text_id=f"xlsr-sls-model-v4:text:{candidate.canonical_text_hash}",
                text_hash=candidate.canonical_text_hash,
                duration_s=asset.duration_s,
                generator_family=(
                    "ruasd_source_tts_families_unverified" if is_spoof else ""
                ),
                generator_name=candidate.source_component if is_spoof else "",
                generator_version=(
                    "source_release_route_metadata_unverified" if is_spoof else ""
                ),
                voice_id="unknown_by_source" if is_spoof else "",
                clone_consent_id="",
                device="unknown",
                capture_route=(
                    "ruasd_source_audio" if is_ruasd else f"ksc2:{candidate.source_component}"
                ),
                original_sr=asset.original_sr,
                codec=asset.codec,
                augmentation_chain="",
                augmentation_seed="",
                created_at=created_at,
            )
        )
    validate_manifest(rows)
    return tuple(rows)


def publish_v4_raw_materialization(
    *,
    raw_destination: Path,
    staged_raw_root: Path,
    inventory_path: Path,
    raw_manifest_path: Path,
    receipt_path: Path,
    decisions: Sequence[V4RawDecision],
    raw_rows: Sequence[ManifestRow],
    data_root: Path,
    license_ledger_path: Path,
    created_at: str,
    bindings: Mapping[str, object],
) -> None:
    """Publish the extracted directory and complete raw accounting as one write-once packet."""

    outputs = (inventory_path, raw_manifest_path, receipt_path)
    if (
        raw_destination.exists()
        or not raw_destination.parent.is_dir()
        or len(set(outputs)) != len(outputs)
        or any(path.exists() or not path.parent.is_dir() for path in outputs)
    ):
        raise V4MaterializationError("Unsafe v4 raw materialization destinations.")
    validate_manifest(raw_rows)
    ledger = load_license_ledger(license_ledger_path)
    validate_manifest_licenses(raw_rows, ledger)
    eligible = [
        decision
        for decision in decisions
        if decision.eligibility_status == "eligible_for_decode_qa"
    ]
    rejection_counts = Counter(
        decision.rejection_reason for decision in decisions if decision.rejection_reason
    )
    raw_published = False
    published_outputs: list[tuple[Path, Path]] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-raw-metadata-", dir=inventory_path.parent
        ) as metadata_stage_name:
            stage = Path(metadata_stage_name)
            staged_inventory = stage / inventory_path.name
            staged_manifest = stage / raw_manifest_path.name
            staged_receipt = stage / receipt_path.name
            with staged_inventory.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=V4_RAW_INVENTORY_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for decision in decisions:
                    asset = decision.asset
                    candidate = asset.candidate
                    writer.writerow(
                        {
                            "selection_rank": candidate.selection_rank,
                            "target_state": candidate.target_state,
                            "language": candidate.language,
                            "label": candidate.label,
                            "candidate_id": candidate.candidate_id,
                            "pair_id": candidate.pair_id,
                            "source_id": candidate.source_id,
                            "source_lineage_id": candidate.source_lineage_id,
                            "source_component": candidate.source_component,
                            "archive_audio_member": candidate.archive_audio_member,
                            "text_hash": candidate.text_hash,
                            "canonical_text_hash": candidate.canonical_text_hash,
                            "parent_group_id": candidate.parent_group_id,
                            "raw_relative_path": asset.raw_relative_path,
                            "raw_audio_sha256": asset.raw_audio_sha256,
                            "raw_size_bytes": asset.raw_size_bytes,
                            "duration_s": asset.duration_s,
                            "original_sr": asset.original_sr,
                            "codec": asset.codec,
                            "eligibility_status": decision.eligibility_status,
                            "rejection_reason": decision.rejection_reason,
                            "duplicate_of_candidate_id": decision.duplicate_of_candidate_id,
                            "historical_exact_hash_matches": (
                                decision.historical_exact_hash_matches
                            ),
                        }
                    )
            write_manifest(staged_manifest, raw_rows)
            receipt = {
                "schema_version": V4_RAW_INVENTORY_SCHEMA_VERSION,
                "protocol_id": "xlsr-sls-model-v4-source-raw-materialization-v1",
                "created_at": created_at,
                "state": "source_raw_materialized_decode_qa_pending",
                "bindings": bindings,
                "outputs": {
                    "raw_directory": raw_destination.relative_to(data_root.parent).as_posix(),
                    "raw_inventory": {
                        "path": inventory_path.as_posix(),
                        "sha256": sha256_file(staged_inventory),
                        "rows": len(decisions),
                    },
                    "raw_manifest": {
                        "path": raw_manifest_path.as_posix(),
                        "sha256": sha256_file(staged_manifest),
                        "rows": len(raw_rows),
                    },
                },
                "accounting": {
                    "selected_source_rows": len(decisions),
                    "eligible_for_decode_qa": len(eligible),
                    "rejected": len(decisions) - len(eligible),
                    "rejection_reason_counts": dict(sorted(rejection_counts.items())),
                    "eligible_cell_counts": dict(
                        sorted(Counter(item.asset.candidate.cell for item in eligible).items())
                    ),
                },
                "claims": {
                    "raw_audio_extraction_performed": True,
                    "historical_exact_raw_hash_screen_performed": True,
                    "within_pool_exact_raw_hash_screen_performed": True,
                    "decoded_audio_hash_screen_performed": False,
                    "near_audio_screen_performed": False,
                    "technical_qa_vad_performed": False,
                    "speaker_independence": "not_verified_speaker_independent",
                    "synthesis_authorized": False,
                    "training_authorized": False,
                    "replacement_or_backfill_performed": False,
                },
                "next_gate": (
                    "decode eligible raw rows to 16-kHz mono PCM WAV; run QA/VAD; compute "
                    "decoded hashes and near-audio fingerprints before synthesis"
                ),
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staged_raw_root.replace(raw_destination)
            raw_published = True
            require_valid_assets(raw_rows, data_root)
            for staged_path, output_path in (
                (staged_inventory, inventory_path),
                (staged_manifest, raw_manifest_path),
                (staged_receipt, receipt_path),
            ):
                os.link(staged_path, output_path)
                published_outputs.append((output_path, staged_path))
    except (OSError, ValueError) as error:
        for output_path, staged_path in reversed(published_outputs):
            try:
                if output_path.samefile(staged_path):
                    output_path.unlink()
            except OSError:
                pass
        if raw_published:
            shutil.rmtree(raw_destination, ignore_errors=True)
        raise V4MaterializationError(f"Cannot publish v4 raw materialization: {error}") from error


def inventory_exposure_binding(exposure: V4ExposureInventory) -> dict[str, object]:
    return {
        "manifest_count": len(exposure.manifest_bindings),
        "rows_with_version_duplicates": exposure.rows,
        "manifests": list(exposure.manifest_bindings),
    }


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4MaterializationError(f"Cannot read {label}: {path}") from error
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4MaterializationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4MaterializationError(f"{label} must be a lowercase SHA-256 digest.")
    return value
